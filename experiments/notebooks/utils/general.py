import numpy as np
import pandas as pd
import jax

from  functools import partial
from itertools import combinations
import random as rnd

from flax.training.checkpoints import restore_checkpoint
from jax import numpy as jnp
from jax import random
from torch.utils.data import Dataset

from src.data import get_image_dataset, NumpyLoader
from src.utils.training import setup_training
from src.utils.notebook_metrics import *
from ml_collections import config_dict


COLOR_DICT = {1.0: "green", 0.5: "orange", 0.0: "red"}



def get_model_object(config: config_dict.ConfigDict, dataset: Dataset):
    model, _ = setup_training(config, random.PRNGKey(0), dataset[0][0], dataset[0][1])
    return model



def get_ood_data(dataset_name: str, config: config_dict.ConfigDict, normalize: bool = True, N_test: int = 10000):
    
    _, test_dataset_notmnist, _ = get_image_dataset(
    dataset_name=dataset_name,
    val_percent=config.val_percent,
    flatten_img=True,
    train_augmentations=[],
    normalize=normalize
    )
    
    test_loader_notmnist = NumpyLoader(test_dataset_notmnist, config.batch_size, num_workers=8)
    X_test_notmnist, y_test_notmnist = list(zip(*test_loader_notmnist.dataset))


    # for EMNIST there are more than 10K test samples, here we sample 10K random samples
    if dataset_name == "EMNIST":
        rnd.seed(0)
        ids = rnd.sample(range(len(X_test_notmnist)), N_test)
        X_test_notmnist, y_test_notmnist = tuple(np.array(X_test_notmnist)[ids]), tuple(np.array(y_test_notmnist)[ids])
        
    return X_test_notmnist, y_test_notmnist


def get_logits(model, state, x):
    pred_fun = partial(
            model.apply,
            {"params": state['params'], **state['model_state']},
            train=False,
            method=model.ens_logits
        )
    logits = jax.vmap(
        pred_fun, axis_name="batch"
    )(jnp.array(x))
    return logits


def compute_metrics(model, model_name, state, random_seed, X_test, y_test, model_size=5):

    logits = get_logits(model, state, X_test)

    s = set(range(model_size))
    power_set = sum(map(lambda r: list(combinations(s, r)), range(1, len(s)+1)), [])

    rows = []
    for indices in power_set:
        n_members = len(indices)
        logits_ = logits[:, indices, :]

        if 'prod' in model_name:
            if "softmax" in model_name:
                entropies = jax.vmap(categorical_entropy_prod_probs_softmax)(logits_)
                nlls_ = jax.vmap(categorical_nll_prod_probs_softmax)(logits_, jnp.array(y_test))
                infs = jnp.isinf(nlls_)
                if infs.sum() > 0:
                    print(f"dropping {infs.sum()} infs for Ens of {n_members}")
                    print(logits_[infs])
                    print(jnp.array(y_test)[infs])
                nlls = nlls_[~infs]
                briers = jax.vmap(categorical_brier_prod_probs_softmax)(logits_, jnp.array(y_test))
                errs = jax.vmap(categorical_err_prod)(logits_, jnp.array(y_test))
                errs_ex_ood = jax.vmap(categorical_err_prod)(logits_[~infs], jnp.array(y_test)[~infs])
                nr_ood = int(infs.sum())
            else:
                entropies = jax.vmap(ovr_entropy)(logits_)
                nlls_ = jax.vmap(ovr_nll)(logits_, jnp.array(y_test))
                infs = jnp.isinf(nlls_)
                print(f"dropping {infs.sum()} infs for prod of {n_members}")
                nlls = nlls_[~infs]
                briers = jax.vmap(ovr_brier)(logits_, jnp.array(y_test))
                zero_preds_ = jax.vmap(ovr_prod_probs)(logits_).sum(axis=1) == 0.
                
                errs_ = jax.vmap(ovr_err)(logits_[~zero_preds_], jnp.array(y_test)[~zero_preds_])
                errs_zero_preds = jax.vmap(max_voting)(logits_[zero_preds_], jnp.array(y_test)[zero_preds_])
                errs = jnp.concatenate([errs_, errs_zero_preds])
                # errs = jax.vmap(ovr_err_)(logits_, jnp.array(y_test))
                
    #             errs_ex_ood = jax.vmap(ovr_err)(logits_[~infs], jnp.array(y_test)[~infs])
    #             nr_ood = int(infs.sum())
                
                
                errs_ex_ood = jax.vmap(ovr_err)(logits_[~zero_preds_], jnp.array(y_test)[~zero_preds_])
                nr_ood = int(zero_preds_.sum())
        elif 'ens' in model_name:
            entropies = jax.vmap(categorical_entropy)(logits_)
            nlls_ = jax.vmap(categorical_nll)(logits_, jnp.array(y_test))
            infs = jnp.isinf(nlls_)
            if infs.sum() > 0:
                print(f"dropping {infs.sum()} infs for Ens of {n_members}")
                print(logits_[infs])
                print(jnp.array(y_test)[infs])
            nlls = nlls_[~infs]
            briers = jax.vmap(categorical_brier)(logits_, jnp.array(y_test))
            errs = jax.vmap(categorical_err)(logits_, jnp.array(y_test))
            errs_ex_ood = jax.vmap(categorical_err)(logits_[~infs], jnp.array(y_test)[~infs])
            nr_ood = int(infs.sum())
        else:
            raise ValueError()
            
        results = {'model_name': model_name,
                    'n_members': n_members,
                    'random_seed': random_seed,
                    'H': entropies.mean(),
                    'nll': nlls.mean(),
                    'err': errs.mean(),
                    'brier': briers.mean(),
                   'err_ex_ood': errs_ex_ood.mean(),
                   'nr_ood': nr_ood
                }

        rows.append(results)
    return rows


def generate_name(model_type, alpha, pretrained, members_ll, finetune_epochs, i, Z=False):
    if model_type == "ens":
        return f"{model_type}_model_{i}"
    else:
        pretrained = "_pretrained" if pretrained else ""
        members_ll = f"_{members_ll}" if members_ll == "soft_ovr" else ""
        # members_ll = f"_{members_ll}"
        finetuned = f"_finetuned{finetune_epochs}" if finetune_epochs is not None else ""
        Z = "_Z" if Z else ""
        return f"{model_type}_model_{i}_{alpha}{pretrained}{members_ll}{finetuned}{Z}"


def metrics_df(config, models, X_test, y_test, ens_model, prod_model):

    results_df = pd.DataFrame(columns=['model_name', 'n_members', 
                                       'random_seed', 'H', 'err', 
                                       'brier', 'nll', 'err_ex_ood', 'nr_ood'])
    
    for model_type, alpha, pretrained, members_ll, ft_epochs, Z in models:
        for i in range(1):
            model_name = generate_name(model_type, alpha, pretrained, members_ll, ft_epochs, i, Z)
            print(model_name)
            model = prod_model if model_type == 'prod' else ens_model
            state = restore_checkpoint(f'dynNN_redux/{model_name}', 1)

            results = compute_metrics(model, model_name, state, i, X_test, y_test, model_size=config.model.size)
            results_df = pd.concat([
                results_df,
                pd.DataFrame(results)],
                ignore_index=True
            )
            
    min_mse_df = results_df[results_df.n_members == config.model.size][['model_name', 
                                                                        'random_seed', 
                                                                        'err', 'nll', 
                                                                        'brier', 
                                                                        'err_ex_ood', 'nr_ood']].rename(
    columns={'err': 'final_err', 'nll': 'final_nll', 
             'brier': 'final_brier', 'err_ex_ood': 
             'final_err_ex_ood', 'nr_ood': 'final_nr_ood'})
    
    
    tmp_df = results_df.merge(min_mse_df, on=['model_name', 'random_seed'], how='left')
    tmp_df['err_diff'] = tmp_df['err'] - tmp_df['final_err'] 
    tmp_df['nll_diff'] = tmp_df['nll'] - tmp_df['final_nll'] 
    tmp_df['brier_diff'] = tmp_df['brier'] - tmp_df['final_brier'] 
    
    agg_df = tmp_df.groupby(by=['model_name', 'n_members']).agg({
    'H': ['mean', 'std', 'count'],
    'err_diff': ['mean', 'std', 'count'],
    'err': ['mean', 'std', 'count'],
    'nll_diff': ['mean', 'std', 'count'],
    'nll': ['mean', 'std', 'count'],
    'brier_diff': ['mean', 'std', 'count'],
    'brier': ['mean', 'std', 'count'],
    'err_ex_ood': ['mean', 'std', 'count'], 
    'nr_ood': ['mean', 'std', 'count']
})

    
    return agg_df



def metrics_df_by_name(models, X_test, y_test, ens_model, prod_model):

    results_df = pd.DataFrame(columns=['model_name', 'n_members', 
                                       'random_seed', 'H', 'err', 
                                       'brier', 'nll', 'err_ex_ood', 'nr_ood'])
    MODEL_SIZE = 5
    RANDOM_SEED = 0
    
    for model_name in models:
        print(model_name)
        model = prod_model if 'prod' in model_name else ens_model
        state = restore_checkpoint(f'dynNN_redux/{model_name}', 1)

        results = compute_metrics(model, model_name, state, RANDOM_SEED, X_test, y_test)
        results_df = pd.concat([
            results_df,
            pd.DataFrame(results)],
            ignore_index=True
        )
            
    min_mse_df = results_df[results_df.n_members == MODEL_SIZE][['model_name', 
                                                                        'random_seed', 
                                                                        'err', 'nll', 
                                                                        'brier', 
                                                                        'err_ex_ood', 'nr_ood']].rename(
    columns={'err': 'final_err', 'nll': 'final_nll', 
             'brier': 'final_brier', 'err_ex_ood': 
             'final_err_ex_ood', 'nr_ood': 'final_nr_ood'})
    
    
    tmp_df = results_df.merge(min_mse_df, on=['model_name', 'random_seed'], how='left')
    tmp_df['err_diff'] = tmp_df['err'] - tmp_df['final_err'] 
    tmp_df['nll_diff'] = tmp_df['nll'] - tmp_df['final_nll'] 
    tmp_df['brier_diff'] = tmp_df['brier'] - tmp_df['final_brier'] 
    
    agg_df = tmp_df.groupby(by=['model_name', 'n_members']).agg({
    'H': ['mean', 'std', 'count'],
    'err_diff': ['mean', 'std', 'count'],
    'err': ['mean', 'std', 'count'],
    'nll_diff': ['mean', 'std', 'count'],
    'nll': ['mean', 'std', 'count'],
    'brier_diff': ['mean', 'std', 'count'],
    'brier': ['mean', 'std', 'count'],
    'err_ex_ood': ['mean', 'std', 'count'], 
    'nr_ood': ['mean', 'std', 'count']
})

    
    return agg_df