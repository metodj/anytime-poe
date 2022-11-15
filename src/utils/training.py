from typing import Callable, Mapping, Optional, Tuple, Union, List, Dict
from functools import partial

import wandb
from tqdm.auto import trange
import jax
from jax import random
from jax import numpy as jnp
import distrax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from flax import struct
import flax.linen as nn
import optax
from chex import Array
from ml_collections import config_dict
from clu import parameter_overview

from src.data import NumpyLoader
import src.models as models
import src.utils.optim
from src.models.common import PRNGKey, powerset

ScalarOrSchedule = Union[float, optax.Schedule]


class TrainState(train_state.TrainState):
    """A Flax TrainState which also tracks model state (e.g. BatchNorm running averages) and schedules β."""
    model_state: FrozenDict
    β: float
    β_val_or_schedule: ScalarOrSchedule = struct.field(pytree_node=False)
    alpha: float
    alpha_val_or_schedule: ScalarOrSchedule = struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            β=_get_param_for_step(self.step, self.β_val_or_schedule),
            alpha=_get_param_for_step(self.step, self.alpha_val_or_schedule),
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, β_val_or_schedule, alpha_val_or_schedule, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            β_val_or_schedule=β_val_or_schedule,
            β=_get_param_for_step(0, β_val_or_schedule),
            alpha_val_or_schedule=alpha_val_or_schedule,
            alpha=_get_param_for_step(0, alpha_val_or_schedule),
            **kwargs,
        )


# Helper which helps us deal with the fact that we can either specify a fixed β
# or a schedule for adjusting β. This is a pattern similar to the one used by
# optax for dealing with LRs either being specified as a constant or schedule.
def _get_param_for_step(step, param_val_or_schedule):
    if callable(param_val_or_schedule):
        return param_val_or_schedule(step)
    else:
        return param_val_or_schedule


def setup_training(
    config: config_dict.ConfigDict,
    rng: PRNGKey,
    init_x: Array,
    init_y: Union[float, int]
) -> Tuple[nn.Module, TrainState]:
    """Helper which returns the model object and the corresponding initialised train state for a given config.
    """
    model_cls = getattr(models, config.model_name)
    model = model_cls(**config.model.to_dict())

    init_rng, rng = random.split(rng)
    variables = model.init(init_rng, init_x, init_y)

    print(parameter_overview.get_parameter_overview(variables))
    # ^ This is really nice for summarising Jax models!

    model_state, params = variables.pop('params')
    del variables

    if config.get('lr_schedule_name', None):
        schedule = getattr(optax, config.lr_schedule_name)
        lr = schedule(
            init_value=config.learning_rate,
            **config.lr_schedule.to_dict()
        )
    else:
        lr = config.learning_rate

    optim = getattr(src.utils.optim, config.optim_name)
    optim = optax.inject_hyperparams(optim)
    # This ^ allows us to access the lr as opt_state.hyperparams['learning_rate'].

    def _parse_schedule(schedule_name: str, param_name: str) -> Union[None, Callable, float]:
        # TODO: add sigmoidal schedule which ramps up more quickly and then gives more time for higher βs
        if config.get(schedule_name, False):
            add = config[schedule_name].end - config[schedule_name].start
            if config[schedule_name].name == 'sigmoid':
                sigmoid = lambda x: 1 / (1 + jnp.exp(x))
                half_steps = config[schedule_name].steps/2
                schedule = lambda step: config[schedule_name].start + add * sigmoid((-step + half_steps)/(config[schedule_name].steps/10))
            else:  # linear
                schedule = lambda step: jnp.minimum(config[schedule_name].start + add / config[schedule_name].steps * step, config[schedule_name].end)
        elif config.get(param_name, False):
            if param_name == 'β':
                schedule = config.β
            elif param_name == 'alpha':
                schedule = config.alpha
            else:
                raise ValueError()
        else:
            schedule = None
        return schedule

    β = _parse_schedule('β_schedule', 'β')
    alpha = _parse_schedule('alpha_schedule', 'alpha')

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optim(learning_rate=lr, **config.optim.to_dict()),
        model_state=model_state,
        β_val_or_schedule=β,
        alpha_val_or_schedule=alpha
    )

    return model, state


def get_ensemble_ids(stochastic_sampling_ens: bool, batch_rng: PRNGKey, sub_ensembles: List[List[int]], M: int) -> List[int]:
    if stochastic_sampling_ens:
        n = len(sub_ensembles)
        probs = jnp.ones(n) / n
        ids_ = distrax.Categorical(probs=probs).sample(seed=batch_rng)
        return sub_ensembles[ids_]
    else:
        return [i for i in range(M)]


def train_loop(
    model: nn.Module,
    state: TrainState,
    config: config_dict.ConfigDict,
    rng: PRNGKey,
    make_loss_fn: Callable,
    make_eval_fn: Callable,
    train_loader: NumpyLoader,
    val_loader: NumpyLoader,
    test_loader: Optional[NumpyLoader] = None,
    wandb_kwargs: Optional[Mapping] = None,
    plot_fn: Optional[Callable] = None,
    plot_freq: int = 10,
    wandb_user: str = 'metodj',
    best_val_error: bool = True,
    stochastic_sampling_ens: bool = False
) -> TrainState:
    """Runs the training loop!
    """
    wandb_kwargs = {
        'project': 'anytime-poe',
        'entity': wandb_user,
        'notes': '',
        # 'mode': 'disabled',
        'config': config.to_dict()
    } | (wandb_kwargs or {})
    # ^ here wandb_kwargs (i.e. whatever the user specifies) takes priority.

    with wandb.init(**wandb_kwargs) as run:

        def _get_kwargs(current_state: TrainState) -> Dict:
            kwargs = {'β': current_state.β} if current_state.β is not None else {}
            kwargs_alpha = {'alpha': current_state.alpha} if current_state.alpha is not None else {}
            return kwargs | kwargs_alpha

        @jax.jit
        def train_step(state, x_batch, y_batch, rng, ensemble_ids):
            kwargs = _get_kwargs(state)

            if config.get('train_data_noise', False):
                x_noise_rng, y_noise_rng = random.split(rng)
                noise_dist = distrax.Uniform(-config.train_data_noise, config.train_data_noise)
                x_noise = noise_dist.sample(seed=x_noise_rng, sample_shape=x_batch.shape)
                y_noise = noise_dist.sample(seed=y_noise_rng, sample_shape=y_batch.shape)
                x_batch = x_batch + x_noise
                y_batch = y_batch + y_noise

            loss_fn = make_loss_fn(model, x_batch, y_batch, train=True, aggregation='mean', ensemble_ids=ensemble_ids, **kwargs)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            (nll, (model_state, err, prod_ll, members_ll)), grads = grad_fn(
                state.params, state.model_state, rng,
            )

            return state.apply_gradients(grads=grads, model_state=model_state), nll * len(x_batch), err * len(x_batch), prod_ll * len(x_batch), members_ll * len(x_batch)

        @jax.jit
        def eval_step(state, x_batch, y_batch, rng):
            kwargs = _get_kwargs(state)

            eval_fn = make_eval_fn(model, x_batch, y_batch, train=False, aggregation='sum', **kwargs)

            nll, (_, err, _, _) = eval_fn(
                state.params, state.model_state, rng,
            )

            return nll, err

        sub_ensembles = powerset(config.model.size)

        train_losses = []
        train_errs = []
        train_prod_ll = []
        train_members_ll = []
        val_losses = []
        val_errs = []
        best_val_loss = jnp.inf
        best_val_err = jnp.inf
        best_state = None
        epochs = trange(1, config.epochs + 1)
        for epoch in epochs:
            batch_losses = []
            batch_errs = []
            batch_prod_lls = []
            batch_members_lls = []
            for i, (x_batch, y_batch) in enumerate(train_loader):
                rng, batch_rng = random.split(rng)
                ensemble_ids = get_ensemble_ids(stochastic_sampling_ens, batch_rng, sub_ensembles, config.model.size)
                state, nll, err, prod_ll, members_ll = train_step(state, x_batch, y_batch, batch_rng, ensemble_ids)
                batch_losses.append(nll)
                batch_errs.append(err)
                batch_prod_lls.append(prod_ll)
                batch_members_lls.append(members_ll)

            N = len(train_loader.dataset)
            train_losses.append(jnp.sum(jnp.array(batch_losses)) / N)
            train_errs.append(jnp.sum(jnp.array(batch_errs)) / N)
            train_prod_ll.append(jnp.sum(jnp.array(batch_prod_lls)) / N)
            train_members_ll.append(jnp.sum(jnp.array(batch_members_lls)) / N)

            batch_losses = []
            batch_errs = []
            for (x_batch, y_batch) in val_loader:
                rng, eval_rng = random.split(rng)
                nll, err = eval_step(state, x_batch, y_batch, eval_rng)
                batch_losses.append(nll)
                batch_errs.append(err)

            val_losses.append(jnp.sum(jnp.array(batch_losses)) / len(val_loader.dataset))
            val_errs.append(jnp.sum(jnp.array(batch_errs)) / len(val_loader.dataset))

            learning_rate = state.opt_state.hyperparams['learning_rate']
            metrics_str = (f'train loss: {train_losses[-1]:7.5f}, val loss: {val_losses[-1]:7.5f}' +
                           f', train err: {train_errs[-1]:6.4f}, val err: {val_errs[-1]:6.4f}' +
                           (f', β: {state.β:.4f}' if state.β is not None else '') +
                           f', lr: {learning_rate:7.5f}' +
                           f', prod_nll: {train_prod_ll[-1]:7.5f}',
                           f', members_nll: {train_members_ll[-1]:7.5f}')
            epochs.set_postfix_str(metrics_str)
            print(f'epoch: {epoch:3} - {metrics_str}')

            metrics = {
                'epoch': epoch,
                'train/loss': train_losses[-1],
                'val/loss': val_losses[-1],
                'train/err': train_errs[-1],
                'val/err': val_errs[-1],
                'β': state.β,
                'alpha': state.alpha,
                'learning_rate': learning_rate,
                'train/prod_nll': train_prod_ll[-1],
                'train/members_nll': train_members_ll[-1],
                'train/nll_ratio': train_members_ll[-1] / train_prod_ll[-1]
            }

            if (epoch % plot_freq == 0) and plot_fn is not None:
                X_train, y_train = list(zip(*train_loader.dataset))
                X_val, y_val = list(zip(*val_loader.dataset))
                fit_plot = plot_fn(
                    model, state, train_losses[-1], val_losses[-1], X_train, y_train, X_val, y_val,
                )

                metrics |= {'fit_plot': wandb.Image(fit_plot)}

            run.log(metrics)

            rng, test_rng = random.split(rng)
            if val_errs[-1] <= best_val_err:
                if config.get('β_schedule', False) and state.β <= 0.9 * config.β_schedule.end:
                    continue

                best_val_err = val_errs[-1]
                print("Best val_err")
                # TODO: add model saving.
                best_state = state

                run.summary['best_epoch'] = epoch
                run.summary['best_val_err'] = val_errs[-1]

                if test_loader is not None:
                    batch_losses = []
                    batch_errs = []
                    for (x_batch, y_batch) in (test_loader):
                        eval_rng, test_rng = random.split(test_rng)
                        nll, err = eval_step(state, x_batch, y_batch, eval_rng)
                        batch_losses.append(nll)

                    test_loss = jnp.sum(jnp.array(batch_losses)) / len(test_loader.dataset)
                    test_err = jnp.sum(jnp.array(batch_errs)) / len(test_loader.dataset)
                    run.summary['test/loss'] = test_loss
                    run.summary['test/err'] = test_err

        if best_state is None:
            best_state = state

    return state, best_state
