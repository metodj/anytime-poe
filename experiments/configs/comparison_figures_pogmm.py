from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.dataset_name = 'gen_simple_1d'
    config.dataset = config_dict.ConfigDict()
    config.dataset.n_samples = 692
    config.dataset.random_seed = 42
    config.dataset.noise_std = 0.25
    config.dataset.heteroscedastic = False

    config.batch_size = 500
    config.epochs = 201

    config.optim_name = 'sgdw'
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4
    config.optim.momentum = 0.9
    config.learning_rate = 1e-4

    config.model_name = 'PoN_Ens_GMM'
    config.model = config_dict.ConfigDict()
    config.model.size = 5
    config.model.learn_weights = False
    config.model.noise = 'homo'
    config.model.alpha = 1.
    config.model.K = 2.
    config.model.learn_weights_gmm = False

    config.model.net = config_dict.ConfigDict()
    config.model.net.depth = 2
    config.model.net.hidden_size = 50
    out_size_mult = 1.
    if config.model.noise == 'hetero':
        out_size_mult += 1
    if config.model.learn_weights_gmm:
        out_size_mult += 1
    config.model.net.out_size = int(config.model.K * out_size_mult)

    return config
