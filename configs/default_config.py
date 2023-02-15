import ml_collections




def get_default_config():
    config = ml_collections.ConfigDict()

    config.device = "cuda"

    # sde configs
    config.sde = sde = ml_collections.ConfigDict()
    sde.type = "vesde"
    sde.sigma = 25.0

    # training configs
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 6
    training.epochs = 1000
    training.log_freq = 25
    training.lr = 1e-4
    training.ema_decay = 0.999

    # sampling configs 
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.snr = 0.05
    sampling.num_steps = 2000
    sampling.sampler = "pc"

    # data configs - specify in other configs
    config.data = ml_collections.ConfigDict()

    # forward operator config - specify in other configs
    config.forward_op = ml_collections.ConfigDict()

    # model configs
    config.model = model = ml_collections.ConfigDict()
    # TODO: add hyperparemeters for model 




    return config