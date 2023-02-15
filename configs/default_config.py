import ml_collections


def get_default_configs():

    config = ml_collections.ConfigDict()
    config.device = "cuda"
    config.seed = 1

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
    training.ema_warm_start_steps = 100 # only start updating ema after this amount of steps 

    # sampling configs 
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1
    sampling.snr = 0.05
    sampling.num_steps = 2000
    sampling.eps = 1e-3
    sampling.sampling_strategy = "predictor_corrector"

    # data configs - specify in other configs
    config.data = ml_collections.ConfigDict()

    # forward operator config - specify in other configs
    config.forward_op = ml_collections.ConfigDict()

    # model configs
    config.model = model = ml_collections.ConfigDict()
    # TODO: add hyperparemeters for model 




    return config