import ml_collections

def get_default_configs(args):

    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1
    # sde configs
    config.sde = sde = ml_collections.ConfigDict()
    sde.type = args.sde # 'vpsde', 'vesde', 'ddpm'
    # the largest noise scale sigma_max was choosen according to Technique 1 from [https://arxiv.org/pdf/2006.09011.pdf], 
    # i.e. to be as large as the maximum eucledian distance between pairs of data -> ~100
    if args.sde in ['vesde', 'vpsde']:
        sde.sigma_min = 0.01
        sde.sigma_max = 100
        sde.beta_min = 0.1
        sde.beta_max = 10
    elif args.sde in ['ddpm']:
        sde.beta_min = 0.0001
        sde.beta_max = 0.02
        sde.num_steps = 1000
    else:
        raise NotImplementedError(args.sde)

    # training configs
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 3
    training.epochs = 100
    training.log_freq = 25
    training.lr = 1e-4
    training.ema_decay = 0.999
    training.ema_warm_start_steps = 400 # only start updating ema after this amount of steps 
    training.save_model_every_n_epoch = 25

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()
    validation.batch_size = 6
    validation.snr = 0.05
    validation.num_steps = 500
    if args.sde in ['ddpm']:
        validation.num_steps = 100
    validation.eps = 1e-3
    validation.sample_freq = 0 #1 # 0 = NO VALIDATION SAMPLES DURING TRAINING

    # sampling configs 
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1
    sampling.eps = 1e-3
    if args.sde in ['ddpm']:
        sampling.travel_length = 1
        sampling.travel_repeat = 1
    
    # data configs - specify in other configs
    config.data = ml_collections.ConfigDict()

    # forward operator config - specify in other configs
    config.forward_op = ml_collections.ConfigDict()

    # model configs
    config.model = model = ml_collections.ConfigDict()
    if args.sde in ['vesde', 'vpsde']:
        model.max_period = 0.005
    elif args.sde in ['ddpm']:
        model.max_period = 1e4
    else:
        raise NotImplementedError(args.sde)

    return config