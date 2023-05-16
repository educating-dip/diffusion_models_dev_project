import ml_collections

def get_default_configs():

    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    # sde configs
    config.sde = sde = ml_collections.ConfigDict()
    sde.type = 'vpsde' #'vesde'
    #sde.sigma = 25.0
    # the largest noise scale sigma_max was choosen according to Technique 1 from [https://arxiv.org/pdf/2006.09011.pdf], 
    # i.e. to be as large as the maximum eucledian distance between pairs of data -> ~100
    sde.sigma_min = 0.01
    sde.sigma_max = 100

    sde.beta_min = 0.1
    sde.beta_max = 10

    # training configs
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 6
    training.epochs = 1000
    training.log_freq = 25
    training.lr = 1e-4
    training.ema_decay = 0.999
    training.ema_warm_start_steps = 400 # only start updating ema after this amount of steps 

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()
    validation.batch_size = 6
    validation.snr = 0.05
    validation.num_steps = 1000
    validation.eps = 1e-3
    validation.sample_freq = 1

    # sampling configs 
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1
    sampling.eps = 1e-3
    
    # data configs - specify in other configs
    config.data = ml_collections.ConfigDict()

    # forward operator config - specify in other configs
    config.forward_op = ml_collections.ConfigDict()

    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.model_name = 'OpenAiUNetModel'
    model.in_channels = 1
    model.model_channels = 64
    model.out_channels = 1
    model.num_res_blocks = 2
    #model.attention_resolutions = [config.data.im_size // 16, config.data.im_size // 8]
    model.channel_mult = (0.5, 1, 1, 2, 2, 2, 4)
    model.conv_resample = True
    model.dims = 2
    model.num_heads = 1
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.use_scale_shift_norm = True 
    model.resblock_updown = False
    model.use_new_attention_order = False

    return config