import ml_collections
from .default_config import get_default_configs

def get_config(args):
  config = get_default_configs(args)

  # data
  data = config.data

  # forward operator
  forward_op = config.forward_op
  
  data = config.data
  data.name = 'AAPM'
  data.im_size = 256
  data.base_path = '/localdata/AlexanderDenker/score_based_baseline/AAPM/256_sorted/256_sorted/L067'
  data.part = 'test'

  data.validation = validation = ml_collections.ConfigDict()
  data.validation.num_images = 56
  data.stddev = 0.01 

  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 60
  forward_op.trafo_name = 'simple_trafo'
  forward_op.impl = 'odl'
  
  sampling = config.sampling 
  sampling.beta_schedule = 'linear'

  model = config.model 
  model.in_channels = 1
  model.out_channels = 2
  model.num_channels = 256
  model.num_heads = 4
  model.num_res_blocks = 1
  model.attention_resolutions = '16'
  model.dropout = 0.0
  model.resamp_with_conv = True
  model.learn_sigma = True
  model.use_scale_shift_norm = True
  model.use_fp16 = False
  model.resblock_updown = True
  model.num_heads_upsample = -1
  model.var_type = 'fixedsmall'
  model.num_head_channels = 64
  model.image_size = 256
  model.use_new_attention_order = False
  model.channel_mult = ''

  return config


