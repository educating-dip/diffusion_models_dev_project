import ml_collections
from .default_config import get_default_configs


def get_config(args):
  config = get_default_configs(args)

  # data
  data = config.data
  data.name = 'DiskDistributedEllipsesDataset'
  data.im_size = 256
  data.length = 32000
  data.val_length = 10
  data.stddev = 0.01
  data.diameter = 0.4745
  data.num_n_ellipse = 140
  data.validation = validation = ml_collections.ConfigDict()
  validation.num_images = 100

  data.part = 'test'

  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 60
  forward_op.trafo_name = 'simple_trafo'
  forward_op.impl = 'odl'

  # model
  model = config.model 
  model.in_channels = 1
  model.out_channels = 1
  model.num_channels = 256
  model.num_heads = 4
  model.num_res_blocks = 1
  model.attention_resolutions = '16'
  model.dropout = 0.0
  model.resamp_with_conv = True
  model.learn_sigma = False
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