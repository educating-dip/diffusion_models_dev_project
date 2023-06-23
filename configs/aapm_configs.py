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
  

  return config


