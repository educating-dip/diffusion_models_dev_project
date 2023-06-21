import ml_collections

from .default_config import get_default_configs


def get_config(args):
  config = get_default_configs(args)

  # data
  data = config.data
  data.name = 'LoDoPabCT'
  data.im_size = 256
  data.stddev = 0.01 #0.025#0.05
  data.part = "val"
  data.validation = validation = ml_collections.ConfigDict()

  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 60
  forward_op.trafo_name = 'simple_trafo'
  forward_op.impl = 'odl' #'iradon'

  # model
  config.model.attention_resolutions = [16, 32]
  config.sampling.load_model_from_path = ''
  config.sampling.model_name = 'model.pt'

  return config