import ml_collections

from .default_config import get_default_configs


def get_config(args):
  config = get_default_configs(args)

  # data
  data = config.data
  data.name = 'LoDoPabCT'
  data.im_size = 501
  data.stddev = 0.05
  data.validation = validation = ml_collections.ConfigDict()
  validation.num_images = 5

  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 200
  forward_op.trafo_name = 'simple_trafo'
  forward_op.impl = 'iradon'

  # model
  config.model.attention_resolutions = [16, 32]
  config.sampling.load_model_from_path = ''
  config.sampling.model_name = 'model.pt'

  return config