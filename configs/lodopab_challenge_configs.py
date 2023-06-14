import ml_collections

from .default_config import get_default_configs


def get_config(args):
  config = get_default_configs(args)

  # data
  data = config.data
  data.name = 'LoDoPabCT'
  data.im_size = 362
  data.validation = validation = ml_collections.ConfigDict()
  validation.num_images = 100

  # forward operator
  forward_op = config.forward_op

  # model
  config.model.attention_resolutions = [8, 16, 32]
  config.sampling.load_model_from_path = ''
  config.sampling.model_name = 'model.pt'

  return config