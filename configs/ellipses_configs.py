import ml_collections

from .default_config import get_default_configs


def get_config():
  config = get_default_configs()

  # data
  data = config.data
  data.name = 'EllipseDatasetFromDival'
  data.im_size = 128
  data.stddev = 0.05
  data.validation = validation = ml_collections.ConfigDict()
  validation.num_images

  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 30
  forward_op.trafo_name = 'simple_trafo'

  config.sampling.load_model_from_path = None
  config.sampling.model_name = None

  config.model.model_name = 'OpenAiUNetModel'

  return config