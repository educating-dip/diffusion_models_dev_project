import ml_collections
from .default_config import get_default_configs


def get_config():
  config = get_default_configs()

  # data
  data = config.data
  data.name = 'Mayo'
  data.im_size = 501
  data.stddev = 0.01
  data.base_path = '/localdata/jleuschn/data/LDCT-and-Projection-data'
  data.part = 'L'
  data.validation = validation = ml_collections.ConfigDict()
  validation.num_images = 5
  
  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 500
  forward_op.trafo_name = 'simple_trafo'
  
  # model
  config.model.attention_resolutions = [16, 32]
  config.sampling.load_model_from_path = '/localdata/AlexanderDenker/score_based_baseline/LoDoPabCT/checkpoints/2023_04_12_10:04'
  config.sampling.model_name = 'model.pt'

  return config