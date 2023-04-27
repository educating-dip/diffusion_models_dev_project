import ml_collections

from .default_config import get_default_configs


def get_config():
  config = get_default_configs()

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

  config.sde = sde = ml_collections.ConfigDict()
  sde.type = 'vpsde'
  sde.beta_min = 0.1
  sde.beta_max = 5

  config.validation.num_steps = 500

  # model
  config.model.attention_resolutions = [16, 32]
  config.sampling.load_model_from_path = '/localdata/AlexanderDenker/score_based_baseline/LoDoPabCT/checkpoints/2023_04_26_14:04/'
  config.sampling.model_name = 'model.pt'

  config.sampling.eps = 5e-4



  return config