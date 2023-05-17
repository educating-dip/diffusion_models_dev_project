import ml_collections
from .default_config import get_default_configs


def get_config():
  config = get_default_configs()

  # data
  data = config.data
  data.name = 'DiskDistributedEllipsesDataset'
  data.im_size = 501
  data.length = 32000
  data.val_length = 10
  data.stddev = 0.05
  data.diameter = 0.4745
  data.num_n_ellipse = 140
  data.validation = validation = ml_collections.ConfigDict()
  validation.num_images = 5

  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 120
  forward_op.trafo_name = 'simple_trafo'

  # model
  config.model.attention_resolutions = [16, 32]
  config.sampling.load_model_from_path = '' #'/localdata/AlexanderDenker/score_based_baseline/DiskDistributedEllipsesDataset/checkpoints/2023_04_14_08:04'
  config.sampling.model_name = 'model.pt'

  return config