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
  data.num_n_ellipse = 70
  data.validation = validation = ml_collections.ConfigDict()
  validation.num_images = 10

  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 30
  forward_op.trafo_name = 'simple_trafo'

  # model
  config.model.attention_resolutions = [config.data.im_size // 16, config.data.im_size // 8]



  config.sampling.load_model_from_path = "/localdata/AlexanderDenker/score_based_baseline/checkpoints/2023_02_17_18:02"
  config.sampling.model_name = "model.pt"


  return config