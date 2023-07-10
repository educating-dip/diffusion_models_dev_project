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
  validation.num_images = 10

  data.part = "val"

  # forward operator
  forward_op = config.forward_op
  forward_op.num_angles = 60
  forward_op.trafo_name = 'simple_trafo'
  forward_op.impl = 'odl'

  # model
  config.model.attention_resolutions = [16, 32]
  config.sampling.load_model_from_path = ''
  config.sampling.model_name = 'model.pt'

  return config