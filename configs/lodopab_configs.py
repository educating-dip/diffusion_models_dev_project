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

  # model
  config.model.attention_resolutions = [16, 32]
  #config.model.attention_resolutions = [32, 64]
  # for im_size of 501 this evaluates to [31, 62] 
  # because of the specific way of how OpenAI builds the network it will only include these attention block if 
  # the resolution is [1, 2, 4, 8, 16, 32, 64]
  # so the choice of [config.data.im_size // 16, config.data.im_size // 8] results in no extra attention blocks
  # in higher scales 
  # the lowest scale "self.middle_block" always has attention



  config.sampling.load_model_from_path = "/localdata/AlexanderDenker/score_based_baseline/LoDoPabCT/checkpoints/2023_04_12_10:04" #2023_02_22_10:02"

  config.sampling.model_name = "model.pt"


  return config