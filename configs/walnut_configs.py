import ml_collections
from .default_config import get_default_configs

def get_config(args):
  config = get_default_configs(args)

  # data
  data = config.data

  # forward operator
  forward_op = config.forward_op
  
  data = config.data
  data.name = 'Walnut'
  data.im_size = 501
  data.new_shape = (256, 256)
  data.data_path = '/localdata/jleuschn/Walnuts/'  # insert "/path/to/Walnuts/", which should contain a sub-folder "Walnut1/" extracted from Walnut1.zip, download from: https://zenodo.org/record/2686726/files/Walnut1.zip?download=1
  data.walnut_id = 1
  data.fold = 'test'
  data.scaling_factor = 14.  # scale values to approximately [0., 1.]
  data.validation = validation = ml_collections.ConfigDict()
  data.validation.num_images = 1 
  data.stddev = 0.05

  # forward operator
  forward_op = config.forward_op
  forward_op.trafo_name = 'walnut_trafo'
  forward_op.orbit_id = 2
  forward_op.angular_sub_sampling = 20  # 1200 -> 60
  forward_op.proj_col_sub_sampling = 6  # 768 -> 128

  # model
  config.model.model_name = 'OpenAiUNetModel'
  config.model.attention_resolutions = [config.data.im_size // 16, config.data.im_size // 8]

  config.sampling.load_model_from_path = ''
  config.sampling.model_name = 'model.pt'


  return config


