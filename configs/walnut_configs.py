from default_config import get_default_configs


def get_config():
  config = get_default_configs()

  # data
  data = config.data

  # forward operator
  forward_op = config.forward_op

  # model
  config.model.model_name = 'OpenAiUNetModel'


  return config