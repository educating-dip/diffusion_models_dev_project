"""
Configs for the original LoDoPab-CT dataset from Dival. 
Using the sinograms of the LoDoPab challenge.

https://jleuschn.github.io/docs.dival/_modules/dival/datasets/lodopab_dataset.html#LoDoPaBDataset
"""
import ml_collections

from .default_config import get_default_configs

def get_config():
    config = get_default_configs()

    # data
    data = config.data
    data.name = 'LoDoPabCT'
    data.im_size = 362

    config.sde.type = 'vpsde'

    config.sde.beta_min = 0.1
    config.sde.beta_max = 20

    # model
    config.sampling.load_model_from_path = None #'/localdata/AlexanderDenker/score_based_baseline/LoDoPabCT/checkpoints/2023_04_12_10:04' #2023_02_22_10:02"
    config.sampling.model_name = 'model.pt'

    config.validation.num_steps = 500
    
    config.model = model = ml_collections.ConfigDict()
    model.n_iter = 4
    model.n_primal = 4
    model.n_dual = 4
    model.n_layer = 6
    model.internal_ch = 32
    model.kernel_size = 3
    model.batch_norm = True

    return config