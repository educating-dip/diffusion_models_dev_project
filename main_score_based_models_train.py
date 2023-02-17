import os
import torch 
import functools 
import numpy as np 
import matplotlib.pyplot as plt 
import numpy as np 
from datetime import datetime 


from src import (loss_fn, ExponentialMovingAverage, marginal_prob_std, diffusion_coeff, 
  OpenAiUNetModel, EllipseDatasetFromDival, get_disk_dist_ellipses_dataset, score_model_simple_trainer)


#from configs.ellipses_configs import get_config
from configs.disk_ellipses_configs import get_config

def coordinator():

  config = get_config()
  marginal_prob_std_fn = functools.partial(
      marginal_prob_std, 
      sigma=config.sde.sigma
    )
  diffusion_coeff_fn = functools.partial(
      diffusion_coeff, 
      sigma=config.sde.sigma
    )

  if config.data.name == 'EllipseDatasetFromDival':
    
    ellipse_dataset = EllipseDatasetFromDival(
        impl="astra_cuda"
      )
    train_dl = ellipse_dataset.get_trainloader(
        batch_size=config.training.batch_size, 
        num_data_loader_workers=0
      )

  elif config.data.name == 'DiskDistributedEllipsesDataset':
    
    train_dl = get_disk_dist_ellipses_dataset(
            fold='train', 
            im_size=config.data.im_size, 
            length=config.data.length,
            diameter=config.data.diameter,
            max_n_ellipse=config.data.num_n_ellipse, 
            device=config.device
          )

  if config.model.model_name == 'OpenAiUNetModel': 

    score_model = OpenAiUNetModel(
                    image_size=config.data.im_size,
                    in_channels=config.model.in_channels,
                    model_channels=config.model.model_channels,
                    out_channels=config.model.out_channels,
                    num_res_blocks=config.model.num_res_blocks,
                    attention_resolutions=config.model.attention_resolutions,
                    marginal_prob_std=marginal_prob_std_fn,
                    channel_mult=config.model.channel_mult,
                    conv_resample=config.model.conv_resample,
                    dims=config.model.dims,
                    num_heads=config.model.num_heads,
                    num_head_channels=config.model.num_head_channels,
                    num_heads_upsample=config.model.num_heads_upsample,
                    use_scale_shift_norm=config.model.use_scale_shift_norm,
                    resblock_updown=config.model.resblock_updown,
                    use_new_attention_order=config.model.use_new_attention_order
                    )
  else:

    raise NotImplementedError 
                  
  print(f" # Parameters: {sum([p.numel() for p in score_model.parameters()]) }")
  today = datetime.now()
  score_model_simple_trainer(
    score_model=score_model.to(config.device),
    marginal_prob_std_fn=marginal_prob_std_fn,
    diffusion_coeff_fn=diffusion_coeff_fn,
    train_dl=train_dl,
    optim_kwargs={
        'epochs': config.training.epochs,
        'lr': config.training.lr,
        'ema_warm_start_steps': config.training.ema_warm_start_steps,
        'log_freq': config.training.log_freq,
        'ema_decay': config.training.ema_decay
      },
      val_kwargs={
        'batch_size': config.validation.batch_size,
        'num_steps': config.validation.num_steps,
        'snr': config.validation.snr,
        'eps': config.validation.eps,
        'sample_freq' : config.validation.sample_freq
      },
    device=config.device,
    log_dir='/localdata/AlexanderDenker/score_based_baseline/checkpoints/' + today.strftime('%Y_%m_%d_%H:%m')
  )


if __name__ == '__main__': 
  coordinator()