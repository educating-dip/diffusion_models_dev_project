''' 
Inspired to https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py 
'''
from typing import Optional, Any, Dict, Tuple

import os
import torchvision
import numpy as np
import torch

from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .utils import _schedule_jump
from ..utils import SDE, _EPSILON_PRED_CLASSES, _SCORE_PRED_CLASSES, PSNR
from ..third_party_models import OpenAiUNetModel

class BaseSampler:
    def __init__(self, 
        score: OpenAiUNetModel, 
        sde: SDE,
        predictor: callable,
        sample_kwargs: Dict,
        init_chain_fn: Optional[callable] = None,
        corrector: Optional[callable] = None,
        device: Optional[Any] = None
        ) -> None:

        self.score = score
        self.sde = sde
        self.predictor = predictor
        self.init_chain_fn = init_chain_fn
        self.sample_kwargs = sample_kwargs
        self.corrector = corrector
        self.device = device
    
    def sample(self,
        logg_kwargs: Dict = {},
        logging: bool = True 
        ) -> Tensor:

        if logging:
            writer = SummaryWriter(log_dir=os.path.join(logg_kwargs['log_dir'], str(logg_kwargs['sample_num'])))
        
        num_steps = self.sample_kwargs['num_steps']
        __iter__ = None
        if any([isinstance(self.sde, classname) for classname in _SCORE_PRED_CLASSES]):
            time_steps = np.linspace(
                1., self.sample_kwargs['eps'], self.sample_kwargs['num_steps'])
            __iter__ = time_steps
        elif any([isinstance(self.sde, classname) for classname in _EPSILON_PRED_CLASSES]):
            assert self.sde.num_steps >= num_steps
            skip = self.sde.num_steps // num_steps
            # if ``self.sample_kwargs['travel_length']'' is 1. and ''self.sample_kwargs['travel_repeat']'' is 1. 
            # ``_schedule_jump'' behaves as ``np.arange(-1. num_steps, 1)[::-1]''
            time_steps = _schedule_jump(num_steps, self.sample_kwargs['travel_length'], self.sample_kwargs['travel_repeat']) 
            time_pairs = list((i*skip, j*skip if j>0 else -1)  for i, j in zip(time_steps[:-1], time_steps[1:]))
            
            # implement early stopping 
            try:
                time_pairs = time_pairs[:int(self.sample_kwargs['early_stopping_pct']*len(time_pairs))]
                print("Use early stopping. Run for ", len(time_pairs), " timesteps. Stop at time step ", time_pairs[-1])
            except KeyError:
                pass

            __iter__= time_pairs
        else:
            raise NotImplementedError(self.sde.__class__ )

        step_size = time_steps[0] - time_steps[1]
        if self.sample_kwargs['start_time_step'] == 0:
            init_x = self.sde.prior_sampling([self.sample_kwargs['batch_size'], *self.sample_kwargs['im_shape']]).to(self.device)
        else:
            assert not any([isinstance(self.sde, classname) for classname in _EPSILON_PRED_CLASSES])
            init_x = self.init_chain_fn(time_steps=time_steps)
        
        if logging:
            writer.add_image('init_x', torchvision.utils.make_grid(init_x, 
                normalize=True, scale_each=True), global_step=0)
            if logg_kwargs['ground_truth'] is not None: writer.add_image(
                'ground_truth', torchvision.utils.make_grid(logg_kwargs['ground_truth'].squeeze(), 
                    normalize=True, scale_each=True), global_step=0)
            if logg_kwargs['filtbackproj'] is not None: writer.add_image(
                'filtbackproj', torchvision.utils.make_grid(logg_kwargs['filtbackproj'].squeeze(), 
                    normalize=True, scale_each=True), global_step=0)
        
        x = init_x
        i = 0
        pbar = tqdm(__iter__)
        for step in pbar:
            ones_vec = torch.ones(self.sample_kwargs['batch_size'], device=self.device)
            if isinstance(step, float): 
                time_step = ones_vec * step # t,
            elif isinstance(step, Tuple):
                time_step = (ones_vec * step[0], ones_vec * step[1]) # (t, tminus1)
            else:
                raise NotImplementedError

            if self.sample_kwargs.get('adapt_freq', None) is not None:  
                self.sample_kwargs['predictor'].update(
                        {'use_adapt': False}
                    )
                if i % self.sample_kwargs['adapt_freq'] == 0:
                    self.sample_kwargs['predictor'].update(
                        {'use_adapt': True}
                    )

            x, x_mean = self.predictor(
                score=self.score,
                sde=self.sde,
                x=x,
                time_step=time_step,
                step_size=step_size,
                datafitscale=step/self.sample_kwargs['num_steps'] if not isinstance(step, Tuple) else 1.,
                **self.sample_kwargs['predictor']
                )

            if self.corrector is not None:
                x = self.corrector(
                    x=x,
                    score=self.score,
                    sde=self.sde,
                    time_step=time_step,
                    datafitscale=step/self.sample_kwargs['num_steps'] if not isinstance(step, Tuple) else 1.,
                    **self.sample_kwargs['corrector']
                    )

            if logging:
                if (i - self.sample_kwargs['start_time_step']) % logg_kwargs['num_img_in_log'] == 0:
                    writer.add_image('reco', torchvision.utils.make_grid(x_mean.squeeze(), normalize=True, scale_each=True), i)
                
                    psnr = PSNR(x_mean[0, 0].cpu().numpy(), logg_kwargs['ground_truth'][0, 0].cpu().numpy())
                
                writer.add_scalar('PSNR', psnr, i)
                pbar.set_postfix({'psnr': psnr})
            i += 1

        if logging:
            writer.add_image(
                'final_reco', torchvision.utils.make_grid(x_mean.squeeze(),
                normalize=True, scale_each=True), global_step=0)

        return x_mean 