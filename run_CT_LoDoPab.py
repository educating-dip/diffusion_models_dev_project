"""
Alex:

In this script the pre-trained score model will be applied to ground truth images of the LoDoPab dataset.
The images are first downsampled (cropped?) to 256x256. 

The radon transform is defined using ODL.

"""

### external libraries
import matplotlib.pyplot as plt 
import numpy as np 

from dival import get_standard_dataset

import torch 
from torch.nn.functional import interpolate

import odl
from odl import uniform_discr
from odl.contrib.torch import OperatorModule
from odl.operator.oputils import power_method_opnorm
import datasets
import time

### internal imports
from models.ema import ExponentialMovingAverage
import controllable_generation
from utils import restore_checkpoint, show_samples_gray, clear, lambda_schedule_const, lambda_schedule_linear
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)

from configs.ve import AAPM_256_ncsnpp_continuous as configs

random_seed = 0

#######################
### Hyperparameters ###
#######################
idx = 29
rel_noise = 0.0 #0.025
num_angles = 90

solver = 'naive' #'MCG'
config_name = 'AAPM_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 185
N = num_scales

ckpt_filename = f"/localdata/AlexanderDenker/checkpoints/{config_name}/checkpoint_{ckpt_num}.pt" 
config = configs.get_config()
config.model.num_scales = N

snr = 0.16 # signal-to-noise ratio of target. Is used in Langevin step to scale the step size (0.16 was used by the authors)
n_steps = 1 # how many Langevin steps per corrector step (default=1)

predictor = ReverseDiffusionPredictor 
corrector = LangevinCorrector
probability_flow = False # if True, then add no noise during sampling (deterministic sampling)

start_N = 1500 # shortcut (come closer, diffuse faster) start_N = 0 <-> normal sampling


# Specify save directory for saving generated samples
save_root = Path(f'./results/LoDoPab/num_angles={num_angles}/solver={solver}/start_N={start_N}/rel_noise={rel_noise}/img={idx}')
save_root.mkdir(parents=True, exist_ok=True)

irl_types = ['input', 'recon', 'gt']
for t in irl_types:
    save_root_f = save_root / t
    save_root_f.mkdir(parents=True, exist_ok=True)

### Load Dataset and create Forward Operator ###

dataset = get_standard_dataset('lodopab', impl="astra_cuda", sorted_by_patient=True)

lodopab_train = dataset.create_torch_dataset(part='train',
                                    reshape=((1,) + dataset.space[0].shape,
                                    (1,) + dataset.space[1].shape))


_, x_gt = lodopab_train[idx]
x_gt = interpolate(x_gt.unsqueeze(0), (256,256), mode="bilinear", align_corners=True)

domain = uniform_discr([-0.13, -0.13], [0.13, 0.13], (256,256), dtype=np.float32)
geometry = odl.tomo.parallel_beam_geometry(domain, num_angles=num_angles)

ray_trafo = odl.tomo.RayTransform(domain, geometry, impl="astra_cuda")

print("OP NORM: ", power_method_opnorm(ray_trafo))

ray_trafo_op = OperatorModule(ray_trafo)
ray_trafo_adjoint_op = OperatorModule(ray_trafo.adjoint)
fbp_op = OperatorModule(odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo, frequency_scaling=0.75, filter_type='Hann'))


y = ray_trafo_op(x_gt)
y_noise = y + rel_noise*torch.mean(torch.abs(y))*torch.randn_like(y)

x_fbp = fbp_op(y_noise)

plt.imsave(save_root / 'gt' / f'{str(idx).zfill(4)}_gt.png', clear(x_gt), cmap='gray')
plt.imsave(save_root / 'gt' / f'{str(idx).zfill(4)}_fbp.png', clear(x_fbp), cmap='gray')

### Set up SDE ###

# step size for data discrepancy part
schedule = 'linear'
start_lamb = 1.0
end_lamb = 0.9#0.6
lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)

sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sde.N = N

batch_size = 1
config.training.batch_size = batch_size
config.eval.batch_size = batch_size


sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config) # scale [0,1] -> [-1,1]
inverse_scaler = datasets.get_data_inverse_scaler(config) # scale [-1,1] -> [0,1]
score_model = mutils.create_model(config)

ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)

state = dict(step=0, model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
ema.copy_to(score_model.parameters())


x_gt = x_gt.to(config.device)
y_noise = y_noise.to(config.device)

if solver == 'MCG':

    pc_MCG = controllable_generation.get_pc_radon_MCG(sde,
                                                      predictor, corrector,
                                                      inverse_scaler,
                                                      snr=snr,
                                                      n_steps=n_steps,
                                                      probability_flow=probability_flow,
                                                      continuous=config.training.continuous,
                                                      denoise=True,
                                                      radon=radon,
                                                      radon_all=radon_all,
                                                      weight=0.1,
                                                      save_progress=True,
                                                      save_root=save_root,
                                                      lamb_schedule=lamb_schedule,
                                                      mask=mask)
    x = pc_MCG(score_model, scaler(img), measurement=sinogram_noise, start_N=start_N, save_freq=4)

elif solver == 'song':
    pc_song = controllable_generation.get_pc_radon_song(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        save_progress=True,
                                                        save_root=save_root,
                                                        denoise=True,
                                                        radon=radon_all,
                                                        lamb=0.7)
    x = pc_song(score_model, scaler(img), mask, measurement=sinogram_full)

elif solver == 'naive' :
    
    pc_naive = controllable_generation.get_pc_radon_naive(sde,
                                                      predictor, corrector,
                                                      inverse_scaler,
                                                      snr=snr,
                                                      n_steps=n_steps,
                                                      probability_flow=probability_flow,
                                                      continuous=config.training.continuous,
                                                      denoise=True,
                                                      ray_trafo=ray_trafo_op,
                                                      ray_trafo_adjoint=ray_trafo_adjoint_op,
                                                      fbp_op=fbp_op,
                                                      save_progress=True,
                                                      save_root=save_root,
                                                      lamb_schedule=lamb_schedule)

    x = pc_naive(score_model, scaler(x_gt), measurement=y_noise, start_N=start_N, save_freq=50)


# Recon
plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x), cmap='gray')
plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}_clip.png'), np.clip(clear(x), 0.1, 1.0), cmap='gray')