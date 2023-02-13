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
from tqdm import tqdm 

### internal imports
from models.ema import ExponentialMovingAverage
import controllable_generation
from utils import restore_checkpoint, show_samples_gray, clear, lambda_schedule_const, lambda_schedule_linear
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from models.utils import get_score_fn


from configs.ve import AAPM_256_ncsnpp_continuous as configs

random_seed = 0

#######################
### Hyperparameters ###
#######################
idx = 400
rel_noise = 0.05 #0.025
num_angles = 180

solver = 'naive' #'MCG'
config_name = 'AAPM_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 185
N = num_scales
eps=1e-5 # t_0 = eps (t_0 = 0 would apparently be unstable [not tested by myself])

ckpt_filename = f"/localdata/AlexanderDenker/checkpoints/{config_name}/checkpoint_{ckpt_num}.pt" 
config = configs.get_config()
config.model.num_scales = N

probability_flow = False # if True, then add no noise during sampling (deterministic sampling)

start_N = 1820 # shortcut (come closer, diffuse faster) start_N = 0 <-> normal sampling

save_progress = True 
save_freq = 20

snr = 0.8#0.16 # for step size in Langevin Corrector 

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


#_, x_gt = lodopab_train[idx]
#x_gt = interpolate(x_gt.unsqueeze(0), (256,256), mode="bilinear", align_corners=True).transpose(-2, -1)
x_gt = torch.from_numpy(np.load("samples/CT/0100.npy"))
h, w = x_gt.shape
x_gt = x_gt.view(1, 1, h, w)
x_gt = x_gt.to(config.device)


print("Range of x_gt: ", torch.min(x_gt), torch.max(x_gt))

domain = uniform_discr([-128, -128], [128, 128], (256,256), dtype=np.float32)
geometry = odl.tomo.parallel_beam_geometry(domain, num_angles=num_angles)

ray_trafo = odl.tomo.RayTransform(domain, geometry, impl="astra_cuda")

norm_const = power_method_opnorm(ray_trafo)

print("OP NORM: ", norm_const)
print("OP NORM of adjoint: ", power_method_opnorm(ray_trafo.adjoint))

ray_trafo_op = OperatorModule(ray_trafo)
ray_trafo_adjoint_op = OperatorModule(ray_trafo.adjoint)

ones = torch.ones(x_gt.shape)
norm_const_ye = torch.mean(ray_trafo_adjoint_op(ray_trafo_op(ones)))

print("OP NORM YE: ", norm_const_ye)

fbp_op = OperatorModule(odl.tomo.fbp_op(ray_trafo, frequency_scaling=0.95))

y = ray_trafo_op(x_gt)
noise_level =  rel_noise*torch.mean(torch.abs(y))
print("NOISE LEVEL: ", noise_level)
y_noise = y + noise_level*torch.randn_like(y)

x_fbp = fbp_op(y_noise)
x_adj = ray_trafo_adjoint_op(y_noise)

print("Range of x_fbp: ", torch.min(x_fbp), torch.max(x_fbp))
print("Range of x_adj: ", torch.min(x_adj), torch.max(x_adj))

plt.imsave(save_root / 'gt' / f'{str(idx).zfill(4)}_gt.png', clear(x_gt), cmap='gray')
plt.imsave(save_root / 'gt' / f'{str(idx).zfill(4)}_fbp.png', clear(x_fbp), cmap='gray')

### Set up SDE ###

# step size for data discrepancy part
schedule = 'linear'
start_lamb = 0.1
end_lamb = 0.1#0.6
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
rho = 1 / noise_level**2


score_fn = get_score_fn(sde, score_model, train=False, continuous=config.training.continuous)

rsde = sde.reverse(score_fn, probability_flow=False)

# IF WE SCALE X WE NEED TO MAKE SURE THAT WE SCALE IT BACK
# ELSE IT WOULDNT FIT TO OUR FORWARD OPERATOR AND MEASUREMENTS?

# reverse diffusion update (predictor)
def reverse_diffusion_update(x, t, y):
    vec_t = torch.ones(x_gt.shape[0], device=x_gt.device) * t

    z = torch.randn_like(x)
        
    x = x.requires_grad_()
    
    f, G, score = rsde.discretize(x, vec_t)

    # x0 hat estimation
    _, bt = sde.marginal_prob(x, vec_t)
    hatx0 = x + (bt ** 2) * score

    norm = torch.norm(ray_trafo_adjoint_op(y - ray_trafo_op(inverse_scaler(hatx0))))
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

    x_mean = x - f - G[:, None, None, None] ** 2 * rho * norm_grad #/ np.sqrt(norm_const)


    x = x_mean + G[:, None, None, None] *  z

    x = x.detach()

    return x, x_mean

# Langevin update (corrector)
# first, no update w.r.t. posterior 
def langevin_update(x, t, y, n_steps=1):
    vec_t = torch.ones(x_gt.shape[0], device=x_gt.device) * t

    alpha = torch.ones_like(vec_t)

    for j in range(n_steps):
        x = x.requires_grad_()

        grad = score_fn(x, vec_t)

        _, bt = sde.marginal_prob(x, vec_t)
        hatx0 = x + (bt ** 2) * grad
        norm = torch.norm(ray_trafo_adjoint_op(y - ray_trafo_op(inverse_scaler(hatx0))))
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

        grad = grad - rho * norm_grad #/ np.sqrt(norm_const)

        noise = torch.randn_like(x)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha

        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        x = x.detach()
        x_mean = x_mean.detach()
    return x, x_mean


# Euler-Maruyama update (predictor)
def euler_maruyama_update(x, t, y):
    vec_t = torch.ones(x_gt.shape[0], device=x_gt.device) * t

    dt = -1. / sde.N 
    z = torch.randn_like(x)
        
    x = x.requires_grad_()
    score = score_fn(x, vec_t)

    # x0 hat estimation
    _, bt = sde.marginal_prob(x, vec_t)
    hatx0 = x + (bt ** 2) * score
    print("bt**2: ", bt ** 2)
    norm = torch.norm(ray_trafo_adjoint_op(y - ray_trafo_op(inverse_scaler(hatx0))))
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
    x = x.detach()

    score_posterior = score - rho * norm_grad #/ norm #/ norm_const

    drift, diffusion = sde.sde(x, vec_t)
    print("Diffusion at step {} : {}".format(t, diffusion))
    drift = drift - diffusion[:, None, None, None]**2 * score_posterior

    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z

    return x, x_mean

timesteps = torch.linspace(sde.T, eps, sde.N)

if start_N == 0:
    x = sde.prior_sampling(x_gt.shape).to(x_gt.device)
else: 
    x_fbp = fbp_op(y_noise)
    x_fbp = scaler(x_fbp)

    x_fbp_mean, x_fbp_std = sde.marginal_prob(x_fbp, torch.ones(x_gt.shape[0], device=x_gt.device) * timesteps[start_N])
    x_fbp_noisy = x_fbp_mean + torch.randn_like(x_fbp) * x_fbp_std[:, None, None, None]
    x = x_fbp_noisy

    plt.imsave(save_root / 'recon' / f'noisy_fbp_start_point_for_sde.png', clear(x), cmap='gray')
    plt.imsave(save_root / 'recon' / f'noisy_fbp_start_point_for_sde_mean.png', clear(x_fbp_mean), cmap='gray')


for i in tqdm(range(start_N, sde.N)):
    t = timesteps[i]
    x, _ = reverse_diffusion_update(x, t, y=y_noise)
    x, x_mean = langevin_update(x, t, y=y_noise, n_steps=1)
    #x, x_mean = euler_maruyama_update(x, t, y=y_noise)
    if save_progress:
        if (i % save_freq) == 0:
            plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x), cmap='gray')


x = inverse_scaler(x_mean)

# LAST DENOISING STEP?        

# Recon
plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x), cmap='gray')
plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}_clip.png'), np.clip(clear(x), 0.1, 1.0), cmap='gray')