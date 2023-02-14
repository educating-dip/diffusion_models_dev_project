
import torch 
from torch.optim import Adam
import functools 
import numpy as np 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 

import odl
from odl import uniform_discr
from odl.contrib.torch import OperatorModule
from odl.operator.oputils import power_method_opnorm

from dataset import EllipseDataset
from model import ScoreNet
from sde import marginal_prob_std, diffusion_coeff
from ema import ExponentialMovingAverage

device = "cuda"
sigma =  25.0


marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
#ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)
#ema.load_state_dict(torch.load("checkpoints/ema_model.pt"))
#ema.copy_to(score_model.parameters())
score_model.load_state_dict(torch.load("checkpoints/model.pt", map_location=torch.device('cpu')))
score_model.eval()
score_model = score_model.to(device)

ellipse_dataset = EllipseDataset(impl="astra_cuda")

### set up forward operator 
num_angles = 30
domain = uniform_discr([-64, -64], [64, 64], (128,128), dtype=np.float32)
geometry = odl.tomo.parallel_beam_geometry(domain, num_angles=num_angles)

ray_trafo = odl.tomo.RayTransform(domain, geometry, impl="astra_cuda")

print(ray_trafo)

norm_const = power_method_opnorm(ray_trafo)

print("OP NORM: ", norm_const)
print("OP NORM of adjoint: ", power_method_opnorm(ray_trafo.adjoint))

ray_trafo_op = OperatorModule(ray_trafo)
ray_trafo_adjoint_op = OperatorModule(ray_trafo.adjoint)

train_dl = ellipse_dataset.get_trainloader(batch_size=1, num_data_loader_workers=0)


_, x_gt = next(iter(train_dl))
x_gt = x_gt.to(device)

y = ray_trafo_op(x_gt)
rho = 0.05*torch.mean(torch.abs(y))
y_noise = y + rho*torch.randn_like(y).to(device)

#fig, (ax1, ax2) = plt.subplots(1,2)
#ax1.imshow(y[0,0,:,:].cpu())
#x_adj = ray_trafo_adjoint_op(y)
#ax2.imshow(x_adj[0,0,:,:].cpu())
#plt.show()

img_size = (1, 128, 128)

num_steps = 1000
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=16, 
                           num_steps=num_steps, 
                           device='cuda', 
                           eps=1e-3):

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 128, 128, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    print(step_size)
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
        # Do not include any noise in the last sampling step.
    return mean_x

signal_to_noise_ratio = 0.05#0.16 

## The number of sampling steps.
num_steps =  3000
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               y_noise, 
               batch_size=64, 
               num_steps=num_steps, 
               snr=signal_to_noise_ratio,                
               device='cuda',
               eps=1e-3):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 128, 128, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  for time_step in tqdm(time_steps):     

    with torch.no_grad():
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

    x = x.requires_grad_()
    s = score_model(x, batch_time_step)
    std = marginal_prob_std_fn(batch_time_step)
    xhat0 = x + s * std[:, None, None, None]
    norm = torch.linalg.norm(ray_trafo_adjoint_op(y_noise - ray_trafo_op(xhat0)))
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

    #im = ax1.imshow(s[0,0,:,:].detach().cpu())
    #fig.colorbar(im, ax=ax1)
    #ax1.set_title("score")

    #im = ax2.imshow(norm_grad[0,0,:,:].detach().cpu())
    #fig.colorbar(im, ax=ax2)
    #ax2.set_title("norm grad")

    #im = ax3.imshow(ray_trafo_adjoint_op(y_noise - ray_trafo_op(xhat0))[0,0,:,:].detach().cpu())
    #fig.colorbar(im, ax=ax3)
    #ax3.set_title("residual")

    #im = ax4.imshow(xhat0[0,0,:,:].detach().cpu())
    #fig.colorbar(im, ax=ax4)
    #ax4.set_title("x hat 0")
    #plt.show() 

    # Predictor step (Euler-Maruyama)
    g = diffusion_coeff(batch_time_step)
    x_mean = x + (g**2)[:, None, None, None] * (s - 1/rho**2 * norm_grad) * step_size
    x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

    x = x.detach()
    x_mean = x_mean.detach()
    # The last step does not include any noise
  return x_mean

    
  
#x_mean = Euler_Maruyama_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, num_steps=num_steps)
x_mean = pc_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, y_noise, num_steps=num_steps, snr=signal_to_noise_ratio, batch_size=1,device=device)

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(x_mean[0,0,:,:].cpu(), cmap="gray")
ax2.imshow(x_gt[0,0,:,:].cpu(), cmap="gray")

plt.show()

"""
fig, axes = plt.subplots(4,4)

for idx, ax in enumerate(axes.ravel()):
    ax.imshow(x_mean[idx, 0,:,:].cpu(), cmap="gray")
    ax.axis("off")


plt.show()
"""