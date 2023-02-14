import torch 
import functools 
import numpy as np 
import matplotlib.pyplot as plt 

import odl
from odl import uniform_discr
from odl.contrib.torch import OperatorModule

from src import (EllipseDatasetFromDival, marginal_prob_std, diffusion_coeff, ScoreNet, pc_sampler)

def coordinator():

  device = "cuda"

  marginal_prob_std_fn = functools.partial(
      marginal_prob_std, 
      sigma=25
    )
  diffusion_coeff_fn = functools.partial(
      diffusion_coeff, 
      sigma=25
    )
  
  score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
  score_model.load_state_dict(
      torch.load("/home/jleuschn/riccardo/mcg_diffusion_baseline/src/model.pt"))
  score_model = score_model.to(device)
  score_model.eval()

  ellipse_dataset = EllipseDatasetFromDival(impl="astra_cuda")

  num_angles = 30
  domain = uniform_discr([-64, -64], [64, 64], (128,128) , dtype=np.float32)
  geometry = odl.tomo.parallel_beam_geometry(
      domain, num_angles=num_angles
    )
  ray_trafo = odl.tomo.RayTransform(
    domain, 
    geometry, 
    impl="astra_cuda"
  )
  ray_trafo_op = OperatorModule(ray_trafo)
  ray_trafo_adjoint_op = OperatorModule(ray_trafo.adjoint)

  # TODO : 
  train_dl = ellipse_dataset.get_trainloader(batch_size=1, num_data_loader_workers=0)
  _, x_gt = next(iter(train_dl))
  x_gt = x_gt.to(device)

  y = ray_trafo_op(x_gt)
  rho = 0.05*torch.mean(torch.abs(y))
  y_noise = y + rho*torch.randn_like(y).to(device)

  img_size = (1, 128, 128)
  signal_to_noise_ratio = 0.05
  ## The number of sampling steps.
  num_steps = 3000

  x_mean = pc_sampler(score_model=score_model, marginal_prob_std=marginal_prob_std_fn, ray_trafo_op=ray_trafo_op, 
      ray_trafo_adjoint_op=ray_trafo_adjoint_op, diffusion_coeff=diffusion_coeff_fn, y_noise=y_noise, rho=rho, 
      batch_size=64, num_steps=num_steps, snr=signal_to_noise_ratio, device=device, eps=1e-3) 

  fig, (ax1, ax2) = plt.subplots(1,2)
  ax1.imshow(x_mean[0,0,:,:].cpu(), cmap="gray")
  ax2.imshow(x_gt[0,0,:,:].cpu(), cmap="gray")
  fig.savefig('test.png')

if __name__ == '__main__':
  coordinator()