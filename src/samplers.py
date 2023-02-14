import torch
import numpy as np 
from tqdm import tqdm 

def pc_sampler(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                y_noise,
                rho,
                ray_trafo_op,
                ray_trafo_adjoint_op,
                batch_size, 
                num_steps, 
                snr,                
                device='cuda',
                eps=1e-3):

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
      std = marginal_prob_std(batch_time_step)
      xhat0 = x + s * std[:, None, None, None]
      norm = torch.linalg.norm(ray_trafo_adjoint_op(y_noise - ray_trafo_op(xhat0)))
      norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      x_mean = x + (g**2)[:, None, None, None] * (s - 1/rho**2 * norm_grad) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

      x = x.detach()
      x_mean = x_mean.detach()
      # The last step does not include any noise

    return x_mean

      