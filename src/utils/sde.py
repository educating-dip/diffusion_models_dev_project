'''
Based on the variance exploding (VE-) SDE.
The derivations are given in [https://arxiv.org/pdf/2011.13456.pdf] Appendix C.
'''
from typing import Any, Optional
import torch 
import numpy as np 
from torch import Tensor

def _marginal_prob_std(t, sigma_min=0.01, sigma_max=50, device='cuda'):
  '''Compute the mean and **standard deviation** of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  '''    
  return sigma_min * (sigma_max / sigma_min) ** t

def _diffusion_coeff(t, sigma_min=0.01, sigma_max=50, device: Optional[Any] = None):
  '''Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  '''
  sigma = sigma_min * (sigma_max / sigma_min) ** t
  return sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=device))

class SDE: 
  def __init__(self, sigma_max: float, sigma_min: float, device: Optional[Any] = None) -> None:
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max 
    self.device = device if device is not None else 'cuda'

  def diffusion_coeff(self, t) -> float: 
    return _diffusion_coeff(
                t=t, 
                sigma_min=self.sigma_min, 
                sigma_max=self.sigma_max, 
                device=self.device
                )

  def marginal_prob_std(self, t) -> Tensor:
    return _marginal_prob_std(
                t=t, 
                sigma_min=self.sigma_min, 
                sigma_max=self.sigma_max, 
                device=self.device
                )