
import torch 
import functools
import numpy as np 

device = 'cuda' 


# based on the variance exploding SDE
# the derivations are given in [https://arxiv.org/pdf/2011.13456.pdf] Appendix C
def marginal_prob_std(t, sigma_min=0.01, sigma_max=50):
  """Compute the mean and **standard deviation** of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t, device=device)
  return sigma_min * (sigma_max / sigma_min) ** t

def diffusion_coeff(t, sigma_min=0.01, sigma_max=50):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  sigma = sigma_min * (sigma_max / sigma_min) ** t
  return sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=device))


"""
Old SDE based on Yang Songs Colab Notebook
dx = sigma^t dw (https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=PpJSwfyY6mJz)


def marginal_prob_std(t, sigma=25.):
 
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma=25.):

  return torch.tensor(sigma**t, device=device)
"""