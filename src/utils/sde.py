'''
Based on the variance exploding (VE-) SDE.
The derivations are given in [https://arxiv.org/pdf/2011.13456.pdf] Appendix C.

Based on: https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py
'''
from typing import Any, Optional
import torch 
import numpy as np 
from torch import Tensor

import abc


class SDE(abc.ABC):
"""SDE abstract class. Functions are designed for a mini-batch of inputs."""
	def __init__(self):
	"""Construct an SDE.

	"""
		super().__init__()

	@abc.abstractmethod
	def sde(self, x, t):
		"""
		Outputs f and G
		"""
		pass

	@abc.abstractmethod
	def marginal_prob(self, x, t):
		"""Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
		pass

	@abs.abstractmethod
	def marginal_prob_std(self, t):
		pass 

	@abc.abstractmethod
	def prior_sampling(self, shape):
		"""Generate one sample from the prior distribution, $p_T(x)$."""
		pass


class VESDE(SDE):
	def __init__(self, sigma_min=0.01, sigma_max=50):
		"""Construct a Variance Exploding SDE.
		Args:
		sigma_min: smallest sigma.
		sigma_max: largest sigma.
		"""
    	super().__init__()

		self.sigma_min = sigma_min
		self.sigma_max = sigma_max

	def diffusion_coeff(self, t)
		sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
		diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
													device=t.device))

		return diffusion 

	def sde(self, x, t):
		drift = torch.zeros_like(x)
		diffusion = self.diffusion_coeff(t)

		return drift, diffusion

	def marginal_prob(self, x, t):
		"""
		mean and standard deviation of p_{0t}(x(t) | x(0))
	
		"""
		std = self.marginal_prob_std(t)
    	mean = x
    	return mean, std

	def marginal_prob_std(self, t):
		"""
		standard deviation of p_{0t}(x(t) | x(0)) is used:
			- in the UNET as a scaling of the output 
		"""
		std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
		return std 

	def prior_sampling(self, shape):
		return torch.randn(*shape) * self.sigma_max


class VPSDE(SDE):
	def __init__(self, beta_min, beta_max)
	"""Construct a Variance Preserving SDE.
	Args:
		beta_min: value of beta(0)
		beta_max: value of beta(1)

    """
	super().__init__()
	self.beta_min = beta_min
	self.beta_max = beta_max

	def diffusion_coeff(self, t)
		beta_t = self.beta_min + t*(self.beta_max - self.beta_min)
		return torch.sqrt(beta_t)

	def sde(self, x, t):
		beta_t = self.beta_min + t*(self.beta_max - self.beta_min)
		drift = -0.5 * beta_t[:, None, None, None] * x 

		diffusion = self.diffusion_coeff(t)
		return drift, diffusion

	def marginal_prob(self, x, t):
		"""
		mean and standard deviation of p_{0t}(x(t) | x(0))
	
		"""
		std = self.marginal_prob_std(t)
		log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
		mean = torch.exp(log_mean_coeff[:, None, None, None]) * x		
    	return mean, std

	def marginal_prob_std(self, t):
		"""
		standard deviation of p_{0t}(x(t) | x(0)) is used:
			- in the UNET as a scaling of the output 
		"""
		log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_max
		std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
		return std 

	def prior_sampling(self, shape):
		return torch.randn(*shape) 