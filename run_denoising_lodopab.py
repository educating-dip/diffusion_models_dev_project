import argparse
import yaml 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import islice
from PIL import Image
import matplotlib.pyplot as plt 

import os
import time
from pathlib import Path

from src import (get_standard_sde, PSNR, SSIM, get_standard_dataset, LoDoPabTrafo,  
	get_standard_score, get_standard_sampler, LoDoPabChallenge, get_standard_path) 

parser = argparse.ArgumentParser(description='conditional sampling')

"""
Check how good the Tweedie estimate really is for this model.

"""

def coordinator(args):
	load_path = os.path.join("/localdata/AlexanderDenker/score_based_baseline/challenge/LoDoPabCT/ddpm/version_02")
	print('load model from: ', load_path)
	with open(os.path.join(load_path, 'report.yaml'), 'r') as stream:
		config = yaml.load(stream, Loader=yaml.UnsafeLoader)
		config.sampling.load_model_from_path = load_path

	if config.seed is not None:
		torch.manual_seed(config.seed) # for reproducible noise in simulate

	print(config)

	sde = get_standard_sde(config=config)
	
	score = get_standard_score(config=config, sde=sde, use_ema=True)
	score = score.to(config.device)
	score.eval()
	
	dataset = LoDoPabChallenge()

	#path = "/localdata/AlexanderDenker/score_results/lodopab"
	#save_root = Path(os.path.join(path, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}'))
	#save_root.mkdir(parents=True, exist_ok=True)

	test_loader = dataset.get_testloader(batch_size=12)
	_, x = next(iter(test_loader))
	print(x.shape)
	x = x.to("cuda")
	denoising_mse = [] 
	with torch.no_grad():
		for i in range(0, sde.num_steps): #range(sde.num_steps-1, 1, -1):
			t = torch.ones(x.shape[0], device=x.device) * i 
			## get noisy sample 
			z = torch.randn_like(x)
			mean, std = sde.marginal_prob(x, t)
			# Diffuse the data for a given number of diffusion steps. In other words, sample from q(x_t | x_0)
			perturbed_x = mean + z * std[:, None, None, None]
			zhat = score(perturbed_x, t)

			div = sde.marginal_prob_mean(t)[:, None, None, None].pow(-1)
			std_t = sde.marginal_prob_std(t)[:, None, None, None]
			#print(div[0,0,0,0].item(), std_t[0,0,0,0].item())
			xhat0 = (perturbed_x - zhat*std_t)*div

			#fig, (ax1, ax2, ax3) = plt.subplots(1,3)
			#fig.suptitle(t[0].item())
			#ax1.imshow(x[3, 0, :, :].cpu().numpy())
			#ax2.imshow(perturbed_x[3, 0, :, :].cpu().numpy())
			#ax3.imshow(xhat0[3, 0, :, :].cpu().numpy())
			#plt.show() 

			mse = torch.mean((x - xhat0)**2)
			print(t[0].item(), mse.item())
			denoising_mse.append(mse.item())

	plt.figure()
	plt.plot(denoising_mse)
	plt.show()

if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)