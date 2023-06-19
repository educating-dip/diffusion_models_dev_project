import argparse
import yaml 
import torch 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import os
from src import (get_standard_sde, get_standard_score, LoDoPabChallenge) 

parser = argparse.ArgumentParser(description='conditional sampling')

def coordinator(args):
	load_path = os.path.join("/localdata/AlexanderDenker/score_based_baseline/challenge/LoDoPabCT/ddpm/version_02")
	print('load model from: ', load_path)
	with open(os.path.join(load_path, 'report.yaml'), 'r') as stream:
		config = yaml.load(stream, Loader=yaml.UnsafeLoader)
		config.sampling.load_model_from_path = load_path

	if config.seed is not None:
		torch.manual_seed(config.seed) # for reproducible noise in simulate


	sde = get_standard_sde(config=config)	
	score = get_standard_score(config=config, sde=sde, use_ema=True)
	score = score.to(config.device)
	score.eval()
	dataset = LoDoPabChallenge()
	test_loader = dataset.get_testloader(batch_size=12)
	_, x = next(iter(test_loader))
	x = x.to('cuda')
	denoising_mse = [] 
	with torch.no_grad():
		for i in range(0, sde.num_steps):
			t = torch.ones(x.shape[0], device=x.device) * i 
			z = torch.randn_like(x)
			mean, std = sde.marginal_prob(x, t)
			# Diffuse the data for a given number of diffusion steps. In other words, sample from q(x_t | x_0)
			perturbed_x = mean + z * std[:, None, None, None]
			zhat = score(perturbed_x, t)

			div = sde.marginal_prob_mean(t)[:, None, None, None].pow(-1)
			std_t = sde.marginal_prob_std(t)[:, None, None, None]
			xhat0 = (perturbed_x - zhat*std_t)*div
			mse = torch.mean((x - xhat0)**2)
			denoising_mse.append(mse.item())

if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)