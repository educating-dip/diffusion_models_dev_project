import os
import argparse
import yaml 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import islice
from itertools import islice
from src import (get_standard_sde, PSNR, SSIM, get_standard_dataset, get_data_from_ground_truth, get_standard_ray_trafo,  
	get_standard_score, get_standard_configs, get_standard_path, get_standard_adapted_sampler) 

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--dataset', default='walnut', help='test-dataset', choices=['walnut', 'lodopab', 'ellipses', 'mayo'])
parser.add_argument('--model_learned_on', default='lodopab', help='model-checkpoint to load', choices=['lodopab', 'ellipses'])
parser.add_argument('--method',  default='naive', choices=['naive', 'dps', 'dds'])
parser.add_argument('--version', default=1, help="version of the model")

parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--num_steps', default=1000)
parser.add_argument('--penalty', default=1, help='reg. penalty used for ``naive'' and ``dps'' only.')
parser.add_argument('--gamma', default=0.01, help='reg. used for ``dds''.')
parser.add_argument('--eta', default=0.15, help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--sde', default='vesde', choices=['vpsde', 'vesde'])

def coordinator(args):
	config, dataconfig = get_standard_configs(args)

	save_root = get_standard_path(args)
	save_root.mkdir(parents=True, exist_ok=True)

	if config.seed is not None:
		torch.manual_seed(config.seed) # for reproducible noise in simulate

	sde = get_standard_sde(config=config)
	score = get_standard_score(config=config, sde=sde, use_ema=args.ema)
	
	score = score.to(config.device)
	score.eval()

	
	ray_trafo = get_standard_ray_trafo(config=dataconfig)
	ray_trafo = ray_trafo.to(device=config.device)
	dataset = get_standard_dataset(config=dataconfig, ray_trafo=ray_trafo)

	for i, data_sample in enumerate(islice(dataset, config.data.validation.num_images)):
		if len(data_sample) == 3:
			observation, ground_truth, filtbackproj = data_sample
			ground_truth = ground_truth.to(device=config.device)
			observation = observation.to(device=config.device)
			filtbackproj = filtbackproj.to(device=config.device)
		else:
			ground_truth, observation, filtbackproj = get_data_from_ground_truth(
				ground_truth=data_sample.to(device=config.device),
				ray_trafo=ray_trafo,
				white_noise_rel_stddev=dataconfig.data.stddev
				)

		logg_kwargs = {'log_dir': save_root, 'num_img_in_log': 10, 'sample_num': 1, 'ground_truth': ground_truth, 'filtbackproj': filtbackproj}

		sampler = get_standard_adapted_sampler(
				args=args,
				config=config,
				score=score,
				sde=sde,
				device=config.device,
				observation = observation,
				ray_trafo = ray_trafo
				)
		
		recon = sampler.sample(logg_kwargs=logg_kwargs)
		
		print(f'reconstruction of sample {i}'	)
		psnr = PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
		ssim = SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())	
		print('PSNR:', psnr)
		print('SSIM:', ssim)
		
		print(recon.shape)
		fig, (ax1, ax2) = plt.subplots(1,2)
		ax1.imshow(ground_truth[0,0,:,:].detach().cpu())
		ax1.axis("off")
		ax1.set_title("Ground truth")
		ax2.imshow(torch.clamp(recon[0,0,:,:], 0, 1).detach().cpu())
		ax2.axis("off")
		ax2.set_title("Adaptation Sampling")
		plt.show() 

if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)