import os
import argparse
import yaml 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import islice
from itertools import islice
from src import (get_sde, PSNR, SSIM, get_standard_dataset, get_data_from_ground_truth, get_standard_ray_trafo,  
	get_standard_score, get_standard_sampler, get_standard_configs, get_standard_path) 

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--dataset', default='walnut', help='test-dataset', choices=['walnut', 'lodopab', 'ellipses'])
parser.add_argument('--model_learned_on', default='lodopab', help='model-checkpoint to load', choices=['lodopab', 'ellipses'])
parser.add_argument('--method',  default='naive', choices=['naive', 'dps', 'dds'])
parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--num_steps', default=1000)
parser.add_argument('--penalty', default=1, help='penalty parameter')
parser.add_argument('--pct_chain_elapsed', default=0,  help='``pct_chain_elapsed'' actives init of chain')

def coordinator(args):

	config, dataconfig = get_standard_configs(args)
	save_root = get_standard_path(args)
	save_root.mkdir(parents=True, exist_ok=True)

	if config.seed is not None:
		torch.manual_seed(config.seed) # for reproducible noise in simulate
	''' 
	This sets the Forward SDE as Variance Exploding SDE refer to Appendix C 
		in ``SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS'' 
		at https://arxiv.org/pdf/2011.13456.pdf. 
	'''
	sde = get_sde(config=config)
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

		logg_kwargs = {'log_dir': save_root, 'num_img_in_log': 10, 'log_freq':10, 
			'sample_num':i, 'ground_truth': ground_truth, 'filtbackproj': filtbackproj}
		
		sampler = get_standard_sampler(
			args=args,
			config=config,
			score=score,
			sde=sde,
			ray_trafo=ray_trafo,
			filtbackproj=filtbackproj,
			observation=observation,
			device=config.device)
		
		recon = sampler.sample(logg_kwargs=logg_kwargs)
		torch.save(		{'recon': recon.cpu().squeeze(), 'ground_truth': ground_truth.cpu().squeeze()}, 
			str(save_root / f'recon_{i}_info.pt')	)
			
		print(	f'reconstruction of sample {i}'	)
		print(	'PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())	)
		print(	'SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())	)
	
	report = {}
	report.update(vars(args))
	report.update(dict(config.items()))
	report.update(dict(dataconfig.items()))

	with open(save_root / 'report.yaml', 'w') as file:
		yaml.dump(report, file)

if __name__ == '__main__':
	
	args = parser.parse_args()
	coordinator(args)