import argparse
import yaml 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import islice
from PIL import Image

import os
import time
from pathlib import Path

from src import (get_standard_sde, PSNR, SSIM, LoDoPabTrafo, get_standard_score, get_standard_sampler, LoDoPabChallenge) 

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--method',  default='dds', choices=['naive', 'dps', 'dds'])
parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--num_steps', default=100)
parser.add_argument('--penalty', default=1, help='reg. penalty used for ``naive'' and ``dps'' only.')
parser.add_argument('--gamma', default=20., help='reg. used for ``dds''.')
parser.add_argument('--eta', default=0.4, help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--pct_chain_elapsed', default=0,  help='``pct_chain_elapsed'' actives init of chain')
parser.add_argument('--sde', default='ddpm', choices=['vpsde', 'vesde', 'ddpm'])
parser.add_argument('--cg_iter', default=4)

def coordinator(args):
	load_path = os.path.join("/localdata/AlexanderDenker/score_based_baseline/challenge/LoDoPabCT/ddpm/version_02")
	print('load model from: ', load_path)
	with open(os.path.join(load_path, 'report.yaml'), 'r') as stream:
		config = yaml.load(stream, Loader=yaml.UnsafeLoader)
		config.sampling.load_model_from_path = load_path

	if config.seed is not None:
		torch.manual_seed(config.seed) # for reproducible noise in simulate

	from configs.lodopab_challenge_configs import get_config
	dataconfig = get_config(args)

	sde = get_standard_sde(config=config)
	score = get_standard_score(config=config, sde=sde, use_ema=True)
	score = score.to(config.device)
	score.eval()
	
	dataset = LoDoPabChallenge()
	ray_trafo = LoDoPabTrafo().to(device=config.device)

	path = "/localdata/AlexanderDenker/score_results/lodopab"
	save_root = Path(os.path.join(path, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}'))
	save_root.mkdir(parents=True, exist_ok=True)

	psnr_list = [] 
	ssim_list = []
	print("number of images in dataset: ", len(dataset.lodopab_test))
	print("num images to test: ", dataconfig.data.validation.num_images)
	for i, data_sample in enumerate(islice(dataset.lodopab_test, dataconfig.data.validation.num_images)):
		observation, ground_truth = data_sample
		ground_truth = ground_truth.to(device=config.device)
		observation = observation.to(device=config.device)
		filtbackproj = ray_trafo.fbp(observation)
		logg_kwargs = {'log_dir': save_root, 'num_img_in_log': 10,
			'sample_num':i, 'ground_truth': ground_truth, 'filtbackproj': filtbackproj}
		sampler = get_standard_sampler(
			args=args,
			config=config,
			score=score,
			sde=sde,
			ray_trafo=ray_trafo,
			filtbackproj=filtbackproj,
			observation=observation,
			device=config.device
			)
		
		recon = sampler.sample(logg_kwargs=logg_kwargs,logging=False)
		recon = torch.clamp(recon, 0, 1)

		print(recon.shape, ground_truth.shape)
		print(f'reconstruction of sample {i}'	)
		psnr = PSNR(recon[0, 0, :, :].cpu().numpy(), ground_truth[0, :, :].cpu().numpy())
		ssim = SSIM(recon[0, 0, :, :].cpu().numpy(), ground_truth[0, :, :].cpu().numpy())	
		psnr_list.append(psnr)
		ssim_list.append(ssim)
		print('PSNR:', psnr)
		print('SSIM:', ssim)
		
		"""
		fig, (ax1, ax2, ax3) = plt.subplots(1,3)
		ax1.imshow(ground_truth[0,:,:].detach().cpu())
		ax1.axis("off")
		ax1.set_title("Ground truth")
		ax2.imshow(torch.clamp(recon[0,0,:,:], 0, 1).detach().cpu())
		ax2.axis("off")
		ax2.set_title(args.method)
		ax3.imshow(filtbackproj[0,:,:].detach().cpu())
		ax3.axis("off")
		ax3.set_title("FBP")
		plt.show() 
		"""

	report = {}
	report.update(dict(dataconfig.items()))
	report.update(dict(config.items()))
	report.update(vars(args))

	report["PSNR"] = float(np.mean(psnr_list))
	report["SSIM"] = float(np.mean(ssim_list))

	with open(save_root / 'report.yaml', 'w') as file:
		yaml.dump(report, file)

if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)