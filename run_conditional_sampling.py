import os
import argparse
import yaml 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import islice
from itertools import islice
from src import (get_standard_sde, PSNR, SSIM, get_standard_dataset, get_data_from_ground_truth, get_standard_ray_trafo,  
	get_standard_score, get_standard_sampler, get_standard_configs, get_standard_path) 

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--dataset', default='walnut', help='test-dataset', choices=['walnut', 'lodopab', 'ellipses', 'mayo'])
parser.add_argument('--model_learned_on', default='lodopab', help='model-checkpoint to load', choices=['lodopab', 'ellipses'])
parser.add_argument('--method',  default='naive', choices=['naive', 'dps', 'dds'])
parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--num_steps', default=1000)
parser.add_argument('--penalty', default=1, help='reg. penalty used for ``naive'' and ``dps'' only.')
parser.add_argument('--gamma', default=0.01, help='reg. used for ``dds''.')
parser.add_argument('--eta', default=0.15, help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--pct_chain_elapsed', default=0,  help='``pct_chain_elapsed'' actives init of chain')
parser.add_argument('--sde', default='vesde', choices=['vpsde', 'vesde'])

def coordinator(args):
	if args.model_learned_on == "ellipses":
		load_path = "/localdata/AlexanderDenker/score_based_baseline/DiskEllipses/checkpoints/2023_05_11_08:05"

		with open(os.path.join(load_path, "report.yaml"), "r") as stream:
			config = yaml.load(stream, Loader=yaml.UnsafeLoader)
			config.sampling.load_model_from_path = load_path

			print(config.sde.type)

	_, dataconfig = get_standard_configs(args)
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

	print("num images: ", config.data.validation.num_images)
	psnr_list = [] 
	ssim_list = []
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

		logg_kwargs = {'log_dir': save_root, 'num_img_in_log': 5,
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
		
		recon = sampler.sample(logg_kwargs=logg_kwargs)
		torch.save(		{'recon': recon.cpu().squeeze(), 'ground_truth': ground_truth.cpu().squeeze()}, 
			str(save_root / f'recon_{i}_info.pt')	)
			
		print(	f'reconstruction of sample {i}'	)
		psnr = PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
		ssim = SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())	
		psnr_list.append(psnr)
		ssim_list.append(ssim)
		print(	'PSNR:', psnr)
		print(	'SSIM:', ssim)
	
		#fig, (ax1, ax2) = plt.subplots(1,2)
		#ax1.imshow(ground_truth[0,0,:,:].detach().cpu())
		#ax1.axis("off")
		#ax1.set_title("Ground truth")
		#ax2.imshow(torch.clamp(recon[0,0,:,:], 0, 1).detach().cpu())
		#ax2.axis("off")
		#ax2.set_title("Naive with penalty = " + str(args.penalty))
		#plt.show() 


	report = {}
	report.update(vars(args))
	report.update(dict(config.items()))
	report.update(dict(dataconfig.items()))

	with open(save_root / 'report.yaml', 'w') as file:
		yaml.dump(report, file)

if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)