import yaml
import argparse
import torch 
import matplotlib.pyplot as plt
import numpy as  np
from PIL import Image

from itertools import islice
from src import (get_standard_sde, PSNR, SSIM, get_standard_dataset, get_data_from_ground_truth, get_standard_ray_trafo,  
	get_standard_score, get_standard_configs, get_standard_path, get_standard_adapted_sampler) 

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--dataset', default='walnut', help='test-dataset', choices=['walnut', 'lodopab', 'ellipses', 'mayo', 'aapm'])
parser.add_argument('--model', default='openai_unet', help='select unet arch.', choices=['dds_unet', 'openai_unet'])
parser.add_argument('--base_path', default='/localdata/AlexanderDenker/score_based_baseline', help='path to model configs')
parser.add_argument('--model_learned_on', default='lodopab', help='model-checkpoint to load', choices=['lodopab', 'ellipses', 'aapm'])
parser.add_argument('--method',  default='naive', choices=['naive', 'dps', 'dds'])
parser.add_argument('--version', default=1, help="version of the model")
parser.add_argument('--noise_level', default=0.01, help="rel. additive gaussian noise.")
parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--num_steps', default=50)
parser.add_argument('--penalty', default=1, help='reg. penalty used for ``naive'' and ``dps'' only.')
parser.add_argument('--tv_penalty', default=1e-6, help='reg. used for ``adapatation''.')
parser.add_argument('--eta', default=0.85, help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--sde', default='vesde', choices=['vpsde', 'vesde', 'ddpm'])
parser.add_argument('--adaptation', default='lora', choices=['decoder', 'full', 'vdkl', 'lora'])
parser.add_argument('--num_optim_step', default=10, help='num. of optimization steps taken per sampl. step')
parser.add_argument('--adapt_freq', default=1, help='freq. of adaptation step in sampl.')
parser.add_argument('--lora_include_blocks', default=['input_blocks','middle_block','output_blocks','out'], nargs='+', help='lora kwargs impl. of arch. blocks included')
parser.add_argument('--lr', default=1e-3, help='learning rate for adaptation')
parser.add_argument('--lora_rank', default=4, help='lora kwargs impl. of rank')
parser.add_argument('--add_cg', action='store_true', help='do DDS steps after adaptation.')
parser.add_argument('--cg_iter', default=5, help='Number of CG steps for DDS update.')
parser.add_argument('--gamma', default=0.01, help='reg. used for ``dds''.')
parser.add_argument('--load_path', help='path to ddpm model.')


def coordinator(args):
	config, dataconfig = get_standard_configs(args, base_path=args.base_path)
	dataconfig.data.stddev = float(args.noise_level)
	save_root = get_standard_path(args, run_type="adapt")
	save_root.mkdir(parents=True, exist_ok=True)
	
	if config.seed is not None:
		torch.manual_seed(config.seed) # for reproducible noise in simulate

	sde = get_standard_sde(config=config)
	score = get_standard_score(config=config, sde=sde, use_ema=args.ema, model_type=args.model)
	score = score.to(config.device).eval()
	ray_trafo = get_standard_ray_trafo(config=dataconfig)
	ray_trafo = ray_trafo.to(device=config.device)
	dataset = get_standard_dataset(config=dataconfig, ray_trafo=ray_trafo)

	dataconfig.data.validation.num_images = len(dataset)
	_psnr, _ssim = [], []
	for i, data_sample in enumerate(islice(dataset, dataconfig.data.validation.num_images)):
		if config.seed is not None:
			torch.manual_seed(config.seed + i)  # for reproducible noise in simulate
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
		recon = sampler.sample(logg_kwargs=logg_kwargs, logging=True)
		recon = torch.clamp(recon, 0)
		torch.save(		{'recon': recon.cpu().squeeze(), 'ground_truth': ground_truth.cpu().squeeze()}, 
		str(save_root / f'recon_{i}_info.pt')	)
		im = Image.fromarray(recon.cpu().squeeze().numpy()*255.).convert("L")
		im.save(str(save_root / f'recon_{i}.png'))

		
		score = get_standard_score(config=config, sde=sde, use_ema=args.ema, model_type=args.model)
		score = score.to(config.device)

		print(f'reconstruction of sample {i}')
		psnr = PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
		ssim = SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
		_psnr.append(psnr)
		_ssim.append(ssim)
		print('PSNR:', psnr)
		print('SSIM:', ssim)

		_, (ax1, ax2) = plt.subplots(1,2)
		ax1.imshow(ground_truth[0,0,:,:].detach().cpu())
		ax1.axis('off')
		ax1.set_title('Ground truth')
		ax2.imshow(torch.clamp(recon[0,0,:,:], 0, 1).detach().cpu())
		ax2.axis('off')
		ax2.set_title('Adaptation Sampling')
		# plt.savefig(f'diag_smpl_{i}.png') 
		
	report = {}
	report.update(dict(dataconfig.items()))
	report.update(vars(args))
	report["PSNR"] = float(np.mean(_psnr))
	report["SSIM"] = float(np.mean(_ssim))

<<<<<<< HEAD
	print("Mean PSNR: ", np.mean(_psnr))
	print("Mean SSIM: ", np.mean(_ssim))

=======
>>>>>>> f97bcce23e38471e17a48b9a199d3b2bd7a5520d
	with open(save_root / 'report.yaml', 'w') as file:
		yaml.dump(report, file)


if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)