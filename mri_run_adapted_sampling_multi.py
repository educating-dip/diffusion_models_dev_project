import random
import json
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import h5py
import os
from torchvision.transforms import Resize
from pathlib import Path

from src import (get_standard_sde, PSNR, SSIM, get_standard_dataset, get_data_from_ground_truth, get_standard_ray_trafo,
				 get_standard_score, get_standard_adapted_sampler, get_standard_configs, get_standard_path, MulticoilMRI)


parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--model', default='openai_unet',
					help='select unet arch.', choices=['dds_unet', 'openai_unet'])
parser.add_argument('--method',  default='naive',
					choices=['naive', 'dps', 'dds'])
parser.add_argument('--sde', default='vesde',
					choices=['vpsde', 'vesde', 'ddpm'])
parser.add_argument('--load_path', help='path to ddpm model.')
parser.add_argument('--base_path', help='path to ddpm model configs.')

parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--num_steps', default=50)
parser.add_argument('--penalty', default=1,
					help='reg. penalty used for ``naive'' and ``dps'' only.')
parser.add_argument('--tv_penalty', default=1e-6,
					help='reg. used for ``adapatation''.')
parser.add_argument('--eta', default=0.85,
					help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--adaptation', default='lora',
					choices=['decoder', 'full', 'vdkl', 'lora'])
parser.add_argument('--num_optim_step', default=10,
					help='num. of optimization steps taken per sampl. step')
parser.add_argument('--adapt_freq', default=1,
					help='freq. of adaptation step in sampl.')
parser.add_argument('--lora_include_blocks', default=['input_blocks', 'middle_block',
													  'output_blocks', 'out'], nargs='+', help='lora kwargs impl. of arch. blocks included')
parser.add_argument('--lr', default=1e-3, help='learning rate for adaptation')
parser.add_argument('--lora_rank', default=4, help='lora kwargs impl. of rank')
parser.add_argument('--add_cg', action='store_true',
					help='do DDS steps after adaptation.')
parser.add_argument('--cg_iter', default=5,
					help='Number of CG steps for DDS update.')
parser.add_argument('--gamma', default=0.01, help='reg. used for ``dds''.')
parser.add_argument('--dc_type', default="cg",
					help='Type of additional DC when computing E[x0|xt, y]')
parser.add_argument('--anatomy', default="knee",
					choices=["knee", "brain"])
parser.add_argument('--mask_type', default="uniform1d",
					choices=["uniform1d", "gaussian1d", "poisson"])
parser.add_argument('--acc_factor', type=int, default=4)


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
	mux_in = size ** 2
	if type.endswith('2d'):
		Nsamp = mux_in // acc_factor
	elif type.endswith('1d'):
		Nsamp = size // acc_factor
	if type == 'gaussian2d':
		mask = torch.zeros_like(img)
		cov_factor = size * (1.5 / 128)
		mean = [size // 2, size // 2]
		cov = [[size * cov_factor, 0], [0, size * cov_factor]]
		if fix:
			samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
			int_samples = samples.astype(int)
			int_samples = np.clip(int_samples, 0, size - 1)
			mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
		else:
			for i in range(batch_size):
				# sample different masks for batch
				samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
				int_samples = samples.astype(int)
				int_samples = np.clip(int_samples, 0, size - 1)
				mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
	elif type == 'uniformrandom2d':
		mask = torch.zeros_like(img)
		if fix:
			mask_vec = torch.zeros([1, size * size])
			samples = np.random.choice(size * size, int(Nsamp))
			mask_vec[:, samples] = 1
			mask_b = mask_vec.view(size, size)
			mask[:, ...] = mask_b
		else:
			for i in range(batch_size):
				# sample different masks for batch
				mask_vec = torch.zeros([1, size * size])
				samples = np.random.choice(size * size, int(Nsamp))
				mask_vec[:, samples] = 1
				mask_b = mask_vec.view(size, size)
				mask[i, ...] = mask_b
	elif type == 'gaussian1d':
		mask = torch.zeros_like(img)
		mean = size // 2
		std = size * (15.0 / 128)
		Nsamp_center = int(size * center_fraction)
		if fix:
			samples = np.random.normal(
				loc=mean, scale=std, size=int(Nsamp * 1.2))
			int_samples = samples.astype(int)
			int_samples = np.clip(int_samples, 0, size - 1)
			mask[..., int_samples] = 1
			c_from = size // 2 - Nsamp_center // 2
			mask[..., c_from:c_from + Nsamp_center] = 1
		else:
			for i in range(batch_size):
				samples = np.random.normal(
					loc=mean, scale=std, size=int(Nsamp*1.2))
				int_samples = samples.astype(int)
				int_samples = np.clip(int_samples, 0, size - 1)
				mask[i, :, :, int_samples] = 1
				c_from = size // 2 - Nsamp_center // 2
				mask[i, :, :, c_from:c_from + Nsamp_center] = 1
	elif type == 'uniform1d':
		mask = torch.zeros_like(img)
		if fix:
			Nsamp_center = int(size * center_fraction)
			samples = np.random.choice(size, int(Nsamp - Nsamp_center))
			mask[..., samples] = 1
			# ACS region
			c_from = size // 2 - Nsamp_center // 2
			mask[..., c_from:c_from + Nsamp_center] = 1
		else:
			for i in range(batch_size):
				Nsamp_center = int(size * center_fraction)
				samples = np.random.choice(size, int(Nsamp - Nsamp_center))
				mask[i, :, :, samples] = 1
				# ACS region
				c_from = size // 2 - Nsamp_center // 2
				mask[i, :, :, c_from:c_from+Nsamp_center] = 1
	else:
		NotImplementedError(f'Mask type {type} is currently not supported.')

	return mask


def create_mask(shape, center_fraction=0.08, acceleration=4):

	num_cols = shape[-2]

	# create the mask
	num_low_freqs = int(round(num_cols * center_fraction))
	prob = (num_cols / acceleration - num_low_freqs) / (
		num_cols - num_low_freqs
	)
	mask = np.random.uniform(size=num_cols) < prob
	pad = (num_cols - num_low_freqs + 1) // 2
	mask[pad: pad + num_low_freqs] = True

	mask = np.repeat(mask[:, None].T, shape[-1], axis=0)

	return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)


def real_to_nchw_comp(x):
	"""
	[1, 2, 320, 320] real --> [1, 1, 320, 320] comp
	"""
	if len(x.shape) == 4:
		x = x[:, 0:1, :, :] + x[:, 1:2, :, :] * 1j
	elif len(x.shape) == 3:
		x = x[0:1, :, :] + x[1:2, :, :] * 1j
	return x

def seed_everything(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)  # type: ignore
	torch.backends.cudnn.deterministic = True  # type: ignore
	torch.backends.cudnn.benchmark = True  # type: ignore


def coordinator(args):
	
	seed_everything()

	with open(args.base_path, 'r') as stream:
		config = yaml.load(stream, Loader=yaml.UnsafeLoader)
		config = OmegaConf.create(config)

	model_type = "dds_unet"
	sde = get_standard_sde(config=config)
	score = get_standard_score(config=config, sde=sde,
							   use_ema=False, model_type="dds_unet", load_model=False)
	score.load_state_dict(torch.load(args.load_path))
	print(f'Model ckpt loaded from {args.load_path}')
	score.convert_to_fp32()
	score.dtype = torch.float32

	score = score.to(config.device)
	score.to("cuda")
	score.eval()
	
	size = 256

	if args.anatomy == "knee":
		# vol_list = ["file1000007", "file1000017", "file1000026", "file1000033", "file1000041", \
        #       		"file1000052", "file1000071", "file1000073", "file1000107", "file1000108"]
		vol_list = ["file1000033"]
		root = Path(f"/media/harry/tomo/fastmri/knee_mvue_{size}_val")
		# data_path = f"/media/harry/tomo/fastmri/knee_mvue_{size}_val/file1000033/slice"
		# data_path_mps = f"/media/harry/tomo/fastmri/knee_mvue_{size}_val/file1000033/mps"
		# x = np.load(os.path.join(data_path, "015.npy"))
		# mps = np.load(os.path.join(data_path_mps, "015.npy"))
	elif args.anatomy == "brain":
		data_path = f"/media/harry/tomo/fastmri/brain_mvue_{size}_val/file_brain_AXT2_200_2000019/slice"
		data_path_mps = f"/media/harry/tomo/fastmri/brain_mvue_{size}_val/file_brain_AXT2_200_2000019/mps"
		# x = np.load(os.path.join(data_path, "005.npy"))
		# mps = np.load(os.path.join(data_path_mps, "005.npy"))
  
	add_cg = True if args.add_cg else False
	save_root = Path(
		f"./results_adapt/{args.anatomy}/{args.mask_type}_acc{args.acc_factor}/add_cg_{add_cg}/lr{args.lr}_Nstep{args.num_optim_step}/tv{args.tv_penalty}")
	save_root.mkdir(exist_ok=True, parents=True)
	"""
	Iterate over dataset
	"""
 
	cnt = 0
	psnr_avg = 0
	ssim_avg = 0
	for vol in vol_list:
		for t in ["input", "recon", "label", "mask"]:
			(save_root / t / f"{vol}").mkdir(exist_ok=True, parents=True)
	
		print(vol)
		data_path = root / f"{vol}" / "slice"
		data_path_mps = root / f"{vol}" / "mps"

		list_fname = sorted(list(data_path.glob("*.npy")))
		for f in list_fname:
			fname = str(f).split('/')[-1][:-4]
			x = np.load(os.path.join(data_path, f"{fname}.npy"))
			mps = np.load(os.path.join(data_path_mps, f"{fname}.npy"))

			mask = get_mask(torch.zeros([1, 1, size, size]), size,
							1, type=args.mask_type,
							acc_factor=args.acc_factor, center_fraction=0.08)
			mask = mask.to(config.device)
			Ncoil, _, _ = mps.shape

			mps = torch.from_numpy(mps)
			mps = mps.view(1, Ncoil, size, size).to(config.device)

			ground_truth = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
			ground_truth = ground_truth.to(config.device)

			ray_trafo = MulticoilMRI(mask=mask, sens=mps)
   
			observation = ray_trafo.trafo(ground_truth)
			filtbackproj = ray_trafo.fbp(observation)


			logg_kwargs = {'log_dir': ".", 'num_img_in_log': 5,
						'sample_num': 0, 'ground_truth': ground_truth, 'filtbackproj': filtbackproj}
			sampler = get_standard_adapted_sampler(
				args=args,
				config=config,
				score=score,
				sde=sde,
				device=config.device,
				observation=observation,
				ray_trafo=ray_trafo,
				complex_y=True
			)

			recon = sampler.sample(logg_kwargs=logg_kwargs, logging=False)
			recon = real_to_nchw_comp(recon)
			recon = np.abs(recon.detach().cpu().numpy())
			meas_img = np.abs(filtbackproj.detach().cpu().numpy())

			psnr = PSNR(recon[0, 0], np.abs(ground_truth[0, 0].cpu().numpy()))
			ssim = SSIM(recon[0, 0], np.abs(ground_truth[0, 0].cpu().numpy()), data_range=np.abs(ground_truth[0, 0].cpu().numpy()).max())
			
			psnr_avg += psnr
			ssim_avg += ssim

			
			import matplotlib.pyplot as plt
			plt.imsave(str(save_root / "input" / f"{vol}" / f"{fname}.png"), meas_img[0, 0], cmap="gray")
			plt.imsave(str(save_root / "mask" / f"{vol}" / f'{fname}.png'), mask[0,0,:,:].detach().cpu(), cmap='gray')
			plt.imsave(str(save_root / "recon" / f"{vol}" / f"{fname}.png"), recon[0, 0], cmap="gray")
			plt.imsave(str(save_root / "label" / f"{vol}" / f"{fname}.png"), np.abs(ground_truth[0, 0].cpu().numpy()), cmap="gray")
			cnt += 1

	summary = {}
	psnr_avg /= cnt
	ssim_avg /= cnt
	summary["results"] = {"PSNR": psnr_avg, "SSIM": ssim_avg}
	with open(str(save_root / f"summary.json"), 'w') as f:
		json.dump(summary, f)


if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)
