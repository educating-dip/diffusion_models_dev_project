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


from src import (get_standard_sde, PSNR, SSIM, get_standard_dataset, get_data_from_ground_truth, get_standard_ray_trafo,  
	get_standard_score, get_standard_adapted_sampler, get_standard_configs, get_standard_path, MulticoilMRI) 

parser = argparse.ArgumentParser(description='adapted sampling')
parser.add_argument('--model', default='openai_unet', help='select unet arch.', choices=['dds_unet', 'openai_unet'])
parser.add_argument('--method',  default='naive', choices=['naive', 'dps', 'dds'])
parser.add_argument('--sde', default='vesde', choices=['vpsde', 'vesde', 'ddpm'])
parser.add_argument('--load_path', help='path to ddpm model.')
parser.add_argument('--base_path', help='path to ddpm model configs.')

parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--num_steps', default=50)
parser.add_argument('--penalty', default=1, help='reg. penalty used for ``naive'' and ``dps'' only.')
parser.add_argument('--tv_penalty', default=1e-6, help='reg. used for ``adapatation''.')
parser.add_argument('--eta', default=0.85, help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--adaptation', default='lora', choices=['decoder', 'full', 'vdkl', 'lora'])
parser.add_argument('--num_optim_step', default=10, help='num. of optimization steps taken per sampl. step')
parser.add_argument('--adapt_freq', default=1, help='freq. of adaptation step in sampl.')
parser.add_argument('--lora_include_blocks', default=['input_blocks','middle_block','output_blocks','out'], nargs='+', help='lora kwargs impl. of arch. blocks included')
parser.add_argument('--lr', default=1e-3, help='learning rate for adaptation')
parser.add_argument('--lora_rank', default=4, help='lora kwargs impl. of rank')
parser.add_argument('--add_cg', action='store_true', help='do DDS steps after adaptation.')
parser.add_argument('--cg_iter', default=5, help='Number of CG steps for DDS update.')
parser.add_argument('--gamma', default=0.01, help='reg. used for ``dds''.')
parser.add_argument('--dc_type', default="cg", choices=["cg", "gd"], help="use cg/gd in adaptation")


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


def real_to_nchw_comp(x):
    """
    [1, 2, 320, 320] real --> [1, 1, 320, 320] comp
    """
    if len(x.shape) == 4:
        x = x[:, 0:1, :, :] + x[:, 1:2, :, :] * 1j
    elif len(x.shape) == 3:
        x = x[0:1, :, :] + x[1:2, :, :] * 1j
    return x

def coordinator(args):
	
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

	data_path = "/localdata/AlexanderDenker/fast_mri/knee_mvue_320_val/file1000033/slice"
	data_path_mps = "/localdata/AlexanderDenker/fast_mri/knee_mvue_320_val/file1000033/mps"

	x = np.load(os.path.join(data_path, "015.npy"))
	mps = np.load(os.path.join(data_path_mps, "015.npy"))

	#mask = create_mask(shape=(256,256), center_fraction=0.08, acceleration=8) 
	mask = get_mask(torch.zeros([1, 1, 256, 256]), 256, 
								1, type="uniform1d",
								acc_factor=8, center_fraction=0.04)
	mask = mask.to(config.device)
	Ncoil, _, _ = mps.shape
	
	resizer = Resize((256,256))
	mps = resizer(torch.from_numpy(mps))

	print(mps.shape)
	mps = mps.view(1, Ncoil, 256, 256).to(config.device)


	#ray_trafo = SingleCoilMRI(mask)

	ray_trafo = MulticoilMRI(mask=mask, sens=mps)


	print("SHAPE OF MPS: ", mps.shape)

	print(x.shape)
	x_torch = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
		


	ground_truth = resizer(x_torch)
	ground_truth = ground_truth.to(config.device)
	print("RANGE OF GT: ", torch.abs(ground_truth).min(), torch.abs(ground_truth).max())

	observation = ray_trafo.trafo(ground_truth)

	filtbackproj = ray_trafo.fbp(observation)

	print(filtbackproj.shape, ground_truth.shape, observation.shape)


	logg_kwargs = {'log_dir': ".", 'num_img_in_log': 5,
		'sample_num':0, 'ground_truth': ground_truth, 'filtbackproj': filtbackproj}
	sampler = get_standard_adapted_sampler(
				args=args,
				config=config,
				score=score,
				sde=sde,
				device=config.device,
				observation = observation,
				ray_trafo = ray_trafo,
				complex_y = True
				)
	
	recon = sampler.sample(logg_kwargs=logg_kwargs, logging=False)
	print("FINISHED SAMPLING")
	print(recon.shape)

	recon = real_to_nchw_comp(recon)
	recon = np.abs(recon.detach().cpu().numpy())

	#recon = torch.clamp(recon, 0)
	#torch.save(		{'recon': recon.cpu().squeeze(), 'ground_truth': ground_truth.cpu().squeeze()}, 
	#	str(save_root / f'recon_{i}_info.pt')	)
	#im = Image.fromarray(recon.cpu().squeeze().numpy()*255.).convert("L")
	#im.save(str(save_root / f'recon_{i}.png'))

	print(f'reconstruction of sample {0}'	)
	psnr = PSNR(recon[0, 0], torch.abs(ground_truth[0, 0]).cpu().numpy())
	ssim = SSIM(recon[0, 0], torch.abs(ground_truth[0, 0]).cpu().numpy())	
	print('PSNR:', psnr)
	print('SSIM:', ssim)
	#_psnr.append(psnr)
	#_ssim.append(ssim)

	_, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
	ax1.imshow(torch.abs(ground_truth[0,0,:,:].detach().cpu()))
	ax1.axis('off')
	ax1.set_title('Ground truth')
	ax2.imshow(recon[0,0,:,:])
	ax2.axis('off')
	ax2.set_title(args.method)
	ax3.imshow(torch.abs(filtbackproj[0,0,:,:].detach().cpu()))
	ax3.axis('off')
	ax3.set_title('FBP')
	ax4.imshow(mask[0,0,:,:].detach().cpu())
	ax4.axis('off')
	ax4.set_title('mask')

	plt.show()	


if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)