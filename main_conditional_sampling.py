import os
import argparse
from pathlib import Path

import torch 
import functools
import numpy as np 
import matplotlib.pyplot as plt 
import yaml 
from itertools import islice
from math import ceil

from itertools import islice
from src import (marginal_prob_std, diffusion_coeff, OpenAiUNetModel,
  	pc_sampler, EllipseDatasetFromDival, simple_trafo, get_walnut_2d_ray_trafo, get_walnut_data_on_device,
   	simulate, PSNR, SSIM, naive_sampling, get_disk_dist_ellipses_dataset, limited_angle_trafo, ExponentialMovingAverage, optimize_sampling)

parser = argparse.ArgumentParser(description='Conditional Sampling.')
parser.add_argument("--dataset", default='walnut', help="which dataset to use")
parser.add_argument("--penalty", default=1, help="penalty parameter")
parser.add_argument("--smpl_start_prc", default=0)
parser.add_argument("--smpl_mthd", default="pc")

parser.add_argument("--limited_angle", action='store_true')
parser.add_argument("--angular_range", default=45)

parser.add_argument("--sparse_view", action='store_true')
parser.add_argument("--angles", default=30)

parser.add_argument("--ema", action='store_true')

parser.add_argument("--num_steps", default=1000)

parser.add_argument("--walnut_trafo", action='store_true')

def coordinator(args):

	if args.dataset == "ellipses":
		from configs.disk_ellipses_configs import get_config

	elif args.dataset == "walnut":
		from configs.walnut_configs import get_config
	else:
		raise NotImplementedError

	config = get_config()

	# set up save path for results 
	if args.limited_angle:
		save_root = Path(f'./results/limited_angle/{args.dataset}/sampler={args.smpl_mthd}/sampling_steps={ceil(float(args.smpl_start_prc) * int(args.num_steps))}_to_{int(args.num_steps)}/penalty={args.penalty}/angular_range={args.angular_range}')
	elif args.sparse_view:
		save_root = Path(f'./results/sparse_view/{args.dataset}/sampler={args.smpl_mthd}/sampling_steps={ceil(float(args.smpl_start_prc) * int(args.num_steps))}_to_{int(args.num_steps)}/penalty={args.penalty}/angles={args.angles}')
	else:
		save_root = Path(f'./results/{args.dataset}/sampler={args.smpl_mthd}/sampling_steps={ceil(float(args.smpl_start_prc) * int(args.num_steps))}_to_{int(args.num_steps)}/penalty={args.penalty}')
	save_root.mkdir(parents=True, exist_ok=True)

	print("Start with penalty: ", float(args.penalty))

	#if config.seed is not None:
	#	torch.manual_seed(config.seed)  # for reproducible noise in simulate
	
	marginal_prob_std_fn = functools.partial(
      marginal_prob_std,
      sigma_min=config.sde.sigma_min,
	  sigma_max=config.sde.sigma_max
    )
	diffusion_coeff_fn = functools.partial(
      diffusion_coeff, 
      sigma_min=config.sde.sigma_min,
	  sigma_max=config.sde.sigma_max
    )
	
	# TODO: getter model func and saving/loading funcs
	if config.model.model_name == 'OpenAiUNetModel':

		score_model = OpenAiUNetModel(
                    image_size=config.data.im_size,
                    in_channels=config.model.in_channels,
                    model_channels=config.model.model_channels,
                    out_channels=config.model.out_channels,
                    num_res_blocks=config.model.num_res_blocks,
                    attention_resolutions=config.model.attention_resolutions,
                    marginal_prob_std=marginal_prob_std_fn,
                    channel_mult=config.model.channel_mult,
                    conv_resample=config.model.conv_resample,
                    dims=config.model.dims,
                    num_heads=config.model.num_heads,
                    num_head_channels=config.model.num_head_channels,
                    num_heads_upsample=config.model.num_heads_upsample,
                    use_scale_shift_norm=config.model.use_scale_shift_norm,
                    resblock_updown=config.model.resblock_updown,
                    use_new_attention_order=config.model.use_new_attention_order
                    )
	else:
		
		raise NotImplementedError


	if (config.sampling.load_model_from_path is not None) and (config.sampling.model_name is not None): 
		print("Load Score Model...")
		if args.ema:
			ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)
			ema.load_state_dict(torch.load(os.path.join(config.sampling.load_model_from_path,"ema_model.pt")))
			ema.copy_to(score_model.parameters())

		else:
			score_model.load_state_dict(torch.load(
				os.path.join(config.sampling.load_model_from_path, config.sampling.model_name))
			)



	score_model = score_model.to(config.device)
	score_model.eval()

	if config.forward_op.trafo_name == 'simple_trafo':
		ray_trafo = simple_trafo(
							im_size=config.data.im_size, 
							num_angles=config.forward_op.num_angles
		)
	elif config.forward_op.trafo_name == 'walnut_trafo':

			ray_trafo = {}
			ray_trafo_obj = get_walnut_2d_ray_trafo(
									data_path=config.data.data_path,
									matrix_path=config.data.data_path,
									walnut_id=config.data.walnut_id,
									orbit_id=config.forward_op.orbit_id,
									angular_sub_sampling=config.forward_op.angular_sub_sampling,
									proj_col_sub_sampling=config.forward_op.proj_col_sub_sampling
			)
			ray_trafo_obj = ray_trafo_obj.to(device=config.device)
			ray_trafo['ray_trafo_module'] = ray_trafo_obj
	else: 
		raise NotImplementedError

	if config.data.name == 'EllipseDatasetFromDival':

		ellipse_dataset = EllipseDatasetFromDival(impl="astra_cuda")
		dataset = ellipse_dataset.get_valloader(
					batch_size=1,
					num_data_loader_workers=0
		)

	elif config.data.name == 'DiskDistributedEllipsesDataset':
    
		dataset = get_disk_dist_ellipses_dataset(
				fold='test', 
				im_size=config.data.im_size, 
				length=config.data.val_length,
				diameter=config.data.diameter,
				max_n_ellipse=config.data.num_n_ellipse, 
				device=config.device
			)
			
	elif config.data.name == 'Walnut':

		dataset = get_walnut_data_on_device(config, ray_trafo_obj)
	else:
		raise NotImplementedError


	if args.limited_angle:
		ray_trafo = limited_angle_trafo(im_size=config.data.im_size, 
							angular_range=int(args.angular_range))
	elif args.sparse_view:
		ray_trafo = simple_trafo(
							im_size=config.data.im_size, 
							num_angles=int(args.angles)
		)
	elif args.walnut_trafo:
		from configs.walnut_configs import get_config
		walnut_config = get_config()
		ray_trafo = {}
		ray_trafo_obj = get_walnut_2d_ray_trafo(
									data_path=walnut_config.data.data_path,
									matrix_path=walnut_config.data.data_path,
									walnut_id=walnut_config.data.walnut_id,
									orbit_id=walnut_config.forward_op.orbit_id,
									angular_sub_sampling=walnut_config.forward_op.angular_sub_sampling,
									proj_col_sub_sampling=walnut_config.forward_op.proj_col_sub_sampling
			)
		ray_trafo_obj = ray_trafo_obj.to(device=config.device)
		ray_trafo['ray_trafo_module'] = ray_trafo_obj
		ray_trafo['fbp_module'] = ray_trafo_obj.fbp

	psnr_list = []
	ssim_list = []

	psnr_fbp_list = []
	ssim_fbp_list = [] 
	for i, data_sample in enumerate(islice(dataset, config.data.validation.num_images)):
		if len(data_sample) == 2 and config.data.name == 'EllipseDatasetFromDival':
			
			_, ground_truth = data_sample
			ground_truth = ground_truth.to(device=config.device)
			observation, noise_level = simulate(
					ground_truth,
					ray_trafo['ray_trafo_module'],
					white_noise_rel_stddev=config.data.stddev
			)
			filterbackproj = ray_trafo["fbp_module"](observation)

		elif len(data_sample) == 3:

			observation, ground_truth, filterbackproj = data_sample
			ground_truth = ground_truth.to(device=config.device)
			observation = observation.to(device=config.device)
			filterbackproj = filterbackproj.to(device=config.device)

			if args.limited_angle or args.sparse_view:
				observation = simulate(
				ground_truth,
				ray_trafo['ray_trafo_module'],
				white_noise_rel_stddev=config.data.stddev,
				return_noise_level=False
				)
				filterbackproj = ray_trafo["fbp_module"](observation)

		elif config.data.name == "DiskDistributedEllipsesDataset":
			ground_truth = data_sample
			ground_truth = ground_truth.unsqueeze(0)
			ground_truth = ground_truth.to(device=config.device)
			observation, noise_level = simulate(
					ground_truth,
					ray_trafo['ray_trafo_module'],
					white_noise_rel_stddev=config.data.stddev,
					return_noise_level=True
			)
			filterbackproj = ray_trafo["fbp_module"](observation)
	

		if args.smpl_mthd == 'pc':

			x_mean = pc_sampler(
							score_model=score_model, 
							marginal_prob_std=marginal_prob_std_fn, 
							ray_trafo=ray_trafo, 
							diffusion_coeff=diffusion_coeff_fn, 
							observation=observation, 
							penalty=float(args.penalty), 
							img_shape=ground_truth.shape[1:],
							batch_size=config.sampling.batch_size, 
							num_steps=int(args.num_steps), 
							snr=config.sampling.snr, 
							device=config.device, 
							eps=config.sampling.eps,
							x_fbp = filterbackproj,
							start_time_step=ceil(float(args.smpl_start_prc) * int(args.num_steps)),
							log_dir = save_root,
							ground_truth = ground_truth			
				)

		elif  args.smpl_mthd == 'naive_smpl':

			x_mean = naive_sampling(
							score_model=score_model, 
							marginal_prob_std=marginal_prob_std_fn, 
							ray_trafo=ray_trafo, 
							diffusion_coeff=diffusion_coeff_fn, 
							observation=observation, 
							penalty=float(args.penalty), 
							img_shape=ground_truth.shape[1:],
							batch_size=config.sampling.batch_size, 
							num_steps=int(args.num_steps), 
							snr=config.sampling.snr, 
							device=config.device, 
							eps=config.sampling.eps,
							x_fbp = filterbackproj,
							start_time_step=ceil(float(args.smpl_start_prc) * int(args.num_steps)),
							log_dir = save_root,
							ground_truth = ground_truth
				)

		else:
				
			raise NotImplementedError
		
		x_mean = torch.clamp(x_mean, 0, 1)
		torch.save(
			{'recon': x_mean.cpu().squeeze(),
			'ground_truth': ground_truth.cpu().squeeze()}, 
			str(save_root / f'recon_{i}_info.pt')
			)
			
		print(f'reconstruction of sample {i}')
		psnr = PSNR(x_mean[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
		ssim = SSIM(x_mean[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
		psnr_list.append(psnr)
		ssim_list.append(ssim)
		print('PSNR:', psnr)
		print('SSIM:', ssim)

		psnr_fbp_list.append(PSNR(filterbackproj[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
		ssim_fbp_list.append(SSIM(filterbackproj[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

		fig, (ax1, ax2,ax3, ax4) = plt.subplots(1,4)
		ax1.imshow(x_mean[0,0,:,:].cpu(), cmap="gray")
		ax1.set_title("Score-based Reco.")
		ax2.imshow(ground_truth[0,0,:,:].cpu(), cmap="gray")
		ax2.set_title("Ground truth")
		ax3.imshow(filterbackproj[0,0,:,:].cpu(), cmap="gray")
		ax3.set_title("FBP Reco.")
		
		ax4.imshow(ground_truth[0,0,:,:].cpu() - x_mean[0,0,:,:].cpu(), cmap="gray")
		ax4.set_title("Residual")
		fig.savefig(f'penaly={args.penalty}.pdf')
		fig.savefig(str(save_root / f'recon_{i}.png'))
		plt.show()
		plt.close("all") 

	print("MEAN PSNR: ", np.mean(psnr_list))
	print("MEAN SSIM: ", np.mean(ssim_list))

	report_dict = {'stettings': {'num_images': config.data.validation.num_images,
								'penalty': float(args.penalty),
								'num_steps': config.sampling.num_steps,
								'start_step': config.sampling.start_time_step},
                    'results': {}}
    
	report_dict['results']['PSNR'] = float(np.mean(psnr_list))
	report_dict['results']['SSIM'] = float(np.mean(ssim_list))

	report_dict['results']['PSNR_FBP'] = float(np.mean(psnr_fbp_list))
	report_dict['results']['SSIM_FBP'] = float(np.mean(ssim_fbp_list))
	print(report_dict)

	with open(save_root / 'report.yaml', 'w') as file:
		documents = yaml.dump(report_dict, file)

if __name__ == '__main__':
	
	args = parser.parse_args()

	coordinator(args)