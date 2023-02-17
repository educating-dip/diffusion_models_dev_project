import os
import torch 
import functools
import numpy as np 
import matplotlib.pyplot as plt 
from itertools import islice

from itertools import islice
from src import (marginal_prob_std, diffusion_coeff, OpenAiUNetModel,
  pc_sampler, EllipseDatasetFromDival, simple_trafo, get_walnut_2d_ray_trafo, get_walnut_data_on_device, simulate, PSNR, SSIM)
from configs.walnut_configs import get_config

def coordinator():
	config = get_config()

	config = get_config()

	if config.seed is not None:
		torch.manual_seed(config.seed)  # for reproducible noise in simulate
	
	marginal_prob_std_fn = functools.partial(
      marginal_prob_std,
      sigma=config.sde.sigma
    )
	diffusion_coeff_fn = functools.partial(
      diffusion_coeff, 
      sigma=config.sde.sigma
    )
	
	# TODO: getter model func and saving/loading funcs
	if config.model.model_name == 'OpenAiUNetModel':

		score_model = OpenAiUNetModel(image_size=config.data.im_size,
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
	if config.forward_op.trafo_name == 'walnut_trafo':
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

	for i, data_sample in enumerate(islice(dataset, config.data.validation.num_images)):

		if len(data_sample) == 2 and config.data.name == 'EllipseDatasetFromDival':
			
			_, ground_truth = data_sample
			ground_truth = ground_truth.to(device=config.device)
			observation, noise_level = simulate(
					ground_truth,
					ray_trafo['ray_trafo_module'],
					white_noise_rel_stddev=config.data.stddev
			)
		
		elif len(data_sample) == 3:

			observation, ground_truth, filterbackproj = data_sample
			ground_truth = ground_truth.to(device=config.device)
			observation = observation.to(device=config.device)
			filterbackproj = filterbackproj.to(device=config.device)

		if config.sampling.sampling_strategy == 'predictor_corrector':

			x_mean = pc_sampler(
							score_model=score_model, 
							marginal_prob_std=marginal_prob_std_fn, 
							ray_trafo=ray_trafo, 
							diffusion_coeff=diffusion_coeff_fn, 
							observation=observation, 
							penalty=1., 
							img_shape=ground_truth.shape[1:],
							batch_size=config.sampling.batch_size, 
							num_steps=config.sampling.num_steps, 
							snr=config.sampling.snr, 
							device=config.device, 
							eps=config.sampling.eps
				)

		else:
				
			raise NotImplementedError

		torch.save(
			{'recon': x_mean.cpu().squeeze(),
			'ground_truth': ground_truth.cpu().squeeze()}, 
			f'recon_{i}_info.pt'
			)
			
		print(f'reconstruction of sample {i}')
		print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
		print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

		fig, (ax1, ax2) = plt.subplots(1,2)
		ax1.imshow(x_mean[0,0,:,:].cpu(), cmap="gray")
		ax2.imshow(x_gt[0,0,:,:].cpu(), cmap="gray")
		fig.savefig(f'recon_{i}.png')

if __name__ == '__main__':
  coordinator()