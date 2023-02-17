import os
import torch 
import functools 
import numpy as np 
import matplotlib.pyplot as plt 
from itertools import islice

from src import (EllipseDatasetFromDival, marginal_prob_std, 
	diffusion_coeff, OpenAiUNetModel, pc_sampler, simple_trafo, simulate, PSNR, SSIM, ExponentialMovingAverage,
	get_disk_dist_ellipses_dataset, simple_sampling)
from configs.disk_ellipses_configs import get_config

def coordinator():
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


  	#score_model.load_state_dict(
  	#    torch.load(os.path.join(config.sampling.load_model_from_path, config.sampling.model_name)))
	#ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)
	#ema.load_state_dict(torch.load("ema_model.pt"))
	#ema.copy_to(score_model.parameters())
	score_model.load_state_dict(torch.load("model.pt"))
	score_model = score_model.to(config.device)
	score_model.eval()

	if config.data.name == 'EllipseDatasetFromDival':
		
		ellipse_dataset = EllipseDatasetFromDival(impl="astra_cuda")
		dataset = ellipse_dataset.get_valloader(
				batch_size=1,
				num_data_loader_workers=0
			)
	elif config.data.name == 'DiskDistributedEllipsesDataset':
		dataset = get_disk_dist_ellipses_dataset(
            fold='validation', 
            im_size=config.data.im_size, 
            length=config.data.length,
            diameter=config.data.diameter,
            device=config.device
          )
	
	else:
		raise NotImplementedError

	if config.forward_op.trafo_name == 'simple_trafo':
		ray_trafo = simple_trafo(
									im_size=config.data.im_size, 
									num_angles=config.forward_op.num_angles
								)
	else: 
		raise NotImplementedError
	
	ground_truth = dataset[0].unsqueeze(0)
	ground_truth = ground_truth.to(device=config.device)
	
	observation, noise_level = simulate(
					ground_truth,
					ray_trafo['ray_trafo_module'],
					white_noise_rel_stddev=config.data.stddev,
					return_noise_level=True
			)
	print(ground_truth.shape, observation.shape)
	
	x_fbp = ray_trafo["fbp_module"](observation)

	for penalty in [0.1, 1, 5, 10, 50, 100, 1000]:
		x_mean = simple_sampling(
						score_model=score_model, 
						marginal_prob_std=marginal_prob_std_fn, 
						ray_trafo=ray_trafo, 
						diffusion_coeff=diffusion_coeff_fn, 
						observation=observation, 
						penalty=penalty, 
						img_shape=ground_truth.shape[1:],
						batch_size=config.sampling.batch_size, 
						num_steps=config.sampling.num_steps, 
						snr=config.sampling.snr, 
						device=config.device, 
						eps=config.sampling.eps,
						start_time_step=int(0.8*config.sampling.num_steps)
				)
		x_mean = torch.clamp(x_mean, 0, 1)
		print(f'reconstruction of sample ')
		psnr = PSNR(x_mean[0, 0].cpu().numpy(), ground_truth[0,0,:,:].cpu().numpy())
		ssim = SSIM(x_mean[0, 0].cpu().numpy(), ground_truth[0,0,:,:].cpu().numpy())
		print('PSNR:', psnr)
		print('SSIM:', ssim)

		fig, (ax1, ax2, ax3) = plt.subplots(1,3)
		fig.suptitle("\n penalty={:.4f}  \n psnr={:.4f} || ssim={:.4f}".format(penalty, psnr, ssim))
		im = ax1.imshow(x_mean[0,0,:,:].cpu(), cmap="gray")
		ax1.axis("off")#
		fig.colorbar(im, ax=ax1)
		im = ax2.imshow(ground_truth[0,0,:,:].cpu(), cmap="gray")
		fig.colorbar(im, ax=ax2)
		ax2.axis("off")
		im = ax3.imshow(x_fbp[0,0,:,:].detach().cpu(), cmap="gray")
		fig.colorbar(im, ax=ax3)
		ax3.axis("off")
		fig.savefig(f'results/recon_penaly={penalty}.png')
		plt.close()
	
if __name__ == '__main__':
  coordinator()