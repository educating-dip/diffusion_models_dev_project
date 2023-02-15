import os
import torch 
import functools 
import numpy as np 
import matplotlib.pyplot as plt 

from src import (EllipseDatasetFromDival, marginal_prob_std, 
	diffusion_coeff, OpenAiUNetModel, pc_sampler, simple_trafo, simulate, PSNR, SSIM)
from configs.ellipses_configs import get_config

def coordinator():

	if cfg.seed is not None:
		torch.manual_seed(config.seed)  # for reproducible noise in simulate
	
	config = get_config()
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

    score_model = OpenAiUNetModel(image_size = config.data.im_size,
                    in_channels = config.model.in_channels,
                    model_channels = config.model,
                    out_channels = config.model,
                    num_res_blocks = config.model,
                    attention_resolutions = config.model ,
                    marginal_prob_std = marginal_prob_std_fn,
                    channel_mult= config.model,
                    conv_resample= config.model,
                    dims= config.model,
                    num_heads= config.model,
                    num_head_channels= config.model,
                    num_heads_upsample= config.model,
                    use_scale_shift_norm=config.model,
                    resblock_updown=config.model,
                    use_new_attention_order=config.model)
	else:
		
		raise NotImplementedError


  score_model.load_state_dict(
      torch.load(os.path.join(config.sampling.load_model_from_path, config.sampling.model_name)))
  score_model = score_model.to(config.device)
  score_model.eval()

	if config.data.name == 'EllipseDatasetFromDival':
  	ellipse_dataset = EllipseDatasetFromDival(impl="astra_cuda")
		dataset = ellipse_dataset.get_valloader(
				batch_size=1,
				num_data_loader_workers=0
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
	
	for i, data_sample in enumerate(islice(dataset), config.data.validation.num_images):

		if len(data_sample) == 2 and config.data.name == 'EllipseDatasetFromDival': 
			
			_, ground_truth = data_sample
			ground_truth = ground_truth.to(device=config.device)
			observation, noise_level = simulate(
					ground_truth,
					ray_trafo['ray_trafo_module'],
					white_noise_rel_stddev=config.data.stddev
			)
		
		elif len(data_sample) == 3:

			observation, ground_truth, _ = data_sample
			ground_truth = ground_truth.to(device=config.device)
			observation = observation.to(device=config.device)

  if config.sampling.sampling_strategy == 'predictor_corrector':

    x_mean = pc_sampler(
					score_model=score_model, 
					marginal_prob_std=marginal_prob_std_fn, 
					ray_trafo=ray_trafo, 
					diffusion_coeff=diffusion_coeff_fn, 
					observation=observation, 
					noise_level=noise_level, 
					img_shape=x.shape[1:],
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