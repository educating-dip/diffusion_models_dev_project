import os
import torch
import argparse
import functools
import matplotlib.pyplot as plt

from src import (marginal_prob_std, diffusion_coeff, OpenAiUNetModel, 
		pred_cor_uncond_sampling, ExponentialMovingAverage)

parser = argparse.ArgumentParser(description='unconditional sampling')
parser.add_argument("--num_steps", default=1000)
parser.add_argument("--batch_size", default=6)
parser.add_argument("--ema", action='store_true')

def coordinator(args):

	from configs.disk_ellipses_configs import get_config
	config = get_config()

	if config.seed is not None:
		torch.manual_seed(config.seed)  # for reproducible noise in simulate
	
	marginal_prob_std_fn = functools.partial(
      marginal_prob_std,
      sigma_min=config.sde.sigma_min, 
	  sigma_max=config.sde.sigma_max)
	diffusion_coeff_fn = functools.partial(
      diffusion_coeff, 
      sigma_min=config.sde.sigma_min, 
	  sigma_max=config.sde.sigma_max)
	
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
                    use_new_attention_order=config.model.use_new_attention_order)
	else:
		raise NotImplementedError


	if (config.sampling.load_model_from_path is not None) and (config.sampling.model_name is not None): 
		print("load score model from path")
		if args.ema:
			ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)
			ema.load_state_dict(torch.load(os.path.join(config.sampling.load_model_from_path,"ema_model.pt")))
			ema.copy_to(score_model.parameters())
		else:
			score_model.load_state_dict(torch.load(
				os.path.join(config.sampling.load_model_from_path, config.sampling.model_name))	)

	score_model = score_model.to(config.device)
	score_model.eval()
	x_mean = pred_cor_uncond_sampling(
				score_model=score_model, 
				marginal_prob_std=marginal_prob_std_fn, 
				diffusion_coeff=diffusion_coeff_fn, 
				img_shape=[1, config.data.im_size, config.data.im_size],
				batch_size=int(args.batch_size),
				num_steps=int(args.num_steps), 
				snr=config.sampling.snr, 
				device=config.device, 
				eps=config.sampling.eps)


	_, axes = plt.subplots(1, x_mean.shape[0])
	for idx, ax in enumerate(axes.ravel()):
		ax.imshow(x_mean[idx, 0,:,:].cpu(), cmap="gray")
		ax.axis("off")
	plt.show()

if __name__ == '__main__':
	
	args = parser.parse_args()

	coordinator(args)