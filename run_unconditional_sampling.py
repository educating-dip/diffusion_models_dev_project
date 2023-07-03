import os
import argparse
import yaml 
import torch 
import matplotlib.pyplot as plt
import functools
from omegaconf import OmegaConf
import numpy as np 

from src import (
		get_standard_sde, get_standard_score, Euler_Maruyama_sde_predictor, wrapper_ddim,
		Langevin_sde_corrector, BaseSampler,  _SCORE_PRED_CLASSES, _EPSILON_PRED_CLASSES, ExponentialMovingAverage
	) 

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--load_path', help='model-checkpoint to load')
parser.add_argument('--base_path', default='/localdata/AlexanderDenker/score_based_baseline', help='path to model configs')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--num_steps', default=100)

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
	
	load_path = args.load_path
	#with open(os.path.join(args.base_path, 'report.yaml'), 'r') as stream:
	with open(args.base_path, 'r') as stream:
		config = yaml.load(stream, Loader=yaml.UnsafeLoader)
		config = OmegaConf.create(config)
	print(config)
	model_type = "dds_unet" 
	sde = get_standard_sde(config=config)
	score = get_standard_score(config=config, sde=sde, 
			use_ema=args.ema, model_type="dds_unet", load_model=False)
	if model_type == "dds_unet":
		if args.ema:
			ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
			ema.load_state_dict(torch.load(args.load_path))
			ema.copy_to(score.parameters())	
		else:
			score.load_state_dict(torch.load(args.load_path))
		print(f'Model ckpt loaded from {args.load_path}')
		score.convert_to_fp32()
		score.dtype = torch.float32

	else:
		if args.ema:
			ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
			ema.load_state_dict(torch.load(args.load_path))
			ema.copy_to(score.parameters())	
		else:
			score.load_state_dict(torch.load(args.load_path))
	score = score.to(config.device)
	score.to("cuda")
	score.eval()

	print(sum([p.numel() for p in score.parameters()]))

	batch_size = 6
	if any([isinstance(sde, classname) for classname in _SCORE_PRED_CLASSES]):
		sampler = BaseSampler(
			score=score,
			sde=sde,
			predictor=functools.partial(Euler_Maruyama_sde_predictor, nloglik = None),
			corrector=functools.partial(Langevin_sde_corrector, nloglik = None),
			init_chain_fn=None,
			sample_kwargs={
						'num_steps': int(args.num_steps),
						'start_time_step': 0,
						'batch_size': batch_size,
						'im_shape': [1, config.data.im_size, config.data.im_size],
						'eps': 1e-4,
						'predictor': {},
						'corrector': {'corrector_steps': 1}
						},
			device=config.device)
	elif any([isinstance(sde, classname) for classname in _EPSILON_PRED_CLASSES]):

		sampler = BaseSampler(
			score=score,
			sde=sde,
			predictor=wrapper_ddim, 
			corrector=None,
			init_chain_fn=None,
			sample_kwargs={
				'num_steps': 100,
				'start_time_step': 0,
				'batch_size': batch_size,
				'im_shape': [config.data.channels, config.data.im_size, config.data.im_size],
				'travel_length': 1, 
				'travel_repeat': 1, 
				'predictor': {}
				},
			device=config.device)
	else: 
		raise NotImplementedError

		
	x_mean = sampler.sample(logging=False)
	_, axes = plt.subplots(2,4)

	"""
	x = real_to_nchw_comp(x_mean)
	print(x.shape)

	print(x_mean.shape)
	for i in range(4):
		axes[0, i].imshow(x_mean[i, 0,:,:].cpu(), cmap='gray')
		axes[1, i].imshow(x_mean[i, 1,:,:].cpu(), cmap='gray')

	plt.show() 
	x_plot = np.abs(x.cpu().numpy())
	print(x_plot.shape)
	"""
	fig, axes = plt.subplots(1,4)
	for idx, ax in enumerate(axes.ravel()):
		ax.imshow(x_mean[idx, 0,:,:].cpu().numpy(), cmap='gray')
		ax.axis('off')

	plt.show()

if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)