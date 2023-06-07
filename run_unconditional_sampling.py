import os
import argparse
import yaml 
import torch 
import matplotlib.pyplot as plt
import functools
from src import (
		get_standard_sde, get_standard_score, Euler_Maruyama_sde_predictor, wrapper_ddim,
		Langevin_sde_corrector, BaseSampler,  _SCORE_PRED_CLASSES, _EPSILON_PRED_CLASSES
	) 

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--load_path', help='model-checkpoint to load')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--num_steps', default=1000)

def coordinator(args):
	
	load_path = args.load_path
	with open(os.path.join(load_path, "report.yaml"), "r") as stream:
		config = yaml.load(stream, Loader=yaml.UnsafeLoader)
		config.sampling.load_model_from_path = load_path

	if config.seed is not None:
		torch.manual_seed(config.seed) # for reproducible noise in simulate

	sde = get_standard_sde(config=config)
	score = get_standard_score(config=config, sde=sde, use_ema=args.ema)
	score = score.to(config.device)
	score.eval()

	batch_size = 4
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
				'im_shape': [1, config.data.im_size, config.data.im_size],
				'travel_length': 1, 
				'travel_repeat': 1, 
				'predictor': {}
				},
			device=config.device)
	else: 
		raise NotImplementedError

		
	x_mean = sampler.sample(logging=False)
	#x_mean = torch.clamp(x_mean, 0)
	_, axes = plt.subplots(1,4)

	for idx, ax in enumerate(axes.ravel()):

		ax.imshow(x_mean[idx, 0,:,:].cpu(), cmap="gray")
		ax.axis("off")

	plt.show()
	print(x_mean.shape)

if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)