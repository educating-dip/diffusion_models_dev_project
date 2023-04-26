import os
import yaml
import torch
import functools

from datetime import datetime
from src import (get_standard_sde, marginal_prob_std, diffusion_coeff, OpenAiUNetModel, EllipseDatasetFromDival, 
	get_disk_dist_ellipses_dataset, score_model_simple_trainer, get_one_ellipses_dataset, get_standard_score, get_standard_configs)

from configs.disk_ellipses_configs import get_config

def coordinator(args):

	config, _ = get_config(args)
	sde = get_standard_sde(config=config)
	score = get_standard_score(config=config, sde=sde, use_ema=False, load_model=False)
	log_dir = './score_based_baseline/checkpoints/' + datetime.now().strftime('%Y_%m_%d_%H:%m')

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	with open(os.path.join(log_dir,'report.yaml'), 'w') as file:
		yaml.dump(config, file)

	train_dl = get_standard_train_dataset(config)
	score_model_simple_trainer(
			score=score.to(config.device),
			sde=sde,
			train_dl=train_dl,
			optim_kwargs={
					'epochs': config.training.epochs,
					'lr': config.training.lr,
					'ema_warm_start_steps': config.training.ema_warm_start_steps,
					'log_freq': config.training.log_freq,
					'ema_decay': config.training.ema_decay
				},
			val_kwargs={
					'batch_size': config.validation.batch_size,
					'num_steps': config.validation.num_steps,
					'snr': config.validation.snr,
					'eps': config.validation.eps,
					'sample_freq' : config.validation.sample_freq
				},
		device=config.device,
		log_dir=log_dir
		)

if __name__ == '__main__':
	coordinator()
