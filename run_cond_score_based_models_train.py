"""
Train a LearnedPrimalDualNetwork as a conditional score model.
"""


import os
import yaml
import torch
import functools

from odl.contrib.torch import OperatorModule


from datetime import datetime
from src import (get_sde, PrimalDualNet, cond_score_model_trainer,
				get_standard_score, LoDoPabDatasetFromDival)


def coordinator():
	from configs.lodopab_dival_configs import get_config

	config = get_config()
	sde = get_sde(config=config)

	dataset = LoDoPabDatasetFromDival(im_size=config.data.im_size , use_transform=False)

	train_dl = dataset.get_trainloader(batch_size=6)

	score = PrimalDualNet(n_iter=config.model.n_iter, 
						op=OperatorModule(dataset.ray_trafo), 
						op_adj=OperatorModule(dataset.ray_trafo.adjoint), 
						sde=sde, 
						n_primal=config.model.n_primal, 
						n_dual=config.model.n_dual,
						n_layer=config.model.n_layer, 
						internal_ch=config.model.internal_ch, 
						kernel_size=config.model.kernel_size,
						batch_norm=config.model.batch_norm)

	log_dir = '/localdata/AlexanderDenker/score_based_baseline/primal_dual/checkpoints/' + datetime.now().strftime('%Y_%m_%d_%H:%m')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	with open(os.path.join(log_dir,'report.yaml'), 'w') as file:
		yaml.dump(config, file)

	cond_score_model_trainer(
			score_model=score.to(config.device),
			sde = sde,
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
