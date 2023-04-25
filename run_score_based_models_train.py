import os
import yaml
import torch
import functools

from datetime import datetime
from src import (get_sde, marginal_prob_std, diffusion_coeff, OpenAiUNetModel, 
				EllipseDatasetFromDival, get_disk_dist_ellipses_dataset, 
				score_model_simple_trainer, get_one_ellipses_dataset,
				get_standard_score)

from configs.disk_ellipses_configs import get_config

def coordinator(args):
	if args.dataset == "ellipses":
		from configs.disk_ellipses_configs import get_config
	elif args.dataset == "lodopab":
		from configs.lodopab_configs import get_config

	config = get_config()
	sde = get_sde(config=config)

	print("DATASET: ", config.data.name)
	if config.data.name == 'EllipseDatasetFromDival':
		#TODO: implement getter for datasets
		ellipse_dataset = EllipseDatasetFromDival(impl="astra_cuda")
		train_dl = ellipse_dataset.get_trainloader(
						batch_size=config.training.batch_size, num_data_loader_workers=0)
	elif config.data.name == 'DiskDistributedEllipsesDataset':
		if config.data.num_n_ellipse > 1:
			dataset = get_disk_dist_ellipses_dataset(
				fold='train',
				im_size=config.data.im_size, 
				length=config.data.length,
				diameter=config.data.diameter,
				max_n_ellipse=config.data.num_n_ellipse,
				device=config.device)
	else:
		dataset = get_one_ellipses_dataset(
				fold='train',
				im_size=config.data.im_size,
				length=config.data.length,
				diameter=config.data.diameter,
				device=config.device)

	train_dl = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)

	score = get_standard_score(config=config, sde=sde, use_ema=False, load_model=False)

	log_dir = './score_based_baseline/checkpoints/' + datetime.now().strftime('%Y_%m_%d_%H:%m')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	with open(os.path.join(log_dir,'report.yaml'), 'w') as file:
		yaml.dump(config, file)

	score_model_simple_trainer(
			score_model=score_model.to(config.device),
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
