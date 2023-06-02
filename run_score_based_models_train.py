import os
import yaml
import argparse
from src import (get_standard_sde, score_model_simple_trainer, get_standard_score, 
		 get_standard_train_dataset)

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--sde', default='vesde', choices=['vpsde', 'vesde', 'ddpm'])
parser.add_argument('--base_path', default='/localdata/AlexanderDenker/score_based_baseline')
parser.add_argument('--train_model_on', default='ellipses', help='training datasets', choices=['lodopab', 'ellipses'])

def coordinator(args):

	if args.train_model_on == 'ellipses': 
		from configs.disk_ellipses_configs import get_config
		config = get_config(args)
	elif args.train_model_on == 'lodopab': 
		from configs.lodopab_configs import get_config
		config = get_config(args)
	else: 
		raise NotImplementedError

	sde = get_standard_sde(config=config)
	score = get_standard_score(config=config, sde=sde, use_ema=False, load_model=False)
	base_path = args.base_path
	if config.data.name == 'LoDoPabCT':
		log_dir = os.path.join(base_path, 'LoDoPabCT')
	elif config.data.name == 'DiskDistributedEllipsesDataset':
		log_dir = os.path.join(base_path, 'DiskEllipses')
	else:
		raise NotImplementedError

	log_dir = os.path.join(log_dir, config.sde.type)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	found_version = False 
	version_num = 1
	while not found_version:
		if os.path.isdir(os.path.join(log_dir, "version_{:02d}".format(version_num))):
			version_num += 1
		else:
			found_version = True 

	log_dir = os.path.join(log_dir, "version_{:02d}".format(version_num))
	print("save model to ", log_dir)
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
	args = parser.parse_args()
	coordinator(args)
