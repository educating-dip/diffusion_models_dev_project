import os 
import torch 
import torchvision
import numpy as np 

from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from .losses import loss_fn, cond_loss_fn
from .ema import ExponentialMovingAverage

def score_model_simple_trainer(
							score_model,
							sde, 
							train_dl, 
							optim_kwargs,
							val_kwargs,
							device, 
							log_dir='./'
							):

	writer = SummaryWriter(log_dir=log_dir, comment='training-score-model')
	optimizer = Adam(score_model.parameters(), lr=optim_kwargs['lr'])

	for epoch in range(optim_kwargs['epochs']):

		avg_loss, num_items = 0, 0
		score_model.train()
		for idx, batch in tqdm(enumerate(train_dl), total = len(train_dl)):

			x = batch.to(device)
			loss = loss_fn(score_model, x, sde)
			optimizer.zero_grad()
			loss.backward()    
			optimizer.step()

			avg_loss += loss.item() * x.shape[0]
			num_items += x.shape[0]
			if idx % optim_kwargs['log_freq'] == 0:
				writer.add_scalar("train/loss", loss.item(), epoch*len(train_dl) + idx) 
			if epoch == 0 and idx == optim_kwargs['ema_warm_start_steps']:
				ema = ExponentialMovingAverage(score_model.parameters(), decay=optim_kwargs["ema_decay"])
			if idx > optim_kwargs['ema_warm_start_steps'] or epoch > 0:
				ema.update(score_model.parameters())


		print('Average Loss: {:5f}'.format(avg_loss / num_items))
		writer.add_scalar("train/mean_loss_per_epoch", avg_loss / num_items, epoch + 1)
		torch.save(score_model.state_dict(), os.path.join(log_dir,'model.pt'))
		torch.save(ema.state_dict(), os.path.join(log_dir, 'ema_model.pt'))

		if  epoch % val_kwargs["sample_freq"]== 0:
			score_model.eval()
			with torch.no_grad():
				x_mean = pred_cor_uncond_sampling(
						score_model=score_model, 
						sde = sde, 
						img_shape=x.shape[1:],
						batch_size=val_kwargs["batch_size"], 
						num_steps=val_kwargs["num_steps"], 
						snr=val_kwargs["snr"], 
						device=device, 
						eps=val_kwargs["eps"]
						)
				sample_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)
				writer.add_image("unconditional samples", sample_grid, global_step=epoch)



def cond_score_model_trainer(
							score_model,
							sde, 
							train_dl, 
							optim_kwargs,
							val_kwargs,
							device, 
							log_dir='./'
							):

	writer = SummaryWriter(log_dir=log_dir, comment='training-score-model')
	optimizer = Adam(score_model.parameters(), lr=optim_kwargs['lr'])

	for epoch in range(optim_kwargs['epochs']):

		avg_loss, num_items = 0, 0
		score_model.train()
		for idx, batch in tqdm(enumerate(train_dl), total = len(train_dl)):
			y, x = batch 
			x = x.to(device)
			y = y.to(device)
			loss = cond_loss_fn(score_model, x, y, sde)
			optimizer.zero_grad()
			loss.backward()    
			optimizer.step()

			avg_loss += loss.item() * x.shape[0]
			num_items += x.shape[0]
			if idx % optim_kwargs['log_freq'] == 0:
				writer.add_scalar("train/loss", loss.item(), epoch*len(train_dl) + idx) 
			if epoch == 0 and idx == optim_kwargs['ema_warm_start_steps']:
				ema = ExponentialMovingAverage(score_model.parameters(), decay=optim_kwargs["ema_decay"])
			if idx > optim_kwargs['ema_warm_start_steps'] or epoch > 0:
				ema.update(score_model.parameters())


		print('Average Loss: {:5f}'.format(avg_loss / num_items))
		writer.add_scalar("train/mean_loss_per_epoch", avg_loss / num_items, epoch + 1)
		torch.save(score_model.state_dict(), os.path.join(log_dir,'model.pt'))
		torch.save(ema.state_dict(), os.path.join(log_dir, 'ema_model.pt'))

		if  epoch % val_kwargs["sample_freq"]== 0:
			score_model.eval()
			with torch.no_grad():
				x_mean = simple_cond_sampler(
						score_model=score_model, 
						sde = sde, 
						img_shape=x.shape[1:],
						y = y, 
						num_steps=val_kwargs["num_steps"], 
						snr=val_kwargs["snr"], 
						eps=val_kwargs["eps"]
						)
				sample_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)
				writer.add_image("unconditional samples", sample_grid, global_step=epoch)


def simple_cond_sampler(score_model, sde, img_shape, y, num_steps, snr, eps):
	batch_size = y.shape[0]

	init_x = sde.prior_sampling([batch_size, *img_shape]).to(y.device)

	time_steps = np.linspace(1., eps, num_steps)
	step_size = time_steps[0] - time_steps[1]

	with torch.no_grad():
		for time_step in tqdm(time_steps):
			batch_time_step = torch.ones(batch_size, device=y.device) * time_step

			grad = score_model(x, y, batch_time_step)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = np.sqrt(np.prod(x.shape[1:]))
			langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
			x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

			# Predictor step (Euler-Maruyama) xt -> xt-1
			f, g = sde.sde(x, batch_time_step)
			x_mean = x - (f - (g**2)[:, None, None, None] * score_model(x, y, batch_time_step)) * step_size
			x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

			# The last step does not include any noise

	return x_mean
