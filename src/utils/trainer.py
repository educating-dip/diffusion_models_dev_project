from tqdm import tqdm 
import torch 
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torchvision
import os 

from .losses import loss_fn
from .ema import ExponentialMovingAverage
from ..samplers import pc_sampler_unconditional

def score_model_simple_trainer(
							score_model,
							marginal_prob_std_fn,
							diffusion_coeff_fn,
							train_dl, 
							optim_kwargs,
							val_kwargs,
							device, 
							log_dir='./'
							):

	writer = SummaryWriter(log_dir=log_dir,
							comment='training-score-model'
							)
	optimizer = Adam(score_model.parameters(), 
					lr=optim_kwargs['lr']
					)
	
	for epoch in range(optim_kwargs['epochs']):

		avg_loss = 0.
		num_items = 0

		score_model.train()
		for idx, batch in tqdm(enumerate(train_dl), total = len(train_dl)):

			if len(batch) == 1:
				x = batch
			else:
				_, x = batch 
			
			x = x.unsqueeze(0).to(device)

			loss = loss_fn(score_model, x, marginal_prob_std_fn)

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

		if val_kwargs["sample_freq"] % (epoch + 1) == 0:
			score_model.eval()
			with torch.no_grad():
				x_mean = pc_sampler_unconditional(
						score_model=score_model, 
						marginal_prob_std=marginal_prob_std_fn, 
						diffusion_coeff=diffusion_coeff_fn, 
						img_shape=x.shape[1:],
						batch_size=val_kwargs["batch_size"], 
						num_steps=val_kwargs["num_steps"], 
						snr=val_kwargs["snr"], 
						device=device, 
						eps=val_kwargs["eps"]
						)
				sample_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)
				
				writer.add_image("unconditional samples", sample_grid, global_step=epoch)