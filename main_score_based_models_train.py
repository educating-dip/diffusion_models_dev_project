import torch 
import functools 
import numpy as np 
import matplotlib.pyplot as plt 
import numpy as np 

from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from src import loss_fn, ExponentialMovingAverage, marginal_prob_std, diffusion_coeff, UNetModel, EllipseDatasetFromDival
from configs.ellipses_configs import get_config

def coordinator():
  config = get_config()

  # TODO:
  device = "cuda"
  batch_size = 6
  n_epochs =   50
  lr=1e-4

  marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25.0)
  diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=25.0)

  ellipse_dataset = EllipseDatasetFromDival(impl="astra_cuda")
  train_dl = ellipse_dataset.get_trainloader(batch_size=batch_size, num_data_loader_workers=0)

  image_size = 128
  score_model = UNetModel(image_size = image_size,
                  in_channels = 1,
                  model_channels = 32,
                  out_channels = 1,
                  num_res_blocks = 2,
                  attention_resolutions = [image_size // 16, image_size // 8] ,
                  marginal_prob_std = marginal_prob_std_fn,
                  channel_mult=(1, 2, 4, 8),
                  conv_resample=True,
                  dims=2,
                  num_heads=1,
                  num_head_channels=-1,
                  num_heads_upsample=-1,
                  use_scale_shift_norm=True,
                  resblock_updown=False,
                  use_new_attention_order=False)
  print("#Parameters: ", sum([p.numel() for p in score_model.parameters()]))
  score_model = score_model.to(device)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)
  
  # TODO: 
  writer = SummaryWriter(log_dir="checkpoints", comment ="Training")
  optimizer = Adam(score_model.parameters(), lr=lr)
  write_every = 15 

  for epoch in range(n_epochs):
    avg_loss = 0.
    num_items = 0

    for idx, batch in tqdm(enumerate(train_dl), total = len(train_dl)):
      _, x = batch
      x = x.to(device)
      loss = loss_fn(score_model, x, marginal_prob_std_fn)
      optimizer.zero_grad()
      loss.backward()    
      optimizer.step()
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]

      if idx % write_every == 0:
        writer.add_scalar("train/loss", loss.item(), epoch*len(train_dl) + idx) 

      ema.update(score_model.parameters())
    # Print the averaged training loss so far.
    print('Average Loss: {:5f}'.format(avg_loss / num_items))
    writer.add_scalar("train/mean_loss_per_epoch", avg_loss / num_items, epoch + 1) 
    
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), 'checkpoints/model.pt')
    torch.save(ema.state_dict(), 'checkpoints/ema_model.pt')

if __name__ == '__main__': 
  coordinator()