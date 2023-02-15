from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from .losses import loss_fn

def score_model_simple_trainer(
    score_model,
    marginal_prob_std_fn,
    train_dl, 
    optim_kwargs,
    device, 
    log_dir='./'
  ):

  writer = SummaryWriter(log_dir=log_dir,
        comment='training-score-model'
    )
  optimizer = Adam(score_model.parameters(), 
      lr=optim_kwargs['lr']
    )
  for epoch in range(optim_kwargs['n_epochs']):
    
    avg_loss = 0.
    num_items = 0

    for idx, batch in tqdm(enumerate(train_dl), total = len(train_dl)):
      _, x = batch
      x = torch.randn(x.shape[0], 1, *x.shape[2:])
      x = x.to(device)

      loss = loss_fn(score_model, x, marginal_prob_std_fn)
      
      optimizer.zero_grad()
      loss.backward()    
      optimizer.step()

      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]

      if idx % optim_kwargs['log_freq'] == 0:
        writer.add_scalar("train/loss", loss.item(), epoch*len(train_dl) + idx) 

      if epoch == 0 and idx == optim_kwargs['ema_warm_start_steps']:
        ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)

      if idx > optim_kwargs['ema_warm_start_steps'] or epoch > 0:
        ema.update(score_model.parameters())
    
    # Print the averaged training loss so far.
    print('Average Loss: {:5f}'.format(avg_loss / num_items))
    writer.add_scalar("train/mean_loss_per_epoch", avg_loss / num_items, epoch + 1)
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), 'model.pt')
    torch.save(ema.state_dict(), 'ema_model.pt')