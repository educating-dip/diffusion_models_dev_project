

import torch 
from torch.optim import Adam
import functools 
import numpy as np 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

from dataset import EllipseDataset
from model import ScoreNet
from sde import marginal_prob_std, diffusion_coeff
from ema import ExponentialMovingAverage 

# Hyperparemters
device = "cuda"
sigma =  25.0
batch_size = 32
n_epochs =   50
## learning rate
lr=1e-4


marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

#model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)

#print(sum(p.numel() for p in model.parameters()))

ellipse_dataset = EllipseDataset(impl="astra_cuda")

train_dl = ellipse_dataset.get_trainloader(batch_size=3, num_data_loader_workers=0)


_, x = next(iter(train_dl))
x = x.to("cuda")
print(x.shape)

t = torch.linspace(1e-5, 1-1e-5, 7)
### look at noisy samples (they should look like Gaussian noise)
fig, axes = plt.subplots(x.shape[0], len(t) + 1)
for i in range(len(t)):
    with torch.no_grad():
        t_ = torch.ones(x.shape[0], device=x.device) * t[i]
        std = marginal_prob_std_fn(t_)
        z = torch.randn_like(x)
        perturbed_x = x + z * std[:, None, None, None]

    axes[0, i].set_title(str(t[i].item()))
    for j in range(x.shape[0]):
        axes[j, i].imshow(perturbed_x[j,0,:,:].cpu(), cmap="gray")
        axes[j,i].axis("off")

        if i == len(t) - 1:
            axes[j, -1].hist(perturbed_x[j,0,:,:].cpu().ravel(), bins="auto")

plt.show()


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


train_dl = ellipse_dataset.get_trainloader(batch_size=batch_size, num_data_loader_workers=0)

  
score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
score_model = score_model.to(device)

ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)


writer = SummaryWriter(log_dir="checkpoints", comment ="Training")


#score_model.load_state_dict(torch.load("model.pt"))


optimizer = Adam(score_model.parameters(), lr=lr)

print("Number of batches per epoch: ", len(train_dl))
write_every = 15 

for epoch in range(n_epochs):
  avg_loss = 0.
  num_items = 0
  for idx, batch in tqdm(enumerate(train_dl)):
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