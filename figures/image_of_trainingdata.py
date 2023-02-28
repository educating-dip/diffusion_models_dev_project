import numpy as np 
import matplotlib.pyplot as plt 

from torchvision.utils import make_grid
import torch 

from PIL import Image


from src import get_disk_dist_ellipses_dataset

from configs.disk_ellipses_configs import get_config


config = get_config()


dataset = get_disk_dist_ellipses_dataset(
              fold='train', 
              im_size=config.data.im_size, 
              length=config.data.length,
              diameter=config.data.diameter,
              max_n_ellipse=config.data.num_n_ellipse, 
              device="cpu"
            )

print(len(dataset))

dl = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False)

x = next(iter(dl))

print(x.shape)

img_grid = make_grid(x, normalize=True, scale_each=True, padding=4, pad_value = 1.0, nrow=6)

im = Image.fromarray(img_grid.numpy()[0,:,:]*255.).convert("L")
im.save("training_data.png")


print(img_grid.shape)

plt.figure()
plt.imshow(img_grid[0,:,:],cmap="gray")
plt.axis("off")


plt.show()