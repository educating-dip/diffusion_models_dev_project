"""
Alex:

In this script the pre-trained score model will be applied to ground truth images of the LoDoPab dataset.
The images are first downsampled (cropped?) to 256x256. 

The radon transform is defined using ODL.

"""

### external libraries
import matplotlib.pyplot as plt 
import numpy as np 

from dival import get_standard_dataset

import torch 
from torch.nn.functional import interpolate

import odl
from odl import uniform_discr
from odl.contrib.torch import OperatorModule
from odl.operator.oputils import power_method_opnorm

from odl.tomo.backends import astra_cuda

print("ODL VERSION: ", odl.__version__)

#######################
### Hyperparameters ###
#######################
idx = 29
rel_noise = 0.025 #0.025
num_angles = 91


### Load Dataset and create Forward Operator ###

dataset = get_standard_dataset('lodopab', impl="astra_cuda", sorted_by_patient=True)

lodopab_train = dataset.create_torch_dataset(part='train',
                                    reshape=((1,) + dataset.space[0].shape,
                                    (1,) + dataset.space[1].shape))


_, x_gt = lodopab_train[idx]
x_gt = interpolate(x_gt.unsqueeze(0), (256,256), mode="bilinear", align_corners=True)

print("Range of x_gt: ", torch.min(x_gt), torch.max(x_gt))

domain = uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=(256,256), dtype=np.float32)
geometry = odl.tomo.parallel_beam_geometry(domain, num_angles=num_angles)

ray_trafo = odl.tomo.RayTransform(domain, geometry, impl="astra_cuda")


s = astra_cuda.astra_cuda_bp_scaling_factor(ray_trafo.range, domain, geometry)

print("SCALING FACTOR: ", s)

#print("OP NORM: ", power_method_opnorm(ray_trafo))

ray_trafo_op = OperatorModule(ray_trafo)
ray_trafo_adjoint_op = OperatorModule(ray_trafo.adjoint)
fbp_op = OperatorModule(odl.tomo.fbp_op(ray_trafo))

print("OP NORM ray trafo: ", power_method_opnorm(ray_trafo))
print("OP NORM fbp op: ", power_method_opnorm(odl.tomo.fbp_op(ray_trafo)))


y = ray_trafo_op(x_gt)

print(y.shape)
print(y.min(), y.max())

y_noise = y + rel_noise*torch.mean(torch.abs(y))*torch.randn_like(y)

x_fbp = fbp_op(y)
x_adj = ray_trafo_adjoint_op(y)

print("Range of x_fbp: ", torch.min(x_fbp), torch.max(x_fbp))
print("Range of x_adj: ", torch.min(x_adj), torch.max(x_adj))
