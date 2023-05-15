import os
import argparse
import yaml 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import islice
from itertools import islice
from src import (get_standard_sde, PSNR, SSIM, get_standard_dataset, get_data_from_ground_truth, get_standard_ray_trafo,  
	get_standard_score, get_standard_sampler, get_standard_configs, get_standard_path) 

from odl.solvers.iterative.iterative import conjugate_gradient
from odl.operator.default_ops import IdentityOperator

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--dataset', default='walnut', help='test-dataset', choices=['walnut', 'lodopab', 'ellipses', 'mayo'])
parser.add_argument('--model_learned_on', default='lodopab', help='model-checkpoint to load', choices=['lodopab', 'ellipses'])

def coordinator(args):
	if args.model_learned_on == "ellipses":
		load_path = "/localdata/AlexanderDenker/score_based_baseline/DiskEllipses/checkpoints/2023_05_07_11:05"

		with open(os.path.join(load_path, "report.yaml"), "r") as stream:
			config = yaml.load(stream, Loader=yaml.UnsafeLoader)
			config.sampling.load_model_from_path = load_path

			print(config.sde.type)

	_, dataconfig = get_standard_configs(args)
	
	ray_trafo = get_standard_ray_trafo(config=dataconfig)
	ray_trafo = ray_trafo.to(device=config.device)
	dataset = get_standard_dataset(config=dataconfig, ray_trafo=ray_trafo)

	for i, data_sample in enumerate(islice(dataset, config.data.validation.num_images)):
		if len(data_sample) == 3:
			observation, ground_truth, filtbackproj = data_sample
			ground_truth = ground_truth.to(device=config.device)
			observation = observation.to(device=config.device)
			filtbackproj = filtbackproj.to(device=config.device)
		else:
			ground_truth, observation, filtbackproj = get_data_from_ground_truth(
				ground_truth=data_sample.to(device=config.device),
				ray_trafo=ray_trafo,
				white_noise_rel_stddev=dataconfig.data.stddev
				)

		xhat0 = filtbackproj
		gamma = 0.9

		ray_trafo_op = ray_trafo.ray_trafo_op_fun.operator
		ray_trafo_adjoint_op = ray_trafo.ray_trafo_adjoint_op_fun.operator
		print(ray_trafo_op, ray_trafo_adjoint_op)


		y_odl = ray_trafo_op.range.element(observation[0,0,:,:].cpu().numpy())
		I = IdentityOperator(ray_trafo_op.domain)
		x = torch.ones(1, 1, 501, 501)
		
		# cg step 
		x_cg = torch.clone(x)
		print(x_cg.shape)
		x_cg_odl = ray_trafo_op.domain.element(x_cg[0,0,:,:].cpu().numpy())

		rhs = x_cg_odl + gamma*ray_trafo_adjoint_op(y_odl)
		operator = I + gamma*ray_trafo_adjoint_op*ray_trafo_op
		conjugate_gradient(operator, x_cg_odl, rhs, niter=4)

		x_proj = torch.from_numpy(np.asarray(x_cg_odl)).to(x.device)
		x_proj = x_proj.unsqueeze(0).unsqueeze(0)

		x = x_proj

		fig, (ax1, ax2, ax3) = plt.subplots(1,3)
		ax1.imshow(ground_truth[0,0,:,:].detach().cpu(), cmap="gray")
		ax1.axis("off")
		ax1.set_title("Ground truth")
		ax2.imshow(x[0,0,:,:].detach().cpu(), cmap="gray")
		ax2.axis("off")
		ax2.set_title("CG")
		ax3.imshow(filtbackproj[0,0,:,:].detach().cpu(), cmap="gray")
		ax3.axis("off")
		ax3.set_title("FBP")

		plt.show() 



if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)