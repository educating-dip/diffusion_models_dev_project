import torch
import numpy as np 
from tqdm import tqdm 
from scipy import integrate

from torch.utils.tensorboard import SummaryWriter
import torchvision

from .utils import PSNR, SSIM


def posterior_sampling(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                observation,
                penalty,
                ray_trafo,
                img_shape,
                batch_size, 
                num_steps, 
                snr,          
				predictor="euler_maruyama",
				corrector="langevin",
				corrector_steps = 1,
				method="naive",
                device='cuda',
                start_time_step = 0,
                eps=1e-3,
                x_fbp = None,
				log_dir = None, 
				num_img_log=10,
				log_freq=10,
				ground_truth = None):
	
	"""
	method = "naive": Jalal et al. (2020) https://arxiv.org/pdf/2209.14687.pdf
	method = "dps": Chung et al. (2022) https://proceedings.neurips.cc/paper/2020/file/07cb5f86508f146774a2fac4373a8e50-Paper.pdf

	"""
	assert method in ["naive", "dps"], f"method {method} is not supported. Use *naive* or *dps*."

	if not log_dir == None:
		writer = SummaryWriter(log_dir=log_dir)

		log_img_interval = (num_steps - start_time_step)/num_img_log

	time_steps = np.linspace(1., eps, num_steps)
	step_size = time_steps[0] - time_steps[1]

	if start_time_step == 0:
		t = torch.ones(batch_size, device=device)
		init_x = torch.randn(batch_size, *img_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
	else:
		if x_fbp == None:
			x_fbp = ray_trafo["fbp_module"](observation)
		std = marginal_prob_std(torch.ones(batch_size, device=device) * time_steps[start_time_step])
		z = torch.randn(batch_size, *img_shape, device=device)
		init_x = x_fbp + z * std[:, None, None, None]

	if not log_dir == None:
		init_x_grid = torchvision.utils.make_grid(init_x, normalize=True, scale_each=True)		
		writer.add_image("init_x", init_x_grid, global_step=0)

		fbp_grid = torchvision.utils.make_grid(x_fbp, normalize=True, scale_each=True)		
		writer.add_image("fbp", fbp_grid, global_step=0)

	x = init_x
	for i in tqdm(range(start_time_step, num_steps)):     
		time_step = time_steps[i]
		batch_time_step = torch.ones(batch_size, device=device) * time_step

		# Corrector step (Langevin MCMC)
		if corrector == "langevin": 
			if method == "naive":
				for _ in range(corrector_steps):
					with torch.no_grad():
						s = score_model(x, batch_time_step)

					x = x.requires_grad_()

					norm = torch.linalg.norm(observation - ray_trafo['ray_trafo_module'](x))
					norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
			
					grad = s - penalty*norm_grad*i/num_steps
					grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
					noise_norm = np.sqrt(np.prod(x.shape[1:]))
					langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
					x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)  
					x = x.detach()

			elif method == "dps":
				for _ in range(corrector_steps):
					x = x.requires_grad_()
					s = score_model(x, batch_time_step)
					std = marginal_prob_std(batch_time_step)
					xhat0 = x + s * std[:, None, None, None]

					norm = torch.linalg.norm(observation - ray_trafo['ray_trafo_module'](xhat0))
					norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

					grad = s - norm_grad * penalty
					grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
					noise_norm = np.sqrt(np.prod(x.shape[1:]))
					langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
					x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)  
					x = x.detach()

		# Predictor step (Euler Maruyama) 
		if predictor == "euler_maruyama":
			if method == "naive":
				with torch.no_grad():
					s = score_model(x, batch_time_step)

				x = x.requires_grad_()

				norm = torch.linalg.norm(observation - ray_trafo['ray_trafo_module'](x))
				norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
		
				grad = s - penalty*norm_grad *i/num_steps
				
				g = diffusion_coeff(batch_time_step)

				x_mean = x + (g**2)[:, None, None, None] * grad * step_size
				x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

				x = x.detach()
				x_mean = x_mean.detach()


			elif method == "dps":
				x = x.requires_grad_()
				s = score_model(x, batch_time_step)
				std = marginal_prob_std(batch_time_step)
				xhat0 = x + s * std[:, None, None, None]**2

				norm = torch.linalg.norm(observation - ray_trafo['ray_trafo_module'](xhat0))
				norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

				grad = s - norm_grad/norm * penalty
				
				g = diffusion_coeff(batch_time_step)

				x_mean = x + (g**2)[:, None, None, None] * grad * step_size
				x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

				x = x.detach()
				x_mean = x_mean.detach()

		if not log_dir == None:
			if (i - start_time_step) % log_img_interval == 0:
				x_grid = torchvision.utils.make_grid(x, normalize=True, scale_each=True)		
				writer.add_image("reco_at_step", x_grid, global_step=i)

			if i % log_freq == 0:
				writer.add_scalar("|Ax-y|", norm.item(), global_step=i)
				psnr = PSNR(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				ssim = SSIM(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				writer.add_scalar("PSNR", psnr, global_step=i)
				writer.add_scalar("SSIM", ssim, global_step=i)


	if not log_dir == None:
		x_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)		
		writer.add_image("reco_at_step", x_grid, global_step=num_steps)

	return x_mean






def pc_sampler(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                observation,
                penalty,
                ray_trafo,
                img_shape,
                batch_size, 
                num_steps, 
                snr,                
                device='cuda',
                eps=1e-3,
				start_time_step=0, 
                x_fbp=None,
				log_dir = None, 
				num_img_log=10,
				log_freq=10,
				ground_truth = None
                ):


	if not log_dir == None:
		writer = SummaryWriter(log_dir=log_dir)

		log_img_interval = (num_steps - start_time_step)/num_img_log

	time_steps = np.linspace(1., eps, num_steps)
	step_size = time_steps[0] - time_steps[1]

	if start_time_step == 0:
		t = torch.ones(batch_size, device=device)
		init_x = torch.randn(batch_size, *img_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
	else:
		if x_fbp == None:
			x_fbp = ray_trafo["fbp_module"](observation)
		std = marginal_prob_std(torch.ones(batch_size, device=device) * time_steps[start_time_step])
		z = torch.randn(batch_size, *img_shape, device=device)
		init_x = x_fbp + z * std[:, None, None, None]

	if not log_dir == None:
		init_x_grid = torchvision.utils.make_grid(init_x, normalize=True, scale_each=True)		
		writer.add_image("init_x", init_x_grid, global_step=0)

	x = init_x
	for i in tqdm(range(start_time_step, num_steps)):
		time_step = time_steps[i]
		batch_time_step = torch.ones(batch_size, device=device) * time_step

		''' We depart from ``` DIFFUSION POSTERIOR SAMPLING FOR GENERAL NOISY INVERSE PROBLEMS'' [https://arxiv.org/pdf/2209.14687.pdf] with the 
			the inclusion of the corrector step which is proposed in Algo. 2 in the appedinx in 
			SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS [https://arxiv.org/pdf/2011.13456.pdf]. '''
      
		#with torch.no_grad():
		#	batch_time_step = torch.ones(batch_size, device=device) * time_step
		#	# Corrector step (Langevin MCMC)
		#	grad = score_model(x, batch_time_step)
		#	grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
		#	noise_norm = np.sqrt(np.prod(x.shape[1:]))
		#	langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
		#	x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

		x = x.requires_grad_()
		s = score_model(x, batch_time_step)
		std = marginal_prob_std(batch_time_step)
		xhat0 = x + s * std[:, None, None, None]

		#norm = torch.linalg.norm(ray_trafo['ray_trafo_adjoint_module'](observation - ray_trafo['ray_trafo_module'](xhat0)))
		norm = torch.linalg.norm(observation - ray_trafo['ray_trafo_module'](xhat0))
		norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

		# Predictor step (Euler-Maruyama)
		g = diffusion_coeff(batch_time_step)
		#x_mean = x + (g**2)[:, None, None, None] * (s - 1/noise_level**2 * norm_grad / 122.) * step_size

		x_mean = x + (g**2)[:, None, None, None] * (s -  norm_grad * penalty) * step_size

		x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

		x = x.detach()
		x_mean = x_mean.detach()
		# The last step does not include any noise

		if not log_dir == None:
			if i - start_time_step % log_img_interval == 0:
				x_grid = torchvision.utils.make_grid(x, normalize=True, scale_each=True)		
				writer.add_image("reco_at_step", x_grid, global_step=i)

			if i % log_freq == 0:
				writer.add_scalar("|Ax-y|", norm.item(), global_step=i)
				psnr = PSNR(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				ssim = SSIM(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				writer.add_scalar("PSNR", psnr, global_step=i)
				writer.add_scalar("SSIM", ssim, global_step=i)


	return x_mean

  
def pc_sampler_unconditional(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                img_shape,
                batch_size, 
                num_steps, 
                snr,                
                device='cuda',
                eps=1e-3):

	''' Sampling from the score function '''

	t = torch.ones(batch_size, device=device)
	init_x = torch.randn(batch_size, *img_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
	time_steps = np.linspace(1., eps, num_steps)
	step_size = time_steps[0] - time_steps[1]

	x = init_x

	with torch.no_grad():
		for time_step in tqdm(time_steps):
			batch_time_step = torch.ones(batch_size, device=device) * time_step
			# Corrector step (Langevin MCMC)
			grad = score_model(x, batch_time_step)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = np.sqrt(np.prod(x.shape[1:]))
			langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
			x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

			# Predictor step (Euler-Maruyama)
			g = diffusion_coeff(batch_time_step)
			x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
			x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

			# The last step does not include any noise

	return x_mean


def naive_sampling(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                observation,
                penalty,
                ray_trafo,
                img_shape,
                batch_size, 
                num_steps, 
                snr,                
                device='cuda',
                start_time_step = 0,
                eps=1e-3,
                x_fbp = None,
				log_dir = None, 
				num_img_log=10,
				log_freq=10,
				ground_truth = None):

	''' Based on ``Robust Compressed Sensing MRI with Deep Generative Priors'' [https://arxiv.org/pdf/2108.01368.pdf] '''

	if not log_dir == None:
		writer = SummaryWriter(log_dir=log_dir)

		log_img_interval = (num_steps - start_time_step)/num_img_log

	time_steps = np.linspace(1., eps, num_steps)
	step_size = time_steps[0] - time_steps[1]

	if start_time_step == 0:
		t = torch.ones(batch_size, device=device)
		init_x = torch.randn(batch_size, *img_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
	else:
		if x_fbp == None:
			x_fbp = ray_trafo["fbp_module"](observation)
		std = marginal_prob_std(torch.ones(batch_size, device=device) * time_steps[start_time_step])
		z = torch.randn(batch_size, *img_shape, device=device)
		init_x = x_fbp + z * std[:, None, None, None]

	if not log_dir == None:
		init_x_grid = torchvision.utils.make_grid(init_x, normalize=True, scale_each=True)		
		writer.add_image("init_x", init_x_grid, global_step=0)

		fbp_grid = torchvision.utils.make_grid(x_fbp, normalize=True, scale_each=True)		
		writer.add_image("fbp", fbp_grid, global_step=0)

	x = init_x
	for i in tqdm(range(start_time_step, num_steps)):     
		time_step = time_steps[i]
		batch_time_step = torch.ones(batch_size, device=device) * time_step

		with torch.no_grad():
			s = score_model(x, batch_time_step)

		x = x.requires_grad_()

		norm = torch.linalg.norm(observation - ray_trafo['ray_trafo_module'](x))
		norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
		
		#norm_grad = norm_grad / torch.linalg.norm(norm_grad)
		#norm_grad = norm_grad * torch.linalg.norm(s)


		# Predictor step (Euler-Maruyama)
		g = diffusion_coeff(batch_time_step)
		#x_mean = x + (g**2)[:, None, None, None] * (s - 1/noise_level**2 * norm_grad / 122.) * step_size

		score_update = (g**2)[:, None, None, None] * s  * step_size
		data_discrepancy_update =  (g**2)[:, None, None, None] * norm_grad * step_size


		x_mean = x + score_update - penalty*data_discrepancy_update*i/num_steps

		x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

		x = x.detach()
		x_mean = x_mean.detach()
		# The last step does not include any noise

		if not log_dir == None:
			if (i - start_time_step) % log_img_interval == 0:
				x_grid = torchvision.utils.make_grid(x, normalize=True, scale_each=True)		
				writer.add_image("reco_at_step", x_grid, global_step=i)

			if i % log_freq == 0:
				writer.add_scalar("|Ax-y|", norm.item(), global_step=i)
				psnr = PSNR(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				ssim = SSIM(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				writer.add_scalar("PSNR", psnr, global_step=i)
				writer.add_scalar("SSIM", ssim, global_step=i)

	if not log_dir == None:
		x_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)		
		writer.add_image("reco_at_step", x_grid, global_step=num_steps)

	return x_mean




def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
				img_shape,
                batch_size=64, 
                atol=1e-5 , 
                rtol=1e-5 , 
                device='cuda', 
                z=None,
                eps=1e-3):
 
	t = torch.ones(batch_size, device=device)
	# Create the latent code
	if z is None:
		init_x = torch.randn(batch_size, *img_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
	else:
		init_x = z

	shape = init_x.shape

	def score_eval_wrapper(sample, time_steps):
		"""A wrapper of the score-based model for use by the ODE solver."""
		sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
		time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
		
		with torch.no_grad():    
			score = score_model(sample, time_steps)
		return score.cpu().numpy().reshape((-1,)).astype(np.float64)

	def ode_func(t, x):        
		"""The ODE function for use by the ODE solver."""
		time_steps = np.ones((shape[0],)) * t    
		g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
		return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

	# Run the black-box ODE solver.
	res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
	print(f"Number of function evaluations: {res.nfev}")
	x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

	return x




def optimize_sampling(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                observation,
                penalty,
                ray_trafo,
                img_shape,
                batch_size, 
                num_steps, 
                snr,                
                device='cuda',
                start_time_step = 0,
                eps=1e-3,
                x_fbp = None,
				log_dir = None, 
				num_img_log=10,
				log_freq=10,
				ground_truth = None):


	if not log_dir == None:
		writer = SummaryWriter(log_dir=log_dir)

		log_img_interval = (num_steps - start_time_step)/num_img_log

	time_steps = np.linspace(1., eps, num_steps)
	step_size = time_steps[0] - time_steps[1]

	if start_time_step == 0:
		t = torch.ones(batch_size, device=device)
		init_x = torch.randn(batch_size, *img_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
	else:
		if x_fbp == None:
			x_fbp = ray_trafo["fbp_module"](observation)
		std = marginal_prob_std(torch.ones(batch_size, device=device) * time_steps[start_time_step])
		z = torch.randn(batch_size, *img_shape, device=device)
		init_x = x_fbp + z * std[:, None, None, None]

	if not log_dir == None:
		init_x_grid = torchvision.utils.make_grid(init_x, normalize=True, scale_each=True)		
		writer.add_image("init_x", init_x_grid, global_step=0)

		fbp_grid = torchvision.utils.make_grid(x_fbp, normalize=True, scale_each=True)		
		writer.add_image("fbp", fbp_grid, global_step=0)

	x = init_x

	score_model.train()
	optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-2)

	for i in tqdm(range(start_time_step, num_steps)):     

		time_step = time_steps[i]
		batch_time_step = torch.ones(batch_size, device=device) * time_step

		for _ in range(3):
			optimizer.zero_grad() 
			# one optimizer step on theta
			s = score_model(x, batch_time_step)
			std = marginal_prob_std(batch_time_step)
			xhat0 = x + s * std[:, None, None, None]

			loss = torch.linalg.norm(observation - ray_trafo['ray_trafo_module'](xhat0))
			loss.backward()
			
			optimizer.step() 

		with torch.no_grad():
			s = score_model(x, batch_time_step)
			g = diffusion_coeff(batch_time_step)

			x_mean = x + (g**2)[:, None, None, None] * s  * step_size 

			x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      



		if not log_dir == None:
			if (i - start_time_step) % log_img_interval == 0:
				x_grid = torchvision.utils.make_grid(x, normalize=True, scale_each=True)		
				writer.add_image("reco_at_step", x_grid, global_step=i)

			if i % log_freq == 0:
				#writer.add_scalar("|Ax-y|", norm.item(), global_step=i)
				psnr = PSNR(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				ssim = SSIM(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				writer.add_scalar("PSNR", psnr, global_step=i)
				writer.add_scalar("SSIM", ssim, global_step=i)

	if not log_dir == None:
		x_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)		
		writer.add_image("reco_at_step", x_grid, global_step=num_steps)
	
	# The last step does not include any noise
	return x_mean


def cg_sampling(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                observation,
                penalty,
                ray_trafo,
                img_shape,
                batch_size, 
                num_steps, 
                snr,                
                device='cuda',
                start_time_step = 0,
                eps=1e-3,
                x_fbp = None,
				log_dir = None, 
				num_img_log=10,
				log_freq=10,
				ground_truth = None,
				time_schedule = 'linear'):

	''' Based on ``DPS'' TODO '''

	if not log_dir == None:
		writer = SummaryWriter(log_dir=log_dir)

		log_img_interval = (num_steps - start_time_step)/num_img_log

	if time_schedule == "linear":
		time_steps = torch.linspace(1., eps, num_steps + 1)
	else:
		rho = 7
		ramp = torch.linspace(0, 1, num_steps + 1)
		time_steps = (1 + ramp*(eps**(1/rho) - 1))**rho


	if start_time_step == 0:
		t = torch.ones(batch_size, device=device)
		init_x = torch.randn(batch_size, *img_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
	else:
		if x_fbp == None:
			x_fbp = ray_trafo["fbp_module"](observation/torch.max(observation))
		std = marginal_prob_std(torch.ones(batch_size, device=device) * time_steps[start_time_step])
		z = torch.randn(batch_size, *img_shape, device=device)
		init_x = x_fbp + z * std[:, None, None, None]

	if not log_dir == None:
		init_x_grid = torchvision.utils.make_grid(init_x, normalize=True, scale_each=True)		
		writer.add_image("init_x", init_x_grid, global_step=0)

		if not x_fbp == None:
			fbp_grid = torchvision.utils.make_grid(x_fbp, normalize=True, scale_each=True)		
			writer.add_image("fbp", fbp_grid, global_step=0)

	x_fbp = ray_trafo["fbp_module"](observation/torch.max(observation))
	y_odl = ray_trafo["ray_trafo"].range.element(observation[0,0,:,:].cpu().numpy())

	x = init_x 
	for i in tqdm(range(start_time_step, num_steps)):     
		time_step = time_steps[i]
		batch_time_step = torch.ones(batch_size, device=device) * time_step
		
		step_size = torch.abs(time_steps[i+1] - time_steps[i])

		# cg step 
		x_cg = torch.clone(x)
		x_cg_odl = ray_trafo["ray_trafo"].domain.element(x[0,0,:,:].cpu().numpy())
		conjugate_gradient_normal(ray_trafo["ray_trafo"], x_cg_odl, y_odl, niter=4)

		x_proj = torch.from_numpy(np.asarray(x_cg_odl)).to(x.device)
		x_proj = x_proj.unsqueeze(0).unsqueeze(0)/torch.max(observation)

		l = 0.
		x = l*x + (1-l)*x_proj

		x = x.requires_grad_()
		with torch.no_grad():
			s = score_model(x, batch_time_step) 
		
		# Predictor step (Euler-Maruyama)
		g = diffusion_coeff(batch_time_step)
		x_mean = x + (g**2)[:, None, None, None] * s  * step_size 
		x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

		x = x.detach()
		x_mean = x_mean.detach()

		with torch.no_grad():
			
			snr = 0.16
			batch_time_step = torch.ones(batch_size, device=device) * time_step
			# Corrector step (Langevin MCMC)
			grad = score_model(x, batch_time_step)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = np.sqrt(np.prod(x.shape[1:]))
			langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
			x_mean = x + langevin_step_size * grad 
			x = x_mean + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)     

		# The last step does not include any noise

		if not log_dir == None:
			if (i - start_time_step) % log_img_interval == 0:
				x_grid = torchvision.utils.make_grid(x, normalize=True, scale_each=True)		
				writer.add_image("reco_at_step", x_grid, global_step=i)

			if i % log_freq == 0:
				psnr = PSNR(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				ssim = SSIM(x[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
				writer.add_scalar("PSNR", psnr, global_step=i)
				writer.add_scalar("SSIM", ssim, global_step=i)

	if not log_dir == None:
		x_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)		
		writer.add_image("reco_at_step", x_grid, global_step=num_steps)

	return x_mean
