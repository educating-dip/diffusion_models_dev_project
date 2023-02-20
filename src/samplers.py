import torch
import numpy as np 
from tqdm import tqdm 
from scipy import integrate



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
                x_fbp=None
                ):


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

	x = init_x
	for i in tqdm(range(start_time_step, num_steps)):
		time_step = time_steps[i]
		batch_time_step = torch.ones(batch_size, device=device) * time_step

		''' We depart from ``` DIFFUSION POSTERIOR SAMPLING FOR GENERAL NOISY INVERSE PROBLEMS'' [https://arxiv.org/pdf/2209.14687.pdf] with the 
			the inclusion of the corrector step which is proposed in Algo. 2 in the appedinx in 
			SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS [https://arxiv.org/pdf/2011.13456.pdf]. '''
      
		with torch.no_grad():
			batch_time_step = torch.ones(batch_size, device=device) * time_step
			# Corrector step (Langevin MCMC)
			grad = score_model(x, batch_time_step)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = np.sqrt(np.prod(x.shape[1:]))
			langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
			x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

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
                x_fbp = None):

	''' Based on ``Robust Compressed Sensing MRI with Deep Generative Priors'' [https://arxiv.org/pdf/2108.01368.pdf] '''

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

	x = init_x
	for i in tqdm(range(start_time_step, num_steps)):     
		time_step = time_steps[i]
		batch_time_step = torch.ones(batch_size, device=device) * time_step

		with torch.no_grad():
			s = score_model(x, batch_time_step)

		x = x.requires_grad_()

		norm = torch.linalg.norm(observation - ray_trafo['ray_trafo_module'](x))
		norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

		# Predictor step (Euler-Maruyama)
		g = diffusion_coeff(batch_time_step)
		#x_mean = x + (g**2)[:, None, None, None] * (s - 1/noise_level**2 * norm_grad / 122.) * step_size

		score_update = (g**2)[:, None, None, None] * s  * step_size
		data_discrepancy_update =  (g**2)[:, None, None, None] * norm_grad * step_size

		x_mean = x + score_update - penalty*data_discrepancy_update

		x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

		x = x.detach()
		x_mean = x_mean.detach()
		# The last step does not include any noise

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