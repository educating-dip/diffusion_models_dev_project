# Score-based Models for Inverse Problems 

## Update 28.02.2023

This repository contains code for experiments on score-based models.

### Model

We use [guided diffusion model](https://github.com/openai/guided-diffusion) from OpenAI based on this [paper](https://arxiv.org/abs/2105.05233). The guided diffusion model is based on the UNet architecture with additional attention models. The original implementation can only handle images of the size $2^k \times 2^k$, we adapted the upsampling part slightly to deal with images of arbitrary size.

By default this model only uses the attention block at the smallest resolution. So also use attention at other resolution one must set the argument **attention_resolutions**. In the configs of guided diffusion this is set 
```python
OpenAiUNetModel( ...,
                attention_resolutions = [im_size // 16, im_size // 8],
                ...)
```

However, because of the way they construct the model, this only has an effect if the image is of size $2^k \times 2^k$. If we want to use use attention models also at different resolution we have to either change the code in the **OpenAiUNetModel** or write the resolution as the nearest power of 2, i.e.
```python
OpenAiUNetModel( ...,
                attention_resolutions = [32, 64],
                ...)
```
This will also include attention blocks at resolution levels $501 / 16$ and $501 / 8$.

### Sampling Methods & Conditioning Methods  

We always have two distinct choices when sampling from the posterior: 
1. How to integrate the measurements?
2. What sampler to use for the reverse SDE?

Currently our code mixes both of these choices and the names in **samplers.py** are confusing. 

#### 1. How to integrate the measurements?

There are different ways to guide the sampling process to produce samples from $p(x|y)$ given measurements $y$.

One line of work decomposes the score of the posterior $\nabla_x \log p_t(x|y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y|x)$ and use different methods to approximate $\nabla_x \log p_t(y|x)$:

- [Jalal et al. 2020](https://proceedings.neurips.cc/paper/2020/file/07cb5f86508f146774a2fac4373a8e50-Paper.pdf): $\nabla_{x_t} \log p_t(y|x_t) \approx \frac{-A^*(A x_t - y)}{\sigma^2 + \gamma_t^2}$ (using Gaussian noise in measurment process) where $\sigma$ is the noise level of the measurements and $\gamma_t$ an additional hyperparameter with $\gamma_t \to 0$ for $t \to 0$.
- [Chung et al. 2022a](https://arxiv.org/pdf/2209.14687.pdf): $\nabla_{x_t} \log p_t(y|x_t) \approx \nabla_{x_t} \log p(y|\hat{x}_0(x_t))$ where $\hat{x}_0(x_t)$ is a denoised version of $x_t$ using Tweedies formula. This results in $\nabla_{x_t} \log p(y|x_t) \approx - \frac{1}{\sigma^2} \nabla_{x_t} \| y - A(\hat{x}_0(x_t)) \|_2^2$ (using Gaussian noise in measurment process). 

Another line of works sample from the unconditional score model and do another projection step on the data consistency manifold $\{x : Ax = y \}$:

- [Song et al. 2022](https://arxiv.org/pdf/2111.08005.pdf): 
- [Chung et al. 2022b](https://arxiv.org/pdf/2206.00941.pdf): (with Yong Chul Ye)
- [Kawar et al. 2022](https://arxiv.org/pdf/2201.11793.pdf): (also with Song & Ermon)
- [Kawar et al. 2021](https://proceedings.neurips.cc/paper/2021/file/b5c01503041b70d41d80e3dbe31bbd8c-Paper.pdf):

For these methods it seems to be important to also take into accounts the resulting diffusion process on $y$ by $y_t = A x_t$. I dont like these methods but maybe this is just because I dont understand them.

#### 2. What sampler to use for the reverse SDE?

Assuming the have some way of integrating the measurements into the sampling process, using some approximation of the likelihood or some projection step, the next question is: how to sample from the reverse SDE?

Notation:

$$ d x = f(x,t) dt + g(t) dw \Leftrightarrow dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) dw $$

The left hand side runs forward in time, i.e. $dt$ is a positive time step. The ride hand side runs backward in time, i.e. $dt$ is a negative time step. 

- Ancestral Sampling: Sampling used in the original DDPM paper. If I understand it right, the process is at follows: In a first step the SDE is discretized and the model is only trained on this specific discretization, during sampling the reverse SDE is discretized in the same way. (everything becomes a discrete Markov Chain). Our model works in a continouos setting, i.e. $s_\theta(x, t)$ with $t \in [0,1]$ instead of $s_\theta(x, \sigma_i)$ for $i=1, \dots, N$, so we do not use this sampling method. 
- Reverse Diffusion (only predictor - Euler Maruyama):
$$ \text{for } t = T, \dots, 0 \\ x^{t} = x^{t+1} + \Delta t (f(x^{t+1}, t+1) - g(t+1)^2 s_\theta(x^{t+1}, t+1)) + g(t+1) \sqrt{|\Delta t|} z \\ z \sim \mathcal{N}(0,I) $$
important to note here is that $\Delta t$ is a **negative** time step.
- Reverse Diffusion (predictor + corrector): Do one step of euler Maruyama to go from $t+1$ to $t$, then do $L$ steps at $t$ to refine the current sample. The corrector step is (this is exactly unadjusted Langevin for $p_t(x)$):
$$ \text{for } i = 1, ..., L \\ 
x^{t}_i = x^{t}_{i-1} + \epsilon_{i-1} s_\theta(x^{t}_{i-1}, t) + \sqrt{2 \epsilon_{i-1}} z \\
z \sim \mathcal{N}(0,I) $$

In this [paper](https://arxiv.org/pdf/2011.13456.pdf#section.3) the author show that 1000 steps of predictor-corrector (L=1, so one step backwards in time, one step with the corrector, i.e. two network evaluations for one predictor-corrector step) works better than 2000 steps of just predictor.


- Hamiltonian Monte Carlo, see [Reduce, Reuse, Recycle](https://arxiv.org/abs/2302.11552). I think this is only possible if we parametrize the score using the energy based formulation.


To put this together, for sampling we have two option: only predictor (Euler Maruyama) or predictor-corrector (Euler-Maruyam and Langevin)