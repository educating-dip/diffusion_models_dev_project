from typing import Optional, Any, Dict, Tuple

import torch
import numpy as np

from torch import Tensor
from src.utils.impl_linear_cg import linear_cg
from src.utils import SDE, VESDE, VPSDE
from src.physics import BaseRayTrafo
from src.third_party_models import OpenAiUNetModel

def Euler_Maruyama_sde_predictor(
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    time_step: Tensor,
    step_size: float,
    nloglik: Optional[callable] = None,
    datafitscale: Optional[float] = None,
    penalty: Optional[float] = None,
    aTweedy: bool = False
    ) -> Tuple[Tensor, Tensor]:
    '''
    Implements the predictor step using Euler-Maruyama 
    (i.e., see Eq.30) in 
        1. @article{song2020score,
            title={Score-based generative modeling through stochastic differential equations},
            author={Song, Yang and Sohl-Dickstein, Jascha and Kingma,
                Diederik P and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
            journal={arXiv preprint arXiv:2011.13456},
            year={2020}
        }, available at https://arxiv.org/abs/2011.13456.
    If ``aTweedy'' is ``False'', it implements: ``Robust Compressed Sensing MRI with Deep Generative Priors''. 
        2. @inproceedings{NEURIPS2021_7d6044e9,
            author = {Jalal, Ajil and Arvinte, Marius and Daras, Giannis and Price, Eric and Dimakis, Alexandros G and Tamir, Jon},
            booktitle = {Advances in Neural Information Processing Systems},
            editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
            pages = {14938--14954},
            publisher = {Curran Associates, Inc.},
            title = {Robust Compressed Sensing MRI with Deep Generative Priors},
            url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/7d6044e95a16761171b130dcb476a43e-Paper.pdf},
            volume = {34},
            year = {2021}
        }. Be aware that the implementation departs from ``Jalal et al.'' as it does not use annealed Langevin MCMC.
    If ``aTweedy`` is ``True'', it implements the predictor method named ``Diffusion Posterior Sampling'', presented in 
        3. @article{chung2022diffusion,
            title={Diffusion posterior sampling for general noisy inverse problems},
            author={Chung, Hyungjin and Kim, Jeongsol and Mccann, Michael T and Klasky, Marc L and Ye, Jong Chul},
            journal={arXiv preprint arXiv:2209.14687},
            year={2022}
        }, available at https://arxiv.org/pdf/2209.14687.pdf.
    '''
    if nloglik is not None: assert (datafitscale is not None) and (penalty is not None)
    x.requires_grad_()
    s = score(x, time_step).detach() if not aTweedy else score(x, time_step)
    if nloglik is not None:
        if aTweedy: xhat0 = _aTweedy(s=s, x=x, sde=sde, time_step=time_step)
        loss = nloglik(x if not aTweedy else xhat0)
        nloglik_grad = torch.autograd.grad(outputs=loss, inputs=x)[0]
    drift, diffusion = sde.sde(x, time_step)
    _s = s
    # if ``penalty == 1/σ2'' and ``aTweedy'' is False : recovers Eq.4 in 1.
    if aTweedy and nloglik is not None: datafitscale = loss.pow(-1)
    if nloglik is not None: _s = _s - penalty*nloglik_grad*datafitscale # minus for negative log-lik.
    x_mean = x - (drift - diffusion[:, None, None, None].pow(2)*_s)*step_size
    noise = torch.sqrt(diffusion[:, None, None, None].pow(2)*step_size)*torch.randn_like(x)
    x = x_mean + noise

    return x.detach(), x_mean.detach()

def Langevin_sde_corrector(
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    time_step: Tensor,
    nloglik: Optional[callable] = None,
    datafitscale: Optional[float] = None,
    penalty: Optional[float] = None,
    corrector_steps: int = 1,
    snr: float = 0.16,
    ) -> Tensor:

    ''' 
    Implements the corrector step using Langevin MCMC   
    '''
    if nloglik is not None: assert (datafitscale is not None) and (penalty is not None)
    for _ in range(corrector_steps):
        x.requires_grad_()
        s = score(x, time_step).detach()
        if nloglik is not None: nloglik_grad = torch.autograd.grad(outputs=nloglik(x), inputs=x)[0]
        overall_grad = s - penalty*nloglik_grad*datafitscale if nloglik is not None else s
        overall_grad_norm = torch.norm(
                overall_grad.reshape(overall_grad.shape[0], -1), 
                dim=-1  ).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / overall_grad_norm)**2
        x = x + langevin_step_size * overall_grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
    return x.detach()

def decomposed_diffusion_sampling_sde_predictor( 
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    rhs: Tensor,
    time_step: Tensor,
    conj_grad_closure: callable,
    eta: float,
    gamma: float,
    step_size: float,
    cg_kwargs: Dict,
    datafitscale: Optional[float] = None # placeholder 
    ) -> Tuple[Tensor, Tensor]:

    '''
    It implements ``Decomposed Diffusion Sampling'' for the VE-SDE model 
        presented in 
            1. @article{chung2023fast,
                title={Fast Diffusion Sampler for Inverse Problems by Geometric Decomposition},
                author={Chung, Hyungjin and Lee, Suhyeon and Ye, Jong Chul},
                journal={arXiv preprint arXiv:2303.05754},
                year={2023}
            },
    available at https://arxiv.org/pdf/2303.05754.pdf. See Algorithm 4 in Appendix. 
    '''
    '''
    Implements the Tweedy denosing step proposed in ``Diffusion Posterior Sampling''.
    '''
    datafitscale = 1. # lace-holder

    s = score(x, time_step).detach()
    xhat0 = _aTweedy(s=s, x=x, sde=sde, time_step=time_step) # Tweedy denoising step
    rhs_flat = rhs.reshape(np.prod(xhat0.shape[2:]), xhat0.shape[0])
    initial_guess = xhat0.reshape(np.prod(xhat0.shape[2:]), xhat0.shape[0])
    reg_rhs_flat = rhs_flat*gamma + initial_guess
    xhat, _= linear_cg(
        matmul_closure=conj_grad_closure, 
        rhs=reg_rhs_flat, 
        initial_guess=initial_guess, 
        **cg_kwargs # early-stop CG
        )
    xhat = xhat.T.view(xhat0.shape[0], 1, *xhat0.shape[2:])
    '''
    It implemets the predictor sampling strategy presented in
        2. @article{song2020denoising,
            title={Denoising diffusion implicit models},
            author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
            journal={arXiv preprint arXiv:2010.02502},
            year={2020}
        }, available at https://arxiv.org/pdf/2010.02502.pdf.
    '''
    x = _ddim_dds(sde=sde, s=s, xhat=xhat, time_step=time_step, step_size=step_size, eta=eta)

    return x.detach(), xhat

def _ddim_dds(
    sde: SDE,
    s: Tensor,
    xhat: Tensor,
    time_step: Tensor,
    step_size: Tensor, 
    eta: float
    ) -> Tensor:
    
    std_t = sde.marginal_prob_std(t=time_step
        )[:, None, None, None]
    std_tminus1 = sde.marginal_prob_std(t=time_step-step_size
        )[:, None, None, None]
    if isinstance(sde, VESDE):
        tbeta = 1 - (std_tminus1.pow(2) * std_t.pow(-2))
        noise_deterministic = - std_tminus1*std_t*torch.sqrt( 1 - tbeta.pow(2)*eta**2 ) * s
        noise_stochastic = eta*tbeta*torch.randn_like(xhat)
    elif isinstance(sde, VPSDE):
        mean_tminus1 = sde.marginal_prob_mean(t=time_step-step_size
            )[:, None, None, None]
        mean_t = sde.marginal_prob_mean(t=time_step
            )[:, None, None, None]
        tbeta = ((1 - mean_tminus1.pow(2)) / ( 1 - mean_t.pow(2) ) ).pow(.5) * (1 - mean_t.pow(2) * mean_tminus1.pow(-2) ).pow(.5) 
        xhat = xhat*mean_tminus1
        noise_deterministic = torch.sqrt( 1 - mean_tminus1.pow(2) - beta.pow(2)*eta**2 )*s
        noise_stochastic = eta*beta*torch.randn_like(xhat)
    else:
        raise NotImplementedError

    return xhat + noise_deterministic + noise_stochastic

def _aTweedy(s: Tensor, x: Tensor, sde: SDE, time_step:Tensor) -> Tensor:

    update = x + s*sde.marginal_prob_std(time_step)[:, None, None, None].pow(2)
    div = sde.marginal_prob_mean(time_step)[:, None, None, None].pow(-1)
    return update*div

def conj_grad_closure(x: Tensor, ray_trafo: BaseRayTrafo, gamma: float = 1e-5):

    batch_size = x.shape[-1]
    x = x.T.reshape(batch_size, 1, *ray_trafo.im_shape)
    return (gamma*ray_trafo.trafo_adjoint(ray_trafo(x)) + x).view(batch_size, np.prod(ray_trafo.im_shape)).T

def chain_simple_init(
    time_steps: Tensor,
    sde: SDE, 
    filtbackproj: Tensor, 
    start_time_step: int, 
    im_shape: Tuple[int, int], 
    batch_size: int, 
    device: Any
    ) -> Tensor:

    t = torch.ones(batch_size, device=device) * time_steps[start_time_step]
    std = sde.marginal_prob_std(t)[:, None, None, None]
    return filtbackproj + torch.randn(batch_size, *im_shape, device=device) * std