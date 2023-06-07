from typing import Optional, Any, Dict, Tuple, Union

import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from src.utils.cg import cg
from src.utils import SDE, VESDE, VPSDE, DDPM, _EPSILON_PRED_CLASSES, _SCORE_PRED_CLASSES
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
    Implements the predictor step using Euler-Maruyama for VE/VP-SDE models
    in  1. @article{song2020score,
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
    assert not any([isinstance(sde,classname) for classname in _EPSILON_PRED_CLASSES])
    if nloglik is not None: assert (datafitscale is not None) and (penalty is not None)
    x.requires_grad_()
    s = score(x, time_step).detach() if not aTweedy else score(x, time_step)
    if nloglik is not None:
        if aTweedy: xhat0 = apTweedy(s=s, x=x, sde=sde, time_step=time_step)
        loss = nloglik(x if not aTweedy else xhat0)
        nloglik_grad = torch.autograd.grad(outputs=loss, inputs=x)[0]
    drift, diffusion = sde.sde(x, time_step)
    _s = s
    # if ``penalty == 1/σ2'' and ``aTweedy'' is False : recovers Eq.4 in 1.
    if aTweedy and nloglik is not None: datafitscale = loss.pow(-1)

    if nloglik is not None and not aTweedy: _s = _s - penalty*nloglik_grad*datafitscale # minus for negative log-lik.
    x_mean = x - (drift - diffusion[:, None, None, None].pow(2)*_s)*step_size
    noise = torch.sqrt(diffusion[:, None, None, None].pow(2)*step_size)*torch.randn_like(x)

    x = x_mean + noise # Algo.1  in 3. line 6
    if aTweedy: x = x - penalty*nloglik_grad*datafitscale # Algo.1 sin 3. line 7

    return x.detach(), x_mean.detach()

def Langevin_sde_corrector(
    score: OpenAiUNetModel,
    sde: SDE, # pylint: disable=unused-variable
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
    assert not any([isinstance(sde,classname) for classname in _EPSILON_PRED_CLASSES])

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
    time_step: Union[Tensor, Tuple[Tensor,Tensor]],
    eta: float,
    gamma: float,
    step_size: float,
    cg_kwargs: Dict,
    datafitscale: Optional[float] = None, # pylint: disable=unused-variable
    use_simplified_eqn: bool = False,
    ray_trafo: callable = None
    ) -> Tuple[Tensor, Tensor]:

    '''
    It implements ``Decomposed Diffusion Sampling'' for the VE/VP-SDE model 
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
    t = time_step if not isinstance(time_step, Tuple) else time_step[0]
    with torch.no_grad():
        s = score(x, t).detach()
        xhat0 = apTweedy(s=s, x=x, sde=sde, time_step=t) # Tweedy denoising step
        rhs = xhat0 + gamma*rhs
        xhat = cg(x=xhat0, ray_trafo=ray_trafo,  rhs=rhs, gamma=gamma, n_iter=cg_kwargs['max_iter'])
        '''
        It implemets the predictor sampling strategy presented in
            2. @article{song2020denoising,
                title={Denoising diffusion implicit models},
                author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
                journal={arXiv preprint arXiv:2010.02502},
                year={2020}
            }, available at https://arxiv.org/pdf/2010.02502.pdf.
        '''
        x = ddim(
            sde=sde, 
            s=s, 
            xhat=xhat, 
            time_step=time_step, 
            step_size=step_size, 
            eta=eta, 
            use_simplified_eqn=use_simplified_eqn,
            )

    return x.detach(), xhat.detach()

def _adapt(
    x: Tensor, 
    score: nn.Module, 
    sde: SDE,
    loss_fn: callable,
    time_step: Tensor, 
    num_steps: int,
    lr: float = 1e-3
    ) -> None:
    
    score.eval()
    optim = torch.optim.Adam(score.parameters(), lr=lr)
    for _ in range(num_steps):
        optim.zero_grad()
        s = score(x, time_step)
        xhat0 = apTweedy(s=s, x=x, sde=sde, time_step=time_step)
        loss = loss_fn(x=xhat0)
        loss.backward()
        optim.step()

def adapted_ddim_sde_predictor( 
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    time_step: Union[Tensor, Tuple[Tensor,Tensor]],
    eta: float,
    step_size: float,
    adapt_fn: callable,
    use_adapt: bool = False,
    datafitscale: Optional[float] = None, # pylint: disable=unused-variable
    use_simplified_eqn: bool = False,
    add_cg: bool = False,
    gamma: float = None,
    cg_kwargs: Dict = None, 
    ray_trafo: callable = None,
    rhs: Tensor = None
    ) -> Tuple[Tensor, Tensor]:

    t = time_step if not isinstance(time_step, Tuple) else time_step[0]
    if use_adapt : adapt_fn(x=x, time_step=t)
    with torch.no_grad():
        s = score(x, t)
        xhat0 = apTweedy(s=s, x=x, sde=sde, time_step=t)

        if add_cg:
            rhs = xhat0 + gamma*rhs
            xhat = cg(x=xhat0, ray_trafo=ray_trafo,  rhs=rhs, gamma=gamma, n_iter=cg_kwargs['max_iter'])

    x = ddim(
        sde=sde,
        s=s,
        xhat=xhat0,
        time_step=time_step,
        step_size=step_size,
        eta=eta, 
        use_simplified_eqn=use_simplified_eqn, 
        )
    
    return x.detach(), xhat0.detach()

def ddim(
    sde: SDE,
    s: Tensor,
    xhat: Tensor,
    time_step: Union[Tensor, Tuple[Tensor,Tensor]],
    step_size: Tensor, 
    eta: float, 
    use_simplified_eqn: bool = False
    ) -> Tensor:
    
    t = time_step if not isinstance(time_step, Tuple) else time_step[0]
    tminus1 = time_step-step_size if not isinstance(time_step,Tuple) else time_step[1]
    std_t = sde.marginal_prob_std(t=t)[:, None, None, None]
    if isinstance(sde, VESDE):
        std_tminus1 = sde.marginal_prob_std(t=tminus1)[:, None, None, None]
        tbeta = 1 - ( std_tminus1.pow(2) * std_t.pow(-2) ) if not use_simplified_eqn else torch.tensor(1.) 
        noise_deterministic = - std_tminus1*std_t*torch.sqrt( 1 - tbeta.pow(2)*eta**2 ) * s
        noise_stochastic = std_tminus1 * eta*tbeta*torch.randn_like(xhat)
    elif any([isinstance(sde, classname) for classname in [VPSDE, DDPM]]):
        mean_tminus1 = sde.marginal_prob_mean(t=tminus1)[:, None, None, None]
        mean_t = sde.marginal_prob_mean(t=t)[:, None, None, None]
        tbeta = ((1 - mean_tminus1.pow(2)) / ( 1 - mean_t.pow(2) ) ).pow(.5) * (1 - mean_t.pow(2) * mean_tminus1.pow(-2) ).pow(.5)
        if any(tbeta.isnan()): tbeta = torch.zeros(*tbeta.shape, device=s.device)
        xhat = xhat*mean_tminus1
        eps_ = _eps_pred_from_s(s, std_t) if isinstance(sde, VPSDE) else s
        # DDIM sampling scheme is derive using a epsilon-matsching parametrization
        noise_deterministic = torch.sqrt( 1 - mean_tminus1.pow(2) - tbeta.pow(2)*eta**2 )*eps_
        noise_stochastic = eta*tbeta*torch.randn_like(xhat)
    else:
        raise NotImplementedError

    return xhat + noise_deterministic + noise_stochastic

def apTweedy(s: Tensor, x: Tensor, sde: SDE, time_step:Tensor) -> Tensor:

    div = sde.marginal_prob_mean(time_step)[:, None, None, None].pow(-1)
    std_t = sde.marginal_prob_std(time_step)[:, None, None, None]
    if any([isinstance(sde, classname) for classname in _SCORE_PRED_CLASSES]):
        s = _eps_pred_from_s(s=s, std_t=std_t) # `s' here is `eps_.'
    update = x - s*std_t

    return update*div

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

def _eps_pred_from_s(s, std_t):
    # based on score-matching = - epsilon-mathcing / std_t
    """s obtained with score-matching, converting to epsilon-prediction"""

    return - std_t * s


def _check_times(times, t_0, num_steps):

    assert times[0] > times[1], (times[0], times[1])

    assert times[-1] == -1, times[-1]

    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= num_steps, (t, num_steps)

def _schedule_jump(num_steps, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, num_steps - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = num_steps
    time_steps = []
    while t >= 1:
        t = t-1
        time_steps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                time_steps.append(t)
    time_steps.append(-1)
    _check_times(time_steps, -1, num_steps)

    return time_steps

def wrapper_ddim(
    score: OpenAiUNetModel, 
    sde: SDE, 
    x: Tensor, 
    time_step: Tensor, 
    step_size: Tensor, 
    datafitscale = 1. # pylint: disable=unused-variable
    ) -> Tuple[Tensor, Tensor]:

    t = time_step if not isinstance(time_step, Tuple) else time_step[0]
    with torch.no_grad():
        s = score(x, t).detach()
        xhat0 = apTweedy(s=s, x=x, sde=sde, time_step=t)
    # setting ``eta'' equals to ``0'' turns ddim into ddpm
    x = ddim(sde=sde, s=s, xhat=xhat0, time_step=time_step, step_size=step_size, eta=0.85, use_simplified_eqn=False)
    
    return x.detach(), xhat0.detach()