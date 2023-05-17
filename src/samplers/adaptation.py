import torch 
import torch.nn as nn
from torch import Tensor

# why is the import not working?
#from .utils import _aTweedy

def _aTweedy(s: Tensor, x: Tensor, sde, time_step:Tensor) -> Tensor:

    update = x + s*sde.marginal_prob_std(time_step)[:, None, None, None].pow(2)
    div = sde.marginal_prob_mean(time_step)[:, None, None, None].pow(-1)
    return update*div



def tv_loss(x):

    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

def adapt_decoder(x: Tensor, 
                score: nn.Module, 
                time_step: Tensor, 
                sde,
                observation: Tensor,
                ray_trafo,
                tv_penalty: float, 
                num_steps: int,
                lr: float = 3e-4) -> nn.Module:
    
    score.train() 
    for name, param in score.named_parameters():
        param.requires_grad = False

    for name, param in score.out.named_parameters():
        if not "emb_layers" in name:
            param.requires_grad = True
        
    for name, param in score.output_blocks.named_parameters():
        if not "emb_layers" in name:
            param.requires_grad = True
        
    all_parameters = sum([p.numel() for p in score.parameters()])
    trainable_parameters = sum([p.numel() for p in score.parameters() if p.requires_grad])

    print("Percent of parameters to re-train: ", trainable_parameters/all_parameters*100., " %")

    optim = torch.optim.Adam(score.parameters(), lr=lr)
    for i in range(num_steps):
        optim.zero_grad()
        s = score(x, time_step)
        xhat0 = _aTweedy(s=s, x=x, sde=sde, time_step=time_step)

        loss = torch.mean((ray_trafo(xhat0) - observation)**2)  + tv_penalty * tv_loss(xhat0)
        loss.backward()
        optim.step()

    score.eval()

    return score 


def full_adapt(x: Tensor, 
                score: nn.Module, 
                time_step: Tensor, 
                sde,
                observation: Tensor,
                ray_trafo,
                tv_penalty: float, 
                num_steps: int,
                lr: float = 3e-4) -> nn.Module:
    
    score.train() 
    for name, param in score.named_parameters():
        param.requires_grad = True
        
    all_parameters = sum([p.numel() for p in score.parameters()])
    trainable_parameters = sum([p.numel() for p in score.parameters() if p.requires_grad])

    print("Percent of parameters to re-train: ", trainable_parameters/all_parameters*100., " %")

    optim = torch.optim.Adam(score.parameters(), lr=lr)
    for i in range(num_steps):
        optim.zero_grad()
        s = score(x, time_step)
        xhat0 = _aTweedy(s=s, x=x, sde=sde, time_step=time_step)

        loss = torch.mean((ray_trafo(xhat0) - observation)**2)  + tv_penalty * tv_loss(xhat0)
        loss.backward()
        optim.step()

    score.eval()

    return score 