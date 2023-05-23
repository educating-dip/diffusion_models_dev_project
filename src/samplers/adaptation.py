import torch 
import torch.nn as nn
from torch import Tensor

from src.utils import SDE

def tv_loss(x):

    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

def _score_model_adpt(
    score: nn.Module, 
    impl: str = 'full'
    ) -> None:
    
    for name, param in score.named_parameters():
        param.requires_grad = False

    if impl == 'full':
        for name, param in score.named_parameters():
            param.requires_grad = True
    elif impl == 'decoder': 
        for name, param in score.out.named_parameters():
            if not "emb_layers" in name:
                param.requires_grad = True
        for name, param in score.output_blocks.named_parameters():
            if not "emb_layers" in name:
                param.requires_grad = True
    elif impl == 'lora':
        pass
    elif impl == 'dif-fit':
        pass
    elif impl == 'vdkl':
        pass 
    else: 
        raise NotImplementedError 