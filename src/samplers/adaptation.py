from typing import Optional, Dict

import torch 
import torch.nn as nn
from src.third_party_models import inject_trainable_lora_extended

def tv_loss(x):

    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

# TODO: eliminate the redundancy here : only lora (@rb876)
def _score_model_adpt(
    score: nn.Module, 
    impl: str = 'full', 
    adpt_kwargs: Optional[Dict] = None,
    verbose: bool = True
    ) -> nn.Module:
    
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
        """ 
        Implement LoRA: https://arxiv.org/pdf/2106.09685.pdf 
        Adding LoRA modules to nn.Conv1d, nn.Conv2d (should we also add to nn.Linear?)
         + retraining all biases (only a negligible number of parameters)
        """
        score.requires_grad_(False)
        for name, param in score.named_parameters():
            if "bias" in name and not "emb_layers" in name:
                param.requires_grad = True
        inject_trainable_lora_extended(score, **adpt_kwargs)
    elif impl == 'dif-fit':
        raise NotImplementedError
    else: 
        raise NotImplementedError

    if impl in ['lora', 'full', 'decoder'] and verbose:
        num_params = sum([p.numel() for p in score.parameters()])
        trainable_params = sum([p.numel() for p in score.parameters() if p.requires_grad])
        print(f'% of trainable params: {trainable_params/num_params*100}')
