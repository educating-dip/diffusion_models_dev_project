import torch 
import torch.nn as nn
from torch import Tensor

from src.utils import SDE
from src.third_party_models import inject_trainable_lora_extended

import itertools

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
        """ 
        Implement LoRA: https://arxiv.org/pdf/2106.09685.pdf 

        Adding LoRA modules to nn.Conv1d, nn.Conv2d (should we also add to nn.Linear?)
         + retraining all biases (only a negligible number of parameters)
        """


        score.requires_grad_(False)

        for name, param in score.named_parameters():
            if "bias" in name and not "emb_layers" in name:
                param.requires_grad = True

        lora_params, train_names = inject_trainable_lora_extended(score) 

        new_num_params = sum([p.numel() for p in score.parameters()])
        trainable_params = sum([p.numel() for p in score.parameters() if p.requires_grad])

        #print("Percent of trainable params: ", trainable_params/new_num_params*100, " %")
        #optim = torch.optim.Adam(itertools.chain(*lora_params), lr=1e-4)
        #optim = torch.optim.Adam(score.parameters(), lr=3e-4)
        #return optim

    elif impl == 'dif-fit':
        pass
    elif impl == 'vdkl':
        pass 
    else: 
        raise NotImplementedError 