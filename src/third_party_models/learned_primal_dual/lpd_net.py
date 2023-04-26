"""
Implementation of a Learned Primal Dual as a conditional score model
Adapted from DiVaL: https://jleuschn.github.io/docs.dival/_modules/dival/reconstructors/networks/iterative.html#IterativeBlock
"""

import torch
import torch.nn as nn

class IterativeBlock(nn.Module):
    def __init__(self, n_in=3, n_out=1, n_memory=5, n_layer=3, internal_ch=32,
                 kernel_size=3, batch_norm=True, prelu=False, lrelu_coeff=0.2):
        super(IterativeBlock, self).__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        modules = []
        if batch_norm:
            modules.append(nn.BatchNorm2d(n_in + n_memory))
        for i in range(n_layer-1):
            input_ch = (n_in + n_memory) if i == 0 else internal_ch
            modules.append(nn.Conv2d(input_ch, internal_ch,
                                     kernel_size=kernel_size, padding=padding))
            if batch_norm:
                modules.append(nn.BatchNorm2d(internal_ch))
            if prelu:
                modules.append(nn.PReLU(internal_ch, init=0.0))
            else:
                modules.append(nn.LeakyReLU(lrelu_coeff, inplace=True))
        modules.append(nn.Conv2d(internal_ch, n_out + n_memory,
                                 kernel_size=kernel_size, padding=padding))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        upd = self.block(x)
        return upd

class PrimalDualNet(nn.Module):
    def __init__(self, n_iter, op, op_adj, sde, n_primal=5, n_dual=5,
                 n_layer=4, internal_ch=32, kernel_size=3,
                 batch_norm=True, prelu=False, lrelu_coeff=0.2):
        super(PrimalDualNet, self).__init__()
        self.n_iter = n_iter
        self.op = op
        self.op_adj = op_adj
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.sde = sde 

        self.primal_blocks = nn.ModuleList()
        self.dual_blocks = nn.ModuleList()
        for it in range(n_iter):
            self.dual_blocks.append(IterativeBlock(
                n_in=3, n_out=1, n_memory=self.n_dual-1, n_layer=n_layer,
                internal_ch=internal_ch, kernel_size=kernel_size,
                batch_norm=batch_norm, prelu=prelu, lrelu_coeff=lrelu_coeff))
            self.primal_blocks.append(IterativeBlock(
                n_in=2, n_out=1, n_memory=self.n_primal-1, n_layer=n_layer,
                internal_ch=internal_ch, kernel_size=kernel_size,
                batch_norm=batch_norm, prelu=prelu, lrelu_coeff=lrelu_coeff))

    def lpd_forward(self, x, y):
        primal_cur = torch.repeat_interleave(x, repeats=self.n_primal, dim=1)

        dual_cur = torch.zeros(y.shape[0], self.n_dual,
                               *self.op_adj.operator.domain.shape,
                               device=y.device)
        for i in range(self.n_iter):
            primal_evalop = self.op(primal_cur[:, 1:2, ...])
            dual_update = torch.cat([dual_cur, primal_evalop, y], dim=1)
            dual_update = self.dual_blocks[i](dual_update)
            dual_cur = dual_cur + dual_update
            # NB: currently only linear op supported
            #     for non-linear op: [d/dx self.op(primal_cur[0:1, ...])]*
            dual_evalop = self.op_adj(dual_cur[:, 0:1, ...])
            primal_update = torch.cat([primal_cur, dual_evalop], dim=1)
            primal_update = self.primal_blocks[i](primal_update)
            primal_cur = primal_cur + primal_update
        x = primal_cur[:, 0:1, ...]

        return x

    def forward(self, x, y, t):
        h = self.lpd_forward(x, y)
        h_mean, std = self.sde.marginal_prob(h, t)
        out = (h_mean - x)/std[:, None, None, None]**2
        return out 