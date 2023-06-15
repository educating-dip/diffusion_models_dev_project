import torch
from torch import Tensor
from ..physics import BaseRayTrafo

"""
Batched Conjugate Gradient in PyTorch for 
    solve (I + gamma A* A) x = gamma * A* y + xhat0

Adapted from ODL: https://github.com/odlgroup/odl/blob/master/odl/solvers/iterative/iterative.py 
"""
def cg(op: callable, x: Tensor, rhs: Tensor, n_iter: int = 5) -> Tensor:
    # solve (I + gamma A* A) x = rhs
    # starting with x 

    # batch x 1 x h x w
    r = op(x)
    r = rhs - r
    p = torch.clone(r)
    d = torch.zeros_like(x)

    # Only recalculate norm after update
    sqnorm_r_old = torch.linalg.norm(r.reshape(r.shape[0], -1), dim=1)**2 #r.norm() ** 2 

    for _ in range(n_iter):
        d = op(p)

        inner_p_d = (p * d).sum(dim=[1,2,3]) 

        alpha = sqnorm_r_old / inner_p_d
        x = x + alpha[:, None,None,None]*p # x = x + alpha*p
        r = r - alpha[:, None,None,None]*d # r = r - alpha*d

        sqnorm_r_new = torch.linalg.norm(r.reshape(r.shape[0], -1), dim=1)**2 

        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new

        p = r + beta[:, None,None,None]*p # p = r + b * p

    return x 