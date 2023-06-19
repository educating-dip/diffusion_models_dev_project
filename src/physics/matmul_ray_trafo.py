"""
From https://github.com/educating-dip/bayes_dip/blob/main/bayes_dip/data/trafo/matmul_ray_trafo.py.
Provides :class:`MatmulRayTrafo`.
"""

from typing import Union, Optional, Callable, Tuple, Any
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any
import torch
from torch import Tensor
import numpy as np
import scipy.sparse

from .base_ray_trafo import BaseRayTrafo

def _convert_to_torch_matrix(matrix):
    matrix = matrix.astype('float32')
    if scipy.sparse.isspmatrix(matrix):
        matrix = matrix.tocoo()
        indices = torch.stack([
                torch.from_numpy(matrix.row),
                torch.from_numpy(matrix.col)])
        values = torch.from_numpy(matrix.data)
        shape = matrix.shape

        matrix = torch.sparse_coo_tensor(indices, values, shape)
        matrix = matrix.coalesce()
    else:
        matrix = torch.from_numpy(matrix)
    return matrix

class ReSize: 
    def __init__(self, shape: Tuple[int, int], mode: str = 'nearest-exact'):
        self.shape = shape
        self.mode = mode

    def __call__(self, x, shape: Optional[Tuple[int, int]]=None) -> Tensor:
        return torch.nn.functional.interpolate(x, size=self.shape if shape is None else shape, mode=self.mode)

class MatmulRayTrafo(BaseRayTrafo):
    """
    Ray transform implemented by (sparse) matrix multiplication.

    Adjoint computations are accurate in this implementation (which is not
    always the case when using back-projection for the adjoint).
    """

    def __init__(self,
            im_shape: Union[Tuple[int, int], Tuple[int, int, int]],
            obs_shape: Union[Tuple[int, int], Tuple[int, int, int]],
            matrix: Union[Tensor, scipy.sparse.spmatrix, np.ndarray],
            fbp_fun: Optional[Callable[[Tensor], Tensor]] = None,
            angles: Optional[ArrayLike] = None, 
            new_shape: Optional[Tuple[int, int]] = None
            ):
        """
        Parameters
        ----------
        im_shape, obs_shape
            See :meth:`BaseRayTrafo.__init__`.
        matrix : tensor, scipy sparse matrix or numpy array
            Matrix representation of the ray transform.
            Shape: ``(np.prod(obs_shape), np.prod(im_shape))``.
        fbp_fun : callable, optional
            Function applying a filtered back-projection, used for providing
            :meth:`fbp`.
        angles : array-like, optional
            Angles of the ray transform, only used for providing the
            :attr:`angles` property; not used for any computations.
        """
        super().__init__(im_shape=im_shape, obs_shape=obs_shape)

        if not isinstance(matrix, Tensor):
            # convert from numpy or scipy sparse matrix
            matrix = _convert_to_torch_matrix(matrix)

        # register for automatic moving to device (access: self.matrix)
        self.register_buffer('matrix', matrix, persistent=False)

        if new_shape is not None:
            # assert isinstance(new_size, Tuple)
            self.resize = ReSize(shape=new_shape)

        if matrix.is_sparse:
            # cannot call .T on sparse torch tensor, so create new tensor and
            # register it for automatic moving to device (access: self.matrix_t)
            indices_t = matrix.indices()[[1, 0], :]  # 2 x ??
            values = matrix.values()
            shape_t = matrix.shape[::-1]
            matrix_t = torch.sparse_coo_tensor(indices_t, values, shape_t)
            matrix_t = matrix_t.coalesce()

            self.register_buffer('matrix_t', matrix_t, persistent=False)

        self.fbp_fun = fbp_fun
        self._angles = angles

    @property
    def angles(self) -> ArrayLike:
        """array-like : The angles (in radian)."""
        if self._angles is not None:
            return self._angles
        raise ValueError('`angles` was not set for `MatmulRayTrafo`')

    def trafo_flat(self, x: Tensor) -> Tensor:
        if self.resize is not None:
           x = self.resize(x.view((x.shape[-1], 1, *self.resize.shape)), shape=self.im_shape)
           x = x.view(np.prod(self.im_shape), x.shape[0])
        if self.matrix.is_sparse:
            observation = torch.sparse.mm(self.matrix, x)
        else:
            observation = torch.matmul(self.matrix, x)
        
        return observation

    def trafo_adjoint_flat(self, observation: Tensor) -> Tensor:
        if self.matrix.is_sparse:
            x = torch.sparse.mm(self.matrix_t, observation)
        else:
            x = torch.matmul(self.matrix.T, observation)
        if self.resize is not None:
            x = self.resize(x.view((observation.shape[-1], 1, *self.im_shape)))
            x = x.view(np.prod(self.resize.shape), observation.shape[-1])
        return x

    def fbp(self, observation: Tensor) -> Tensor:
        fbp = self.fbp_fun(observation)
        if self.resize is not None:
            _fbp = fbp.view((*observation.shape[:2], *self.im_shape))
            fbp = self.resize(_fbp)
        return fbp

    trafo = BaseRayTrafo._trafo_via_trafo_flat
    trafo_adjoint = BaseRayTrafo._trafo_adjoint_via_trafo_adjoint_flat
