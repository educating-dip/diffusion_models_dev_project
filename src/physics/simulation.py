"""
Provides simulation by applying a ray transform and white noise.
From https://github.com/educating-dip/subspace_dip_learning/blob/main/subspace_dip/data/simulation.py
"""

import torch 
import numpy as np 
from torch import Tensor

from typing import Union, Sequence, Iterable, Optional, Any, Tuple, Iterator

def simulate(x: Tensor, ray_trafo, white_noise_rel_stddev: float, rng  = None, return_noise_level: bool = False):

    observation = ray_trafo(x)

    if rng is None:
        rng = np.random.default_rng()
    
    noise_level = white_noise_rel_stddev * torch.mean(torch.abs(observation)).item()
    noise = torch.from_numpy(rng.normal(
            scale=noise_level,
            size=observation.shape)).to(
                    dtype=observation.dtype, device=observation.device)
    
    noisy_observation = observation + noise

    return (noisy_observation, noise_level) if return_noise_level else noisy_observation

class SimulatedDataset(torch.utils.data.Dataset):

    def __init__(self,
            image_dataset: Union[Sequence[Tensor], Iterable[Tensor]],
            ray_trafo,
            white_noise_rel_stddev: float,
            use_fixed_seeds_starting_from: Optional[int] = 1,
            rng: Optional[np.random.Generator] = None,
            device: Optional[Any] = None):

        super().__init__()

        self.image_dataset = image_dataset
        self.ray_trafo = ray_trafo
        self.white_noise_rel_stddev = white_noise_rel_stddev
        if rng is not None:
            assert use_fixed_seeds_starting_from is None, (
                    'must not use fixed seeds when passing a custom rng')
        self.rng = rng
        self.use_fixed_seeds_starting_from = use_fixed_seeds_starting_from
        self.device = device

    def __len__(self) -> Union[int, float]:
        return len(self.image_dataset)

    def _generate_item(self, idx: int, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        if self.rng is None:
            seed = (self.use_fixed_seeds_starting_from + idx
                    if self.use_fixed_seeds_starting_from is not None else None)
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        x = x.to(device=self.device)
        noisy_observation = simulate(x[None],
                ray_trafo=self.ray_trafo,
                white_noise_rel_stddev=self.white_noise_rel_stddev,
                rng=rng)[0].to(device=self.device)
        filtbackproj = self.ray_trafo.fbp(noisy_observation[None])[0].to(
                device=self.device)

        return noisy_observation, x, filtbackproj

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        for idx, x in enumerate(self.image_dataset):
            yield self._generate_item(idx, x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self._generate_item(idx, self.image_dataset[idx])
