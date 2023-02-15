import torch 
import numpy as np 
from torch import Tensor

def simulate(x: Tensor, ray_trafo: BaseRayTrafo, white_noise_rel_stddev: float,
        rng: Optional[np.random.Generator] = None):

    observation = ray_trafo(x)
    if rng is None:
        rng = np.random.default_rng()
    noise = torch.from_numpy(rng.normal(
            scale=white_noise_rel_stddev * torch.mean(torch.abs(observation)).item(),
            size=observation.shape)).to(
                    dtype=observation.dtype, device=observation.device)

    noisy_observation = observation + noise
    return noisy_observation, white_noise_rel_stddev * torch.mean(torch.abs(observation)).item()