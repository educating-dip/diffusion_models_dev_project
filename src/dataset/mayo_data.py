'''
Taken from https://github.com/educating-dip/svd_dip/blob/2c3c5eafb782e132b22b7fbf1e9ba1ca265bc39a/src/dataset/standard.py#L790
'''
from typing import Tuple

import os 
import numpy as np 
import torch

from pydicom.filereader import dcmread
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

class MayoDataset(Dataset):
    def __init__(self, 
        part: str, 
        base_path: str, 
        im_shape: Tuple[int, int], 
        seed: int = 1, 
        ) -> None:
        
        assert part in ['N', 'L', 'C']
        
        self.part = part
        self.im_shape = im_shape
        self.base_path = base_path
        self.full_subjects = {
            'N': ['N001', 'N003', 'N005', 'N012', 'N017', 'N021', 'N024', 'N025', 'N029', 'N030'],
            'L': ['L004', 'L006', 'L012', 'L014', 'L019', 'L024', 'L027', 'L030', 'L033', 'L035'], 
            'C': ['C001', 'C002', 'C004', 'C009', 'C012', 'C016', 'C019', 'C021', 'C023', 'C027']
        }
        self.subjetcs = self.full_subjects[self.part]
        self.seed = seed

    def __len__(self):
        return len(self.subjetcs)

    def __getitem__(self, idx: int):
        subject = self.subjetcs[idx]
        dirs = os.listdir(
            os.path.join(self.base_path, subject))
        full_dose_image_dirs = [d for d in os.listdir(os.path.join(self.base_path, subject, dirs[0])) if 'Full Dose' in d]
        assert len(full_dose_image_dirs) == 1
        path = os.path.join(self.base_path, subject, dirs[0], full_dose_image_dirs[0])
        dcm_files = os.listdir(path)
        dcm_files.sort(key=lambda f: float(dcmread(os.path.join(path, f), specific_tags=['SliceLocation'])['SliceLocation'].value))
        sample_slice = (len(dcm_files) - 1) // 2 # middle slice 
        dcm_dataset = dcmread(os.path.join(path, dcm_files[sample_slice]))

        rng = np.random.default_rng(self.seed)
        # like lodopab preprocessing
        array = dcm_dataset.pixel_array[75:-75,75:-75].astype(np.float32).T
        array *= dcm_dataset.RescaleSlope
        array += dcm_dataset.RescaleIntercept
        array += rng.uniform(0., 1., size=array.shape)
        # convert values
        MU_WATER = 20
        MU_AIR = 0.02
        MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
        array *= (MU_WATER - MU_AIR) / 1000
        array += MU_WATER
        array /= MU_MAX
        np.clip(array, 0., 1., out=array)
        array = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
        
        x = interpolate(array, size=self.im_shape, mode='bilinear').squeeze()

        return x.unsqueeze(0)
