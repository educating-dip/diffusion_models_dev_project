import sys, os
sys.path.insert(0, './bart-0.6.00/python')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOOLBOX_PATH"]    = './bart-0.6.00'
sys.path.append('./bart-0.6.00/python')


import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
from utils import normalize, crop_center, normalize_mvue, unnormalize_mvue, get_mvue, normalize_complex, clear
import sigpy as sp
import sigpy.mri as mr
from bart import bart

"""
This script prepares 320 x 320 sized fastmri MVUE estimate, acquired from multi-coil data
"""

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=0) / np.sqrt(
        np.sum(np.square(np.abs(s_maps)), axis=0))

# Save
types = ['train', 'val']
save_path_type = ['slice', 'mps']
save_path = Path('/media/harry/tomo/fastmri')
sz = 320
# # #
for t in types:
    print(f'Processing {t}')
    root_t = Path(f'/media/harry/tomo/fastmri/knee_multicoil_h5_{t}')
    files = list(root_t.glob('*.h5'))
    for f in tqdm(files):
        vol_name = str(f).split('/')[-1][:-3]
        save_path_tv = save_path / f'knee_mvue_{sz}_{t}' / vol_name
        for spt in save_path_type:
            save_path_tvt = save_path_tv / spt
            save_path_tvt.mkdir(parents=True, exist_ok=True)
        with h5py.File(f, 'r') as hf:
            target = hf['kspace']
            depth = target.shape[0]
            for idx in range(5, depth):
                full_kspace = target[idx, ...]
                full_img = sp.ifft(full_kspace, axes=(-1, -2))
                comp_img = crop_center(full_img, sz, sz)
                comp_kspace = sp.fft(full_img, axes=(-1, -2))

                # check if mps from bart satisfy the normalization constraints
                s_mps = bart(1, 'ecalib -m1 -W -c0', comp_kspace.transpose((1, 2, 0))[None, ...]).transpose(
                    (3, 1, 2, 0)).squeeze()
                mvue = get_mvue(comp_kspace, s_mps)
                mvue = torch.from_numpy(mvue)

                mvue = unnormalize_mvue(mvue, mvue)
                np.save(str(save_path_tv / 'slice' / f'{idx:03d}.npy'), mvue.numpy())
                np.save(str(save_path_tv / 'mps' / f'{idx:03d}.npy'), s_mps)

                # n2_mvue = normalize_complex(mvue)

                # debug
                # sum = np.sqrt(np.sum(np.square(np.abs(s_mps)), axis=0))

                #
                # plt.imshow(np.abs(mvue).squeeze(), cmap='gray')
                # plt.show()
                # import ipdb;
                # ipdb.set_trace()



