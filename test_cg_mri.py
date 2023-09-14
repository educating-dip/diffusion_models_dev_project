import argparse
import yaml 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import islice
from PIL import Image
from omegaconf import OmegaConf
import numpy as np 
import h5py 
import os 
from torchvision.transforms import Resize
from PIL import Image
import time 
from pathlib import Path

from src import (cg, PSNR, SSIM, SingleCoilMRI, get_standard_sde, get_standard_score, get_standard_sampler) 

parser = argparse.ArgumentParser(description='conditional sampling for single-coil MRI')
parser.add_argument('--method',  default='dds', choices=['naive', 'dps', 'dds'])
parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--num_steps', default=200  )
parser.add_argument('--penalty', default=1, help='reg. penalty used for ``naive'' and ``dps'' only.')
parser.add_argument('--gamma', default=2.0, help='reg. used for ``dds''.')
parser.add_argument('--eta', default=0.85, help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--pct_chain_elapsed', default=0,  help='``pct_chain_elapsed'' actives init of chain')
parser.add_argument('--sde', default='ddpm', choices=['vpsde', 'vesde', 'ddpm'])
parser.add_argument('--cg_iter', default=5)

args = parser.parse_args()

def get_mask(img, size, batch_size, type='uniform1d', acc_factor=8, center_fraction=0.04, fix=False):
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
    if type == 'gaussian2d':
        mask = torch.zeros_like(img)
        cov_factor = size * (1.5 / 128)
        mean = [size // 2, size // 2]
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        if fix:
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
        else:
            for i in range(batch_size):
                # sample different masks for batch
                samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
    elif type == 'uniformrandom2d':
        mask = torch.zeros_like(img)
        if fix:
            mask_vec = torch.zeros([1, size * size])
            samples = np.random.choice(size * size, int(Nsamp))
            mask_vec[:, samples] = 1
            mask_b = mask_vec.view(size, size)
            mask[:, ...] = mask_b
        else:
            for i in range(batch_size):
                # sample different masks for batch
                mask_vec = torch.zeros([1, size * size])
                samples = np.random.choice(size * size, int(Nsamp))
                mask_vec[:, samples] = 1
                mask_b = mask_vec.view(size, size)
                mask[i, ...] = mask_b
    elif type == 'gaussian1d':
        mask = torch.zeros_like(img)
        mean = size // 2
        std = size * (15.0 / 128)
        Nsamp_center = int(size * center_fraction)
        if fix:
            samples = np.random.normal(
                loc=mean, scale=std, size=int(Nsamp * 1.2))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                samples = np.random.normal(
                    loc=mean, scale=std, size=int(Nsamp*1.2))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, :, int_samples] = 1
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from + Nsamp_center] = 1
    elif type == 'uniform1d':
        mask = torch.zeros_like(img)
        if fix:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[..., samples] = 1
            # ACS region
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                Nsamp_center = int(size * center_fraction)
                samples = np.random.choice(size, int(Nsamp - Nsamp_center))
                mask[i, :, :, samples] = 1
                # ACS region
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from+Nsamp_center] = 1
    else:
        NotImplementedError(f'Mask type {type} is currently not supported.')

    return mask


def real_to_nchw_comp(x):
    """
    [1, 2, 320, 320] real --> [1, 1, 320, 320] comp
    """
    if len(x.shape) == 4:
        x = x[:, 0:1, :, :] + x[:, 1:2, :, :] * 1j
    elif len(x.shape) == 3:
        x = x[0:1, :, :] + x[1:2, :, :] * 1j
    return x


device = "cuda"

save_root = "/home/adenker/projects/diffusion_models_dev_project/single_coil"
save_root = os.path.join(save_root, 'knee_knee', "dds", "num_steps=" + str(args.num_steps), "eta=" + str(args.eta), "gamma=" + str(args.gamma))
save_root = Path(os.path.join(save_root, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}'))

results = {}

print("save to: ", save_root)
save_root.mkdir(parents=True, exist_ok=True)

#mask = create_mask(shape=(256,256), center_fraction=0.08, acceleration=8) 
#mask = get_mask(torch.zeros([1, 1, 256, 256]), 256, 
#                            1, type="uniform1d",
#                            acc_factor=4, center_fraction=0.08)
#torch.save(mask, "uniform1d_4x.pt")
mask = torch.load("uniform1d_4x.pt")
mask = mask.to(device)
ray_trafo = SingleCoilMRI(mask)

img_file = "/localdata/AlexanderDenker/MRI/Dataset/Brain/label/file_brain_AXFLAIR_200_6002447/007.png" #"/localdata/AlexanderDenker/MRI/Dataset/Knee/label/file1000041/025.png"

x = np.array(Image.open(img_file).convert("L"))/255.

x_torch = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
x_torch = x_torch.type(torch.complex64)    

ground_truth = x_torch.to(device)

observation = ray_trafo.trafo(ground_truth)

noise = torch.randn_like(observation)
noise_level = torch.sum(torch.abs(observation))/torch.sum(mask)
observation = observation + 0.01*noise_level.item()*noise



filtbackproj = ray_trafo.fbp(observation)


#def op(x):
#    return ray_trafo.trafo_adjoint(ray_trafo(x)) 
gamma = 0.1
rhs = filtbackproj + gamma*ray_trafo.trafo_adjoint(observation)

def op(x):
    return x + gamma*ray_trafo.trafo_adjoint(ray_trafo(x))

x_init = torch.zeros_like(ground_truth)
reco = cg(op, x=x_init, rhs =rhs, n_iter=5)

print(reco.shape)

psnr = PSNR(torch.abs(filtbackproj[0, 0]).cpu().numpy(), torch.abs(ground_truth[0, 0]).cpu().numpy())
ssim = SSIM(torch.abs(filtbackproj[0, 0]).cpu().numpy(), torch.abs(ground_truth[0, 0]).cpu().numpy())	
print('FBP PSNR:', psnr)
print('FBP SSIM:', ssim)

results["FBP"] = {}
results["FBP"]["psnr"] = float(psnr)
results["FBP"]["ssim"] = float(ssim)

psnr = PSNR(torch.abs(reco[0, 0]).cpu().numpy(), torch.abs(ground_truth[0, 0]).cpu().numpy())
ssim = SSIM(torch.abs(reco[0, 0]).cpu().numpy(), torch.abs(ground_truth[0, 0]).cpu().numpy())	
print('CG PSNR:', psnr)
print('CG SSIM:', ssim)

results["CG"] = {}
results["CG"]["psnr"] = float(psnr)
results["CG"]["ssim"] = float(ssim)

### using score model DDS

base_path = "/home/adenker/projects/diffusion_models_dev_project/fastmri_configs/ddpm/fastmri_knee_320_complex.yml"
load_path = "/localdata/AlexanderDenker/score_based_baseline/FastMRI/fastmri_knee_320_complex_1m.pt"
with open(base_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.UnsafeLoader)
    config = OmegaConf.create(config)

model_type = "dds_unet" 
sde = get_standard_sde(config=config)
score = get_standard_score(config=config, sde=sde, 
        use_ema=False, model_type="dds_unet", load_model=False)
score.load_state_dict(torch.load(load_path))
print(f'Model ckpt loaded from {load_path}')
score.convert_to_fp32()
score.dtype = torch.float32

score = score.to(config.device)
score.to("cuda")
score.eval()

logg_kwargs = {'log_dir': ".", 'num_img_in_log': 5,
		'sample_num':0, 'ground_truth': ground_truth, 'filtbackproj': filtbackproj}
sampler = get_standard_sampler(
    args=args,
    config=config,
    score=score,
    sde=sde,
    ray_trafo=ray_trafo,
    filtbackproj=filtbackproj,
    observation=observation,
    device=config.device
    )

recon_dds = sampler.sample(logg_kwargs=logg_kwargs, logging=False)
print("FINISHED SAMPLING")
recon_dds = real_to_nchw_comp(recon_dds)
recon_dds = np.abs(recon_dds.detach().cpu().numpy())
print(recon_dds.shape)

psnr = PSNR(recon_dds[0, 0], torch.abs(ground_truth[0, 0]).cpu().numpy())
ssim = SSIM(recon_dds[0, 0], torch.abs(ground_truth[0, 0]).cpu().numpy())	
print('DDS PSNR:', psnr)
print('DDS SSIM:', ssim)

results["DDS"] = {}
results["DDS"]["psnr"] = float(psnr)
results["DDS"]["ssim"] = float(ssim)


with open(save_root / 'report.yaml', 'w') as file:
    yaml.dump(results, file)

im = Image.fromarray(torch.abs(ground_truth[0,0,:,:].detach().cpu()).numpy()*255.).convert("L")
im.save(str(save_root / f'gt.png'))

im = Image.fromarray(torch.abs(filtbackproj[0,0,:,:].detach().cpu()).numpy()*255.).convert("L")
im.save(str(save_root / f'fbp.png'))

im = Image.fromarray(torch.abs(reco[0,0,:,:].detach().cpu()).numpy()*255.).convert("L")
im.save(str(save_root / f'cg.png'))

im = Image.fromarray(recon_dds[0,0,:,:]*255.).convert("L")
im.save(str(save_root / f'dds.png'))

im = Image.fromarray(mask[0,0,:,:].detach().cpu().numpy()*255.).convert("L")
im.save(str(save_root / f'mask.png'))

_, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
ax1.imshow(torch.abs(ground_truth[0,0,:,:].detach().cpu()), cmap="gray")
ax1.axis('off')
ax1.set_title('Ground truth')
ax2.imshow(torch.abs(filtbackproj[0,0,:,:].detach().cpu()), cmap="gray")
ax2.axis('off')
ax2.set_title('FBP')
ax3.imshow(torch.abs(reco[0,0,:,:].detach().cpu()), cmap="gray")
ax3.axis('off')
ax3.set_title('CG')
ax4.imshow(recon_dds[0,0,:,:], cmap="gray")
ax4.axis('off')
ax4.set_title('DDS')
ax5.imshow(mask[0,0,:,:].detach().cpu())
ax5.axis('off')
ax5.set_title('mask')

plt.show()	
