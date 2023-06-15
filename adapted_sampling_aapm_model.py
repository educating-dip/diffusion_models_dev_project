
import torch 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import yaml 
from src import UNetModel, DDPM, BaseSampler, wrapper_ddim, SimpleTrafo, adapted_ddim_sde_predictor, tv_loss, _adapt, _score_model_adpt, PSNR
from src import get_standard_score, get_standard_configs, get_standard_adapted_sampler
import argparse
import functools 

parser = argparse.ArgumentParser(description='adapted sampling')
parser.add_argument('--dataset', default='walnut', help='test-dataset', choices=['walnut', 'lodopab', 'ellipses', 'mayo'])
parser.add_argument('--model_learned_on', default='aapm', help='model-checkpoint to load', choices=['lodopab', 'ellipses', 'aapm'])
parser.add_argument('--method',  default='dds', choices=['naive', 'dps', 'dds'])
parser.add_argument('--add_corrector_step', action='store_true')

parser.add_argument('--num_steps', default=50)
parser.add_argument('--penalty', default=1, help='reg. penalty used for ``naive'' and ``dps'' only.')
parser.add_argument('--tv_penalty', default=1e-5, help='reg. used for ``adapatation''.')
parser.add_argument('--eta', default=0.15, help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--sde', default='ddpm', choices=['vpsde', 'vesde', 'ddpm'])
parser.add_argument('--adaptation', default='lora', choices=['decoder', 'full', 'vdkl', 'lora'])
parser.add_argument('--num_optim_step', default=5, help='num. of optimization steps taken per sampl. step')
parser.add_argument('--adapt_freq', default=1, help='freq. of adaptation step in sampl.')
parser.add_argument('--lora_include_blocks', default=['input_blocks','middle_block','output_blocks','out'], nargs='+', help='lora kwargs impl. of arch. blocks included')
parser.add_argument('--lora_rank', default=4, help='lora kwargs impl. of rank')
parser.add_argument('--add_cg', action='store_true', help="do DDS steps after adaptation.")
parser.add_argument('--cg_iter', default=1, help="Number of CG steps for DDS update.")
parser.add_argument('--gamma', default=0.6, help='reg. used for ``dds''.')




def main(args):

    config, dataconfig = get_standard_configs(args)
    config.device = "cuda"


    model = get_standard_score(model_type="dds_unet", config=config, sde=None, use_ema=False, load_model=True)
    model.to(config.device)
    model.eval()


    sde = DDPM(beta_min=config.diffusion.beta_start, beta_max=config.diffusion.beta_end, num_steps=config.diffusion.num_diffusion_timesteps)


    x = torch.from_numpy(np.load("/localdata/AlexanderDenker/score_based_baseline/AAPM/walnut/walnut.npy")).unsqueeze(0).unsqueeze(0).to(config.device)

    ray_trafo = SimpleTrafo((256, 256), num_angles=60)

    y = ray_trafo.trafo(x)
    y_noise = y + 0.05*torch.mean(torch.abs(y))*torch.randn_like(y)

    x_fbp = ray_trafo.fbp(y_noise)


    sampler = get_standard_adapted_sampler(
			args=args,
			config=config,
			score=model,
			sde=sde,
			ray_trafo=ray_trafo,
			observation=y_noise,
			device=config.device
			)


    x_mean = sampler.sample(logging=False)

    print("PSNR: ", PSNR(x_mean[0, 0].cpu().numpy(), x[0, 0].cpu().numpy()))

    #x_mean = torch.clamp(x_mean, 0)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)


    ax1.imshow(x[0,0,:,:].cpu())
    ax2.imshow(x_fbp[0,0,:,:].cpu())
    ax3.imshow(x_mean[0,0,:,:].cpu())



    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)