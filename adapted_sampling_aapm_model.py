
import torch 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import yaml 
from src import UNetModel, DDPM, BaseSampler, wrapper_ddim, SimpleTrafo, adapted_ddim_sde_predictor, tv_loss, _adapt, _score_model_adpt, PSNR
from src import get_standard_score, get_standard_configs, get_standard_adapted_sampler, AAPMDataset, get_standard_sampler
import argparse
import functools 
import time 
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser(description='adapted sampling')
parser.add_argument('--dataset', default='walnut', help='test-dataset', choices=['walnut', 'lodopab', 'ellipses', 'mayo'])
parser.add_argument('--model_learned_on', default='aapm', help='model-checkpoint to load', choices=['aapm'])
parser.add_argument('--method',  default='dds', choices=['naive', 'dps', 'dds'])
parser.add_argument('--add_corrector_step', action='store_true')
parser.add_argument('--load_path', default="/localdata/AlexanderDenker/score_based_baseline/AAPM/vp/AAPM256_1M.pt")
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
parser.add_argument('--pct_chain_elapsed', default=0.)

parser.add_argument("--sampling_type", default="cond", choices=["adapt", "cond"])
parser.add_argument("--relative_noise", default=0.01)

parser.add_argument('--data', default="aapm", choices=["aapm", "walnut"])

def main(args):

    config, dataconfig = get_standard_configs(args)
    config.device = "cuda"

    config.sampling.eps = 1e-4
    config.sampling.travel_length = 1
    config.sampling.travel_repeat = 1

    model = get_standard_score(model_type="dds_unet", config=config, sde=None, use_ema=False, load_model=True)
    model.to(config.device)
    model.eval()


    save_root = os.path.join("/localdata/AlexanderDenker/score_model_results/", args.data)

    if args.sampling_type == "cond":
        save_root = os.path.join(save_root, args.sampling_type, args.method, "num_steps=" + str(args.num_steps), "gamma=" + str(args.gamma), "cg_iter=" 
        + str(args.cg_iter), f'{time.strftime("%d-%m-%Y-%H-%M-%S")}')
    elif args.sampling_type == "adapt":
        save_root = os.path.join(save_root, args.sampling_type, args.adaptation, "num_steps=" + str(args.num_steps), 
                        "gamma=" + str(args.gamma),  "optim_steps=" + str(args.num_optim_step), f'{time.strftime("%d-%m-%Y-%H-%M-%S")}')
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    
    with open(save_root / 'settings.yaml', 'w') as file:
        yaml.dump(args, file)

    sde = DDPM(beta_min=config.diffusion.beta_start, beta_max=config.diffusion.beta_end, num_steps=config.diffusion.num_diffusion_timesteps)

    if args.data == "aapm":
        dataset = AAPMDataset(part="val")
    elif args.data == "walnut":
        x = torch.from_numpy(np.load("/localdata/AlexanderDenker/score_based_baseline/AAPM/walnut/walnut.npy")).unsqueeze(0)
        dataset = [x]

    #x = torch.from_numpy(np.load("/localdata/AlexanderDenker/score_based_baseline/AAPM/walnut/walnut.npy")).unsqueeze(0).unsqueeze(0).to(config.device)
    print(len(dataset))

    ray_trafo = SimpleTrafo((256, 256), num_angles=60)

    psnr_list = []
    time_list = []
    for idx in range(len(dataset)):
        x = dataset[idx]
        x = x.unsqueeze(0).to(config.device)

        y = ray_trafo.trafo(x)
        y_noise = y + args.relative_noise*torch.mean(torch.abs(y))*torch.randn_like(y)

        x_fbp = ray_trafo.fbp(y_noise)

        if args.sampling_type == "cond":
            sampler = get_standard_sampler(
                args=args,
                config=config,
                score=model,
                sde=sde,
                ray_trafo=ray_trafo,
                filtbackproj=x_fbp,
                observation=y_noise,
                device=config.device
                )
        elif args.sampling_type == "adapt":
            sampler = get_standard_adapted_sampler(
                    args=args,
                    config=config,
                    score=model,
                    sde=sde,
                    ray_trafo=ray_trafo,
                    observation=y_noise,
                    device=config.device
                    )
            
        time_start = time.time() 
        x_mean = sampler.sample(logging=False)
        time_end = time.time() 
        duration = float(time_end - time_start)
        time_list.append(duration)

        x_mean = torch.clamp(x_mean, 0)
        psnr = PSNR(x_mean[0, 0].cpu().numpy(), x[0, 0].cpu().numpy())
        print("PSNR: ", psnr, " Time: ", duration, " s")
        psnr_list.append(psnr)
        im = Image.fromarray(x_mean.cpu().squeeze().numpy()*255.).convert("L")
        im.save(str(save_root / f'recon_{idx}.png'))

    res = {}
    res["PSNR"] = float(np.mean(psnr_list))
    res["Mean Sampling Time"] = float(np.mean(time_list))
    with open(save_root / 'result.yaml', 'w') as file:
        yaml.dump(res, file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)