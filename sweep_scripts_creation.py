import os 
import argparse

parser = argparse.ArgumentParser(description='conditional sampling')
parser.add_argument('--dataset', default='walnut', help='test-dataset', choices=['walnut', 'lodopab', 'ellipses', 'mayo', 'aapm'])
parser.add_argument('--model_learned_on', default='lodopab', help='model-checkpoint to load', choices=['lodopab', 'ellipses', 'aapm'])
parser.add_argument('--method',  default='dds', choices=['dds'])
parser.add_argument('--num_steps', default=[50, 75])
parser.add_argument('--tv_penalty', default=[1e-5,1e-6,1e-7,0], help='reg. used for ``adapatation''.')
parser.add_argument('--eta', default=[0.75,0.85,0.95], help='reg. used for ``dds'' weighting stochastic and deterministic noise.')
parser.add_argument('--sde', default='ddpm', choices=['ddpm'])
parser.add_argument('--adaptation', default='lora', choices=['decoder', 'full', 'vdkl', 'lora'])
parser.add_argument('--num_optim_step', default=[1,2,5,10,15,20], help='num. of optimization steps taken per sampl. step')
parser.add_argument('--lr', default=[1e-3,1e-4], help='learning rate for adaptation')
parser.add_argument('--lora_rank', default=4, help='lora kwargs impl. of rank')
parser.add_argument('--add_cg', action='store_true', help='do DDS steps after adaptation.')
parser.add_argument('--cg_iter', default=[1,2,3,4,5,10], help='Number of CG steps for DDS update.')
parser.add_argument('--gamma', default=[0.01,0.1,1,10], help='reg. used for ``dds''.')
parser.add_argument('--load_path', help='path to ddpm model.')
parser.add_argument('--path_to_script', help='path to ddpm model.')
parser.add_argument('--cuda_idx', help='cuda device')

args = parser.parse_args()
f = open(f'{args.dataset}_{args.method}_sweep.txt', 'w+')
for method in args.method:
    for eta in args.eta: 
        for gamma in args.gamma:
            for num_steps in args.num_steps:
                for num_cg_steps in args.cg_iter:
                    for sde in args.sde:
                        line = f'CUDA_VISIBLE_DEVICES={args.cuda_idx} python {args.path_to_script}/run_conditional_sampling.py --model_learned_on={args.model_learned_on} --model dds_unet --dataset={args.dataset} --eta={eta} --gamma={gamma} --method={method} --ema --num_steps={num_steps} --cg_iter={num_cg_steps} --sde={sde} --load_path={args.load_path}/model.pt'
                        f.write(line)
                        f.write(' && ')
f.truncate(f.tell() - 4)

f = open(f'{args.dataset}_adaptation_from_{args.model_learned_on}_sweep.txt', 'w+')
for method in args.method:
    for sde in args.method:
        for eta in [0.85]: 
            for gamma in [1]:
                for lr in args.lr: 
                    for num_steps in args.num_steps:
                        for num_optim_step in args.num_optim_step:
                            for tvP in args.tv_penalty:
                                    line = f"CUDA_VISIBLE_DEVICES={args.cuda_idx} python {args.path_to_script}/run_adapted_sampling.py --model_learned_on={args.model_learned_on} --model dds_unet --dataset={args.dataset} --eta={eta} --method={method} --ema --num_steps={num_steps} --sde={sde} --load_path={args.load_path}/model.pt --tv_penalty {tvP} --adapt_freq 1 --num_optim_step={num_optim_step} --lora_rank={args.lora_rank} --lr={lr} --add_cg --gamma={gamma} --cg_iter=1"
                                    f.write(line)
                                    f.write(' && ')
f.truncate(f.tell() - 4)