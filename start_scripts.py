
import os 
cuda_idx = 3
LOAD_PATH = '/localdata/AlexanderDenker/score_based_baseline/DiskEllipses/dds_unet/ddpm/version_01/model.pt' #'/localdata/AlexanderDenker/score_based_baseline/AAPM/vp/AAPM256_1M.pt'

# f = open('aapm_dps_sweep.txt', 'w+')
# for method in ["dps"]:
#     for eta in [0.15, 0.85]: 
#         for gamma in [0.1, 1.0, 10.]:
#             for num_steps in [50, 75]:
#                 for num_cg_steps in [1, 2, 5, 10]:
#                     for sde in ['ddmp']:
#                         line = f"CUDA_VISIBLE_DEVICES={cuda_idx} python run_conditional_sampling.py --model_learned_on=aapm --model dds_unet --dataset=aapm --eta={eta} --method={method} --ema --num_steps={num_steps} --sde={sde} --load_path={LOAD_PATH}"

#                         f.write(line)
#                         f.write('\n')

#f = open('ellipses_dds_sweep.txt', 'w+')
#for method in ['dds']:
#    for eta in [0.75, 0.85, 0.95]: 
#        for gamma in [0.01, 0.1, 1.0, 10.]:
#            for num_steps in [50, 75, 100]:
#                for num_cg_steps in [1, 2, 3, 5, 10]:
#                    for sde in ['ddpm']:
#                        line = f"CUDA_VISIBLE_DEVICES={cuda_idx} python run_conditional_sampling.py --model_learned_on=ellipses --model dds_unet --dataset=walnut --eta={eta} --gamma={gamma} --method={method} --ema --version=1 --num_steps={num_steps} --cg_iter={num_cg_steps} --sde={sde} --load_path={LOAD_PATH}"#                        
#                        f.write(line)
#                        f.write(' && ')


f = open('ellipses_lora_sweep.txt', 'w+')
for method in ['dds']:
    for eta in [0.85]: 
        for gamma in [0.1]:
            for num_steps in [50]:
                for num_optim_step in [20, 15, 10, 5]: #[20, 15,10,5]:
                    for tv_penalty in [5e-6, 1e-6, 1e-7, 0]:
                        for sde in ['ddpm']:
                            line = f"CUDA_VISIBLE_DEVICES={cuda_idx} python run_adapted_sampling.py --model_learned_on=ellipses --model dds_unet --dataset=walnut --eta={eta} --gamma={gamma} --method={method} --ema --version=1 --num_steps={num_steps} --cg_iter=1 --sde={sde} --load_path={LOAD_PATH} --add_cg --num_optim_step={num_optim_step} --dc_type=cg --lr=1e-3 --lora_rank=4 --tv_penalty={tv_penalty}"
                            
                            f.write(line)
                            f.write(' && ')


"""
f = open('aapm_dds_sweep.txt', 'w+')
for method in ['dds']:
    for eta in [0.15, 0.85]: 
        for gamma in [0.1, 1.0, 10.]:
            for num_steps in [50, 75]:
                for num_cg_steps in [1, 2, 5, 10]:
                    for sde in ['ddpm']:
                        line = f"CUDA_VISIBLE_DEVICES={cuda_idx} python /home/jleuschn/riccardo/diffusion_models_dev_project/run_conditional_sampling.py --model_learned_on=aapm --model dds_unet --dataset=aapm --eta={eta} --gamma={gamma} --method={method} --ema --num_steps={num_steps} --cg_iter={num_cg_steps} --sde={sde} --load_path={LOAD_PATH}"
                        
                        f.write(line)
                        f.write(' && ')

f = open('walnut_dds_sweep.txt', 'w+')
for method in ['dds']:
    for eta in [0.15, 0.85]: 
        for gamma in [0.1, 1.0, 10.]:
            for num_steps in [50, 75]:
                for num_cg_steps in [1, 2, 5, 10]:
                    for sde in ['ddpm']:
                        line = f"CUDA_VISIBLE_DEVICES={cuda_idx} python /home/jleuschn/riccardo/diffusion_models_dev_project/run_conditional_sampling.py --model_learned_on=aapm --model dds_unet --dataset=walnut --eta={eta} --gamma={gamma} --method={method} --ema --num_steps={num_steps} --cg_iter={num_cg_steps} --sde={sde} --load_path={LOAD_PATH} "
                        
                        f.write(line)
                        f.write(' && ')

num_cg_steps = 1
lora_rank = 4
f = open('aapm_lora_sweep.txt', 'w+')
for method in ['dds']:
    for sde in ['ddpm']:
        for eta in [0.85]: 
            for gamma in [1]:
                for lr in [1e-3, 1e-4]: 
                    for num_steps in [50]:
                        for num_optim_step in [1, 3, 5, 10]:
                            for tvP in [1e-5, 1e-6, 1e-7]:
                                    line = f"CUDA_VISIBLE_DEVICES={cuda_idx} python /home/jleuschn/riccardo/diffusion_models_dev_project/run_adapted_sampling.py --model_learned_on=aapm --model dds_unet --dataset=aapm --eta={eta} --gamma={gamma} --method={method} --ema --num_steps={num_steps} --cg_iter={num_cg_steps} --sde={sde} --load_path={LOAD_PATH} --tv_penalty {tvP} --adapt_freq 1 --num_optim_step {num_optim_step}  --lora_rank={lora_rank} --lr={lr} --add_cg --gamma={gamma} --cg_iter={num_cg_steps} "

                                    f.write(line)
                                    f.write(' && ')

num_cg_steps = 1
lora_rank = 4
f = open('walnut_lora_sweep.txt', 'w+')
for method in ['dds']:
    for sde in ['ddpm']:
        for eta in [0.85]: 
            for gamma in [1.0]:
                for lr in [1e-3, 1e-4]: 
                    for num_steps in [50]:
                        for num_optim_step in [1, 3, 5, 10]:
                            for tvP in [1e-5, 1e-6, 1e-7]:
                                    line = f"CUDA_VISIBLE_DEVICES={cuda_idx} python /home/jleuschn/riccardo/diffusion_models_dev_project/run_adapted_sampling.py --model_learned_on=aapm --model dds_unet --dataset=walnut --eta={eta} --gamma={gamma} --method={method} --ema --num_steps={num_steps} --cg_iter={num_cg_steps} --sde={sde} --load_path={LOAD_PATH} --tv_penalty {tvP} --adapt_freq 1 --num_optim_step {num_optim_step}  --lora_rank={lora_rank} --lr={lr} --add_cg --gamma={gamma} --cg_iter={num_cg_steps} "

                                    f.write(line)
                                    f.write(' && ')
"""