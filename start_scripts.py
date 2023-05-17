
import os 
cuda_idx = 7


"""
for penalty in [750., 1000., 2000.]:
    for smpl_start_prc in [0]:
        for num_steps in [200]:
            os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python main_conditional_sampling.py --dataset {dataset} --penalty {penalty} \
                        --smpl_mthd {smpl_mthd} --smpl_start_prc {smpl_start_prc} --ema --num_steps {num_steps}")
"""


for method in ["naive"]:
    for penalty in [75.,80.,85.,90.]:
        for num_steps in [150, 300, 600]:
            for sde in ["vesde", "vpsde"]:
                os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python run_conditional_sampling.py --model_learned_on=ellipses \
                        --dataset=walnut --penalty={penalty} \
                        --method=naive --ema --num_steps={num_steps} --sde={sde} --version=1")


"""
for method in ["dds"]:
    for eta in [0.15, 0.5, 0.85]: #[0.15, 0.5, 0.85]:
        for gamma in [0.05, 0.1, 0.25, 0.85, 1.0, 2.0, 10.]: #[0.001, 0.1, 0.5, 0.8, 0.9, 0.99]:
            for num_steps in [50, 75]: #[25, 50, 75, 100]:
                for sde in ["vesde", "vpsde"]:
                    os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python run_conditional_sampling.py --model_learned_on=ellipses \
                            --dataset=walnut --eta={eta} --gamma={gamma} \
                            --method={method} --ema --num_steps={num_steps} --cg_iter=6 --sde={sde} --version=1")
"""

"""
for method in ["dps"]:
    for penalty in [0.005, 0.008, 0.02, 0.1, 0.5, 2., 5.]:
        for num_steps in [50, 200, 400]:
            for sde in ["vesde", "vpsde"]:
                os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python run_conditional_sampling.py --model_learned_on=ellipses \
                        --dataset=walnut --penalty={penalty} \
                        --method=dps --ema --num_steps={num_steps} --sde={sde} --version=1")
"""