
import os 
cuda_idx = 0


"""
for penalty in [750., 1000., 2000.]:
    for smpl_start_prc in [0]:
        for num_steps in [200]:
            os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python main_conditional_sampling.py --dataset {dataset} --penalty {penalty} \
                        --smpl_mthd {smpl_mthd} --smpl_start_prc {smpl_start_prc} --ema --num_steps {num_steps}")
"""


#for method in ["naive"]:
#    for penalty in [10, 20, 30, 40, 50, 60, 70]:
#        for num_steps in [50, 150, 300, 600]:
#            os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python run_conditional_sampling.py --model=ellipses \
#                        --dataset=ellipses --penalty={penalty} \
#                        --method=naive --ema --num_steps={num_steps}")


for method in ["dds"]:
    for eta in [0.15, 0.5, 0.85]:
        for gamma in [0.001, 0.1, 0.5, 0.8, 0.9, 0.99]:
            for num_steps in [25, 50, 75, 100]:
                os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python run_conditional_sampling.py --model=ellipses \
                            --dataset=ellipses --eta={eta} --gamma={gamma} \
                            --method={method} --ema --num_steps={num_steps}")