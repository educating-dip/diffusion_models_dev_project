
import os 

dataset = "lodopab"
cuda_idx = 0

for penalty in [750., 1000., 2000.]:
    for smpl_start_prc in [0]:
        for num_steps in [200]:
            os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python main_conditional_sampling.py --dataset {dataset} --penalty {penalty} \
                        --smpl_mthd {smpl_mthd} --smpl_start_prc {smpl_start_prc} --ema --num_steps {num_steps}")
        