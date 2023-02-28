
import os 

dataset = "ellipses"

sparse_view = False 
limited_angle = True
smpl_mthd = 'pc' # 'naive_smpl' #"pc"

cuda_idx = 0

for penalty in [100.]:
    for smpl_start_prc in [0, 0.8]:
        if sparse_view:
            for angles in [50, 100, 200]:
                print("Start Script")
                if sparse_view:
                    os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python main_conditional_sampling.py --dataset {dataset} --penalty {penalty} \
                            --smpl_mthd {smpl_mthd} --smpl_start_prc {smpl_start_prc} --sparse_view --angles {angles} ")
        elif limited_angle: 
            for angular_range in [45, 90]:
                print("Start Script")
                os.system(f"CUDA_VISIBLE_DEVICES={cuda_idx} python main_conditional_sampling.py --dataset {dataset} --penalty {penalty} \
                                --smpl_mthd {smpl_mthd} --smpl_start_prc {smpl_start_prc} --limited_angle --angular_range {angular_range} ")

