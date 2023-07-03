#!/bin/bash

lr_list=(5e-4 5e-5)
iter_list=(3 5)
tvp_list=(0)

anatomy_list=('knee')
acc_factor_list=(4)

for lr in "${lr_list[@]}"; do
    for iter in "${iter_list[@]}"; do
        for tvp in "${tvp_list[@]}"; do
            for anatomy in "${anatomy_list[@]}"; do
                for acc_factor in "${acc_factor_list[@]}"; do
                    python mri_run_adapted_sampling_multi.py \
                    --method 'dds' \
                    --num_steps 50 \
                    --gamma 5.0 \
                    --eta 0.85 \
                    --cg_iter 5 \
                    --add_cg \
                    --anatomy $anatomy \
                    --mask_type 'uniform1d' \
                    --acc_factor $acc_factor \
                    --load_path './ckpts/fastmri_knee_320_complex_1m.pt' \
                    --base_path './fastmri_configs/ddpm/fastmri_knee_320_complex.yml' \
                    --lr $lr \
                    --num_optim_step $iter \
                    --tv_penalty $tvp
                done
            done
        done
    done
done