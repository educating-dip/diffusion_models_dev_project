#!/bin/bash

python mri_run_conditional_sampling_multi.py \
--method 'dds' \
--num_steps 50 \
--gamma 5.0 \
--eta 0.85 \
--cg_iter 5 \
--anatomy 'knee' \
--mask_type 'uniform1d' \
--acc_factor 4 \
--load_path './ckpts/fastmri_knee_320_complex_1m.pt' \
--base_path './fastmri_configs/ddpm/fastmri_knee_320_complex.yml'