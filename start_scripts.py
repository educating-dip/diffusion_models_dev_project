
import os 

for penalty in [0.001, 0.01, 0.1, 1.0, 5.0, 10., 100., 500., 1000.]:
    print("Start Script")
    os.system(f"CUDA_VISIBLE_DEVICES=3 python main_conditional_sampling.py ellipses {penalty}")