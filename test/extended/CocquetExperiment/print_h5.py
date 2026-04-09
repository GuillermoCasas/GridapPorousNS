import h5py
import numpy as np

f = h5py.File('results/convergence_cocquet.h5', 'r')
N_list = f['N_list'][:]
for method in f.keys():
    if method == "N_list": continue
    print(f"--- Method: {method} ---")
    for k in f[method].keys():
        errors = f[method][k]['errors_l2_u'][:]
        print(f"  k={k}: l2_u errors = {errors}")
        for i in range(len(errors)-1):
            h_ratio = N_list[i+1] / N_list[i]
            e_ratio = errors[i] / errors[i+1]
            slope = np.log(e_ratio) / np.log(h_ratio)
            print(f"    slope {N_list[i]}->{N_list[i+1]} = {slope:.3f}")
f.close()
