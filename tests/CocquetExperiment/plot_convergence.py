# tests/CocquetExperiment/plot_convergence.py
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cocquet():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    h5_file = os.path.join(results_dir, 'convergence_cocquet.h5')
    
    if not os.path.exists(h5_file):
        print(f"HDF5 file not found: {h5_file}")
        return

    plt.figure(figsize=(10, 8))
    
    colors = ['b', 'g']
    with h5py.File(h5_file, 'r') as file:
        N_list = np.array(file["N_list"])
        
        for idx, k in enumerate([1, 2]):
            group = file[f"P{k}P{k}"]
            err_u = np.array(group["errors_l2_u"])
            err_p = np.array(group["errors_l2_p"])
            
            plt.loglog(N_list, err_u, f'{colors[idx]}o-', linewidth=2, markersize=8, label=f'P{k}/P{k} L2 Velocity')
            plt.loglog(N_list, err_p, f'{colors[idx]}s--', linewidth=2, markersize=8, label=f'P{k}/P{k} L2 Pressure')
            
            # Print expected slopes correctly anchored
            order = k + 1 # theoretically optimal velocity L2 convergence
            offset_ideal = err_u[0] * (N_list[0]**order)
            plt.loglog(N_list, offset_ideal * (N_list**(-float(order))), 'k--', alpha=0.5, label=f'O(h^{order})')
            
    plt.xlabel('N')
    plt.ylabel('L2 error')
    plt.title('Convergence Analysis: Equal-Order P1 and P2 (Re=500, Da=1.0)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plot_file = os.path.join(results_dir, 'convergence_cocquet.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Convergence plotted and saved to {plot_file}")

if __name__ == "__main__":
    plot_cocquet()
