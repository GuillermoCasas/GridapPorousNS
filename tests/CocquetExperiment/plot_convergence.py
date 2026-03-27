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
    
    colors = ['C0', 'C1']
    markers_u = ['o', 'o']
    markers_p = ['s', 's']

    with h5py.File(h5_file, 'r') as file:
        N_list = np.array(file["N_list"])
        h = 1.0 / N_list
        
        for idx, k in enumerate([1, 2]):
            group = file[f"P{k}P{k}"]
            err_u = np.array(group["errors_l2_u"])
            err_p = np.array(group["errors_l2_p"])
            
            c = colors[idx]
            opt_u = k + 1
            opt_p = k
            
            plt.loglog(N_list, err_u, f'{c}{markers_u[idx]}-', linewidth=2, markersize=8, 
                       label=fr'P{k}/P{k} L2 Velocity (opt: $\mathcal{{O}}(h^{opt_u})$)')
            plt.loglog(N_list, err_p, f'{c}{markers_p[idx]}--', linewidth=2, markersize=8, 
                       label=fr'P{k}/P{k} L2 Pressure (opt: $\mathcal{{O}}(h^{opt_p})$)')
            
            # Print expected slopes correctly anchored relative to N
            offset_ideal = err_u[0] * (N_list[0]**opt_u)
            plt.loglog(N_list, offset_ideal * (N_list**(-float(opt_u))), f'{c}:', alpha=0.5, label=f'O(h^{opt_u}) reference')
            
            # Annotate local slopes for each segment
            for i in range(len(N_list) - 1):
                slope_u = (np.log(err_u[i+1]) - np.log(err_u[i])) / (np.log(h[i+1]) - np.log(h[i]))
                slope_p = (np.log(err_p[i+1]) - np.log(err_p[i])) / (np.log(h[i+1]) - np.log(h[i]))
                
                N_mid = np.exp((np.log(N_list[i]) + np.log(N_list[i+1])) / 2.0)
                err_u_mid = np.exp((np.log(err_u[i]) + np.log(err_u[i+1])) / 2.0)
                err_p_mid = np.exp((np.log(err_p[i]) + np.log(err_p[i+1])) / 2.0)
                
                plt.annotate(f'{slope_u:.2f}', xy=(N_mid, err_u_mid), xytext=(-15, 15), 
                             textcoords='offset points', color=c, fontweight='bold')
                plt.annotate(f'{slope_p:.2f}', xy=(N_mid, err_p_mid), xytext=(15, -15), 
                             textcoords='offset points', color=c, fontweight='bold')
            
    plt.xlabel('Linear Mesh Elements (N)')
    plt.ylabel(r'$L^2$ Error Norm')
    plt.title('Convergence Analysis: Equal-Order P1 and P2 ASGS (Re=500, Da=1.0)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plot_file = os.path.join(results_dir, 'convergence_cocquet.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Convergence plotted and saved to {plot_file}")

if __name__ == "__main__":
    plot_cocquet()
