import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

def plot_convergence():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    h5_file = os.path.join(results_dir, 'convergence_data.h5')
    
    if not os.path.exists(h5_file):
        print(f"Error file not found: {h5_file}")
        return

    with h5py.File(h5_file, 'r') as f:
        for group_name in f.keys():
            g = f[group_name]
            
            # Read Datasets
            h = g['h'][:]
            err_u_l2 = g['err_u_l2'][:]
            err_p_l2 = g['err_p_l2'][:]
            err_u_h1 = g['err_u_h1'][:]
            err_p_h1 = g['err_p_h1'][:]
            
            # Read Metadata
            Re = g.attrs.get('Re', 'N/A')
            Da = g.attrs.get('Da', 'N/A')
            kv = g.attrs.get('k_velocity', 'N/A')
            kp = g.attrs.get('k_pressure', 'N/A')
            alpha_0 = g.attrs.get('alpha_0', 'N/A')
            etype = g.attrs.get('element_type', 'N/A')
            if isinstance(etype, bytes):
                etype = etype.decode('utf-8')
            
            opt_u_l2 = kv + 1
            opt_u_h1 = kv
            opt_p_l2 = kp
            
            plt.figure(figsize=(10, 8))
            plt.loglog(h, err_u_l2, 'o-', color='C0', label=fr'Velocity $L^2$ Error (optimal: $\mathcal{{O}}(h^{{{opt_u_l2}}})$)')
            plt.loglog(h, err_u_h1, 's-', color='C1', label=fr'Velocity $H^1$ Error (optimal: $\mathcal{{O}}(h^{{{opt_u_h1}}})$)')
            plt.loglog(h, err_p_l2, '^-', color='C2', label=fr'Pressure $L^2$ Error (optimal: $\mathcal{{O}}(h^{{{opt_p_l2}}})$)')
            
            # Annotate local slopes for each segment
            for i in range(len(h) - 1):
                slope_u_l2 = (np.log(err_u_l2[i+1]) - np.log(err_u_l2[i])) / (np.log(h[i+1]) - np.log(h[i]))
                slope_u_h1 = (np.log(err_u_h1[i+1]) - np.log(err_u_h1[i])) / (np.log(h[i+1]) - np.log(h[i]))
                slope_p_l2 = (np.log(err_p_l2[i+1]) - np.log(err_p_l2[i])) / (np.log(h[i+1]) - np.log(h[i]))
                
                h_mid = np.exp((np.log(h[i]) + np.log(h[i+1])) / 2.0)
                err_u_l2_mid = np.exp((np.log(err_u_l2[i]) + np.log(err_u_l2[i+1])) / 2.0)
                err_u_h1_mid = np.exp((np.log(err_u_h1[i]) + np.log(err_u_h1[i+1])) / 2.0)
                err_p_l2_mid = np.exp((np.log(err_p_l2[i]) + np.log(err_p_l2[i+1])) / 2.0)
                
                plt.annotate(f'{slope_u_l2:.2f}', xy=(h_mid, err_u_l2_mid), xytext=(-10, 10), 
                             textcoords='offset points', color='C0', fontweight='bold')
                plt.annotate(f'{slope_u_h1:.2f}', xy=(h_mid, err_u_h1_mid), xytext=(10, 10), 
                             textcoords='offset points', color='C1', fontweight='bold')
                plt.annotate(f'{slope_p_l2:.2f}', xy=(h_mid, err_p_l2_mid), xytext=(-10, -15), 
                             textcoords='offset points', color='C2', fontweight='bold')
            
            plt.xlabel(r'Mesh size ($h$)')
            plt.ylabel('Error Norms')
            title_str = fr'Convergence ($Re: {Re}$, $Da: {Da}$, $\alpha_0: {alpha_0}$, $k_v: {kv}$, $k_p: {kp}$, Mesh: {etype})'
            plt.title(title_str)
            plt.legend()
            plt.grid(True, which="both", ls="--")
            
            plot_file = os.path.join(results_dir, f'convergence_Re{Re}_Da{Da}_a{alpha_0}_kv{kv}_kp{kp}_{etype}.png')
            plt.savefig(plot_file)
            print(f"Convergence plot saved to: {plot_file}")
            plt.close()

if __name__ == "__main__":
    plot_convergence()
