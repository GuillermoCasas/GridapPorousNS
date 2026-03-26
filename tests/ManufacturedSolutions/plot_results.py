import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_convergence():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    err_file = os.path.join(results_dir, 'errors.csv')
    
    if not os.path.exists(err_file):
        print(f"Error file not found: {err_file}")
        return

    df = pd.read_csv(err_file, sep='\t')
    
    h = df['h'].values
    err_u_l2 = df['err_u_l2'].values
    err_p_l2 = df['err_p_l2'].values
    err_u_h1 = df['err_u_h1'].values
    err_p_h1 = df['err_p_h1'].values
    
    import json
    config_file = os.path.join(os.path.dirname(__file__), 'data', 'test_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        k_u = config.get('discretization', {}).get('k_velocity', 2)
        k_p = config.get('discretization', {}).get('k_pressure', 1)
        opt_u_l2 = k_u + 1
        opt_u_h1 = k_u
        opt_p_l2 = k_p
        
        Re = config.get('physical_parameters', {}).get('Re', "N/A")
        Da = config.get('physical_parameters', {}).get('Da', "N/A")
        alpha_0 = config.get('porosity_field', {}).get('alpha_0', "N/A")
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    plt.figure(figsize=(10, 8))
    plt.loglog(h, err_u_l2, 'o-', color='C0', label=fr'Velocity $L^2$ Error (optimal: $\mathcal{{O}}(h^{{{opt_u_l2}}})$)')
    plt.loglog(h, err_u_h1, 's-', color='C1', label=fr'Velocity $H^1$ Error (optimal: $\mathcal{{O}}(h^{{{opt_u_h1}}})$)')
    plt.loglog(h, err_p_l2, '^-', color='C2', label=fr'Pressure $L^2$ Error (optimal: $\mathcal{{O}}(h^{{{opt_p_l2}}})$)')
    
    # Annotate local slopes for each segment
    for i in range(len(h) - 1):
        # Calculate slope
        slope_u_l2 = (np.log(err_u_l2[i+1]) - np.log(err_u_l2[i])) / (np.log(h[i+1]) - np.log(h[i]))
        slope_u_h1 = (np.log(err_u_h1[i+1]) - np.log(err_u_h1[i])) / (np.log(h[i+1]) - np.log(h[i]))
        slope_p_l2 = (np.log(err_p_l2[i+1]) - np.log(err_p_l2[i])) / (np.log(h[i+1]) - np.log(h[i]))
        
        # Find midpoint in log space for text placement
        h_mid = np.exp((np.log(h[i]) + np.log(h[i+1])) / 2.0)
        err_u_l2_mid = np.exp((np.log(err_u_l2[i]) + np.log(err_u_l2[i+1])) / 2.0)
        err_u_h1_mid = np.exp((np.log(err_u_h1[i]) + np.log(err_u_h1[i+1])) / 2.0)
        err_p_l2_mid = np.exp((np.log(err_p_l2[i]) + np.log(err_p_l2[i+1])) / 2.0)
        
        # Annotate
        plt.annotate(f'{slope_u_l2:.2f}', xy=(h_mid, err_u_l2_mid), xytext=(-10, 10), 
                     textcoords='offset points', color='C0', fontweight='bold')
        plt.annotate(f'{slope_u_h1:.2f}', xy=(h_mid, err_u_h1_mid), xytext=(10, 10), 
                     textcoords='offset points', color='C1', fontweight='bold')
        plt.annotate(f'{slope_p_l2:.2f}', xy=(h_mid, err_p_l2_mid), xytext=(-10, -15), 
                     textcoords='offset points', color='C2', fontweight='bold')
    
    plt.xlabel(r'Mesh size ($h$)')
    plt.ylabel('Error Norms')
    plt.title(fr'Convergence Analysis ($Re: {Re}$, $Da: {Da}$, $\alpha_0: {alpha_0}$)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    plot_file = os.path.join(results_dir, 'convergence.png')
    plt.savefig(plot_file)
    print(f"Convergence plot saved to: {plot_file}")

if __name__ == "__main__":
    plot_convergence()
