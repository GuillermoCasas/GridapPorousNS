# test/long/CocquetExperiment/plot_convergence.py
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

    # Create a 1x2 subplot layout to cleanly separate L2 and H1 metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    with h5py.File(h5_file, 'r') as file:
        N_list = np.array(file["N_list"])
        h = 1.0 / N_list
        
        delta = None
        for k in [1, 2]:
            for method in ["ASGS", "OSGS"]:
                group_name = f"{method}/P{k}P{k}"
                if group_name in file:
                    if "outlet_truncation_delta" in file[group_name].attrs:
                        delta = file[group_name].attrs["outlet_truncation_delta"]
                        break
            if delta is not None:
                break
        
        for k in [1, 2]:
            for method in ["ASGS", "OSGS"]:
                group_name = f"{method}/P{k}P{k}"
                if group_name not in file:
                    continue
                
                group = file[group_name]
                err_u = np.array(group["errors_l2_u"])
                err_p = np.array(group["errors_l2_p"])
                
                err_u_h1 = np.array(group["errors_h1_u"])
                err_p_h1 = np.array(group["errors_h1_p"])
                
                marker = '^' if k == 1 else 's'
                ls = '-' if method == "ASGS" else '--'
                
                # Theoretical Rates
                opt_u_l2 = k + 1
                opt_p_l2 = k
                opt_u_h1 = k
                opt_p_h1 = k
                
                # Plot L2 Data
                ax1.loglog(N_list, err_u, color='blue', marker=marker, linestyle=ls, linewidth=2, markersize=8, 
                           label=fr'{method} P{k}/P{k} $L_2$ Velocity ({opt_u_l2})')
                ax1.loglog(N_list, err_p, color='red', marker=marker, linestyle=ls, linewidth=2, markersize=8, 
                           markerfacecolor='white', label=fr'{method} P{k}/P{k} $L_2$ Pressure ({opt_p_l2})')

                # Plot H1 Data
                ax2.loglog(N_list, err_u_h1, color='blue', marker=marker, linestyle=ls, linewidth=2, markersize=8, 
                           label=fr'{method} P{k}/P{k} $H_1$ Velocity ({opt_u_h1})')
                ax2.loglog(N_list, err_p_h1, color='red', marker=marker, linestyle=ls, linewidth=2, markersize=8, 
                           markerfacecolor='white', label=fr'{method} P{k}/P{k} $H_1$ Pressure ({opt_p_h1})')
                
                # Annotate local slopes for each segment dynamically handling collisions
                for i in range(len(N_list) - 1):
                    slope_u = ((np.log(err_u[i+1]) - np.log(err_u[i])) / (np.log(h[i+1]) - np.log(h[i]))) / opt_u_l2
                    slope_p = ((np.log(err_p[i+1]) - np.log(err_p[i])) / (np.log(h[i+1]) - np.log(h[i]))) / opt_p_l2
                    
                    slope_u_h1 = ((np.log(err_u_h1[i+1]) - np.log(err_u_h1[i])) / (np.log(h[i+1]) - np.log(h[i]))) / opt_u_h1
                    slope_p_h1 = ((np.log(err_p_h1[i+1]) - np.log(err_p_h1[i])) / (np.log(h[i+1]) - np.log(h[i]))) / opt_p_h1
                    
                    # Split ASGS and OSGS anchors precisely on the interpolating log segment to prevent crashes
                    alpha_point = 0.50 if method == "ASGS" else 0.75
                    log_N_mid = np.log(N_list[i]) * (1-alpha_point) + np.log(N_list[i+1]) * alpha_point
                    N_mid = np.exp(log_N_mid)

                    # --- L2 Midpoints ---
                    log_u_mid = np.log(err_u[i]) * (1-alpha_point) + np.log(err_u[i+1]) * alpha_point
                    log_p_mid = np.log(err_p[i]) * (1-alpha_point) + np.log(err_p[i+1]) * alpha_point
                    err_u_mid = np.exp(log_u_mid)
                    err_p_mid = np.exp(log_p_mid)
                    
                    # --- H1 Midpoints ---
                    log_u_mid_h1 = np.log(err_u_h1[i]) * (1-alpha_point) + np.log(err_u_h1[i+1]) * alpha_point
                    log_p_mid_h1 = np.log(err_p_h1[i]) * (1-alpha_point) + np.log(err_p_h1[i+1]) * alpha_point
                    err_u_mid_h1 = np.exp(log_u_mid_h1)
                    err_p_mid_h1 = np.exp(log_p_mid_h1)
                    
                    # Compute constant pixel offsets mathematically using tangential vector rotation
                    d_log_N = np.log(N_list[i+1]) - np.log(N_list[i])
                    offset_pts = 15.0 # Set perfectly distinct layout natively tracking log-log screens spacing
                    
                    # L2 Text Rotation
                    d_log_u = np.log(err_u[i+1]) - np.log(err_u[i])
                    len_u = np.sqrt(d_log_N**2 + d_log_u**2)
                    nu_x, nu_y = -d_log_u / len_u, d_log_N / len_u
                    
                    d_log_p = np.log(err_p[i+1]) - np.log(err_p[i])
                    len_p = np.sqrt(d_log_N**2 + d_log_p**2)
                    np_x, np_y = d_log_p / len_p, -d_log_N / len_p
                    
                    ax1.annotate(f'{slope_u:.2f}', xy=(N_mid, err_u_mid), xycoords='data', xytext=(nu_x*offset_pts, nu_y*offset_pts), textcoords='offset points', color='blue', fontweight='bold', ha='center', va='center')
                    ax1.annotate(f'{slope_p:.2f}', xy=(N_mid, err_p_mid), xycoords='data', xytext=(np_x*offset_pts, np_y*offset_pts), textcoords='offset points', color='red', fontweight='bold', ha='center', va='center')
                    
                    # H1 Text Rotation
                    d_log_u_h1 = np.log(err_u_h1[i+1]) - np.log(err_u_h1[i])
                    len_u_h1 = np.sqrt(d_log_N**2 + d_log_u_h1**2)
                    nu_x_h1, nu_y_h1 = -d_log_u_h1 / len_u_h1, d_log_N / len_u_h1
                    
                    d_log_p_h1 = np.log(err_p_h1[i+1]) - np.log(err_p_h1[i])
                    len_p_h1 = np.sqrt(d_log_N**2 + d_log_p_h1**2)
                    np_x_h1, np_y_h1 = d_log_p_h1 / len_p_h1, -d_log_N / len_p_h1

                    ax2.annotate(f'{slope_u_h1:.2f}', xy=(N_mid, err_u_mid_h1), xycoords='data', xytext=(nu_x_h1*offset_pts, nu_y_h1*offset_pts), textcoords='offset points', color='blue', fontweight='bold', ha='center', va='center')
                    ax2.annotate(f'{slope_p_h1:.2f}', xy=(N_mid, err_p_mid_h1), xycoords='data', xytext=(np_x_h1*offset_pts, np_y_h1*offset_pts), textcoords='offset points', color='red', fontweight='bold', ha='center', va='center')
            
    ax1.set_xlabel(r'$N$')
    ax1.set_ylabel(r'$L_2$-norm error')
    ax1.set_title(r'Spatial convergence: $L_2$-norms')
    ax1.grid(True, which="both", ls="--")
    ax1.legend(handlelength=4.0)

    ax2.set_xlabel(r'$N$')
    ax2.set_ylabel(r'$H_1$-seminorm error')
    ax2.set_title(r'Spatial convergence: $H_1$-seminorms')
    ax2.grid(True, which="both", ls="--")
    ax2.legend(handlelength=4.0)
    
    delta_str = f", $\delta=${delta}" if delta is not None else ""
    plt.suptitle(rf'Cocquet Experiment Convergence Analysis: P1 and P2 (Re=500, c_in=0.5{delta_str})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_file = os.path.join(results_dir, 'convergence_cocquet.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Convergence plotted and saved to {plot_file}")

if __name__ == "__main__":
    plot_cocquet()
