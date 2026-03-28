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

    # Dictionary to store data for tables
    # keys: (etype, kv, kp)
    # values: list of dicts with Re, Da, alpha_0, slope_u, fme_u, slope_p, fme_p
    table_data = {}

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
            
            # For summary tables: compute slope from two finest meshes and FME natively
            if len(h) >= 2:
                slope_finest_u = (np.log(err_u_l2[-1]) - np.log(err_u_l2[-2])) / (np.log(h[-1]) - np.log(h[-2]))
                slope_finest_p = (np.log(err_p_l2[-1]) - np.log(err_p_l2[-2])) / (np.log(h[-1]) - np.log(h[-2]))
                fme_u = err_u_l2[-1]
                fme_p = err_p_l2[-1]
            else:
                slope_finest_u = 0.0
                slope_finest_p = 0.0
                fme_u = err_u_l2[-1] if len(h) > 0 else 0.0
                fme_p = err_p_l2[-1] if len(h) > 0 else 0.0
                
            key = (etype, kv, kp)
            if key not in table_data:
                table_data[key] = []
            
            table_data[key].append({
                'Re': float(Re),
                'Da': float(Da),
                'alpha_0': float(alpha_0),
                'slope_u': slope_finest_u,
                'fme_u': fme_u,
                'slope_p': slope_finest_p,
                'fme_p': fme_p
            })
            
            # Plotting logic
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
            plt.close()

    # Generate the text tables
    write_summary_tables(table_data, results_dir)

def write_summary_tables(table_data, results_dir):
    summary_file = os.path.join(results_dir, 'summary_tables.txt')
    
    with open(summary_file, 'w') as f:
        # Sort keys to ensure deterministic ordering (e.g. QUAD/TRI, P1/P2)
        sorted_keys = sorted(table_data.keys(), key=lambda x: (x[0], x[1], x[2]))
        
        for key in sorted_keys:
            etype, kv, kp = key
            opt_u = kv + 1
            opt_p = kp
            
            data = table_data[key]
            # Standard theoretical format sorts by ascending Da, ascending Re, descending alpha_0
            data.sort(key=lambda x: (x['Da'], x['Re'], -x['alpha_0']))
            
            # Write the table headers explicitly mirroring theoretical boundaries
            f.write(f"Table: Observed convergence rates and normalized finest mesh error (FME) for P{kv}/P{kp} elements ({etype})\n")
            f.write("--------------------------------------------------------------------------------------------------------\n\n")
            
            # Velocity Table Matrix
            f.write("velocity\n")
            f.write(f"                        slope ({opt_u})      FME\n")
            f.write("Re          Da          α0      ASGS          ASGS\n")
            for row in data:
                Re_str = format_scientific(row['Re'])
                Da_str = format_scientific(row['Da'])
                a0_str = f"{row['alpha_0']:.2f}"
                slope = f"{row['slope_u']:.2f}"
                fme = f"{row['fme_u']:.2e}"
                f.write(f"{Re_str:<11} {Da_str:<11} {a0_str:<7} {slope:<13} {fme:<13}\n")
            
            f.write("\n")
            
            # Pressure Table Matrix
            f.write("pressure\n")
            f.write(f"                        slope ({opt_p})      FME\n")
            f.write("Re          Da          α0      ASGS          ASGS\n")
            for row in data:
                Re_str = format_scientific(row['Re'])
                Da_str = format_scientific(row['Da'])
                a0_str = f"{row['alpha_0']:.2f}"
                slope = f"{row['slope_p']:.2f}"
                fme = f"{row['fme_p']:.2e}"
                f.write(f"{Re_str:<11} {Da_str:<11} {a0_str:<7} {slope:<13} {fme:<13}\n")
            
            f.write("\n========================================================================================================\n\n")
            
    print(f"Summary tables structurally generated and saved to: {summary_file}")

def format_scientific(val):
    if val == 1.0:
        return "1"
    else:
        # e.g., 1e-06 -> 10^-6 natively
        s = f"{val:.0e}"
        s = s.replace("e-0", "e-").replace("e+0", "e")
        base, exp = s.split("e")
        if base == "1":
            return f"10^{exp}"
        return s

if __name__ == "__main__":
    plot_convergence()
