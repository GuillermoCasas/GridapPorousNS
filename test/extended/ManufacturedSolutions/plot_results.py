import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

import time

def robust_open_h5(filepath, mode='r', retries=10, delay=2.0):
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    for attempt in range(retries):
        try:
            return h5py.File(filepath, mode, swmr=True)
        except (OSError, RuntimeError) as e:
            if attempt < retries - 1:
                print(f"HDF5 file busy/locked. Retrying in {delay}s... ({attempt+1}/{retries})")
                time.sleep(delay)
            else:
                print(f"Failed to open HDF5 file after {retries} attempts: {e}. Skipping plot generation.")
                import sys
                sys.exit(0)

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

    with robust_open_h5(h5_file, 'r') as f:
        # Group runs by config_idx
        import collections
        config_dict = collections.defaultdict(dict)
        for group_name in f.keys():
            parts = group_name.split('_')
            if len(parts) >= 3 and parts[0] == 'config':
                c_idx = parts[1]
                method = parts[-1] 
                config_dict[c_idx][method] = f[group_name]

        for c_idx, methods in config_dict.items():
            # We'll use the metadata from the first method we find (they are identical except for method)
            first_g = next(iter(methods.values()))
            
            # Read Metadata once per config
            Re = first_g.attrs.get('Re', 'N/A')
            Da = first_g.attrs.get('Da', 'N/A')
            kv = first_g.attrs.get('k_velocity', 'N/A')
            kp = first_g.attrs.get('k_pressure', 'N/A')
            alpha_0 = first_g.attrs.get('alpha_0', 'N/A')
            etype = first_g.attrs.get('element_type', 'N/A')
            if isinstance(etype, bytes):
                etype = etype.decode('utf-8')
                
            opt_u_l2 = kv + 1
            opt_u_h1 = kv
            opt_p_l2 = kp
            
            plt.figure(figsize=(10, 8))
            
            table_row = {
                'Re': float(Re), 'Da': float(Da), 'alpha_0': float(alpha_0)
            }
            
            for method, g in methods.items():
                h = g['h'][:]
                err_u_l2 = g['err_u_l2'][:]
                err_p_l2 = g['err_p_l2'][:]
                err_u_h1 = g['err_u_h1'][:]
                err_p_h1 = g['err_p_h1'][:]
                
                # Filter out pure-zero arrays avoiding nan logs
                valid_mask = (err_u_l2 > 0) & (err_p_l2 > 0)
                h = h[valid_mask]
                err_u_l2 = err_u_l2[valid_mask]
                err_p_l2 = err_p_l2[valid_mask]
                err_u_h1 = err_u_h1[valid_mask]
                err_p_h1 = err_p_h1[valid_mask]
                
                ls = '-' if method == 'ASGS' else '--'
                
                if len(h) >= 2:
                    slope_finest_u = (np.log(err_u_l2[-1]) - np.log(err_u_l2[-2])) / (np.log(h[-1]) - np.log(h[-2]))
                    slope_finest_p = (np.log(err_p_l2[-1]) - np.log(err_p_l2[-2])) / (np.log(h[-1]) - np.log(h[-2]))
                    fme_u = err_u_l2[-1]
                    fme_p = err_p_l2[-1]
                else:
                    slope_finest_u = float('nan')
                    slope_finest_p = float('nan')
                    fme_u = err_u_l2[-1] if len(h) > 0 else float('nan')
                    fme_p = err_p_l2[-1] if len(h) > 0 else float('nan')
                
                table_row[f'slope_u_{method}'] = slope_finest_u / opt_u_l2 if opt_u_l2 else float('nan')
                table_row[f'fme_u_{method}'] = fme_u
                table_row[f'slope_p_{method}'] = slope_finest_p / opt_p_l2 if opt_p_l2 else float('nan')
                table_row[f'fme_p_{method}'] = fme_p
                
                plt.loglog(h, err_u_l2, marker='o', linestyle=ls, color='blue', linewidth=2, markersize=8,
                           label=f'{method} $L_2$ Velocity ({opt_u_l2})')
                plt.loglog(h, err_u_h1, marker='D', linestyle=ls, color='blue', linewidth=2, markersize=8,
                           label=f'{method} $H^1$ Velocity ({opt_u_h1})')
                plt.loglog(h, err_p_l2, marker='o', linestyle=ls, color='red', linewidth=2, markersize=8, markerfacecolor='white',
                           label=f'{method} $L_2$ Pressure ({opt_p_l2})')

                # Annotate local slopes for each segment
                alpha = 0.50 if method == 'ASGS' else 0.75

                for i in range(len(h)-1):
                    # Points in log space
                    h_pt = np.exp((1.0 - alpha) * np.log(h[i]) + alpha * np.log(h[i+1]))
                    
                    # Log distances to estimate visual direction
                    d_log_h = np.log(h[i+1]) - np.log(h[i])
                    
                    def annotate_slope(err_arr, c_plot, err_name, opt_slope):
                        err_pt = np.exp((1.0 - alpha) * np.log(err_arr[i]) + alpha * np.log(err_arr[i+1]))
                        slope_val = (np.log(err_arr[i+1]) - np.log(err_arr[i])) / (np.log(h[i+1]) - np.log(h[i]))
                        slope_ratio = slope_val / opt_slope if opt_slope else float('nan')
                        
                        d_log_e = np.log(err_arr[i+1]) - np.log(err_arr[i])
                        
                        # Orient tangent from left to right (visual axis +x)
                        if d_log_h < 0:
                            t_x, t_y = -d_log_h, -d_log_e
                        else:
                            t_x, t_y = d_log_h, d_log_e
                            
                        # Normalize direction exactly as plotted proportions
                        length = np.sqrt(t_x**2 + t_y**2)
                        if length == 0: length = 1.0
                        
                        # Normal vector "up and left" from tangent
                        n_x = -t_y / length
                        n_y = t_x / length
                        
                        offset_pts = 15.0 if method == 'ASGS' else -15.0
                        
                        # Dynamically adjust side if another method exists
                        for other_method, other_g in methods.items():
                            if other_method != method:
                                if err_name in other_g and 'h' in other_g:
                                    other_h_full = other_g['h'][:]
                                    other_err_full = other_g[err_name][:]
                                    other_mask = (other_g['err_u_l2'][:] > 0) & (other_g['err_p_l2'][:] > 0)
                                    other_h = other_h_full[other_mask]
                                    other_err = other_err_full[other_mask]
                                    
                                    idx1 = np.where(np.isclose(other_h, h[i]))[0]
                                    idx2 = np.where(np.isclose(other_h, h[i+1]))[0]
                                    if len(idx1) > 0 and len(idx2) > 0:
                                        other_err1 = other_err[idx1[0]]
                                        other_err2 = other_err[idx2[0]]
                                        other_err_pt = np.exp((1.0 - alpha) * np.log(other_err1) + alpha * np.log(other_err2))
                                        
                                        if err_pt > other_err_pt:
                                            offset_pts = 15.0
                                        elif err_pt < other_err_pt:
                                            offset_pts = -15.0
                                        break
                        
                        # Generate the visual fraction representation
                        annot_text = f'{slope_ratio:.2f}'
                        
                        plt.annotate(annot_text, xy=(h_pt, err_pt), 
                                     xytext=(n_x * offset_pts, n_y * offset_pts),
                                     textcoords='offset points', ha='center', va='center', 
                                     fontsize=8, color=c_plot, fontweight='bold')

                    annotate_slope(err_u_l2, 'blue', 'err_u_l2', opt_u_l2)
                    annotate_slope(err_u_h1, 'blue', 'err_u_h1', opt_u_h1)
                    annotate_slope(err_p_l2, 'red', 'err_p_l2', opt_p_l2)
            
            key = (etype, kv, kp)
            if key not in table_data:
                table_data[key] = []
            table_data[key].append(table_row)
            
            plt.xlabel(r'Mesh size ($h$)')
            plt.ylabel('Error Norms')
            title_str = fr'Convergence ($Re: {Re}$, $Da: {Da}$, $\alpha_0: {alpha_0}$, $k: {kv}$, Mesh: {etype})'
            plt.title(title_str)
            plt.legend(handlelength=4.0)
            plt.grid(True, which="both", ls="--")
            
            plot_file = os.path.join(results_dir, f'convergence_config{c_idx}_Re{Re}_Da{Da}_a{alpha_0}_{etype}.png')
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
            f.write(f"                        ASGS ({opt_u})                   OSGS ({opt_u})    \n")
            f.write("Re          Da          α0      slope        FME          slope        FME\n")
            for row in data:
                Re_str = format_scientific(row['Re'])
                Da_str = format_scientific(row['Da'])
                a0_str = f"{row['alpha_0']:.2f}"
                
                def fmt_slope(v): return "N/A" if np.isnan(v) else f"{v:.2f}"
                def fmt_fme(v): return "N/A" if np.isnan(v) else f"{v:.2e}"
                
                slope_a = fmt_slope(row.get('slope_u_ASGS', np.nan))
                fme_a = fmt_fme(row.get('fme_u_ASGS', np.nan))
                slope_o = fmt_slope(row.get('slope_u_OSGS', np.nan))
                fme_o = fmt_fme(row.get('fme_u_OSGS', np.nan))
                f.write(f"{Re_str:<11} {Da_str:<11} {a0_str:<7} {slope_a:<12} {fme_a:<12} {slope_o:<12} {fme_o:<12}\n")
            
            f.write("\n")
            
            # Pressure Table Matrix
            f.write("pressure\n")
            f.write(f"                        ASGS ({opt_p})                   OSGS ({opt_p})    \n")
            f.write("Re          Da          α0      slope        FME          slope        FME\n")
            for row in data:
                Re_str = format_scientific(row['Re'])
                Da_str = format_scientific(row['Da'])
                a0_str = f"{row['alpha_0']:.2f}"
                
                def fmt_slope(v): return "N/A" if np.isnan(v) else f"{v:.2f}"
                def fmt_fme(v): return "N/A" if np.isnan(v) else f"{v:.2e}"
                
                slope_a = fmt_slope(row.get('slope_p_ASGS', np.nan))
                fme_a = fmt_fme(row.get('fme_p_ASGS', np.nan))
                slope_o = fmt_slope(row.get('slope_p_OSGS', np.nan))
                fme_o = fmt_fme(row.get('fme_p_OSGS', np.nan))
                f.write(f"{Re_str:<11} {Da_str:<11} {a0_str:<7} {slope_a:<12} {fme_a:<12} {slope_o:<12} {fme_o:<12}\n")
            
            f.write("\n========================================================================================================\n\n")
            
    print(f"Summary tables structurally generated and saved to: {summary_file}")

def generate_markdown_report(h5_file, results_dir):
    report_file = os.path.join(results_dir, "convergence_report.md")
    with open(report_file, "w") as io:
        io.write("# Convergence Rate and FME Table\n\n")
        io.write("| Config | Method | Re | Da | α_0 | k | Elem | Iters | Converged | epsilon_pert | rate_u_L2 (opt) | rate_p_L2 (opt) | rate_u_H1 (opt) | rate_p_H1 (opt) | FME u_L2 | FME p_L2 | FME u_H1 | FME p_H1 |\n")
        io.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        
        with robust_open_h5(h5_file, 'r') as h5f:
            groups = sorted(h5f.keys(), key=lambda x: int(x.split('_')[1]))
            for group_name in groups:
                g = h5f[group_name]
                c_idx = group_name.split('_')[1]
                method = group_name.split('_')[2]
                
                re = float(g.attrs['Re'])
                da = float(g.attrs['Da'])
                a0 = float(g.attrs['alpha_0'])
                kv = int(g.attrs['k_velocity'])
                etype = g.attrs['element_type']
                if isinstance(etype, bytes): etype = etype.decode('utf-8')
                
                h_arr = g['h'][:]
                err_u_l2_arr = g['err_u_l2'][:]
                err_p_l2_arr = g['err_p_l2'][:]
                err_u_h1_arr = g['err_u_h1'][:]
                err_p_h1_arr = g['err_p_h1'][:]
                
                # Filter valid points
                mask = (err_u_l2_arr > 0) & (err_p_l2_arr > 0)
                h_arr = h_arr[mask]
                err_u_l2_arr = err_u_l2_arr[mask]
                err_p_l2_arr = err_p_l2_arr[mask]
                err_u_h1_arr = err_u_h1_arr[mask]
                err_p_h1_arr = err_p_h1_arr[mask]
                
                if len(h_arr) >= 2:
                    opt_u_l2 = float(kv + 1)
                    opt_p_l2 = float(kp if 'kp' in locals() else kv) # fallback if kp not available directly
                    opt_u_h1 = float(kv)
                    opt_p_h1 = float(kv - 1)
                    
                    rate_u_l2 = ((np.log(err_u_l2_arr[-1]) - np.log(err_u_l2_arr[-2])) / (np.log(h_arr[-1]) - np.log(h_arr[-2]))) / opt_u_l2
                    rate_p_l2 = ((np.log(err_p_l2_arr[-1]) - np.log(err_p_l2_arr[-2])) / (np.log(h_arr[-1]) - np.log(h_arr[-2]))) / opt_p_l2 if opt_p_l2 else float('nan')
                    rate_u_h1 = ((np.log(err_u_h1_arr[-1]) - np.log(err_u_h1_arr[-2])) / (np.log(h_arr[-1]) - np.log(h_arr[-2]))) / opt_u_h1 if opt_u_h1 else float('nan')
                    rate_p_h1 = ((np.log(err_p_h1_arr[-1]) - np.log(err_p_h1_arr[-2])) / (np.log(h_arr[-1]) - np.log(h_arr[-2]))) / opt_p_h1 if opt_p_h1 else float('nan')
                else:
                    rate_u_l2 = rate_p_l2 = rate_u_h1 = rate_p_h1 = float('nan')
                
                err_u_l2_last = err_u_l2_arr[-1] if len(err_u_l2_arr) > 0 else float('nan')
                err_p_l2_last = err_p_l2_arr[-1] if len(err_p_l2_arr) > 0 else float('nan')
                err_u_h1_last = err_u_h1_arr[-1] if len(err_u_h1_arr) > 0 else float('nan')
                err_p_h1_last = err_p_h1_arr[-1] if len(err_p_h1_arr) > 0 else float('nan')
                
                opt_u_l2 = float(kv + 1)
                opt_p_l2 = float(kv)  # assuming equal order
                opt_u_h1 = float(kv)
                opt_p_h1 = float(kv - 1)
                
                # We already divided rates by opt, so target is 1.0. We pass 1.0 to format_rate to indicate optimum
                def format_rate(rate, opt):
                    if np.isnan(rate):
                        return f"  N/A ({int(opt)})"
                    threshold = 0.90 # Since normalized, threshold is 0.9
                    if rate < threshold:
                        return f"<b style='color:red'>{rate:5.2f}</b> (1.0)"
                    return f" {rate:5.2f} (1.0)"
                
                def format_sci(v):
                    if np.isnan(v): return "NaN"
                    return f"{v:.4e}"

                def jl_fmt(val):
                    if val == 1.0: return "1e+00"
                    if val == 1e-6: return "1e-06"
                    if val == 1e6: return "1e+06"
                    s = f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e+")
                    return s

                ru2 = format_rate(rate_u_l2, opt_u_l2)
                rp2 = format_rate(rate_p_l2, opt_p_l2)
                ru1 = format_rate(rate_u_h1, opt_u_h1)
                rp1 = format_rate(rate_p_h1, opt_p_h1)
                
                eval_iters = g['eval_iters'][:] if 'eval_iters' in g else []
                total_iters = int(g.attrs.get('total_iters', sum(eval_iters) if len(eval_iters) > 0 else 0))
                
                eval_eps = g['eval_eps'][:] if 'eval_eps' in g else []
                eps_val = np.min(eval_eps) if len(eval_eps) > 0 else float('nan')
                
                if method == 'ASGS':
                    converged = "<b style='color:red'>No</b>" if any(it >= 15 for it in eval_iters) else "Yes"
                else: # OSGS takes max 4 iters per outer, typically 3 outer loops
                    converged = "<b style='color:red'>No</b>" if any(it >= 12 for it in eval_iters) else "Yes"
                
                io.write(f"| C{c_idx} | {method} | {jl_fmt(re)} | {jl_fmt(da)} | {a0:.2f} | {kv} | {etype} | {total_iters} | {converged} | {format_sci(eps_val)} | {ru2} | {rp2} | {ru1} | {rp1} | {format_sci(err_u_l2_last)} | {format_sci(err_p_l2_last)} | {format_sci(err_u_h1_last)} | {format_sci(err_p_h1_last)} |\n")
                
    print(f"Convergence Markdown report generated explicitly securely to: {report_file}")

def format_scientific(val):
    if val == 1.0:
        return "1"
    else:
        s = f"{val:.0e}"
        s = s.replace("e-0", "e-").replace("e+0", "e")
        base, exp = s.split("e")
        if base == "1":
            return f"10^{exp}"
        return s

if __name__ == "__main__":
    plot_convergence()
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    h5_file = os.path.join(results_dir, 'convergence_data.h5')
    if os.path.exists(h5_file):
        generate_markdown_report(h5_file, results_dir)
