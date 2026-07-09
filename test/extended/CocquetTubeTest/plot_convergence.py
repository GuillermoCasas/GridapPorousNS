# test/extended/CocquetTubeTest/plot_convergence.py
#
# Consolidated convergence plotter for the unified Cocquet tube-flow test. Every variant writes
# results/<name>/convergence.h5 with the SAME schema, so one plotter serves them all.
#
# Usage:
#   python plot_convergence.py structured            # plots results/structured/convergence.h5
#   python plot_convergence.py unstructured_gmsh     # any variant name under results/
#   python plot_convergence.py path/to/convergence.h5
#
# Plots every (method / P<kv>P<kp>) group present in the file — equal-order P1/P1, P2/P2 and
# mixed Taylor-Hood P2/P1 alike — so it never silently produces an empty figure. The S3 / corner-
# excluded probes are drawn on the third axis when present (they always are, for this driver).
import sys
import os
import re
import h5py
import matplotlib.pyplot as plt
import numpy as np


def _resolve_h5(arg):
    """Accept a variant name (-> results/<name>/convergence.h5) or a direct path."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if arg is None:
        arg = 'structured'
    if os.path.isabs(arg) or os.path.exists(arg):
        return arg
    if arg.endswith('.h5'):
        return os.path.join(results_dir, arg)
    # bare variant name
    return os.path.join(results_dir, arg, 'convergence.h5')


def _discover_groups(file):
    """Yield (method, pair_name, kv, kp, group) for every P<kv>P<kp> leaf group."""
    pat = re.compile(r'^P(\d+)P(\d+)$')
    for method in file.keys():
        node = file[method]
        if not isinstance(node, h5py.Group):
            continue
        for pair_name in node.keys():
            m = pat.match(pair_name)
            if m and isinstance(node[pair_name], h5py.Group):
                yield method, pair_name, int(m.group(1)), int(m.group(2)), node[pair_name]


def plot_cocquet(h5_arg=None):
    h5_file = _resolve_h5(h5_arg)
    if not os.path.exists(h5_file):
        print(f"HDF5 file not found: {h5_file}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))
    markers = ['^', 's', 'o', 'D', 'v', '*']
    plotted = 0
    diag_present = False
    flagged_unconverged = False   # [2b] any non-converged-but-finite mesh marked with an x

    with h5py.File(h5_file, 'r') as file:
        if "N_list" not in file:
            print(f"'N_list' missing from {h5_file}; nothing to plot.")
            return
        N_list = np.array(file["N_list"], dtype=float)
        h = 1.0 / N_list

        groups = list(_discover_groups(file))
        if not groups:
            print(f"No P<kv>P<kp> groups found in {h5_file}. Top-level keys: {list(file.keys())}")
            return

        # Physical scales / variant identity for the title are read from the file (no hard-coding).
        Re = file.attrs.get("Re")
        c_in = file.attrs.get("c_in")
        delta = file.attrs.get("outlet_truncation_delta")
        variant = file.attrs.get("config_file")
        midx = 0
        for method, pair_name, kv, kp, group in sorted(groups, key=lambda t: (t[1], t[0])):
            if delta is None and "outlet_truncation_delta" in group.attrs:
                delta = group.attrs["outlet_truncation_delta"]

            err_u = np.array(group["errors_l2_u"]); err_p = np.array(group["errors_l2_p"])
            err_u_h1 = np.array(group["errors_h1_u"]); err_p_h1 = np.array(group["errors_h1_p"])
            # [2b] per-mesh converged flag (back-compat: absent -> all True). "success" here = the
            # nonlinear SOLVE converged (the tube uses a reference-solution error metric, not an MMS root).
            succ = (np.array(group["mesh_success"], dtype=bool) if "mesh_success" in group
                    else np.ones(len(N_list), dtype=bool))

            marker = markers[midx % len(markers)]; midx += 1
            # Distinct line style per method: ASGS solid, OSGS dashed, Galerkin* (Cocquet) dotted.
            ls = '-' if method == "ASGS" else (':' if method.startswith("Galerkin") else '--')

            opt_u_l2, opt_u_h1 = kv + 1, kv
            opt_p_l2, opt_p_h1 = kp + 1, kp
            tag = f"Cocquet(Galerkin) P{kv}/P{kp}" if method.startswith("Galerkin") else f"{method} P{kv}/P{kp}"

            ax1.loglog(N_list, err_u, color='blue', marker=marker, linestyle=ls, lw=2, ms=8,
                       label=fr'{tag} $L_2$ Vel (opt {opt_u_l2})')
            ax1.loglog(N_list, err_p, color='red', marker=marker, linestyle=ls, lw=2, ms=8,
                       markerfacecolor='white', label=fr'{tag} $L_2$ Pre (opt {opt_p_l2})')
            ax2.loglog(N_list, err_u_h1, color='blue', marker=marker, linestyle=ls, lw=2, ms=8,
                       label=fr'{tag} $H_1$ Vel (opt {opt_u_h1})')
            ax2.loglog(N_list, err_p_h1, color='red', marker=marker, linestyle=ls, lw=2, ms=8,
                       markerfacecolor='white', label=fr'{tag} $H_1$ Pre (opt {opt_p_h1})')

            def annotate(ax, err, color, dy):
                for i in range(len(N_list) - 1):
                    # [2b] never fit a rate across / into a non-converged or non-finite mesh
                    if not (succ[i] and succ[i+1] and np.isfinite(err[i]) and np.isfinite(err[i+1])
                            and err[i] > 0 and err[i+1] > 0):
                        continue
                    raw = (np.log(err[i+1]) - np.log(err[i])) / (np.log(h[i+1]) - np.log(h[i]))
                    Nm = np.exp(0.5 * (np.log(N_list[i]) + np.log(N_list[i+1])))
                    em = np.exp(0.5 * (np.log(err[i]) + np.log(err[i+1])))
                    ax.annotate(f'{raw:.2f}', xy=(Nm, em), xytext=(0, dy), textcoords='offset points',
                                color=color, fontsize=8, fontweight='bold', ha='center',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6))
            annotate(ax1, err_u, 'blue', 7)
            annotate(ax1, err_p, 'red', -11)
            annotate(ax2, err_u_h1, 'blue', 7)
            annotate(ax2, err_p_h1, 'red', -11)
            # [2b] mark non-converged-but-finite meshes with a distinct x (kept + flagged, not dropped)
            for ax, err, col in ((ax1, err_u, 'blue'), (ax1, err_p, 'red'),
                                 (ax2, err_u_h1, 'blue'), (ax2, err_p_h1, 'red')):
                fin = np.isfinite(err) & (err > 0)
                bad = fin & ~succ
                if np.any(bad):
                    ax.loglog(N_list[bad], err[bad], marker='x', linestyle='none', color=col,
                              ms=13, markeredgewidth=2.5, zorder=6)
                    flagged_unconverged = True
            plotted += 1

            # S3 / corner-excluded probes on ax3 (present for every run of this driver).
            if "chi_Omega_u" in group and "l2_eu_corner_excl" in group:
                diag_present = True
                chi_u = np.array(group["chi_Omega_u"])
                chi_p = np.array(group["chi_Omega_p"])
                radii = np.array(group.attrs.get("corner_excl_radii", [0.05, 0.1, 0.2]))
                r_idx_plot = int(np.argmin(np.abs(radii - 0.1)))
                R_plot = radii[r_idx_plot]
                l2_u_excl = np.array(group["l2_eu_corner_excl"])[r_idx_plot, :]
                ax3.plot(N_list, chi_u, color='blue', marker=marker, linestyle=ls, lw=2, ms=8,
                         label=fr'{tag} $\chi_\Omega$ (u)')
                ax3.plot(N_list, chi_p, color='red',  marker=marker, linestyle=ls, lw=2, ms=8,
                         markerfacecolor='white', label=fr'{tag} $\chi_\Omega$ (p)')
                if not hasattr(ax3, '_twin'):
                    ax3._twin = ax3.twinx()
                    ax3._twin.set_yscale('log')
                    ax3._twin.set_ylabel(rf'$\|e_h\|_{{L^2(\Omega\setminus B_{{{R_plot:g}}})}}$ (log)')
                ax3._twin.plot(N_list, l2_u_excl, color='green', marker=marker, linestyle=ls, lw=2, ms=8,
                               markerfacecolor='white', label=fr'{tag} $L_2(u)\,/\,B_{{{R_plot:g}}}$ excl.')

    for ax, ttl, yl in ((ax1, r'$L_2$-norms', r'$L_2$-norm error'),
                        (ax2, r'$H_1$-seminorms', r'$H_1$-seminorm error')):
        ax.set_xlabel(r'$N$'); ax.set_ylabel(yl); ax.set_title(f'Spatial convergence: {ttl}')
        if flagged_unconverged:   # [2b] legend note for the x markers
            ax.plot([], [], marker='x', linestyle='none', color='0.35', ms=11, markeredgewidth=2.5,
                    label='not fully converged (excluded from rate)')
        ax.grid(True, which="both", ls="--"); ax.legend(handlelength=3.0, fontsize=8)

    if diag_present:
        ax3.set_xscale('log')
        ax3.set_xlabel(r'$N$')
        ax3.set_ylabel(r'domain-mean share $\chi_\Omega = |\Omega|\,\|\bar e_\Omega\|^2/\|e_h\|^2$')
        ax3.set_title(r'Magnitude-gap probes: $\chi_\Omega$ (left, linear) and corner-excluded $L^2(u)$ (right, log)')
        ax3.set_ylim(0.0, 1.05)
        ax3.grid(True, which='both', ls='--')
        h1_h, h1_l = ax3.get_legend_handles_labels()
        h2_h, h2_l = ax3._twin.get_legend_handles_labels() if hasattr(ax3, '_twin') else ([], [])
        ax3.legend(h1_h + h2_h, h1_l + h2_l, handlelength=3.0, fontsize=8, loc='upper right')
    else:
        ax3.set_visible(False)

    # Variant name = the results subfolder holding this h5.
    name = os.path.basename(os.path.dirname(h5_file))
    parts = [name]
    if variant is not None:
        parts.append(variant.decode() if isinstance(variant, bytes) else str(variant))
    if Re is not None:
        parts.append(rf'Re={Re:g}')
    if c_in is not None:
        parts.append(rf'c_in={c_in:g}')
    if delta is not None:
        parts.append(rf'$\delta=${delta:g}')
    plt.suptitle(rf'Cocquet Tube-Flow Convergence ({", ".join(parts)})', fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_file = os.path.splitext(h5_file)[0] + '.png'
    plt.savefig(plot_file, dpi=200)
    print(f"Plotted {plotted} group(s); saved to {plot_file}")


if __name__ == "__main__":
    plot_cocquet(sys.argv[1] if len(sys.argv) > 1 else None)
