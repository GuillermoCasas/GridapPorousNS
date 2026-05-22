# test/extended/CocquetDeviatoric/plot_convergence.py
#
# Usage:
#   python plot_convergence.py                          # plots results/convergence_deviatoric.h5
#   python plot_convergence.py some_other.h5            # any file under results/ (or an absolute path)
#
# Plots every (method / P<kv>P<kp>) group present in the file — equal-order P1/P1, P2/P2 and
# mixed Taylor-Hood P2/P1 alike — so it never silently produces an empty figure when the group
# names don't match a hard-coded list.
import sys
import os
import re
import h5py
import matplotlib.pyplot as plt
import numpy as np


def _resolve_h5(arg):
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if arg is None:
        return os.path.join(results_dir, 'convergence_deviatoric.h5')
    if os.path.isabs(arg) or os.path.exists(arg):
        return arg
    # bare name (with or without .h5) resolved under results/
    if not arg.endswith('.h5'):
        arg += '.h5'
    return os.path.join(results_dir, arg)


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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
    markers = ['^', 's', 'o', 'D', 'v', '*']
    plotted = 0

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

        # Physical scales for the title are read from the file (fall back to file-level then
        # per-group attrs for older files); nothing is hard-coded here.
        Re = file.attrs.get("Re")
        c_in = file.attrs.get("c_in")
        delta = file.attrs.get("outlet_truncation_delta")
        midx = 0
        for method, pair_name, kv, kp, group in sorted(groups, key=lambda t: (t[1], t[0])):
            if delta is None and "outlet_truncation_delta" in group.attrs:
                delta = group.attrs["outlet_truncation_delta"]

            err_u = np.array(group["errors_l2_u"]); err_p = np.array(group["errors_l2_p"])
            err_u_h1 = np.array(group["errors_h1_u"]); err_p_h1 = np.array(group["errors_h1_p"])

            marker = markers[midx % len(markers)]; midx += 1
            ls = '-' if method == "ASGS" else '--'

            # Optimal rates: velocity ~ (kv+1) in L2, kv in H1; pressure ~ (kp+1) in L2, kp in H1.
            opt_u_l2, opt_u_h1 = kv + 1, kv
            opt_p_l2, opt_p_h1 = kp + 1, kp
            tag = f"{method} P{kv}/P{kp}"

            ax1.loglog(N_list, err_u, color='blue', marker=marker, linestyle=ls, lw=2, ms=8,
                       label=fr'{tag} $L_2$ Vel (opt {opt_u_l2})')
            ax1.loglog(N_list, err_p, color='red', marker=marker, linestyle=ls, lw=2, ms=8,
                       markerfacecolor='white', label=fr'{tag} $L_2$ Pre (opt {opt_p_l2})')
            ax2.loglog(N_list, err_u_h1, color='blue', marker=marker, linestyle=ls, lw=2, ms=8,
                       label=fr'{tag} $H_1$ Vel (opt {opt_u_h1})')
            ax2.loglog(N_list, err_p_h1, color='red', marker=marker, linestyle=ls, lw=2, ms=8,
                       markerfacecolor='white', label=fr'{tag} $H_1$ Pre (opt {opt_p_h1})')

            # Raw local slopes (d log err / d log h), annotated at every segment midpoint,
            # on every curve. Velocity labels sit above the line (blue), pressure below (red).
            def annotate(ax, err, color, dy):
                for i in range(len(N_list) - 1):
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
            plotted += 1

    for ax, ttl, yl in ((ax1, r'$L_2$-norms', r'$L_2$-norm error'),
                        (ax2, r'$H_1$-seminorms', r'$H_1$-seminorm error')):
        ax.set_xlabel(r'$N$'); ax.set_ylabel(yl); ax.set_title(f'Spatial convergence: {ttl}')
        ax.grid(True, which="both", ls="--"); ax.legend(handlelength=3.0, fontsize=8)

    parts = [os.path.basename(h5_file)]
    if Re is not None:
        parts.append(rf'Re={Re:g}')
    if c_in is not None:
        parts.append(rf'c_in={c_in:g}')
    if delta is not None:
        parts.append(rf'$\delta=${delta:g}')
    plt.suptitle(rf'Cocquet Convergence ({", ".join(parts)})', fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_file = os.path.splitext(h5_file)[0] + '.png'
    plt.savefig(plot_file, dpi=200)
    print(f"Plotted {plotted} group(s); saved to {plot_file}")


if __name__ == "__main__":
    plot_cocquet(sys.argv[1] if len(sys.argv) > 1 else None)
