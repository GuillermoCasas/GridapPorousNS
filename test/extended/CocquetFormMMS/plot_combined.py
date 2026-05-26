#!/usr/bin/env python3
"""Combined convergence plot — all formulations overlaid on one figure.

Reads an h5 produced by `run_test.jl` (config_<idx>_<METHOD> groups, same schema as
the ManufacturedSolutions sweeps) and writes a single PNG with three side-by-side
log–log panels: L²_u, H¹_u, L²_p. Each formulation is a separate curve, distinguished
by colour. Reference slope lines (h^{k+1} for L²_u and L²_p, h^k for H¹_u) are
overlaid in light grey for visual comparison.

Default target: results/cocquet_form_mms_comparison.h5 → results/combined_comparison.png.

Usage:
  python plot_combined.py [--h5 <path>] [--out <png>]
"""
import argparse
import os
import glob

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _label_for(attrs):
    method = attrs['method'].decode() if isinstance(attrs['method'], bytes) else str(attrs['method'])
    kv = int(attrs['k_velocity']); kp = int(attrs['k_pressure'])
    return f"P{kv}/P{kp} {method}"


def process_file(h5_path, out_path=None):
    with h5py.File(h5_path, 'r') as f:
        groups = sorted(f.keys())
        if not groups:
            print(f"Warning: no groups in {h5_path}, skipping")
            return
        # gather one bundle per group
        bundles = []
        for g in groups:
            grp = f[g]
            a = grp.attrs
            bundles.append(dict(
                label=_label_for(a),
                kv=int(a['k_velocity']),
                kp=int(a['k_pressure']),
                Re=float(a['Re']), Da=float(a['Da']), a0=float(a['alpha_0']),
                h=grp['h'][:],
                eu_l2=grp['err_u_l2'][:],
                eu_h1=grp['err_u_h1'][:],
                ep_l2=grp['err_p_l2'][:],
                rate_l2u=float(a.get('rate_u_l2', np.nan)),
                rate_h1u=float(a.get('rate_u_h1', np.nan)),
                rate_l2p=float(a.get('rate_p_l2', np.nan)),
            ))

    # One physics point per figure title
    Re, Da, a0 = bundles[0]['Re'], bundles[0]['Da'], bundles[0]['a0']
    
    if not out_path:
        base = os.path.basename(h5_path)
        base_no_ext = os.path.splitext(base)[0]
        out_name = base_no_ext.replace('cocquet_form_mms', 'combined') + '.png'
        out_path = os.path.join(HERE, 'results', out_name)

    fig, axes = plt.subplots(3, 1, figsize=(6, 15), sharex=True)
    # Reference rates (textbook, smooth-solution case):
    #   u_L2 ~ h^{kv+1}  (Aubin–Nitsche)
    #   u_H1 ~ h^{kv}
    #   p_L2 ~ h^{min(kv, kp+1)}   ← TH and stabilized equal-order share the same rule:
    #                                stabilized Pₖ/Pₖ gives h^{kp}; TH Pₖ/Pₖ₋₁ gives h^{kp+1}=h^{kv}.
    norm_specs = [
        ('eu_l2', r'$\| u - u_h \|_{L^2}$',  lambda kv, kp: kv + 1,           'Velocity $L^2$'),
        ('eu_h1', r'$\| u - u_h \|_{H^1}$',  lambda kv, kp: kv,               'Velocity $H^1$'),
        ('ep_l2', r'$\| p - p_h \|_{L^2}$',  lambda kv, kp: min(kv, kp + 1),  'Pressure $L^2$'),
    ]

    for ax, (key, ylabel, expected_order, short) in zip(axes, norm_specs):
        for bundle in bundles:
            h = bundle['h']
            y = bundle[key]
            mask = np.isfinite(y) & (y > 0)
            if not mask.any():
                continue
                
            h_m = h[mask]
            y_m = y[mask]
            
            method = bundle['label'].split()[-1]
            ls = '-' if method == 'ASGS' else '--'
            kref = expected_order(bundle['kv'], bundle['kp'])
            
            label = f"{bundle['label']} ({kref})"
            
            kv, kp = bundle['kv'], bundle['kp']
            if kv == 1 and kp == 1:
                marker = '^'
                color = 'blue'
            elif kv == 2 and kp == 1:
                marker = 'D'
                color = 'black'
            elif kv == 2 and kp == 2:
                marker = 's'
                color = 'red'
            else:
                marker = 'o'
                color = 'gray'
            
            ax.loglog(h_m, y_m, marker=marker, linestyle=ls, linewidth=2, color=color,
                      markersize=8, label=label)
            
            # Segment slope labels
            for i in range(len(h_m) - 1):
                hm = np.sqrt(h_m[i] * h_m[i + 1])
                em = np.sqrt(y_m[i] * y_m[i + 1])
                sv = (np.log(y_m[i + 1]) - np.log(y_m[i])) / (np.log(h_m[i + 1]) - np.log(h_m[i]))
                ax.annotate(f'{(sv/kref if kref else float("nan")):.2f}', xy=(hm, em),
                            xytext=(0, 10 if method == 'ASGS' else -12),
                            textcoords='offset points', ha='center', fontsize=8,
                            color=color, fontweight='bold')
                            
        ax.set_ylabel(ylabel)
        ax.set_title(short)
        ax.grid(True, which='both', linestyle='--')
        ax.legend(fontsize=8, loc='best')

    axes[-1].set_xlabel(r"$h$")
    fig.suptitle(rf"$Re={Re:g}$, $Da={Da:g}$, $\alpha_0={a0:g}$", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[combined] -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5', default=None, help='Input H5 file or pattern (defaults to results/cocquet_form_mms_comparison*.h5)')
    ap.add_argument('--out', default=None, help='Output path (only used if a single h5 file is processed)')
    args = ap.parse_args()

    if args.h5:
        h5_files = glob.glob(args.h5)
        if not h5_files:
            # Try as exact path
            if os.path.exists(args.h5):
                h5_files = [args.h5]
            else:
                raise SystemExit(f"No files matched pattern: {args.h5}")
    else:
        pattern = os.path.join(HERE, 'results', 'cocquet_form_mms_comparison*.h5')
        h5_files = glob.glob(pattern)
        if not h5_files:
            # Fallback to default file
            fallback = os.path.join(HERE, 'results', 'cocquet_form_mms_comparison.h5')
            if os.path.exists(fallback):
                h5_files = [fallback]
            else:
                raise SystemExit(f"No H5 files found matching: {pattern}")

    # Process all matched files
    for h5_file in sorted(h5_files):
        # Only pass args.out if we have a single file (to avoid overwriting the same output file multiple times)
        out_path = args.out if len(h5_files) == 1 else None
        process_file(h5_file, out_path)


if __name__ == '__main__':
    main()
