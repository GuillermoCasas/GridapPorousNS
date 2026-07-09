#!/usr/bin/env python3
"""Combined convergence plots ACROSS MULTIPLE h5 files, grouped per physics cell.

Overlays every method-family (P1/P1 & P2/P2 ASGS & OSGS, P2/P1 Taylor-Hood/Galerkin)
for a given (Re, Da, alpha_0) cell, reading several h5 files at once (vms + taylorhood).

Convention:
  ELEMENT -> marker :  k=1 (P1/P1) = round 'o' ,  k=2 (P2/P2) = square 's' ,  Taylor-Hood (P2/P1) = cross 'X'.
  METHOD  -> colour + line style :  ASGS = blue/solid ,  OSGS = red/dashed ,  Taylor-Hood (Galerkin) = black/dotted.
  FIELD   -> velocity = FILLED marker ,  pressure = HOLLOW marker.
  Two plots per cell:  L^2 norms (…_L2.png)  and  H^1 seminorms (…_H1.png).

Usage:
  python plot_combined_all.py                       # defaults to vms + taylorhood
  python plot_combined_all.py --h5 results/a.h5 results/b.h5 --out-dir results/combined
"""
import argparse, os, glob, math
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# METHOD -> (colour, linestyle):  ASGS blue/solid, OSGS red/dashed, Taylor-Hood (Galerkin) black/dotted.
def _method_style(method):
    m = method.upper()
    if m == 'ASGS': return 'blue', '-'
    if m == 'OSGS': return 'red',  '--'
    return 'black', ':'

# ELEMENT -> marker:  P1/P1 round, P2/P2 square, Taylor-Hood (P2/P1) cross.
def _elem_marker(kv, kp, method):
    if method.upper() == 'GALERKIN' or (kv, kp) == (2, 1): return 'X'
    if kv == 1: return 'o'
    if kv == 2: return 's'
    return 'P'

# the two norm groups: (h5 key, axis label, velocity?/pressure?, expected order f(kv,kp))
GROUPS = {
    'L2': [('err_u_l2', r'$L^2$ velocity', True,  lambda kv, kp: kv + 1),
           ('err_p_l2', r'$L^2$ pressure', False, lambda kv, kp: min(kv, kp + 1))],
    'H1': [('err_u_h1', r'$H^1$ velocity', True,  lambda kv, kp: kv),
           ('err_p_h1', r'$H^1$ pressure', False, lambda kv, kp: max(kp, 1))],
}

def gather(files):
    cells = {}   # (Re,Da,a0) -> list of bundles
    for fn in files:
        if not os.path.exists(fn): continue
        with h5py.File(fn, 'r') as f:
            for g in f.keys():
                a = f[g].attrs
                kv, kp = int(a['k_velocity']), int(a['k_pressure'])
                method = a['method'].decode() if isinstance(a['method'], bytes) else str(a['method'])
                key = (round(float(a['Re']), 12), round(float(a['Da']), 12), round(float(a['alpha_0']), 12))
                b = dict(kv=kv, kp=kp, method=method, label=f"P{kv}/P{kp} {method}", h=f[g]['h'][:])
                for grp in GROUPS.values():
                    for fkey, *_ in grp:
                        b[fkey] = f[g][fkey][:] if fkey in f[g] else np.full_like(b['h'], np.nan)
                b['fold'] = f[g]['fold'][:] if 'fold' in f[g] else np.zeros_like(b['h'], dtype=bool)
                cells.setdefault(key, []).append(b)
    return cells

def plot_group(key, bundles, grp_name, out_dir):
    Re, Da, a0 = key
    fields = GROUPS[grp_name]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    morder = {'ASGS': 0, 'OSGS': 1, 'GALERKIN': 2}
    bundles = sorted(bundles, key=lambda b: (morder.get(b['method'].upper(), 9), b['kv'], b['kp']))
    drew = False
    any_unconverged = False   # [2b] any fold-flagged (unconverged-but-finite) point marked
    for b in bundles:
        color, ls = _method_style(b['method'])
        marker = _elem_marker(b['kv'], b['kp'], b['method'])
        h = np.asarray(b['h'])
        foldv = np.asarray(b.get('fold', np.zeros(len(h), dtype=bool)), dtype=bool)
        for fkey, flabel, is_vel, ford in fields:
            y = np.asarray(b[fkey])
            m = np.isfinite(y) & (y > 0) & np.isfinite(h) & (h > 0)
            if m.sum() < 2: continue
            hm, ym, cm = h[m], y[m], (~foldv)[m]      # cm[i]=True ⇒ CONVERGED (not fold-flagged)
            o = np.argsort(-hm); hm, ym, cm = hm[o], ym[o], cm[o]
            mfc = color if is_vel else 'white'        # velocity filled, pressure hollow
            kref = ford(b['kv'], b['kp'])
            ax.loglog(hm, ym, marker=marker, linestyle=ls, color=color, lw=2.0, ms=8,
                      markerfacecolor=mfc, markeredgecolor=color,
                      label=f"{b['label']} · {flabel} ($h^{{{kref}}}$)")
            drew = True
            # [2b] mark fold-flagged (unconverged-but-finite) points with a distinct ✕; never fit a
            # rate across / into one. Mirrors ManufacturedSolutions3D/plot_convergence3d.py.
            bad = ~cm
            if bad.any():
                ax.loglog(hm[bad], ym[bad], marker='x', linestyle='none', color=color, ms=13,
                          markeredgewidth=2.5, zorder=6)
                any_unconverged = True
            for i in range(len(hm) - 1):              # observed per-segment rate (converged pairs only)
                if not (cm[i] and cm[i + 1]):
                    continue
                xa = math.sqrt(hm[i] * hm[i + 1]); ya = math.sqrt(ym[i] * ym[i + 1])
                sv = math.log(ym[i + 1] / ym[i]) / math.log(hm[i + 1] / hm[i])
                ax.annotate(f'{sv:.2f}', xy=(xa, ya), xytext=(0, 7), textcoords='offset points',
                            ha='center', fontsize=6.5, color=color)
    if not drew:
        plt.close(fig); return
    if any_unconverged:                           # [2b] legend note for the ✕ (non-converged) markers
        ax.plot([], [], marker='x', linestyle='none', color='0.35', ms=11, markeredgewidth=2.5,
                label='not fully converged (excluded from rate)')
    folded = any(bool(np.any(b.get('fold', False))) for b in bundles)
    norm_lbl = r'$L^2$ norms' if grp_name == 'L2' else r'$H^1$ seminorms'
    ttl = rf"Cocquet-form MMS, {norm_lbl} — $Re={Re:g}$, $Da={Da:g}$, $\alpha_0={a0:g}$"
    if folded: ttl += "   [some VMS cells FOLD]"
    ax.set_xlabel(r'Mesh size $h$'); ax.set_ylabel('Error')
    ax.set_title(ttl, fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"combined_Re{Re:g}_Da{Da:g}_a{a0:g}_{grp_name}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[combined] {out}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5', nargs='+', default=[
        os.path.join(HERE, 'results', 'cocquet_form_mms_vms.h5'),
        os.path.join(HERE, 'results', 'cocquet_form_mms_vms_k2.h5'),
        os.path.join(HERE, 'results', 'cocquet_form_mms_taylorhood.h5')])
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'results', 'combined'))
    args = ap.parse_args()
    files = []
    for p in args.h5:
        files += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    if not files:
        raise SystemExit(f"No h5 files found: {args.h5}")
    print(f"[combined] reading: {[os.path.basename(f) for f in files]}")
    cells = gather(files)
    for key in sorted(cells):
        for grp in GROUPS:
            plot_group(key, cells[key], grp, args.out_dir)

if __name__ == '__main__':
    main()
