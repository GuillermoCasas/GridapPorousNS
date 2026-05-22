#!/usr/bin/env python3
"""Detect MMS sweep cells whose asymptotic convergence is NOT verified on the two finest meshes.

A cell (config group) is VERIFIED iff the two finest meshes are BOTH true roots AND the
finest-pair L2/H1 slopes match the optimal orders (k+1 and k) within tolerance. Everything else
is FLAGGED for Phase-2 continuation. Detection is post-hoc on the recorded per-mesh residuals
(robust to the noise-floor false-success), with the stored status datasets used only as a
cross-check.

Usage:
  python detect_flagged_cells.py --h5 'results/phase1_*.h5' --config data/phase1_quad_k1.json \
      --out results/flagged_cells.json [--slope-tol 0.25]
"""
import argparse
import glob
import json
import os

import numpy as np

from mms_convergence_lib import (robust_open_h5, is_true_root,
                                  consecutive_slope, load_solver_constants, decode)


def analyze_group(g, gname, kv_attr_name='k_velocity'):
    h = g['h'][:]
    if len(h) == 0:
        return None
    kv = int(g.attrs[kv_attr_name])
    res = g['eval_residuals'][:] if 'eval_residuals' in g else np.full(len(h), np.nan)
    eu = g['err_u_l2'][:]
    euh = g['err_u_h1'][:]
    eps = g['eval_eps'][:] if 'eval_eps' in g else np.full(len(h), np.nan)
    ovs = g['overall_verification_success'][:] if 'overall_verification_success' in g else None
    Ns = np.round(1.0 / h).astype(int)
    order = np.argsort(Ns)  # ascending N
    return dict(kv=kv, N=Ns[order], h=h[order], res=res[order], eu=eu[order],
                euh=euh[order], eps=eps[order],
                ovs=(ovs[order] if ovs is not None else None))


def detect(h5_glob, config_path, slope_tol, out_json):
    c = load_solver_constants(config_path)
    files = sorted(glob.glob(h5_glob))
    if not files:
        print(f"[detect] no h5 files match {h5_glob}")
        return []
    flagged = []
    summary = dict(total=0, verified=0, flagged=0, only_one_mesh=0, disagreements=0)
    for fpath in files:
        with robust_open_h5(fpath, 'r') as f:
            for gname in sorted(f.keys()):
                parts = gname.split('_')
                if len(parts) < 3 or parts[0] != 'config':
                    continue
                a = analyze_group(f[gname], gname)
                if a is None:
                    continue
                g = f[gname]
                method = parts[-1]
                kv = a['kv']
                N, h, res, eu, euh, eps = a['N'], a['h'], a['res'], a['eu'], a['euh'], a['eps']
                tr = np.array([is_true_root(res[i], N[i], kv, c) for i in range(len(N))])
                summary['total'] += 1

                why = []
                if len(N) >= 2:
                    ci, fi = len(N) - 2, len(N) - 1  # two finest present
                else:
                    ci = fi = len(N) - 1
                    summary['only_one_mesh'] += 1
                    why.append("only_one_mesh")
                pair = (int(N[ci]), int(N[fi]))

                if not tr[ci]:
                    why.append(f"not_root@{int(N[ci])}")
                if not tr[fi]:
                    why.append(f"not_root@{int(N[fi])}")
                sL2 = consecutive_slope(h[ci], eu[ci], h[fi], eu[fi])
                sH1 = consecutive_slope(h[ci], euh[ci], h[fi], euh[fi])
                if not (np.isfinite(sL2) and abs(sL2 - (kv + 1)) <= slope_tol):
                    why.append(f"slope_L2={'nan' if not np.isfinite(sL2) else round(sL2,2)}(exp{kv+1})")
                if not (np.isfinite(sH1) and abs(sH1 - kv) <= slope_tol):
                    why.append(f"slope_H1={'nan' if not np.isfinite(sH1) else round(sH1,2)}(exp{kv})")
                valid_eps = eps[np.isfinite(eps)]
                if len(valid_eps) and np.all(valid_eps < 0):
                    why.append("total_failure")

                # cross-check the stored honest flag (warn, don't decide)
                if a['ovs'] is not None and len(a['ovs']) == len(N):
                    stored_ok = bool(a['ovs'][fi] == 1)
                    if stored_ok != bool(tr[fi]):
                        summary['disagreements'] += 1

                tr_idx = np.where(tr)[0]
                finest_tr = (dict(N=int(N[tr_idx[-1]]), err_u_l2=float(eu[tr_idx[-1]]),
                                  err_u_h1=float(euh[tr_idx[-1]]), residual=float(res[tr_idx[-1]]))
                             if len(tr_idx) else None)
                fin = np.where(np.isfinite(eu) & (eu > 0))[0]
                min_err = (dict(N=int(N[fin[-1]]), err_u_l2=float(eu[fin[-1]]),
                                err_u_h1=float(euh[fin[-1]])) if len(fin) else None)

                if len(why) == 0:
                    summary['verified'] += 1
                    continue
                summary['flagged'] += 1
                # Categorize WHY it's flagged — this drives Phase-2 strategy + report status:
                #   total_failure : no true root at any mesh.
                #   fold          : the finest mesh has no true root (a coarser one may) -> continuation
                #                   must find a root at a FINER mesh.
                #   suboptimal_rate: both finest meshes ARE true roots but the slope is off -> the
                #                   solution exists; finer meshes will tell pre-asymptotic from genuine.
                #   incomplete    : only one mesh available.
                if not tr.any():
                    category = "total_failure"
                elif not bool(tr[fi]):
                    category = "fold"
                elif bool(tr[ci]) and bool(tr[fi]):
                    category = "suboptimal_rate"
                elif "only_one_mesh" in why:
                    category = "incomplete"
                else:
                    category = "other"
                summary[category] = summary.get(category, 0) + 1
                flagged.append(dict(
                    h5=os.path.basename(fpath), group=gname, config_idx=int(parts[1]),
                    Re=float(g.attrs['Re']), Da=float(g.attrs['Da']), alpha_0=float(g.attrs['alpha_0']),
                    k_velocity=kv, k_pressure=int(g.attrs['k_pressure']),
                    element_type=decode(g.attrs['element_type']), method=method,
                    category=category, pair_checked=list(pair), why_flagged=why,
                    true_root_meshes=[int(x) for x in N[tr]],
                    finest_true_root=finest_tr, min_error_attained=min_err,
                    slope_L2=(None if not np.isfinite(sL2) else round(sL2, 4)),
                    slope_H1=(None if not np.isfinite(sH1) else round(sH1, 4)),
                ))
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w') as fh:
        json.dump(flagged, fh, indent=2)
    print(f"[detect] {summary}")
    print(f"[detect] wrote {len(flagged)} flagged cells -> {out_json}")
    for fc in flagged:
        print(f"  C{fc['config_idx']:>3} {fc['method']:<4} Re={fc['Re']:.0e} Da={fc['Da']:.0e} "
              f"a={fc['alpha_0']:.2f} k={fc['k_velocity']} {fc['element_type']:<4} "
              f"[{fc['category']:<15}] pair={fc['pair_checked']}: {fc['why_flagged']}")
    return flagged


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default=os.path.join(here, 'results', 'phase1_*.h5'))
    ap.add_argument('--config', default=os.path.join(here, 'data', 'phase1_quad_k1.json'))
    ap.add_argument('--out', default=os.path.join(here, 'results', 'flagged_cells.json'))
    ap.add_argument('--slope-tol', type=float, default=0.25)
    args = ap.parse_args()
    detect(args.h5, args.config, args.slope_tol, args.out)
