#!/usr/bin/env python3
"""Paper-ready merged convergence table: Phase-1 sweep + Phase-2 continuation rescue.

Per (physics cell, method):
  - VERIFIED (no mark): the report mesh (default N=320) is a TRUE root AND the finest-pair slopes
    are optimal. Report its L2/H1 error and the (160,320) slope.
  - `*`  needs-fine-mesh: not verified at N=320, but Phase-2 continuation reached >=2 consecutive
    true roots at finer meshes. Report the finest-true-root error (mesh N noted) and the Phase-2 slope.
  - `**` unverified: a true root was attained somewhere (Phase-1 or a single Phase-2 base) but the
    asymptotic rate was not established. Report the best true-root error (informational).
  - `N/A`: no true root anywhere.

Usage:
  python merged_report.py --h5 'results/phase1_*.h5' --config data/phase1_quad_k1.json \
      --flagged results/flagged_cells.json --phase2 results/phase2_results.json \
      --out results/merged_convergence_report.md [--report-N 320]
"""
import argparse
import glob
import json
import os

import numpy as np

from mms_convergence_lib import (robust_open_h5, is_true_root,
                                  consecutive_slope, load_solver_constants, decode)


def _key(Re, Da, alpha_0, kv, etype):
    return (f"{float(Re):.3e}", f"{float(Da):.3e}", f"{float(alpha_0):.4f}", int(kv), str(etype))


def load_phase2(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = json.load(fh)
    return {_key(r["Re"], r["Da"], r["alpha_0"], r["k_velocity"], r["element_type"]): r for r in rows}


def load_flagged(path):
    """Map (config_idx, method) -> flagged record (for the `category` field)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = json.load(fh)
    return {(int(r["config_idx"]), r["method"]): r for r in rows}


def _slope_ok(s, target, tol):
    return s is not None and np.isfinite(s) and abs(s - target) <= tol


def fmt(x):
    return "---" if (x is None or not np.isfinite(x)) else f"{x:.3e}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default=os.path.join(here, 'results', 'phase1_*.h5'))
    ap.add_argument('--config', default=os.path.join(here, 'data', 'phase1_quad_k1.json'))
    ap.add_argument('--flagged', default=os.path.join(here, 'results', 'flagged_cells.json'))
    ap.add_argument('--phase2', default=os.path.join(here, 'results', 'phase2_results.json'))
    ap.add_argument('--out', default=os.path.join(here, 'results', 'merged_convergence_report.md'))
    ap.add_argument('--report-N', type=int, default=320)
    ap.add_argument('--slope-tol', type=float, default=0.25)
    args = ap.parse_args()

    c = load_solver_constants(args.config)
    phase2 = load_phase2(args.phase2)
    flagged_map = load_flagged(args.flagged)
    rows = []

    for fpath in sorted(glob.glob(args.h5)):
        with robust_open_h5(fpath, 'r') as f:
            for gname in sorted(f.keys()):
                parts = gname.split('_')
                if len(parts) < 3 or parts[0] != 'config':
                    continue
                g = f[gname]
                method = parts[-1]
                h = g['h'][:]
                if len(h) == 0:
                    continue
                kv = int(g.attrs['k_velocity']); kp = int(g.attrs['k_pressure'])
                etype = decode(g.attrs['element_type'])
                Re = float(g.attrs['Re']); Da = float(g.attrs['Da']); a0 = float(g.attrs['alpha_0'])
                res = g['eval_residuals'][:] if 'eval_residuals' in g else np.full(len(h), np.nan)
                eu, ep, euh = g['err_u_l2'][:], g['err_p_l2'][:], g['err_u_h1'][:]
                N = np.round(1.0 / h).astype(int)
                o = np.argsort(N)
                N, h, res, eu, ep, euh = N[o], h[o], res[o], eu[o], ep[o], euh[o]
                tr = np.array([is_true_root(res[i], N[i], kv, c) for i in range(len(N))])

                # report mesh = report-N if present else finest present
                ridx = int(np.where(N == args.report_N)[0][0]) if args.report_N in N else len(N) - 1
                report_N = int(N[ridx])
                # finest-pair slopes (the two finest meshes present)
                if len(N) >= 2:
                    sL2 = consecutive_slope(h[-2], eu[-2], h[-1], eu[-1])
                    sH1 = consecutive_slope(h[-2], euh[-2], h[-1], euh[-1])
                else:
                    sL2 = sH1 = float('nan')

                verified = (tr[ridx] and len(N) >= 2 and tr[-1] and tr[-2]
                            and _slope_ok(sL2, kv + 1, args.slope_tol)
                            and _slope_ok(sH1, kv, args.slope_tol))

                # Robust least-squares rate over the finest up-to-3 TRUE-root Phase-1 meshes
                # (less noisy than a single consecutive pair) — informational alongside sL2/sH1.
                tri = np.where(tr)[0]
                def ls_rate(errs):
                    idx = tri[-3:]
                    if len(idx) < 2 or np.any(errs[idx] <= 0) or np.any(~np.isfinite(errs[idx])):
                        return float('nan')
                    return float(np.polyfit(np.log(h[idx]), np.log(errs[idx]), 1)[0])
                lsL2, lsH1 = ls_rate(eu), ls_rate(euh)

                fl = flagged_map.get((int(parts[1]), method))
                cat = fl.get("category") if fl else None
                p2 = phase2.get(_key(Re, Da, a0, kv, etype))
                tol = args.slope_tol
                row = dict(cidx=int(parts[1]), Re=Re, Da=Da, a0=a0, k=kv, etype=etype, method=method,
                           lsL2=lsL2, lsH1=lsH1)

                if verified:
                    row.update(status="verified", mark="", N_rep=report_N,
                               l2u=float(eu[ridx]), l2p=float(ep[ridx]), h1u=float(euh[ridx]), sL2=sL2, sH1=sH1)
                elif p2 and p2.get("status") == "rescued":
                    ft = p2["finest_true_root"]; pL2 = p2.get("slope_L2"); pH1 = p2.get("slope_H1")
                    opt = _slope_ok(pL2, kv + 1, tol) and _slope_ok(pH1, kv, tol)
                    # Phase-2 reached finer meshes: optimal there ⇒ was pre-asymptotic/fold (now verified);
                    # still off ⇒ GENUINE sub-optimal rate even at the finest mesh.
                    row.update(status=("verified@fine-mesh" if opt else "SUBOPTIMAL-RATE@fine"),
                               mark=("*" if opt else "‡"), N_rep=int(ft["N"]),
                               l2u=ft["l2u"], l2p=ft.get("l2p"), h1u=ft["h1u"], sL2=pL2, sH1=pH1)
                elif cat == "suboptimal_rate":
                    # Root EXISTS at the report mesh (not a fold); the issue is the rate. Report the
                    # report-mesh error + the finest-pair (sub-optimal) rate. Prominent ‡ mark.
                    row.update(status="SUBOPTIMAL-RATE", mark="‡", N_rep=report_N,
                               l2u=float(eu[ridx]), l2p=float(ep[ridx]), h1u=float(euh[ridx]), sL2=sL2, sH1=sH1)
                elif p2 and p2.get("status") == "base_only":
                    ft = p2["finest_true_root"]
                    row.update(status="best-root(1 mesh)", mark="**", N_rep=int(ft["N"]),
                               l2u=ft["l2u"], l2p=ft.get("l2p"), h1u=ft["h1u"], sL2=float('nan'), sH1=float('nan'))
                elif len(tri):
                    j = int(tri[-1])
                    row.update(status="unverified(fold)", mark="**", N_rep=int(N[j]),
                               l2u=float(eu[j]), l2p=float(ep[j]), h1u=float(euh[j]), sL2=float('nan'), sH1=float('nan'))
                else:
                    row.update(status="no-root", mark="", N_rep=report_N,
                               l2u=None, l2p=None, h1u=None, sL2=float('nan'), sH1=float('nan'))
                rows.append(row)

    rows.sort(key=lambda r: (r['etype'], r['k'], r['cidx'], r['method']))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    def rfmt(s):
        return "---" if (s is None or not np.isfinite(s)) else f"{s:.2f}"
    with open(args.out, 'w') as io:
        io.write("# Merged MMS convergence table (Phase-1 sweep + Phase-2 continuation)\n\n")
        io.write("Marks: *(none)*=verified (true root + optimal slope, k+1 / k); ")
        io.write("`*`=true root recovered by continuation at the noted finer mesh, slope then optimal; ")
        io.write("`‡`=**converges but SUB-OPTIMAL rate** (true root exists; slope stays below k+1/k even at the finest mesh — a genuine finding, not a fold); ")
        io.write("`**`=best true root attained, asymptotics not established; `N/A`=no true root anywhere.\n\n")
        io.write("rate_* = slope on the finest available mesh PAIR; lsq_* = least-squares slope over the finest ≤3 true-root meshes.\n\n")
        io.write("| Config | Re | Da | α₀ | k | Elem | Method | Status | N_rep | L2_u | L2_p | H1_u | rate_L2u | rate_H1u | lsq_L2u | lsq_H1u |\n")
        io.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            l2u = (fmt(r['l2u']) + r['mark']) if r['l2u'] is not None else "N/A"
            io.write("| C{} | {:.0e} | {:.0e} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                r['cidx'], r['Re'], r['Da'], r['a0'], r['k'], r['etype'], r['method'], r['status'],
                r['N_rep'], l2u, fmt(r['l2p']), fmt(r['h1u']),
                rfmt(r['sL2']), rfmt(r['sH1']), rfmt(r.get('lsL2')), rfmt(r.get('lsH1'))))
    n_ver = sum(1 for r in rows if r['status'] == 'verified')
    n_star = sum(1 for r in rows if r['mark'] == '*')
    n_subopt = sum(1 for r in rows if r['mark'] == '‡')
    n_ss = sum(1 for r in rows if r['mark'] == '**')
    n_na = sum(1 for r in rows if r['status'] == 'no-root')
    print(f"[merged] sub-optimal-rate(‡): {n_subopt}")
    print(f"[merged] {len(rows)} rows: {n_ver} verified, {n_star} needs-fine-mesh(*), "
          f"{n_ss} best-root(**), {n_na} no-root. -> {args.out}")


if __name__ == '__main__':
    main()
