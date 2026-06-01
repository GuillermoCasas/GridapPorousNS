#!/usr/bin/env python3
"""analyze_results.py — single entry point for ALL MMS-sweep analysis & reporting.

Supersedes (and folds together) the former plot_results.py + merged_report.py +
detect_flagged_cells.py + mms_convergence_lib.py. One run produces:

  1. TRUE-ROOT DETECTION + categorization  -> flagged_cells.json  (input to the Phase-2 Julia
     continuation step). A mesh counts as a true root only if its recorded ‖R‖ ≤ k_nf·dynamic_ftol
     (the honest-exit gate), NOT the solver's own — noise-floor-foolable — "converged" flag.
  2. ANNOTATED MERGED TABLE  -> merged_convergence_report.md  (the paper table). Joins Phase-2
     rescue results. Marks per (cell, method):
        (none) verified  | `*` recovered at a finer mesh by continuation, then optimal
        `‡` genuine sub-optimal rate (true root, slope stays below k+1/k even at the finest mesh)
        `ˢ` velocity-L² rate ABOVE nominal k+1 (super-convergent) — not a failure, but can signal
            the asymptotic regime is not yet established; confirm with one further refinement
        `**` best true root attained, asymptotics not established (fold) | `N/A` no true root
  3. PER-CONFIG CONVERGENCE PLOTS  -> convergence_config*.png  (log-log, slope annotations).
  4. DETAILED PER-CONFIG TABLE  -> convergence_report.md  (rates / FME / time / iters, with an
     HONEST true-root "Converged" column) + fixed-width summary_tables.txt.

Acceptance is ONE-SIDED: a slope at or above the optimal order passes (super-convergence is not a
failure); only a slope below (target − tol) is sub-optimal.

Usage (run after a Phase-1 sweep; re-run after Phase-2 to fold in rescues):
  python analyze_results.py --h5 'results/phase1_quad_k1.h5' --config data/phase1_quad_k1.json \
      [--phase2 results/phase2_quad_k1.json] [--report-N 320] [--outdir results] [--no-plots]
"""
import argparse
import collections
import glob
import json
import os
import time

import numpy as np

try:                                            # plots are optional — the tables never depend on them
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# ======================================================================================
# Shared helpers (folded from mms_convergence_lib.py)
# ======================================================================================
def robust_open_h5(filepath, mode='r', retries=10, delay=2.0):
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    import h5py
    for attempt in range(retries):
        try:
            return h5py.File(filepath, mode, swmr=True)
        except (OSError, RuntimeError):
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


# base_config.json fallbacks (used only if a phase-1 config omits a field; the phase-1 configs
# set ftol/ceiling/safety/k_nf explicitly, so these rarely apply).
# `relative_residual_tol` and `initial_residual_floor` parameterise the pure-relative gate
# `‖R_final‖ / ‖R_initial‖ ≤ τ_rel` that replaces the unscaled-residual gate when iter-0
# residuals are recorded (sweeps from 2026-05-31 onward). The relative form is the
# canonical Newton-convergence test from Dennis–Schnabel / Kelley — it discriminates between
# "scale-correct deep convergence" (which any Re=10⁶ cell achieves trivially with a
# 15-order relative reduction) and "stall at the algebraic noise floor" (where Newton drops
# the residual by only a few orders before saturating, as in the C7 N=320 case).
# The floor handles the degenerate `‖R₀‖ → 0` case (perfect IC, trivial problem, or re-solve
# from a converged state): below the floor we fall back to the absolute gate so the relative
# test does not divide-by-noise.
_BASE = dict(ftol=1e-8, dynamic_ftol_ceiling=1e-4, dynamic_ftol_spatial_safety_factor=1e-2,
             noise_floor_success_max_ftol_multiple=1e30,
             relative_residual_tol=1e-6,
             initial_residual_floor=1e-13)


def load_solver_constants(config_path):
    """Read the dynamic_ftol / honest-exit constants from a phase-1 JSON config."""
    with open(config_path) as fh:
        cfg = json.load(fh)
    sol = cfg.get("numerical_method", {}).get("solver", {})
    return {k: float(sol.get(k, v)) for k, v in _BASE.items()}


def dynamic_ftol(N, kv, c):
    """Mirror run_test.jl: max(ftol, min(ceiling, safety * h^(kv+1)))."""
    h = 1.0 / N
    return max(c["ftol"], min(c["dynamic_ftol_ceiling"],
                              c["dynamic_ftol_spatial_safety_factor"] * h ** (kv + 1)))


def is_true_root(residual, N, kv, c, initial_residual=float('nan')):
    """A mesh reached a TRUE root iff Newton drove ‖R‖ sufficiently small.

    Preferred test (pure relative, scale-invariant): ‖R_final‖/‖R_initial‖ ≤ relative_residual_tol.
    This is the canonical Newton-convergence test from Dennis–Schnabel / Kelley. It correctly
    handles both:
      - Re=10⁶ cells where ‖R_initial‖ ~ 10¹⁵ and ‖R_final‖ ~ 1 (relative ratio ~10⁻¹⁵ → pass,
        even though raw ‖R_final‖ is huge by the absolute gate's standard);
      - Noise-floor stalls (C7 N=320, ‖R_initial‖ = 5.74e-12, ‖R_final‖ = 9.16e-15 → relative
        ratio 1.6e-3 → fail, correctly diagnosed as Newton not achieving deep convergence even
        though the absolute residual is at machine precision).

    Floor on ‖R_initial‖ (`initial_residual_floor`): when the iter-0 residual is itself at
    machine precision (perfect IC, trivial problem, re-solve from a converged state), the
    relative form would divide-by-noise. In that regime there is nothing meaningful for Newton
    to converge; fall back to the absolute gate.

    Legacy fallback: when `initial_residual` is missing entirely (NaN — HDF5s written before
    2026-05-31 do not carry `eval_initial_residuals`), use the absolute gate so existing
    sweeps remain readable. Known to false-flag Re=10⁶ cells; re-run those cells with the
    new harness to get the relative gate behavior."""
    if not np.isfinite(residual):
        return False
    if not np.isfinite(initial_residual) or initial_residual <= c["initial_residual_floor"]:
        return bool(residual <= c["noise_floor_success_max_ftol_multiple"] * dynamic_ftol(N, kv, c))
    return bool(residual <= c["relative_residual_tol"] * initial_residual)


def consecutive_slope(h_coarse, e_coarse, h_fine, e_fine):
    """log-log slope between two meshes; NaN if either error is non-finite or non-positive."""
    if not (np.isfinite(e_coarse) and np.isfinite(e_fine) and e_coarse > 0 and e_fine > 0):
        return float('nan')
    return float(np.log(e_coarse / e_fine) / np.log(h_coarse / h_fine))


def decode(x):
    return x.decode('utf-8') if isinstance(x, bytes) else str(x)


def _slope_ok(s, target, tol):
    # One-sided: a rate AT or ABOVE the optimal order is acceptable (super-convergence is not a
    # failure). Only a rate below (target - tol) is sub-optimal.
    return s is not None and np.isfinite(s) and s >= target - tol


def _is_superconv(s, target, tol):
    # A velocity-L² rate well ABOVE nominal k+1. Not a failure, but it can signal the asymptotic
    # regime is not yet established, so it is surfaced with a visible `ˢ` caveat rather than
    # silently counted as clean.
    return s is not None and np.isfinite(s) and s > target + tol


def phase1_status(N, h, res, eu, euh, kv, c, report_N, slope_tol, res0=None):
    """Phase-1-only diagnosis of ONE (config, method), from its per-mesh residuals + errors.
    Returns (status, flag) where status ∈ {optimal, sub-optimal-rate, fold, no-root, incomplete}
    and flag ∈ {'', 'ˢ'} (super-convergent velocity-L²). This is the SAME true-root + one-sided
    slope logic the detection and merged table use, so the detailed table agrees with them. It does
    NOT include the Phase-2 rescue verdict (that is in merged_convergence_report.md).

    `res0` (optional): per-mesh iter-0 residuals for the relative-residual gate. Pass `None` or
    NaN-filled when reading legacy HDF5s that lack `eval_initial_residuals`."""
    n = len(N)
    if n == 0:
        return ("no-data", "")
    r0 = res0 if res0 is not None else np.full(n, np.nan)
    tr = np.array([is_true_root(res[i], N[i], kv, c, r0[i]) for i in range(n)])
    if n >= 2:
        sL2 = consecutive_slope(h[-2], eu[-2], h[-1], eu[-1])
        sH1 = consecutive_slope(h[-2], euh[-2], h[-1], euh[-1])
    else:
        sL2 = sH1 = float('nan')
    flag = "ˢ" if _is_superconv(sL2, kv + 1, slope_tol) else ""
    ridx = int(np.where(N == report_N)[0][0]) if report_N in N else n - 1
    if n < 2:
        return ("incomplete", flag)
    if (tr[ridx] and tr[-1] and tr[-2]
            and _slope_ok(sL2, kv + 1, slope_tol) and _slope_ok(sH1, kv, slope_tol)):
        return ("optimal", flag)
    if not tr.any():
        return ("no-root", "")
    if not tr[-1]:
        return ("fold", "")                 # finest mesh has no true root (a coarser one may)
    if tr[-1] and tr[-2]:
        return ("sub-optimal-rate", "")     # roots exist at both finest meshes, slope below optimal
    return ("partial-root", "")             # roots at some meshes but not the finest pair


# ======================================================================================
# 1. Detection — true-root + slope check, categorize, write flagged_cells.json
# ======================================================================================
def _analyze_group(g):
    h = g['h'][:]
    if len(h) == 0:
        return None
    kv = int(g.attrs['k_velocity'])
    res = g['eval_residuals'][:] if 'eval_residuals' in g else np.full(len(h), np.nan)
    # `eval_initial_residuals` was added 2026-05-31 to enable the relative-residual gate
    # ‖R_final‖/‖R_initial‖. Missing on legacy HDF5s — fall back to NaN so the gate falls
    # back to the unscaled (legacy) absolute test for those rows.
    res0 = g['eval_initial_residuals'][:] if 'eval_initial_residuals' in g else np.full(len(h), np.nan)
    eu = g['err_u_l2'][:]
    euh = g['err_u_h1'][:]
    eps = g['eval_eps'][:] if 'eval_eps' in g else np.full(len(h), np.nan)
    ovs = g['overall_verification_success'][:] if 'overall_verification_success' in g else None
    Ns = np.round(1.0 / h).astype(int)
    o = np.argsort(Ns)
    return dict(kv=kv, N=Ns[o], h=h[o], res=res[o], res0=res0[o], eu=eu[o], euh=euh[o], eps=eps[o],
                ovs=(ovs[o] if ovs is not None else None))


def detect(h5_glob, config_path, slope_tol, out_json, numbering):
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
                a = _analyze_group(f[gname])
                if a is None:
                    continue
                g = f[gname]
                method = parts[-1]
                kv = a['kv']
                N, h, res, eu, euh, eps = a['N'], a['h'], a['res'], a['eu'], a['euh'], a['eps']
                res0 = a['res0']
                tr = np.array([is_true_root(res[i], N[i], kv, c, res0[i]) for i in range(len(N))])
                summary['total'] += 1

                why = []
                if len(N) >= 2:
                    ci, fi = len(N) - 2, len(N) - 1
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
                # One-sided acceptance: a slope AT or ABOVE the optimal order passes (super-
                # convergence is not a failure — cf. C24's velocity-L²≈3.0 at k=1). Only a slope
                # BELOW (target - tol) flags the cell as sub-optimal.
                if not (np.isfinite(sL2) and sL2 >= (kv + 1) - slope_tol):
                    why.append(f"slope_L2={'nan' if not np.isfinite(sL2) else round(sL2,2)}(<exp{kv+1})")
                if not (np.isfinite(sH1) and sH1 >= kv - slope_tol):
                    why.append(f"slope_H1={'nan' if not np.isfinite(sH1) else round(sH1,2)}(<exp{kv})")
                valid_eps = eps[np.isfinite(eps)]
                if len(valid_eps) and np.all(valid_eps < 0):
                    why.append("total_failure")

                if a['ovs'] is not None and len(a['ovs']) == len(N):     # cross-check (warn only)
                    if bool(a['ovs'][fi] == 1) != bool(tr[fi]):
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
                # Categorize WHY flagged — drives Phase-2 strategy + report status:
                #   total_failure  : no true root at any mesh.
                #   fold           : finest mesh has no true root (a coarser one may) -> continuation
                #                    must find a root at a FINER mesh.
                #   suboptimal_rate: both finest meshes ARE true roots but the slope is off -> the
                #                    solution exists; finer meshes tell pre-asymptotic from genuine.
                #   incomplete     : only one mesh available.
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
                    h5=os.path.basename(fpath), group=gname, config_idx=numbering.get(_cell_id(g), 0),
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


# ======================================================================================
# 2. Annotated merged table (Phase-1 + Phase-2 continuation rescue)
# ======================================================================================
def _key(Re, Da, alpha_0, kv, etype):
    return (f"{float(Re):.3e}", f"{float(Da):.3e}", f"{float(alpha_0):.4f}", int(kv), str(etype))


def _cell_key(parts):
    """Stable per-physics-cell identity, used to group a cell's ASGS/OSGS halves and to join the
    flagged map. New-format groups are config_<idx>_<tag>_<method> (len>=4): the <tag> content-
    addresses the physics cell and is identical across concurrent shards, so it is the robust key.
    Legacy groups are config_<idx>_<method> (len==3): fall back to <idx>. This makes pairing correct
    even when concurrent shards assign different positional <idx> values to the same cell."""
    return parts[2] if len(parts) >= 4 else parts[1]


def _cell_id(g):
    """Canonical, method-agnostic physics-cell identity (Re, Da, α, kv, kp, etype). Uses the harness's
    `cell_signature` attribute when present (new-format groups), else rebuilds it from attributes
    (legacy). Matches run_test.jl's _cell_signature format."""
    cs = g.attrs.get('cell_signature')
    if cs is not None:
        return decode(cs)
    return "Re=%.12e|Da=%.12e|a0=%.12e|kv=%d|kp=%d|et=%s" % (
        float(g.attrs['Re']), float(g.attrs['Da']), float(g.attrs['alpha_0']),
        int(g.attrs['k_velocity']), int(g.attrs['k_pressure']), decode(g.attrs['element_type']))


def build_cell_numbering(h5_glob):
    """Deterministic, globally-unique display index per physics cell across all matched HDF5 files —
    independent of the stored config_<idx> (a per-process counter that repeats under sharded runs).
    Sorted by (kv, kp, etype, Re, Da, alpha_0) so the numbering is stable and reproducible.
    Returns {cell_id -> 1-based int}."""
    keys = {}
    for fpath in sorted(glob.glob(h5_glob)):
        with robust_open_h5(fpath, 'r') as f:
            for gname in f.keys():
                parts = gname.split('_')
                if len(parts) < 3 or parts[0] != 'config':
                    continue
                g = f[gname]
                cid = _cell_id(g)
                if cid not in keys:
                    keys[cid] = (int(g.attrs['k_velocity']), int(g.attrs['k_pressure']),
                                 decode(g.attrs['element_type']), float(g.attrs['Re']),
                                 float(g.attrs['Da']), float(g.attrs['alpha_0']))
    return {cid: i + 1 for i, cid in enumerate(sorted(keys, key=lambda c: keys[c]))}


def _load_phase2(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = json.load(fh)
    return {_key(r["Re"], r["Da"], r["alpha_0"], r["k_velocity"], r["element_type"]): r for r in rows}


def _load_flagged(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = json.load(fh)
    # Key by the stable cell identity (content tag for new-format groups, idx for legacy) so the
    # merged-table join below is robust to per-process idx drift under concurrent sharded writes.
    return {(_cell_key(r["group"].split('_')), r["method"]): r for r in rows}


def _fmt(x):
    return "---" if (x is None or not np.isfinite(x)) else f"{x:.3e}"


def merged_table(h5_glob, config_path, flagged_path, phase2_path, out_path, report_N, slope_tol, numbering):
    c = load_solver_constants(config_path)
    phase2 = _load_phase2(phase2_path)
    flagged_map = _load_flagged(flagged_path)
    rows = []
    for fpath in sorted(glob.glob(h5_glob)):
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
                kv = int(g.attrs['k_velocity'])
                etype = decode(g.attrs['element_type'])
                Re = float(g.attrs['Re']); Da = float(g.attrs['Da']); a0 = float(g.attrs['alpha_0'])
                res = g['eval_residuals'][:] if 'eval_residuals' in g else np.full(len(h), np.nan)
                res0 = g['eval_initial_residuals'][:] if 'eval_initial_residuals' in g else np.full(len(h), np.nan)
                # `eval_eps` (per-mesh) = the largest eps_pert in {1.0, 0.1, ..., 0} that
                # Newton actually absorbed at this N under the homotopy outer loop's
                # break-at-first-success cascade. It is the per-cell robustness fingerprint;
                # 1.0 = solver handled a full perturbation; 0 = only the eps=0 (u_ex) fallback
                # converged (pre-asymptotic / noise-floor limited regime).
                eps = g['eval_eps'][:] if 'eval_eps' in g else np.full(len(h), np.nan)
                eu, ep, euh = g['err_u_l2'][:], g['err_p_l2'][:], g['err_u_h1'][:]
                N = np.round(1.0 / h).astype(int)
                o = np.argsort(N)
                N, h, res, res0, eu, ep, euh, eps = N[o], h[o], res[o], res0[o], eu[o], ep[o], euh[o], eps[o]
                tr = np.array([is_true_root(res[i], N[i], kv, c, res0[i]) for i in range(len(N))])

                ridx = int(np.where(N == report_N)[0][0]) if report_N in N else len(N) - 1
                rep_N = int(N[ridx])
                if len(N) >= 2:
                    sL2 = consecutive_slope(h[-2], eu[-2], h[-1], eu[-1])
                    sH1 = consecutive_slope(h[-2], euh[-2], h[-1], euh[-1])
                else:
                    sL2 = sH1 = float('nan')

                verified = (tr[ridx] and len(N) >= 2 and tr[-1] and tr[-2]
                            and _slope_ok(sL2, kv + 1, slope_tol) and _slope_ok(sH1, kv, slope_tol))

                tri = np.where(tr)[0]

                def ls_rate(errs):
                    idx = tri[-3:]
                    if len(idx) < 2 or np.any(errs[idx] <= 0) or np.any(~np.isfinite(errs[idx])):
                        return float('nan')
                    return float(np.polyfit(np.log(h[idx]), np.log(errs[idx]), 1)[0])
                lsL2, lsH1 = ls_rate(eu), ls_rate(euh)

                fl = flagged_map.get((_cell_key(parts), method))
                cat = fl.get("category") if fl else None
                p2 = phase2.get(_key(Re, Da, a0, kv, etype))
                tol = slope_tol
                # eps_used per the homotopy cascade: the eval_eps at the finest mesh (the
                # hardest case in the convergence ladder) is the most diagnostic single number.
                eps_finest = float(eps[-1]) if len(eps) and np.isfinite(eps[-1]) else float('nan')
                eps_min = float(np.nanmin(eps)) if len(eps) and np.any(np.isfinite(eps)) else float('nan')
                row = dict(cidx=numbering.get(_cell_id(g), 0), Re=Re, Da=Da, a0=a0, k=kv, etype=etype, method=method,
                           lsL2=lsL2, lsH1=lsH1, rate_mark="",
                           eps_finest=eps_finest, eps_min=eps_min)

                if verified:
                    row.update(status="verified", mark="", N_rep=rep_N,
                               l2u=float(eu[ridx]), l2p=float(ep[ridx]), h1u=float(euh[ridx]), sL2=sL2, sH1=sH1,
                               rate_mark=("ˢ" if _is_superconv(sL2, kv + 1, tol) else ""))
                elif p2 and p2.get("status") == "rescued":
                    ft = p2["finest_true_root"]; pL2 = p2.get("slope_L2"); pH1 = p2.get("slope_H1")
                    opt = _slope_ok(pL2, kv + 1, tol) and _slope_ok(pH1, kv, tol)
                    # Phase-2 reached finer meshes: optimal there ⇒ was pre-asymptotic/fold (now verified);
                    # still off ⇒ GENUINE sub-optimal rate even at the finest mesh.
                    row.update(status=("verified@fine-mesh" if opt else "SUBOPTIMAL-RATE@fine"),
                               mark=("*" if opt else "‡"), N_rep=int(ft["N"]),
                               l2u=ft["l2u"], l2p=ft.get("l2p"), h1u=ft["h1u"], sL2=pL2, sH1=pH1,
                               rate_mark=("ˢ" if _is_superconv(pL2, kv + 1, tol) else ""))
                elif cat == "suboptimal_rate":
                    # Root EXISTS at the report mesh (not a fold); the issue is the rate. Report the
                    # report-mesh error + the finest-pair (sub-optimal) rate. Prominent ‡ mark.
                    row.update(status="SUBOPTIMAL-RATE", mark="‡", N_rep=rep_N,
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
                    row.update(status="no-root", mark="", N_rep=rep_N,
                               l2u=None, l2p=None, h1u=None, sL2=float('nan'), sH1=float('nan'))
                rows.append(row)

    rows.sort(key=lambda r: (r['etype'], r['k'], r['cidx'], r['method']))
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    def rfmt(s):
        return "---" if (s is None or not np.isfinite(s)) else f"{s:.2f}"
    def epsfmt(s):
        # eps_pert values are exact decimals from {1.0, 0.1, 0.01, ..., 0.0}; format compactly.
        return "---" if (s is None or not np.isfinite(s)) else (f"{s:.0e}" if 0 < s < 0.01 else f"{s:g}")
    with open(out_path, 'w') as io:
        io.write("# Merged MMS convergence table (Phase-1 sweep + Phase-2 continuation)\n\n")
        io.write("Marks: *(none)*=verified (true root + optimal slope, k+1 / k); ")
        io.write("`*`=true root recovered by continuation at the noted finer mesh, slope then optimal; ")
        io.write("`‡`=**converges but SUB-OPTIMAL rate** (true root exists; slope stays below k+1/k even at the finest mesh — a genuine finding, not a fold); ")
        io.write("`**`=best true root attained, asymptotics not established; `N/A`=no true root anywhere.\n\n")
        io.write("rate_* = slope on the finest available mesh PAIR; lsq_* = least-squares slope over the finest ≤3 true-root meshes.\n\n")
        io.write("Rate caveat: a velocity-L² rate marked `ˢ` is ABOVE the nominal k+1 (super-convergent). ")
        io.write("This is not a failure (acceptance is one-sided, slope ≥ target − tol), but a rate well above nominal ")
        io.write("can signal the asymptotic regime is not yet established — confirm with one further refinement.\n\n")
        io.write("**eps_finest / eps_min** columns report the per-cell robustness fingerprint: the largest "
                 "eps_pert in {1.0, 0.1, ..., 0} that the homotopy outer loop accepted at the FINEST mesh "
                 "(eps_finest) and the SMALLEST such value across the mesh ladder (eps_min). 1 means the "
                 "solver absorbed a full ‖u_ex‖-scale perturbation of the initial guess; 0 means only the "
                 "interpolated-u_ex fallback converged (pre-asymptotic / noise-floor limited).\n\n")
        io.write("| Config | Re | Da | α₀ | k | Elem | Method | Status | N_rep | L2_u | L2_p | H1_u | rate_L2u | rate_H1u | lsq_L2u | lsq_H1u | eps_finest | eps_min |\n")
        io.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            l2u = (_fmt(r['l2u']) + r['mark']) if r['l2u'] is not None else "N/A"
            io.write("| C{} | {:.0e} | {:.0e} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                r['cidx'], r['Re'], r['Da'], r['a0'], r['k'], r['etype'], r['method'], r['status'],
                r['N_rep'], l2u, _fmt(r['l2p']), _fmt(r['h1u']),
                rfmt(r['sL2']) + r.get('rate_mark', ''), rfmt(r['sH1']), rfmt(r.get('lsL2')), rfmt(r.get('lsH1')),
                epsfmt(r.get('eps_finest')), epsfmt(r.get('eps_min'))))
    n_ver = sum(1 for r in rows if r['status'] == 'verified')
    n_star = sum(1 for r in rows if r['mark'] == '*')
    n_subopt = sum(1 for r in rows if r['mark'] == '‡')
    n_ss = sum(1 for r in rows if r['mark'] == '**')
    n_na = sum(1 for r in rows if r['status'] == 'no-root')
    n_super = sum(1 for r in rows if r.get('rate_mark') == 'ˢ')
    print(f"[merged] sub-optimal-rate(‡): {n_subopt}; super-convergent-L²(ˢ, confirm asymptotics): {n_super}")
    print(f"[merged] {len(rows)} rows: {n_ver} verified, {n_star} needs-fine-mesh(*), "
          f"{n_ss} best-root(**), {n_na} no-root. -> {out_path}")
    return rows


# ======================================================================================
# 3. Per-config convergence plots + 4. detailed per-config table (honest converged)
# ======================================================================================
def _format_scientific(val):
    if val == 1.0:
        return "1"
    s = f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e")
    base, exp = s.split("e")
    return f"10^{exp}" if base == "1" else s


def make_plots_and_detailed(h5_glob, config_path, outdir, report_N, slope_tol, numbering, do_plots=True):
    """Per-config log-log plots (PNG) + a detailed markdown table (honest true-root 'Converged')
    + fixed-width summary_tables.txt. Reads all h5 files matching the glob."""
    c = load_solver_constants(config_path)
    files = sorted(glob.glob(h5_glob))
    if not files:
        print(f"[plots] no h5 files match {h5_glob}")
        return
    os.makedirs(outdir, exist_ok=True)
    if do_plots and not _HAVE_MPL:
        print("[plots] matplotlib not available — skipping PNGs (tables still written).")
    table_data = collections.defaultdict(list)          # (etype,kv,kp) -> rows for summary_tables
    detailed_rows = []
    config_file_to_cidx = collections.defaultdict(set)

    for fpath in files:
        with robust_open_h5(fpath, 'r') as f:
            config_dict = collections.defaultdict(dict)
            for gname in f.keys():
                parts = gname.split('_')
                if len(parts) >= 3 and parts[0] == 'config':
                    config_dict[_cell_key(parts)][parts[-1]] = gname
            for cell_key, methods in config_dict.items():
                first_gname = next(iter(methods.values()))
                first_g = f[first_gname]
                # Grouping above is by the stable cell_key (content tag for new-format groups), so
                # ASGS/OSGS pair correctly even when concurrent shards assigned different positional
                # <idx> values. c_idx here is just a human-readable display label from the group name.
                c_idx = numbering.get(_cell_id(first_g), 0)
                Re = float(first_g.attrs['Re']); Da = float(first_g.attrs['Da'])
                a0 = float(first_g.attrs['alpha_0'])
                kv = int(first_g.attrs['k_velocity']); kp = int(first_g.attrs['k_pressure'])
                etype = decode(first_g.attrs['element_type'])
                opt_u_l2, opt_u_h1, opt_p_l2, opt_p_h1 = kv + 1, kv, kp, max(kv - 1, 0)

                if do_plots and _HAVE_MPL:
                    plt.figure(figsize=(10, 8))
                trow = {'c_idx': c_idx, 'Re': Re, 'Da': Da, 'alpha_0': a0}

                for method, gname in methods.items():
                    g = f[gname]
                    h = g['h'][:]; eu = g['err_u_l2'][:]; ep = g['err_p_l2'][:]
                    euh = g['err_u_h1'][:]; eph = g['err_p_h1'][:]
                    res = g['eval_residuals'][:] if 'eval_residuals' in g else np.full(len(h), np.nan)
                    res0 = g['eval_initial_residuals'][:] if 'eval_initial_residuals' in g else np.full(len(h), np.nan)
                    # eval_eps: per-mesh, largest eps_pert in {1.0, 0.1, ..., 0} that the homotopy
                    # outer loop accepted at this N. The per-cell robustness fingerprint.
                    eps_arr = g['eval_eps'][:] if 'eval_eps' in g else np.full(len(h), np.nan)
                    Ns = np.round(1.0 / h).astype(int)
                    # Diagnosis from the FULL per-mesh arrays (sorted by N, NO eu>0 filter) so the
                    # Status column matches detection / the merged table.
                    so = np.argsort(Ns)
                    p1status, p1flag = phase1_status(Ns[so], h[so], res[so], eu[so], euh[so],
                                                     kv, c, report_N, slope_tol, res0=res0[so])
                    # Masked + sorted arrays for plotting / rate / FME (drop non-positive errors).
                    m = (eu > 0) & (ep > 0)
                    h, eu, ep, euh, eph, res, res0, Ns, eps_arr = h[m], eu[m], ep[m], euh[m], eph[m], res[m], res0[m], Ns[m], eps_arr[m]
                    order = np.argsort(Ns)
                    h, eu, ep, euh, eph, res, res0, Ns, eps_arr = (a[order] for a in (h, eu, ep, euh, eph, res, res0, Ns, eps_arr))

                    cf = g.attrs.get('config_file', b'unknown.json')
                    config_file_to_cidx[decode(cf)].add(c_idx)

                    def pair_rate(arr):
                        return consecutive_slope(h[-2], arr[-2], h[-1], arr[-1]) if len(h) >= 2 else float('nan')
                    ru2, rp2, ru1, rp1 = pair_rate(eu), pair_rate(ep), pair_rate(euh), pair_rate(eph)

                    trow[f'slope_u_{method}'] = ru2 / opt_u_l2 if opt_u_l2 else float('nan')
                    trow[f'fme_u_{method}'] = eu[-1] if len(eu) else float('nan')
                    trow[f'slope_p_{method}'] = rp2 / opt_p_l2 if opt_p_l2 else float('nan')
                    trow[f'fme_p_{method}'] = ep[-1] if len(ep) else float('nan')
                    eval_times = g['eval_times'][:] if 'eval_times' in g else []
                    trow[f'time_{method}'] = float(g.attrs.get('total_time_s',
                                                               sum(eval_times) if len(eval_times) else 0.0))

                    # HONEST converged: finest mesh is a true root (‖R‖ ≤ k_nf·dynamic_ftol), NOT
                    # the solver's homotopy-sentinel heuristic.
                    honest_root = bool(len(Ns) and is_true_root(res[-1], int(Ns[-1]), kv, c, res0[-1] if len(res0) else float('nan')))
                    total_iters = int(g.attrs.get('total_iters', 0))
                    total_time = float(g.attrs.get('total_time_s', 0.0))
                    fres = res[-1] if len(res) and np.isfinite(res[-1]) else float('nan')
                    eps_finest_det = float(eps_arr[-1]) if len(eps_arr) and np.isfinite(eps_arr[-1]) else float('nan')
                    eps_min_det = float(np.nanmin(eps_arr)) if len(eps_arr) and np.any(np.isfinite(eps_arr)) else float('nan')
                    detailed_rows.append(dict(
                        cidx=int(c_idx), method=method, cf=decode(cf), Re=Re, Da=Da, a0=a0, kv=kv,
                        etype=etype, time=total_time, iters=total_iters, converged=honest_root,
                        status=p1status, flag=p1flag,
                        fres=fres, ftol=dynamic_ftol(int(Ns[-1]), kv, c) if len(Ns) else float('nan'),
                        ru2=ru2, rp2=rp2, ru1=ru1, rp1=rp1,
                        opt=(opt_u_l2, opt_p_l2, opt_u_h1, opt_p_h1),
                        fme=(eu[-1] if len(eu) else np.nan, ep[-1] if len(ep) else np.nan,
                             euh[-1] if len(euh) else np.nan, eph[-1] if len(eph) else np.nan),
                        eps_finest=eps_finest_det, eps_min=eps_min_det))

                    if do_plots and _HAVE_MPL and len(h):
                        ls = '-' if method == 'ASGS' else '--'
                        plt.loglog(h, eu, marker='o', linestyle=ls, color='blue', linewidth=2,
                                   markersize=8, label=f'{method} $L_2$ Velocity ({opt_u_l2})')
                        plt.loglog(h, euh, marker='D', linestyle=ls, color='blue', linewidth=2,
                                   markersize=8, label=f'{method} $H^1$ Velocity ({opt_u_h1})')
                        plt.loglog(h, ep, marker='o', linestyle=ls, color='red', linewidth=2,
                                   markersize=8, markerfacecolor='white',
                                   label=f'{method} $L_2$ Pressure ({opt_p_l2})')
                        for i in range(len(h) - 1):
                            for arr, col, opt in ((eu, 'blue', opt_u_l2), (euh, 'blue', opt_u_h1),
                                                  (ep, 'red', opt_p_l2)):
                                hm = np.sqrt(h[i] * h[i + 1])
                                em = np.sqrt(arr[i] * arr[i + 1])
                                sv = (np.log(arr[i + 1]) - np.log(arr[i])) / (np.log(h[i + 1]) - np.log(h[i]))
                                plt.annotate(f'{(sv/opt if opt else float("nan")):.2f}', xy=(hm, em),
                                             xytext=(0, 10 if method == 'ASGS' else -12),
                                             textcoords='offset points', ha='center', fontsize=8,
                                             color=col, fontweight='bold')

                key = (etype, kv, kp)
                table_data[key].append(trow)
                if do_plots and _HAVE_MPL:
                    plt.xlabel(r'Mesh size ($h$)'); plt.ylabel('Error Norms')
                    plt.title(fr'Convergence ($Re: {Re:g}$, $Da: {Da:g}$, $\alpha_0: {a0}$, $k: {kv}$, {etype})')
                    if plt.gca().get_legend_handles_labels()[0]:    # skip empty legend (failed cells)
                        plt.legend(handlelength=4.0)
                    plt.grid(True, which="both", ls="--")
                    plot_subdir = os.path.join(outdir, f'k{kv}', etype)
                    os.makedirs(plot_subdir, exist_ok=True)
                    png = os.path.join(plot_subdir, f'convergence_config{c_idx}_Re{Re:g}_Da{Da:g}_a{a0}_k{kv}{kp}.png')
                    plt.savefig(png); plt.close()

    _write_detailed_md(detailed_rows, config_file_to_cidx, os.path.join(outdir, 'convergence_report.md'))
    _write_summary_tables(table_data, os.path.join(outdir, 'summary_tables.txt'))
    if do_plots and _HAVE_MPL:
        print(f"[plots] per-config PNGs written under {outdir}/k{{kv}}/{{etype}}/")


def _write_detailed_md(rows, config_file_to_cidx, out_file):
    def fsci(v):
        return "NaN" if (v is None or not np.isfinite(v)) else f"{v:.4e}"

    def frate(rate, opt):
        if not np.isfinite(rate):
            return "  N/A"
        s = f"{rate:4.2f} ({opt:.0f})"
        return f"<b style='color:red'>{s}</b>" if rate < 0.90 * opt else s

    def jl(val):
        return {1.0: "1e+00", 1e-6: "1e-06", 1e6: "1e+06"}.get(val,
               f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e+"))

    def fstatus(status, flag):
        base = {"optimal": "optimal", "sub-optimal-rate": "sub-optimal-rate ‡",
                "fold": "fold (**)", "no-root": "no-root (N/A)", "incomplete": "incomplete",
                "partial-root": "partial-root", "no-data": "no-data"}.get(status, status)
        label = f"{base} {flag}" if flag else base
        if status in ("fold", "no-root", "no-data"):
            return f"<b style='color:red'>{label}</b>"
        if status == "sub-optimal-rate":
            return f"<b style='color:#b8860b'>{label}</b>"
        return label
    rows.sort(key=lambda r: (r['etype'], r['kv'], r['cidx'], r['method']))
    with open(out_file, 'w') as io:
        io.write("# Convergence Rate and FME Table\n\n")
        io.write("**Status** is the Phase-1 per-config diagnosis (same true-root + one-sided-slope "
                 "logic as the merged table, WITHOUT the Phase-2 continuation rescue — for the "
                 "post-rescue verdict see `merged_convergence_report.md`):\n")
        io.write("- `optimal` — true root at the finest meshes and slope ≥ optimal (k+1 / k). "
                 "`ˢ` = velocity-L² rate ABOVE nominal (super-convergent; confirm asymptotics).\n")
        io.write("- `sub-optimal-rate ‡` — true root exists at both finest meshes, but the slope is "
                 "below optimal (candidate for a finer-mesh check).\n")
        io.write("- `fold (**)` — the finest mesh has NO true root (a coarser one may); the discrete "
                 "branch folds → needs continuation at a finer mesh.\n")
        io.write("- `no-root (N/A)` — no true root at any mesh. `incomplete` — only one mesh.\n\n")
        io.write("'Converged' is the HONEST true-root test (‖R‖ ≤ k_nf·dynamic_ftol) at the finest "
                 "mesh, not the solver's noise-floor-foolable flag. Final Res. shows (‖R‖ vs "
                 "dynamic_ftol at that mesh). Rates/FME are the Phase-1 finest-mesh values.\n\n"
                 "'eps_finest' / 'eps_min' is the homotopy outer loop's per-cell robustness fingerprint: "
                 "the largest eps_pert ∈ {1.0, 0.1, ..., 0} that Newton absorbed at the finest mesh "
                 "(eps_finest) and the smallest such value across all meshes (eps_min). 1 = solver "
                 "handled a full perturbation; 0 = only the interpolated-u_ex fallback worked.\n\n")
        def epsfmt_det(v):
            return "---" if (v is None or not np.isfinite(v)) else (f"{v:.0e}" if 0 < v < 0.01 else f"{v:g}")
        io.write("| Config | Method | Source JSON | Re | Da | α_0 | k | Elem | Time (s) | Iters | "
                 "Converged | Status | Final Res. | rate_u_L2 | rate_p_L2 | rate_u_H1 | rate_p_H1 | "
                 "FME u_L2 | FME p_L2 | FME u_H1 | FME p_H1 | eps_finest | eps_min |\n")
        io.write("|" + "---|" * 23 + "\n")
        for r in rows:
            ou2, op2, ou1, op1 = r['opt']
            conv = "Yes" if r['converged'] else "<b style='color:red'>No</b>"
            io.write("| C{} | {} | {} | {} | {} | {:.2f} | {} | {} | {:.1f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                r['cidx'], r['method'], r['cf'], jl(r['Re']), jl(r['Da']), r['a0'], r['kv'], r['etype'],
                r['time'], r['iters'], conv, fstatus(r['status'], r.get('flag', '')),
                f"{fsci(r['fres'])} ({fsci(r['ftol'])})",
                frate(r['ru2'], ou2), frate(r['rp2'], op2), frate(r['ru1'], ou1), frate(r['rp1'], op1),
                fsci(r['fme'][0]), fsci(r['fme'][1]), fsci(r['fme'][2]), fsci(r['fme'][3]),
                epsfmt_det(r.get('eps_finest')), epsfmt_det(r.get('eps_min'))))
        # config reference map
        io.write("\n## Simulation Configuration Reference Map\n")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out_file))), 'data')
        for jf in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
            bname = os.path.basename(jf)
            cids = sorted(config_file_to_cidx.get(bname, []), key=int)
            cstr = ", ".join(f"C{x}" for x in cids) if cids else "N/A"
            try:
                with open(jf) as fh:
                    io.write(f"\n<details><summary><b>{bname} (Config IDs Map: {cstr})</b></summary>\n")
                    io.write('<pre><code class="json">\n')
                    io.write(json.dumps(json.load(fh), indent=4))
                    io.write("\n</code></pre>\n</details>\n")
            except Exception as e:
                io.write(f"\n<!-- Error loading config {bname}: {e} -->\n")
    print(f"[detailed] per-config table -> {out_file}")


def _write_summary_tables(table_data, out_file):
    with open(out_file, 'w') as f:
        for key in sorted(table_data.keys(), key=lambda x: (x[0], x[1], x[2])):
            etype, kv, kp = key
            opt_u, opt_p = kv + 1, kp
            data = sorted(table_data[key], key=lambda x: (x['Da'], x['Re'], -x['alpha_0']))
            f.write(f"Table: Observed (normalized) convergence rates and finest-mesh error (FME) "
                    f"for P{kv}/P{kp} elements ({etype})\n")
            f.write("-" * 104 + "\n\n")
            for field, label, opt in (('u', 'velocity', opt_u), ('p', 'pressure', opt_p)):
                f.write(f"{label}\n")
                f.write(f"                              ASGS ({opt})                                  OSGS ({opt})\n")
                f.write("Cfg   Re          Da          α0      slope        FME          Time(s)      slope        FME          Time(s)\n")
                for row in data:
                    def sl(v): return "N/A" if (v is None or np.isnan(v)) else f"{v:.2f}"
                    def fe(v): return "N/A" if (v is None or np.isnan(v)) else f"{v:.2e}"
                    def tm(v): return "N/A" if (v is None or np.isnan(v)) else f"{v:.1f}"
                    sa, fa = sl(row.get(f'slope_{field}_ASGS', np.nan)), fe(row.get(f'fme_{field}_ASGS', np.nan))
                    ta = tm(row.get('time_ASGS', np.nan))
                    so, fo = sl(row.get(f'slope_{field}_OSGS', np.nan)), fe(row.get(f'fme_{field}_OSGS', np.nan))
                    to = tm(row.get('time_OSGS', np.nan))
                    f.write(f"C{row['c_idx']:<4} {_format_scientific(row['Re']):<11} {_format_scientific(row['Da']):<11} "
                            f"{row['alpha_0']:<7.2f} {sa:<12} {fa:<12} {ta:<12} {so:<12} {fo:<12} {to:<12}\n")
                f.write("\n")
            f.write("=" * 104 + "\n\n")
    print(f"[summary] fixed-width tables -> {out_file}")


# ======================================================================================
# main
# ======================================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="One-stop MMS sweep analysis: detect + merged table + plots + detailed table.")
    ap.add_argument('--h5', default=os.path.join(here, 'results', '*.h5'),
                    help="glob of result HDF5 DBs (default results/*.h5; pass one DB to analyze a single study)")
    ap.add_argument('--config', default=os.path.join(here, 'data', 'test_config.json'),
                    help="a sweep JSON config (for the dynamic_ftol / k_nf constants)")
    ap.add_argument('--flagged', default=os.path.join(here, 'results', 'flagged_cells.json'),
                    help="output path for the detected flagged_cells.json (Phase-2 input)")
    ap.add_argument('--phase2', default=os.path.join(here, 'results', 'phase2_results.json'),
                    help="Phase-2 continuation results to join (optional; ignored if absent)")
    ap.add_argument('--out', default=os.path.join(here, 'results', 'merged_convergence_report.md'),
                    help="output path for the annotated merged table")
    ap.add_argument('--outdir', default=os.path.join(here, 'results'),
                    help="directory for plots + detailed/summary tables")
    ap.add_argument('--report-N', type=int, default=320, help="mesh whose error is reported for verified cells")
    ap.add_argument('--slope-tol', type=float, default=0.25)
    ap.add_argument('--no-plots', action='store_true', help="skip PNG plots (tables still written)")
    ap.add_argument('--no-detailed', action='store_true', help="skip the detailed per-config table + summary + plots")
    ap.add_argument('--only', choices=['detect', 'merged', 'detailed', 'plots'], default=None,
                    help="run only one stage instead of everything")
    args = ap.parse_args()

    do = args.only
    numbering = build_cell_numbering(args.h5)   # deterministic, shard-independent config numbers
    if do in (None, 'detect'):
        detect(args.h5, args.config, args.slope_tol, args.flagged, numbering)
    if do in (None, 'merged'):
        merged_table(args.h5, args.config, args.flagged, args.phase2, args.out, args.report_N, args.slope_tol, numbering)
    if do in ('detailed', 'plots') or (do is None and not args.no_detailed):
        make_plots_and_detailed(args.h5, args.config, args.outdir, args.report_N, args.slope_tol, numbering,
                                do_plots=(do != 'detailed' and not args.no_plots))


if __name__ == '__main__':
    main()
