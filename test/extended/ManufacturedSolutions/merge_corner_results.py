#!/usr/bin/env python3
"""
merge_corner_results.py — merge the direct-solve CORNER results (the Re=1e6/α₀=0.05 cells that the
standard sweep `skip_cells` and that `run_corner_article.jl`/`run_corner_osgs.jl` produce as JSON in
the TRACKED `data/corner/` dir) INTO the sweep HDF5 DBs, as proper content-addressed groups.

Why: `analyze_results.py` (per-cell convergence PNGs + reports) and any other h5 consumer read the
sweep h5 only, so the corner cells — living in separate JSONs — get no plot/report. After this merge
they are first-class h5 groups, so `analyze_results.py` plots and reports all 27 grid cells per family.
(The LaTeX table generator already merged them; this brings the h5/PNG pipeline to parity.)

Idempotent: re-running deletes and rewrites the corner groups. It only ADDS corner groups; the swept
groups are untouched. Run, then `python analyze_results.py` to regenerate the plots.
"""
import hashlib
import json
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DEBUG = os.path.join(RESULTS, "debug_results")
# Corner direct-solve provenance lives in a TRACKED dir (these JSONs feed the official plots), not in the
# gitignored results/debug_results/ scratch — see data/corner/README.md.
CORNER = os.path.join(HERE, "data", "corner")

# corner JSON l2u/l2p/h1u/h1p  ->  h5 err_u_l2/err_p_l2/err_u_h1/err_p_h1
_NORM = {"err_u_l2": "l2u", "err_p_l2": "l2p", "err_u_h1": "h1u", "err_p_h1": "h1p"}
_RATE = {"rate_u_l2": "slope_l2u", "rate_p_l2": "slope_l2p",
         "rate_u_h1": "slope_h1u", "rate_p_h1": "slope_h1p"}


def _cell_signature(Re, Da, a0, kv, kp, etype):
    # byte-identical to run_test.jl `_cell_signature` (%.12e canonicalises the IEEE double)
    return "Re=%.12e|Da=%.12e|a0=%.12e|kv=%d|kp=%d|et=%s" % (Re, Da, a0, kv, kp, etype)


def _content_tag(sig):
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def _load_corner_records(json_specs):
    """Read corner JSONs into a list of dicts keyed by (Re,Da,a0,method). Adds the Forchheimer
    OSGS Da=1 alias (≡ Da=1e-6) when a Da=1 record is absent — so all 3 Da are plotted."""
    recs = {}
    for path, default_method in json_specs:
        if not os.path.isfile(path):
            continue
        for r in json.load(open(path)):
            if r.get("status") == "base_only" or r.get("l2u") is None:
                continue
            method = str(r.get("method", default_method)).upper()
            key = (float(r["Re"]), float(r["Da"]), float(r["alpha_0"]), method)
            recs[key] = r
    # OSGS Da=1 ← Da=1e-6 (σ∝Da negligible vs convection at this corner)
    for (Re, Da, a0, m), r in list(recs.items()):
        if m == "OSGS" and abs(Da - 1e-6) < 1e-12:
            k1 = (Re, 1.0, a0, m)
            if k1 not in recs:
                recs[k1] = dict(r, Da=1.0, _aliased=True)
    return recs


def inject(h5_path, json_specs, kv, etype, idx_base=900):
    if not os.path.isfile(h5_path):
        print(f"[merge] sweep h5 not found, skipping: {h5_path}")
        return 0
    recs = _load_corner_records(json_specs)
    kp = kv
    n_written = 0
    with h5py.File(h5_path, "a") as f:
        for i, ((Re, Da, a0, method), r) in enumerate(sorted(recs.items())):
            sig = _cell_signature(Re, Da, a0, kv, kp, etype)
            name = f"config_{idx_base + i}_{_content_tag(sig)}_{method}"
            if name in f:
                del f[name]
            g = f.create_group(name)
            hs = np.asarray(r["hs"], float)
            n = len(hs)
            g["h"] = hs
            for h5k, jk in _NORM.items():
                g[h5k] = np.asarray(r[jk], float)
            iters = [int(x) for x in r.get("iters", [0] * n)]
            g["eval_iters"] = np.asarray(iters, dtype=np.int64)
            g["eval_eps"] = np.zeros(n)                       # direct solve: ε_pert = 0
            g["eval_times"] = np.zeros(n)
            g["eval_residuals"] = np.asarray(r.get("residuals", [np.nan] * n), float)
            g["eval_initial_residuals"] = np.full(n, np.nan)
            g["overall_verification_success"] = np.ones(n, dtype=np.int8)   # reached true roots
            g["mms_plateau_success"] = np.full(n, np.int8(-1))             # plateau verifier not run
            g.attrs["Re"] = Re
            g.attrs["Da"] = Da
            g.attrs["alpha_0"] = a0
            g.attrs["k_velocity"] = int(kv)
            g.attrs["k_pressure"] = int(kp)
            g.attrs["element_type"] = etype
            g.attrs["method"] = method
            g.attrs["cell_signature"] = sig
            g.attrs["config_file"] = "corner_direct_solve"
            g.attrs["total_iters"] = int(sum(iters))
            g.attrs["total_time_s"] = 0.0
            g.attrs["physical_epsilon"] = 0.0
            g.attrs["numerical_epsilon_coeff"] = 0.0
            g.attrs["target_ftol"] = 1e-8
            g.attrs["osgs_tolerance"] = 1e-8
            for rk, jk in _RATE.items():
                if r.get(jk) is not None:
                    g.attrs[rk] = float(r[jk])
            n_written += 1
            tag = " (alias Da=1←1e-6)" if r.get("_aliased") else ""
            print(f"  + {name}  Re={Re:.0e} Da={Da:.0e} a={a0} {method}  N={[int(round(1/x)) for x in hs]}{tag}")
    print(f"[merge] {os.path.basename(h5_path)}: wrote {n_written} corner groups")
    return n_written


def main():
    # [layout 2026-06-27] DBs live per-(kv,etype) as results/k<kv>/<etype>/results.h5.
    total = 0
    total += inject(
        os.path.join(RESULTS, "k1", "TRI", "results.h5"),
        [(os.path.join(CORNER, "corner_tri_k1_a005.json"), "ASGS"),
         (os.path.join(CORNER, "corner_tri_k1_a005_osgs.json"), "OSGS"),
         (os.path.join(CORNER, "corner_tri_k1_a005_osgs_da1e6.json"), "OSGS")],
        kv=1, etype="TRI")
    total += inject(
        os.path.join(RESULTS, "k2", "QUAD", "results.h5"),
        [(os.path.join(CORNER, "corner_quad_k2_a005.json"), "ASGS"),
         (os.path.join(CORNER, "corner_quad_k2_a005_osgs.json"), "OSGS")],
        kv=2, etype="QUAD")
    print(f"\n[merge] done — {total} corner groups merged. "
          f"Now run:  python analyze_results.py  (regenerates PNGs + reports with the corner cells).")


if __name__ == "__main__":
    main()
