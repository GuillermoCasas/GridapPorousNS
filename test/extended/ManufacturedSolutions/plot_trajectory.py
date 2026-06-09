#!/usr/bin/env python3
"""Render nonlinear-solver trajectory diagrams for this MMS sweep — one PNG per eps_pert attempt.

Test-specific wrapper around the shared renderer tools/trajectory_viz/trajectory_plot.py. It reads
the per-run trajectory JSON sidecars written by run_test.jl under results/traces/ (one per
cell+method+N) and, for EACH homotopy eps_pert attempt of a run, writes one independent PNG showing
that attempt's path through the nonlinear orchestration (Alg. O -> B/C -> A) with per-iteration
residual dots. The eps_pert / attempt notion is MMS-harness-specific and lives here; the shared tool
only knows how to render a list of algorithm stages. See tools/trajectory_viz/README.md.

Usage:
    python plot_trajectory.py                                      # every attempt of every trace
    python plot_trajectory.py --cell Re=1e6,Da=1e-6,a0=0.5 --N 160 --method ASGS
    python plot_trajectory.py --file results/traces/traj_....json
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools", "trajectory_viz"))
import trajectory_plot as tp  # noqa: E402

GREEN, RED = "#2e7d32", "#c62828"


def _plot_dir_for(trace_path):
    """Output dir for a trace's plots: a 'plots/' subfolder INSIDE the trace's own 'traces/' dir
    (results/k<kv>/<etype>/traces/plots/), so each trajectory's JSON sidecar and its rendered PNG live
    together under traces/. (The per-(kv,etype) convergence PNGs stay at the k<kv>/<etype> top level.)
    A run-name subfolder is appended by the caller, giving results/k<kv>/<etype>/traces/plots/<run>/."""
    return os.path.join(os.path.dirname(os.path.abspath(trace_path)), "plots")


def _match(cell, N, cell_sel, n_sel, method_sel):
    if method_sel and str(cell.get("method", "")).upper() != method_sel.upper():
        return False
    if n_sel is not None and int(N) != int(n_sel):
        return False
    if cell_sel:
        for tok in cell_sel.split(","):
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k = {"a0": "alpha_0", "alpha0": "alpha_0"}.get(k.strip().lower(), k.strip().lower())
            try:
                if not math.isclose(float(cell.get(k, float("nan"))), float(v), rel_tol=1e-3):
                    return False
            except (TypeError, ValueError):
                if str(cell.get(k, "")).lower() != v.strip().lower():
                    return False
    return True


def _title(cell, N):
    return (rf"$Re={tp.fmt_pow(cell['Re'])}$, $Da={tp.fmt_pow(cell['Da'])}$, $\alpha_0={cell['alpha_0']:g}$"
            rf"   |   $P_{{{cell['kv']}}}/P_{{{cell['kp']}}}$ {cell['etype']} {cell['method']}"
            rf"   |   $N={N}$   ({cell.get('encoding', '')})")


def main():
    ap = argparse.ArgumentParser(description="MMS solver trajectory diagrams (one PNG per eps_pert attempt).")
    ap.add_argument("--traces", default=os.path.join(HERE, "results"),
                    help="root to search for trajectory JSON sidecars (default results/; recurses into k<kv>/<etype>/traces/ and debug_results/)")
    ap.add_argument("--file", default=None, help="plot a single trace JSON file")
    ap.add_argument("--out", default=None,
                    help="force a single flat output dir (override). Default: each plot mirrors its trace's "
                         "location — written to the 'plots/' sibling of the trace's 'traces/' dir "
                         "(e.g. results/k1/QUAD/traces/foo.json -> results/k1/QUAD/plots/foo_att1.png), "
                         "so plots are organized per-(kv,etype) alongside vtk/ and traces/.")
    ap.add_argument("--run", default=None,
                    help="OVERRIDE the run-name subfolder. By default each plot is grouped under the trace's "
                         "own `run` stamp (written by run_test.jl = the results-DB basename), e.g. "
                         "results/k1/QUAD/plots/k1_quad_freeze/foo_att1.png — so you normally need NOTHING. "
                         "Pass --run only to force a different grouping name (e.g. for pre-2026-06-05 traces "
                         "that lack the stamp).")
    ap.add_argument("--cell", default=None, help="filter, e.g. Re=1e6,Da=1e-6,a0=0.5")
    ap.add_argument("--N", type=int, default=None, help="filter by mesh resolution N")
    ap.add_argument("--method", default=None, help="filter by method (ASGS/OSGS)")
    args = ap.parse_args()

    files = [args.file] if args.file else sorted(glob.glob(os.path.join(args.traces, "**", "traj_*.json"), recursive=True))
    if not files:
        print(f"[traj] no trace files in {args.traces}")
        return

    n = 0
    out_dirs = set()
    for f in files:
        try:
            with open(f) as fh:
                trace = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[traj] skip {os.path.basename(f)}: {e}")
            continue
        cell, N = trace["cell"], trace["N"]
        if not args.file and not _match(cell, N, args.cell, args.N, args.method):
            continue
        # Per-trace output dir: mirror the trace's location so plots sit under k<kv>/<etype>/plots/
        # alongside vtk/ and traces/ (consistent with run_test.jl's per-cell artifact layout). Each run's
        # plots are then grouped under a run-name subfolder, taken AUTOMATICALLY from the trace's own `run`
        # stamp (written by run_test.jl = the results-DB basename) — so you never have to pass a flag. `--run`
        # overrides the stamp; `--out` forces a single flat dir (ad-hoc use). Traces with no `run` stamp
        # (pre-2026-06-05) fall back to a flat plots/ dir.
        base_out = args.out or _plot_dir_for(f)
        run = args.run or trace.get("run")
        out_dir = os.path.join(base_out, run) if run else base_out
        # [per-config folder] each cell (Re,Da,α,kv,kp,etype,method) gets its OWN subfolder holding all of
        # its trace plots — every mesh N and every homotopy attempt — so one config's convergence story sits
        # in one place: plots/<run>/<cell_id>/. The cell id is the trace stem minus the leading "traj_" and
        # the trailing "_N<mesh>".
        base = os.path.splitext(os.path.basename(f))[0]
        cell_stem = base[len("traj_"):] if base.startswith("traj_") else base
        cell_id = re.sub(r"_N\d+$", "", cell_stem)
        out_dir = os.path.join(out_dir, cell_id)
        os.makedirs(out_dir, exist_ok=True)
        out_dirs.add(out_dir)
        title = _title(cell, N)
        attempts = trace.get("attempts", []) or []
        for ai, att in enumerate(attempts):
            ok = att.get("success", False)
            eps = att.get("eps_pert")
            eps_tag = ("%.0e" % eps) if isinstance(eps, (int, float)) else str(eps)
            sub = (rf"attempt {ai + 1}/{len(attempts)}:  $\varepsilon_{{\mathrm{{pert}}}}={tp.sci(eps)}$  "
                   + (r"$\checkmark$ converged" if ok else r"$\times$ failed")
                   + (r"   ($\varepsilon_{\mathrm{pert}}\!\downarrow$ fallback)" if ai > 0 else ""))
            out_path = os.path.join(out_dir, f"{base}_att{ai + 1}_eps{eps_tag}.png")
            tp.plot_attempt(att.get("stages", []), out_path, title=title, subtitle=sub,
                            subtitle_color=(GREEN if ok else RED),
                            osgs_outer=att.get("osgs_outer"),
                            base_conv_k=att.get("base_conv_outer_iter"),
                            mms_relchange=att.get("mms_relchange"), success=ok)
            print(f"[traj] {os.path.relpath(out_path, args.traces)}")
            n += 1
    if len(out_dirs) == 1:
        print(f"[traj] wrote {n} PNG(s) to {next(iter(out_dirs))}")
    else:
        print(f"[traj] wrote {n} PNG(s) across {len(out_dirs)} per-(kv,etype) plots/ dirs under {args.traces}")


if __name__ == "__main__":
    main()
