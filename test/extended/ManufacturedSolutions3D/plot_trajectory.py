#!/usr/bin/env python3
"""Render nonlinear-solver trajectory diagrams for the 3D MMS harness — one PNG per solve.

3D analog of test/extended/ManufacturedSolutions/plot_trajectory.py: a thin, test-specific wrapper
around the SHARED renderer tools/trajectory_viz/trajectory_plot.py (same tool, same schema, same
output layout as the 2D harness). It reads the per-solve trajectory JSON sidecars written by
smoke3d.jl under results/k<kv>/TET/traces/ (one per cell+method+mesh) and, for EACH attempt, writes
one PNG showing that solve's path through the nonlinear orchestration (Alg. O -> B/C -> A) with
per-iteration residual dots. The 3D solve is a single DIRECT exact-guess solve, so each trace has one
attempt with eps_pert = 0; ASGS and the coupled OSGS solve both render as a flat stage column.

Usage:
    python plot_trajectory.py                                   # every trace under results/
    python plot_trajectory.py --cell kv=2,method=OSGS --N 23
    python plot_trajectory.py --method ASGS
    python plot_trajectory.py --file results/k1/TET/traces/traj_....json
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
    (results/k<kv>/TET/traces/plots/), so each trajectory's JSON sidecar and its rendered PNG live
    together — mirroring the 2D harness layout. A run-name subfolder is appended by the caller."""
    return os.path.join(os.path.dirname(os.path.abspath(trace_path)), "plots")


def _match(cell, N, cell_sel, n_sel, method_sel):
    """Filter a trace by --cell (key=value, comma-separated), --N, --method (mirrors the 2D wrapper)."""
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
    nc = cell.get("ncells")
    tets = rf"   ({nc} tets)" if nc else ""
    return (rf"$Re={tp.fmt_pow(cell['Re'])}$, $Da={tp.fmt_pow(cell['Da'])}$, $\alpha_0={cell['alpha_0']:g}$"
            rf"   |   $\mathbb{{P}}_{{{cell['kv']}}}/\mathbb{{P}}_{{{cell['kp']}}}$ {cell['etype']} {cell['method']}"
            rf"   |   $N={N}${tets}")


def main():
    ap = argparse.ArgumentParser(description="3D MMS solver trajectory diagrams (one PNG per solve).")
    ap.add_argument("--traces", default=os.path.join(HERE, "results"),
                    help="root to search for trajectory JSON sidecars (default results/; recurses into "
                         "k<kv>/TET/traces/)")
    ap.add_argument("--file", default=None, help="plot a single trace JSON file")
    ap.add_argument("--out", default=None, help="force a single flat output dir (override the per-trace layout)")
    ap.add_argument("--run", default=None, help="override the run-name subfolder (default: the trace's `run` stamp)")
    ap.add_argument("--cell", default=None, help="filter, e.g. kv=2,method=OSGS or alpha0=0.5")
    ap.add_argument("--N", type=int, default=None, help="filter by effective resolution N=round(1/h)")
    ap.add_argument("--method", default=None, help="filter by method (ASGS/OSGS)")
    args = ap.parse_args()

    files = [args.file] if args.file else sorted(
        glob.glob(os.path.join(args.traces, "**", "traj_*.json"), recursive=True))
    if not files:
        print(f"[traj] no trace files in {args.traces} — run a sweep first (smoke3d.jl writes traces under "
              "results/k<kv>/TET/traces/)")
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
        # Install this trace's scale-free gate thresholds so the plotter draws the true tol_M/tol_C lines.
        tp.set_tols(trace.get("tol_M"), trace.get("tol_C"))
        base_out = args.out or _plot_dir_for(f)
        run = args.run or trace.get("run")
        out_dir = os.path.join(base_out, run) if run else base_out
        # Per-config folder: each cell gets its own subfolder (trace stem minus "traj_" and trailing "_N<..>").
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
            # 3D is a single direct exact-guess solve (eps_pert = 0); keep the subtitle explicit.
            sub = (r"direct exact-guess solve  ($\varepsilon_{\mathrm{pert}}=0$):  "
                   + (r"$\checkmark$ converged" if ok else r"$\times$ failed"))
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
        print(f"[traj] wrote {n} PNG(s) across {len(out_dirs)} per-cell plots/ dirs under {args.traces}")


if __name__ == "__main__":
    main()
