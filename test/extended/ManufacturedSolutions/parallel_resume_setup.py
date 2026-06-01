#!/usr/bin/env python3
"""
parallel_resume_setup.py — partition pending phase1_quad_k1 cells across workers.

Reads `results/phase1_quad_k1.h5` to identify cells whose N=320 row is still
missing.  Computes a projected wall-time for each pending cell using the
observed N=160 time and the median N=320/N=160 ratio measured on the α=1
cells already done (stratified by Reynolds band).  Greedy-partitions the
pending cells into N_WORKERS bins to roughly balance total projected cost,
then for each worker:

  * copies the master HDF5 to `results/phase1_quad_k1_w{k}.h5` so the
    harness's resume-aware logic naturally skips the 21 already-done cells;
  * writes `data/phase1_quad_k1_w{k}.json` — a copy of the master config
    with `h5_filename` overridden and `skip_cells` listing every (Re, Da, α)
    triple NOT assigned to worker k.

The orchestrator does NOT launch Julia; it just prepares files and prints
the four commands the user can run in parallel (one per terminal/screen,
or backgrounded).

Run from `test/extended/ManufacturedSolutions/`:
    python3 parallel_resume_setup.py [--n-workers 4]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = HERE / "data"
MASTER_H5 = RESULTS / "phase1_quad_k1.h5"
MASTER_JSON = DATA / "phase1_quad_k1.json"


def load_master_state():
    f = h5py.File(MASTER_H5, "r")
    done, pending = {}, {}
    re_ratios = {}  # Re-band -> list of (t320/t160) from α=1 done cells
    for k in sorted(f.keys()):
        g = f[k]
        et = g["eval_times"][:]
        Re = float(g.attrs["Re"]); Da = float(g.attrs["Da"]); a = float(g.attrs["alpha_0"])
        if len(et) == 6 and et[4] > 0:
            done[k] = (Re, Da, a, et[5], et[4])
            if abs(a - 1.0) < 1e-6:
                re_ratios.setdefault(Re, []).append(et[5] / et[4])
        else:
            pending[k] = (Re, Da, a, et[4] if len(et) >= 5 else 0.0)
    f.close()
    # Median ratio per Re band; fallback 30x if a band is empty.
    median_ratio = {re: float(np.median(rs)) for re, rs in re_ratios.items()}
    return done, pending, median_ratio


def project_cost(re_val, t160, ratios, fallback=30.0):
    r = ratios.get(re_val, fallback)
    return r * t160


def partition_greedy(pending_with_cost, n_workers):
    """Greedy LPT bin-packing: assign each cell to the currently-emptiest bin."""
    bins = [[] for _ in range(n_workers)]
    bin_totals = [0.0] * n_workers
    # Heaviest first
    sorted_cells = sorted(pending_with_cost, key=lambda x: -x[1])
    for cell_key, cost in sorted_cells:
        # Pick the bin with the lowest current total
        k = min(range(n_workers), key=lambda i: bin_totals[i])
        bins[k].append((cell_key, cost))
        bin_totals[k] += cost
    return bins, bin_totals


def build_assignment_by_triple(pending, bins):
    """Group cells per worker by their (Re, Da, alpha) triple, dropping method."""
    # pending[cell_key] = (Re, Da, alpha, t160)
    worker_triples = []  # list of sets of (Re, Da, alpha) per worker
    for bin_cells in bins:
        triples = set()
        for cell_key, _ in bin_cells:
            Re, Da, a, _ = pending[cell_key]
            triples.add((Re, Da, a))
        worker_triples.append(triples)
    return worker_triples


def write_worker_artifacts(worker_idx, assigned_triples, all_grid_triples):
    """Per worker: copy master HDF5, write JSON config with skip_cells."""
    worker_h5 = RESULTS / f"phase1_quad_k1_w{worker_idx}.h5"
    worker_json = DATA / f"phase1_quad_k1_w{worker_idx}.json"

    # Copy master HDF5 → worker HDF5 (so harness resume-aware skips already-done cells).
    shutil.copy2(MASTER_H5, worker_h5)

    # Load master JSON and customize.
    with open(MASTER_JSON, "r") as f:
        cfg = json.load(f)

    # Override h5 filename.
    cfg["h5_filename"] = worker_h5.name

    # skip_cells = every (Re, Da, α) triple in the grid NOT assigned to this worker.
    skip_list = []
    for (Re, Da, a) in sorted(all_grid_triples):
        if (Re, Da, a) not in assigned_triples:
            skip_list.append([Re, Da, a])
    # Preserve any existing skip_cells from master (cells already excluded by config).
    existing_skip = cfg.get("skip_cells", [])
    for entry in existing_skip:
        triple = (float(entry[0]), float(entry[1]), float(entry[2]))
        if triple not in assigned_triples and list(triple) not in [list(s) for s in skip_list]:
            skip_list.append(list(triple))
    cfg["skip_cells"] = skip_list

    # erase_past_results MUST be False so the harness preserves the pre-populated done cells.
    cfg["erase_past_results"] = False

    with open(worker_json, "w") as f:
        json.dump(cfg, f, indent=4)

    return worker_h5, worker_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    args = ap.parse_args()

    done, pending, ratios = load_master_state()
    print(f"Master state: {len(done)} done @ N=320, {len(pending)} pending")
    print(f"Re-band ratios (t320/t160, from α=1 done): {ratios}")

    # Build the full grid of (Re, Da, α) triples from master + pending.
    all_triples = set()
    for k, (Re, Da, a, *_rest) in {**done, **pending}.items():
        all_triples.add((Re, Da, a))
    print(f"Grid: {len(all_triples)} (Re, Da, α) triples")

    # Project cost per pending cell.
    pending_with_cost = []
    for cell_key, (Re, Da, a, t160) in pending.items():
        cost = project_cost(Re, t160, ratios)
        pending_with_cost.append((cell_key, cost))

    total_projected = sum(c for _, c in pending_with_cost)
    print(f"\nTotal projected cost for 27 pending cells: {total_projected:.0f}s = {total_projected/3600:.1f}h")

    # Greedy partition.
    bins, totals = partition_greedy(pending_with_cost, args.n_workers)

    print(f"\nPartition across {args.n_workers} workers (greedy LPT):")
    for i, (bin_cells, total) in enumerate(zip(bins, totals)):
        triples = set()
        for k, _ in bin_cells:
            Re, Da, a, _ = pending[k]
            triples.add((Re, Da, a))
        print(f"  worker {i}: {len(bin_cells):2d} cells across {len(triples)} (Re,Da,α) triples | projected = {total/3600:6.2f}h")
        # Show cells in order of cost
        for k, cost in sorted(bin_cells, key=lambda x: -x[1]):
            Re, Da, a, t160 = pending[k]
            print(f"    {k:18s} Re={Re:7.0e} Da={Da:7.0e} α={a:5.2f} | t160={t160/60:5.1f}min, proj t320={cost/60:7.1f}min")

    longest = max(totals)
    print(f"\nLongest worker projected: {longest/3600:.2f}h  (= wall-clock if all start together)")

    # Build per-worker assignment by triple.
    worker_triples = build_assignment_by_triple(pending, bins)

    # Write worker artifacts.
    print(f"\nWriting per-worker artifacts...")
    cmds = []
    for i in range(args.n_workers):
        h5, jsn = write_worker_artifacts(i, worker_triples[i], all_triples)
        cmds.append((i, h5.name, jsn.name))
        print(f"  worker {i}: results/{h5.name} + data/{jsn.name}")

    # Print launch commands.
    print(f"\n{'='*72}\nLaunch commands (run from {HERE}):\n")
    for i, h5n, jsn in cmds:
        print(f"  julia --project=../../.. run_test.jl {jsn} > logs/phase1_w{i}.log 2>&1 &")
    print(f"  wait  # all four workers in parallel")
    print(f"\nAfter all four finish, merge with:\n  python3 parallel_resume_merge.py")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
