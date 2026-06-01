#!/usr/bin/env python3
"""
parallel_resume_merge.py — merge per-worker HDF5 outputs back into master.

After the four `phase1_quad_k1_w{k}.h5` workers complete, this script walks
each worker's HDF5 and for every `config_*` group whose `eval_times` row is
LONGER than the master's (i.e., the worker advanced this cell beyond what
master had), copies the worker's group over the master's.

The merge is conservative: it never overwrites a cell whose master entry
already has more (or equal) N values than the worker's.  Worker cells that
were not assigned (and therefore unchanged by the worker) are skipped
silently because their length matches master's.

Run from `test/extended/ManufacturedSolutions/`:
    python3 parallel_resume_merge.py [--n-workers 4] [--dry-run]
"""
import argparse
import shutil
from pathlib import Path

import h5py

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MASTER_H5 = RESULTS / "phase1_quad_k1.h5"


def merge_one_worker(master_h5_path, worker_h5_path, dry_run=False):
    """Copy each worker config group that is more advanced than master's."""
    updates = []
    with h5py.File(worker_h5_path, "r") as wf, h5py.File(master_h5_path, "r" if dry_run else "r+") as mf:
        for key in sorted(wf.keys()):
            wg = wf[key]
            if not hasattr(wg, "keys") or "eval_times" not in wg:
                continue
            wn = wg["eval_times"].shape[0]
            if key not in mf:
                updates.append((key, 0, wn))
                if not dry_run:
                    mf.copy(wg, key)
                continue
            mg = mf[key]
            mn = mg["eval_times"].shape[0] if "eval_times" in mg else 0
            if wn > mn:
                updates.append((key, mn, wn))
                if not dry_run:
                    del mf[key]
                    mf.copy(wg, key)
    return updates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Pre-merge safety backup.
    safety_backup = MASTER_H5.with_suffix(".h5.bak_pre_merge")
    if not args.dry_run:
        shutil.copy2(MASTER_H5, safety_backup)
        print(f"Safety backup written to {safety_backup}")

    total_updates = []
    for k in range(args.n_workers):
        wpath = RESULTS / f"phase1_quad_k1_w{k}.h5"
        if not wpath.exists():
            print(f"  worker {k}: {wpath.name} missing — SKIPPED")
            continue
        ups = merge_one_worker(MASTER_H5, wpath, dry_run=args.dry_run)
        print(f"  worker {k}: {len(ups)} cells advanced")
        for key, m_old, m_new in ups:
            print(f"    {key:20s}  {m_old} -> {m_new}  (N entries)")
        total_updates.extend(ups)

    print(f"\nTotal cells advanced: {len(total_updates)} ({'DRY RUN' if args.dry_run else 'COMMITTED'})")
    print(f"Master HDF5 path: {MASTER_H5}")


if __name__ == "__main__":
    main()
