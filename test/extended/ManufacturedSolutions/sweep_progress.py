#!/usr/bin/env python3
"""Progress of the MMS sweep, by introspecting the new-layout DBs (results/k<kv>/<etype>/results.h5).

Method-aware: ASGS is REUSED (migrated from the pre-JFNK DBs — shown as done), so this focuses on the
OSGS re-run that actually computes. For each family it reports, per mesh N, how many OSGS cells have
reached it, against the OSGS target (the config grid minus skip_cells = 24 cells). Robust to cells whose
N-ladder exceeds the config (the direct-solve corner cells go to N=512/768; migrated QUAD-k1 ASGS to 640)
— it reports the ACTUAL N values present, never synthetic buckets. Works at any time; just reads the DBs.
Usage:  python3 sweep_progress.py
"""
import json, os, h5py, numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FAMILIES = [  # (new-layout subdir, config that defines the OSGS grid)
    ("k1/TRI ", "phase1_tri_k1.json"),
    ("k2/QUAD", "phase1_quad_k2.json"),
    ("k1/QUAD", "phase1_quad_k1.json"),
]

def osgs_grid(cfg_path):
    d = json.load(open(cfg_path))
    pp, dom, nm = d["physical_properties"], d["domain"], d["numerical_method"]
    skip = {tuple(round(float(x), 10) for x in s) for s in d.get("skip_cells", [])}
    ncells = sum(1 for a in dom["alpha_0"] for da in pp["Da"] for r in pp["Re"]
                 if (round(float(r),10), round(float(da),10), round(float(a),10)) not in skip)
    Ns = sorted(int(n) for n in nm["mesh"]["convergence_partitions"])
    return ncells, Ns

def counts_by_method(db):
    by = defaultdict(Counter); tot = Counter()
    f = h5py.File(db, "r", swmr=True)
    try:
        for k in f:
            if "method" not in f[k].attrs: continue
            m = f[k].attrs["method"]; m = m.decode() if isinstance(m, bytes) else m
            tot[m] += 1
            for x in np.array(f[k]["h"]):
                by[m][int(round(1/x))] += 1
    finally:
        f.close()
    return tot, by

print("MMS sweep progress  (ASGS = migrated/done; OSGS = the re-run actually computing)")
print("=" * 84)
for rel, cfg in FAMILIES:
    db = os.path.join(RES, rel.strip(), "results.h5")
    ntarget, Ns = osgs_grid(os.path.join(HERE, "data", cfg))
    if not os.path.isfile(db):
        print(f"{rel} | DB not created yet  (OSGS target: {ntarget} cells x N{Ns})"); continue
    try:
        tot, by = counts_by_method(db)
    except OSError:
        print(f"{rel} | (DB locked mid-write — retry in a moment)"); continue
    osgs_done = sum(1 for n in [Ns[-1]] for _ in range(by["OSGS"].get(Ns[-1], 0)))  # cells at finest N
    perN = "  ".join(f"N{n}:{by['OSGS'].get(n,0)}/{ntarget}" for n in Ns)
    osgs_started = tot.get("OSGS", 0)
    status = "DONE" if by["OSGS"].get(Ns[-1], 0) >= ntarget else ("running" if osgs_started else "pending")
    print(f"{rel} | OSGS {status:7} {osgs_done}/{ntarget} at N{Ns[-1]} | {perN}")
    # ASGS one-liner (migrated): show count + any extra N beyond the OSGS grid (corner 512/768, k1 640)
    extra = sorted(n for n in by["ASGS"] if n not in Ns)
    extra_s = f" (+ extra N {extra} from corner/640)" if extra else ""
    print(f"{'':8} | ASGS migrated: {tot.get('ASGS',0)} cells (done){extra_s}")
print("=" * 84)
print("OSGS target = config grid minus skip_cells (24). ASGS shows >24 where corner cells are merged in.")
