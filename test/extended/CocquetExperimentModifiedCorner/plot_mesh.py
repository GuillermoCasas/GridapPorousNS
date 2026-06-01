#!/usr/bin/env python3
"""Plot the structured meshes used by this Cocquet convergence study.

Cocquet convention: mesh-density N -> [2N, N] TRI cells over [0,2]x[0,1]
(see this dir's run_convergence.jl, "partition = [2*N, N]"). One PNG per
convergence level is written to results/meshes/. The [400,200] reference mesh
is huge, so it is skipped unless you pass --ref.

Usage:
    python plot_mesh.py                # all convergence levels
    python plot_mesh.py 20 40          # only N=20 and N=40
    python plot_mesh.py --show         # display the first interactively
    python plot_mesh.py --ref          # also render the reference mesh
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools", "mesh_viz"))
import _wrapper_common as wc  # noqa: E402

CONFIG = os.path.join(HERE, "data/paper_comparison_modified_corner.json")
OUT_DIR = os.path.join(HERE, "results", "meshes")

if __name__ == "__main__":
    wc.run_cartesian(CONFIG, OUT_DIR, wc.mp.cocquet_partition, sys.argv[1:])
