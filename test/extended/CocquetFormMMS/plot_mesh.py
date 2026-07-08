#!/usr/bin/env python3
"""Plot the structured meshes used by the CocquetForm MMS study.

This sibling is manufactured-solution based: density N -> [N, N] TRI cells over
the origin-centred box [-0.5,0.5]^2 (square convention, unlike the [2N,N]
Cocquet benchmark). One PNG per convergence level is written to results/meshes/.

Usage:
    python plot_mesh.py                # all convergence levels of cocquet_form_mms_vms.json
    python plot_mesh.py 20 40          # only N=20 and N=40
    python plot_mesh.py --show         # display the first interactively
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools", "mesh_viz"))
import _wrapper_common as wc  # noqa: E402

CONFIG = os.path.join(HERE, "data", "cocquet_form_mms_vms.json")
OUT_DIR = os.path.join(HERE, "results", "meshes")

if __name__ == "__main__":
    wc.run_cartesian(CONFIG, OUT_DIR, wc.mp.square_partition, sys.argv[1:])
