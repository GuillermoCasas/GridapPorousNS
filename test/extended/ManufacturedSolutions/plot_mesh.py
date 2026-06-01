#!/usr/bin/env python3
"""Plot the structured meshes used by the Manufactured Solutions sweep.

MMS convention: density N -> [N, N] cells over the origin-centred box
[-0.5,0.5]^2 (h = L/N). The config sweeps BOTH element types (QUAD and the
simplexified TRI), so both are rendered for each level. One PNG per
(element-type, level) is written to results/meshes/.

Usage:
    python plot_mesh.py                # all convergence levels, both QUAD & TRI
    python plot_mesh.py 20 40          # only N=20 and N=40
    python plot_mesh.py --show         # display the first interactively
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools", "mesh_viz"))
import _wrapper_common as wc  # noqa: E402

CONFIG = os.path.join(HERE, "data", "test_config.json")
OUT_DIR = os.path.join(HERE, "results", "meshes")

if __name__ == "__main__":
    # element_types=None -> taken from the config's ["QUAD", "TRI"] list.
    wc.run_cartesian(CONFIG, OUT_DIR, wc.mp.square_partition, sys.argv[1:])
