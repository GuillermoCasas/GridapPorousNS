#!/usr/bin/env python3
"""Plot the meshes used by a Cocquet tube-flow variant.

One consolidated mesh plotter for the unified test. Pass a variant name (a folder under
data/); the script reads that variant's config and dispatches to the right renderer:

  * config has a non-empty "freefem_mesh_dir"  -> render the on-disk FreeFem .msh sequence
  * config has "mesh_generator": "UNSTRUCTURED" -> gmsh meshes are generated at solve time,
                                                   nothing on disk to plot (skipped with a note)
  * otherwise (structured Cartesian variants)   -> render the [2N, N] TRI meshes from the config

PNGs are written to results/<variant>/meshes/.

Usage:
    python plot_mesh.py structured            # all convergence levels of the structured variant
    python plot_mesh.py structured 20 40      # only N=20, N=40
    python plot_mesh.py freefem_meshes        # the FreeFem .msh sequence
    python plot_mesh.py structured --show     # display the first interactively
"""
import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools", "mesh_viz"))
import _wrapper_common as wc  # noqa: E402

MS_DIR = os.path.join(HERE, "data")


def _variant_config(variant):
    cfgs = glob.glob(os.path.join(MS_DIR, variant, "*.json"))
    if not cfgs:
        raise SystemExit(f"No config JSON found in data/{variant}/")
    return cfgs[0]


def main(argv):
    if not argv or argv[0].startswith('-'):
        raise SystemExit("Usage: python plot_mesh.py <variant> [N ...] [--show]\n"
                         f"variants: {', '.join(sorted(os.listdir(MS_DIR)))}")
    variant, rest = argv[0], argv[1:]
    config = _variant_config(variant)
    out_dir = os.path.join(HERE, "results", variant, "meshes")
    with open(config) as f:
        cfg = json.load(f)

    freefem_dir = cfg.get("freefem_mesh_dir", "")
    if freefem_dir:
        msh_dir = os.path.join(os.path.dirname(config), freefem_dir)
        wc.run_freefem(msh_dir, out_dir, rest)
    elif cfg.get("mesh_generator", "STRUCTURED") == "UNSTRUCTURED":
        print(f"[{variant}] mesh_generator=UNSTRUCTURED: gmsh meshes are generated at solve time "
              f"(no .msh files on disk), so there is nothing to render here.")
    else:
        wc.run_cartesian(config, out_dir, wc.mp.cocquet_partition, rest)


if __name__ == "__main__":
    main(sys.argv[1:])
