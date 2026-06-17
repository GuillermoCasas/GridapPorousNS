#!/usr/bin/env python3
"""Draw the 3D tetrahedral meshes exported by mesh3d.jl (HDF5) using matplotlib.

Each results/mesh/mesh_level{k}.h5 holds:
  node_coords : (N,3) vertex coordinates   (Julia wrote (3,N) -> h5py reads (N,3))
  tets        : (K,4) tet connectivity, 1-BASED node indices (we subtract 1)
  attrs: nnodes, ncells, hmean

We derive the BOUNDARY surface (triangular faces belonging to exactly one tet) and render it,
so the refinement of the unstructured tet mesh is visible level by level.

Usage:
  python plot_mesh3d.py                          # all results/mesh/mesh_level*.h5 in one figure
  python plot_mesh3d.py --dir results/mesh --out results/mesh/mesh_levels.png
"""
import argparse, glob, os, re
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
_TET_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def boundary_faces(tets):
    """Triangular faces appearing in exactly one tet = the mesh boundary surface."""
    count = {}
    for t in tets:
        for combo in _TET_FACES:
            f = tuple(sorted((int(t[combo[0]]), int(t[combo[1]]), int(t[combo[2]]))))
            count[f] = count.get(f, 0) + 1
    return [f for f, c in count.items() if c == 1]


def load(path):
    with h5py.File(path, "r") as f:
        coords = np.asarray(f["node_coords"][:], dtype=float)   # (N,3)
        tets = np.asarray(f["tets"][:], dtype=np.int64) - 1      # (K,4) -> 0-based
        attrs = {k: f.attrs[k] for k in f.attrs}
    if coords.shape[0] == 3 and coords.shape[1] != 3:           # safety: undo if (3,N)
        coords = coords.T
    if tets.shape[1] != 4 and tets.shape[0] == 4:
        tets = tets.T
    return coords, tets, attrs


def draw(ax, coords, tets, title):
    bf = boundary_faces(tets)
    polys = [coords[list(f)] for f in bf]
    pc = Poly3DCollection(polys, facecolor="#9ecae1", edgecolor="k", linewidths=0.15, alpha=0.55)
    ax.add_collection3d(pc)
    mn, mx = coords.min(0), coords.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    try:
        ax.set_box_aspect((mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]))
    except Exception:
        pass
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-58)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(HERE, "results", "mesh"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "mesh_level*.h5")),
                   key=lambda p: int(re.search(r"level(\d+)", p).group(1)))
    if not files:
        raise SystemExit(f"No mesh_level*.h5 in {args.dir}")
    out = args.out or os.path.join(args.dir, "mesh_levels.png")

    n = len(files)
    fig = plt.figure(figsize=(5.2 * n, 5.0))
    for i, path in enumerate(files):
        coords, tets, attrs = load(path)
        lvl = int(re.search(r"level(\d+)", path).group(1))
        hmean = float(attrs.get("hmean", np.nan)); ncells = int(attrs.get("ncells", len(tets)))
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        draw(ax, coords, tets, f"level {lvl}\n{ncells} tets, h̄={hmean:.4g}")
        print(f"[plot] level {lvl}: {ncells} tets, {coords.shape[0]} nodes, hmean={hmean:.4g}")
    fig.suptitle("3D nested tetrahedral mesh family (boundary surface)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
