#!/usr/bin/env python3
"""Draw the 3D tetrahedral meshes exported by mesh3d.jl (HDF5) using matplotlib.

Each results/mesh/mesh_level{k}.h5 holds:
  node_coords : (N,3) vertex coordinates   (Julia wrote (3,N) -> h5py reads (N,3))
  tets        : (K,4) tet connectivity, 1-BASED node indices (we subtract 1)
  attrs: nnodes, ncells, hmean

We derive the BOUNDARY surface (triangular faces belonging to exactly one tet) and render it,
so the refinement of the unstructured tet mesh is visible level by level.

Mesh sequences live under results/meshes/<sequence>/ (e.g. structured, nested_red, frontal), each
holding that sequence's mesh_level*.h5 and a mesh_mosaic.png drawn here.

Usage:
  python plot_mesh3d.py                          # mosaic for EVERY sequence under results/meshes/*/
  python plot_mesh3d.py structured               # mosaic for results/meshes/structured/
  python plot_mesh3d.py --dir <d> --out <p.png>  # explicit dir/output (back-compat)
"""
import argparse, glob, os, re
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
# Render ALL text (axis labels, tick numbers, titles, symbols) with real LaTeX for consistency.
try:
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
except Exception:
    pass
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
    # only the first and last tick per axis (domain endpoints) — less clutter
    ax.set_xticks([mn[0], mx[0]]); ax.set_yticks([mn[1], mx[1]]); ax.set_zticks([mn[2], mx[2]])
    ax.tick_params(labelsize=12)
    try:
        ax.set_box_aspect((mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]))
    except Exception:
        pass
    ax.set_title(title, fontsize=14, y=0.90)   # closer to the mesh (less title-to-mesh gap)
    ax.set_xlabel(r"$x$", fontsize=14); ax.set_ylabel(r"$y$", fontsize=14); ax.set_zlabel(r"$z$", fontsize=14)
    ax.view_init(elev=22, azim=-58)


def build_mosaic(seq_dir, out=None, seq_name=None):
    """Mosaic of all mesh_level*.h5 in seq_dir -> seq_dir/mesh_mosaic.png. Returns True if drawn."""
    files = sorted(glob.glob(os.path.join(seq_dir, "mesh_level*.h5")),
                   key=lambda p: int(re.search(r"level(\d+)", p).group(1)))
    if not files:
        return False
    out = out or os.path.join(seq_dir, "mesh_mosaic.png")
    seq_name = seq_name or os.path.basename(os.path.normpath(seq_dir))
    # 2-column mosaic with tight gaps — 3D axes reserve a lot of whitespace, so pull panels
    # together with negative hspace (no content overlap).
    n = len(files); ncols = 2; nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.8 * ncols, 4.4 * nrows))
    for i, path in enumerate(files):
        coords, tets, attrs = load(path)
        lvl = int(re.search(r"level(\d+)", path).group(1))
        hmean = float(attrs.get("hmean", np.nan)); ncells = int(attrs.get("ncells", len(tets)))
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        draw(ax, coords, tets, fr"level {lvl}:  {ncells} tets,  $\bar{{h}} = {hmean:.4g}$")
        print(f"[plot] {seq_name} level {lvl}: {ncells} tets, {coords.shape[0]} nodes, hmean={hmean:.4g}")
    seq_label = seq_name.replace("_", r"\_")
    fig.suptitle(fr"3D tetrahedral mesh sequence: \texttt{{{seq_label}}} (boundary surface)", fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.15, hspace=-0.05)
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="?", default=None,
                    help="mesh-sequence tag under results/meshes/ (omit = every sequence)")
    ap.add_argument("--dir", default=None, help="explicit sequence dir (overrides positional)")
    ap.add_argument("--out", default=None, help="explicit output PNG (with --dir)")
    args = ap.parse_args()
    meshes_root = os.path.join(HERE, "results", "meshes")

    if args.dir:
        if not build_mosaic(args.dir, out=args.out):
            raise SystemExit(f"No mesh_level*.h5 in {args.dir}")
    elif args.sequence:
        seq_dir = os.path.join(meshes_root, args.sequence)
        if not build_mosaic(seq_dir):
            raise SystemExit(f"No mesh_level*.h5 in {seq_dir} — export it first (julia mesh3d.jl {args.sequence})")
    else:
        # no arg: a mosaic for EVERY sequence that has exported meshes
        seqs = sorted(d for d in glob.glob(os.path.join(meshes_root, "*"))
                      if glob.glob(os.path.join(d, "mesh_level*.h5")))
        if not seqs:
            raise SystemExit(f"No mesh sequences under {meshes_root} — export one (julia mesh3d.jl <sequence>)")
        for d in seqs:
            build_mosaic(d)


if __name__ == "__main__":
    main()
