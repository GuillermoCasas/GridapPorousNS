#!/usr/bin/env python3
"""
mesh_plot.py — visualise the meshes used by the porous-NS tests.

Two mesh families are supported, matching the two ways meshes enter the solver
(see ``src/run_simulation.jl``):

  1. Structured Cartesian meshes built procedurally from a JSON config's
     ``bounding_box`` + ``partition`` + ``element_type`` ("QUAD" or "TRI",
     the latter being the Cartesian mesh ``simplexify``-d into right triangles).

  2. Explicit unstructured FreeFem ``.msh`` files (the Cocquet irregular-mesh
     experiments), parsed in the same ``nv nt nbe`` layout that
     ``freefem_to_gmsh`` consumes.

The figure is annotated with the data a reader actually wants when looking at a
mesh: element type, element / vertex counts, domain extent, mean cell area, and
the **average element characteristic length** ``h_K`` — computed with the SAME
formula the solver uses for the stabilization parameter τ:

    h_K = sqrt(area_K)        for QUAD cells
    h_K = sqrt(2 * area_K)    for TRI cells (simplexified or unstructured)

For FreeFem meshes the paper's reported ``h = sqrt(2)/N`` is also shown when the
mesh-density ``N`` can be recovered from the filename (``..._N40.msh``).

This module is meant to be imported by the tiny per-test wrapper scripts, but it
is also a self-contained CLI:

    # Cartesian mesh straight from a config (uses that config's own partition):
    python mesh_plot.py --config path/to/config.json

    # ... overriding the partition (e.g. one convergence level, Cocquet 2N x N):
    python mesh_plot.py --config cfg.json --partition 80 40

    # Explicit FreeFem mesh file:
    python mesh_plot.py --msh path/to/mesh_fig2_N40.msh

    # Save instead of showing interactively:
    python mesh_plot.py --msh foo.msh --out foo.png
"""

import argparse
import json
import os
import re

import numpy as np
import matplotlib

# Use a non-interactive backend automatically when there is no display / an
# output file is requested; resolved again in `show_or_save`.
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection


# --------------------------------------------------------------------------- #
# Geometry builders
# --------------------------------------------------------------------------- #
def cartesian_mesh(bounding_box, partition, element_type):
    """Build a structured Cartesian mesh.

    Parameters
    ----------
    bounding_box : (x0, x1, y0, y1)
    partition    : (nx, ny) number of cells in each direction
    element_type : "QUAD" or "TRI"

    Returns
    -------
    verts : (Nv, 2) float array of node coordinates
    cells : list of index arrays (length-4 for QUAD, length-3 for TRI)
    meta  : dict with dx, dy and the cell area
    """
    x0, x1, y0, y1 = map(float, bounding_box)
    nx, ny = int(partition[0]), int(partition[1])
    if nx <= 0 or ny <= 0:
        raise ValueError(f"partition must be positive, got {partition}")

    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")  # shape (nx+1, ny+1)
    verts = np.column_stack([gx.ravel(), gy.ravel()])

    def vid(i, j):
        return i * (ny + 1) + j

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    cells = []
    et = element_type.upper()
    for i in range(nx):
        for j in range(ny):
            bl = vid(i, j)
            br = vid(i + 1, j)
            tr = vid(i + 1, j + 1)
            tl = vid(i, j + 1)
            if et == "QUAD":
                cells.append(np.array([bl, br, tr, tl]))
            elif et == "TRI":
                # simplexify each quad into two right triangles along the
                # bottom-left -> top-right diagonal. Both halves have equal
                # area, so the per-cell h_K is identical regardless of the
                # diagonal chosen.
                cells.append(np.array([bl, br, tr]))
                cells.append(np.array([bl, tr, tl]))
            else:
                raise ValueError(f"element_type must be QUAD or TRI, got {element_type!r}")

    meta = {"dx": dx, "dy": dy, "cell_area_quad": dx * dy}
    return verts, cells, meta


def read_freefem_msh(path):
    """Parse a FreeFem ``savemesh`` ``.msh`` file.

    Layout (1-based vertex indices), matching ``freefem_to_gmsh``:
        line 1            : nv nt nbe
        next nv lines     : x y label
        next nt lines     : v1 v2 v3 region
        next nbe lines    : e1 e2 label   (1=inlet, 2=outlet, 3/4=walls)

    Returns
    -------
    verts : (nv, 2) float array
    tris  : (nt, 3) int array, 0-based
    edges : list of (i, j, label) with 0-based vertex indices
    """
    with open(path, "r") as f:
        tokens = f.read().split()
    it = iter(tokens)

    def nexti():
        return int(next(it))

    def nextf():
        return float(next(it))

    nv, nt, nbe = nexti(), nexti(), nexti()

    verts = np.empty((nv, 2), dtype=float)
    for k in range(nv):
        verts[k, 0] = nextf()
        verts[k, 1] = nextf()
        next(it)  # vertex label (unused)

    tris = np.empty((nt, 3), dtype=int)
    for k in range(nt):
        tris[k, 0] = nexti() - 1
        tris[k, 1] = nexti() - 1
        tris[k, 2] = nexti() - 1
        next(it)  # region label (unused)

    edges = []
    for _ in range(nbe):
        a = nexti() - 1
        b = nexti() - 1
        lab = nexti()
        edges.append((a, b, lab))

    return verts, tris, edges


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _poly_area(coords):
    """Shoelace area (absolute) of a polygon given as (n, 2) coords."""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def cell_characteristic_lengths(verts, cells, element_type):
    """Per-cell h_K using the solver's own definition.

    h_K = sqrt(area)      for QUAD
    h_K = sqrt(2 * area)  for TRI
    """
    et = element_type.upper()
    factor = 2.0 if et == "TRI" else 1.0
    areas = np.array([_poly_area(verts[c]) for c in cells])
    h = np.sqrt(factor * areas)
    return h, areas


def mesh_stats(verts, cells, element_type, h, areas):
    """Assemble the annotation dictionary shown on the figure."""
    x0, x1 = verts[:, 0].min(), verts[:, 0].max()
    y0, y1 = verts[:, 1].min(), verts[:, 1].max()
    return {
        "element_type": element_type.upper(),
        "n_cells": len(cells),
        "n_vertices": len(verts),
        "domain": (x0, x1, y0, y1),
        "h_mean": float(np.mean(h)),
        "h_min": float(np.min(h)),
        "h_max": float(np.max(h)),
        "area_mean": float(np.mean(areas)),
        "area_total": float(np.sum(areas)),
    }


def _stats_text(stats, extra=None):
    x0, x1, y0, y1 = stats["domain"]
    lines = [
        f"element type : {stats['element_type']}",
        f"elements     : {stats['n_cells']:,}",
        f"vertices     : {stats['n_vertices']:,}",
        f"domain       : [{x0:g}, {x1:g}] x [{y0:g}, {y1:g}]",
        f"avg h_K      : {stats['h_mean']:.4g}",
        f"  (min/max)  : {stats['h_min']:.4g} / {stats['h_max']:.4g}",
        f"avg cell area: {stats['area_mean']:.4g}",
    ]
    if extra:
        lines.extend(extra)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _draw_cells(ax, verts, cells, fill, lw):
    polys = [verts[c] for c in cells]
    # For big meshes, drawing edge lines is far cheaper than filled patches.
    if fill and len(cells) <= 5000:
        pc = PolyCollection(
            polys, facecolors="#cfe3f7", edgecolors="#1f4e79", linewidths=lw
        )
        ax.add_collection(pc)
    else:
        segs = []
        for p in polys:
            n = len(p)
            for k in range(n):
                segs.append([p[k], p[(k + 1) % n]])
        lc = LineCollection(segs, colors="#1f4e79", linewidths=lw)
        ax.add_collection(lc)


def _draw_freefem_boundary(ax, verts, edges):
    """Colour the labelled FreeFem boundary edges (inlet/outlet/walls)."""
    colors = {1: "#2ca02c", 2: "#d62728", 3: "#7f7f7f", 4: "#7f7f7f"}
    labels = {1: "inlet", 2: "outlet", 3: "walls", 4: "walls"}
    drawn = set()
    by_color = {}
    for a, b, lab in edges:
        by_color.setdefault(lab, []).append([verts[a], verts[b]])
    for lab, segs in by_color.items():
        c = colors.get(lab, "#000000")
        name = labels.get(lab, f"label {lab}")
        lbl = name if name not in drawn else None
        drawn.add(name)
        ax.add_collection(LineCollection(segs, colors=c, linewidths=2.0, label=lbl))


def plot_mesh(
    verts,
    cells,
    element_type,
    title="",
    extra_stats=None,
    edges=None,
    fill=True,
    out=None,
):
    """Draw a mesh + annotation box; show or save the figure."""
    h, areas = cell_characteristic_lengths(verts, cells, element_type)
    stats = mesh_stats(verts, cells, element_type, h, areas)

    fig, ax = plt.subplots(figsize=(10, 6))
    lw = 0.4 if len(cells) > 3000 else 0.8
    _draw_cells(ax, verts, cells, fill=fill, lw=lw)
    if edges:
        _draw_freefem_boundary(ax, verts, edges)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    x0, x1, y0, y1 = stats["domain"]
    pad_x = 0.03 * (x1 - x0 or 1.0)
    pad_y = 0.03 * (y1 - y0 or 1.0)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    ax.text(
        0.01,
        0.99,
        _stats_text(stats, extra_stats),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#888"),
    )

    fig.tight_layout()
    show_or_save(fig, out)
    return stats


def show_or_save(fig, out):
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[mesh_plot] wrote {out}")
    else:
        try:
            plt.show()
        except Exception:
            fallback = "mesh.png"
            fig.savefig(fallback, dpi=150, bbox_inches="tight")
            print(f"[mesh_plot] no display; wrote {fallback}")


# --------------------------------------------------------------------------- #
# Convenience entry points used by the per-test wrappers
# --------------------------------------------------------------------------- #
def plot_cartesian_from_config(
    config_path, partition=None, element_type=None, title=None, out=None, fill=True
):
    """Plot a structured mesh described by a JSON config.

    The config supplies ``domain.bounding_box`` and, unless overridden,
    ``numerical_method.mesh.{partition, element_type}``. Pass an explicit
    ``partition`` to render a single convergence level (each test owns its own
    N -> partition convention, e.g. Cocquet's ``[2N, N]``).
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    bbox = cfg["domain"]["bounding_box"]
    mesh_cfg = cfg["numerical_method"]["mesh"]
    if partition is None:
        partition = mesh_cfg["partition"]
    if element_type is None:
        et = mesh_cfg.get("element_type", "QUAD")
        # configs may carry a list of element types for a sweep; take the first.
        element_type = et[0] if isinstance(et, list) else et

    verts, cells, meta = cartesian_mesh(bbox, partition, element_type)
    extra = [
        f"partition    : {int(partition[0])} x {int(partition[1])}",
        f"dx x dy      : {meta['dx']:.4g} x {meta['dy']:.4g}",
    ]
    if title is None:
        title = f"{os.path.basename(config_path)} — {element_type} {int(partition[0])}x{int(partition[1])}"
    return plot_mesh(
        verts, cells, element_type, title=title, extra_stats=extra, fill=fill, out=out
    )


def plot_freefem(msh_path, title=None, out=None, fill=False):
    """Plot an explicit FreeFem ``.msh`` mesh file."""
    verts, tris, edges = read_freefem_msh(msh_path)
    cells = [tris[k] for k in range(tris.shape[0])]

    extra = []
    m = re.search(r"_N(\d+)", os.path.basename(msh_path))
    if m:
        N = int(m.group(1))
        extra.append(f"density N    : {N}")
        extra.append(f"h = sqrt2/N  : {np.sqrt(2.0) / N:.4g}  (paper)")

    if title is None:
        title = os.path.basename(msh_path)
    return plot_mesh(
        verts,
        cells,
        "TRI",
        title=title,
        extra_stats=extra,
        edges=edges,
        fill=fill,
        out=out,
    )


# --------------------------------------------------------------------------- #
# Sweep helpers for the per-test wrappers
# --------------------------------------------------------------------------- #
# Each test owns its own "mesh-density N -> Cartesian partition" convention.
# Encoding it as a named rule keeps the wrapper scripts trivial while making the
# convention explicit at the call site.
def square_partition(N):
    """MMS / unit-square convention: N x N cells (h = L/N)."""
    return (N, N)


def cocquet_partition(N):
    """Cocquet [0,2]x[0,1] convention: 2N x N cells keeps cells square (h = 1/N)."""
    return (2 * N, N)


def sweep_from_config(
    config_path,
    partition_rule,
    out_dir,
    element_types=None,
    only_N=None,
    include_reference=True,
    show=False,
):
    """Render every mesh a convergence study uses, one PNG per level.

    Reads ``domain.bounding_box``, ``numerical_method.mesh.convergence_partitions``
    (a list of densities N) and the reference ``partition``. For each N it builds
    the Cartesian partition via ``partition_rule(N)`` and saves a figure into
    ``out_dir``. ``element_types`` defaults to whatever the config declares (a
    config may list several, e.g. ["QUAD", "TRI"] — each is rendered).
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    bbox = cfg["domain"]["bounding_box"]
    mesh_cfg = cfg["numerical_method"]["mesh"]
    if element_types is None:
        et = mesh_cfg.get("element_type", "QUAD")
        element_types = et if isinstance(et, list) else [et]

    Ns = list(only_N) if only_N else list(mesh_cfg.get("convergence_partitions", []))
    if not Ns and not include_reference:
        Ns = [mesh_cfg["partition"]]  # single explicit partition, no sweep

    base = os.path.splitext(os.path.basename(config_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    written = []

    jobs = [(N, partition_rule(N)) for N in Ns]
    if include_reference and "partition" in mesh_cfg:
        jobs.append(("ref", tuple(mesh_cfg["partition"])))

    first = True
    for et in element_types:
        for tag, partition in jobs:
            label = f"N{tag}" if tag != "ref" else "ref"
            out = os.path.join(out_dir, f"mesh_{base}_{et}_{label}.png")
            title = f"{base} — {et} {int(partition[0])}x{int(partition[1])} ({label})"
            plot_cartesian_from_config(
                config_path,
                partition=partition,
                element_type=et,
                title=title,
                out=None if (show and first) else out,
            )
            if not (show and first):
                written.append(out)
            first = False
    return written


def sweep_freefem_dir(msh_dir, out_dir, only_N=None, show=False):
    """Render the FreeFem ``.msh`` files in a directory, one PNG each.

    ``only_N`` restricts to files whose ``_N<digits>`` density matches the list
    (handy to skip the multi-hundred-thousand-element reference mesh).
    """
    os.makedirs(out_dir, exist_ok=True)
    wanted = set(only_N) if only_N else None

    def keep(fname):
        if not fname.endswith(".msh"):
            return False
        if wanted is None:
            return True
        m = re.search(r"_N(\d+)", fname)
        return bool(m) and int(m.group(1)) in wanted

    files = sorted(f for f in os.listdir(msh_dir) if keep(f))
    written = []
    for i, fname in enumerate(files):
        path = os.path.join(msh_dir, fname)
        out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")
        plot_freefem(path, out=None if (show and i == 0) else out)
        if not (show and i == 0):
            written.append(out)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Visualise porous-NS test meshes.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="JSON config describing a Cartesian mesh")
    src.add_argument("--msh", help="explicit FreeFem .msh file")
    p.add_argument(
        "--partition",
        nargs=2,
        type=int,
        metavar=("NX", "NY"),
        help="override the Cartesian partition (cells per direction)",
    )
    p.add_argument(
        "--element-type", choices=["QUAD", "TRI"], help="override the element type"
    )
    p.add_argument("--title", help="figure title")
    p.add_argument("--out", help="output image path (otherwise shown interactively)")
    p.add_argument(
        "--no-fill", action="store_true", help="draw edges only (faster for big meshes)"
    )
    args = p.parse_args(argv)

    if args.config:
        plot_cartesian_from_config(
            args.config,
            partition=args.partition,
            element_type=args.element_type,
            title=args.title,
            out=args.out,
            fill=not args.no_fill,
        )
    else:
        plot_freefem(args.msh, title=args.title, out=args.out, fill=not args.no_fill)


if __name__ == "__main__":
    main()
