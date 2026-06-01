"""Shared bootstrap for the per-test ``plot_mesh.py`` wrappers.

Each test directory holds a tiny wrapper that knows only its own config path and
mesh-density convention; everything else lives in ``mesh_plot``. This module
resolves the import path to ``mesh_plot`` regardless of how deep the test dir is
and exposes a couple of thin argv helpers so the wrappers stay ~10 lines.
"""

import os
import sys

# tools/mesh_viz/ — the dir this file lives in — holds mesh_plot.py.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import mesh_plot as mp  # noqa: E402  (path set above)


def parse_argv(argv):
    """Return (only_N, show) from a wrapper's argv list.

    Bare integers select specific densities (e.g. ``plot_mesh.py 20 40``);
    ``--show`` displays the first figure interactively instead of saving it.
    """
    only_N = [int(a) for a in argv if a.lstrip("-").isdigit()]
    show = "--show" in argv
    return (only_N or None), show


def run_cartesian(config, out_dir, partition_rule, argv, element_types=None,
                  include_reference=False):
    only_N, show = parse_argv(argv)
    written = mp.sweep_from_config(
        config,
        partition_rule,
        out_dir,
        element_types=element_types,
        only_N=only_N,
        include_reference=include_reference or ("--ref" in argv),
        show=show,
    )
    for w in written:
        print(w)


def run_freefem(msh_dir, out_dir, argv):
    only_N, show = parse_argv(argv)
    written = mp.sweep_freefem_dir(msh_dir, out_dir, only_N=only_N, show=show)
    for w in written:
        print(w)
