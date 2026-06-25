#!/usr/bin/env python3
"""3D MMS convergence plots, in the same format as the 2D ManufacturedSolutions harness
(`analyze_results.py`'s per-cell convergence PNGs): log-log error-vs-h, ASGS solid / OSGS dashed,
L² Velocity (blue ●), H¹ Velocity (blue ◆), L² Pressure (red ○), each segment annotated with its
slope ÷ optimal rate (1.0 = optimal). One plot per element type (P1, P2) — the "different meshes
employed" (here the single nested red-refined tet family at two polynomial orders).

Layout: results from each mesh sequence are conserved under their own folder INSIDE each element type:
    results/k<kv>/TET/<mesh_sequence>/convergence3d_results.json  ->  .../convergence3d_P<kv>.png
so e.g. the structured P1 plot is results/k1/TET/structured/convergence3d_P1.png.

Run:
  python plot_convergence3d.py <mesh_sequence>          # plot results/k<kv>/TET/<seq>/ for kv in 1,2
  python plot_convergence3d.py <mesh_sequence> <root>   # ...under a custom results root
  python plot_convergence3d.py <dir>                    # back-compat: a dir holding an aggregate JSON
  python plot_convergence3d.py                          # back-compat: legacy results/convergence3d_results.json
"""
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KV_VALUES = (1, 2)
ETYPE = "TET"


def _seg_slope(h, e, i):
    return (math.log(e[i + 1]) - math.log(e[i])) / (math.log(h[i + 1]) - math.log(h[i]))


def plot_cell(results_dir):
    """Read results_dir/convergence3d_results.json and write a convergence3d_P<kv>.png per kv into it."""
    json_path = os.path.join(results_dir, "convergence3d_results.json")
    if not os.path.exists(json_path):
        return False
    recs = json.load(open(json_path))
    by_kv = {}
    for r in recs:
        by_kv.setdefault(int(r["kv"]), {})[str(r["method"]).upper()] = r
    _plot_by_kv(by_kv, results_dir)
    return True


def _plot_by_kv(by_kv, out_dir):
    for kv, methods in sorted(by_kv.items()):
        opt_u_l2, opt_u_h1, opt_p_l2 = kv + 1, kv, kv      # P_kv: L²u~h^{kv+1}, H¹u~h^{kv}, L²p~h^{kv}
        plt.figure(figsize=(10, 8))
        ref = next(iter(methods.values()))
        a0, Re, Da = ref["alpha_0"], ref["Re"], ref["Da"]
        for method, r in sorted(methods.items()):
            ls = "-" if method == "ASGS" else "--"
            h = np.asarray(r["hs"], float)
            eu, euh, ep = (np.asarray(r[k], float) for k in ("l2u", "h1u", "l2p"))
            plt.loglog(h, eu, marker="o", linestyle=ls, color="blue", linewidth=2, markersize=8,
                       label=fr"{method} $L_2$ Velocity ({opt_u_l2})")
            plt.loglog(h, euh, marker="D", linestyle=ls, color="blue", linewidth=2, markersize=8,
                       label=fr"{method} $H^1$ Velocity ({opt_u_h1})")
            plt.loglog(h, ep, marker="o", linestyle=ls, color="red", linewidth=2, markersize=8,
                       markerfacecolor="white", label=fr"{method} $L_2$ Pressure ({opt_p_l2})")
            for i in range(len(h) - 1):
                for arr, col, opt in ((eu, "blue", opt_u_l2), (euh, "blue", opt_u_h1), (ep, "red", opt_p_l2)):
                    hm, em = math.sqrt(h[i] * h[i + 1]), math.sqrt(arr[i] * arr[i + 1])
                    sv = _seg_slope(h, arr, i)
                    plt.annotate(f"{(sv / opt if opt else float('nan')):.2f}", xy=(hm, em),
                                 xytext=(0, 10 if method == "ASGS" else -12), textcoords="offset points",
                                 ha="center", fontsize=8, color=col, fontweight="bold")
        plt.xlabel(r"Mesh size ($h$)")
        plt.ylabel("Error Norms")
        mesh_tag = str(ref.get("mesh", "")).replace("_", " ")
        plt.title(fr"Convergence 3D ($\alpha_0: {a0:g}$, $Re: {Re:g}$, $Da: {Da:g}$, "
                  fr"$\mathbb{{P}}_{kv}$, TET, {mesh_tag})")
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(handlelength=4.0)
        plt.grid(True, which="both", ls="--")
        out = os.path.join(out_dir, f"convergence3d_P{kv}.png")
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"[plot] wrote {out}")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    root = _resolve(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, "results")

    if arg is None:
        # no sequence given: plot ALL available ones — every convergence3d_results.json under results/
        # (any layout: results/k<kv>/TET/<seq>/, a legacy results/<seq>/, or the legacy aggregate results/).
        n = 0
        for jp in sorted(glob.glob(os.path.join(root, "**", "convergence3d_results.json"), recursive=True)):
            if plot_cell(os.path.dirname(jp)):
                n += 1
        if n == 0:
            raise SystemExit(f"[plot] no convergence3d_results.json found anywhere under {root} — run a sweep first")
    elif os.path.isdir(_resolve(arg)) and os.path.exists(os.path.join(_resolve(arg), "convergence3d_results.json")):
        # back-compat: an explicit dir holding an aggregate JSON (e.g. results/structured)
        plot_cell(_resolve(arg))
    else:
        # arg is a mesh_sequence tag: plot each element-type cell results/k<kv>/TET/<seq>/
        n = 0
        for kv in KV_VALUES:
            cell = os.path.join(root, f"k{kv}", ETYPE, arg)
            if plot_cell(cell):
                n += 1
        if n == 0:
            raise SystemExit(f"[plot] no convergence3d_results.json under {root}/k*/{ETYPE}/{arg}/ "
                             f"— run a sweep with mesh_sequence={arg!r}")
