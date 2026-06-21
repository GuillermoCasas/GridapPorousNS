#!/usr/bin/env python3
"""3D MMS convergence plots, in the same format as the 2D ManufacturedSolutions harness
(`analyze_results.py`'s per-cell convergence PNGs): log-log error-vs-h, ASGS solid / OSGS dashed,
L² Velocity (blue ●), H¹ Velocity (blue ◆), L² Pressure (red ○), each segment annotated with its
slope ÷ optimal rate (1.0 = optimal). One plot per element type (P1, P2) — the "different meshes
employed" (here the single nested red-refined tet family at two polynomial orders).

Reads results/convergence3d_results.json (produced by `julia smoke3d.jl sweep`).
Run: python plot_convergence3d.py  ->  results/convergence3d_P1.png, results/convergence3d_P2.png
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
JSON_PATH = os.path.join(RESULTS, "convergence3d_results.json")

if not os.path.exists(JSON_PATH):
    raise SystemExit(f"[plot] no sweep results at {JSON_PATH} — run: julia --project=../../.. smoke3d.jl sweep")

recs = json.load(open(JSON_PATH))
by_kv = {}
for r in recs:
    by_kv.setdefault(int(r["kv"]), {})[str(r["method"]).upper()] = r


def _seg_slope(h, e, i):
    return (math.log(e[i + 1]) - math.log(e[i])) / (math.log(h[i + 1]) - math.log(h[i]))


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
    plt.title(fr"Convergence 3D ($\alpha_0: {a0:g}$, $Re: {Re:g}$, $Da: {Da:g}$, "
              fr"$\mathbb{{P}}_{kv}$, TET, nested)")
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(handlelength=4.0)
    plt.grid(True, which="both", ls="--")
    out = os.path.join(RESULTS, f"convergence3d_P{kv}.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] wrote {out}")
