#!/usr/bin/env python3
"""3D MMS convergence — results SO FAR (P1).

Data measured from the runs in this folder (provenance in each block):
  - interp_test.jl 1   : pure P1 interpolation of u_ex on the CUBE (0,1)^3, qdeg=14, vs measured h
                         for three mesh families (nested red-refined gmsh / structured simplexified
                         Cartesian / independent gmsh remeshes).
  - smoke3d.jl 1 ASGS Deviatoric 1.0 box 2 0.2 : full ASGS solve on the BOX (0,1)x(0,1)x(0,0.3),
                         nested family, levels 0-2.
Error is rate vs MEASURED mean h (regular-tet edge from cell volume). Optimal P1: L2~h^2, H1~h^1.
Run: python plot_convergence3d.py  ->  results/convergence3d.png
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- measured data (h, L2u, H1u) ---------------------------------------------------------------
# CUBE, pure interpolation (no solve), P1
nested_interp = dict(  # nested red-refined gmsh family (quality-optimized base)
    h=[0.2025, 0.1012, 0.05062, 0.02531], L2u=[0.03647, 0.01242, 0.003845, 0.001108],
    H1u=[0.7676, 0.5242, 0.314, 0.1835])
structured_interp = dict(  # structured simplexified Cartesian (known-good control)
    h=[0.2245, 0.1122, 0.05612], L2u=[0.03607, 0.01289, 0.003063],
    H1u=[0.7513, 0.4526, 0.2232])
independent_interp = dict(  # OLD strategy: independent gmsh remeshes
    h=[0.2207, 0.118, 0.06055], L2u=[0.03725, 0.01331, 0.004362],
    H1u=[0.7959, 0.4971, 0.3003])
# BOX (0,1)x(0,1)x(0,0.3), full ASGS solve on nested family, P1, levels 0-2
box_solve = dict(
    h=[0.1737, 0.08685, 0.04342], L2u=[0.03468, 0.01356, 0.006336],
    H1u=[0.4183, 0.2465, 0.1433], L2p=[0.2586, 0.2417, 0.1151])

SERIES = [
    ("nested gmsh (interp, cube)",      nested_interp,      "tab:green",  "o", "-"),
    ("structured Cartesian (interp)",   structured_interp,  "tab:blue",   "s", "-"),
    ("independent gmsh (interp, cube)", independent_interp, "tab:red",    "^", "--"),
    ("nested gmsh (SOLVE, box)",        box_solve,          "black",      "D", "-"),
]


def seg_rates(h, e):
    h, e = np.asarray(h), np.asarray(e)
    return [np.log(e[i]/e[i+1])/np.log(h[i]/h[i+1]) for i in range(len(h)-1)]


def panel(ax, key, opt, title):
    for label, d, color, mk, ls in SERIES:
        if key not in d:
            continue
        h, e = np.asarray(d["h"], float), np.asarray(d[key], float)
        r = seg_rates(h, e)
        ax.loglog(h, e, marker=mk, ls=ls, color=color, lw=1.8, ms=7,
                  label=f"{label}  (last rate {r[-1]:.2f})")
        for i in range(len(h)-1):  # per-segment slope annotation
            ax.annotate(f"{r[i]:.2f}", xy=(np.sqrt(h[i]*h[i+1]), np.sqrt(e[i]*e[i+1])),
                        xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=7, color=color)
    # reference slope (anchored to the structured series' coarsest point)
    h0 = max(np.max(d["h"]) for _, d, *_ in SERIES if key in d)
    e0 = np.max([d[key][np.argmax(d["h"])] for _, d, *_ in SERIES if key in d])
    hh = np.array([h0, h0/8])
    ax.loglog(hh, e0*(hh/h0)**opt, "k:", lw=1.2, alpha=0.6, label=f"ref $h^{opt}$ (optimal)")
    ax.set_xlabel(r"mean $h$ (tet edge)"); ax.set_ylabel(title)
    ax.set_title(title); ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=7.5, loc="lower right")


fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
panel(axes[0], "L2u", 2, r"Velocity $L^2$  ($u-u_h$)")
panel(axes[1], "H1u", 1, r"Velocity $H^1$  ($\nabla(u-u_h)$)")
fig.suptitle("3D MMS convergence so far (P1): mesh-strategy comparison + box solve\n"
             "independent gmsh = FLAT (~1.66/0.76);  nested = CLIMBING;  structured = OPTIMAL (2.07/1.02)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "results", "convergence3d.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140); plt.close(fig)
print(f"[plot] wrote {out}")
