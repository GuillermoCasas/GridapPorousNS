#!/usr/bin/env python3
"""Plot the ASGS momentum-block coercivity probe (coercivity_probe.jl output).

Reads results/debug_results/coercivity_probe*.json and draws lambda_min(c1_mult) for each
(partition, h_conv) family, with the coercivity boundary lambda_min=0 and the paper c1=4k^4
(c1_mult=1) marked. DIAGNOSTIC figure -> results/debug_results/ (gitignored).

    python3 plot_coercivity_probe.py
"""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.join(HERE, "results", "debug_results")


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(DBG, "coercivity_probe*.json"))):
        if f.endswith("_run.json"):
            continue
        with open(f) as fh:
            rows += json.load(fh).get("rows", [])
    return rows


def main():
    rows = load()
    # group by (partition, h_conv); prefer picard (== newton, and it is the coercivity form)
    fams = {}
    for r in rows:
        if r["lin_mode"] != "picard":
            continue
        key = (tuple(r["partition"]), r["h_conv"])
        fams.setdefault(key, []).append((r["c1_mult"], r["lambda_min"]))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    conv_color = {"shortest_edge": "#1f77b4", "regular_tet": "#ff7f0e", "diameter": "#d62728"}
    conv_label = {"shortest_edge": "h=shortest edge (=a, production)",
                  "regular_tet": "h=(6√2·V)^⅓ (≈1.12a, sweep default)",
                  "diameter": "h=diameter (≈1.73a, paper h_K)"}
    seen_conv = set()
    for (part, hc), pts in sorted(fams.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        lbl = conv_label.get(hc, hc) if hc not in seen_conv else None
        seen_conv.add(hc)
        ax.plot(xs, ys, "-o", color=conv_color.get(hc, "gray"), ms=4,
                label=lbl, alpha=0.5 + 0.5 * (part[0] >= 12))
        ax.annotate(f"{part}", (xs[-1], ys[-1]), fontsize=6, color=conv_color.get(hc, "gray"))

    ax.axhline(0.0, color="k", lw=1.0)
    ax.axvline(1.0, color="0.5", ls="--", lw=1.0)
    ax.text(1.02, ax.get_ylim()[0] * 0.9, "paper c₁=4k⁴", fontsize=8, color="0.4")
    ax.text(0.28, 0.06, "COERCIVE (λ_min>0)", fontsize=8, color="green")
    ax.text(0.28, -0.55, "NOT coercive", fontsize=8, color="red")
    ax.set_xlabel("c₁ multiplier  (c₂ held fixed — c₁-only)")
    ax.set_ylabel("λ_min  of  (½(Aᵤᵤ+Aᵤᵤᵀ),  Galerkin deviatoric+reaction Gram)")
    ax.set_title("ASGS momentum-block coercivity vs c₁ — Kuhn P2 (Re=Da=1, α₀=0.5)\n"
                 "paper c₁ is sub-coercive; threshold is h-convention-dependent (≈1.4×..4×)")
    ax.set_ylim(-1.5, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    out = os.path.join(DBG, "coercivity_probe.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print("[wrote]", out)


if __name__ == "__main__":
    main()
