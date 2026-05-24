"""
H-A diagnostic plot: overlay our Galerkin P2/P1 L²(u) and (L²(u))² against the
paper Cocquet Fig. 2 right-panel hand-read values.

Hypothesis under test: the paper's y-axis is the SQUARED L² norm, not the L²
norm. If true, squaring our values should reproduce both the magnitude and
the slope of the paper's curve.

Generates results/h_a_squared_overlay.png. No code under src/ is touched;
reads the existing freefem-divs HDF5 only.

Usage:  python plot_h_a_squared_overlay.py
"""

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(__file__)
H5_PATH = os.path.join(HERE, "results", "convergence_paper_comparison_irregular_freefem_divs.h5")
OUT_PATH = os.path.join(HERE, "results", "h_a_squared_overlay.png")

# Hand-read values from Cocquet et al. (HAL hal-02561058) Figure 2 right panel,
# the GREEN curve (Re=500, c_in=0.5, P2/P1 Galerkin on FreeFem mesh). Reading
# the gridlines: y-axis is 10^-7 (bottom) to 10^-4 (top), x-axis N=10..100.
PAPER_N = np.array([10, 20, 40, 80, 100], dtype=float)
PAPER_L2_U = np.array([3.0e-5, 1.0e-5, 4.0e-6, 5.0e-7, 1.5e-7])


def main():
    with h5py.File(H5_PATH, "r") as f:
        N = np.array(f["N_list"], dtype=float)
        err = np.array(f["Galerkin/P2P1/errors_l2_u"])
    err2 = err ** 2

    raw_slope = (np.log(err[-1]) - np.log(err[0])) / (np.log(N[-1]) - np.log(N[0]))
    sq_slope = (np.log(err2[-1]) - np.log(err2[0])) / (np.log(N[-1]) - np.log(N[0]))
    paper_slope = (np.log(PAPER_L2_U[-1]) - np.log(PAPER_L2_U[0])) / (np.log(PAPER_N[-1]) - np.log(PAPER_N[0]))

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.loglog(N, err, "o-", color="C0", lw=2, ms=10,
              label=fr"Ours: $\|e_u\|_{{L^2}}$  (slope $\approx${raw_slope:.2f})")
    ax.loglog(N, err2, "s--", color="C1", lw=2, ms=10,
              label=fr"Ours squared: $\|e_u\|_{{L^2}}^2$  (slope $\approx${sq_slope:.2f})")
    ax.loglog(PAPER_N, PAPER_L2_U, "^:", color="C2", lw=2, ms=10,
              label=fr"Cocquet Fig. 2 (hand-read)  (slope $\approx${paper_slope:.2f})")

    for Ni, ei in zip(N, err):
        ax.annotate(f"{ei:.2e}", (Ni, ei), textcoords="offset points",
                    xytext=(8, 8), color="C0", fontsize=8)
    for Ni, ei in zip(N, err2):
        ax.annotate(f"{ei:.2e}", (Ni, ei), textcoords="offset points",
                    xytext=(8, -14), color="C1", fontsize=8)
    for Ni, ei in zip(PAPER_N, PAPER_L2_U):
        ax.annotate(f"{ei:.2e}", (Ni, ei), textcoords="offset points",
                    xytext=(-46, -4), color="C2", fontsize=8)

    ax.set_xlabel("N")
    ax.set_ylabel(r"$\|e_u\|$ or $\|e_u\|^2$")
    ax.set_title("H-A diagnostic: does squaring our $L^2$ velocity error\n"
                 "reproduce Cocquet Fig. 2?  (Galerkin P2/P1, freefem-divs mesh)")
    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.legend(fontsize=10, loc="upper right")

    ratio_raw = err / np.interp(N, PAPER_N, PAPER_L2_U)
    ratio_sq = err2 / np.interp(N, PAPER_N, PAPER_L2_U)
    txt = "Ratios vs paper (N=10,20,40,80,100):\n"
    txt += "  raw / paper : " + ", ".join(f"{r:.1f}x" for r in ratio_raw) + "\n"
    txt += "  squared / paper : " + ", ".join(f"{r:.2f}x" for r in ratio_sq)
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9,
            family="monospace", verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"))

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    print(f"Saved: {OUT_PATH}")
    print()
    print(f"  raw slope    : {raw_slope:+.3f}")
    print(f"  squared slope: {sq_slope:+.3f}")
    print(f"  paper slope  : {paper_slope:+.3f}")
    print()
    print("Ratios (lower=closer to paper):")
    print(f"  raw / paper    : {ratio_raw}")
    print(f"  squared / paper: {ratio_sq}")


if __name__ == "__main__":
    main()
