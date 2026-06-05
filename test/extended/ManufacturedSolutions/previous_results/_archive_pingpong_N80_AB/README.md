# Iterator-optimization A/B (N<=80), 2026-06-05

`phase1_quad_k1_pingpong.h5` — k=1 sweep, N<=80, **optimized iterator ON** (ping-pong + plateau
short-circuit + stall_window=2), all 24 cells x 2 methods. A/B vs the covariant-complete snapshot
(`../_archive_postFix_covariant_complete/phase1_quad_k1.h5`): **errors equal-or-better on all 48
(cell,method); total nonlinear iters 2595 -> 2024 (-22%)**, concentrated on Re=1e6 (-37%); mild cells
inert (0%). The headline evidence for the P0-P6 nonlinear-iterator optimization.

NOTE: produced with the OSGS outer loop still running its full budget ("always 10") — i.e. BEFORE the
OSGS per-iteration-projection / convergence-detection rework. Kept as the pre-rework baseline.
