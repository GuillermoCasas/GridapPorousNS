# probe_stiff_diagnose.jl — findings (v2: A1 J–FD consistency, A2 heavy solves, A3 τ-landscape)

Run: 2026-05-21T08:52:53.196

Decisive re-diagnosis of C24 (`Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS`). Supersedes probe_stiff_raw.md / probe_stiff_findings.md (whose harness was self-flagged unreliable).

## A0 benign gate — `Re=1, Da=1, α=0.5, k=1, n=10, QUAD, ASGS`

**GATE PASSED.**

### Supplementary — cond(J), residual at u_ex

- ‖R(u_ex)‖_∞ = 7.669537e-02,  ‖R(u_ex)‖_2 = 3.586576e-01,  n_dofs = 283
- cond(J) = 3.148379e+10,  σ_max = 6.673509e+00,  σ_min = 2.119665e-10

### A1 — Jacobian-vs-FD consistency (decisive)

ExactNewton Jacobian **CONSISTENT** (best full relerr 2.632122e-12).

| direction | best full relerr | u-block | p-block |
|---|---|---|---|
| random_2 | 2.990828e-12 | 2.990869e-12 | 2.756869e-12 |
| random_1 | 2.632122e-12 | 2.632051e-12 | 3.191409e-12 |
| random_3 | 3.171750e-12 | 3.171660e-12 | 3.733770e-12 |
| pressure_only | 8.701867e-11 | 8.749636e-11 | 1.374326e-11 |
| velocity_only | 2.779893e-12 | 2.779933e-12 | 2.380804e-12 |

### A2 — heavy Picard from u_ex

- iters=4  stop=`ftol_reached`  reached_root=true
- final ‖R‖ = 3.612157e-10,  L2 u = 3.140418e-02,  L2 p = 1.417673e-01,  H1 u = 7.823354e-01

### A2 — heavy Newton from u_ex

- iters=2  stop=`ftol_reached`  reached_root=true
- final ‖R‖ = 9.391016e-12,  L2 u = 3.140418e-02,  L2 p = 1.417673e-01,  H1 u = 7.823354e-01

### Supplementary — Newton-step line probe

‖δ‖_2 = 2.313398e+01,  min at s* = +1.000,  ‖R‖_∞(s*) = 7.039548e-05

## Stiff cell C24 — `Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS`

### Supplementary — cond(J), residual at u_ex

- ‖R(u_ex)‖_∞ = 2.152972e-01,  ‖R(u_ex)‖_2 = 3.302287e-01,  n_dofs = 283
- cond(J) = 2.716489e+07,  σ_max = 1.541135e+01,  σ_min = 5.673261e-07

### A1 — Jacobian-vs-FD consistency (decisive)

ExactNewton Jacobian **CONSISTENT** (best full relerr 4.817400e-12).

| direction | best full relerr | u-block | p-block |
|---|---|---|---|
| random_2 | 6.551181e-10 | 1.839041e-10 | 6.548502e-10 |
| random_1 | 3.294946e-10 | 1.656395e-10 | 3.295284e-10 |
| random_3 | 3.635588e-10 | 1.607221e-10 | 3.635989e-10 |
| pressure_only | 4.817400e-12 | 1.176019e-11 | 4.812889e-12 |
| velocity_only | 5.429341e-10 | 9.050388e-10 | 5.258228e-10 |

### A2 — heavy Picard from u_ex

- iters=4  stop=`linesearch_failed`  reached_root=false
- final ‖R‖ = 4.838721e-02,  L2 u = 1.200116e-01,  L2 p = 2.445125e-02,  H1 u = 2.989407e+00

### A2 — heavy Newton from u_ex

- iters=49  stop=`linesearch_failed`  reached_root=false
- final ‖R‖ = 7.664679e-02,  L2 u = 1.674043e-01,  L2 p = 4.400258e-02,  H1 u = 3.457645e+00

### Supplementary — Newton-step line probe

‖δ‖_2 = 1.405699e+01,  min at s* = +0.500,  ‖R‖_∞(s*) = 1.811943e-01

## Verdict

- A1: ExactNewton Jacobian is CONSISTENT at u_ex (best full relerr 4.82e-12).
- A2 heavy Picard: reached_root=false, final ‖R‖=4.84e-02, L2 u=1.200e-01 (stop=linesearch_failed).
- A2 heavy Newton: reached_root=false, final ‖R‖=7.66e-02, L2 u=1.674e-01 (stop=linesearch_failed).
- VERDICT → Path 2: J CONSISTENT but NO nearby discrete root at ftol — landscape genuinely hard. Da-continuation from C27 and/or smooth the τ/velocity-floor (see A3 dominance/threshold).

