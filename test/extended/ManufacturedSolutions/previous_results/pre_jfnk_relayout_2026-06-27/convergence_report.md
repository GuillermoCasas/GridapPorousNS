# Convergence Rate and FME Table

**Status** is the Phase-1 per-config diagnosis (same true-root + one-sided-slope logic as the merged table, WITHOUT the Phase-2 continuation rescue — for the post-rescue verdict see `merged_convergence_report.md`):
- `optimal` — true root at the finest meshes and slope ≥ optimal (k+1 / k). `ˢ` = velocity-L² rate ABOVE nominal (super-convergent; confirm asymptotics).
- `sub-optimal-rate ‡` — true root exists at both finest meshes, but the slope is below optimal (candidate for a finer-mesh check).
- `fold (**)` — the finest mesh has NO true root (a coarser one may); the discrete branch folds → needs continuation at a finer mesh.
- `no-root (N/A)` — no true root at any mesh. `incomplete` — only one mesh.

'Converged' is the HONEST true-root test (‖R‖ ≤ k_nf·dynamic_ftol) at the finest mesh, not the solver's noise-floor-foolable flag. Final Res. shows (‖R‖ vs dynamic_ftol at that mesh). Rates/FME are the Phase-1 finest-mesh values.

'eps_finest' / 'eps_min' is the homotopy outer loop's per-cell robustness fingerprint: the largest eps_pert ∈ {1.0, 0.1, ..., 0} that Newton absorbed at the finest mesh (eps_finest) and the smallest such value across all meshes (eps_min). 1 = solver handled a full perturbation; 0 = only the interpolated-u_ex fallback worked.

| Config | Method | Source JSON | Re | Da | α_0 | k | Elem | Time (s) | Iters | Converged | Status | Final Res. | rate_u_L2 | rate_p_L2 | rate_u_H1 | rate_p_H1 | FME u_L2 | FME p_L2 | FME u_H1 | FME p_H1 | eps_finest | eps_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e-06 | 0.05 | 1 | QUAD | 455.6 | 7 | Yes | optimal | 2.9814e-12 (2.4414e-08) | 1.98 (2) | 1.99 (1) | 1.02 (1) | 1.98 (0) | 2.1698e-04 | 6.3926e-03 | 6.7472e-02 | 8.0024e-01 | 1 | 1 |
| C1 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e-06 | 1e-06 | 0.05 | 1 | QUAD | 1660.7 | 105 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.6903e-09 (2.4414e-08) | 2.03 (2) | 2.02 (1) | 1.00 (1) | 2.03 (0) | 8.0765e-05 | 2.0707e-03 | 6.7220e-02 | 2.5952e-01 | 1 | 1 |
| C2 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e-06 | 0.50 | 1 | QUAD | 430.9 | 7 | Yes | optimal | 1.4916e-11 (2.4414e-08) | 2.00 (2) | 2.10 (1) | 1.00 (1) | 0.96 (0) | 1.6239e-05 | 6.7497e-04 | 1.3058e-02 | 1.7487e-01 | 1 | 1 |
| C2 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e-06 | 1e-06 | 0.50 | 1 | QUAD | 2567.0 | 189 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 4.1610e-09 (2.4414e-08) | 2.04 (2) | 2.10 (1) | 1.00 (1) | 0.85 (0) | 5.3253e-06 | 2.1016e-04 | 1.3046e-02 | 8.8862e-02 | 1 | 1 |
| C3 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e-06 | 1.00 | 1 | QUAD | 514.8 | 7 | Yes | optimal | 5.6451e-11 (2.4414e-08) | 2.00 (2) | 1.72 (1) | 1.00 (1) | 0.56 (0) | 9.4016e-06 | 3.5148e-04 | 4.4593e-03 | 3.3215e-01 | 1 | 1 |
| C3 | OSGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e-06 | 1.00 | 1 | QUAD | 3712.9 | 237 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.5344e-09 (2.4414e-08) | 2.00 (2) | 1.71 (1) | 1.00 (1) | 0.66 (0) | 1.8049e-06 | 1.2839e-04 | 4.4526e-03 | 1.8139e-01 | 1 | 1 |
| C4 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e+00 | 0.05 | 1 | QUAD | 454.3 | 7 | Yes | optimal | 3.0727e-12 (2.4414e-08) | 1.96 (2) | 1.93 (1) | 1.01 (1) | 1.89 (0) | 1.5540e-04 | 6.4976e-03 | 6.7475e-02 | 8.5136e-01 | 1 | 1 |
| C4 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e-06 | 1e+00 | 0.05 | 1 | QUAD | 2662.6 | 121 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.6196e-09 (2.4414e-08) | 2.00 (2) | 2.01 (1) | 1.00 (1) | 2.03 (0) | 1.2241e-04 | 1.9048e-03 | 6.7223e-02 | 2.3545e-01 | 1 | 1 |
| C5 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e+00 | 0.50 | 1 | QUAD | 439.3 | 7 | Yes | optimal | 1.4924e-11 (2.4414e-08) | 1.99 (2) | 1.95 (1) | 1.00 (1) | 1.13 (0) | 1.6403e-05 | 6.1947e-04 | 1.3058e-02 | 1.1064e-01 | 1 | 1 |
| C5 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e-06 | 1e+00 | 0.50 | 1 | QUAD | 3833.3 | 204 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.0987e-08 (2.4414e-08) | 2.00 (2) | 1.98 (1) | 1.00 (1) | 0.71 (0) | 6.4352e-06 | 1.8170e-04 | 1.3046e-02 | 6.8435e-02 | 1 | 1 |
| C6 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e+00 | 1.00 | 1 | QUAD | 434.9 | 7 | Yes | optimal | 5.6507e-11 (2.4414e-08) | 2.00 (2) | 1.58 (1) | 1.00 (1) | 0.50 (0) | 9.4186e-06 | 2.5046e-04 | 4.4574e-03 | 1.9391e-01 | 1 | 1 |
| C6 | OSGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e+00 | 1.00 | 1 | QUAD | 4783.2 | 255 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.0700e-08 (2.4414e-08) | 2.00 (2) | 1.54 (1) | 1.00 (1) | 0.51 (0) | 1.8274e-06 | 1.0220e-04 | 4.4524e-03 | 1.3420e-01 | 1 | 1 |
| C7 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e+06 | 0.05 | 1 | QUAD | 458.8 | 7 | Yes | optimal | 5.6389e-11 (2.4414e-08) | 1.98 (2) | 2.42 (1) | 1.02 (1) | 1.00 (0) | 1.0691e-04 | 1.3093e-06 | 6.7657e-02 | 3.1479e-03 | 1 | 1 |
| C7 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e-06 | 1e+06 | 0.05 | 1 | QUAD | 1643.6 | 193 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.4198e-04 (2.4414e-08) | 2.49 (2) | 2.41 (1) | 1.14 (1) | 1.00 (0) | 2.3228e-05 | 1.3158e-06 | 7.1322e-02 | 3.1514e-03 | 1 | 1 |
| C8 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e+06 | 0.50 | 1 | QUAD | 435.8 | 7 | Yes | optimal | 4.1919e-10 (2.4414e-08) | 2.04 (2) | 2.44 (1) | 1.01 (1) | 1.00 (0) | 2.1557e-05 | 7.2165e-07 | 1.3066e-02 | 3.1479e-03 | 1 | 1 |
| C8 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e-06 | 1e+06 | 0.50 | 1 | QUAD | 1514.2 | 75 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 1.4198e-04 (2.4414e-08) | 2.60 (2) | 2.44 (1) | 1.60 (1) | 1.00 (0) | 1.8704e-05 | 7.2530e-07 | 2.7116e-02 | 3.1514e-03 | 1 | 1 |
| C9 | ASGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e+06 | 1.00 | 1 | QUAD | 436.2 | 7 | Yes | optimal | 9.6878e-10 (2.4414e-08) | 2.05 (2) | 2.27 (1) | 1.01 (1) | 1.00 (0) | 9.6928e-06 | 6.8596e-07 | 4.4656e-03 | 3.1479e-03 | 1 | 1 |
| C9 | OSGS | _run_quad_k1_postrefactor.json | 1e-06 | 1e+06 | 1.00 | 1 | QUAD | 1370.9 | 68 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.3318e-04 (2.4414e-08) | 2.75 (2) | 2.27 (1) | 1.85 (1) | 1.00 (0) | 1.6753e-05 | 6.8935e-07 | 2.1825e-02 | 3.1510e-03 | 1 | 1 |
| C10 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e-06 | 0.05 | 1 | QUAD | 633.2 | 16 | Yes | optimal | 3.9996e-10 (2.4414e-08) | 1.93 (2) | 2.02 (1) | 1.01 (1) | 1.99 (0) | 1.7388e-04 | 5.9226e-03 | 6.7472e-02 | 7.7327e-01 | 1 | 1 |
| C10 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+00 | 1e-06 | 0.05 | 1 | QUAD | 2931.6 | 119 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.5702e-09 (2.4414e-08) | 2.06 (2) | 2.09 (1) | 1.00 (1) | 2.12 (0) | 1.2881e-04 | 1.7308e-03 | 6.7223e-02 | 2.1453e-01 | 1 | 1 |
| C11 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e-06 | 0.50 | 1 | QUAD | 793.3 | 21 | Yes | optimal | 3.9302e-16 (2.4414e-08) | 1.99 (2) | 1.96 (1) | 1.00 (1) | 1.14 (0) | 1.6376e-05 | 6.1327e-04 | 1.3058e-02 | 1.1026e-01 | 1 | 1 |
| C11 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+00 | 1e-06 | 0.50 | 1 | QUAD | 4052.4 | 218 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.0848e-08 (2.4414e-08) | 2.01 (2) | 1.99 (1) | 1.00 (1) | 0.71 (0) | 6.4694e-06 | 1.8031e-04 | 1.3046e-02 | 6.8177e-02 | 1 | 1 |
| C12 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e-06 | 1.00 | 1 | QUAD | 864.2 | 21 | Yes | optimal | 9.9985e-16 (2.4414e-08) | 2.00 (2) | 1.58 (1) | 1.00 (1) | 0.51 (0) | 9.4119e-06 | 2.4937e-04 | 4.4574e-03 | 1.9356e-01 | 1 | 1 |
| C12 | OSGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e-06 | 1.00 | 1 | QUAD | 5548.8 | 269 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.0575e-08 (2.4414e-08) | 2.00 (2) | 1.54 (1) | 1.00 (1) | 0.52 (0) | 1.8310e-06 | 1.0178e-04 | 4.4524e-03 | 1.3376e-01 | 1 | 1 |
| C13 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e+00 | 0.05 | 1 | QUAD | 637.5 | 16 | Yes | optimal | 3.6680e-10 (2.4414e-08) | 1.96 (2) | 1.92 (1) | 1.01 (1) | 1.89 (0) | 1.5513e-04 | 4.3446e-03 | 6.7474e-02 | 5.6937e-01 | 1 | 1 |
| C13 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+00 | 1e+00 | 0.05 | 1 | QUAD | 2993.6 | 119 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.6215e-09 (2.4414e-08) | 2.00 (2) | 2.01 (1) | 1.00 (1) | 2.03 (0) | 1.2265e-04 | 1.2740e-03 | 6.7223e-02 | 1.5754e-01 | 1 | 1 |
| C14 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e+00 | 0.50 | 1 | QUAD | 775.6 | 21 | Yes | optimal | 3.9308e-16 (2.4414e-08) | 1.99 (2) | 1.95 (1) | 1.00 (1) | 1.13 (0) | 1.6396e-05 | 4.1297e-04 | 1.3058e-02 | 7.3797e-02 | 1 | 1 |
| C14 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+00 | 1e+00 | 0.50 | 1 | QUAD | 4307.3 | 217 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.1001e-08 (2.4414e-08) | 2.00 (2) | 1.98 (1) | 1.00 (1) | 0.71 (0) | 6.4362e-06 | 1.2120e-04 | 1.3046e-02 | 4.5701e-02 | 1 | 1 |
| C15 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e+00 | 1.00 | 1 | QUAD | 785.4 | 21 | Yes | optimal | 1.0716e-15 (2.4414e-08) | 2.00 (2) | 1.58 (1) | 1.00 (1) | 0.50 (0) | 9.4140e-06 | 1.6693e-04 | 4.4574e-03 | 1.2927e-01 | 1 | 1 |
| C15 | OSGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e+00 | 1.00 | 1 | QUAD | 4944.4 | 262 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.2658e-08 (2.4414e-08) | 2.00 (2) | 1.54 (1) | 1.00 (1) | 0.52 (0) | 1.8274e-06 | 6.8145e-05 | 4.4524e-03 | 8.9512e-02 | 1 | 1 |
| C16 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e+06 | 0.05 | 1 | QUAD | 626.6 | 13 | Yes | optimal | 5.1298e-11 (2.4414e-08) | 1.98 (2) | 2.42 (1) | 1.02 (1) | 1.00 (0) | 1.0686e-04 | 1.3093e-06 | 6.7657e-02 | 3.1479e-03 | 1 | 1 |
| C16 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+00 | 1e+06 | 0.05 | 1 | QUAD | 1759.6 | 203 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.4198e-04 (2.4414e-08) | 2.49 (2) | 2.41 (1) | 1.14 (1) | 1.00 (0) | 2.3228e-05 | 1.3157e-06 | 7.1322e-02 | 3.1514e-03 | 1 | 1 |
| C17 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e+06 | 0.50 | 1 | QUAD | 633.1 | 15 | Yes | optimal | 1.9548e-10 (2.4414e-08) | 2.04 (2) | 2.44 (1) | 1.01 (1) | 1.00 (0) | 2.1547e-05 | 7.2161e-07 | 1.3066e-02 | 3.1479e-03 | 1 | 1 |
| C17 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+00 | 1e+06 | 0.50 | 1 | QUAD | 1798.4 | 83 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 1.4198e-04 (2.4414e-08) | 2.60 (2) | 2.44 (1) | 1.60 (1) | 1.00 (0) | 1.8703e-05 | 7.2526e-07 | 2.7115e-02 | 3.1514e-03 | 1 | 1 |
| C18 | ASGS | _run_quad_k1_postrefactor.json | 1e+00 | 1e+06 | 1.00 | 1 | QUAD | 601.0 | 14 | Yes | optimal | 3.1685e-10 (2.4414e-08) | 2.04 (2) | 2.27 (1) | 1.01 (1) | 1.00 (0) | 9.6877e-06 | 6.8594e-07 | 4.4656e-03 | 3.1479e-03 | 1 | 1 |
| C18 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+00 | 1e+06 | 1.00 | 1 | QUAD | 1521.1 | 75 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.3318e-04 (2.4414e-08) | 2.75 (2) | 2.27 (1) | 1.85 (1) | 1.00 (0) | 1.6750e-05 | 6.8933e-07 | 2.1822e-02 | 3.1510e-03 | 1 | 1 |
| C19 | ASGS | _run_quad_k1_postrefactor.json | 1e+06 | 1e-06 | 0.50 | 1 | QUAD | 813.8 | 47 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.9159e-05 (2.4414e-08) | 2.08 (2) | 2.07 (1) | 1.01 (1) | 1.00 (0) | 2.8831e-06 | 7.2044e-07 | 1.3061e-02 | 3.1479e-03 | 0.1 | 0.1 |
| C19 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+06 | 1e-06 | 0.50 | 1 | QUAD | 1827.5 | 195 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 5.2375e-04 (2.4414e-08) | 2.15 (2) | 2.06 (1) | 1.02 (1) | 1.00 (0) | 3.0718e-06 | 7.3634e-07 | 1.3155e-02 | 3.1506e-03 | 0.1 | 0.1 |
| C20 | ASGS | _run_quad_k1_postrefactor.json | 1e+06 | 1e-06 | 1.00 | 1 | QUAD | 1282.8 | 66 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.9335e-05 (2.4414e-08) | 2.07 (2) | 2.02 (1) | 1.01 (1) | 1.00 (0) | 1.7524e-06 | 1.1805e-06 | 4.4797e-03 | 3.1485e-03 | 0.1 | 0.1 |
| C20 | OSGS | _run_quad_k1_postrefactor.json | 1e+06 | 1e-06 | 1.00 | 1 | QUAD | 2255.4 | 186 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.0173e-04 (2.4414e-08) | 2.08 (2) | 2.00 (1) | 1.04 (1) | 1.01 (0) | 1.8677e-06 | 1.1921e-06 | 4.5697e-03 | 3.1581e-03 | 0.1 | 0.1 |
| C21 | ASGS | _run_quad_k1_postrefactor.json | 1e+06 | 1e+00 | 0.50 | 1 | QUAD | 814.5 | 36 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.9159e-05 (2.4414e-08) | 2.08 (2) | 2.07 (1) | 1.01 (1) | 1.00 (0) | 2.8833e-06 | 7.2059e-07 | 1.3061e-02 | 3.1479e-03 | 0.1 | 0.1 |
| C21 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+06 | 1e+00 | 0.50 | 1 | QUAD | 1826.9 | 184 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 5.2376e-04 (2.4414e-08) | 2.15 (2) | 2.06 (1) | 1.02 (1) | 1.00 (0) | 3.0720e-06 | 7.3654e-07 | 1.3155e-02 | 3.1506e-03 | 0.1 | 0.1 |
| C22 | ASGS | _run_quad_k1_postrefactor.json | 1e+06 | 1e+00 | 1.00 | 1 | QUAD | 1261.4 | 63 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.9336e-05 (2.4414e-08) | 2.07 (2) | 2.02 (1) | 1.01 (1) | 1.00 (0) | 1.7524e-06 | 1.1805e-06 | 4.4797e-03 | 3.1485e-03 | 0.1 | 0.1 |
| C22 | OSGS | _run_quad_k1_postrefactor.json | 1e+06 | 1e+00 | 1.00 | 1 | QUAD | 2213.9 | 183 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.0175e-04 (2.4414e-08) | 2.08 (2) | 2.00 (1) | 1.04 (1) | 1.01 (0) | 1.8677e-06 | 1.1921e-06 | 4.5697e-03 | 3.1581e-03 | 0.1 | 0.1 |
| C23 | ASGS | _run_quad_k1_postrefactor.json | 1e+06 | 1e+06 | 0.50 | 1 | QUAD | 2579.4 | 71 | Yes | optimal | 1.5141e-05 (2.4414e-08) | 2.10 (2) | 2.01 (1) | 1.01 (1) | 1.00 (0) | 2.9466e-06 | 6.6477e-07 | 1.3072e-02 | 3.1479e-03 | 1 | 1 |
| C23 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+06 | 1e+06 | 0.50 | 1 | QUAD | 3723.0 | 178 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 9.5737e-03 (2.4414e-08) | 2.20 (2) | 2.01 (1) | 1.03 (1) | 1.00 (0) | 3.5670e-06 | 6.6771e-07 | 1.3390e-02 | 3.1502e-03 | 1 | 1 |
| C24 | ASGS | _run_quad_k1_postrefactor.json | 1e+06 | 1e+06 | 1.00 | 1 | QUAD | 1287.7 | 58 | Yes | partial-root | 3.2225e-06 (2.4414e-08) | 2.10 (2) | 2.00 (1) | 1.02 (1) | 1.00 (0) | 1.7904e-06 | 8.0062e-07 | 4.5138e-03 | 3.1480e-03 | 0.1 | 0.1 |
| C24 | OSGS | /Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/test/extended/ManufacturedSolutions/data/_run_quad_k1_postrefactor.json | 1e+06 | 1e+06 | 1.00 | 1 | QUAD | 2151.7 | 156 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.3947e-03 (2.4414e-08) | 2.16 (2) | 2.01 (1) | 1.07 (1) | 1.00 (0) | 2.1157e-06 | 8.0151e-07 | 4.8186e-03 | 3.1521e-03 | 0.1 | 0.1 |
| C52 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e-06 | 0.05 | 2 | QUAD | 300.9 | 12 | Yes | optimal | 5.3874e-17 (3.0518e-10) | 2.99 (3) | 1.81 (2) | 2.00 (2) | 0.93 (1) | 2.3855e-06 | 1.9543e-04 | 4.9676e-03 | 4.7955e-01 | 1 | 1 |
| C52 | ASGS | phase1_quad_k2.json | 1e-06 | 1e-06 | 0.05 | 2 | QUAD | 275.2 | 6 | Yes | optimal | 3.3329e-12 (3.0518e-10) | 2.99 (3) | 1.81 (2) | 2.00 (2) | 0.93 (1) | 2.3855e-06 | 1.9543e-04 | 4.9676e-03 | 4.7955e-01 | 1 | 1 |
| C52 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e-06 | 0.05 | 2 | QUAD | 4178.5 | 284 | Yes | optimal | 7.9373e-13 (3.0518e-10) | 2.98 (3) | 3.16 (2) | 1.99 (2) | 2.01 (1) | 2.3867e-06 | 4.9004e-04 | 4.9498e-03 | 1.1648e+00 | 1 | 1 |
| C52 | OSGS | phase1_quad_k2.json | 1e-06 | 1e-06 | 0.05 | 2 | QUAD | 2256.5 | 135 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 9.1708e-10 (3.0518e-10) | 2.98 (3) | 3.16 (2) | 1.99 (2) | 2.01 (1) | 2.3867e-06 | 4.9006e-04 | 4.9498e-03 | 1.1648e+00 | 1 | 1 |
| C53 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e-06 | 0.50 | 2 | QUAD | 314.3 | 12 | Yes | optimal | 5.9635e-16 (3.0518e-10) | 3.00 (3) | 1.88 (2) | 2.00 (2) | <b style='color:red'>0.87 (1)</b> | 2.0648e-07 | 1.9111e-05 | 4.2879e-04 | 4.7083e-02 | 1 | 1 |
| C53 | ASGS | phase1_quad_k2.json | 1e-06 | 1e-06 | 0.50 | 2 | QUAD | 274.4 | 6 | Yes | optimal | 1.6615e-11 (3.0518e-10) | 3.00 (3) | 1.88 (2) | 2.00 (2) | <b style='color:red'>0.87 (1)</b> | 2.0659e-07 | 1.9112e-05 | 4.2879e-04 | 4.7083e-02 | 1 | 1 |
| C53 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e-06 | 0.50 | 2 | QUAD | 3511.4 | 297 | Yes | optimal | 7.5099e-13 (3.0518e-10) | 3.00 (3) | 3.08 (2) | 2.00 (2) | 1.97 (1) | 2.0680e-07 | 2.9659e-05 | 4.2843e-04 | 7.1755e-02 | 1 | 1 |
| C53 | OSGS | phase1_quad_k2.json | 1e-06 | 1e-06 | 0.50 | 2 | QUAD | 1777.9 | 99 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 7.2332e-10 (3.0518e-10) | 3.00 (3) | 3.08 (2) | 2.00 (2) | 1.97 (1) | 2.0680e-07 | 2.9657e-05 | 4.2843e-04 | 7.1751e-02 | 1 | 1 |
| C54 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e-06 | 1.00 | 2 | QUAD | 372.4 | 12 | Yes | optimal | 1.1965e-15 (3.0518e-10) | 3.00 (3) | 2.00 (2) | 2.00 (2) | 1.00 (1) | 5.4414e-09 | 5.3181e-07 | 1.1285e-05 | 1.3182e-03 | 1 | 1 |
| C54 | ASGS | phase1_quad_k2.json | 1e-06 | 1e-06 | 1.00 | 2 | QUAD | 361.8 | 6 | Yes | <b style='color:#b8860b'>sub-optimal-rate ‡</b> | 6.2915e-11 (3.0518e-10) | <b style='color:red'>1.69 (3)</b> | <b style='color:red'>1.66 (2)</b> | 2.00 (2) | 1.00 (1) | 1.4054e-08 | 6.8538e-07 | 1.1286e-05 | 1.3182e-03 | 1 | 1 |
| C54 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e-06 | 1.00 | 2 | QUAD | 2749.3 | 306 | Yes | optimal | 1.2460e-12 (3.0518e-10) | 3.00 (3) | 2.47 (2) | 2.00 (2) | 1.44 (1) | 5.4455e-09 | 9.8123e-07 | 1.1291e-05 | 2.3011e-03 | 1 | 1 |
| C54 | OSGS | phase1_quad_k2.json | 1e-06 | 1e-06 | 1.00 | 2 | QUAD | 906.8 | 54 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.3452e-09 (3.0518e-10) | 3.00 (3) | 2.56 (2) | 2.00 (2) | 1.53 (1) | 5.4446e-09 | 8.8654e-07 | 1.1289e-05 | 2.0961e-03 | 1 | 1 |
| C55 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+00 | 0.05 | 2 | QUAD | 297.2 | 12 | Yes | optimal | 8.0163e-17 (3.0518e-10) | 3.02 (3) | 1.81 (2) | 2.00 (2) | <b style='color:red'>0.79 (1)</b> | 2.4010e-06 | 1.0542e-04 | 4.9568e-03 | 2.5572e-01 | 1 | 1 |
| C55 | ASGS | phase1_quad_k2.json | 1e-06 | 1e+00 | 0.05 | 2 | QUAD | 275.8 | 6 | Yes | optimal | 3.4147e-12 (3.0518e-10) | 3.02 (3) | 1.81 (2) | 2.00 (2) | <b style='color:red'>0.79 (1)</b> | 2.4010e-06 | 1.0542e-04 | 4.9568e-03 | 2.5572e-01 | 1 | 1 |
| C55 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+00 | 0.05 | 2 | QUAD | 2715.6 | 263 | Yes | optimal | 5.6354e-13 (3.0518e-10) | 3.01 (3) | 3.44 (2) | 2.00 (2) | 2.28 (1) | 2.4175e-06 | 3.1272e-04 | 4.9708e-03 | 7.4013e-01 | 1 | 1 |
| C55 | OSGS | phase1_quad_k2.json | 1e-06 | 1e+00 | 0.05 | 2 | QUAD | 1577.1 | 124 | Yes | optimal | 5.9326e-10 (3.0518e-10) | 3.01 (3) | 3.44 (2) | 2.00 (2) | 2.28 (1) | 2.4175e-06 | 3.1271e-04 | 4.9708e-03 | 7.4013e-01 | 1 | 1 |
| C56 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+00 | 0.50 | 2 | QUAD | 296.5 | 12 | Yes | optimal | 8.7915e-16 (3.0518e-10) | 3.00 (3) | 1.89 (2) | 2.00 (2) | <b style='color:red'>0.88 (1)</b> | 2.0657e-07 | 9.7484e-06 | 4.2854e-04 | 2.3975e-02 | 1 | 1 |
| C56 | ASGS | phase1_quad_k2.json | 1e-06 | 1e+00 | 0.50 | 2 | QUAD | 273.9 | 6 | Yes | optimal | 1.6621e-11 (3.0518e-10) | 3.00 (3) | 1.89 (2) | 2.00 (2) | <b style='color:red'>0.88 (1)</b> | 2.0667e-07 | 9.7489e-06 | 4.2854e-04 | 2.3975e-02 | 1 | 1 |
| C56 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+00 | 0.50 | 2 | QUAD | 2618.5 | 287 | Yes | optimal | 1.8898e-12 (3.0518e-10) | 3.01 (3) | 3.13 (2) | 2.00 (2) | 2.02 (1) | 2.0756e-07 | 1.6672e-05 | 4.2907e-04 | 4.0063e-02 | 1 | 1 |
| C56 | OSGS | phase1_quad_k2.json | 1e-06 | 1e+00 | 0.50 | 2 | QUAD | 1402.2 | 94 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 8.2979e-10 (3.0518e-10) | 3.01 (3) | 3.13 (2) | 2.00 (2) | 2.02 (1) | 2.0756e-07 | 1.6672e-05 | 4.2907e-04 | 4.0064e-02 | 1 | 1 |
| C57 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+00 | 1.00 | 2 | QUAD | 296.2 | 12 | Yes | optimal | 1.6654e-15 (3.0518e-10) | 3.00 (3) | 2.00 (2) | 2.00 (2) | 1.00 (1) | 5.4414e-09 | 2.6593e-07 | 1.1285e-05 | 6.5914e-04 | 1 | 1 |
| C57 | ASGS | phase1_quad_k2.json | 1e-06 | 1e+00 | 1.00 | 2 | QUAD | 269.7 | 6 | Yes | <b style='color:#b8860b'>sub-optimal-rate ‡</b> | 6.2954e-11 (3.0518e-10) | <b style='color:red'>1.70 (3)</b> | <b style='color:red'>1.66 (2)</b> | 2.00 (2) | 1.00 (1) | 1.3990e-08 | 3.4267e-07 | 1.1286e-05 | 6.5915e-04 | 1 | 1 |
| C57 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+00 | 1.00 | 2 | QUAD | 2816.0 | 302 | Yes | optimal | 2.0485e-12 (3.0518e-10) | 3.00 (3) | 2.41 (2) | 2.00 (2) | 1.40 (1) | 5.4452e-09 | 5.4705e-07 | 1.1290e-05 | 1.2625e-03 | 1 | 1 |
| C57 | OSGS | phase1_quad_k2.json | 1e-06 | 1e+00 | 1.00 | 2 | QUAD | 963.4 | 56 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.8186e-09 (3.0518e-10) | 3.00 (3) | 2.48 (2) | 2.00 (2) | 1.46 (1) | 5.4446e-09 | 5.1142e-07 | 1.1289e-05 | 1.1849e-03 | 1 | 1 |
| C58 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+06 | 0.05 | 2 | QUAD | 170.2 | 6 | Yes | optimal | 1.5002e-10 (3.0518e-10) | 3.00 (3) | 3.00 (2) | 2.01 (2) | 2.00 (1) | 2.4398e-06 | 3.8488e-09 | 5.1035e-03 | 7.9979e-06 | 1 | 1 |
| C58 | ASGS | phase1_quad_k2.json | 1e-06 | 1e+06 | 0.05 | 2 | QUAD | 275.9 | 6 | Yes | optimal | 1.5002e-10 (3.0518e-10) | 3.00 (3) | 3.00 (2) | 2.01 (2) | 2.00 (1) | 2.4398e-06 | 3.8488e-09 | 5.1035e-03 | 7.9979e-06 | 1 | 1 |
| C58 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+06 | 0.05 | 2 | QUAD | 1224.6 | 138 | Yes | optimal | 2.9232e-07 (3.0518e-10) | 2.95 (3) | 3.02 (2) | 2.10 (2) | 2.01 (1) | 2.3909e-06 | 3.9382e-09 | 5.1372e-03 | 8.2408e-06 | 1 | 1 |
| C58 | OSGS | phase1_quad_k2.json | 1e-06 | 1e+06 | 0.05 | 2 | QUAD | 333.6 | 34 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 1.2858e-05 (3.0518e-10) | 2.93 (3) | 3.06 (2) | 2.11 (2) | 2.05 (1) | 2.4398e-06 | 3.8488e-09 | 5.1035e-03 | 7.9979e-06 | 1 | 1 |
| C59 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+06 | 0.50 | 2 | QUAD | 169.3 | 6 | Yes | optimal | 5.9937e-10 (3.0518e-10) | 3.17 (3) | 3.00 (2) | 2.18 (2) | 2.00 (1) | 6.6291e-07 | 3.8429e-09 | 1.5162e-03 | 7.9819e-06 | 1 | 1 |
| C59 | ASGS | phase1_quad_k2.json | 1e-06 | 1e+06 | 0.50 | 2 | QUAD | 274.9 | 6 | Yes | optimal | 5.9937e-10 (3.0518e-10) | 3.17 (3) | 3.00 (2) | 2.18 (2) | 2.00 (1) | 6.6291e-07 | 3.8429e-09 | 1.5162e-03 | 7.9819e-06 | 1 | 1 |
| C59 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+06 | 0.50 | 2 | QUAD | 1325.9 | 140 | Yes | optimal ˢ | 2.9232e-07 (3.0518e-10) | 3.58 (3) | 3.00 (2) | 2.59 (2) | 2.01 (1) | 6.1827e-07 | 3.8438e-09 | 1.5058e-03 | 8.0038e-06 | 1 | 1 |
| C59 | OSGS | phase1_quad_k2.json | 1e-06 | 1e+06 | 0.50 | 2 | QUAD | 435.6 | 40 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.7387e-06 (3.0518e-10) | 3.52 (3) | 3.00 (2) | 2.62 (2) | 2.00 (1) | 6.6291e-07 | 3.8429e-09 | 1.5162e-03 | 7.9819e-06 | 1 | 1 |
| C60 | ASGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+06 | 1.00 | 2 | QUAD | 175.4 | 6 | Yes | optimal ˢ | 1.4482e-09 (3.0518e-10) | 3.28 (3) | 3.00 (2) | 2.28 (2) | 2.00 (1) | 6.7442e-07 | 3.8428e-09 | 1.5569e-03 | 7.9818e-06 | 1 | 1 |
| C60 | ASGS | phase1_quad_k2.json | 1e-06 | 1e+06 | 1.00 | 2 | QUAD | 274.4 | 6 | Yes | optimal ˢ | 1.4482e-09 (3.0518e-10) | 3.28 (3) | 3.00 (2) | 2.28 (2) | 2.00 (1) | 6.7442e-07 | 3.8428e-09 | 1.5569e-03 | 7.9818e-06 | 1 | 1 |
| C60 | OSGS | _consistency_k2_lowmidRe.json | 1e-06 | 1e+06 | 1.00 | 2 | QUAD | 1312.2 | 119 | Yes | optimal ˢ | 2.9232e-07 (3.0518e-10) | 3.70 (3) | 3.00 (2) | 2.70 (2) | 2.01 (1) | 5.9857e-07 | 3.8436e-09 | 1.4829e-03 | 8.0034e-06 | 1 | 1 |
| C60 | OSGS | phase1_quad_k2.json | 1e-06 | 1e+06 | 1.00 | 2 | QUAD | 422.9 | 36 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.7387e-06 (3.0518e-10) | 3.57 (3) | 3.00 (2) | 2.67 (2) | 2.00 (1) | 6.7442e-07 | 3.8428e-09 | 1.5569e-03 | 7.9818e-06 | 1 | 1 |
| C61 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e-06 | 0.05 | 2 | QUAD | 428.4 | 18 | Yes | optimal | 6.5312e-17 (3.0518e-10) | 3.02 (3) | 1.81 (2) | 2.00 (2) | <b style='color:red'>0.79 (1)</b> | 2.4026e-06 | 1.0566e-04 | 4.9570e-03 | 2.5638e-01 | 1 | 1 |
| C61 | ASGS | phase1_quad_k2.json | 1e+00 | 1e-06 | 0.05 | 2 | QUAD | 409.8 | 13 | Yes | optimal | 4.4434e-10 (3.0518e-10) | 3.02 (3) | 1.81 (2) | 2.00 (2) | <b style='color:red'>0.79 (1)</b> | 2.4139e-06 | 1.0588e-04 | 4.9570e-03 | 2.5637e-01 | 1 | 1 |
| C61 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e-06 | 0.05 | 2 | QUAD | 2843.5 | 269 | Yes | optimal | 6.4241e-13 (3.0518e-10) | 3.01 (3) | 3.45 (2) | 2.00 (2) | 2.28 (1) | 2.4165e-06 | 3.1108e-04 | 4.9698e-03 | 7.3638e-01 | 1 | 1 |
| C61 | OSGS | phase1_quad_k2.json | 1e+00 | 1e-06 | 0.05 | 2 | QUAD | 1701.8 | 130 | Yes | optimal | 6.2244e-10 (3.0518e-10) | 3.01 (3) | 3.45 (2) | 2.00 (2) | 2.28 (1) | 2.4165e-06 | 3.1107e-04 | 4.9697e-03 | 7.3638e-01 | 1 | 1 |
| C62 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e-06 | 0.50 | 2 | QUAD | 444.4 | 18 | Yes | optimal | 6.6541e-16 (3.0518e-10) | 3.00 (3) | 1.90 (2) | 2.00 (2) | <b style='color:red'>0.88 (1)</b> | 2.0657e-07 | 9.7848e-06 | 4.2854e-04 | 2.4064e-02 | 1 | 1 |
| C62 | ASGS | phase1_quad_k2.json | 1e+00 | 1e-06 | 0.50 | 2 | QUAD | 545.7 | 18 | Yes | optimal | 6.6541e-16 (3.0518e-10) | 3.00 (3) | 1.90 (2) | 2.00 (2) | <b style='color:red'>0.88 (1)</b> | 2.0657e-07 | 9.7848e-06 | 4.2854e-04 | 2.4064e-02 | 1 | 1 |
| C62 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e-06 | 0.50 | 2 | QUAD | 2849.7 | 293 | Yes | optimal | 1.8833e-12 (3.0518e-10) | 3.01 (3) | 3.13 (2) | 2.00 (2) | 2.02 (1) | 2.0756e-07 | 1.6700e-05 | 4.2907e-04 | 4.0135e-02 | 1 | 1 |
| C62 | OSGS | phase1_quad_k2.json | 1e+00 | 1e-06 | 0.50 | 2 | QUAD | 1674.1 | 106 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 8.3448e-10 (3.0518e-10) | 3.01 (3) | 3.13 (2) | 2.00 (2) | 2.02 (1) | 2.0756e-07 | 1.6700e-05 | 4.2907e-04 | 4.0136e-02 | 1 | 1 |
| C63 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e-06 | 1.00 | 2 | QUAD | 426.3 | 18 | Yes | optimal | 1.6864e-15 (3.0518e-10) | 3.00 (3) | 2.00 (2) | 2.00 (2) | 1.00 (1) | 5.4414e-09 | 2.6689e-07 | 1.1285e-05 | 6.6152e-04 | 1 | 1 |
| C63 | ASGS | phase1_quad_k2.json | 1e+00 | 1e-06 | 1.00 | 2 | QUAD | 544.7 | 18 | Yes | optimal | 1.6864e-15 (3.0518e-10) | 3.00 (3) | 2.00 (2) | 2.00 (2) | 1.00 (1) | 5.4414e-09 | 2.6689e-07 | 1.1285e-05 | 6.6152e-04 | 1 | 1 |
| C63 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e-06 | 1.00 | 2 | QUAD | 2944.1 | 308 | Yes | optimal | 2.0412e-12 (3.0518e-10) | 3.00 (3) | 2.41 (2) | 2.00 (2) | 1.40 (1) | 5.4452e-09 | 5.4740e-07 | 1.1290e-05 | 1.2636e-03 | 1 | 1 |
| C63 | OSGS | phase1_quad_k2.json | 1e+00 | 1e-06 | 1.00 | 2 | QUAD | 1244.7 | 68 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.8132e-09 (3.0518e-10) | 3.00 (3) | 2.48 (2) | 2.00 (2) | 1.46 (1) | 5.4446e-09 | 5.1185e-07 | 1.1289e-05 | 1.1862e-03 | 1 | 1 |
| C64 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+00 | 0.05 | 2 | QUAD | 427.0 | 18 | Yes | optimal | 6.5535e-17 (3.0518e-10) | 3.02 (3) | 1.81 (2) | 2.00 (2) | <b style='color:red'>0.79 (1)</b> | 2.4010e-06 | 7.0556e-05 | 4.9568e-03 | 1.7114e-01 | 1 | 1 |
| C64 | ASGS | phase1_quad_k2.json | 1e+00 | 1e+00 | 0.05 | 2 | QUAD | 408.0 | 12 | Yes | optimal | 4.0756e-10 (3.0518e-10) | 3.01 (3) | 1.81 (2) | 2.00 (2) | <b style='color:red'>0.79 (1)</b> | 2.4101e-06 | 7.0678e-05 | 4.9568e-03 | 1.7114e-01 | 1 | 1 |
| C64 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+00 | 0.05 | 2 | QUAD | 2817.4 | 267 | Yes | optimal | 5.5942e-13 (3.0518e-10) | 3.01 (3) | 3.44 (2) | 2.00 (2) | 2.28 (1) | 2.4175e-06 | 2.0883e-04 | 4.9709e-03 | 4.9426e-01 | 1 | 1 |
| C64 | OSGS | phase1_quad_k2.json | 1e+00 | 1e+00 | 0.05 | 2 | QUAD | 1710.3 | 126 | Yes | optimal | 5.9152e-10 (3.0518e-10) | 3.01 (3) | 3.44 (2) | 2.00 (2) | 2.28 (1) | 2.4175e-06 | 2.0883e-04 | 4.9709e-03 | 4.9426e-01 | 1 | 1 |
| C65 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+00 | 0.50 | 2 | QUAD | 425.8 | 18 | Yes | optimal | 6.9290e-16 (3.0518e-10) | 3.00 (3) | 1.90 (2) | 2.00 (2) | <b style='color:red'>0.88 (1)</b> | 2.0657e-07 | 6.5237e-06 | 4.2854e-04 | 1.6044e-02 | 1 | 1 |
| C65 | ASGS | phase1_quad_k2.json | 1e+00 | 1e+00 | 0.50 | 2 | QUAD | 540.2 | 18 | Yes | optimal | 6.9290e-16 (3.0518e-10) | 3.00 (3) | 1.90 (2) | 2.00 (2) | <b style='color:red'>0.88 (1)</b> | 2.0657e-07 | 6.5237e-06 | 4.2854e-04 | 1.6044e-02 | 1 | 1 |
| C65 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+00 | 0.50 | 2 | QUAD | 2751.5 | 291 | Yes | optimal | 1.8900e-12 (3.0518e-10) | 3.01 (3) | 3.13 (2) | 2.00 (2) | 2.02 (1) | 2.0756e-07 | 1.1138e-05 | 4.2907e-04 | 2.6767e-02 | 1 | 1 |
| C65 | OSGS | phase1_quad_k2.json | 1e+00 | 1e+00 | 0.50 | 2 | QUAD | 1672.1 | 104 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 8.2696e-10 (3.0518e-10) | 3.01 (3) | 3.13 (2) | 2.00 (2) | 2.02 (1) | 2.0756e-07 | 1.1138e-05 | 4.2907e-04 | 2.6768e-02 | 1 | 1 |
| C66 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+00 | 1.00 | 2 | QUAD | 426.3 | 18 | Yes | optimal | 1.4393e-15 (3.0518e-10) | 3.00 (3) | 2.01 (2) | 2.00 (2) | 1.00 (1) | 5.4414e-09 | 1.7795e-07 | 1.1285e-05 | 4.4105e-04 | 1 | 1 |
| C66 | ASGS | phase1_quad_k2.json | 1e+00 | 1e+00 | 1.00 | 2 | QUAD | 534.7 | 18 | Yes | optimal | 1.4393e-15 (3.0518e-10) | 3.00 (3) | 2.01 (2) | 2.00 (2) | 1.00 (1) | 5.4414e-09 | 1.7795e-07 | 1.1285e-05 | 4.4105e-04 | 1 | 1 |
| C66 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+00 | 1.00 | 2 | QUAD | 3039.2 | 302 | Yes | optimal | 2.0487e-12 (3.0518e-10) | 3.00 (3) | 2.41 (2) | 2.00 (2) | 1.40 (1) | 5.4452e-09 | 3.6509e-07 | 1.1290e-05 | 8.4272e-04 | 1 | 1 |
| C66 | OSGS | phase1_quad_k2.json | 1e+00 | 1e+00 | 1.00 | 2 | QUAD | 1241.6 | 68 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.8187e-09 (3.0518e-10) | 3.00 (3) | 2.48 (2) | 2.00 (2) | 1.46 (1) | 5.4446e-09 | 3.4136e-07 | 1.1289e-05 | 7.9103e-04 | 1 | 1 |
| C67 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+06 | 0.05 | 2 | QUAD | 297.2 | 12 | Yes | optimal | 5.5314e-11 (3.0518e-10) | 3.00 (3) | 3.00 (2) | 2.01 (2) | 2.00 (1) | 2.4398e-06 | 3.8489e-09 | 5.1035e-03 | 7.9980e-06 | 1 | 1 |
| C67 | ASGS | phase1_quad_k2.json | 1e+00 | 1e+06 | 0.05 | 2 | QUAD | 409.3 | 12 | Yes | optimal | 5.5314e-11 (3.0518e-10) | 3.00 (3) | 3.00 (2) | 2.01 (2) | 2.00 (1) | 2.4398e-06 | 3.8489e-09 | 5.1035e-03 | 7.9980e-06 | 1 | 1 |
| C67 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+06 | 0.05 | 2 | QUAD | 1347.7 | 144 | Yes | optimal | 2.9232e-07 (3.0518e-10) | 2.95 (3) | 3.02 (2) | 2.10 (2) | 2.01 (1) | 2.3909e-06 | 3.9383e-09 | 5.1372e-03 | 8.2413e-06 | 1 | 1 |
| C67 | OSGS | phase1_quad_k2.json | 1e+00 | 1e+06 | 0.05 | 2 | QUAD | 472.2 | 40 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 1.2856e-05 (3.0518e-10) | 2.93 (3) | 3.06 (2) | 2.11 (2) | 2.05 (1) | 2.4398e-06 | 3.8489e-09 | 5.1035e-03 | 7.9980e-06 | 1 | 1 |
| C68 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+06 | 0.50 | 2 | QUAD | 296.9 | 12 | Yes | optimal | 6.9366e-11 (3.0518e-10) | 3.17 (3) | 3.00 (2) | 2.18 (2) | 2.00 (1) | 6.6283e-07 | 3.8429e-09 | 1.5160e-03 | 7.9819e-06 | 1 | 1 |
| C68 | ASGS | phase1_quad_k2.json | 1e+00 | 1e+06 | 0.50 | 2 | QUAD | 409.1 | 12 | Yes | optimal | 6.9366e-11 (3.0518e-10) | 3.17 (3) | 3.00 (2) | 2.18 (2) | 2.00 (1) | 6.6283e-07 | 3.8429e-09 | 1.5160e-03 | 7.9819e-06 | 1 | 1 |
| C68 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+06 | 0.50 | 2 | QUAD | 1403.2 | 144 | Yes | optimal ˢ | 2.9232e-07 (3.0518e-10) | 3.58 (3) | 3.00 (2) | 2.59 (2) | 2.01 (1) | 6.1822e-07 | 3.8438e-09 | 1.5057e-03 | 8.0038e-06 | 1 | 1 |
| C68 | OSGS | phase1_quad_k2.json | 1e+00 | 1e+06 | 0.50 | 2 | QUAD | 531.0 | 44 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.7378e-06 (3.0518e-10) | 3.52 (3) | 3.00 (2) | 2.62 (2) | 2.00 (1) | 6.6283e-07 | 3.8429e-09 | 1.5160e-03 | 7.9819e-06 | 1 | 1 |
| C69 | ASGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+06 | 1.00 | 2 | QUAD | 313.7 | 14 | Yes | optimal ˢ | 1.3326e-10 (3.0518e-10) | 3.28 (3) | 3.00 (2) | 2.28 (2) | 2.00 (1) | 6.7426e-07 | 3.8428e-09 | 1.5566e-03 | 7.9818e-06 | 1 | 1 |
| C69 | ASGS | phase1_quad_k2.json | 1e+00 | 1e+06 | 1.00 | 2 | QUAD | 407.3 | 12 | Yes | optimal ˢ | 1.3326e-10 (3.0518e-10) | 3.28 (3) | 3.00 (2) | 2.28 (2) | 2.00 (1) | 6.7426e-07 | 3.8428e-09 | 1.5566e-03 | 7.9818e-06 | 1 | 1 |
| C69 | OSGS | _consistency_k2_lowmidRe.json | 1e+00 | 1e+06 | 1.00 | 2 | QUAD | 1446.3 | 127 | Yes | optimal ˢ | 2.9231e-07 (3.0518e-10) | 3.70 (3) | 3.00 (2) | 2.70 (2) | 2.01 (1) | 5.9848e-07 | 3.8436e-09 | 1.4827e-03 | 8.0034e-06 | 1 | 1 |
| C69 | OSGS | phase1_quad_k2.json | 1e+00 | 1e+06 | 1.00 | 2 | QUAD | 515.2 | 42 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.7368e-06 (3.0518e-10) | 3.57 (3) | 3.00 (2) | 2.67 (2) | 2.00 (1) | 6.7426e-07 | 3.8428e-09 | 1.5566e-03 | 7.9818e-06 | 1 | 1 |
| C70 | ASGS | corner_direct_solve | 1e+06 | 1e-06 | 0.05 | 2 | QUAD | 0.0 | 2 | Yes | <b style='color:#b8860b'>sub-optimal-rate ‡</b> | 1.2943e-11 (3.0518e-10) | 2.90 (3) | 2.81 (2) | <b style='color:red'>1.69 (2)</b> | 1.66 (1) | 5.0873e-06 | 8.0134e-07 | 1.2229e-02 | 1.9259e-03 | 0 | 0 |
| C70 | OSGS | corner_direct_solve | 1e+06 | 1e-06 | 0.05 | 2 | QUAD | 0.0 | 0 | Yes | partial-root | 1.4314e-09 (3.0518e-10) | 2.90 (3) | 2.81 (2) | <b style='color:red'>1.69 (2)</b> | 1.66 (1) | 5.0873e-06 | 8.0134e-07 | 1.2229e-02 | 1.9259e-03 | 0 | 0 |
| C71 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e-06 | 0.50 | 2 | QUAD | 549.8 | 34 | Yes | optimal | 2.7064e-10 (3.0518e-10) | 2.82 (3) | 2.90 (2) | 1.81 (2) | 1.87 (1) | 4.5613e-07 | 7.5783e-08 | 1.1013e-03 | 1.8614e-04 | 0.1 | 0.1 |
| C71 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e-06 | 0.50 | 2 | QUAD | 549.8 | 34 | Yes | optimal | 2.7064e-10 (3.0518e-10) | 2.82 (3) | 2.90 (2) | 1.81 (2) | 1.87 (1) | 4.5613e-07 | 7.5783e-08 | 1.1013e-03 | 1.8614e-04 | 0.1 | 0.1 |
| C71 | ASGS | phase1_quad_k2.json | 1e+06 | 1e-06 | 0.50 | 2 | QUAD | 547.7 | 29 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 7.7764e-05 (3.0518e-10) | <b style='color:red'>2.34 (3)</b> | 1.87 (2) | 1.81 (2) | 1.87 (1) | 6.4385e-07 | 1.5977e-07 | 1.1015e-03 | 1.8615e-04 | 0.1 | 0.1 |
| C71 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e-06 | 0.50 | 2 | QUAD | 2076.9 | 326 | Yes | optimal | 1.5370e-07 (3.0518e-10) | 2.83 (3) | 3.04 (2) | 1.81 (2) | 2.00 (1) | 4.5734e-07 | 7.9141e-08 | 1.1024e-03 | 1.9424e-04 | 0.1 | 0.1 |
| C71 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e-06 | 0.50 | 2 | QUAD | 2076.9 | 326 | Yes | optimal | 1.5370e-07 (3.0518e-10) | 2.83 (3) | 3.04 (2) | 1.81 (2) | 2.00 (1) | 4.5734e-07 | 7.9141e-08 | 1.1024e-03 | 1.9424e-04 | 0.1 | 0.1 |
| C71 | OSGS | phase1_quad_k2.json | 1e+06 | 1e-06 | 0.50 | 2 | QUAD | 806.1 | 104 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.9119e-04 (3.0518e-10) | 2.83 (3) | 3.04 (2) | 1.81 (2) | 2.00 (1) | 4.5731e-07 | 7.8954e-08 | 1.1024e-03 | 1.9379e-04 | 0.1 | 0.1 |
| C72 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e-06 | 1.00 | 2 | QUAD | 1766.3 | 47 | Yes | optimal ˢ | 5.1801e-08 (3.0518e-10) | 3.66 (3) | 3.01 (2) | 2.07 (2) | 2.00 (1) | 1.1873e-08 | 5.9132e-09 | 2.5834e-05 | 1.2116e-05 | 1 | 0.1 |
| C72 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e-06 | 1.00 | 2 | QUAD | 1766.3 | 47 | Yes | optimal ˢ | 5.1801e-08 (3.0518e-10) | 3.66 (3) | 3.01 (2) | 2.07 (2) | 2.00 (1) | 1.1873e-08 | 5.9132e-09 | 2.5834e-05 | 1.2116e-05 | 1 | 0.1 |
| C72 | ASGS | phase1_quad_k2.json | 1e+06 | 1e-06 | 1.00 | 2 | QUAD | 1908.7 | 42 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 2.4391e-04 (3.0518e-10) | <b style='color:red'>1.74 (3)</b> | 4.10 (2) | <b style='color:red'>0.82 (2)</b> | 2.00 (1) | 7.0211e-08 | 6.4230e-09 | 6.1696e-05 | 1.2153e-05 | 1 | 0.1 |
| C72 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e-06 | 1.00 | 2 | QUAD | 3068.9 | 309 | Yes | optimal ˢ | 3.8505e-07 (3.0518e-10) | 3.34 (3) | 3.01 (2) | 2.02 (2) | 2.01 (1) | 1.3565e-08 | 7.1937e-09 | 2.5844e-05 | 1.2159e-05 | 1 | 0.1 |
| C72 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e-06 | 1.00 | 2 | QUAD | 3068.9 | 309 | Yes | optimal ˢ | 3.8505e-07 (3.0518e-10) | 3.34 (3) | 3.01 (2) | 2.02 (2) | 2.01 (1) | 1.3565e-08 | 7.1937e-09 | 2.5844e-05 | 1.2159e-05 | 1 | 0.1 |
| C72 | OSGS | phase1_quad_k2.json | 1e+06 | 1e-06 | 1.00 | 2 | QUAD | 1886.1 | 80 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.3964e-04 (3.0518e-10) | <b style='color:red'>1.74 (3)</b> | 4.10 (2) | <b style='color:red'>0.82 (2)</b> | 2.00 (1) | 7.0211e-08 | 6.4230e-09 | 6.1696e-05 | 1.2153e-05 | 1 | 0.1 |
| C73 | ASGS | corner_direct_solve | 1e+06 | 1e+00 | 0.05 | 2 | QUAD | 0.0 | 2 | Yes | <b style='color:#b8860b'>sub-optimal-rate ‡</b> | 1.2943e-11 (3.0518e-10) | 2.90 (3) | 2.81 (2) | <b style='color:red'>1.69 (2)</b> | 1.66 (1) | 5.0873e-06 | 8.0134e-07 | 1.2229e-02 | 1.9259e-03 | 0 | 0 |
| C73 | OSGS | corner_direct_solve | 1e+06 | 1e+00 | 0.05 | 2 | QUAD | 0.0 | 0 | Yes | partial-root | 1.4314e-09 (3.0518e-10) | 2.90 (3) | 2.81 (2) | <b style='color:red'>1.69 (2)</b> | 1.66 (1) | 5.0873e-06 | 8.0134e-07 | 1.2229e-02 | 1.9259e-03 | 0 | 0 |
| C74 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e+00 | 0.50 | 2 | QUAD | 554.5 | 34 | Yes | optimal | 2.7074e-10 (3.0518e-10) | 2.82 (3) | 2.90 (2) | 1.81 (2) | 1.87 (1) | 4.5613e-07 | 7.5788e-08 | 1.1013e-03 | 1.8615e-04 | 0.1 | 0.1 |
| C74 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e+00 | 0.50 | 2 | QUAD | 554.5 | 34 | Yes | optimal | 2.7074e-10 (3.0518e-10) | 2.82 (3) | 2.90 (2) | 1.81 (2) | 1.87 (1) | 4.5613e-07 | 7.5788e-08 | 1.1013e-03 | 1.8615e-04 | 0.1 | 0.1 |
| C74 | ASGS | phase1_quad_k2.json | 1e+06 | 1e+00 | 0.50 | 2 | QUAD | 545.1 | 29 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 7.7763e-05 (3.0518e-10) | <b style='color:red'>2.34 (3)</b> | 1.87 (2) | 1.81 (2) | 1.87 (1) | 6.4385e-07 | 1.5977e-07 | 1.1015e-03 | 1.8616e-04 | 0.1 | 0.1 |
| C74 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e+00 | 0.50 | 2 | QUAD | 2075.6 | 326 | Yes | optimal | 1.5377e-07 (3.0518e-10) | 2.83 (3) | 3.04 (2) | 1.81 (2) | 2.00 (1) | 4.5734e-07 | 7.9147e-08 | 1.1024e-03 | 1.9425e-04 | 0.1 | 0.1 |
| C74 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e+00 | 0.50 | 2 | QUAD | 2075.6 | 326 | Yes | optimal | 1.5377e-07 (3.0518e-10) | 2.83 (3) | 3.04 (2) | 1.81 (2) | 2.00 (1) | 4.5734e-07 | 7.9147e-08 | 1.1024e-03 | 1.9425e-04 | 0.1 | 0.1 |
| C74 | OSGS | phase1_quad_k2.json | 1e+06 | 1e+00 | 0.50 | 2 | QUAD | 805.7 | 104 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.9119e-04 (3.0518e-10) | 2.83 (3) | 3.04 (2) | 1.81 (2) | 2.00 (1) | 4.5731e-07 | 7.8960e-08 | 1.1024e-03 | 1.9381e-04 | 0.1 | 0.1 |
| C75 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e+00 | 1.00 | 2 | QUAD | 1710.2 | 47 | Yes | optimal ˢ | 5.1799e-08 (3.0518e-10) | 3.66 (3) | 3.01 (2) | 2.07 (2) | 2.00 (1) | 1.1872e-08 | 5.9116e-09 | 2.5829e-05 | 1.2111e-05 | 1 | 0.1 |
| C75 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e+00 | 1.00 | 2 | QUAD | 1710.2 | 47 | Yes | optimal ˢ | 5.1799e-08 (3.0518e-10) | 3.66 (3) | 3.01 (2) | 2.07 (2) | 2.00 (1) | 1.1872e-08 | 5.9116e-09 | 2.5829e-05 | 1.2111e-05 | 1 | 0.1 |
| C75 | ASGS | phase1_quad_k2.json | 1e+06 | 1e+00 | 1.00 | 2 | QUAD | 1920.4 | 42 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 2.4391e-04 (3.0518e-10) | <b style='color:red'>1.74 (3)</b> | 4.10 (2) | <b style='color:red'>0.82 (2)</b> | 2.00 (1) | 7.0210e-08 | 6.4215e-09 | 6.1694e-05 | 1.2148e-05 | 1 | 0.1 |
| C75 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e+00 | 1.00 | 2 | QUAD | 3043.7 | 309 | Yes | optimal ˢ | 3.8543e-07 (3.0518e-10) | 3.34 (3) | 3.01 (2) | 2.02 (2) | 2.01 (1) | 1.3563e-08 | 7.1924e-09 | 2.5840e-05 | 1.2155e-05 | 1 | 0.1 |
| C75 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e+00 | 1.00 | 2 | QUAD | 3043.7 | 309 | Yes | optimal ˢ | 3.8543e-07 (3.0518e-10) | 3.34 (3) | 3.01 (2) | 2.02 (2) | 2.01 (1) | 1.3563e-08 | 7.1924e-09 | 2.5840e-05 | 1.2155e-05 | 1 | 0.1 |
| C75 | OSGS | phase1_quad_k2.json | 1e+06 | 1e+00 | 1.00 | 2 | QUAD | 1934.0 | 80 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.3964e-04 (3.0518e-10) | <b style='color:red'>1.74 (3)</b> | 4.10 (2) | <b style='color:red'>0.82 (2)</b> | 2.00 (1) | 7.0210e-08 | 6.4215e-09 | 6.1694e-05 | 1.2148e-05 | 1 | 0.1 |
| C76 | ASGS | corner_direct_solve | 1e+06 | 1e+06 | 0.05 | 2 | QUAD | 0.0 | 2 | Yes | <b style='color:#b8860b'>sub-optimal-rate ‡</b> | 2.9909e-12 (3.0518e-10) | <b style='color:red'>2.63 (3)</b> | 2.76 (2) | <b style='color:red'>1.64 (2)</b> | 1.67 (1) | 4.2617e-06 | 4.0342e-07 | 1.0223e-02 | 9.7190e-04 | 0 | 0 |
| C76 | OSGS | corner_direct_solve | 1e+06 | 1e+06 | 0.05 | 2 | QUAD | 0.0 | 0 | Yes | partial-root | 1.4245e-09 (3.0518e-10) | <b style='color:red'>2.63 (3)</b> | 2.76 (2) | <b style='color:red'>1.64 (2)</b> | 1.67 (1) | 4.2617e-06 | 4.0342e-07 | 1.0223e-02 | 9.7190e-04 | 0 | 0 |
| C77 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e+06 | 0.50 | 2 | QUAD | 810.4 | 57 | Yes | optimal | 7.7307e-12 (3.0518e-10) | 2.79 (3) | 2.91 (2) | <b style='color:red'>1.79 (2)</b> | 1.88 (1) | 4.4329e-07 | 3.8161e-08 | 1.0687e-03 | 9.3638e-05 | 0.1 | 0.1 |
| C77 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e+06 | 0.50 | 2 | QUAD | 810.4 | 57 | Yes | optimal | 7.7307e-12 (3.0518e-10) | 2.79 (3) | 2.91 (2) | <b style='color:red'>1.79 (2)</b> | 1.88 (1) | 4.4329e-07 | 3.8161e-08 | 1.0687e-03 | 9.3638e-05 | 0.1 | 0.1 |
| C77 | ASGS | phase1_quad_k2.json | 1e+06 | 1e+06 | 0.50 | 2 | QUAD | 853.9 | 52 | Yes | optimal | 1.5405e-06 (3.0518e-10) | 2.79 (3) | 2.91 (2) | <b style='color:red'>1.79 (2)</b> | 1.88 (1) | 4.4341e-07 | 3.8212e-08 | 1.0687e-03 | 9.3638e-05 | 0.1 | 0.1 |
| C77 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e+06 | 0.50 | 2 | QUAD | 2155.5 | 268 | Yes | optimal | 3.0928e-07 (3.0518e-10) | 2.80 (3) | 3.04 (2) | <b style='color:red'>1.80 (2)</b> | 2.00 (1) | 4.4506e-07 | 3.9796e-08 | 1.0712e-03 | 9.7593e-05 | 0.1 | 0.1 |
| C77 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e+06 | 0.50 | 2 | QUAD | 2155.5 | 268 | Yes | optimal | 3.0928e-07 (3.0518e-10) | 2.80 (3) | 3.04 (2) | <b style='color:red'>1.80 (2)</b> | 2.00 (1) | 4.4506e-07 | 3.9796e-08 | 1.0712e-03 | 9.7593e-05 | 0.1 | 0.1 |
| C77 | OSGS | phase1_quad_k2.json | 1e+06 | 1e+06 | 0.50 | 2 | QUAD | 1084.7 | 94 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.9269e-05 (3.0518e-10) | 2.80 (3) | 3.04 (2) | <b style='color:red'>1.80 (2)</b> | 2.00 (1) | 4.4500e-07 | 3.9701e-08 | 1.0711e-03 | 9.7371e-05 | 0.1 | 0.1 |
| C78 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e+06 | 1.00 | 2 | QUAD | 1504.9 | 56 | Yes | optimal | 9.9973e-12 (3.0518e-10) | 3.09 (3) | 3.00 (2) | 1.99 (2) | 2.00 (1) | 1.3727e-08 | 5.0953e-09 | 3.2339e-05 | 1.1286e-05 | 1 | 1 |
| C78 | ASGS | _surgical_k2_hiRe.json | 1e+06 | 1e+06 | 1.00 | 2 | QUAD | 1504.9 | 56 | Yes | optimal | 9.9973e-12 (3.0518e-10) | 3.09 (3) | 3.00 (2) | 1.99 (2) | 2.00 (1) | 1.3727e-08 | 5.0953e-09 | 3.2339e-05 | 1.1286e-05 | 1 | 1 |
| C78 | ASGS | phase1_quad_k2.json | 1e+06 | 1e+06 | 1.00 | 2 | QUAD | 1649.8 | 51 | Yes | optimal ˢ | 4.9777e-06 (3.0518e-10) | 3.38 (3) | 3.01 (2) | 1.99 (2) | 2.00 (1) | 1.3809e-08 | 5.0951e-09 | 3.2377e-05 | 1.1286e-05 | 1 | 1 |
| C78 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e+06 | 1.00 | 2 | QUAD | 2661.8 | 221 | Yes | optimal | 7.6319e-07 (3.0518e-10) | 3.04 (3) | 3.00 (2) | 1.98 (2) | 2.00 (1) | 1.4453e-08 | 5.3077e-09 | 3.2363e-05 | 1.1321e-05 | 1 | 1 |
| C78 | OSGS | _surgical_k2_hiRe.json | 1e+06 | 1e+06 | 1.00 | 2 | QUAD | 2661.8 | 221 | Yes | optimal | 7.6319e-07 (3.0518e-10) | 3.04 (3) | 3.00 (2) | 1.98 (2) | 2.00 (1) | 1.4453e-08 | 5.3077e-09 | 3.2363e-05 | 1.1321e-05 | 1 | 1 |
| C78 | OSGS | phase1_quad_k2.json | 1e+06 | 1e+06 | 1.00 | 2 | QUAD | 1662.5 | 77 | Yes | optimal ˢ | 1.7495e-05 (3.0518e-10) | 3.38 (3) | 3.01 (2) | 1.99 (2) | 2.00 (1) | 1.3809e-08 | 5.0951e-09 | 3.2377e-05 | 1.1286e-05 | 1 | 1 |
| C25 | ASGS | phase1_tri_k1.json | 1e-06 | 1e-06 | 0.05 | 1 | TRI | 199.2 | 6 | Yes | optimal | 1.1936e-11 (9.7656e-08) | 1.90 (2) | 1.97 (1) | 1.01 (1) | 1.95 (0) | 6.8384e-04 | 2.0823e-02 | 1.7611e-01 | 2.6054e+00 | 1 | 1 |
| C25 | OSGS | phase1_tri_k1.json | 1e-06 | 1e-06 | 0.05 | 1 | TRI | 1208.3 | 132 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.4031e-08 (9.7656e-08) | 2.08 (2) | 2.04 (1) | 1.00 (1) | 2.07 (0) | 6.3293e-04 | 1.7555e-02 | 1.7575e-01 | 2.2409e+00 | 1 | 1 |
| C26 | ASGS | phase1_tri_k1.json | 1e-06 | 1e-06 | 0.50 | 1 | TRI | 197.5 | 6 | Yes | optimal | 5.9624e-11 (9.7656e-08) | 2.01 (2) | 2.20 (1) | 1.00 (1) | 1.56 (0) | 6.9544e-05 | 2.3454e-03 | 3.5038e-02 | 3.1617e-01 | 1 | 1 |
| C26 | OSGS | phase1_tri_k1.json | 1e-06 | 1e-06 | 0.50 | 1 | TRI | 1944.5 | 234 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 5.4903e-08 (9.7656e-08) | 2.12 (2) | 2.20 (1) | 1.00 (1) | 1.63 (0) | 3.6249e-05 | 1.8607e-03 | 3.5020e-02 | 2.6482e-01 | 1 | 1 |
| C27 | ASGS | phase1_tri_k1.json | 1e-06 | 1e-06 | 1.00 | 1 | TRI | 458.2 | 6 | Yes | optimal | 2.2569e-10 (9.7656e-08) | 2.00 (2) | 1.79 (1) | 1.00 (1) | 0.59 (0) | 3.8278e-05 | 1.1632e-03 | 1.5439e-02 | 4.9890e-01 | 1 | 1 |
| C27 | OSGS | phase1_tri_k1.json | 1e-06 | 1e-06 | 1.00 | 1 | TRI | 3039.2 | 283 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.7488e-08 (9.7656e-08) | 2.03 (2) | 1.88 (1) | 1.00 (1) | 0.79 (0) | 9.2517e-06 | 6.5231e-04 | 1.5425e-02 | 4.2985e-01 | 1 | 1 |
| C28 | ASGS | phase1_tri_k1.json | 1e-06 | 1e+00 | 0.05 | 1 | TRI | 203.7 | 6 | Yes | optimal | 1.2669e-11 (9.7656e-08) | 1.89 (2) | 1.76 (1) | 1.00 (1) | 1.68 (0) | 4.8334e-04 | 2.1013e-02 | 1.7613e-01 | 2.7125e+00 | 1 | 1 |
| C28 | OSGS | phase1_tri_k1.json | 1e-06 | 1e+00 | 0.05 | 1 | TRI | 1542.8 | 86 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 3.1696e-08 (9.7656e-08) | 2.00 (2) | 2.00 (1) | 1.00 (1) | 2.02 (0) | 8.7416e-04 | 1.3920e-02 | 1.7583e-01 | 1.7569e+00 | 1 | 1 |
| C29 | ASGS | phase1_tri_k1.json | 1e-06 | 1e+00 | 0.50 | 1 | TRI | 197.2 | 6 | Yes | optimal | 5.9697e-11 (9.7656e-08) | 1.98 (2) | 1.88 (1) | 1.00 (1) | 1.47 (0) | 7.3736e-05 | 2.0005e-03 | 3.5038e-02 | 2.1893e-01 | 1 | 1 |
| C29 | OSGS | phase1_tri_k1.json | 1e-06 | 1e+00 | 0.50 | 1 | TRI | 2795.2 | 251 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 5.9087e-08 (9.7656e-08) | 2.00 (2) | 2.01 (1) | 1.00 (1) | 1.46 (0) | 4.5652e-05 | 1.3319e-03 | 3.5022e-02 | 1.8061e-01 | 1 | 1 |
| C30 | ASGS | phase1_tri_k1.json | 1e-06 | 1e+00 | 1.00 | 1 | TRI | 197.8 | 6 | Yes | optimal | 2.2612e-10 (9.7656e-08) | 1.99 (2) | 1.60 (1) | 1.00 (1) | 0.51 (0) | 3.8493e-05 | 7.5405e-04 | 1.5436e-02 | 2.8095e-01 | 1 | 1 |
| C30 | OSGS | phase1_tri_k1.json | 1e-06 | 1e+00 | 1.00 | 1 | TRI | 3552.9 | 297 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 4.4177e-08 (9.7656e-08) | 2.00 (2) | 1.64 (1) | 1.00 (1) | 0.58 (0) | 9.6297e-06 | 4.5202e-04 | 1.5425e-02 | 2.8360e-01 | 1 | 1 |
| C31 | ASGS | phase1_tri_k1.json | 1e-06 | 1e+06 | 0.05 | 1 | TRI | 201.7 | 6 | Yes | optimal | 1.2576e-10 (9.7656e-08) | 1.95 (2) | 2.17 (1) | 1.02 (1) | 1.00 (0) | 4.3731e-04 | 1.6187e-05 | 1.7684e-01 | 1.0907e-02 | 1 | 1 |
| C31 | OSGS | phase1_tri_k1.json | 1e-06 | 1e+06 | 0.05 | 1 | TRI | 1043.0 | 329 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.4955e-04 (9.7656e-08) | 2.06 (2) | 2.17 (1) | 1.04 (1) | 1.00 (0) | 1.9219e-04 | 1.6207e-05 | 2.0965e-01 | 1.0923e-02 | 1 | 1 |
| C32 | ASGS | phase1_tri_k1.json | 1e-06 | 1e+06 | 0.50 | 1 | TRI | 196.9 | 6 | Yes | optimal | 1.9146e-09 (9.7656e-08) | 2.12 (2) | 2.27 (1) | 1.02 (1) | 1.00 (0) | 1.0716e-04 | 8.7624e-06 | 3.5166e-02 | 1.0904e-02 | 1 | 1 |
| C32 | OSGS | phase1_tri_k1.json | 1e-06 | 1e+06 | 0.50 | 1 | TRI | 882.1 | 86 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 3.4954e-04 (9.7656e-08) | 2.03 (2) | 2.27 (1) | 1.08 (1) | 1.00 (0) | 1.7209e-04 | 8.7628e-06 | 1.1889e-01 | 1.0921e-02 | 1 | 1 |
| C33 | ASGS | phase1_tri_k1.json | 1e-06 | 1e+06 | 1.00 | 1 | TRI | 198.2 | 6 | Yes | optimal | 6.7901e-09 (9.7656e-08) | 2.21 (2) | 2.28 (1) | 1.05 (1) | 1.00 (0) | 7.3538e-05 | 8.1666e-06 | 1.5714e-02 | 1.0904e-02 | 1 | 1 |
| C33 | OSGS | phase1_tri_k1.json | 1e-06 | 1e+06 | 1.00 | 1 | TRI | 879.0 | 80 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 3.4954e-04 (9.7656e-08) | 2.03 (2) | 2.28 (1) | 1.08 (1) | 1.00 (0) | 1.7108e-04 | 8.1815e-06 | 1.1466e-01 | 1.0921e-02 | 1 | 1 |
| C34 | ASGS | phase1_tri_k1.json | 1e+00 | 1e-06 | 0.05 | 1 | TRI | 274.9 | 15 | Yes | optimal | 1.5871e-09 (9.7656e-08) | 1.88 (2) | 1.79 (1) | 1.00 (1) | 1.71 (0) | 5.0853e-04 | 2.0440e-02 | 1.7612e-01 | 2.6403e+00 | 1 | 1 |
| C34 | OSGS | phase1_tri_k1.json | 1e+00 | 1e-06 | 0.05 | 1 | TRI | 1606.2 | 90 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 3.1607e-08 (9.7656e-08) | 2.01 (2) | 2.01 (1) | 1.01 (1) | 2.05 (0) | 9.5300e-04 | 1.3433e-02 | 1.7584e-01 | 1.7120e+00 | 1 | 1 |
| C35 | ASGS | phase1_tri_k1.json | 1e+00 | 1e-06 | 0.50 | 1 | TRI | 346.5 | 18 | Yes | optimal | 4.2219e-16 (9.7656e-08) | 1.98 (2) | 1.88 (1) | 1.00 (1) | 1.47 (0) | 7.3610e-05 | 1.9936e-03 | 3.5038e-02 | 2.1846e-01 | 1 | 1 |
| C35 | OSGS | phase1_tri_k1.json | 1e+00 | 1e-06 | 0.50 | 1 | TRI | 2929.2 | 263 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 5.9116e-08 (9.7656e-08) | 2.00 (2) | 2.01 (1) | 1.00 (1) | 1.46 (0) | 4.6018e-05 | 1.3312e-03 | 3.5022e-02 | 1.8047e-01 | 1 | 1 |
| C36 | ASGS | phase1_tri_k1.json | 1e+00 | 1e-06 | 1.00 | 1 | TRI | 356.0 | 18 | Yes | optimal | 2.0101e-15 (9.7656e-08) | 1.99 (2) | 1.60 (1) | 1.00 (1) | 0.51 (0) | 3.8455e-05 | 7.5258e-04 | 1.5436e-02 | 2.8074e-01 | 1 | 1 |
| C36 | OSGS | phase1_tri_k1.json | 1e+00 | 1e-06 | 1.00 | 1 | TRI | 3746.1 | 308 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 4.4354e-08 (9.7656e-08) | 2.00 (2) | 1.65 (1) | 1.00 (1) | 0.59 (0) | 9.6593e-06 | 4.5207e-04 | 1.5425e-02 | 2.8363e-01 | 1 | 1 |
| C37 | ASGS | phase1_tri_k1.json | 1e+00 | 1e+00 | 0.05 | 1 | TRI | 275.4 | 11 | Yes | optimal | 1.4554e-09 (9.7656e-08) | 1.89 (2) | 1.76 (1) | 1.00 (1) | 1.68 (0) | 4.8285e-04 | 1.4009e-02 | 1.7613e-01 | 1.8082e+00 | 1 | 1 |
| C37 | OSGS | phase1_tri_k1.json | 1e+00 | 1e+00 | 0.05 | 1 | TRI | 1591.7 | 88 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 3.1710e-08 (9.7656e-08) | 2.00 (2) | 2.00 (1) | 1.00 (1) | 2.02 (0) | 8.7465e-04 | 9.2898e-03 | 1.7583e-01 | 1.1726e+00 | 1 | 1 |
| C38 | ASGS | phase1_tri_k1.json | 1e+00 | 1e+00 | 0.50 | 1 | TRI | 347.5 | 18 | Yes | optimal | 4.3390e-16 (9.7656e-08) | 1.98 (2) | 1.88 (1) | 1.00 (1) | 1.47 (0) | 7.3693e-05 | 1.3327e-03 | 3.5038e-02 | 1.4611e-01 | 1 | 1 |
| C38 | OSGS | phase1_tri_k1.json | 1e+00 | 1e+00 | 0.50 | 1 | TRI | 2852.0 | 259 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 6.7153e-08 (9.7656e-08) | 2.00 (2) | 2.01 (1) | 1.00 (1) | 1.46 (0) | 4.5660e-05 | 8.8824e-04 | 3.5022e-02 | 1.2068e-01 | 1 | 1 |
| C39 | ASGS | phase1_tri_k1.json | 1e+00 | 1e+00 | 1.00 | 1 | TRI | 348.6 | 18 | Yes | optimal | 1.9837e-15 (9.7656e-08) | 1.99 (2) | 1.60 (1) | 1.00 (1) | 0.51 (0) | 3.8457e-05 | 5.0234e-04 | 1.5436e-02 | 1.8741e-01 | 1 | 1 |
| C39 | OSGS | phase1_tri_k1.json | 1e+00 | 1e+00 | 1.00 | 1 | TRI | 3617.0 | 303 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 5.0325e-08 (9.7656e-08) | 2.00 (2) | 1.64 (1) | 1.00 (1) | 0.59 (0) | 9.6303e-06 | 3.0153e-04 | 1.5425e-02 | 1.8938e-01 | 1 | 1 |
| C40 | ASGS | phase1_tri_k1.json | 1e+00 | 1e+06 | 0.05 | 1 | TRI | 272.5 | 11 | Yes | optimal | 5.0948e-10 (9.7656e-08) | 1.95 (2) | 2.17 (1) | 1.02 (1) | 1.00 (0) | 4.3698e-04 | 1.6187e-05 | 1.7684e-01 | 1.0907e-02 | 1 | 1 |
| C40 | OSGS | phase1_tri_k1.json | 1e+00 | 1e+06 | 0.05 | 1 | TRI | 991.2 | 183 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 3.4955e-04 (9.7656e-08) | 2.06 (2) | 2.17 (1) | 1.04 (1) | 1.00 (0) | 1.9219e-04 | 1.6207e-05 | 2.0965e-01 | 1.0923e-02 | 1 | 1 |
| C41 | ASGS | phase1_tri_k1.json | 1e+00 | 1e+06 | 0.50 | 1 | TRI | 272.0 | 12 | Yes | optimal | 8.0450e-09 (9.7656e-08) | 2.12 (2) | 2.27 (1) | 1.02 (1) | 1.00 (0) | 1.0709e-04 | 8.7614e-06 | 3.5166e-02 | 1.0904e-02 | 1 | 1 |
| C41 | OSGS | phase1_tri_k1.json | 1e+00 | 1e+06 | 0.50 | 1 | TRI | 952.4 | 92 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 3.4954e-04 (9.7656e-08) | 2.03 (2) | 2.27 (1) | 1.08 (1) | 1.00 (0) | 1.7207e-04 | 8.7618e-06 | 1.1888e-01 | 1.0921e-02 | 1 | 1 |
| C42 | ASGS | phase1_tri_k1.json | 1e+00 | 1e+06 | 1.00 | 1 | TRI | 273.3 | 12 | Yes | optimal | 3.0695e-08 (9.7656e-08) | 2.21 (2) | 2.28 (1) | 1.05 (1) | 1.00 (0) | 7.3497e-05 | 8.1649e-06 | 1.5714e-02 | 1.0904e-02 | 1 | 1 |
| C42 | OSGS | phase1_tri_k1.json | 1e+00 | 1e+06 | 1.00 | 1 | TRI | 953.3 | 86 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 3.4954e-04 (9.7656e-08) | 2.03 (2) | 2.28 (1) | 1.08 (1) | 1.00 (0) | 1.7105e-04 | 8.1799e-06 | 1.1464e-01 | 1.0921e-02 | 1 | 1 |
| C43 | ASGS | corner_direct_solve | 1e+06 | 1e-06 | 0.05 | 1 | TRI | 0.0 | 6 | Yes | optimal ˢ | 3.8081e-12 (1.6954e-08) | 3.03 (2) | 3.04 (1) | 1.02 (1) | 1.02 (0) | 7.9411e-05 | 7.3834e-06 | 7.3338e-02 | 4.5510e-03 | 0 | 0 |
| C43 | OSGS | corner_direct_solve | 1e+06 | 1e-06 | 0.05 | 1 | TRI | 0.0 | 4 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 8.7163e-07 (1.6954e-08) | 3.09 (2) | 3.10 (1) | 1.05 (1) | 1.02 (0) | 7.9411e-05 | 7.3834e-06 | 7.3338e-02 | 4.5510e-03 | 0 | 0 |
| C44 | ASGS | phase1_tri_k1.json | 1e+06 | 1e-06 | 0.50 | 1 | TRI | 360.4 | 36 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.6523e-04 (9.7656e-08) | 2.09 (2) | 2.13 (1) | 1.01 (1) | 1.00 (0) | 1.7445e-05 | 4.8375e-06 | 3.5263e-02 | 1.0905e-02 | 0.1 | 0.1 |
| C44 | OSGS | phase1_tri_k1.json | 1e+06 | 1e-06 | 0.50 | 1 | TRI | 1187.9 | 172 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.3616e-03 (9.7656e-08) | 2.05 (2) | 2.05 (1) | 1.03 (1) | 1.00 (0) | 2.5705e-05 | 6.2036e-06 | 3.6454e-02 | 1.0920e-02 | 0.1 | 0.1 |
| C45 | ASGS | phase1_tri_k1.json | 1e+06 | 1e-06 | 1.00 | 1 | TRI | 1025.3 | 60 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 9.4419e-03 (9.7656e-08) | 1.84 (2) | 1.88 (1) | 1.00 (1) | 1.00 (0) | 1.8594e-05 | 8.9765e-06 | 1.5643e-02 | 1.0907e-02 | 1 | 0.1 |
| C45 | OSGS | phase1_tri_k1.json | 1e+06 | 1e-06 | 1.00 | 1 | TRI | 1985.2 | 219 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.5920e-03 (9.7656e-08) | 1.82 (2) | 1.83 (1) | 1.02 (1) | 1.01 (0) | 4.0831e-05 | 1.7821e-05 | 1.6508e-02 | 1.0947e-02 | 1 | 0.1 |
| C46 | ASGS | corner_direct_solve | 1e+06 | 1e+00 | 0.05 | 1 | TRI | 0.0 | 6 | Yes | optimal ˢ | 3.8081e-12 (1.6954e-08) | 3.03 (2) | 3.04 (1) | 1.02 (1) | 1.02 (0) | 7.9406e-05 | 7.3831e-06 | 7.3338e-02 | 4.5510e-03 | 0 | 0 |
| C46 | OSGS | corner_direct_solve | 1e+06 | 1e+00 | 0.05 | 1 | TRI | 0.0 | 4 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 8.7163e-07 (1.6954e-08) | 3.09 (2) | 3.10 (1) | 1.05 (1) | 1.02 (0) | 7.9411e-05 | 7.3834e-06 | 7.3338e-02 | 4.5510e-03 | 0 | 0 |
| C47 | ASGS | phase1_tri_k1.json | 1e+06 | 1e+00 | 0.50 | 1 | TRI | 393.6 | 38 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 1.6523e-04 (9.7656e-08) | 2.09 (2) | 2.13 (1) | 1.01 (1) | 1.00 (0) | 1.7445e-05 | 4.8390e-06 | 3.5263e-02 | 1.0905e-02 | 0.1 | 0.1 |
| C47 | OSGS | phase1_tri_k1.json | 1e+06 | 1e+00 | 0.50 | 1 | TRI | 1191.7 | 174 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.3616e-03 (9.7656e-08) | 2.05 (2) | 2.05 (1) | 1.03 (1) | 1.00 (0) | 2.5706e-05 | 6.2050e-06 | 3.6454e-02 | 1.0920e-02 | 0.1 | 0.1 |
| C48 | ASGS | phase1_tri_k1.json | 1e+06 | 1e+00 | 1.00 | 1 | TRI | 1162.8 | 62 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 9.4309e-03 (9.7656e-08) | 1.84 (2) | 1.88 (1) | 1.00 (1) | 1.00 (0) | 1.8595e-05 | 8.9766e-06 | 1.5643e-02 | 1.0907e-02 | 1 | 0.1 |
| C48 | OSGS | phase1_tri_k1.json | 1e+06 | 1e+00 | 1.00 | 1 | TRI | 2151.4 | 221 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 6.6097e-03 (9.7656e-08) | 1.82 (2) | 1.83 (1) | 1.02 (1) | 1.01 (0) | 4.0834e-05 | 1.7823e-05 | 1.6508e-02 | 1.0947e-02 | 1 | 0.1 |
| C49 | ASGS | corner_direct_solve | 1e+06 | 1e+06 | 0.05 | 1 | TRI | 0.0 | 6 | Yes | optimal | 4.3199e-11 (1.6954e-08) | 2.11 (2) | 2.50 (1) | 1.00 (1) | 1.00 (0) | 1.2056e-05 | 9.0975e-07 | 7.3177e-02 | 4.5439e-03 | 0 | 0 |
| C49 | OSGS | corner_direct_solve | 1e+06 | 1e+06 | 0.05 | 1 | TRI | 0.0 | 5 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 7.9649e-07 (1.6954e-08) | 2.63 (2) | 2.50 (1) | 1.06 (1) | 1.00 (0) | 1.2266e-05 | 9.1756e-07 | 7.3260e-02 | 4.5444e-03 | 0 | 0 |
| C50 | ASGS | phase1_tri_k1.json | 1e+06 | 1e+06 | 0.50 | 1 | TRI | 390.2 | 39 | Yes | optimal | 3.5710e-06 (9.7656e-08) | 2.10 (2) | 2.05 (1) | 1.01 (1) | 1.00 (0) | 1.8537e-05 | 4.1597e-06 | 3.5397e-02 | 1.0905e-02 | 0.1 | 0.1 |
| C50 | OSGS | phase1_tri_k1.json | 1e+06 | 1e+06 | 0.50 | 1 | TRI | 1157.8 | 146 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 2.1566e-02 (9.7656e-08) | 2.11 (2) | 2.06 (1) | 1.03 (1) | 1.00 (0) | 2.8244e-05 | 4.2548e-06 | 3.8273e-02 | 1.0919e-02 | 0.1 | 0.1 |
| C51 | ASGS | phase1_tri_k1.json | 1e+06 | 1e+06 | 1.00 | 1 | TRI | 860.7 | 45 | <b style='color:red'>No</b> | <b style='color:red'>fold (**)</b> | 6.2591e-04 (9.7656e-08) | 2.02 (2) | 2.01 (1) | 1.01 (1) | 1.00 (0) | 1.2725e-05 | 4.7279e-06 | 1.5818e-02 | 1.0905e-02 | 1 | 1 |
| C51 | OSGS | phase1_tri_k1.json | 1e+06 | 1e+06 | 1.00 | 1 | TRI | 1664.7 | 183 | <b style='color:red'>No</b> | <b style='color:red'>no-root (N/A)</b> | 1.2809e-02 (9.7656e-08) | 1.99 (2) | 1.99 (1) | 1.03 (1) | 1.00 (0) | 2.1950e-05 | 5.5695e-06 | 1.7926e-02 | 1.0925e-02 | 1 | 1 |

## Simulation Configuration Reference Map

<details><summary><b>_consistency_k2_lowmidRe.json (Config IDs Map: C52, C53, C54, C55, C56, C57, C58, C59, C60, C61, C62, C63, C64, C65, C66, C67, C68, C69)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": true,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "phase1_quad_k2_lowmidRe_tight.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": false,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "physical_properties": {
        "Re": [
            1e-06,
            1.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                2
            ],
            "k_pressure": [
                2
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-12,
            "ftol": 1e-12,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5,
            "eps_tol_momentum": 1e-09
        },
        "stabilization": {
            "method": [
                "ASGS",
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k2_lowmidRe_tight"
    }
}
</code></pre>
</details>

<details><summary><b>_surgical_k2_hiRe.json (Config IDs Map: C71, C72, C74, C75, C77, C78)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": true,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "phase1_quad_k2_hiRe_tight.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": false,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "physical_properties": {
        "Re": [
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                2
            ],
            "k_pressure": [
                2
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-12,
            "ftol": 1e-12,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5,
            "eps_tol_momentum": 1e-09
        },
        "stabilization": {
            "method": [
                "ASGS",
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k2_hiRe_tight"
    }
}
</code></pre>
</details>

<details><summary><b>_surgical_k2_tight.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": true,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "phase1_quad_k2_tight.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": false,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "physical_properties": {
        "Re": [
            1e-06,
            1.0
        ],
        "Da": [
            1e-06,
            1.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                2
            ],
            "k_pressure": [
                2
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-12,
            "ftol": 1e-12,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5,
            "eps_tol_momentum": 1e-09
        },
        "stabilization": {
            "method": [
                "ASGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k2_tight"
    }
}
</code></pre>
</details>

<details><summary><b>continuation_c24.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "target": {
        "Re": 1000000.0,
        "Da": 1.0,
        "alpha_0": 0.05
    },
    "continuation": {
        "axis": "alpha",
        "start": 1.0,
        "nsteps": 30,
        "adaptive": true,
        "min_step_ratio": 0.03,
        "max_iters_per_step": 60
    },
    "element_spaces": {
        "k_velocity": 1,
        "k_pressure": 1
    },
    "mesh": {
        "convergence_partitions": [
            80,
            160,
            320
        ],
        "element_type": "QUAD"
    }
}
</code></pre>
</details>

<details><summary><b>continuation_c24_rate.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "target": {
        "Re": 1000000.0,
        "Da": 1.0,
        "alpha_0": 0.05
    },
    "continuation": {
        "nsteps": 32,
        "min_step_ratio": 0.03,
        "max_iters_per_step": 80
    },
    "mesh_continuation": {
        "base_N": 512,
        "fine_Ns": [
            768,
            1024
        ],
        "mesh_max_iters": 120
    },
    "element_spaces": {
        "k_velocity": 1,
        "k_pressure": 1
    },
    "mesh": {
        "element_type": "QUAD"
    }
}
</code></pre>
</details>

<details><summary><b>jfnk_quad_k1.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": true,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "debug_results/jfnk_quad_k1.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                1
            ],
            "k_pressure": [
                1
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320,
                640
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5,
            "osgs_jfnk_enabled": true
        },
        "stabilization": {
            "method": [
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k1"
    }
}
</code></pre>
</details>

<details><summary><b>jfnk_quad_k1_fine.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": false,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "debug_results/jfnk_quad_k1_n160.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                1
            ],
            "k_pressure": [
                1
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320,
                640
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5,
            "osgs_jfnk_enabled": true
        },
        "stabilization": {
            "method": [
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k1"
    }
}
</code></pre>
</details>

<details><summary><b>jfnk_quad_k2.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": true,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "debug_results/jfnk_quad_k2.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                2
            ],
            "k_pressure": [
                2
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5,
            "osgs_jfnk_enabled": true
        },
        "stabilization": {
            "method": [
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k2"
    }
}
</code></pre>
</details>

<details><summary><b>jfnk_quad_k2_fine.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": false,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "debug_results/jfnk_quad_k2_n160.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                2
            ],
            "k_pressure": [
                2
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5,
            "osgs_jfnk_enabled": true
        },
        "stabilization": {
            "method": [
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k2"
    }
}
</code></pre>
</details>

<details><summary><b>jfnk_quad_k2_off.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": false,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "debug_results/jfnk_quad_k2_off.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                2
            ],
            "k_pressure": [
                2
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5,
            "osgs_jfnk_enabled": false
        },
        "stabilization": {
            "method": [
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k2"
    }
}
</code></pre>
</details>

<details><summary><b>phase1_quad_k1.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": false,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "phase1_quad_k1.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                1
            ],
            "k_pressure": [
                1
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320,
                640
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5
        },
        "stabilization": {
            "method": [
                "ASGS",
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k1"
    }
}
</code></pre>
</details>

<details><summary><b>phase1_quad_k2.json (Config IDs Map: C52, C53, C54, C55, C56, C57, C58, C59, C60, C61, C62, C63, C64, C65, C66, C67, C68, C69, C71, C72, C74, C75, C77, C78)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": false,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "phase1_quad_k2.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                2
            ],
            "k_pressure": [
                2
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5
        },
        "stabilization": {
            "method": [
                "ASGS",
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_quad_k2"
    }
}
</code></pre>
</details>

<details><summary><b>phase1_tri_k1.json (Config IDs Map: C25, C26, C27, C28, C29, C30, C31, C32, C33, C34, C35, C36, C37, C38, C39, C40, C41, C42, C44, C45, C47, C48, C50, C51)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": false,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "phase1_tri_k1.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                1
            ],
            "k_pressure": [
                1
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "TRI"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001,
            "newton_residual_divergence_patience": 2,
            "pingpong_enabled": true,
            "pingpong_max_swaps": 4,
            "pingpong_picard_gain_orders": 1.5
        },
        "stabilization": {
            "method": [
                "ASGS",
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "phase1_tri_k1"
    }
}
</code></pre>
</details>

<details><summary><b>test_config.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "mms_dynamic_budget": {
        "newton_re_iterations": 150
    },
    "erase_past_results": false,
    "encoding_strategy": "minmax",
    "solver_stall_window": 20,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "mms_sweep.h5",
    "epsilon_pert": [
        1.0
    ],
    "max_n_pert": 1,
    "mms_verification_enabled": true,
    "mms_tau_err": 0.0001,
    "mms_eps_u_l2": 1e-12,
    "mms_eps_u_h1": 1e-12,
    "mms_eps_p_l2": 1e-12,
    "mms_max_extra_cycles": 5,
    "mms_require_consecutive_passes": 2,
    "mms_rate_check_factor": 100.0,
    "skip_cells": [
        [
            1000000.0,
            1e-06,
            0.05
        ],
        [
            1000000.0,
            1.0,
            0.05
        ],
        [
            1000000.0,
            1000000.0,
            0.05
        ]
    ],
    "physical_properties": {
        "Re": [
            1e-06,
            1.0,
            1000000.0
        ],
        "Da": [
            1e-06,
            1.0,
            1000000.0
        ],
        "physical_epsilon": [
            0.0
        ],
        "numerical_epsilon_coefficient": [
            0.0001
        ],
        "f_x": 0.0,
        "f_y": 0.0
    },
    "domain": {
        "alpha_0": [
            1.0,
            0.5,
            0.05
        ],
        "r_1": 0.2,
        "r_2": 0.4,
        "bounding_box": [
            -0.5,
            0.5,
            -0.5,
            0.5
        ]
    },
    "numerical_method": {
        "element_spaces": {
            "k_velocity": [
                1,
                2
            ],
            "k_pressure": [
                1,
                2
            ],
            "equal_order_only": true
        },
        "mesh": {
            "partition": [
                10,
                10
            ],
            "convergence_partitions": [
                10,
                20,
                40,
                80,
                160,
                320
            ],
            "element_type": [
                "QUAD",
                "TRI"
            ]
        },
        "solver": {
            "picard_iterations": 2,
            "newton_iterations": 10,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
            "armijo_c1": 0.0001
        },
        "stabilization": {
            "method": [
                "ASGS",
                "OSGS"
            ]
        }
    },
    "output": {
        "directory": "test/extended/ManufacturedSolutions/results",
        "basename": "mms_sweep"
    }
}
</code></pre>
</details>
