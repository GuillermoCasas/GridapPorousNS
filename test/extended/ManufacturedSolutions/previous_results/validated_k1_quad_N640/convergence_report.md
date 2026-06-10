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

## Simulation Configuration Reference Map

<details><summary><b>_run_quad_k1_postrefactor.json (Config IDs Map: C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24)</b></summary>
<pre><code class="json">
{
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
            "dynamic_newton_re_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "dynamic_ftol_ceiling": 0.0001,
            "dynamic_ftol_spatial_safety_factor": 0.01,
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

<details><summary><b>_scratch_step4.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "erase_past_results": false,
    "write_vtk": false,
    "trace_convergence_norms": true,
    "encoding_strategy": "minmax",
    "solver_stall_window": 2,
    "solver_stall_min_rel_improvement": 0.01,
    "h5_filename": "debug_results/step4_test.h5",
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
    "skip_cells": [],
    "physical_properties": {
        "Re": [
            1.0,
            1000000.0
        ],
        "Da": [
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
                20
            ],
            "element_type": [
                "QUAD"
            ]
        },
        "solver": {
            "picard_iterations": 15,
            "newton_iterations": 150,
            "dynamic_newton_re_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "dynamic_ftol_ceiling": 0.0001,
            "dynamic_ftol_spatial_safety_factor": 0.01,
            "stagnation_noise_floor": 1e-05,
            "linesearch_alpha_min": 0.0001,
            "max_linesearch_iterations": 25,
            "linesearch_contraction_factor": 0.5,
            "max_increases": 3,
            "divergence_merit_factor": 1.05,
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
        "basename": "step4_test"
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

<details><summary><b>phase1_quad_k1.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "erase_past_results": true,
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
            "dynamic_newton_re_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "dynamic_ftol_ceiling": 0.0001,
            "dynamic_ftol_spatial_safety_factor": 0.01,
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

<details><summary><b>phase1_quad_k2.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
    "erase_past_results": false,
    "encoding_strategy": "minmax",
    "solver_stall_window": 20,
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
            "picard_iterations": 2,
            "newton_iterations": 10,
            "dynamic_newton_re_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "dynamic_ftol_ceiling": 0.0001,
            "dynamic_ftol_spatial_safety_factor": 0.01,
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
        "basename": "phase1_quad_k2"
    }
}
</code></pre>
</details>

<details><summary><b>test_config.json (Config IDs Map: N/A)</b></summary>
<pre><code class="json">
{
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
            "dynamic_newton_re_iterations": 150,
            "noise_floor_success_max_ftol_multiple": 10.0,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "dynamic_ftol_ceiling": 0.0001,
            "dynamic_ftol_spatial_safety_factor": 0.01,
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
