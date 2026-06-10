# Merged MMS convergence table (Phase-1 sweep + Phase-2 continuation)

Marks: *(none)*=verified (true root + optimal slope, k+1 / k); `*`=true root recovered by continuation at the noted finer mesh, slope then optimal; `‡`=**converges but SUB-OPTIMAL rate** (true root exists; slope stays below k+1/k even at the finest mesh — a genuine finding, not a fold); `**`=best true root attained, asymptotics not established; `N/A`=no true root anywhere.

rate_* = slope on the finest available mesh PAIR; lsq_* = least-squares slope over the finest ≤3 true-root meshes.

Rate caveat: a velocity-L² rate marked `ˢ` is ABOVE the nominal k+1 (super-convergent). This is not a failure (acceptance is one-sided, slope ≥ target − tol), but a rate well above nominal can signal the asymptotic regime is not yet established — confirm with one further refinement.

**eps_finest / eps_min** columns report the per-cell robustness fingerprint: the largest eps_pert in {1.0, 0.1, ..., 0} that the homotopy outer loop accepted at the FINEST mesh (eps_finest) and the SMALLEST such value across the mesh ladder (eps_min). 1 means the solver absorbed a full ‖u_ex‖-scale perturbation of the initial guess; 0 means only the interpolated-u_ex fallback converged (pre-asymptotic / noise-floor limited).

| Config | Re | Da | α₀ | k | Elem | Method | Status | N_rep | L2_u | L2_p | H1_u | rate_L2u | rate_H1u | lsq_L2u | lsq_H1u | eps_finest | eps_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 1e-06 | 1e-06 | 0.05 | 1 | QUAD | ASGS | verified | 320 | 8.554e-04 | 2.536e-02 | 1.364e-01 | 1.98 | 1.02 | 1.95 | 1.03 | 1 | 1 |
| C1 | 1e-06 | 1e-06 | 0.05 | 1 | QUAD | OSGS | unverified(fold) | 20 | 5.524e-02** | 6.849e-01 | 2.052e+00 | --- | --- | --- | --- | 1 | -1 |
| C2 | 1e-06 | 1e-06 | 0.50 | 1 | QUAD | ASGS | verified | 320 | 6.475e-05 | 2.897e-03 | 2.619e-02 | 2.00 | 1.00 | 1.99 | 1.01 | 1 | 1 |
| C2 | 1e-06 | 1e-06 | 0.50 | 1 | QUAD | OSGS | unverified(fold) | 20 | 5.877e-03** | 1.849e-01 | 3.897e-01 | --- | --- | 1.99 | 0.99 | 1 | 1 |
| C3 | 1e-06 | 1e-06 | 1.00 | 1 | QUAD | ASGS | verified | 320 | 3.756e-05 | 1.158e-03 | 8.934e-03 | 2.00 | 1.00 | 2.00 | 1.00 | 1 | 1 |
| C3 | 1e-06 | 1e-06 | 1.00 | 1 | QUAD | OSGS | unverified(fold) | 40 | 4.692e-04** | 1.661e-02 | 7.147e-02 | --- | --- | 2.02 | 1.01 | 1 | 1 |
| C4 | 1e-06 | 1e+00 | 0.05 | 1 | QUAD | ASGS | verified | 320 | 6.058e-04 | 2.470e-02 | 1.363e-01 | 1.96 | 1.01 | 1.92 | 1.02 | 1 | 1 |
| C4 | 1e-06 | 1e+00 | 0.05 | 1 | QUAD | OSGS | unverified(fold) | 20 | 5.371e-02** | 3.491e-01 | 2.054e+00 | --- | --- | --- | --- | 1 | -1 |
| C5 | 1e-06 | 1e+00 | 0.50 | 1 | QUAD | ASGS | verified | 320 | 6.522e-05 | 2.400e-03 | 2.619e-02 | 1.99 | 1.00 | 1.98 | 1.01 | 1 | 1 |
| C5 | 1e-06 | 1e+00 | 0.50 | 1 | QUAD | OSGS | unverified(fold) | 10 | 2.330e-02** | 1.609e-01 | 7.754e-01 | --- | --- | --- | --- | 1 | 1 |
| C6 | 1e-06 | 1e+00 | 1.00 | 1 | QUAD | ASGS | verified | 320 | 3.760e-05 | 7.479e-04 | 8.928e-03 | 2.00 | 1.00 | 2.00 | 1.00 | 1 | 1 |
| C6 | 1e-06 | 1e+00 | 1.00 | 1 | QUAD | OSGS | unverified(fold) | 40 | 4.686e-04** | 8.408e-03 | 7.147e-02 | --- | --- | 2.02 | 1.01 | 1 | 1 |
| C7 | 1e-06 | 1e+06 | 0.05 | 1 | QUAD | ASGS | verified | 320 | 4.217e-04 | 6.985e-06 | 1.377e-01 | 1.98 | 1.02 | 1.95 | 1.04 | 1 | 1 |
| C7 | 1e-06 | 1e+06 | 0.05 | 1 | QUAD | OSGS | unverified(fold) | 10 | 6.026e-02** | 6.474e-03 | 2.664e+00 | --- | --- | --- | --- | 1 | 1 |
| C8 | 1e-06 | 1e+06 | 0.50 | 1 | QUAD | ASGS | verified | 320 | 8.879e-05 | 3.918e-06 | 2.625e-02 | 2.04 | 1.01 | 2.03 | 1.01 | 1 | 1 |
| C8 | 1e-06 | 1e+06 | 0.50 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 1 | 1 |
| C9 | 1e-06 | 1e+06 | 1.00 | 1 | QUAD | ASGS | verified | 320 | 4.002e-05 | 3.313e-06 | 8.988e-03 | 2.05 | 1.01 | 2.04 | 1.01 | 1 | 1 |
| C9 | 1e-06 | 1e+06 | 1.00 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 1 | 1 |
| C10 | 1e+00 | 1e-06 | 0.05 | 1 | QUAD | ASGS | verified | 320 | 6.626e-04 | 2.401e-02 | 1.363e-01 | 1.93 | 1.01 | 1.90 | 1.02 | 1 | 1 |
| C10 | 1e+00 | 1e-06 | 0.05 | 1 | QUAD | OSGS | unverified(fold) | 20 | 5.784e-02** | 3.543e-01 | 2.064e+00 | --- | --- | --- | --- | 1 | -1 |
| C11 | 1e+00 | 1e-06 | 0.50 | 1 | QUAD | ASGS | verified | 320 | 6.511e-05 | 2.392e-03 | 2.619e-02 | 1.99 | 1.00 | 1.98 | 1.01 | 1 | 1 |
| C11 | 1e+00 | 1e-06 | 0.50 | 1 | QUAD | OSGS | unverified(fold) | 10 | 2.342e-02** | 1.626e-01 | 7.755e-01 | --- | --- | --- | --- | 1 | 1 |
| C12 | 1e+00 | 1e-06 | 1.00 | 1 | QUAD | ASGS | verified | 320 | 3.756e-05 | 7.464e-04 | 8.928e-03 | 2.00 | 1.00 | 1.99 | 1.00 | 1 | 1 |
| C12 | 1e+00 | 1e-06 | 1.00 | 1 | QUAD | OSGS | unverified(fold) | 40 | 4.696e-04** | 8.472e-03 | 7.147e-02 | --- | --- | 2.02 | 1.01 | 1 | 1 |
| C13 | 1e+00 | 1e+00 | 0.05 | 1 | QUAD | ASGS | verified | 320 | 6.049e-04 | 1.647e-02 | 1.363e-01 | 1.96 | 1.01 | 1.92 | 1.02 | 1 | 1 |
| C13 | 1e+00 | 1e+00 | 0.05 | 1 | QUAD | OSGS | unverified(fold) | 20 | 5.385e-02** | 2.324e-01 | 2.053e+00 | --- | --- | --- | --- | 1 | -1 |
| C14 | 1e+00 | 1e+00 | 0.50 | 1 | QUAD | ASGS | verified | 320 | 6.517e-05 | 1.599e-03 | 2.619e-02 | 1.99 | 1.00 | 1.98 | 1.01 | 1 | 1 |
| C14 | 1e+00 | 1e+00 | 0.50 | 1 | QUAD | OSGS | unverified(fold) | 10 | 2.330e-02** | 1.077e-01 | 7.753e-01 | --- | --- | --- | --- | 1 | 1 |
| C15 | 1e+00 | 1e+00 | 1.00 | 1 | QUAD | ASGS | verified | 320 | 3.757e-05 | 4.982e-04 | 8.928e-03 | 2.00 | 1.00 | 1.99 | 1.00 | 1 | 1 |
| C15 | 1e+00 | 1e+00 | 1.00 | 1 | QUAD | OSGS | unverified(fold) | 40 | 4.686e-04** | 5.638e-03 | 7.147e-02 | --- | --- | 2.02 | 1.01 | 1 | 1 |
| C16 | 1e+00 | 1e+06 | 0.05 | 1 | QUAD | ASGS | verified | 320 | 4.214e-04 | 6.985e-06 | 1.376e-01 | 1.98 | 1.02 | 1.95 | 1.04 | 1 | 1 |
| C16 | 1e+00 | 1e+06 | 0.05 | 1 | QUAD | OSGS | unverified(fold) | 10 | 6.026e-02** | 6.474e-03 | 2.664e+00 | --- | --- | --- | --- | 1 | 1 |
| C17 | 1e+00 | 1e+06 | 0.50 | 1 | QUAD | ASGS | verified | 320 | 8.871e-05 | 3.918e-06 | 2.625e-02 | 2.04 | 1.01 | 2.03 | 1.01 | 1 | 1 |
| C17 | 1e+00 | 1e+06 | 0.50 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 1 | 1 |
| C18 | 1e+00 | 1e+06 | 1.00 | 1 | QUAD | ASGS | verified | 320 | 3.998e-05 | 3.312e-06 | 8.987e-03 | 2.04 | 1.01 | 2.04 | 1.01 | 1 | 1 |
| C18 | 1e+00 | 1e+06 | 1.00 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 1 | 1 |
| C19 | 1e+06 | 1e-06 | 0.50 | 1 | QUAD | ASGS | unverified(fold) | 80 | 2.345e-04** | 6.696e-05 | 1.060e-01 | --- | --- | 2.23 | 0.98 | 0.1 | 0.1 |
| C19 | 1e+06 | 1e-06 | 0.50 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 0.1 | 0.1 |
| C20 | 1e+06 | 1e-06 | 1.00 | 1 | QUAD | ASGS | unverified(fold) | 20 | 2.070e-03** | 1.318e-03 | 1.478e-01 | --- | --- | --- | --- | 0.1 | 0.1 |
| C20 | 1e+06 | 1e-06 | 1.00 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 0.1 | 0.1 |
| C21 | 1e+06 | 1e+00 | 0.50 | 1 | QUAD | ASGS | unverified(fold) | 80 | 2.346e-04** | 6.699e-05 | 1.060e-01 | --- | --- | 2.23 | 0.98 | 0.1 | 0.1 |
| C21 | 1e+06 | 1e+00 | 0.50 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 0.1 | 0.1 |
| C22 | 1e+06 | 1e+00 | 1.00 | 1 | QUAD | ASGS | unverified(fold) | 20 | 2.070e-03** | 1.318e-03 | 1.478e-01 | --- | --- | --- | --- | 0.1 | 0.1 |
| C22 | 1e+06 | 1e+00 | 1.00 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 0.1 | 0.1 |
| C23 | 1e+06 | 1e+06 | 0.50 | 1 | QUAD | ASGS | verified | 320 | 1.265e-05 | 2.671e-06 | 2.627e-02 | 2.10 | 1.01 | 2.09 | 1.01 | 1 | 1 |
| C23 | 1e+06 | 1e+06 | 0.50 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 1 | 1 |
| C24 | 1e+06 | 1e+06 | 1.00 | 1 | QUAD | ASGS | unverified(fold) | 640 | 1.790e-06** | 8.006e-07 | 4.514e-03 | --- | --- | 2.05 | 1.01 | 0.1 | 0.1 |
| C24 | 1e+06 | 1e+06 | 1.00 | 1 | QUAD | OSGS | no-root | 320 | N/A | --- | --- | --- | --- | --- | --- | 0.1 | 0.1 |
