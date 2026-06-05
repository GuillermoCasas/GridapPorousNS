# Normalization / scaling audit of the solver gates (P5, 2026-06-04)

The user's brief included "look for normalization mistakes in the various checks." This audit classifies
every convergence / drift / divergence / plateau gate in the nonlinear solver by **whether both sides of
its inequality carry the same units** (dimensionless ratio = encoding-invariant) or one side is an
**absolute** number compared to a scale-dependent quantity (the bug class fixed for the inner gate on
2026-06-02). Verdict legend: **OK** = dimensionally sound; **SUSPECT** = absolute-vs-scaled, but see notes;
**BUG** = an unambiguous dimensional error to fix.

Convention reminders for the MMS encoding sweep (`run_test.jl`): velocity residual/iterate `∝ U`, pressure
`∝ P_c ∝ U²`, so a gate that compares a residual or drift to an **absolute** constant is encoding-dependent.

| # | Gate | Code | Inequality | LHS / RHS units | Verdict |
|---|------|------|-----------|-----------------|---------|
| 1 | Inner per-field ftol | `_residual_meets_per_field_ftol` + `effective_ftol_per_field` (`nonlinear.jl`) | `‖R_k‖∞ ≤ max(ftol, rel_k·‖R₀_k‖∞)` | both ∝ residual (per field) | **OK** |
| 2 | Inner per-field noise floor | `effective_noise_floor_per_field` + `_residual_meets_per_field_noise_floor` | `‖R_k‖∞ ≤ max(nf, rel_nf_k·‖R₀_k‖∞)` | both ∝ residual | **OK** |
| 3 | Honest-exit gate | `noise_floor_success_max_ftol_multiple` + `_residual_meets_per_field_honest_exit_gate` | `‖R_k‖∞ ≤ k_nf·effective_ftol_k` | both ∝ residual (k_nf dimensionless) | **OK** |
| 4 | Divergence safeguard | `eval_safeguard_termination_bounds!` | Newton: `Φ_new > Φ_old·f`; Picard: `‖b‖∞,new > ‖b‖∞,old·f` | ratio → dimensionless | **OK** |
| 5 | OSGS outer state-drift | `_compute_state_drift` + `_decide_osgs_convergence` | `x_diff ≤ max(osgs_tol, stagnation_tol)` | LHS ∝ U (L²-mass) or mixed (ℓ∞); RHS absolute | **SUSPECT** (inert — see below) |
| 6 | OSGS projection-drift | `pi_u_drift`/`pi_p_drift` in `_update_and_project!` + `_decide_osgs_convergence` | `max(π_u_drift, π_p_drift) ≤ max(osgs_proj_tol, stagnation_tol)` | LHS ∝ mass-weighted Δπ; RHS absolute | **SUSPECT** (inert — see below) |
| 7 | MMS plateau ratios | `_run_*_mms_extension!` / `_run_osgs_relaxation!` | `\|E_k−E_{k-1}\| / max(E_k,E_{k-1},ε·h^p) < τ_err` | ratio → dimensionless | **OK** |

## Gate-by-gate notes

**#1 Inner per-field ftol — OK (the 2026-06-02 fix).** `effective_ftol_per_field[k] = max(ftol,
relative_ftol_per_field[k]·solution_scale_per_field[k])` with `solution_scale_per_field = copy(norm_b_per_field)`
= the **frozen initial residual** `‖R₀_k‖∞`. So the working target is the dimensionless relative reduction
`‖R_k‖∞ ≤ rel_k·‖R₀_k‖∞` — both sides per-field residual, encoding-invariant. Confirmed still `‖R₀‖`-based
(not `‖x‖`). **The old `‖x‖`-fallback helper `_resolve_solution_scale_per_field` (Bernoulli/total-head zero
proxy) was DEAD and has been removed in P5** — it only made sense for a solution-magnitude-scaled gate,
which no longer exists. Guarded by `test/quick/encoding_invariance_quick_test.jl`.

**#2 Noise-floor gate — OK.** Same `_initialize_effective_thresholds` family as #1, scaled by `‖R₀_k‖`.

**#3 Honest-exit gate — OK.** `‖R_k‖∞ ≤ k_nf·effective_ftol_k`; both sides residual-scaled, `k_nf`
(`noise_floor_success_max_ftol_multiple`) dimensionless.

**#4 Divergence safeguard — OK; do not regress the Picard branch.** In `:newton` mode the test is a merit
*ratio* `Φ_new/Φ_old > divergence_merit_factor`; in `:picard` mode it is a residual-inf-norm *ratio*. Both
dimensionless. The mode split is load-bearing: a Φ-based test in Picard mode caused spurious
`merit_divergence_escaped` exits (see `test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md`).
Confirmed the `:picard` branch is taken in `:picard` mode.

**#5 / #6 OSGS outer state-drift & projection-drift — SUSPECT but empirically inert; NOT changed in P5.**
These are the prime dimensional suspects: `x_diff` (either the L²-mass functional `√∫(e_u·e_u)dΩ`, which
scales `∝ U`, or the raw `ℓ∞` DOF norm, which mixes the velocity/pressure scales) and the mass-weighted
projection drift are compared to **absolute** tolerances `max(osgs_tol, stagnation_tol)` /
`max(osgs_projection_tolerance, stagnation_tol)`. Dimensionally this is the same class as the inner-gate bug.
**However:**
- The 2026-06-02 ledger records that making the outer state-drift gate *relative* "changed NOTHING" for
  encoding covariance, because in the tested regime the outer loop **runs its full iteration budget**
  rather than early-exiting on this gate — so the gate only governs an early-exit that does not occur, and
  changing it moves no final error.
- The open OSGS rate-stagnation defect is in the **discrete staggered map** `Π(R(U(π)))`, **not** this gate
  (established 2026-06-02 + the 2026-06-04 complete-sweep analysis in `docs/known_issues.md`). So fixing the
  gate cannot help the rate.
- Therefore a relative rewrite has **zero demonstrated benefit** and a real **error-movement risk** (it
  changes *which* outer iteration the loop declares converged, hence the converged OSGS fixed point on any
  cell that does early-exit). Per the P5 rule ("only fix gates that are unambiguously dimensional bugs AND
  can be fixed without moving any final error"), it is **deferred, not applied.**

  **Recommended relative form (when a demonstrated need + full A/B justify it):** normalize the state drift
  by the iterate magnitude, `x_diff / max(‖U_h‖, floor) ≤ rel_drift`, or by the first-iteration drift
  `x_diff^{(m)} / x_diff^{(1)} ≤ rel_drift`; analogously normalize the projection drift by `‖π_h‖`. Acceptance:
  `encoding_invariance_quick_test.jl` stays green AND every final `err_*` is unchanged on the full k=1 A/B;
  if any error moves, that is a convergence-behaviour change (back out and document), not a normalization fix.

**#7 MMS plateau ratios — OK.** `r = |E_k − E_{k-1}| / max(E_k, E_{k-1}, ε·h^p)` is a dimensionless
relative change; the `h`-scaled `ε` only sets the denominator's noise floor (the §5.1 fix). Compared to the
dimensionless `tau_err`.

## Summary

Five of seven gates (1–4, 7) are dimensionally sound. The two OSGS outer drift gates (5, 6) are
dimensionally suspect (absolute tolerance vs scale-dependent drift) but **empirically inert** for covariance
in the tested regime and **not** the cause of the open OSGS rate defect — so they are documented with a
recommended relative form and deferred rather than changed, to avoid moving converged solutions for no
demonstrated benefit. The only P5 code change is the removal of the dead `_resolve_solution_scale_per_field`
helper and its unused buffer.
