# Normalization / scaling audit of the solver gates (P5, 2026-06-04)

> **STATUS — SUPERSEDED for production (gates #1–#3).** The inner per-field ftol (#1), noise floor (#2),
> and honest-exit (#3) gates analyzed below are NO LONGER the authoritative inner convergence gate in
> production. The solver now stops on a **scale-free `ε_M`/`ε_C` criterion**:
> converged ⇔ `ε_M ≤ tol_M` **and** `ε_C ≤ tol_C`, where `ε_M = ‖r_M‖/D_M` (momentum residual over the
> dynamic force-magnitude envelope) and `ε_C = ‖r_C‖/D_C` — the Route-B "Philosophy-A" algebraic mass
> gate, symmetric with `ε_M` (see [`docs/mms/route-b-2d-sweep-status.md`](../mms/route-b-2d-sweep-status.md);
> the earlier strong-form `‖ε p + ∇·(α u) − g‖ / (‖∇(α u)‖ + ‖g‖)` is now the diagnostic `eps_C_strong`,
> not the gate). It is implemented in
> [`src/solvers/convergence_criterion.jl`](../../src/solvers/convergence_criterion.jl) and specified in
> [`docs/solver/nonlinear-convergence-criterion-prompt.md`](nonlinear-convergence-criterion-prompt.md).
> It is injected by `solve_system` (`src/solvers/solver_core.jl`) as `conv_probe` and is the authoritative
> success gate in `_safe_solve_inner!` (`src/solvers/nonlinear.jl`, guarded by `scale_free = conv_probe !== nothing`).
> **Post-`d1fac8e`:** the per-field RELATIVE re-anchoring (gate #1's `rel_k·‖R₀_k‖`) was removed; the fallback gate #1 is now the uniform scalar `ftol` (the relative formulas below are historical).
>
> Gates #1–#3 below survive ONLY as the `conv_probe === nothing` fallback (Cocquet unstabilized-Galerkin
> runs + kernel unit tests, which use the scalar ftol) and to feed the `f_norm` trace diagnostic. The
> per-field-ftol-vs-`‖R₀‖` dimensional analysis below is retained as historical record of that fallback
> path; it is sound on its own terms but is not the production gate.

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
| 1 | Inner per-field ftol *(fallback-only; superseded by ε_M/ε_C)* | `_residual_meets_per_field_ftol` + `effective_ftol_per_field` (`nonlinear.jl`) | `‖R_k‖∞ ≤ max(ftol, rel_k·‖R₀_k‖∞)` | both ∝ residual (per field) | **OK** |
| 2 | Inner per-field noise floor *(fallback-only; superseded)* | `effective_noise_floor_per_field` + `_residual_meets_per_field_noise_floor` | `‖R_k‖∞ ≤ max(nf, rel_nf_k·‖R₀_k‖∞)` | both ∝ residual | **OK** |
| 3 | Honest-exit gate *(fallback-only; superseded)* | `noise_floor_success_max_ftol_multiple` + `_residual_meets_per_field_honest_exit_gate` | `‖R_k‖∞ ≤ k_nf·effective_ftol_k` | both ∝ residual (k_nf dimensionless) | **OK** |
| 4 | Divergence safeguard | `eval_safeguard_termination_bounds!` | Newton: `Φ_new > Φ_old·f`; Picard: `‖b‖∞,new > ‖b‖∞,old·f` | ratio → dimensionless | **OK** |
| 5 | ~~OSGS outer state-drift~~ | ~~`_compute_state_drift` + `_decide_osgs_convergence`~~ | — | — | **REMOVED** (gate deleted with the staggered loop, 2026-06-07 leaning; no longer applicable) |
| 6 | ~~OSGS projection-drift~~ | ~~`pi_u_drift`/`pi_p_drift` in `_update_and_project!` + `_decide_osgs_convergence`~~ | — | — | **REMOVED** (gate deleted with the staggered loop, 2026-06-07 leaning; no longer applicable) |
| 7 | MMS plateau ratios | `_run_*_mms_extension!` / `_run_osgs_relaxation!` *(dead symbol — now `solve_osgs_stage!`)* | `\|E_k−E_{k-1}\| / max(E_k,E_{k-1},ε·h^p) < τ_err` | ratio → dimensionless | **OK** |

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

**#5 / #6 OSGS outer state-drift & projection-drift — REMOVED with the staggered loop (2026-06-07 leaning).**
Both gates, and the helpers that computed them (`_compute_state_drift`, `_update_and_project!`,
`_decide_osgs_convergence`), were **deleted** when the OSGS solver was leaned to a single "coupled" mode (one
Newton solve that re-projects `π = Π(R(u))` at every residual evaluation). There is no longer an outer
relaxation loop, so neither the state-drift nor the projection-drift early-exit gate exists. The earlier
dimensional analysis (absolute tolerance vs scale-dependent drift) and the recommended relative-form rewrite
are **moot** and have been removed along with the gates.

**#7 MMS plateau ratios — OK.** `r = |E_k − E_{k-1}| / max(E_k, E_{k-1}, ε·h^p)` is a dimensionless
relative change; the `h`-scaled `ε` only sets the denominator's noise floor (the §5.1 fix). Compared to the
dimensionless `tau_err`.

## Summary

All five surviving gates (1–4, 7) are dimensionally sound. **Note (production gate):** gates #1–#3 are
no longer the authoritative inner stopping criterion — production uses the scale-free `ε_M`/`ε_C` gate
(`src/solvers/convergence_criterion.jl`; spec `docs/solver/nonlinear-convergence-criterion-prompt.md`), and
#1–#3 now run only on the `conv_probe === nothing` fallback (Cocquet unstabilized-Galerkin + kernel tests).
See the status banner at the top. The two OSGS outer drift gates (5, 6) — which
were dimensionally suspect (absolute tolerance vs scale-dependent drift) but empirically inert for covariance
— no longer exist: they were deleted with the entire staggered outer loop in the 2026-06-07 leaning to the
single coupled OSGS mode. The only P5 code change was the removal of the dead
`_resolve_solution_scale_per_field` helper and its unused buffer.
