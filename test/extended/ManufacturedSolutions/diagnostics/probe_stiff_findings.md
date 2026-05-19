# probe_stiff_diagnose.jl — findings (2026-05-19)

Run via `cd test/extended/ManufacturedSolutions && julia --project=../../.. probe_stiff_diagnose.jl`.
Mirrors the FE setup of `run_test.jl:368-516` for the cell deferred Phase 6 §2.1 was
investigating: `(Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS, eps_pert=0)`.

Companion to [theory/algorithm-improvement-progress.md](../../../../theory/algorithm-improvement-progress.md)
§"Phase 6 §2.1 — DEFERRED", which named three live hypotheses (H2 / H3 / H4) and three
unrun probes (C / D / E).

Raw probe output (auto-dumped each script run) lives in `probe_stiff_raw.md`; this file
is the interpretive write-up.

---

## TL;DR — Phase 6 §2.1 deferral overturned; failing cell now converges

The cell the previous session ruled "irrecoverable by continuation" **converges
cleanly** after two small solver fixes:

1. **Fix 1 (landed).** Picard mode's divergence safeguard now compares `‖R‖_∞` growth
   instead of Φ growth (which is a Newton-mode metric that fluctuates spuriously
   across Picard iterations as the Jacobian-diagonal weights shift).
2. **Fix 2 (landed).** Added `dynamic_newton_re_threshold` / `dynamic_newton_re_iterations`
   config fields mirroring the existing `dynamic_picard_re_*` pattern. At Re ≥ 10⁴ the
   Newton iteration budget auto-expands to 60 (default), enough to span both the
   stiff-regime smoothing-out phase and the quadratic-convergence tail.

After both fixes, the cell `(Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS, eps_pert=0)`
converges in the first homotopy attempt with `‖R‖_∞ → 5.47e-4` at iter 60 of the second
Newton pass, exiting cleanly via `stagnation_noise_floor_reached`. L2 u/p errors
0.127 / 0.025 (the rate-validity of these errors against MMS is a separate Phase 5
question; convergence vs. failure is the bar this session was investigating).

---

## H2/H3/H4 verdict

| Hypothesis | Status | Evidence |
|---|---|---|
| **H2 — Jacobian ill-conditioning at the asymptotic corner** | **RULED OUT** | Probe D: `cond(J) = 2.72e7` at u_ex on the stiff cell. The easy cell at the same α₀=0.05 actually shows `cond = 2.25e18` because of P_c scaling — cond alone is not a useful sanity criterion in this codebase. The stiff cell is **11 orders better-conditioned** than the easy cell. |
| **H3 — no discrete root nearby** | **RULED OUT** | Probe C (post-Fix-1) and the full cascade both reach the root: ‖R‖_∞ goes from 0.215 to 5.47e-4 in finite iterations. |
| **H4 — τ-stabilisation breakdown** | **PARTIAL / SECONDARY** | Probe E reveals isolated spikes at `s=-0.15` and `s=+0.25` where `‖R(x0 - s·δ)‖_∞` jumps 2-10× vs neighbours — compatible with `SmoothVelocityFloor` / τ-regularisation thresholds being crossed along the Newton line. **Not the deep H4 the original framing predicted** (pure-convective basis loss): the underlying landscape is smooth between spikes, and Newton converges through them with enough iteration budget. |

**Actual root cause: two compounding solver-safeguard bugs.** Detailed below.

---

## Probe results (initial pre-fix run on the stiff cell)

### Easy-cell sanity check (`Re=1e-6, Da=1e6, α=0.05, k=1, n=10, QUAD, ASGS`)

| Probe | Outcome |
|---|---|
| D | `‖R(u_ex)‖_∞ = 2.18e8` (large — P_c ~ 1e12 scaling), `cond(J) = 2.25e18`. |
| C | Picard converges in **3 iters at α=1**, `ftol_reached`, final `‖R‖ = 3.05e-5`. Drops 13 orders of magnitude. |
| E | Clean V-shape, `s* = 1.0` exactly. ‖R‖ at s=0 is 2.18e8; at s=1 is 1.10e2. **6-order reduction at full Newton step.** |

Harness validated. The easy cell behaves textbook.

### Stiff failing cell (`Re=1e6, Da=1, α=0.05, k=1, n=10, QUAD, ASGS`) — pre-fix run

#### Probe D — Jacobian condition number at u_ex

| Metric | Value |
|---|---|
| ‖R(u_ex)‖_∞ | 2.15e-01 |
| ‖R(u_ex)‖_2 | 3.30e-01 |
| n_dofs | 283 |
| cond(J) | **2.72e+07** |
| σ_max(J) | 1.54e+01 |
| σ_min(J) | 5.67e-07 |

Well-conditioned. The "asymptotic corner Jacobian ill-conditioning" framing of the
original investigation was wrong.

#### Probe C — Picard from u_ex (pre-Fix-1)

```
Iter 0: ‖R‖_∞ = 0.215   Φ = 0.0545
Iter 1: ‖R‖_∞ = 0.131   Φ = 10.30   (α=1, line search accepted on ‖R‖)
Iter 2: ‖R‖_∞ = 0.100   Φ = 10.59   ← divergence_merit_factor triggered (Φ grew 200×)
                                        Picard exits: merit_divergence_escaped
```

‖R‖_∞ reduction per iter ≈ 0.62 (factor 1.6× per iter). Φ growth is spurious —
between iterations the Picard Jacobian's diagonal differs from Newton's, so the
merit weights `w = diag(J)` shift, and `Φ = ½‖b/w‖²` is non-stationary.

#### Probe E — Newton-step line probe at u_ex (full landscape)

Newton step magnitude `‖δ‖_2 = 14.06`. Optimal `s* = 0.50`, ‖R‖_∞(s*) = 0.181 (16%
reduction). Compared to ‖R‖_∞(s=0) = 0.215 and ‖R‖_∞(s=1) = 0.343 (full Newton step
makes it 60% worse).

Key features (full table in [probe_stiff_raw.md](probe_stiff_raw.md)):

| s | ‖R‖_∞ | Φ | Note |
|---|---|---|---|
| -0.15 | 2.06 | 2.54 | **Spike** (‖R‖ jumps 10× vs neighbouring s=-0.10) |
| 0.00  | 0.215 | 0.054 | (u_ex) |
| +0.10 | 0.198 | 0.046 | Descent |
| +0.20 | 0.189 | 0.045 | Descent |
| +0.25 | 0.431 | 0.138 | **Spike** (‖R‖ jumps 2.3× vs neighbouring s=0.20) |
| +0.30 | 0.185 | 0.030 | Recovered descent — actual Φ-minimum |
| +0.35 | 0.183 | 0.029 | Plateau |
| +0.50 | 0.181 | 0.060 | ‖R‖_∞ minimum |
| +1.00 | 0.343 | 0.156 | Full Newton step — 60% worse |
| +1.50 | 0.430 | 0.296 | Continuing uphill |

The spike at s=+0.25 sits precisely where Armijo contraction lands on the second
bisection (α = 1 → 0.5 → 0.25). The spike at s=-0.15 (against direction) is irrelevant
for Newton but confirms the irregularity is structural.

The two minima at `s ≈ 0.20` and `s ≈ 0.30-0.50` straddle the spike. With enough
Newton iteration budget, the line search eventually finds a step in a smooth window
and Newton makes progress.

---

## Why the original framing was wrong

The previous session noted h-/p-refinement didn't rescue the cell, and concluded
the discrete root must not exist. Probe D explains why the refinement test was
inconclusive: the FE consistency residual `‖R(u_ex)‖_∞` shrinks at O(h^{k+1}) as
expected (n=10: 0.215; n=160: 1.95e-4 per the progress doc), but **the solver
safeguards fire at any value of `‖R‖`** because the failure mode is in the iteration
control, not the FE error budget. Mesh/order refinement makes the residual smaller
but doesn't change Picard's merit-divergence misfire or Newton's iteration-budget
exhaustion.

---

## Fix 1 — Picard divergence safeguard tracks ‖R‖_∞ (landed)

[src/solvers/nonlinear.jl:166-180](../../../../src/solvers/nonlinear.jl#L166) before:

```julia
if state.phi_x_new > state.phi_x * solver.divergence_merit_factor
    state.inc_count += 1
    ...
```

After:

```julia
diverged = if solver.mode === :picard
    state.norm_b_new_inf > state.norm_b_inf * solver.divergence_merit_factor
else
    state.phi_x_new > state.phi_x * solver.divergence_merit_factor
end
if diverged
    state.inc_count += 1
    ...
```

Newton-mode bit-identical (the `else` branch preserves the exact prior comparison).
Probe C post-Fix-1: Picard now makes 4 iterations (was 2), reducing ‖R‖ from 0.215 →
0.048 (4.4×) before hitting an honest `linesearch_failed` on iter 4 (where the
Picard direction is genuinely uphill at every α).

## Fix 2 — Dynamic Newton iteration budget at high Re (landed)

The cascade in `porous_solver.jl:230-262` re-engages Newton from Picard's improved
state. With the default `newton_iterations = 20`, the second Newton pass trends
down (Iter 19: ‖R‖ = 0.014) but runs out of budget. Adding 40+ iterations was
needed to span both:

- The transient phase (iters 0-30) where Newton bounces around `‖R‖ ~ 0.05-0.3`
  due to the SmoothVelocityFloor / τ-spike line-search backtracking.
- The quadratic-convergence tail (iters 30+) where Newton accepts α=1 and
  drops residual by ~10× per iter until the noise floor.

New config fields added (mirroring `dynamic_picard_re_*` pattern):

- `dynamic_newton_re_threshold` (default `1e4`)
- `dynamic_newton_re_iterations` (default `60`)

Plumbed through:
- [src/config.jl:65-67](../../../../src/config.jl#L65) (`SolverConfig` struct)
- [src/config.jl:135-136](../../../../src/config.jl#L135) (`validate!` assertions)
- [base_config.json](../../../../base_config.json) (default values)
- [test/extended/ManufacturedSolutions/run_test.jl:485-498](../../run_test.jl#L485) (selection logic)

Wiring `run_simulation.jl` (the non-MMS production entry point) is **out of scope**
for this session — its current `newton_iterations` plumbing has no Re/Da awareness
because it derives physics from `cfg.phys.nu` directly rather than from sweep-level
`(Re, Da)` parameters. A future change can hook the dynamic budget into
`run_simulation.jl` once a characteristic-Re estimator lands.

---

## Verification

Pre-fix → post-fix on `probe_stiff_failing.json` (`Re=1e6, Da=1, α=0.05, k=1, n=10, QUAD, ASGS`):

| Stage | Pre-fix | Post Fix 1 + Fix 2 |
|---|---|---|
| Outcome | ❌ ALL 7 attempts fail | ✅ converged, first homotopy attempt |
| Final `‖R‖_∞` | NaN | 5.47e-4 |
| Stop reason | `max_iters_stagnation` / `linesearch_failed` | `stagnation_noise_floor_reached` |
| Iters at success | — | 3 Picard + 60 Newton |
| L2 u/p | NaN / NaN | 0.127 / 0.025 |

(Verification cadence per [CLAUDE.md](../../../../CLAUDE.md): Blitz + Quick + probe_k1
bit-identicality + 2×2×2 stiff diag — tracked in
[theory/algorithm-improvement-progress.md](../../../../theory/algorithm-improvement-progress.md).)

---

## Deferred — what's still unsettled

- **The line-probe spikes themselves.** Probe E reveals non-smooth points at
  `s=-0.15` and `s=+0.25` where the residual jumps 2-10×. Likely
  `SmoothVelocityFloor` / τ-regularisation thresholds. They don't block convergence
  (Newton iterates through them with enough budget) but they're a latent source
  of fragility for tighter regimes. Not investigated here.
- **`run_simulation.jl` Newton budget.** The non-MMS production entry point doesn't
  thread `(Re, Da)` into the budget calculation. Low-priority — only matters if
  someone runs production simulations at Re ≥ 10⁴.
- **The line-search "best-α fallback" (originally Fix 2 in this doc's earlier
  draft).** Probe E suggests it could give faster convergence, but Fix 2 (Newton
  budget bump) is sufficient on the cell tested. Land later if a more pathological
  case appears.
