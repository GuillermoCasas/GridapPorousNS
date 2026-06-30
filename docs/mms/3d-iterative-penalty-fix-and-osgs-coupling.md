# 3D MMS (§5.2): the iterative-penalty fix, and the OSGS ∂π/∂u coupling problem

> **Status: CANONICAL for the 3D tetrahedral MMS convergence behaviour (paper §5.2).** Supersedes the
> c₁/coercivity verdict of [3d-p2-convergence-investigation.md](3d-p2-convergence-investigation.md) — that
> doc's "paper c₁ under-budgets coercivity for 3D tets" conclusion was **wrong**; it was masking a missing
> term. Investigation dates 2026-06-28 → 2026-06-30. Harness: `test/extended/ManufacturedSolutions3D/`
> (`smoke3d.jl`). Author steer: the paper's author (Casas), who pointed directly at the iterative penalty
> and confirmed the 3D case runs fine at paper c₁ in Kratos.

## TL;DR

1. **Root cause of the 3D P2 failure = the missing Codina ITERATIVE PENALTY in the mass residual.** NOT a
   dimension-aware c₁ (there is no such thing — c₁=4k⁴ is dimension-independent; Kratos solves this 3D case
   at paper c₁). The paper (article.tex §5.2, **line ~1383**) uses ε>0 as the iterative penalty for the 3D
   case because at ε=0 the discrete problem is ill-posed (constant-pressure indeterminacy + BC/mass
   incompatibility). The penalty adds `ε_num·pⁿ` to the mass-eq LHS and `ε_num·pⁿ⁻¹` (PREVIOUS nonlinear
   iterate's pressure) to the RHS, so the residual carries `ε_num·(pⁿ − pⁿ⁻¹)`: nonzero during iterations
   (pins the pressure mode), vanishing at convergence (`pⁿ=pⁿ⁻¹`) so the **converged solution is unaltered**.
2. **The bug:** `continuous_problem.jl` had `ε_num` ONLY in the Jacobian (`mass_term_jac`,
   `(eps_val+numerical_epsilon)·dp`) and NOT in the residual (`mass_term` used `eps_val·p`, eps_val=0). The
   code comment "lagging ε_num·p cancels in the residual" conflated `pⁿ⁻¹` (previous iterate) with the Newton
   increment `dp` — they are not the same; the term only cancels AT convergence, not during iterations.
3. **ASGS-3D is SOLVED** by the penalty: converges to the correct root at PAPER c₁ with optimal rate
   (structured (12,12,3)→(16,16,4): L²u rate 5.25, ≥ optimal 3), and is robust through the eps_pert homotopy.
4. **OSGS-3D is NOT fully solved.** The penalty is necessary but not sufficient: there is a separate, genuine
   **∂π/∂u-coupling** solver problem (stronger in 3D than 2D) that prevents robust OSGS convergence from a
   far initial guess. See §4. (2D k=2 OSGS already needs JFNK for the same ∂π/∂u reason.)

## 1. Symptom & the two false leads

3D §5.2: z-extruded field on (0,1)×(0,1)×(0,0.3), (α₀,Re,Da)=(0.5,1,1), ConstantSigma, Deviatoric, structured
Kuhn tets, P2. At paper c₁:
- **ASGS** converges (`ok=true`) but to a **wrong/erratic** solution (L²u non-monotone, ~0.05) — *converged
  but wrong*.
- **OSGS** the coupled Newton **overshoots** (step inf-norm ~268, merit ~1e4–1e6, line-search depletes) —
  *never converges*.

**False lead #1 — c₁/coercivity** ([3d-p2-convergence-investigation.md](3d-p2-convergence-investigation.md)):
c₁×4 makes both converge, so that doc concluded paper c₁ under-budgets coercivity for 3D tets. But increasing
c₁ and shrinking the τ-h are the SAME lever (both raise c₁ν/h²), and c₁×4 only shrank the error *constant*,
not the rate — it MASKED the real defect. Kratos works at paper c₁ ⇒ no dimensional c₁.

**False lead #2 — pressure null mode via `eps_num` (Jacobian-only)**: tested `eps_mult` 1→1000; never fixed
the OSGS overshoot (and corrupted velocity, so not a pure pressure-gauge null mode). The Jacobian-only ε_num
is not the iterative penalty.

The viscous operator (deviatoric, incl. the 3D grad-div `(0.5−1/D)∇(∇·u)`), τ₁/τ₂, quadrature degree, and
the discrete L²-projection were all **verified correct** vs the paper and the working 2D code (grad-div tested
directly: `u=(x₂²,0,0)` gives the correct `(2,0,0)`, not the Laplacian-bug `(2.667,0,0)`).

## 2. Root cause & fix (the iterative penalty)

article.tex line 1383: *"we also add the previous-step value of the compressibility term to the RHS (i.e., we
add ε pⁿ⁻¹ to the mass source) at every nonlinear iteration. This iterative penalty method ensures that the
manufactured solution is not altered."* And line 1375: at ε=0 the 3D problem is ill-posed → they use ε>0.
2D uses ε=0 (line 1098) — which is why 2D never exposed the bug.

**Fix (landed, all gated default-OFF → byte-identical, Blitz 240/240):**
- `src/formulations/continuous_problem.jl`: `build_stabilized_weak_form_residual` gains a `p_prev` kwarg;
  when given, the mass residual adds `iter_penalty = numerical_epsilon·(p − p_prev)`. The matching
  `ε_num·dp` was already in the Jacobian, so residual+Jacobian are now consistent.
- `src/solvers/solver_core.jl`: `solve_system` wraps the solve in an OUTER iterative-penalty loop
  (`_one_pass()` extracted), holding `p_prev` fixed within a pass and updating it between passes; stops when
  relative pressure drift < `xtol`. Gated by `iterative_penalty_enabled`.
- `src/solvers/osgs_solver.jl`: `p_prev` threaded into the coupled (`res_fn_coupled`) and Anderson
  (`_osgs_anderson_outer!`) OSGS residual closures.
- New config (schema + `config.jl` + `base_config.json`, all default-OFF): `iterative_penalty_enabled`,
  `iterative_penalty_max_iters`, `osgs_skip_asgs_boot`.
- 3D harness `smoke3d.jl`: `build_config`/`solve_one` gain `iterative_penalty=true` (default), `osgs_skip_boot`,
  plus the ported 2D-style **eps_pert homotopy** (`eps_pert_base`, `max_n_pert`).

**Validation:** ASGS-3D production, paper c₁, structured (12,12,3)→(16,16,4): `ok=true`, outer penalty loop
converges in 2 passes (drift 0.98→0.0), rate L²u 5.25 / H¹u 4.08 / L²p 5.14. Converged values match the
no-penalty run (penalty vanishes at convergence — solution unaltered, as the paper guarantees).

## 3. The eps_pert homotopy (3D now runs like 2D)

Ported the 2D `execute_outer_homotopy_perturbation_loop!` into `solve_one`: initial guess
`u0 = u_ex + eps_p·(‖u_ex‖/‖h_pert‖)·h_pert`, `eps_p = eps_pert_base/10^attempt` down to 0 (hard→easy), break
at first success. `h_pert` = boundary-vanishing bubble × oscillatory field (so `u0=u_ex` on ∂Ω). This is the
robustness test: *how far from the exact solution can we start and still converge*. (Do NOT "fix"
convergence by starting from the interpolant — that defeats the test. The point is robustness from an
arbitrary start.)

Observation: at `eps_pert=1` ASGS converges (deep ‖R‖) but can land in an ALTERNATE/spurious root
(L²u=0.14 vs the correct 0.0493) — the same "noise-floor pseudo-root from a generic guess" the 2D harness
documents; the correct root is at small eps_pert.

## 4. The OPEN problem: OSGS-3D ∂π/∂u coupling (far-guess non-robustness)

The OSGS coupled tangent drops the dense `∂π/∂u` coupling (frozen-π inexact Newton). This is benign-ish in 2D
but **genuinely worse in 3D**. Findings:

- **The ASGS Stage-I boot is HARMFUL for OSGS** (it is a code-side globalization safeguard, NOT in the paper
  algorithm, which runs the OSGS staggered iteration directly from the guess). The boot converges ASGS to the
  ASGS root (a DIFFERENT fixed point), and OSGS overshoots from there at every eps_pert. Skipping it
  (`osgs_skip_asgs_boot`) lets OSGS run from the guess directly; the eps_pert homotopy supplies the
  globalization the boot was for.
- With **boot-skip**, the OSGS FIRST staggered inner solve **converges even from eps_pert=1** (‖R‖→8.6e-10) —
  so OSGS is robust *to the start* once the boot stops hijacking it. **BUT** the staggered π-UPDATE between
  outer iterations diverges in 3D (outer 2: ‖R‖→2.06, merit 1.6e6) — the dropped ∂π/∂u makes the π-iteration
  non-contractive (the production Anderson over-extrapolation likely worsens it; a PLAIN Picard staggered may
  be more stable — a manual plain-staggered test converged outer-1 but outer-2 was never observed).
- **JFNK** (recovers ∂π/∂u, no staggering) is the principled fix and is the 2D-k2 recipe — but from a far
  guess (eps_pert=1, no boot) its frozen-π preconditioner has no traction (GMRES doesn't converge), it falls
  back to frozen-π coupled, which overshoots; and the matrix-free re-projecting mat-vecs make failing attempts
  slow. From a NEAR guess (eps_pert≈0) penalty+JFNK makes good progress (L²u=0.0045) but its merit line search
  trips (re-projecting merit jumps while ‖R‖ drops). NB the merit (block-equilibrated `Φ=½Σ(bᵢ/wᵢ)²`,
  `wᵢ`=Jacobian diag, `_update_merit_weights!`) is NOT broken — it works in 2D; in 3D it is correctly
  backtracking a genuinely bad ∂π/∂u step.

**Where OSGS-3D stands (KEY):** the **correct OSGS root IS reachable at PAPER c₁** — JFNK+boot-skip+penalty
on (12,12,3) reaches **L²u=0.0012187** (H¹u=0.059, L²p=0.0029), *exactly the c₁×4 value* ⇒ the solution is
right and **c₁ is genuinely not needed**. But the solve reports **`ok=false`**: it's a "good solution,
`ok=false`" situation. The blocker is purely **solver convergence-DETECTION / robustness**, NOT the
discretization (ASGS is optimal at paper c₁; the OSGS root matches c₁×4):
- JFNK's GMRES doesn't fully converge (weak 3D frozen-π preconditioner; a `maxiter=20` cap makes it bail —
  and from FAR guesses, `eps_pert`=1/0.1/0.01, it gets no traction at all → only `eps_pert=0` reaches the root).
- The merit-based line search depletes **near the root** — the re-projecting merit jumps in 3D where it
  doesn't in 2D (the ∂π/∂u-coupled residual). The merit (`_update_merit_weights!`, block-equilibrated) is not
  "broken" (works in 2D); near the 3D OSGS root it backtracks a step that the looser frozen-π fallback can't
  improve either.

So the remaining OSGS-3D work is solver-engineering only: (a) a real **saddle-point/MG preconditioner** for the
coupled tangent so JFNK's GMRES converges (and from any guess) — the Kratos-matching, principled fix; or
(b) a **stabilized (damped, plain-Picard, no-Anderson-extrapolation) staggering** (the manual plain-staggered
inner solve converged); and (c) the **merit/gate near the OSGS root** for the re-projecting residual. The
discretization, c₁, the penalty, and the operators are all confirmed correct. Not a quick knob.

## 5. Next steps (ranked)

1. **OSGS far-guess robustness via the homotopy descent** — if eps_pert=1 fails, descend (0.1, 0.01, 0) and
   record the largest survivor (in progress). Make doomed attempts fail fast (small JFNK budget / a
   divergence-patience guard) so the descent is practical.
2. **A real preconditioner for the OSGS coupled tangent** (saddle-point/MG) so JFNK converges from a far state
   — the principled, Kratos-matching fix.
3. **Plain/damped staggered π-iteration** (no Anderson over-extrapolation; relaxation<1) — cheaper than JFNK;
   test whether it makes the π-update contractive in 3D.
4. Confirm the c₁=paper-value OSGS *converged root* is optimal (it should be, since ASGS at paper c₁ is
   optimal and the discretization is shared) once a robust solver reaches it.

## Pointers

- Fix: `continuous_problem.jl` (`p_prev`), `solver_core.jl` (`_one_pass` + outer penalty loop +
  `osgs_skip_asgs_boot`), `osgs_solver.jl` (`p_prev` threading).
- Config flags: `iterative_penalty_enabled`, `iterative_penalty_max_iters`, `osgs_skip_asgs_boot` (all default
  OFF; `base_config.json` + schema + `config.jl`).
- Harness: `smoke3d.jl` `solve_one` (eps_pert homotopy + the kwargs). 3D structured result JSONs under
  `results/k*/TET/structured/` are gitignored.
- Memory: `jfnk-osgs-cost-model-and-preconditioner-question`, `k2-needs-tighter-convergence-gate`.
