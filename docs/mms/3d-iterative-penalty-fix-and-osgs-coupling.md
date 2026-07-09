# 3D MMS (§5.2): the iterative-penalty fix, and the OSGS ∂π/∂u coupling problem

> **Status: CANONICAL for the 3D iterative-penalty (well-posedness) fix + the OSGS ∂π/∂u coupling problem.**
> ⚠️ **Scope correction (2026-07-05).** The iterative-penalty fix below is real and canonical — it restores
> 3D all-Dirichlet **well-posedness** (ε=0 is ill-posed) — and was always a *separate* fix from the P2 accuracy
> defect. This doc's original instinct that "c₁ is NOT the root cause" is now **VINDICATED**: the paper's
> **first author** confirms **Kratos runs the FULL subscale at paper c₁ = 4k⁴ on tetrahedra and solves the 3D
> §5.2 P2 case OPTIMALLY** (both ASGS and OSGS), so **paper c₁ is CORRECT** and the intervening (2026-07-03)
> "element-family c₁ coercivity deficit" claim is **REFUTED**. The Gridap c₁×4-fixes behaviour is a **symptom**:
> Gridap **under-stabilizes P2-3D relative to the paper**, and c₁×4 merely MASKS a **Gridap↔paper
> implementation discrepancy**. **Root cause is OPEN** — a term-level code↔paper discrepancy (most likely, not
> certainly, the P2-3D viscous 2nd-derivative subscale; possibly broader) — and it is **neither c₁ nor the
> penalty** (the penalty fixed well-posedness, a distinct real fix). **Root-cause canonical doc:
> [3d-p2-instability-investigation.md](3d-p2-instability-investigation.md).**
> Investigation dates 2026-06-28 → 2026-06-30. Harness: `test/extended/ManufacturedSolutions3D/` (`smoke3d.jl`).

## TL;DR

1. **The iterative penalty fixes 3D WELL-POSEDNESS — NOT the P2 accuracy defect.** The missing Codina
   ITERATIVE PENALTY in the mass residual made the 3D all-Dirichlet problem ill-posed at ε=0; adding it is a
   real, necessary fix (this doc's core contribution). It does *not* fix the P2 converged-but-wrong *accuracy*
   defect — but neither does c₁. **(2026-07-05, authoritative)** The paper's first author confirms **Kratos
   runs the FULL subscale at paper c₁ = 4k⁴ on tets and solves 3D §5.2 P2 optimally** ⇒ **paper c₁ is CORRECT**;
   the intervening "element-family c₁ coercivity deficit" claim is **REFUTED**, and this doc's *original*
   instinct ("NOT c₁ / no dimensional c₁ — Kratos runs paper c₁") is **VINDICATED**. The P2 accuracy defect is
   an **OPEN Gridap↔paper implementation discrepancy** that c₁×4 merely MASKS (most likely, not certainly, the
   P2-3D viscous 2nd-derivative subscale; possibly broader). See
   [3d-p2-instability-investigation.md](3d-p2-instability-investigation.md). The paper (article.tex §5.2, **line ~1383**) uses ε>0 as the iterative penalty for the 3D
   case because at ε=0 the discrete problem is ill-posed (constant-pressure indeterminacy + BC/mass
   incompatibility). The penalty adds `ε_num·pⁿ` to the mass-eq LHS and `ε_num·pⁿ⁻¹` (PREVIOUS nonlinear
   iterate's pressure) to the RHS, so the residual carries `ε_num·(pⁿ − pⁿ⁻¹)`: nonzero during iterations
   (pins the pressure mode), vanishing at convergence (`pⁿ=pⁿ⁻¹`) so the **converged solution is unaltered**.
2. **The bug:** `continuous_problem.jl` had `ε_num` ONLY in the Jacobian (`mass_term_jac`,
   `(physical_epsilon+numerical_epsilon)·dp`) and NOT in the residual (`mass_term` used `physical_epsilon·p`, physical_epsilon=0). The
   code comment "lagging ε_num·p cancels in the residual" conflated `pⁿ⁻¹` (previous iterate) with the Newton
   increment `dp` — they are not the same; the term only cancels AT convergence, not during iterations.
3. **ASGS-3D is well-posed at PAPER c₁** by the penalty and robust through the eps_pert homotopy. **P1**:
   `eps_used=1` (converges from the hardest start) on all 4 meshes, H¹u and L²p **optimal** (→1.0), L²u ~1.4
   (the documented structured-tet P1 limitation, mesh-quality-bound). **P2**: converges, but the rate is **not
   clean across the full ladder** — L0→L1 super-optimal (5.24/4.08/5.14) then L1→L2 the error *grows*
   (non-monotone). That residual is the known **paper-c₁ P2 under-stabilization** (a τ·c₁ stabilization issue,
   ORTHOGONAL to the penalty — the penalty fixes well-posedness, not stabilization). The earlier "rate 5.25"
   verdict was the L0→L1 segment only; the third mesh exposes the non-monotonicity.
4. **OSGS-3D: P1 SOLVED, P2 RESOLVED 2026-07-09 (§6).** The combination **iterative penalty + boot-skip + JFNK +
   reference-root homotopy** makes **OSGS P1 robust AND fully optimal** at paper c₁ (`eps_used=1` all 4 meshes;
   L²u→2.0, H¹u→1.0, L²p→1.8; ~2–4× more accurate than ASGS). **OSGS P2** was the **∂π/∂u-coupling** problem —
   `ρ(J_frozen⁻¹·∂π/∂u) ≈ 1249` at paper c₁ made JFNK-GMRES stall, so the solver **sat at the exact-guess
   interpolant** (the "accurate solution, `ok=false`" was the *interpolation* error of a stuck solver, not a
   reached root). **Fixed** by a solution-preserving **preconditioner-only c₁×4 inflation**
   (`osgs_jfnk_precond_c1_mult`): ρ_prec → 3.8, `success=true`, `eps_used=1`, quadratic Newton — **see §6**. This
   *exposes* a residual §3 question: the paper-c₁ P2-3D root has accurate velocity but under-stabilized pressure
   (L²p=0.045). (2D k=2 OSGS needs JFNK for the same ∂π/∂u reason.)

## 1. Symptom & the two false leads

3D §5.2: z-extruded field on (0,1)×(0,1)×(0,0.3), (α₀,Re,Da)=(0.5,1,1), ConstantSigma, Deviatoric, structured
Kuhn tets, P2. At paper c₁:
- **ASGS** converges (`ok=true`) but to a **wrong/erratic** solution (L²u non-monotone, ~0.05) — *converged
  but wrong*.
- **OSGS** the coupled Newton **overshoots** (step inf-norm ~268, merit ~1e4–1e6, line-search depletes) —
  *never converges*.

**"False lead #1 — c₁/coercivity"** ([3d-p2-convergence-investigation.md](3d-p2-convergence-investigation.md))
— ✅ **This framing STANDS (VINDICATED 2026-07-05).** c₁ is NOT the true cause of the P2 accuracy defect. The
reasoning below is the correct instinct: c₁×4 makes both converge, so a superseded reading concluded paper c₁
under-budgets coercivity for 3D tets. But increasing c₁ and shrinking the τ-h are the SAME lever (both raise
c₁ν/h²), and c₁×4 only shrank the error *constant* — it MASKS the real defect. **Kratos works at paper c₁ with
the full subscale (first author, 2026-07-05) ⇒ paper c₁ is CORRECT; no dimensional c₁.** The intervening
2026-07-03 "element-family c₁ coercivity deficit" restatement is **REFUTED**. What c₁×4 masks is an **OPEN
Gridap↔paper implementation discrepancy** in the P2-3D case (most likely, not certainly, the P2-3D viscous
2nd-derivative subscale; possibly broader) — NOT c₁, and NOT the iterative penalty (which fixed a separate
thing, well-posedness). See [3d-p2-instability-investigation.md](3d-p2-instability-investigation.md).

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

Observation: at `eps_pert=1` ASGS P2 converges (deep ‖R‖) but can land in an ALTERNATE/spurious root
(L²u=0.14, H¹u≈5.6 vs the correct ~0.05) — the same "noise-floor pseudo-root from a generic guess" the 2D
harness documents.

**Fix (`smoke3d.jl`, commit `5ecc0ca`): reference-root matching.** "First solver-success wins" is unsafe in
3D — it accepted that spurious root and recorded it. Now the exact-guess (`eps_pert=0`) start, which always
lands in the TRUE root's basin, is solved FIRST as the reference; the perturbed starts are then descended
hard→easy and the largest whose converged field matches the reference (relative-L² ≤ `ROOT_MATCH_TOL`=1e-3 —
same-root agreement is ~solver-tol ≈1e-6, a spurious root is O(1)) sets `eps_used`. Errors are always reported
from the reference. In the official sweep this rejected exactly one spurious root (ASGS P2 L0, eps_pert=1,
rel=0.295 → fell back to eps_pert=0.1).

## 3.5 Official structured sweep — full convergence map (2026-06-30)

The official §5.2 sweep (`smoke3d.jl sweep_structured`, regular Kuhn-tet mesh, paper c₁, eps_pert homotopy +
iterative penalty; ASGS = default coupled+boot, OSGS = boot-skip+JFNK; written self-describing to
`results/k*/TET/structured/`). Rates are consecutive-segment slopes; `eps_used` = largest perturbation that
still reached the true root (1 = hardest start).

| method | P | robustness | L²u rates (opt) | H¹u rates (opt) | L²p rates (opt) | verdict |
|---|---|---|---|---|---|---|
| ASGS | P1 | `eps_used=1` all | 1.16 → 1.28 → 1.40 (2) | 0.82 → 0.84 → 0.92 (1) | 0.84 → 0.97 → 1.09 (1) | ✅ robust; H¹u/L²p optimal; L²u structured-tet-limited |
| OSGS | P1 | `eps_used=1` all | **2.01 → 1.92 → 1.91** (2) | **0.95 → 0.92 → 0.96** (1) | 1.93 → 1.74 → 1.77 (1) | ✅✅ robust + **fully optimal**, ~2–4× more accurate than ASGS |
| ASGS | P2 | 0.1, 1, 1 | 5.24 → **−0.82** (3) | 4.08 → **−1.95** (2) | 5.14 → **−1.13** (2) | ⚠️ converges but **non-monotone** at the fine mesh (paper-c₁ under-stabilization) |
| OSGS | P2 | `eps_used=0`, `ok=false` | 2.39 → 2.92 (3) | 1.61 → 1.73 (2) | erratic (tiny) | ⚠️ **accurate** + near-optimal solution but solver GATE not met (∂π/∂u, §4) |

**Reading it:** P1 is a clean win for both methods (OSGS optimal and robust — the headline). At P2, OSGS
produces the accurate solution (small, near-optimal-rate errors) where ASGS under-stabilizes (non-monotone),
but neither is "clean" at paper c₁: ASGS by the rate, OSGS by the convergence gate. The P2 cases are the two
remaining open items (ASGS-P2 stabilization is the c₁ lever the author rejects as a fix; OSGS-P2 gate is §4).

> **UPDATE 2026-07-01:** the "ASGS-P2 = paper-c₁ under-stabilization" reading here was **not confirmed** by the
> follow-up investigation. ASGS-P2-3D is **converged-but-wrong and config-independent** (fails on Frontal meshes,
> Taylor-Hood, and all viscous operators; c₁/inf-sup/mesh-quality all refuted). See the canonical
> [3d-p2-instability-investigation.md](3d-p2-instability-investigation.md) — root cause still open.

## 4. The OPEN problem: OSGS-3D P2 ∂π/∂u coupling (gate not met)

> **Scope (updated 2026-06-30):** the official sweep shows this is now a **P2-only** problem — **OSGS P1 is
> robust and optimal** at paper c₁ (§3.5). The far-guess non-robustness below was observed on P2 (12,12,3) and
> does **not** generalize to P1. P1 is the easier case (lower order, milder ∂π/∂u coupling); the JFNK + boot-skip
> + reference-homotopy recipe converges it from the hardest start.

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

## 6. RESOLVED (2026-07-09): ρ_prec diagnosis + the c₁-inflated JFNK preconditioner

The §4 blocker is **fixed**. A full re-diagnosis on current code (single-cell probes on the (12,12,3) cell)
settled the mechanism to **one number** and found a cheap, solution-preserving fix.

**The mechanism — ρ_prec = 1249.** The spectral radius `ρ(J_frozen⁻¹·∂π/∂u)` — the dropped coupling measured
against the frozen-π preconditioner — is **1249** at paper c₁ (2D reference ≈ 0.88). This one number governs
*both* solver strategies: JFNK's preconditioned GMRES has an eigenvalue ≈1250 (can't converge at any practical
budget), and the staggered Picard π-update's contraction rate *is* ρ_prec = 1249 (so it diverges — matching the
§4 "outer 2 diverges" observation, and refuting the damped-staggering idea, whose rate is the same).

**Everything else refuted (probes, this cell):**
- **Gate** is ready but *starved* — the `3b76864` residual-floor valve can only accept a solve the solver drives
  to the floor; here the solver never descends, so `success=false` is *correct*.
- **The "accurate solution" was the interpolation error of a STUCK solver.** ‖f‖ never leaves 1.7e-3; the solver
  rolls back to the exact-guess interpolant `[u_ex,p_ex]`, whose error (L²u=0.0012) *coincides* with a discrete
  root's only because both are O(hᵏ⁺¹). The doc's "OSGS reaches L²u=0.0012187 ⇒ root correct" reasoning was an
  artifact of starting at the exact solution's interpolant.
- **FD noise** — refuted (the mat-vec is flat across ε∈[1e-11,1e-4]; `fd_epsilon=1e-8` is fine).
- **Weak-pin / stronger ε_num** — refuted (an `eps_mult` 1→1000 sweep made GMRES *worse*: ε_num pins the
  preconditioner but not the residual tangent ⇒ more mismatch).
- **Constant-pressure gauge deflation** — refuted: the frozen-π tangent *is* near-singular on the pure
  constant-pressure mode (σ_min=1.14e-9, cond=9.4e8, alignment 1.0000), but deflating it leaves ρ_prec=1249
  unchanged — so the blowup is NOT the gauge mode (unlike 2D, where deflation gave ρ_defl=0.74).

**The fix — a preconditioner-ONLY c₁ inflation.** A classic Schur/approximate-`J_frozen` preconditioner cannot
help (the current preconditioner is already *exact* `J_frozen⁻¹` via LU; the problem is the *dropped* `∂π/∂u`,
not inverting `J_frozen`). What works: assemble the frozen-π preconditioner with **c₁×4** while the residual `F`
and its matrix-free full tangent stay at **paper c₁**. A larger preconditioner c₁ shrinks the subscale/∂π/∂u
relative to the preconditioner — ρ_prec falls **1178 → 3.8** (a U-shaped optimum at ×4: ×2→93, ×4→3.8, ×8→7.5,
×64→58). Because `F` is untouched, the **converged root is the paper-c₁ solution** (provably solution-preserving:
‖F‖→1.4e-12). Result on (12,12,3): **`success=true`, `eps_used=1`** (robust from the *hardest* perturbed start,
matching OSGS-P1), **quadratic Newton (5 iterations to machine zero)**.

**Implementation (landed, default-off ⇒ byte-identical; Blitz 272/272, Quick 85/85):**
- New config `osgs_jfnk_precond_c1_mult` (schema + `config.jl` + `base_config.json`, default **1.0** = off).
- `osgs_solver.jl`: `solve_osgs_stage!` builds a c₁×mult preconditioner Jacobian closure (`jac_precond_fn`,
  reusing the physical-c₁ live π — its tangent effect is 2nd-order near the root); `_osgs_jfnk_solve!` uses it for
  the GMRES left-preconditioner while the mat-vec keeps differencing the physical-c₁ residual (so the root is
  unchanged) and the C.1 fallback keeps the true paper-c₁ tangent.
- `smoke3d.jl` `run_sweep_structured`: OSGS-**P2** now uses `jfnk_precond_c1_mult=4` + `jfnk_maxiter=80`
  (OSGS-P1 unchanged, mult=1); recorded in `solver_prov` (`recipe="boot_skip+JFNK+precond_c1x4"`).

**What this EXPOSES (a §3 hand-off, not a Q4 failure).** Now that the solver reaches the paper-c₁ discrete root,
its true error is visible: velocity is accurate (L²u=0.00123 ≈ interpolant) but **pressure is ~15× larger
(L²p=0.0446 vs the stuck-interpolant's 0.0029)**. The paper-c₁ P2-3D *pressure* is genuinely under-stabilized —
which is the [§3 accuracy question](3d-p2-instability-investigation.md), now directly measurable for the first
time (c₁×4 *in the residual* gives L²p=0.0029). Q4 (can the solver converge) is resolved; the pressure accuracy
is §3. **Next:** run the full 4-mesh structured sweep with the fix to get OSGS-P2-3D convergence *rates* and see
whether the L²p defect is uniform (§3) or converges.

## Pointers

- Fix: `continuous_problem.jl` (`p_prev`), `solver_core.jl` (`_one_pass` + outer penalty loop +
  `osgs_skip_asgs_boot`), `osgs_solver.jl` (`p_prev` threading).
- Config flags: `iterative_penalty_enabled`, `iterative_penalty_max_iters`, `osgs_skip_asgs_boot` (all default
  OFF; `base_config.json` + schema + `config.jl`).
- Harness: `smoke3d.jl` `solve_one` (eps_pert homotopy + the kwargs). 3D structured result JSONs under
  `results/k*/TET/structured/` are gitignored.
- Memory: `jfnk-osgs-cost-model-and-preconditioner-question`, `k2-needs-tighter-convergence-gate`.
