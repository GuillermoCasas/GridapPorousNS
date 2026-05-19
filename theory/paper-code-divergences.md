# Code vs Paper Divergences Log

This document serves as the canonical map for all divergences between the literal mathematical theory detailed in the referenced `theory/article.tex` and the concrete `Gridap.jl` numerical codebase execution. It strictly adheres to the definitions imposed by the `porousns-doc-architect` framework.

Any discrepancies introduced due to numerical stability limits, algebraic bounds, discrete compilation behaviors, or Julia/LLVM restrictions MUST be securely recorded here.

---

## 1. Sub-Grid Mass Balancing Approximation
**Location**: `src/formulations/continuous_problem.jl`

**Apparent Divergence**: The strict analytical integration by parts of the subgrid convective velocities inside the Galerkin test derivations forces the theoretically exact test-side adjoint mapping to include the scalar compressibility term: $(1/\alpha)\nabla \cdot (\alpha \mathbf{u}) v$. In the code, `convective_adjoint` omits this term explicitly.

**Paper Alignment**: `[paper-faithful]` This is **not a divergence**. In `theory/article.tex` (Line 800), the authors explicitly justify the removal of this term across the entire mathematical theory to preserve stability:  
> *"Note that, strictly speaking, one has the term $\frac{1}{\alpha}\nabla \cdot (\alpha \boldsymbol{a}) \boldsymbol{v}_h$ in the expansion of $\mathcal{L}^* V_h$. The inclusion of such term generates a number of crossed terms in (\ref{eq:StabilityEstimate}) that actually harm stability. [...] Here, we have opted for simplifying the formulation by removing the aforementioned term from $\mathcal{L}^* V_h$, leading to a simpler formulation with similar stability properties."*  
Thus, the code omitting this term is a literal, faithful transcription of the exact simplified VMS operator specified in the theoretical methodology.

## 2. Viscous Operator and its Formal Adjoint — `[paper-faithful]`
**Location**: [src/formulations/viscous_operators.jl](../src/formulations/viscous_operators.jl)

**Paper Theory**: The strong viscous operator and its formal adjoint `L*V` per
[article.tex:479](article.tex#L479) both involve the divergence of the (deviatoric or full
symmetric) strain tensor, which expands as

$$\nabla\cdot \varepsilon^{\mathrm{d}}(u) \;=\; \tfrac{1}{2}\Delta u \;+\; \bigl(\tfrac{1}{2} - \tfrac{1}{d}\bigr)\, \nabla(\nabla\cdot u)$$

for the deviatoric case, or

$$\nabla\cdot \varepsilon(u) \;=\; \tfrac{1}{2}\Delta u \;+\; \tfrac{1}{2}\, \nabla(\nabla\cdot u)$$

for the symmetric-gradient case. The `∇(∇·u)` contribution is dimension-dependent for the
deviatoric variant (coefficient `0` in 2D, `+1/6` in 3D) and always present for the
symmetric-gradient variant.

**Code Reality**: `[paper-faithful]` Both the strong operator and its formal adjoint use
the same dimension-aware Hessian-evaluation Operations:

- [`strong_viscous_operator(::DeviatoricSymmetricViscosity, ...)`](../src/formulations/viscous_operators.jl#L145) — `EvalDivDevSymOp(Δ(u), ∇∇(u))`
- [`adjoint_viscous_operator(::DeviatoricSymmetricViscosity, ...)`](../src/formulations/viscous_operators.jl#L163) — `EvalDivDevSymOp(Δ(v), ∇∇(v))`
- [`strong_viscous_operator(::SymmetricGradientViscosity, ...)`](../src/formulations/viscous_operators.jl#L93) — `EvalStrongViscSymOp(Δ(u), ∇∇(u))`
- [`adjoint_viscous_operator(::SymmetricGradientViscosity, ...)`](../src/formulations/viscous_operators.jl#L111) — `EvalStrongViscSymOp(Δ(v), ∇∇(v))`

Both `EvalDivDevSymOp` and `EvalStrongViscSymOp` are dimension-dispatched callable structs
(one method each for `d = 2` and `d = 3`) that compute the exact divergence-of-strain
expansion. Gridap evaluates `∇∇(u)` for trial fields and `∇∇(v)` for test fields cleanly
on Lagrangian elements; for `k_v = 1` the Hessian is identically zero so the
`∇(∇·)` contribution vanishes regardless of dimension, exactly as the analytic operator
demands.

**Historical note (audit P-004)**: Before the Fix 4a commit, the adjoint dropped the
`∇(∇·v)` contribution and returned only `0.5·Δ(v)`. In 2D for the deviatoric variant this
was harmless (`(1/2 − 1/d) = 0`); in 3D for the deviatoric variant and in any dimension
for the symmetric-gradient variant it was a loss of formal symmetry against the paper's
`L*V`. The fix re-uses the strong-operator machinery on `v`, so the adjoint is now exact
in any dimension. The orthogonality smoke test (`osgs_orthogonality_quick_test.jl`) uses
`SymmetricGradientViscosity` and consequently saw a small numerical shift after the fix
(OSGS L2 `3.39545818e-02 → 3.33665493e-02`); the orthogonality property
`‖Π_h(R_h(u))‖ ≈ 10^{-14}` is unchanged. The MMS sweep (which uses
`DeviatoricSymmetricViscosity` per `base_config.json`) is bit-identical to the
post-`fa8aaec` baseline.

## 3. Adjoint Streamline Mapping Positivity
**Location**: `src/formulations/continuous_problem.jl`

**Apparent Divergence**: Naively, one expects the formal convective adjoint operator $\mathcal{L}^*_{conv}$ to evaluate to $-\alpha \mathbf{a} \cdot \nabla \mathbf{v}$. However, the codebase explicitly evaluates `convective_adjoint` with a positive sign (evaluating as $+ \alpha \mathbf{a} \cdot \nabla \mathbf{v}$). Reversing the code sign triggers catastrophic numeric divergence at high Reynolds numbers.

**Paper Alignment**: `[paper-faithful]` This is **not a divergence**, but a mandatory requirement of the stability proof's coercivity. In `theory/article.tex` (Eq 39 / Line 554), the VMS stabilization bilinear form is explicitly constructed by **subtracting** the adjoint: $- \sum_{K}{\langle\mathcal{L}^* U_h, \boldsymbol{\tau} \mathcal{L} U_h\rangle}$. 
When evaluated for the velocity test function, $-\mathcal{L}_{conv}^* \mathbf{u}_h$ correctly evaluates to mathematically positive $A = + \alpha \mathbf{a} \cdot \nabla \mathbf{u}_h$. When multiplied by the strong residual which contains the same positive $A$, this forms the critical $(A - B) \cdot (A + B) = A^2 - B^2$ symmetry (Eq 50 / Line 797 in `article.tex`), creating the positive-definite stability bound $+ \Big\| \tau_1^{1/2} \alpha X(U_h)\Big\|_h^2$. 
Therefore, returning the positive evaluation in the code is structurally identical to evaluating $-\mathcal{L}^*$ in the paper's VMS inner product definition.

## 4. Jacobian Scalar Singularities Regularizations
**Location**: `src/models/reaction.jl`, `src/solvers/nonlinear.jl`

**Paper Theory**: The Jacobian bounds over limits of non-linear parameter expansions are considered mathematically continuously differentiable over the local phase transitions.

**Code Reality**: `[code-actual]` A strictly applied numerical flooring coefficient natively injects a safe non-zero structural element bounded securely by $O(1e-15)$ (via `SmoothVelocityFloor`) within geometric evaluations. It mathematically governs exact continuous Jacobians limits when structural magnitudes approach algebraic zero limits precisely where analytical derivatives of absolute norms structurally fracture.

## 4b. Simplified Stabilization Parameters (eq:Tau1Final / eq:Tau2Final) — [paper-faithful]
**Location**: [src/stabilization/tau.jl](../src/stabilization/tau.jl) — `compute_tau_1`, `compute_tau_2`.

**Apparent Divergence**: Compared to the full paper definitions, the code drops:
1. The `ε h²` contribution from `τ₂`'s denominator (full form: [article.tex:755 `eq:Tau2`](article.tex#L755); simplified: [`eq:Tau2Final` L778](article.tex#L778)).
2. The porosity-gradient `(h/|k_0|)|∇α|` term inside `C_α` for `τ₁` (full form: [article.tex:754 `eq:Tau1`](article.tex#L754); simplified: [`eq:Tau1Final` L777](article.tex#L777)).

**Paper Alignment**: `[paper-faithful]` Both omissions are explicitly justified in the paper itself:

- For `τ₂`, [article.tex L762](article.tex#L762) states: *"The second term in `eq:Tau2` is only strictly necessary for large `ε`."*
- For `τ₁`, [article.tex L764–768](article.tex#L764) states: *"the second term in the definition of `C_α` is in fact unnecessary if `(h/|k_0|)|∇α| ≲ α`. That is, if the porosity changes are well resolved by the mesh. We will assume this to hold in the following, leaving issues related to steep porosity gradients to future work."*

The simplified analysis in §4.2 uses `eq:Tau1Final` and `eq:Tau2Final` directly ([article.tex L775](article.tex#L775)). [tau.jl:5–16](../src/stabilization/tau.jl#L5) implements those final forms verbatim.

**Empirical verification of the well-resolved porosity assumption**: every MMS sweep config in the current regression suite uses `SmoothRadialPorosity` with `r₁=0.2, r₂=0.4` (transition width `Δr = 0.2`). The worst-case ratio is

$$\frac{h\,|\nabla\alpha|}{\alpha} \approx \frac{h \cdot (1-\alpha_0)/\Delta r}{\alpha_0}.$$

At the coarsest mesh (`h = 0.1`, `α₀ = 0.5`) this gives `~0.5`; at the finest (`h = 0.003`, `α₀ = 0.5`) it gives `~0.015`. The Cocquet experiment uses `α ≡ 1` so `|∇α| = 0` trivially. The assumption holds across the entire current sweep.

**Regression anchor**: [test/blitz/tau_blitz_test.jl](../test/blitz/tau_blitz_test.jl) `@testset "Tau1/Tau2 simplified paper form is intentional [P-001, P-008]"` locks in both simplifications:
1. `compute_tau_2` does not take `eps_val` and produces values exactly matching the closed-form `h²/(c₁ α τ_NS + reg)`.
2. `compute_tau_1` is bit-identical when `med.grad_alpha` varies by orders of magnitude while local `α` is held fixed.

A future audit that re-raises P-001 ("τ₂ missing ε·h²") or P-008 ("τ₁ missing C_α") should reach this section first and then fail the anchor test rather than file a regression.

### Re-evaluation triggers (when to switch to the full forms)

- **Switch to `eq:Tau2` (with `ε h²`)** if a future config drives `ε` large enough that `ε h²` becomes comparable to `c₁ α τ_NS`. The `eps_val` field already flows through `phys_cfg`; the switch is a local change in `compute_tau_2` and its derivative `compute_dtau_2_du`.
- **Switch to `eq:Tau1` (with `C_α`)** if a future config has `h|∇α|/α ≳ 1` (steep porosity gradient under-resolved). `grad_alpha` is already plumbed through `MediumState`, so the implementation is unblocked when triggered.

## 5. OSGS Preconditioning \u0026 Linearization Architecture
**Location**: `src/solvers/porous_solver.jl`

**Paper Theory**: In section 6.2 (Eq. 107a-107c), the theoretical staggered iterative scheme for the Orthogonal Subgrid Scale (OSGS) implicitly defines dual architectural boundaries: 
1. The global momentum mapping applies a standard Picard (Oseen) linearization $B_S(\mathbf{u}^{m-1}, U_h^m)$.
2. The orthogonal tracking subspace implies projection tests natively down generic finite element bounds utilizing standard physical topologies $\langle W_h, \boldsymbol{\pi}_h^m \rangle$.

**Code Reality**: `[code-divergent-superior]` The numerical codebase explicitly diverges across both domains to enforce rigid numerical convergence guarantees missing from standard Picard iteration boundaries:
1. **Exact-Newton Tangent Mapping**: Rather than settling for slow, linear Picard evaluation during the primary operator resolution, the nested subgrid extraction evaluates a fully consistent, exact Newton-Raphson tangent derivative Jacobian recursively (`ExactNewtonMode()`). This rigorously accelerates the sub-scale adaptation organically scaling cleanly quadratic internally.
2. **Topologically Unconstrained Orthogonal Projection Bounds**: Classic geometric mapping assumes $W_h \subset \mathcal{V}_h(\Omega)$, inherently inheriting Dirichlet walls. In manufactured mathematical benchmarks, forcing boundary nodes to mirror extreme Dirichlet velocity assumptions annihilates the exact $L^2$-projection, causing an unphysical $O(1)$ residual explosion isolating $O(h^4)$ bounds explicitly on boundaries. To protect analytical limits perfectly, the codebase executes the projection bounds on dynamically unbound spaces (`V_free/Q_free` totally structurally stripped of geometric zero conditions). This protects pure optimal interpolation mathematically and empirically identically.

---

## 6. OSGS Pressure Projection — Constant-Mode Treatment (DEFERRED, OPEN QUESTION)

**Location**: [src/solvers/porous_solver.jl](../src/solvers/porous_solver.jl) — `discrete_l2_projection` call site for `pi_p_next` (~line 596). Also informs the projection-space selection at the `Q_proj = Q_free` assignment (~lines 381–383).

**Status**: `[deferred]` — explicitly considered, intentionally **not** applied as of the Phase 5 batch. Documented here for future reconsideration. The companion plan item, **Phase 5 §3.6** of [algorithm-improvement-plan.md](algorithm-improvement-plan.md), is excluded from the Phase 5 batch but is kept on the long-term ledger.

### The discrete operator the code currently uses

For each OSGS outer iteration, the pressure-side projection $\pi_h^p$ is computed by an unweighted $L^2$ projection of the strong mass residual $R_p$ onto the FE space $\mathcal{Q}_h$. The projection space `Q_proj` is built from `Q_free` ([src/run_simulation.jl](../src/run_simulation.jl)) — a `TestFESpace` with **no Dirichlet constraint and no zero-mean constraint**. So the resulting $\pi_h^p$ in general carries a non-zero constant mode equal to the $L^2$-mean of $R_p$.

### What the paper says about the pressure gauge

[theory/article.tex](article.tex#L407) (line ~407) defines the continuous pressure space in two regimes:

> The pressure will be assumed to belong to $\mathcal{Q}_0 \coloneqq L^2(\Omega)$ in general, while $\mathcal{Q}_0 \coloneqq \{q \in L^2(\Omega) \mid \int_\Omega q \, d\Omega = 0\}$ when the boundary conditions in the problem are all-Dirichlet (as with the regular Navier–Stokes system, constraining the solution to this subspace fixes the free constant when $\varepsilon = 0$; for $\varepsilon > 0$ this condition is met automatically).

So in the **all-Dirichlet velocity** regime — which is the regime of every MMS test config currently in the repo (`small_test_config.json`, `test_config.json`, all `probe_k*.json`) — the *continuous* pressure space is $L^2_0(\Omega)$ (mean zero), and the paper's gauge-fixing mechanism is the $\varepsilon > 0$ penalty in the perturbed continuity equation, which "imposes that the average pressure is zero" ([theory/article.tex](article.tex#L1333), §about $\varepsilon$ choice).

The OSGS projection operator $\pi_h$ is defined in `eq:NonlinearResidualProjection` as the $L^2$ projection of the residual onto $\mathcal{X}_{h0}$ (the FE space), with no explicit mean-removal clause.

### The candidate change — §3.6 of the plan

The deferred patch is the cheap post-hoc form from [algorithm-improvement-plan.md §3.6](algorithm-improvement-plan.md):

```julia
pi_p_next = discrete_l2_projection(R_p, ...)
pi_p_next_mean = sum(∫(pi_p_next)dΩ) / sum(∫(1.0)dΩ)
pi_p_next = pi_p_next - pi_p_next_mean
```

i.e. subtract the $L^2$ mean from the projected scalar field, restricting $\pi_h^p$ to $\mathcal{Q}_h \ominus \{1\}$.

### Why it is plausibly beneficial

The constant mode in $\pi_h^p$ is **inert for the velocity-side stabilization gradient** (the gradient annihilates constants), but it **inflates the projection-drift metric $d_\pi^m$** used as an OSGS convergence test ([osgs_algorithm.tex](osgs_algorithm.tex), §"OSGS stopping mode"). Mean removal would keep $d_\pi^m$ clean of constant-mode noise and is expected to slightly improve pressure-rate measurements in MMS sweeps where $\alpha_0$ is small and the constant-mode pollution scales unfavourably.

### Why §3.6 is **not** in Phase 5

Three independent concerns, all flagged during the planning discussion that preceded this batch:

1. **Regime dependence.** The paper's pressure-space definition switches at the boundary-condition boundary:
   - **All-Dirichlet velocity**: $\mathcal{Q}_0 = L^2_0$. Mean removal in $\pi_h^p$ is *more* paper-faithful than the current unconstrained projection.
   - **Mixed BCs (open outlet, e.g. Cocquet experiment)**: $\mathcal{Q}_0 = L^2$, no constraint. Mean removal would *remove a physically meaningful constant mode*, making the code *less* faithful in that regime.

   A correct §3.6 must therefore be conditional on the BC regime — detected from the configured Dirichlet tags at config-load time. The plan's recipe is unconditional.

2. **Two distinct implementations were identified.**
   - **Option A** (plan's recipe, cheap): compute $\pi_h^p$ on the full `Q_free`, then subtract its $L^2$ mean. One extra inner product per OSGS iteration; mathematically a post-hoc orthogonalization, not a re-definition of the projection.
   - **Option B** (cleaner, slightly more invasive): construct the projection space as `Q_free` $\ominus \{1\}$ explicitly — via a Lagrange multiplier on the Gram matrix, or via a constrained `TestFESpace`. This makes the discrete operator definition coincide structurally with "projection onto a closed subspace", which is unambiguously paper-compatible.

   Option B is the rigorous form. Option A would be defensible only if accompanied by **this** divergence entry documenting the post-hoc mean removal as engineering hygiene, not a redefinition of $\pi_h$.

3. **The continuous gauge is already fixed by $\varepsilon > 0$.** The mass equation contains an $\varepsilon \cdot p$ penalty term ([base_config.json](../base_config.json), `eps_val`). At convergence this drives $p$ toward zero-mean. The §3.6 patch therefore addresses **intermediate-iteration pollution** of the convergence diagnostic, not a steady-state correctness issue. The expected MMS-rate improvement is in the noise floor of the measurement, not in the converged solution.

### Re-evaluation triggers

§3.6 should be re-considered when **any one** of the following becomes true:

1. **A non-all-Dirichlet test config is added to the regression suite** (e.g. an open-outlet MMS variant, a Cocquet-style benchmark with traction BCs). At that point the regime branch is unavoidable and the right time to land Option B with a `bc_regime`-aware switch.
2. **Post-Phase-5 MMS sweep shows pressure-rate sub-optimality traceable to constant-mode noise in $d_\pi^m$.** Symptom: OSGS outer iterations stall at large $d_\pi^m$ values whose $L^2$ decomposition is dominated by the constant mode. Diagnostic: dump `sum(∫(pi_p_next)dΩ)` per outer iteration on a coarse mesh and compare against the full $\|\pi_h^p\|_{L^2}$.
3. **Phase 6 §2.1 continuation runs at $\alpha_0 = 0.05$** (narrow channel) show pressure-rate degradation in the corner cells. The constant-mode pollution scales unfavourably at small $\alpha_0$ per the critical-analysis discussion.
4. **The paper-code-divergences ledger gains another mass-side entry** that interacts with the pressure space definition (different mass-residual form, different penalty, different test space). At that point the pressure-space treatment becomes part of a larger formal review and §3.6 should be settled definitively.

### What **not** to do (lessons captured for future sessions)

- **Do not** apply mean removal to the pressure variable $p$ itself (mutating `final_x0`). The $\varepsilon > 0$ gauge fixing is a softly-enforced feedback; slamming the pressure to zero mean every iteration would fight it and destabilize Newton. §3.6 modifies only $\pi_h^p$ (the *projection* used inside the stabilization term), not the pressure variable.

- **Do not** land Option A without also landing the conditional on BC regime. An unconditional mean removal at a future open-outlet test would degrade rates silently.

- **Do not** treat §3.6 as a prerequisite for Phase 6. Phase 6 §2.1 (continuation driver) can proceed without it; if the corner-cell pressure-rate symptom in trigger (3) above appears, that becomes the natural moment to revisit.

### What is in scope for the current Phase 5 batch (without §3.6)

§3.1, §3.5, §5.1, §5.2 only. §3.6 is excluded by user direction in the planning session and is captured here for the next iteration. The expected effect of *not* landing §3.6 in this batch: pressure rates may not improve as much as theoretically possible; they are not expected to *regress*. The post-Phase-5 baseline becomes the reference against which a future §3.6 commit is judged for actual numerical improvement.

---

## 7. Three-Way Asymmetry in the Legacy `max_iters_caught` Exception Path — `[code-actual]`

**Location**: [src/solvers/porous_solver.jl](../src/solvers/porous_solver.jl) — `_initialize_asgs_state!` (Stage I), `_run_osgs_inner_cascade!` (Stage II inner), `_run_asgs_mms_extension!` (MMS extension).

**Paper Theory**: Algorithm B (`theory/osgs_algorithm.tex` §"The shared cascade") describes a Newton→Picard→Newton cascade that is reused at three call sites. The pseudocode is mode-agnostic with respect to *how* a non-converged Newton exit is detected — only the success/failure binary matters.

**Apparent Divergence**: The legacy Gridap exception path (`"Reached maximum iterations"`, caught as `:max_iters_caught` by `safe_fe_solve!`) is treated *differently* at the three sites:
- Stage I: structural failure → Picard fallback fires (matches the "Stage I quadratic-basin guarantee" addendum).
- Stage II inner: single-iteration partial success, `iter_count` increments, no fallback, log line emitted.
- ASGS-MMS extension: single-iteration partial success, `iter_count` increments, no fallback, **no** log line emitted.

**Paper Alignment**: `[code-actual]` — These three policies are documented additions to Algorithm B, not divergences from it. The modern `SafeNewtonSolver` exits cleanly with `stop_reason = "max_iters_stagnation"` on the `:ok` path; the exception path is purely defensive against legacy or non-`SafeNewtonSolver` solver instances and is normally never taken. The paper now explicitly enumerates the three policies in `theory/osgs_algorithm.tex` §"The shared cascade" (paragraph "Legacy `max_iters_caught` exception path").

**Re-evaluation trigger**: If a custom non-`SafeNewtonSolver` is plugged into the solver stack, or if Gridap's exception contract changes, re-audit which policy each site should apply.

---

## 8. `eval_time` Reporting Convention — `[code-actual]`

**Location**: [src/solvers/porous_solver.jl](../src/solvers/porous_solver.jl), the `eval_time` return-tuple slot of `solve_system`.

**Paper Theory**: Algorithm O describes the orchestration without committing to a particular wall-clock reporting boundary.

**Implementation Reality**: `eval_time` measures the cumulative wall time of three regions:
1. Stage I cascade (`@elapsed` around `_initialize_asgs_state!`).
2. ASGS-MMS extension cycle loop (`@elapsed` *inside* `_run_asgs_mms_extension!`, around the cycle loop only).
3. OSGS outer staggered loop (`@elapsed` *inside* `_run_osgs_relaxation!`, around the for-loop only).

Crucially, `eval_time` **excludes**:
- OSGS mass-matrix assembly and Cholesky factorisation (run-once setup; lives outside the `@elapsed` block in `solve_system`).
- The ASGS-MMS extension's setup (oracle call, local-solver construction).
- The OSGS post-loop `mms_budget_exhausted` check.
- All `diag_cache` writes after the timed regions.

**Paper Alignment**: `[code-actual]` — `eval_time` is the *iterative* wall time, not the *total* per-call wall time. Use a wall-clock `@elapsed` wrapper around the whole `solve_system` call if total cost is needed; that figure will be larger than `eval_time` by the OSGS setup cost (which can be a non-trivial chunk on large meshes).

**Re-evaluation trigger**: If a `setup_eval_time` field is ever added to the return tuple (or a wider diagnostics-cache refactor lands), this entry should be updated to reflect the new split.

---

## 9. `mms_budget_exhausted` Reports Solver Success, Not Verification Success — `[paper-faithful]` (post P-007/Fix-6)

**Location**: [src/solvers/porous_solver.jl](../src/solvers/porous_solver.jl), end of `_run_osgs_relaxation!`.

**Paper Theory**: `theory/osgs_algorithm.tex` Algorithm C (OSGS branch with MMS hook) and Algorithm D (plateau verifier). The new paragraph "Budget exhaustion is not a verification failure of the solver" in §"How Algorithm D hooks into the core" now states the contract explicitly.

**Implementation Reality**: When the OSGS outer-loop budget exhausts without the plateau test firing, the solver returns
`(S_solver, S_plat, …) = (True, False, …)` with `diag_cache["mms_stop_reason"] = "mms_budget_exhausted"`. The base OSGS fixed point converged (necessary for the post-loop check to even fire); the plateau verifier ran out of samples.

**Paper Alignment**: `[paper-faithful]` — the documented two-flag split (`solver_success, mms_plateau_success`) is the resolution of audit finding P-007 / Fix 6 ("solver success conflated with verification success"). Callers must read both flags. The P-007 follow-up in the audit-findings triage plan migrates *callers* to use the second flag explicitly; the solver-side contract is already in this final form.

**Re-evaluation trigger**: If the audit plan's Fix 6 caller migration changes the return-tuple shape or adds an `overall_verification_success` convenience flag, update this entry.
