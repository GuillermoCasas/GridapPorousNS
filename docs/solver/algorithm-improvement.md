# Algorithm improvement — consolidated (analysis → plan → progress)

> **HISTORICAL PROCESS DOC.** Consolidates the original critique → plan → progress chain (formerly three separate files). Resolved items are reflected in the code. Canonical paper↔code references: [paper-code-divergences.md](paper-code-divergences.md) and [algorithm-code-mapping.md](algorithm-code-mapping.md).
>
> The Part 3 progress log records past verification runs against diagnostic/scratch files
> (`probe_stiff*.json`, `small_test_config.json`, `predictions_small_test.md`, `probe_stiff_findings.md`)
> that were **removed in a later cleanup**. Those names are left as a historical record of what was run,
> not as live files — they appear here un-linked (the live diagnostic script `probe_stiff_diagnose.jl`
> still exists).

---

## Part 1 — Critical analysis

**Scope.** This report evaluates the orchestrated solver cascade documented in
[theory/osgs_algorithm.tex](../../theory/osgs_algorithm.tex) against the standard
literature on FEM, VMS stabilization, and globalized nonlinear iteration. It
identifies (i) theoretical inconsistencies that I believe constitute genuine
bugs from the perspective of optimal convergence, (ii) robustness gaps that
prevent the cascade from reaching every viable state, and (iii)
quality-of-implementation choices that could be made differently for better
behaviour. The user's stated priorities are robustness and optimal
convergence in every case, not raw speed; I have ranked findings on that
basis.

A balancing note: the cascade is, on the whole, well-designed. Diagonal
merit equilibration, the Newton ↔ Picard fallback, OSGS staggered
relaxation with block-isolated Anderson acceleration, and noise-floor-aware
safeguards are all standard and correctly motivated. The criticisms below
are about *details* and *missing pieces*, not the overall architecture.

---

### 1. Theoretically problematic items (highest priority)

#### 1.1 The line search applies the Newton Gauss-Newton identity to Picard steps

**Where.** [src/solvers/nonlinear.jl:216](../../src/solvers/nonlinear.jl#L216),
inside `_safe_solve_inner!`:

```julia
dir_deriv = -2.0 * state.phi_x
```

This value is then passed to `eval_armijo_linesearch_pass!`, which tests the
Armijo predicate
$\hat\Phi \le \Phi + c_1 \alpha D$ with $D = -2\Phi$.

**Why this is wrong for Picard.** The identity $D = -2\Phi$ is the
directional derivative of $\Phi_w(\boldsymbol{x})
= \tfrac{1}{2}\|W^{-1}\boldsymbol{b}\|^2$ along an *exact Newton step*
$\delta\boldsymbol{x} = \mathcal{J}^{-1}\boldsymbol{b}$:

$$
D \;=\; (\nabla\Phi)^\top(-\delta\boldsymbol{x})
  \;=\; (W^{-2}\boldsymbol{b})^\top\mathcal{J}\,(-\mathcal{J}^{-1}\boldsymbol{b})
  \;=\; -\|W^{-1}\boldsymbol{b}\|^2
  \;=\; -2\Phi.
$$

The cancellation requires the search direction to satisfy
$\mathcal{J}\delta\boldsymbol{x} = \boldsymbol{b}$ with the *true* Jacobian.
The Picard fallback solves
$\mathcal{J}_{\mathrm{P}}\delta\boldsymbol{x} = \boldsymbol{b}$ where
$\mathcal{J}_{\mathrm{P}}$ drops the convective and Forchheimer
linearisations. For a Picard step the directional derivative is

$$
D_{\mathrm{P}} = -(W^{-2}\boldsymbol{b})^\top \mathcal{J}\mathcal{J}_{\mathrm{P}}^{-1}\boldsymbol{b},
$$

which is in general neither $-2\Phi$ nor even negative. The current code
uses $-2\Phi$ unconditionally, so the Armijo test becomes
$\hat\Phi \le \Phi(1 - 2c_1\alpha)$, which has nothing to do with the actual
descent property of the Picard step.

**Concrete failure mode.** When $\mathcal{J}_{\mathrm{P}}^{-1}\boldsymbol{b}$
happens to *increase* $\Phi$ along the step (the Picard direction is not a
descent direction at this point), the line search will backtrack until
$\hat\Phi \le \Phi(1 - 2c_1\alpha)$ is satisfied, which may never happen
within $N_{\mathrm{ls}}$ tries. The line search then exits with
`linesearch_failed`, and the orchestration aborts. *But this exit is an
artefact of an incorrect descent condition, not a genuine failure of
Picard.* In well-known regimes the Picard iteration is convergent in a
weighted norm (not necessarily in $\Phi_w$).

**Recommendation.** Make `_safe_solve_inner!` aware of the linearisation
mode and use the correct line-search criterion:

- For Newton: keep `D = -2Φ` (cheap, exact, correct).
- For Picard: either (a) compute the true $D_{\mathrm{P}} = \nabla\Phi^\top
  (-\delta\boldsymbol{x})$ explicitly (one extra matrix-vector with
  $\mathcal{J}$), or (b) replace the Armijo test with a **monotone
  residual** condition $\|\hat{\boldsymbol{b}}\|_\infty \le
  (1-c_1\alpha)\|\boldsymbol{b}\|_\infty$ that is reasonable for any
  fixed-point map. Option (b) is preferable because it removes the merit
  function from Picard's accept/reject decision, which is the heart of the
  mismatch.

This is, in my reading, the single most consequential theoretical defect
in the cascade. It very likely costs convergence on hard cases where Picard
is *almost* working.

#### 1.2 The OSGS short-circuit silently breaks the `Mode_stop = "both"` semantics

**Where.** [src/solvers/porous_solver.jl:531-534](../../src/solvers/porous_solver.jl#L531-L534):

```julia
if x_diff_inf <= stagnation_tol && osgs_iter >= 2
    overall_converged = true
end
```

This rule fires *after* the regular convergence predicate is evaluated and
unconditionally overrides it. In `Mode_stop = "both"` (the default) the
predicate is $S_{\mathrm{state}} \wedge S_{\mathrm{proj}}$; the short-circuit
sets `overall_converged = true` solely on the basis of $\ell^\infty$ DOF
drift, regardless of $d_\pi^m$.

**Why this is a problem.** OSGS is a fixed point on the *pair*
$(U_h, \boldsymbol{\pi}_h)$. The projection can drift even when $U_h$ has
stopped moving — this happens when the freezing of $\boldsymbol{\pi}_h$
inside the inner Newton produces a state that re-projects back near, but
not at, the previous projection. In that regime, $\ell^\infty(U_h^m -
U_h^{m-1})$ can fall below $\textstag$ while $d_\pi^m$ is still
$\gg \tau_{\mathrm{proj}}$. Declaring convergence here returns a non-fixed
point.

**Where it bites.** Anywhere $(U_h, \boldsymbol{\pi}_h)$ converges
component-wise at very different rates. High-Da regimes are the classic
example: the reaction term anchors $U_h$ quickly, but the projection of
the convective residual slowly accommodates the freezing-thawing structure
of $\boldsymbol{\pi}_h$.

**Recommendation.** The short-circuit should be gated by the active
`Mode_stop`. The simplest correct rule is:

```julia
if x_diff_inf <= stagnation_tol && osgs_iter >= 2
    # Only short-circuit when state drift is the binding criterion.
    if stab_cfg.osgs_stopping_mode == "state_drift"
        overall_converged = true
    end
end
```

Better still: drop the short-circuit entirely and rely on the existing
`max(τ_OSGS, stag)` and `max(τ_proj, stag)` floors. The short-circuit is a
defence against "limit-cycle oscillation never cleanly drops below
$\tau_{\mathrm{OSGS}}$", but the noise-floor floors already provide that
defence in the original predicate.

#### 1.3 The OSGS inner Newton has no Picard fallback

**Where.** [src/solvers/porous_solver.jl:449-453](../../src/solvers/porous_solver.jl#L449-L453):

```julia
if nls_cache.result.stop_reason == "linesearch_failed" ||
   nls_cache.result.stop_reason == "merit_divergence_escaped" ||
   nls_cache.result.stop_reason == "linear_solve_nan"
    println("-> OSGS Inner Newton sweep failed algebraically...")
    success = false
    break
end
```

Stage I has an elaborate Newton → Picard → Newton cascade for *exactly*
these exit reasons. The OSGS inner Newton aborts on them. This means: the
ASGS root might be reachable through Picard globalization, but if a single
OSGS staggered iteration's inner Newton hits the same wall, the whole
outer loop dies without trying the fallback that worked at Stage I.

**Why this matters.** The OSGS Jacobian at staggered iteration $m$ is the
ASGS Jacobian (with frozen $\boldsymbol{\pi}_h$). The same convective
cusp that nearly broke Stage I will reappear inside the OSGS loop, with
the additional perturbation of a stale projection. It is *more* likely,
not less, that the inner Newton there needs the Picard fallback.

**Recommendation.** Wrap the inner Newton call in the same `PicardFallback`
(Algorithm B) that Stage I uses. The Picard Jacobian during OSGS would
naturally freeze the projection (same as the inner Newton already does)
and drop the convective/Forchheimer linearisations, so the implementation
is straightforward.

#### 1.4 Picard runs to the same tight `ftol` as Newton

**Where.** [src/run_simulation.jl:174](../../src/run_simulation.jl#L174):

```julia
nls_picard = SafeNewtonSolver(p_ls, sol_cfg.picard_iterations,
                              sol_cfg.max_increases, sol_cfg.xtol,
                              sol_cfg.ftol, ...)
```

Picard uses `sol_cfg.ftol`. Theoretically, Picard converges *linearly*,
with rate determined by the contraction constant of the iteration map.
Driving it to a tight tolerance like $10^{-10}$ requires roughly
$\log_{1-c}(10^{-10}) \approx 23/c$ iterations for a contraction constant
$c$, which on stiff cases can be several hundred. Newton, by contrast,
needs only to be in the basin of attraction.

**The role of Picard in this cascade is globalization, not solving.** Once
the iterate is in Newton's basin, Newton's quadratic convergence is far
cheaper than continuing Picard.

**Recommendation.** Add a separate Picard tolerance, e.g.
`sol.picard_handoff_ftol`, set to a loose value like $10^{-3}\textftol$ or
even $\sqrt{\textftol}$. Hand off to Newton as soon as that threshold is
hit. The current rigid `picard_iterations` cap is a crude proxy for this
but is set conservatively, with the result that Picard often runs all its
iterations needlessly.

#### 1.5 The bootstrap projection $\boldsymbol{\pi}_h^0$ uses the ASGS state

**Where.** [src/solvers/porous_solver.jl:400-412](../../src/solvers/porous_solver.jl#L400-L412):

```julia
u_h, p_h = final_x0   # final_x0 is the ASGS-converged state
R_u = inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
R_p = inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf)
...
pi_u = FEFunction(U_proj, x_u)
pi_p = FEFunction(P_proj, x_p)
```

**The issue.** ASGS and OSGS converge to *different* discrete solutions.
$U_h^{\mathrm{ASGS}}$ satisfies $\mathcal{R}_{\mathrm{ASGS}}(U_h) = 0$,
which in OSGS notation is "$\mathcal{R}(U_h) - \boldsymbol{\pi}_h = 0$ with
$\boldsymbol{\pi}_h$ being the *whole* residual." Bootstrapping
$\boldsymbol{\pi}_h^0$ from $U_h^{\mathrm{ASGS}}$ therefore *over-shoots*
the OSGS fixed-point projection in regimes where ASGS and OSGS differ
significantly.

The empirical fix has been to add the OSGS warmup phase (Sec.~6.5 of the
.tex), which relaxes the inner Newton tolerance until the projection
settles. But this is treating a symptom: the loop spends the first
$m_{\mathrm{w}}$ iterations *unwinding* an aggressive bootstrap.

**Recommendation.** Initialize $\boldsymbol{\pi}_h^0 = 0$ instead. The
first inner Newton iteration then runs the ASGS problem (which is
already converged), produces the same $U_h^{\mathrm{ASGS}}$, and the first
projection $\boldsymbol{\pi}_h^1$ is computed from $U_h^{\mathrm{ASGS}}$
— exactly the current bootstrap value. But now the iteration sees this
projection for the *first* time, with full natural damping, rather than
having it injected as an initial condition before any staggered update.

The conceptual savings: the warmup phase becomes redundant; the cascade
is a pure continuation from ASGS ($\boldsymbol{\pi}_h = 0$) to OSGS
($\boldsymbol{\pi}_h \neq 0$).

---

### 2. Robustness gaps

#### 2.1 No automatic homotopy in $\mathrm{Re}$, $\mathrm{Da}$, or $\alpha$

For the steady porous Navier–Stokes problem at high $\mathrm{Re}$ and high
$\mathrm{Da}$, the standard FEM-literature defense against divergence is
**parameter continuation**: solve the problem at a tame $(\mathrm{Re}_0,
\mathrm{Da}_0)$, then ramp to the target values, using each converged
state as the initial guess for the next.

The current cascade has no such mechanism in `run_simulation`. If the user
asks for $\mathrm{Re} = 10^6$ and the cascade can't converge from
$\boldsymbol{x}^0 = 0$, the run fails. The user must manually solve a
sequence of intermediate problems and feed each into the next.

The MMS sweep harness has a related mechanism
([test/extended/ManufacturedSolutions/run_test.jl:499](../../test/extended/ManufacturedSolutions/run_test.jl#L499))
called `execute_outer_homotopy_perturbation_loop!`, but it perturbs the
exact initial guess against the manufactured solution — it doesn't ramp
$\mathrm{Re}/\mathrm{Da}$. That mechanism is also harness-only.

**Recommendation.** Add an optional continuation wrapper around Stage I
(triggered when the unbroken Stage I fails). A natural design:

1. Detect Stage I failure at the trigger reason.
2. Build a sequence $(\mathrm{Re}_i, \mathrm{Da}_i)$ with $\mathrm{Re}_0
   = 1$, $\mathrm{Da}_0 = 1$ and geometric growth to the target.
3. Run Stage I at each $(\mathrm{Re}_i, \mathrm{Da}_i)$, using the
   previous converged state as the initial guess.
4. If still failing at a target step, abort with diagnostics.

This is standard in commercial CFD codes (Fluent's pseudo-time stepping,
COMSOL's parametric solver) and would close the largest robustness gap
in the current cascade.

#### 2.2 No fallback for `xtol_stagnation` / `max_iters_stagnation`

The Picard fallback (Algorithm B) is invoked when *any* Newton non-success
exit reason fires (this is correct, as I noted in the rewrite). But the
fallback itself has only one Picard pass and one Newton retry. If the
retry hits `xtol_stagnation` or `max_iters_stagnation` again, the
orchestrator gives up.

A more aggressive design would:

1. On second-Newton failure, run *another* Picard pass with a different
   coefficient mixing (e.g. a continuation between $\boldsymbol{u}^k$ and
   $\boldsymbol{u}^0$ used for the convective coefficient).
2. Or run Newton with a strong damping (a hand-picked $\alpha < 1$
   trust-region surrogate).
3. Or trigger the homotopy of §2.1.

Currently there is no second-chance behaviour beyond the single
Newton → Picard → Newton ladder.

#### 2.3 No best-state retention across cascade stages

`SafeNewtonSolver` retains the best-seen $\boldsymbol{x}_{\mathrm{best}}$
*within* a single safeguarded Newton call. But the *cascade* does not
retain a best state across stages. If Newton makes substantial progress
but exits at `linesearch_failed`, and then Picard accidentally degrades
the state, the Picard-degraded state — not the Newton-best state — is
the seed for the second Newton call.

**Recommendation.** The orchestrator should compare residuals across the
cascade and use whichever stage produced the lowest residual as the seed
for the next. This is a simple `if final_res < best_residual_so_far: ...`
pattern at the top level.

#### 2.4 The catastrophic-state guard is asymmetric

After Picard, the code checks
[`any(!isfinite, x0)`](../../src/solvers/porous_solver.jl#L218). This same
check is *not* applied after the first Newton attempt. If the linear
solver in Newton produces a non-NaN but Inf-contaminated step (this can
happen with marginally singular matrices and certain BLAS code paths),
the Picard pass starts with garbage.

**Recommendation.** Promote the `any(!isfinite)` check to a wrapper
applied after *every* stage of the cascade.

---

### 3. Numerical-quality concerns (FEM theory)

#### 3.1 Merit-function equilibration is diagonal, not block-diagonal

The weights $w_k = \max(|\mathcal{J}_{kk}|, \varepsilon\,\|\text{diag}\mathcal{J}\|_\infty, \varepsilon)$
rescale each row by its own diagonal entry. For a saddle-point system
$\begin{pmatrix} A & B^\top \\ B & 0 \end{pmatrix}$, the pressure rows
have *zero* on the diagonal (block $C = 0$); the stabilisation term
contributes a non-zero diagonal of order $\tau h^{-2}$, but the resulting
weights are dimensionally inconsistent with the velocity weights.

**Why this matters.** The Armijo predicate
$\hat\Phi_w \le \Phi_w(1 - 2c_1\alpha)$ becomes biased: a step that
substantially reduces the momentum residual but slightly increases the
mass residual passes the test more easily than a balanced step. In MMS
runs this manifests as $L^2$-pressure rates degrading before
$L^2$-velocity rates.

**Recommendation.** Use a block-equilibrated merit:
$\Phi(\boldsymbol{x}) = \tfrac{1}{2}(\|W_u^{-1}\boldsymbol{b}_u\|^2 +
\|W_p^{-1}\boldsymbol{b}_p\|^2)$ with $W_u, W_p$ computed separately. This
respects the block structure and the natural scaling of momentum vs mass
balance. The implementation cost is one extra pass over the DOF index map
to identify velocity vs pressure rows.

#### 3.2 Mass matrices are LU-factored even though they are SPD

[src/solvers/porous_solver.jl:374-379](../../src/solvers/porous_solver.jl#L374-L379):

```julia
ls_u = LUSolver()
ls_p = LUSolver()
num_u_fac = numerical_setup(symbolic_setup(ls_u, M_u), M_u)
num_p_fac = numerical_setup(symbolic_setup(ls_p, M_p), M_p)
```

The $L^2$ mass matrices are SPD. Cholesky factorization is roughly $2\times$
faster than LU for the same accuracy and is provably more numerically
stable (it avoids partial pivoting, which can introduce small
permutation-induced perturbations). On large meshes the projection cost
becomes non-trivial; using LU here costs a constant factor of 2 in the
OSGS loop's projection time.

This is a small but real numerical defect. The relevant Gridap solver is
`CholeskySolver()` (or a wrapper around `cholesky(...)` if not exposed).

#### 3.3 Anderson's hardcoded safety factor and lack of restart

[src/solvers/accelerators.jl:83](../../src/solvers/accelerators.jl#L83):

```julia
if norm(x_next .- g_k, Inf) > 10.0 * norm(f_k, Inf)
```

The factor of `10.0` is a magic number. Worse, when this safety check
fires, the history is *not cleared*: the very next iteration will try the
same near-singular least-squares problem with one more column appended.
The standard Anderson-with-restart pattern would empty `history_X` and
`history_F` on triggering, so the accelerator rebuilds from scratch.

Additionally, the least-squares solve
[`A_ls \ b_ls`](../../src/solvers/accelerators.jl#L69) has no
regularization. When the history is mature, $\Delta F$ becomes
near-rank-deficient, and the normal-equations matrix $\Delta F^\top M
\Delta F$ becomes ill-conditioned. A truncated-SVD solve or a small
$\lambda I$ regularization would prevent the resulting $\gamma$ from being
dominated by noise.

**Recommendation.** Three independent fixes:

1. Expose the safety factor as a config field (`accelerator.safety_factor`).
2. Clear `history_X` and `history_F` on safety-fallback trigger.
3. Use a regularized solve: `gamma = (A_ls + λ*I) \ b_ls` with
   `λ = ε_mach * tr(A_ls)/size(A_ls,1)` or equivalent.

#### 3.4 `freeze_jacobian_cusp` is opaque

The flag at
[src/stabilization/tau.jl:18-19](../../src/stabilization/tau.jl#L18-L19)
controls whether to skip the linearisation of $\partial\tau/\partial u$
through the $|\boldsymbol{u}|$ cusp. When `freeze_cusp = true`, the
derivative term is dropped; when `false`, it is computed via the
regularized smooth velocity floor.

Theoretically: freezing the cusp converts the linearisation locally from
Newton to a (partial) Picard scheme. This is fine for robustness but
degrades the local convergence rate from quadratic to linear in
neighbourhoods of $|\boldsymbol{u}| \to 0$. If `freeze_cusp` is left
`true` by default, the cascade silently loses Newton's quadratic rate on
problems with significant stagnation regions.

**Recommendation.** Document the rate trade-off clearly in the .tex
appendix and the config schema. Optionally: make the freezing
*automatic*, triggered only on those quadrature points where
$|\boldsymbol{u}| < c \cdot u_{\mathrm{floor}}$, rather than globally.

#### 3.5 Quadrature degree is not adaptive to nonlinear terms

`get_quadrature_degree` returns a fixed degree based on the velocity
polynomial order $k_v$. The Forchheimer term contributes a $|\boldsymbol{u}|
\boldsymbol{u}$ factor whose effective polynomial degree is non-integer (it
behaves like $\boldsymbol{u}^{2}$ in the smooth regime but loses regularity
near zero velocity). For high-order discretizations ($k \ge 3$), under-quadrature
of this term can pollute MMS convergence rates.

**Recommendation.** For Forchheimer flow, bump the quadrature degree by at
least $\lfloor k/2 \rfloor$ above the linear-NS default. This is cheap
($\mathcal{O}(k^d)$ growth in cell-wise cost) and removes a subtle source of
rate degradation.

#### 3.6 Pressure is unpinned in the OSGS projection

`Q_free` is unconstrained — no Dirichlet, no zero-mean. For problems where
the pressure has a free constant mode (pure Dirichlet-velocity walls,
which is the dominant case), the projection $\Pi_{Q_{\mathrm{free}}}(R_p)$
absorbs that constant mode into $\pi_{h,p}$ without obstruction. This is
not "wrong" — the constant mode contributes nothing to the stabilisation
gradient — but it makes the projection drift $d_\pi^m$ noisier than it
needs to be.

**Recommendation.** Project onto $Q_{\mathrm{free}} \ominus \{1\}$ (the
$L^2$-orthogonal complement of the constant function). The implementation
cost is one mass-weighted inner-product subtraction per projection.

---

### 4. Architectural observations

#### 4.1 OSGS warmup damps the inner Newton but not the projection update

The warmup phase only loosens $\textftol^{\,\mathrm{inner}}$. It doesn't
damp the projection update $\boldsymbol{\pi}_h^m \gets \Pi(\mathcal{R}(U_h^m))$.
A more symmetric warmup would use a relaxation factor
$\theta_m < 1$ during the warmup window:

$$
\boldsymbol{\pi}_h^m \;\gets\; \theta_m\,\Pi(\mathcal{R}(U_h^m))
              \,+\, (1-\theta_m)\,\boldsymbol{\pi}_h^{m-1},
$$

ramping $\theta_m \to 1$ as $m \to m_{\mathrm{w}}$. This is the textbook
under-relaxed Picard scheme and would smooth the projection's transient
behaviour, reducing the risk that the inner Newton has to chase a
projection that is itself oscillating.

#### 4.2 No cross-parameter consistency validation

`validate!` in `src/config.jl` enforces individual parameter ranges
($0 < c_1 < 1$, etc.) but does not check inter-parameter consistency. The
following obvious inconsistencies would pass validation:

- `ftol = 1e-10` paired with `osgs_tolerance = 1e-3`. The outer loop will
  declare convergence long before the inner Newton has resolved its own
  state.
- `stagnation_noise_floor = 1e-5` paired with `ftol = 1e-3`. The noise
  floor is *below* the target tolerance, so it is never tested.
- `picard_iterations = 1` paired with `newton_iterations = 2`. Stage I has
  almost no budget; the cascade will fail trivially on any non-trivial
  problem.

**Recommendation.** Add cross-parameter validation enforcing the natural
ordering $\textftol \le \textstag \le \tau_{\mathrm{OSGS}}$ and minimum
budgets.

#### 4.3 The cascade does not export partial results on failure

Stage-I failure at `solve_system` ends the run without writing fields.
For long-running simulations, exporting the best-seen state with a
`_failed.vtu` suffix would dramatically help post-mortem diagnosis. The
necessary state (`best_x` inside `_safe_solve_inner!`) is already kept in
memory; only the wiring to `export_results` is missing.

#### 4.4 The `Picard` Jacobian is a single hard-coded variant

The Picard linearisation in this codebase drops $\partial(u\cdot\nabla u)/
\partial u$ and $\partial\sigma/\partial u$ entirely. This is the
"oldest" Picard variant. There is a richer literature on partial
linearisations:

- **Oseen iteration**: keep the convective coefficient frozen at $u^k$
  but include the cross-derivative $\partial u/\partial u^k$. This is the
  current variant.
- **Streamline Upwind Picard**: include SUPG-like stabilization in the
  Picard Jacobian. Improves convergence for high-Re.
- **Stabilized Defect Correction**: alternate between a stabilized Jacobian
  for the linear solve and an unstabilized residual. Faster for some
  classes of problems.

The current code commits to one variant globally. A configurable choice
would let users tune for their problem class.

---

### 5. The MMS plateau verifier (Algorithm D)

#### 5.1 Per-norm floors do not scale with $h$

The floors $\varepsilon_{u,L^2}$, $\varepsilon_{u,H^1}$, $\varepsilon_{p,L^2}$
are static constants. On fine meshes the FE error itself drops below the
floors, and the relative-change ratio becomes meaningless. The plateau
test then either trivially passes (the floors dominate every denominator)
or oscillates (the floors are crossed in some iterations and not others).

**Recommendation.** Make the floors $h$-dependent: $\varepsilon_\bullet =
\varepsilon^{\mathrm{base}}_\bullet \cdot h^{p_\bullet}$ where
$p_{L^2} = k+1$ and $p_{H^1} = k$. This tracks the discretisation scale
naturally.

#### 5.2 The plateau test is not rate-aware

A genuine plateau verification asks two questions: *(i)* has the iterate
stopped moving in the FE norm? *(ii)* is the FE error at the expected
$O(h^{k+1})$ level? The current test only addresses (i). It will declare
a plateau even when the iterate stalls at a discretisation-error level
that is much *larger* than expected (e.g., due to under-quadrature or
boundary-layer under-resolution).

**Recommendation.** Add a secondary check: the plateau is valid only if
$E_{u,L^2}^m \le C_{\mathrm{rate}} \cdot h^{k+1}$ for a configurable
constant $C_{\mathrm{rate}}$. If this fails, report "plateau reached at
suboptimal rate" rather than success.

---

### 6. Recommended priority

Ordered by impact on the user's stated criteria (robustness and
correctness, with cheating-free convergence rates):

| # | Item | Impact | Effort |
|---|---|---|---|
| 1 | Fix the Armijo line search for Picard steps (§1.1) | Major. Affects every case where Picard globalization is required. | Medium. Two-line change to use mode-aware `D`, plus testing. |
| 2 | Add Picard fallback inside the OSGS inner Newton (§1.3) | Major. Closes the OSGS robustness gap. | Medium. Wrap the inner call in `PicardFallback`. |
| 3 | Add $(\mathrm{Re}, \mathrm{Da})$ continuation in `run_simulation` (§2.1) | Major. Enables previously-unreachable cases. | High. New homotopy driver. |
| 4 | Fix the OSGS short-circuit's interaction with `Mode_stop = "both"` (§1.2) | High. Silent false-convergence bug. | Low. Two-line guard. |
| 5 | Loosen Picard's `ftol` (§1.4) | High. Wastes iterations; can cause the Picard cap to fire spuriously. | Low. Add new config field. |
| 6 | Switch mass matrices to Cholesky (§3.2) | Medium. Speed and stability. | Low. One-line change once `CholeskySolver()` is wired. |
| 7 | Block-equilibrated merit (§3.1) | Medium. Affects MMS pressure rates. | Medium. Requires block index map. |
| 8 | Anderson regularization + restart (§3.3) | Medium. Closes silent failure modes. | Low. Add `+ λI` and `empty!` calls. |
| 9 | Initialize $\boldsymbol{\pi}_h^0 = 0$ (§1.5) | Medium. Simplifies and removes warmup-coupling artefacts. | Low. Replace 12 lines with one. |
| 10 | $h$-scaling MMS floors (§5.1) | Medium-low. Affects MMS rate plots. | Low. |
| 11 | Best-state retention across cascade stages (§2.3) | Low-medium. Marginal robustness. | Low. |
| 12 | Cross-parameter validation (§4.2) | Low. Helps users catch config errors. | Low. |
| 13 | Pressure-mean removal in projection (§3.6) | Low. Cleaner $d_\pi^m$. | Low. |
| 14 | Adaptive quadrature for Forchheimer (§3.5) | Low. Marginal MMS rate cleanup. | Low. |
| 15 | Configurable Picard linearisation variants (§4.4) | Low. Research-flexibility. | Medium-high. |
| 16 | Partial export on failure (§4.3) | Low. Quality-of-life. | Low. |

The top six items are, in my judgement, the ones that most directly limit
the cascade's ability to converge every viable case with optimal rates.
Items 1–5 each address an underlying theoretical mismatch or a missing
piece of the standard FEM nonlinear-solver toolkit. Item 6 is a clean
factorization-quality improvement.

---

### 7. Things the cascade gets right

For balance, a partial list of choices that align well with FEM-solver
best practice:

- **Diagonal equilibration of the merit function.** The right idea even
  though the diagonal-only implementation could be sharpened (§3.1).
- **Best-state retention inside `SafeNewtonSolver`.** A correct and
  cheap defence against transient overshooting.
- **The asymmetric `IterationCap` vs `NoiseFloor` classification.** The
  false-equivalency failsafe (§9 of the .tex) is theoretically essential
  for honest MMS rate reporting and is implemented correctly.
- **Block-isolated Anderson on the mass-weighted inner product.** This is
  the principled choice for the OSGS projection fixed point. (The
  hardcoded safety factor and lack of restart, §3.3, are details on top
  of the right structure.)
- **Pre-factored mass matrices, reused across the OSGS loop.** Correct
  cost amortization. (Could be Cholesky instead of LU, §3.2.)
- **Projection onto unconstrained spaces $V_{\mathrm{free}}, Q_{\mathrm{free}}$.**
  Avoids the boundary residual that would otherwise destroy MMS rates.
  This is one of the cascade's most important correct choices.
- **Two distinct $L^\infty$ and $L^2$ drift modes, with the $L^2$ mode as
  the recommended default.** The right physics-respecting metric for
  mixed velocity-pressure systems.
- **A globally consistent equal-order Lagrangian pair + VMS stabilization
  rather than ad-hoc inf-sup-stable pairs.** Standard, correct.
- **Strict configuration validation that rejects unknown keys and
  enforces parameter ranges.** A small thing that catches countless
  user-configuration errors that would otherwise turn into silent
  miscomputations.

These choices are why the cascade works as well as it does. The
criticisms in §§1–4 are about going from "works for most cases" to
"works for every viable case with optimal rates."

---

### 8. Closing thought

The cascade is a textbook-faithful implementation of a Newton ↔ Picard
↔ OSGS stack with sensible safeguards. Its weakness is not in any of the
individual algorithms but in the *seams* between them: the Picard
line-search inheriting Newton's directional derivative; the OSGS inner
Newton lacking the Picard fallback that Stage I has; the OSGS bootstrap
injecting a state that the warmup phase must then unwind; the absence of
parameter continuation as a higher-level fallback. Closing these seams
— particularly items 1–5 of the priority list — would substantially
expand the set of cases for which the cascade converges without manual
intervention, which is what the user has asked for.

---

## Part 2 — Improvement plan

**Companion to** [algorithm-critical-analysis.md](algorithm-improvement.md).

This plan translates each finding of the critical analysis into a concrete
code change, sequenced so that each phase is self-contained: it can be
implemented, tested against Blitz / Quick / Extended, and merged before the
next phase begins. Phases are ordered by a combination of (a) impact on
robustness, (b) dependency between items, and (c) likelihood of shifting MMS
convergence rates — phases that *will* shift numbers come together, so that
re-baselining only happens once.

The numbering matches the section numbers of the critical-analysis report
so cross-reference is easy.

### Strategy and ordering

Twelve items, grouped into six phases:

| Phase | Theme | Items |
|---|---|---|
| **P1** | Pure bug fixes (no behaviour shift on passing cases) | §1.2, §1.5 |
| **P2** | Numerical-correctness, isolated | §3.2, §3.3 |
| **P3** | Picard line-search semantics | §1.1 |
| **P4** | OSGS inner robustness | §1.3, §1.4 |
| **P5** | Convergence-rate-affecting changes (batched, re-baseline once) | §3.1, §3.5, §3.6, §5.1, §5.2 |
| **P6** | Higher-level robustness (continuation) | §2.1, §2.3, §2.4 |
| **P7** | Quality-of-life | §3.4, §4.2, §4.3, §4.4 |

**Why this order.**

- **P1 first** because both items are silent-bug fixes that should not
  change passing tests but will fix specific failure modes (false
  convergence in OSGS, and warmup-phase artefacts). Closing them first
  removes confounders for later phases.
- **P2 second** because Cholesky and Anderson hardening are isolated and
  improve the noise floor of every downstream test.
- **P3 before P4** because the OSGS Picard fallback (§1.3) inherits the
  same line-search logic, so we want the Picard line search correct
  before adding more Picard call sites.
- **P5 batched** because each item affects MMS rates individually; bundling
  them lets you re-run the convergence sweep once instead of five times.
- **P6 late** because the continuation driver is the largest change and
  should sit on top of an already-robust Stage I.
- **P7 last** because none of these items affect correctness; they just
  make the codebase easier to use and debug.

For each item below: **What** (the change), **Where** (files and lines),
**How** (the concrete edit), and **Verify** (which test tier to run and
what to look for).

---

### Phase 1 — Silent-bug fixes

#### Item §1.2 — Fix the OSGS short-circuit's `Mode_stop` interaction

**What.** The unconditional override
([porous_solver.jl:531-534](../../src/solvers/porous_solver.jl#L531-L534))
can declare convergence while the projection is still drifting in
`Mode_stop = "both"` mode.

**Where.** [src/solvers/porous_solver.jl:531-534](../../src/solvers/porous_solver.jl#L531-L534).

**How.**

```julia
if x_diff_inf <= stagnation_tol && osgs_iter >= 2
    if stab_cfg.osgs_stopping_mode == "state_drift"
        overall_converged = true
    end
end
```

That is: the short-circuit fires only in the mode where state drift is
the binding criterion. For `"projection_drift"` and `"both"` the regular
predicate stands. (Optionally: drop the short-circuit entirely; the
`max(τ, stag)` floors in the regular predicate already provide noise-floor
defence. I recommend keeping it but gated.)

**Verify.**
1. Run Blitz: no behaviour change expected.
2. Run an OSGS MMS sweep at a stiff `(Re, Da)` point in `"both"` mode and
   inspect the `outer_osgs_diagnostics`. Before the fix: any case where
   `pi_u_drift > τ_proj` while `x_diff_inf < stag` declared converged.
   After the fix: those cases run to either genuine convergence or
   `max_osgs_iters`.

#### Item §1.5 — Initialise `π_h^0 = 0` and let the first iteration do the bootstrap

**What.** Bootstrapping `π_h^0` from the ASGS state injects an
off-distribution projection that the warmup phase then has to unwind.
Replacing it with `π_h^0 = 0` makes the cascade a clean ASGS → OSGS
continuation.

**Where.** [src/solvers/porous_solver.jl:399-412](../../src/solvers/porous_solver.jl#L399-L412).

**How.** Replace the explicit `R_u`, `R_p`, `b_u`, `b_p`, `solve!` block
with

```julia
pi_u = zero(FEFunction(U_proj, zeros(num_free_dofs(U_proj))))
pi_p = zero(FEFunction(P_proj, zeros(num_free_dofs(P_proj))))
```

or whatever the idiomatic zero-construction is for your `Q_proj` /
`U_proj`. The first iteration of the outer loop will then re-evaluate
`U_h` against `π_h = 0` (which is exactly the ASGS problem, already
converged) and produce the first non-trivial projection naturally.

**Verify.**
1. Run Quick on an OSGS test. Inner-Newton iteration count at `m = 1`
   should drop to roughly zero (the ASGS state already satisfies
   `π_h = 0` so Newton converges in one step).
2. Run an OSGS MMS sweep; the converged solution must be identical (up
   to round-off). If it isn't, the rest of the warmup logic has been
   silently relying on the off-distribution bootstrap and you'll see a
   drift you need to investigate.
3. With `m_warmup = 0` in the config, the result should now also be
   identical — i.e., the warmup phase becomes mathematically redundant.
   Confirm this by running with `m_warmup = 0` and comparing against the
   default.

---

### Phase 2 — Numerical infrastructure

#### Item §3.2 — Switch mass-matrix factorization to Cholesky

**Where.** [src/solvers/porous_solver.jl:374-379](../../src/solvers/porous_solver.jl#L374-L379).

**How.** Replace `LUSolver()` with whatever Cholesky wrapper Gridap
exposes. If Gridap doesn't expose one directly, drop into
`LinearAlgebra.cholesky` against the assembled sparse matrix and wrap in a
custom `LinearSolver`. The wrapper needs to honor Gridap's
`symbolic_setup` / `numerical_setup` / `solve!` protocol.

A reasonable implementation sketch:

```julia
using LinearAlgebra: cholesky, Cholesky
using SparseArrays

struct CholeskySolver <: LinearSolver end
struct CholeskySymbolicSetup <: SymbolicSetup end
struct CholeskyNumericalSetup{T} <: NumericalSetup
    fac::T
end

Gridap.Algebra.symbolic_setup(::CholeskySolver, ::AbstractMatrix) = CholeskySymbolicSetup()
function Gridap.Algebra.numerical_setup(::CholeskySymbolicSetup, A::AbstractMatrix)
    CholeskyNumericalSetup(cholesky(Symmetric(A)))
end
Gridap.Algebra.solve!(x, ns::CholeskyNumericalSetup, b) = (x .= ns.fac \ b; x)
```

(Adapt to your Gridap version's exact interface.)

**Verify.**
1. Run Quick. No change in projection drift magnitudes (Cholesky and LU
   produce the same solution up to round-off).
2. Profile the OSGS outer loop on a `n=80` mesh; the projection step
   should be measurably faster.
3. Run on a deliberately ill-conditioned mesh (large aspect-ratio cells)
   and confirm that Cholesky does not fail (it would only fail on a
   genuinely non-SPD matrix, which would indicate an assembly bug).

#### Item §3.3 — Anderson regularization, restart, and configurable safety factor

**Where.** [src/solvers/accelerators.jl](../../src/solvers/accelerators.jl).

**How.** Three independent fixes:

1. **Configurable safety factor.** Move the hard-coded `10.0` to a new
   field of `AcceleratorConfig` in [src/config.jl](../../src/config.jl):

   ```julia
   Base.@kwdef struct AcceleratorConfig
       type::String
       m::Int
       relaxation_factor::Float64
       safety_factor::Float64  # NEW
   end
   ```

   Add to schema and `validate!` (require `safety_factor >= 1`).

2. **History restart on safety trigger.** In
   [accelerators.jl:83-85](../../src/solvers/accelerators.jl#L83-L85):

   ```julia
   if norm(x_next .- g_k, Inf) > acc.safety_factor * norm(f_k, Inf)
       empty!(acc.history_X)
       empty!(acc.history_F)
       acc.iter = 0
       return x_k .+ acc.relaxation_factor .* f_k
   end
   ```

3. **Regularized least-squares.** In
   [accelerators.jl:67-69](../../src/solvers/accelerators.jl#L67-L69):

   ```julia
   A_ls = DeltaF' * (acc.M_mat * DeltaF)
   b_ls = DeltaF' * (acc.M_mat * f_k)
   λ = eps(Float64) * (tr(A_ls) / size(A_ls, 1))
   gamma = (A_ls + λ * I) \ b_ls
   ```

   The Tikhonov term is tiny in well-conditioned cases (no observable
   effect on rates) but prevents NaN/Inf when the history is near-rank-
   deficient.

**Verify.**
1. Run an OSGS sweep with Anderson enabled at high `m` (history depth
   5–10) on a difficult case. Before the fix: occasional NaN propagation
   or stalling. After: clean convergence.
2. Run a case where Anderson would historically trigger the safety
   fallback frequently. Confirm history is reset (you can add a
   diagnostic counter).

---

### Phase 3 — Picard line-search correctness

#### Item §1.1 — Mode-aware line search

**What.** Newton uses `D = -2Φ` for the Armijo predicate (correct).
Picard uses the same value (incorrect; the cancellation requires the true
Jacobian). The line search must know which Jacobian fed the step.

**Where.**
- [src/solvers/nonlinear.jl:7-19](../../src/solvers/nonlinear.jl#L7-L19): `SafeNewtonSolver` struct.
- [src/solvers/nonlinear.jl:82-116](../../src/solvers/nonlinear.jl#L82-L116): `eval_armijo_linesearch_pass!`.
- [src/solvers/nonlinear.jl:165-249](../../src/solvers/nonlinear.jl#L165-L249): `_safe_solve_inner!`.
- [src/run_simulation.jl:174,177](../../src/run_simulation.jl#L174-L177): solver construction.

**How.** I recommend **monotone residual** rather than computing the true
$D_P$, because it removes the merit function from Picard's accept/reject
loop entirely and is well-justified in the literature for fixed-point
maps (Bank & Rose, 1980; classical reference).

Add a mode field to the solver:

```julia
struct SafeNewtonSolver <: NonlinearSolver
    ls::LinearSolver
    max_iters::Int
    ...
    mode::Symbol  # :newton or :picard
end
```

Make `eval_armijo_linesearch_pass!` dispatch on `solver.mode`:

```julia
if solver.mode === :newton
    # Existing Armijo test on merit
    if state.phi_x_new <= state.phi_x + solver.c1 * alpha * dir_deriv
        state.ls_success = true; break
    end
else  # :picard
    # Monotone residual condition
    if state.norm_b_new_inf <= (1.0 - solver.c1 * alpha) * state.norm_b_inf
        state.ls_success = true; break
    end
end
```

In `run_simulation.jl`, construct the two solvers with their respective
modes:

```julia
nls_picard = SafeNewtonSolver(p_ls, ..., :picard)
nls_newton = SafeNewtonSolver(p_ls, ..., :newton)
```

Same for the OSGS-local solver at
[porous_solver.jl:425](../../src/solvers/porous_solver.jl#L425).

**Verify.**
1. Run Blitz: no behaviour change expected (Picard alone is rarely the
   binding constraint in unit tests).
2. Run the MMS sweep on a deliberately stiff case (high Re *and* high Da)
   where Stage I previously hit `linesearch_failed` on the second-Newton
   retry. Before the fix: `r = "linesearch_failed"` on Picard step.
   After: Picard runs to its iteration cap or hands off cleanly.
3. Inspect `inner_osgs_diagnostics` and confirm that Picard-induced
   merit fluctuations no longer trigger the safeguard's
   `"merit_divergence_escaped"` exit.

---

### Phase 4 — OSGS inner robustness

#### Item §1.4 — Loosen Picard's `ftol`

**Where.** [src/config.jl:58-84](../../src/config.jl#L58-L84), [src/run_simulation.jl:174](../../src/run_simulation.jl#L174).

**How.** Add a field to `SolverConfig`:

```julia
picard_handoff_ftol::Float64  # NEW
```

Validate `picard_handoff_ftol >= ftol`. In `run_simulation.jl`:

```julia
nls_picard = SafeNewtonSolver(p_ls, sol_cfg.picard_iterations,
                              sol_cfg.max_increases, sol_cfg.xtol,
                              sol_cfg.picard_handoff_ftol,  # not ftol
                              ...)
```

Default in `base_config.json`: `picard_handoff_ftol = 1e-3` (i.e., loose).

**Verify.**
1. Run a previously-difficult case. The Picard pass should exit with
   `"ftol_reached"` after a small number of iterations and the
   subsequent Newton retry should converge in 3–6 steps.
2. Run a case where Picard previously hit `picard_iterations` cap;
   confirm it now exits via `ftol_reached` instead.

#### Item §1.3 — Picard fallback inside OSGS inner Newton

**Where.** [src/solvers/porous_solver.jl:425-463](../../src/solvers/porous_solver.jl#L425-L463).

**How.** Extract the Stage-I Newton → Picard → Newton cascade into a
reusable function:

```julia
function picard_fallback_solve!(x, op_newton, op_picard,
                                solver_newton, solver_picard,
                                x_backup)
    # Attempt Newton
    try
        res = solve!(x, solver_newton, op_newton)
        cache = res isa Tuple ? res[2] : res
        nls = cache isa Tuple ? cache[2] : cache
        if nls.result.stop_reason in ("ftol_reached", "initial_ftol",
                                       "stagnation_noise_floor_reached")
            return true, nls
        end
    catch e
        # fall through
    end

    # Restore and try Picard
    get_free_dof_values(x) .= x_backup
    try
        res = solve!(x, solver_picard, op_picard)
        ...
    catch e
        ...
    end

    if any(!isfinite, get_free_dof_values(x))
        get_free_dof_values(x) .= x_backup
        return false, nothing
    end

    # Re-engage Newton
    try
        res = solve!(x, solver_newton, op_newton)
        ...
    catch e
        return false, nothing
    end
end
```

Replace the OSGS inner-Newton call at
[porous_solver.jl:435-463](../../src/solvers/porous_solver.jl#L435-L463) with
a call to `picard_fallback_solve!`. The cascade then uses the same
robustness machinery at every layer.

You'll need to construct an OSGS-local Picard solver alongside the
existing OSGS-local Newton solver
([porous_solver.jl:425](../../src/solvers/porous_solver.jl#L425)). The
Picard Jacobian for OSGS is the same as the Stage-I Picard Jacobian but
with `pi_u`, `pi_p` passed in (rather than `nothing`):

```julia
jac_picard_osgs(x, dx, y) = build_picard_jacobian(x, dx, y, setup,
                            formulation, phys_cfg;
                            pi_u=pi_u, pi_p=pi_p)
op_picard_osgs = FEOperator(res_fn_osgs, jac_picard_osgs, X, Y)
```

**Verify.**
1. Run a stiff OSGS case that previously aborted with
   `"OSGS Inner Newton sweep failed algebraically"`. After the change,
   the outer loop should continue.
2. Confirm that on easy cases there is no regression — Newton converges
   first time, Picard never runs.

---

### Phase 5 — Convergence-rate-affecting changes (re-baseline once)

This phase bundles five items that each will shift MMS error magnitudes
slightly. Do them together so you re-baseline the convergence sweep once.

#### Item §3.1 — Block-equilibrated merit function

**Where.** [src/solvers/nonlinear.jl:172-202](../../src/solvers/nonlinear.jl#L172-L202).

**How.** The current code computes `w` from the global Jacobian diagonal.
Change it to compute `w_u`, `w_p` separately and combine. You'll need
access to the velocity/pressure DOF index ranges. Gridap exposes this
through the `MultiFieldFESpace` (`get_dof_to_field` or equivalent).

Pseudocode:

```julia
u_dofs, p_dofs = field_dof_ranges(X)  # build once outside the loop
...
d_A = diag(A)
d_A_u = view(d_A, u_dofs)
d_A_p = view(d_A, p_dofs)
w_u_max = max(eps(Float64), norm(d_A_u, Inf) * eps(Float64))
w_p_max = max(eps(Float64), norm(d_A_p, Inf) * eps(Float64))
w[u_dofs] .= max.(abs.(d_A_u), w_u_max)
w[p_dofs] .= max.(abs.(d_A_p), w_p_max)
```

(Plus the absolute `eps(Float64)` clamp from the original.)

This respects the block structure of the saddle-point system.

#### Item §3.5 — Adaptive quadrature for Forchheimer

**Where.** Wherever `get_quadrature_degree` is defined (likely
[src/formulations/](../../src/formulations/)).

**How.** Bump the degree by `floor(k/2)` for the Forchheimer reaction
case. For linear elements this is a no-op; for cubic and above this adds
a few quadrature points per cell.

Document the trade-off in the function's docstring.

#### Item §3.6 — Pressure mean removal in projection — **DEFERRED**

**Status.** Excluded from the Phase 5 batch by user direction during planning;
the deferral with rationale, regime dependence, two implementation options
(post-hoc mean removal vs. constrained projection space), and re-evaluation
triggers is captured in
[paper-code-divergences.md §6](paper-code-divergences.md). When this item
is taken up again, that ledger entry is the canonical source for what
needs to happen, not the recipe below.

**Where.** [src/solvers/porous_solver.jl:485-488](../../src/solvers/porous_solver.jl#L485-L488)
(the `discrete_l2_projection` call for `pi_p_next`).

**How (Option A — plan's original recipe; cheap, post-hoc).** After
computing `pi_p_next`, subtract its $L^2$ mean:

```julia
pi_p_next_mean = sum(∫(pi_p_next)dΩ) / sum(∫(1.0)dΩ)
pi_p_next = pi_p_next - pi_p_next_mean  # FE-function subtraction
```

This removes the constant mode that would otherwise pollute `d_π^m`.

Optional but recommended: do the same for the bootstrap projection (or
the new `π_h^0 = 0` if Phase 1 is done — in which case nothing to do).

**Caveats applying when this is eventually landed.**

1. Option A as above is only paper-compatible in the **all-Dirichlet
   velocity** regime; it would *degrade* paper compatibility under mixed
   BCs (open outlet) where the continuous pressure space is full $L^2$.
   A real implementation must therefore branch on the configured Dirichlet
   tags and only apply mean removal when all velocity DOFs are Dirichlet-
   constrained at the domain boundary.
2. **Option B** (project onto `Q_free` $\ominus \{1\}$ explicitly via a
   Lagrange multiplier on the Gram matrix) is the rigorous form and is
   the recommended path if/when this item is reopened. See
   [paper-code-divergences.md §6](paper-code-divergences.md).
3. Do NOT apply mean removal to the pressure variable itself
   (`final_x0`). The $\varepsilon > 0$ gauge fixing in the mass equation
   is a softly-enforced feedback; mean-removing the pressure each
   iteration would fight it. §3.6 modifies only the *projection*
   $\pi_h^p$.

#### Item §5.1 — `h`-scaling MMS floors

**Where.** The MMS config-handling code (search for `eps_u_l2`,
`eps_p_l2`, `eps_u_h1` use sites in
[src/solvers/porous_solver.jl](../../src/solvers/porous_solver.jl)).

**How.** Currently the floors are static constants. Replace them with
`h`-scaled values:

```julia
h_local = 1.0 / partition_n  # or extracted from setup
ε_u_l2_eff = mms_cfg.eps_u_l2 * h_local^(kv + 1)
ε_p_l2_eff = mms_cfg.eps_p_l2 * h_local^(kv + 1)
ε_u_h1_eff = mms_cfg.eps_u_h1 * h_local^kv
```

Then use `ε_*_eff` in the `r_u2`, `r_u1`, `r_p2` denominators. The config
field becomes a *baseline* floor that gets multiplied by the expected
discretisation scale.

#### Item §5.2 — Rate-aware plateau verification

**Where.** Same code as §5.1.

**How.** Add a sanity check after the plateau-pass count test:

```julia
## After: if pass_count >= mms_cfg.require_consecutive_passes ...
if E_u2_k > mms_cfg.rate_check_factor * h_local^(kv + 1)
    diag_cache["mms_stop_reason"] = "mms_plateau_at_suboptimal_rate"
    # Still success, but flagged for inspection
end
```

Add `rate_check_factor` to `MMSConfig` (default `100.0`, i.e. up to 2
orders of magnitude above the expected discretisation scale). This
catches the case where the iterate has converged to a non-FE-optimal
state and would otherwise be silently accepted.

**Verify (full Phase 5).**
1. Re-run the MMS convergence sweep. Expect MMS rates to be at least as
   good as before. Pressure rates in particular should *improve*
   slightly because of §3.1 and §3.6.
2. Any test that fails should be investigated — the changes are
   theoretically rate-preserving, so a regression points to either an
   indexing bug in the block-merit code or a missing case in the
   h-scaling.
3. Verify that the `"mms_plateau_at_suboptimal_rate"` flag fires on
   intentionally under-resolved cases (e.g., very high Re on a coarse
   mesh).

---

### Phase 6 — Higher-level robustness

#### Item §2.1 — Automatic `(Re, Da)` continuation — **DEFERRED**

**Status (2026-05-18).** Investigated and deferred without implementation.
The canonical record of the deferral, including the empirical evidence
that overturned the original motivation, the live hypotheses about what
the failing-case symptoms actually mean (Jacobian conditioning vs.
no-nearby-root vs. τ stabilisation breakdown), and the probes (C/D/E)
that would settle which hypothesis is correct, lives in the "Phase 6
§2.1 — DEFERRED" section of
[algorithm-improvement-progress.md](algorithm-improvement.md).
The recipe below is preserved as a starting point if §2.1 is ever
revisited under different evidence (e.g., a case where the failure
pathology is demonstrably a Newton-basin failure that continuation
can fix).

**What.** When Stage I fails, wrap it in a continuation driver that ramps
`Re` and `Da` from tame values to the target.

**Where.** New module `src/solvers/continuation.jl`, called from
[src/solvers/porous_solver.jl:solve_system](../../src/solvers/porous_solver.jl#L140)
on Stage-I failure.

**How.**

```julia
function continuation_solve!(x0, cfg_target, setup, ...)
    # Build ramp
    re_target = effective_reynolds(cfg_target)
    da_target = effective_damkohler(cfg_target)
    n_steps = ceil(Int, max(log10(re_target), log10(da_target)))
    re_steps = exp.(range(0, log(re_target); length = n_steps + 1))
    da_steps = exp.(range(0, log(da_target); length = n_steps + 1))

    for i in 1:(n_steps+1)
        cfg_local = scale_cfg(cfg_target, re_steps[i], da_steps[i])
        ok, x = try_stage_I(x0, cfg_local, setup, ...)
        if !ok
            return false, x0  # report failure at this ramp step
        end
        x0 = x  # warm-start next ramp step
    end
    return true, x0
end
```

The exact "effective Re/Da" extraction depends on how the config encodes
flow conditions; you may need to scale `phys_cfg.nu` (inverse Re) and the
reaction-law constants (Da) directly.

Add a new field `solver.continuation_enabled::Bool` (default `true`) and
`solver.continuation_max_steps::Int` (default `8`).

Trigger the driver from `solve_system` when:
- Stage I returns `success == false`
- `sol_cfg.continuation_enabled` is true
- We have not already entered the continuation loop (prevent infinite
  recursion)

**Verify.**
1. Run a deliberately-stiff case from `x0 = 0` that previously failed.
   With continuation enabled, expect convergence (possibly after several
   ramp steps).
2. Run an easy case and confirm continuation does not trigger (the
   driver only activates on Stage-I failure).
3. Run the MMS sweep with continuation enabled. Cases that were
   previously borderline (lower-right corner of the (Re, Da) grid)
   should now converge reliably.

#### Item §2.3 — Best-state retention across cascade stages

**Where.** [src/solvers/porous_solver.jl:169-246](../../src/solvers/porous_solver.jl#L169-L246).

**How.** After each cascade stage (initial Newton, Picard, second
Newton), record:

```julia
if final_res < best_residual_so_far
    best_residual_so_far = final_res
    best_x0 .= get_free_dof_values(x0)
end
```

After Stage I, if `success == false` but
`best_residual_so_far < some_threshold`, restore `x0 .= best_x0` and
log a "Stage-I returned best-seen state" message. The threshold should
be a multiple of `stagnation_noise_floor`.

#### Item §2.4 — Symmetric `!isfinite` guard

**Where.** [src/solvers/porous_solver.jl:218-220](../../src/solvers/porous_solver.jl#L218-L220)
(already exists for Picard), should be replicated after each Newton
attempt.

**How.** Wrap each `solve!` call in a check:

```julia
if any(!isfinite, get_free_dof_values(x0))
    get_free_dof_values(x0) .= x0_backup
    newton_success = false
end
```

Apply at all three sites (initial Newton, Picard, second Newton).

**Verify (full Phase 6).**
1. Run the MMS sweep with continuation enabled at very high
   `(Re, Da) = (10^6, 10^6)`. Previously: most cases at the edge of
   the grid failed. After: at least 80% of these cases should converge,
   though the time-to-solution will be 5–10× longer than easy cases.
2. Run a deliberately divergent case to verify the `!isfinite` guards
   prevent runaway. Expected behaviour: clean abort with diagnostics,
   not NaN propagation through the pipeline.

---

### Phase 7 — Quality-of-life

These items are low-risk and low-effort. I recommend doing them in a
single PR.

#### Item §3.4 — Element-wise `freeze_jacobian_cusp`

**Where.** [src/stabilization/tau.jl:18-50](../../src/stabilization/tau.jl#L18-L50).

**How.** Currently `freeze_cusp` is a global Boolean. Make it an
element-wise predicate:

```julia
function compute_dtau_1_du(kin, med, du_val, ν, c_1, c_2, tau_reg_lim,
                            freeze_cusp_threshold, rxn_law)
    if norm(kin.u_v) < freeze_cusp_threshold
        return zero_term(...)
    end
    # else: full derivative
end
```

Replace the config field with `freeze_cusp_threshold::Float64`, defaulting
to a small velocity scale (e.g., `1e-6` in non-dimensional units). This
restores Newton's quadratic rate on the bulk of the domain while
preserving robustness near stagnation regions.

#### Item §4.2 — Cross-parameter validation

**Where.** [src/config.jl:118-150](../../src/config.jl#L118-L150) (`validate!`).

**How.** Add assertions:

```julia
@assert sol.ftol <= sol.stagnation_noise_floor "ftol must be <= stagnation_noise_floor"
@assert stab.osgs_tolerance >= sol.ftol "osgs_tolerance must be >= ftol"
@assert stab.osgs_projection_tolerance >= sol.ftol "osgs_projection_tolerance must be >= ftol"
@assert sol.picard_iterations >= 5 "picard_iterations must be >= 5 (Picard is a fallback, not a solver)"
@assert sol.newton_iterations >= 3 "newton_iterations must be >= 3"
@assert sol.picard_handoff_ftol >= sol.ftol "picard_handoff_ftol must be >= ftol"
```

(The last one assumes §1.4 has been done.)

#### Item §4.3 — Partial export on failure

**Where.** [src/run_simulation.jl:185-198](../../src/run_simulation.jl#L185-L198).

**How.**

```julia
success, final_x0, iters, eval_time = solve_system(...)
if !success
    @warn "Simulation failed; exporting best-seen state as *_failed.vtu"
    export_results(cfg, model, u_h, p_h; suffix="_failed")
    return u_h, p_h, model, Ω, dΩ
end
export_results(cfg, model, u_h, p_h)
```

`export_results` should accept the optional `suffix` keyword.

#### Item §4.4 — Configurable Picard linearisation variants

**Where.** New abstract type `PicardVariant` in
[src/formulations/](../../src/formulations/), with subtypes
`OseenPicard`, `FullyFrozenPicard` (current default), `DefectCorrectionPicard`.

**How.** This is the largest of the QoL items. It's a research-flexibility
feature, not a correctness fix. I'd defer it to a future iteration unless
you specifically need to compare variants.

**Verify (Phase 7).**
1. Run the full suite (Blitz, Quick, Extended) to confirm no regressions.
2. Try a deliberately-invalid config (e.g., `osgs_tolerance < ftol`) and
   confirm `validate!` rejects it with a clear message.
3. Manually fail a run and confirm `_failed.vtu` is produced with the
   best-seen state.

---

### Cross-cutting concerns

#### Documentation

After each phase: update
[theory/osgs_algorithm.tex](../../theory/osgs_algorithm.tex) to reflect any structural
change. Specifically:

- Phase 1: update Algorithm C bootstrap and short-circuit text.
- Phase 3: add a note to Algorithm A.2 distinguishing Newton vs Picard
  line search.
- Phase 4: update Algorithm C to embed PicardFallback inside the inner
  Newton, and add `picard_handoff_ftol` to `tab:params`.
- Phase 5: update Algorithm D for `h`-scaling floors and rate check.
- Phase 6: add Algorithm O' (the continuation wrapper) before
  Algorithm O, and discuss in §3.

#### Testing protocol

Each phase ends with the same test cadence:

1. **Blitz** (5s/file): must pass with no new failures or tier warnings.
2. **Quick** (≤ 2min/file): must pass; investigate any new tier warnings.
3. **Extended** at a representative `(Re, Da)`: confirm convergence and
   record iteration counts (for cross-phase regression detection).
4. **MMS sweep**: optional, but mandatory after Phase 5. Compare rates
   against the previous baseline (one row per `(Re, Da, n)` cell).

Maintain a per-phase results log (a `phaseN_results.csv` in
`results/improvement_phases/`) so cross-phase regressions are easy to
diagnose.

#### Suggested commit/PR pattern

One PR per phase, with the phase number in the title (e.g.
`feat(solver): phase 1 — silent-bug fixes`). The PR description
references the relevant `algorithm-critical-analysis.md` items and
includes:

- Summary of changes (1–3 sentences).
- The acceptance criteria from this plan.
- The actual Blitz/Quick/Extended results.
- For Phase 5: side-by-side MMS rate tables.

#### Lessons-learned ledger

Per [CLAUDE.md](../../CLAUDE.md): append to
[docs/lessons_learned.md](../../docs/lessons_learned.md) whenever a change
fixes a previous regression or surfaces a non-obvious invariant. Phase 1
and Phase 4 in particular are likely candidates.

---

### Summary timing estimate

If you tackle one phase per week with normal coverage and one MMS sweep
per phase (more for Phase 5):

| Phase | Estimated effort |
|---|---|
| P1 | 1 day |
| P2 | 1–2 days |
| P3 | 2–3 days |
| P4 | 2–3 days |
| P5 | 4–5 days (mostly re-baselining) |
| P6 | 5–7 days |
| P7 | 1–2 days |

Roughly three to four weeks total, well-paced. The single most
impactful day is probably the one that fixes §1.1 (the Picard line
search). The single most impactful week is Phase 6 (continuation).

---

### What to defer or skip

A few items in the original critique are deliberately *not* in this plan:

- **§4.4 — Configurable Picard variants**: research flexibility, not
  correctness. Defer unless you have an active research need.
- **§3.5 — Adaptive quadrature**: included in P5 but only as a `floor(k/2)`
  bump. A more elaborate scheme is over-engineering for now.
- **Anderson-with-Type-II / NEPv variants**: out of scope. The current
  Anderson with the P2 fixes is sufficient.

These can be revisited if Phases 1–6 prove insufficient for some specific
case class, but I would not invest in them up-front.

---

## Part 3 — Progress & status

**Companion to** [algorithm-improvement-plan.md](algorithm-improvement.md).

This file tracks which plan items have landed, what verification was done,
what state is pending, and the recommended order for resuming work in a
future session. Update it after each phase commit.

Last updated: **2026-05-19**.

---

### What has landed (this session, 2026-05-17)

| Commit | Plan item | Files | Verification |
|---|---|---|---|
| `dc83e28` | (none — pure plumbing) | `src/solvers/porous_solver.jl` | Blitz 32/32, probe_k1 shows real residuals |
| `dfb8470` | Phase 1 — §1.2 + §1.5 | `src/solvers/porous_solver.jl`, `predictions_small_test.md` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `1caa633` | Phase 3 — §1.1 | `src/solvers/nonlinear.jl`, `src/run_simulation.jl`, 3 test files | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `9c4e1db` | Phase 4 part 1 — §1.4 | `src/config.jl`, `porous_ns.schema.json`, `base_config.json`, `src/run_simulation.jl` | Blitz 32/32, Quick 10/10 |
| `110e0d7` | Phase 4 part 2 — §1.3 | `src/solvers/porous_solver.jl` | Blitz 32/32, Quick 10/10 |
| `e702c16` | Phase 2 — §3.3 Anderson hardening | `src/solvers/accelerators.jl`, `src/solvers/porous_solver.jl`, `src/config.jl`, `porous_ns.schema.json`, `base_config.json` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `41c55ec` | Phase 6 part 1 — §2.4 `!isfinite` guards | `src/solvers/porous_solver.jl` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `dc8f302` | Phase 5 — §3.6 deferral (docs only) | `theory/paper-code-divergences.md`, plan, progress | — |
| `a307bcd` | Phase 5 part 1 — §3.5 reaction-law-aware quadrature | `src/models/reaction.jl`, `src/formulations/continuous_problem.jl`, `src/run_simulation.jl`, 2 test runners | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `f72c819` | Phase 5 part 2 — §3.1 block-equilibrated merit | `src/solvers/nonlinear.jl` | Blitz 32/32, Quick 10/10, probe_k1 state bit-identical (Φ values differ as expected, line search never fires here) |
| `af0bf07` | Phase 5 part 3 — §5.1 h-scaled MMS plateau floors | `src/solvers/porous_solver.jl`, `test/extended/ManufacturedSolutions/run_test.jl` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `af05173` | Phase 5 part 4 — §5.2 rate-aware plateau verification | `src/solvers/porous_solver.jl`, `test/extended/ManufacturedSolutions/run_test.jl` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |

### What has landed (next session, 2026-05-19) — Phase 6 §2.1 *resolution*

| Commit (pending) | Plan item | Files | Verification |
|---|---|---|---|
| (uncommitted) | Phase 6 §2.1 — diagnostic harness + Fix 1 (Picard safeguard) + Fix 2 (dynamic Newton budget) | `src/solvers/nonlinear.jl` (Picard divergence safeguard tracks `‖R‖_∞`), `src/config.jl` + `base_config.json` (new `dynamic_newton_re_threshold`/`dynamic_newton_re_iterations` fields), `test/extended/ManufacturedSolutions/run_test.jl` (wire dynamic Newton budget), `test/extended/ManufacturedSolutions/probe_stiff_diagnose.jl` (new diagnostic script), `test/extended/ManufacturedSolutions/data/probe_stiff_failing.json` (new probe config), `test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md` + `probe_stiff_raw.md` (write-up) | Blitz 32/32, Quick 10/10 bit-identical (ASGS L2 3.12525176e-02, OSGS L2 3.39545818e-02, proj norm ~1.5e-14), probe_k1 bit-identical (iters=[2,1], residuals=[6.87e-5, 1.85e-4], u_L2/p_L2 match), probe_stiff_diag.json 2×2×2 grid now **8/8** (was 7/8 — the historically failing `(Re=1e6,Da=1,α=0.05)` cell converges, `‖R‖_∞ = 5.47e-4` at iter 91), probe_stiff_failing.json now ✅ converges (was ❌ all 7 attempts fail) |

### What has landed (next session, 2026-05-20) — external-audit response

External AI raised 11 paper-vs-code claims (theory/Code Audit Findings.md). Triage in
`/Users/guillermocasasgonzalez/.claude/plans/please-read-carefully-the-silly-lark.md` —
verdicts: 1 real math defect (3D viscous adjoint), 4 real software defects, 1 naming
fix, 2 doc/test traceability gaps, 3 misreads.

| Commit (pending) | Plan item | Files | Verification |
|---|---|---|---|
| (uncommitted) | Audit-response Fix 4b (eps_val floor), Fix 5 (OSGS budget), Fix 6 (MMS plateau split, 2 commits), Fix 7 (strict config loader), Fix 8 (rename) — **all in one bundle since they're all paper-vs-code-honesty fixes** | **Code**: `src/formulations/continuous_problem.jl` (drop `eps_floor` default + clamp; validate `eps_val ≥ 0`); `src/solvers/porous_solver.jl` (drop silent `max(osgs_iters, newton_iters+5)` expansion; split `solve_system` return to `(solver_success, mms_plateau_success::Union{Bool,Nothing}, …)`; P-003 code-comment back-pointer at L496); `src/stabilization/projection.jl` (rename `ProjectResidualWithoutMassPenalty` → `ProjectResidualWithoutPressurePenalty`); `src/config.jl` (rename `load_config_with_defaults` → `load_config_with_base_template`; flip `load_config` shim to strict `load_frozen_config`; drop `eps_floor` field); `src/run_simulation.jl` (use strict loader); `src/formulations/viscous_operators.jl` (Fix 4a: adjoint reuses `EvalDivDevSymOp(Δ(v), ∇∇(v))` / `EvalStrongViscSymOp(Δ(v), ∇∇(v))` — paper-faithful in any `d`). **Schema/configs**: `porous_ns.schema.json` (synced with struct: added missing `dynamic_picard_*`, `dynamic_newton_re_*`, `dynamic_ftol_*`, `condition_noise_floor_*`, `osgs_inner_newton_iters`, `osgs_projection_tolerance`, `osgs_stopping_mode`, `osgs_state_drift_scale`; removed `eps_floor`); `base_config.json` (dropped `eps_floor`); 13 MMS / probe / Cocquet configs (bumped `osgs_iterations` to previous-effective `max(osgs, newton+5)` to preserve numerical baseline). **Tests**: new `test/blitz/config_strict_loader_blitz_test.jl` (2 subtests locking the strict-loader contract); `test/blitz/tau_blitz_test.jl` (P-001/P-008 invariance subtests — `τ₂` independent of `eps_val`, `τ₁` independent of `∇α`); `test/blitz/projection_blitz_test.jl` (rename); `test/quick/osgs_orthogonality_quick_test.jl` (uses `load_config_with_base_template` explicitly, osgs_iterations=25); `test/quick/formulation_smoke_quick_test.jl` (drop `eps_floor` from inline phys_cfg); `test/extended/ManufacturedSolutions/run_test.jl` (caller-logic migration: receive + record `mms_plateau_success` + `overall_verification_success`); `test/extended/ManufacturedSolutions/parse_test.jl` (unpacking signature); `test/extended/ManufacturedSolutions/probe_stiff_diagnose.jl` (constructor signature); `test/extended/CocquetExperiment/run_convergence.jl` (unpacking signature). **Docs**: `theory/paper-code-divergences.md` (new §4b for simplified τ forms with paper-line refs + worst-case `h\|∇α\|/α` bound + re-evaluation triggers; rewritten §2 for viscous adjoint now exact in any d); `theory/osgs_algorithm.tex` (stopping-mode short-circuit gated to `state_drift` in both prose and Algorithm C pseudocode). | Blitz **37/37** (was 32; +2 strict-loader, +3 tau invariance). Quick **10/10**. probe_k1 bit-identical (u_L2=[1.902, 1.254e-3], p_L2=[2.63e-3, 4.065e-5]). Orthogonality smoke: OSGS proj norm 1.61e-14 ✓; OSGS L2 shifted `3.39545818e-02 → 3.33665493e-02` (-1.7%) because the test uses `SymmetricGradientViscosity` where the now-restored `0.5·∇(∇·v)` adjoint term is non-zero in 2D — paper-correct, not a regression. MMS sweep `DeviatoricSymmetricViscosity` 2D unaffected (`(0.5 − 1/d) = 0` in 2D). |

#### Headline impact — paper-faithful 3D viscous adjoint + no-silent-defaults config

The audit-response bundle closes five real defects (one mathematical, four
software/configuration) and adds three regression-anchor tests. The big-picture
shifts:

1. **Formal viscous adjoint is now exact in any dimension** (Fix 4a). Before:
   `adjoint_viscous_operator` for `DeviatoricSymmetric` and `SymmetricGradient`
   variants dropped the `∇(∇·v)` contribution, which is paper-faithful in 2D for
   Deviatoric only (`(0.5 − 1/d) = 0`) but incomplete in 3D for Deviatoric
   (`+1/6`) and in any dimension for SymmetricGradient (`+0.5`). After: the
   adjoint reuses the same dimension-aware `EvalDivDevSymOp` / `EvalStrongViscSymOp`
   machinery the strong operator already uses, so `L*V` matches
   [article.tex:479](../../theory/paper/article.tex#L479) exactly.

2. **No silent defaults in the PDE** (Fix 4b). The `eps_val = max(eps_val, 1e-8)`
   constructor floor that silently changed `ε = 0` into `ε = 10⁻⁸` is removed.
   The `eps_floor` config field is gone from the schema, struct, and `base_config.json`.

3. **No silent OSGS budget expansion** (Fix 5). The `max(stab.osgs_iterations,
   sol.newton_iterations + 5)` clamp is gone. Each MMS/probe/Cocquet config now
   declares an explicit `osgs_iterations` value matching its previous-effective
   budget — so the baseline is bit-identical but the policy is now auditable.

4. **MMS plateau success vs solver success are now separated** (Fix 6). `solve_system`
   returns `(solver_success, mms_plateau_success::Union{Bool,Nothing}, …)`. The MMS
   sweep records both per-cell and computes `overall_verification_success`
   accordingly. Previously a budget-exhausted MMS plateau would still record
   `success = true`; now it records `solver_success = true, mms_plateau_success = false`
   and prints a `[⚠]` flag.

5. **`load_config` is strict** (Fix 7). The production single-file loader no longer
   merges with `base_config.json`. Template-based test configs use the explicitly
   named `load_config_with_base_template` (`osgs_orthogonality_quick_test.jl` is the
   one such consumer; MMS sweeps build configs at runtime via `load_config_from_dict`
   which is unchanged). The `porous_ns.schema.json` is now in 3-way agreement with
   `SolverConfig`/`StabilizationConfig` + `base_config.json` for every field.

6. **Misleading type name fixed** (Fix 8). `ProjectResidualWithoutMassPenalty` →
   `ProjectResidualWithoutPressurePenalty`. The policy strips `ε·p` from the
   pressure-side projection — that's a pressure-equation gauge term, not a reaction
   term.

7. **Documentation traceability** (Doc 1, 3a, 3b, 3c).
   [paper-code-divergences.md §4b](paper-code-divergences.md) traces the simplified
   `eq:Tau1Final` / `eq:Tau2Final` forms used in `compute_tau_1` / `compute_tau_2`,
   records the empirical well-resolved-porosity bound (worst-case `h|∇α|/α ≈ 0.5`
   at the coarsest mesh, well within `eq:SmallPorosityGradient`), and gives explicit
   re-evaluation triggers. [paper-code-divergences.md §2](paper-code-divergences.md)
   is rewritten to reflect the post-Fix-4a state. A blitz regression test locks
   in both simplifications (`τ₂` invariant under `eps_val`, `τ₁` invariant under
   `∇α`) so a future audit reaches the divergence-ledger note and fails the test
   rather than re-raising the same finding. `osgs_algorithm.tex` §"Choosing the
   stopping mode" + Algorithm C pseudocode now state correctly that the ℓ∞
   short-circuit is gated to `state_drift` mode only.

### What has landed (next session, 2026-05-20 cont.) — Gemini-review follow-ups #4 + #2

External Gemini review (2026-05-20) raised four additional points. The user
vetoed #1 (`run_simulation.jl` dynamic budgets — non-trivial generalization,
extra params). #3 (pressure-gauge §3.6) stays deferred per `paper-code-divergences.md
§6` and Tier 1 results. #4 (cascade refactor) and #2 (Cholesky) landed here.

| Commit (pending) | Plan item | Files | Verification |
|---|---|---|---|
| (uncommitted) | Gemini #4 — extract cascade boilerplate into `safe_fe_solve!` | [src/solvers/porous_solver.jl](../../src/solvers/porous_solver.jl) — new `safe_fe_solve!` helper (~50 LoC) above `solve_system`, 6 call sites converted (Stage I Newton 1, Stage I Picard, Stage I Newton 2 retry; OSGS inner Newton 1, OSGS Picard fallback, OSGS Newton 2 retry). The duplicated try/catch + tuple-unwrap + non-finite-DOF guard + max-iters-exception classification is now in one place; the call-site-specific success/failure criteria and structured diagnostic recording stay inline at each site because they legitimately differ. [theory/osgs_algorithm.tex](../../theory/osgs_algorithm.tex) — Algorithm A3 signature gains `\|b\|_∞` and mode `\mathfrak{m}`; the merit-divergence rule is rewritten mode-aware (Newton: `\hat\Phi > F_div Φ`; Picard: `\|\hat b\|_∞ > F_div \|b\|_∞`); the "Picard mode invalidates the identity" paragraph after §"directional derivative" extends to A3 with an explicit reference to `probe_stiff_findings.md` 2026-05-19; new `dynamic_newton_re_*` row added to the test-harness parameter table. | Blitz **37/37**. Quick **10/10**, OSGS L2 **3.33665493e-02** bit-identical to post-Fix-4a baseline, proj norm 1.61e-14 ✓. probe_k1 bit-identical (u_L2 = [1.902, 1.254e-3], p_L2 = [2.63e-3, 4.065e-5]). |
| (uncommitted) | Gemini #2 — Cholesky for OSGS mass-matrix factorization (Phase 2 §3.2) | [src/solvers/linear_solvers.jl](../../src/solvers/linear_solvers.jl) — new `CholeskySolver <: Gridap.Algebra.LinearSolver` with `symbolic_setup`/`numerical_setup`/`numerical_setup!`/`solve!` methods that dispatch through Julia's `cholesky(Symmetric(M))` → CHOLMOD for sparse SPD, LAPACK POTRF for dense. The in-place `numerical_setup!` uses CHOLMOD's `cholesky!` fast-path with a try/catch fallback to a fresh factorization if the symbolic structure changed. `solve!` uses `x .= fac \ b` because CHOLMOD doesn't implement the 2-arg `ldiv!` that `LinearAlgebra.ldiv!(x, A, b)` would dispatch to internally. [src/solvers/porous_solver.jl](../../src/solvers/porous_solver.jl) — `LUSolver()` replaced with `CholeskySolver()` at the OSGS mass-matrix setup ([porous_solver.jl:478](../../src/solvers/porous_solver.jl#L478)) and at the convenience overload [`discrete_l2_projection`](../../src/solvers/porous_solver.jl#L83). | Blitz **37/37**. Quick **10/10**, ASGS L2 and OSGS L2 bit-identical to displayed precision; OSGS proj norm shifted machine-epsilon-level (`1.61e-14 → 1.76e-14`) as expected from a different SPD factorization. probe_k1 bit-identical (u_L2/p_L2 to 4 sig figs). |

#### Headline impact

**#4 (cascade refactor)** removed 6× duplicated `try`/`catch` + tuple-unwrap +
non-finite-DOF guard + max-iters-exception classification. The control-flow
semantics at each call site (Stage I vs OSGS, with different success criteria
and diagnostic recording) remain inline — that variation isn't accidental
duplication, it's the actual call-site logic. The refactor reduces the line
count, makes the boilerplate auditable in one place, and gives future readers
a single function to test. Bit-identical verification on Blitz + Quick + probe_k1
brings the regression risk to ~0.

**#2 (Cholesky)** brings the L² projection mass-matrix factorization into
mathematical alignment with the fact that `M_u, M_p` are SPD. The previous
`LUSolver()` (UMFPack with partial pivoting) did extra work and could introduce
asymmetric permutation. The Cholesky path is faster (no pivoting tree) and
mathematically honest. The numerical change is at machine-epsilon level
(visible only in the projection norm at the 1e-14 scale) — way below the FE
error budget, so all L2/H1 rate measurements are unaffected.

**Algorithm document is now in sync with the fa8aaec divergence-safeguard fix.**
The .tex previously described Algorithm A3 with an unconditional `\hat\Phi >
F_div Φ` divergence test. After `fa8aaec` the code branched this by mode; the
algorithm doc only described the prose of "Picard mode invalidates the
identity" as it applies to A.2 (line search), missing the parallel logic for
A.3 (safeguard arbiter). The 2026-05-20 update extends the explanation,
updates the A3 pseudocode signature, and adds the `dynamic_newton_re_*`
parameter row.

#### Items deliberately NOT addressed

- **Gemini #1** (`run_simulation.jl` dynamic budgets): the user vetoed — fix
  requires either a characteristic-Re config field (new schema params) or a
  manual override switch, both of which add complexity. Users running production
  at high effective Re can simply bump `newton_iterations` in their config
  directly.
- **Gemini #3** (pressure-gauge §3.6): stays deferred per
  [paper-code-divergences.md §6](paper-code-divergences.md). Tier 1 sweep
  (running) will reveal whether constant-mode pollution at α₀=0.05 actually
  degrades pressure rates; re-decide after the sweep.

---

### Original audit findings deliberately NOT changed (preserved from 2026-05-20 entry)

#### Audit findings the bundle deliberately does NOT change

- **P-001** (τ₂ missing ε·h²) and **P-008** (τ₁ missing C_α): paper itself
  documents these as deliberate simplifications (`eq:Tau2Final`, `eq:Tau1Final`).
  Code matches the simplified forms. Documented + regression-anchored.
- **P-003** (OSGS treating `xtol_stagnation` / `max_iters_stagnation` as
  progress): the algorithm spec explicitly says these are progress
  ([osgs_algorithm.tex §1.2.4 L1118](../../theory/osgs_algorithm.tex)). Code follows the
  spec. Code-comment back-pointer added at the inner-failure classification
  site so future readers don't re-raise.
- **P-009** (pressure projection on unconstrained `Q_free`): still deferred per
  [paper-code-divergences.md §6](paper-code-divergences.md). Re-evaluation
  triggers documented there have not fired.
- **P-010** (auditor's claim that `ProjectResidualWithoutMassPenalty` was broader
  than the paper's reaction trim): auditor misread the file. The policy was
  always just the documented reaction-term trim — the name was misleading and
  has been fixed by Fix 8.

#### Headline result — Phase 6 §2.1 deferral overturned

The cell the previous session (2026-05-18) ruled "irrecoverable by continuation"
— `(Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS, eps_pert=0)` — now **converges
cleanly** under defaults. ‖R‖_∞ drops 0.215 → 5.47e-4, exits via
`stagnation_noise_floor_reached`, L2 u/p errors 0.127 / 0.025. The H2/H3/H4
diagnostic settled the question:

- **H2 (Jacobian ill-conditioning) ruled out.** Probe D: `cond(J) = 2.72e7` at
  u_ex — well-conditioned. The easy cell at the same α₀=0.05 actually has
  `cond = 2.25e18` (P_c scaling dominates), so cond is not a useful sanity
  criterion in this codebase.
- **H3 (no discrete root) ruled out.** Probe C: Picard makes monotone ‖R‖
  progress; full cascade reaches the root once safeguards stop misfiring.
- **H4 (τ-stabilisation breakdown) partial / secondary.** Probe E shows isolated
  residual-landscape spikes along the Newton line, compatible with
  `SmoothVelocityFloor` / τ-regularisation thresholds — but they don't block
  convergence, just slow it (more line-search backtracking → more Newton iters
  needed).

**Root cause: two compounding solver-safeguard bugs**:

1. **Picard mode's `divergence_merit_factor` safeguard tracks Φ growth.** Φ is
   a Newton-merit metric (`½‖b/w‖²` with `w = diag(J_Newton)`). Across Picard
   iterations the weights `w` shift because the Picard Jacobian's diagonal
   differs from Newton's, making Φ non-stationary even when `‖R‖_∞` shrinks
   monotonically. Fix: in `:picard` mode, compare `‖R‖_∞` growth instead.
   [src/solvers/nonlinear.jl:166-180](../../src/solvers/nonlinear.jl#L166).
2. **`newton_iterations = 20` is too tight at high Re.** With Fix 1 in place,
   the second Newton pass (after Picard fallback) trends down to ‖R‖_∞ ≈ 0.014
   but hits the 20-iter cap. Adding 40+ iterations covers both the transient
   smoothing-out phase (where line-search backtracks through spikes) and the
   quadratic tail. Fix: new `dynamic_newton_re_*` config pair mirroring the
   existing `dynamic_picard_re_*` pattern, default `threshold = 1e4`,
   `iterations = 60`. Low-Re regimes unchanged.

Full write-up with probe data, line-probe landscape, and verdict in
`test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md`.
Raw probe output auto-dumped to `probe_stiff_raw.md` each script run.

#### Headline empirical result

Re-running `small_test_config.json`
after P1 + P3 + §1.4 dropped total wall-clock from **~9.7 hours → ~18 minutes**
(~30× overall). Per-config highlights:

| Config | Pre-P1 OSGS time | Post-P1+P3+§1.4 OSGS time | Speedup |
|---|---|---|---|
| C3 k=3 QUAD | 5776 s | 119 s | **48×** |
| C4 k=1 TRI | 13126 s (3.6 hr) | 265 s | **49×** |
| C5 k=2 TRI | 3650 s (1.0 hr) | 53 s | **69×** |
| C3 k=3 QUAD ASGS | 9559 s (2.7 hr) | 118 s | **81×** |

Errors **bit-identical** across all 12 cases between pre-P1 and post-P4. This
empirically validates §1.5 (`π_h^0 = 0`) in the wall-clock dimension that
matters: the OSGS-on-TRI bootstrap pathology is eliminated.

---

### Verification baselines for regression detection

If a future change is supposed to be rate- and error-preserving, it should
not move these reference values. If it does, investigate before continuing.

#### probe_k1.json (Re=1e-6, Da=1e6, k=1, QUAD, n=[10, 80], OSGS+ASGS)

```
config_1_ASGS / OSGS:
  iters     = [2, 1]
  residuals = [6.87e-5, 1.85e-4]
  u_L2      = [1.90179591, 1.25406420e-03]
  p_L2      = [2.62977746e-03, 4.06508762e-05]
```

ASGS and OSGS must produce identical numbers to ≥ 8 decimal places (constant-σ
trim makes the projection ≈ 0).

#### small_test_config.json finest-mesh u_L2 (post-Phase-5, `af05173`)

| Config | k | Elem | u_L2 (n=160) |
|---|---|---|---|
| C1 | 1 | QUAD | 3.304e-04 |
| C2 | 2 | QUAD | 5.938e-06 |
| C3 | 3 | QUAD | 3.868e-08 |
| C4 | 1 | TRI  | 3.356e-04 |
| C5 | 2 | TRI  | 1.544e-05 |
| C6 | 3 | TRI  | 6.695e-08 |

Bit-identical to the post-Phase-4 baseline across all six configs (only
display precision differs — the underlying HDF5 numbers match to ≥ 4
displayed digits in every case). All 60 (config × mesh × method) entries
in the sweep hit `base_convergence_only` → the Phase 5 plateau verifier
(§5.1, §5.2) was dead code in this Re=1e-6 regime; the §3.1 line-search
merit never fires here (Newton converges in 1–2 iters at α=1 throughout);
and §3.5's Forchheimer quadrature bump is a no-op because the MMS runner
uses `ConstantSigmaLaw`. The Phase 5 changes are therefore *latent* — they
hardened the harness for stiff-regime tests Phase 6 will unlock, without
shifting any visible number in the current regression suite. Wall-clock
~19 minutes (matching the post-P4 baseline within noise).

#### Quick tier orthogonality test

```
ASGS: L2 Velocity Error = 3.12525176e-02
OSGS: L2 Velocity Error = 3.39545818e-02 ; Proj(Subgrid_R) ≈ 1.5e-14
```

---

### Pending work — what's open, in recommended order

#### Phase 2 — Numerical infrastructure (deferred earlier as higher-effort)

| Sub-item | Effort | Risk | Notes |
|---|---|---|---|
| ~~§3.3 Anderson hardening~~ | ~~Small~~ | ~~Low~~ | **Done** in the row above. Three fixes landed: configurable `safety_factor` (was hard-coded `10.0`), history clear on safety trigger, Tikhonov shift on the LS solve. probe_k1 bit-identical confirms Anderson code path (probe_k1 has `type: "Anderson"`, `m: 10`, `β: 0.8`) is unchanged in well-conditioned regime. |
| §3.2 Cholesky factorization | Medium (1 day) | Medium | Requires Gridap LinearSolver interface adaptation; may be a rabbit hole in Gridap 0.18.6. |

§3.3 landed: configurable `safety_factor` field added to `AcceleratorConfig`
following the §1.4 pattern (schema + base_config + validate! assertion
`safety_factor >= 1.0`), history-clearing safety trigger in
[src/solvers/accelerators.jl:88-93](../../src/solvers/accelerators.jl#L88-L93),
and Tikhonov shift $\lambda = \varepsilon_{\mathrm{mach}}\,\mathrm{tr}(A)/n$
on both LS branches in
[src/solvers/accelerators.jl:65-77](../../src/solvers/accelerators.jl#L65-L77).
Default `safety_factor = 10.0` preserves prior behaviour bit-identical.
Theory write-up updated in
[theory/osgs_algorithm.tex §"Block-wise Anderson acceleration"](../../theory/osgs_algorithm.tex)
(Tikhonov paragraph + safety-fallback paragraph rewritten; parameter
table extended with $C_{\mathrm{sf}}^{\mathrm{And}}$).

Picard-side payoff is unverified in the current Re=1e-6 regime (Anderson
extrapolates cleanly without ever tripping the safety check). The hardened
paths will be exercised properly by Phase 6 continuation runs and the
stiff-regime probe config that the progress doc flags as needed.

#### Phase 5 — Rate-affecting batch — **LANDED** (`a307bcd`–`af05173`)

Four of the five planned items landed; §3.6 was explicitly deferred
(documented in [paper-code-divergences.md §6](paper-code-divergences.md)
and the plan's §3.6 entry).

- ~~§3.1 block-equilibrated merit function~~ — landed in `f72c819`.
- ~~§3.5 adaptive quadrature for Forchheimer~~ — landed in `a307bcd`
  via a `min_quadrature_degree(::AbstractReactionLaw, k_v)` trait that
  each reaction law overrides (default 0; Forchheimer adds ⌊k_v/2⌋).
- ~~§3.6 pressure mean removal in projection~~ — **deferred**, see
  [paper-code-divergences.md §6](paper-code-divergences.md).
- ~~§5.1 h-scaling MMS floors~~ — landed in `af0bf07`.
- ~~§5.2 rate-aware plateau verification~~ — landed in `af05173` with
  configurable `mms_rate_check_factor` (default 100×).

Re-baseline observation. small_test_config.json re-ran in ~19 min
(matching pre-P5 wall-clock within noise). All six configs at n=160
produced **bit-identical** u_L2 values vs the post-P4 reference
(see the table above). All 60 (config × mesh × method) cases exited
via `base_convergence_only` — the plateau verifier was never engaged
in this regime, so §5.1 and §5.2 were dead code throughout. §3.1's
line-search merit never fires here (Newton always succeeds in 1–2
iters at α=1), and §3.5's Forchheimer bump is a no-op because the MMS
runner uses `ConstantSigmaLaw` exclusively. The Phase 5 changes are
therefore **latent harness improvements** — they preserve current
behaviour bit-identical, and their effect surfaces only when:

1. Newton-OSGS has to iterate enough times for the merit's velocity-vs-
   pressure bias to matter (stiff regimes Phase 6 unlocks).
2. The plateau verifier is engaged (also stiff-regime / corner cells).
3. A Forchheimer MMS variant is added (currently every MMS config
   builds `ConstantSigmaLaw` via `build_mms_formulation`).

The post-P5 baseline above is the reference against which a future §3.6
commit will be judged for actual numerical improvement. The bridge
floors in small_test_config.json are still in place — they're solver
tolerances, not MMS plateau floors; reverting them is a separate cleanup
that can happen once Phase 6 has built enough confidence to do so safely.

#### Phase 6 — Continuation driver (§2.1, §2.3, ~~§2.4~~)

Biggest remaining phase. Stage-I cascade currently bails on hard divergence;
Phase 6 §2.1 adds a `(Re, Da)` ramp continuation wrapper, §2.3 adds best-state
retention, and §2.4 (the `!isfinite` guards) has now landed — see the
"What has landed" table. §2.4 was promoted from Phase 6 because it's
independent of the continuation driver and "essentially free" per this doc's
"Strongly recommended" section; landing it now means high-Re/high-Da
continuation runs (when §2.1 lands) will fail-cleanly instead of NaN-leaking.

§2.4 covers six guard sites in [src/solvers/porous_solver.jl](../../src/solvers/porous_solver.jl):
three in the Stage I Newton-Picard-Newton cascade (initial Newton, Picard
fallback, second Newton) and three in the OSGS inner cascade (added by §1.3
in `110e0d7`: inner Newton, Picard fallback, inner second Newton). Each
guard checks `any(!isfinite, get_free_dof_values(x))` immediately after
the corresponding `solve!`, restores from the appropriate backup
(`x0_backup` in Stage I, `x_prev` in OSGS inner), and short-circuits the
local control flow to the natural failure branch (fall-through to Picard
for the first Newton sites, `success = false` + `break` elsewhere). One
pre-existing Stage I Picard guard was strengthened to also restore from
`x0_backup`; the §1.3 OSGS Picard guard was already correct and is
unchanged.

Verification in current Re=1e-6 regime is "no regression": Anderson,
Newton, Picard, and OSGS all stay well-conditioned, so the guards never
trip. The defensive value is realised only in the stiff regimes Phase 6
§2.1 will unlock — at which point a deliberately divergent case should
exit with a clean abort and a "produced non-finite state" diagnostic
rather than propagating NaN through `diag_cache` and the MMS pipeline.

#### Phase 6 §2.1 — Continuation driver — **DEFERRAL CONFIRMED; cell unblocked via solver-safeguard fixes (2026-05-19)**

**Status as of 2026-05-19:** §2.1 (Re/Da ramp continuation) remains not landed,
**and the cell that motivated it now converges without it.** Diagnostic probes
C/D/E (deliberately not run on 2026-05-18) settled the live H2/H3/H4 hypotheses
and pointed at two solver-safeguard bugs, not the basin-of-attraction issue
continuation would have addressed.

**Fix 1 — Picard divergence safeguard tracks `‖R‖_∞`** (was Φ growth, a
Newton-merit metric that fluctuates spuriously across Picard iterations as the
Jacobian-diagonal weights shift).
[src/solvers/nonlinear.jl:166-180](../../src/solvers/nonlinear.jl#L166).
Newton-mode bit-identical.

**Fix 2 — Dynamic Newton iteration budget at high Re.** Added
`dynamic_newton_re_threshold` / `dynamic_newton_re_iterations` config pair
mirroring the existing `dynamic_picard_re_*` pattern. Default
`threshold = 10000`, `iterations = 60`. Low-Re regimes unchanged. Wired into
[run_test.jl:485-498](../../test/extended/ManufacturedSolutions/run_test.jl#L485);
non-MMS `run_simulation.jl` not yet plumbed (its `cfg.phys.nu` doesn't expose
characteristic Re).

After both fixes, `probe_stiff_failing.json`
(the exact failing cell, `Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS`)
converges on the first homotopy attempt with `‖R‖_∞ → 5.47e-4`, stop reason
`stagnation_noise_floor_reached`, L2 u/p errors `0.127 / 0.025`. Full
investigation lives in
`test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md`.

§2.1 remains a valid future item *if* a case appears where the **discrete root
exists** AND **no warm-start within Newton's basin can be constructed from the
existing initial-guess machinery** AND **the cell can't be unlocked by larger
solver budgets**. None of those conditions are currently demonstrated in this
codebase.

---

#### Phase 6 §2.1 — DEFERRED (pre-2026-05-19 framing, preserved for context)

The remainder of this section preserves the 2026-05-18 investigation framing for historical reference. The verdict above supersedes it.

Status as of 2026-05-18: not landed; **not justified by current empirical
evidence**.

##### Pre-investigation framing

The plan and earlier progress-doc revisions described §2.1 as "the
highest-impact remaining item" on the grounds that the high-Re/high-Da
corner cells of [test_config.json](../../test/extended/ManufacturedSolutions/data/test_config.json)
would *fail outright* without continuation. That framing was inherited
from the original critical-analysis document; **it was never
empirically tested**.

##### What was actually tested

A focused diagnostic sweep at the asymptotic corner of the
parameter space, executed via four scaffolding configs in
[test/extended/ManufacturedSolutions/data/](../../test/extended/ManufacturedSolutions/data/):

- `probe_stiff.json`
  — single case `(Re=1e6, Da=1e6, α₀=0.05, k=1, n=10, ASGS)`.
- `probe_stiff_diag.json`
  — 2×2×2 isolation grid `(Re, Da, α₀) ∈ {tame, stiff}³` at `k=1, n=10`.
- `probe_stiff_n80.json`
  — h-refinement sweep `n ∈ {40, 80, 160}` on the failing case.
- `probe_stiff_psweep.json`
  — p-refinement sweep `k ∈ {1, 2, 3}` at `n=10` on the failing case.

##### Findings

**The 2×2×2 isolation revealed only ONE genuinely difficult cell:**

| α | Re | Da | Outcome | Required `eps_pert` |
|---|---|---|---|---|
| 0.5 | × | × | × | All 4 cells: ✅ clean, `eps_pert = 0.1` (first attempt) |
| 0.05 | 1 | 1 | ✅ clean | `eps_pert = 0.1` |
| 0.05 | 1e6 | **1** | ❌ ALL 7 attempts fail | nothing works |
| 0.05 | 1 | 1e6 | ✅ clean | `eps_pert = 0.1` |
| 0.05 | 1e6 | 1e6 | ⚠️ noise-floor saturation | `eps_pert = 1e-5` |

The "expected corner cell" `(α=0.05, Re=1e6, Da=1e6)` is **not** the hard
one — it converges, just slowly. The genuinely hard cell is the
*opposite of intuition*: **high Re + narrow channel + LOW Da**. With
`Da = 1` the reaction term `σ ≈ 1e-6` is effectively switched off,
removing the Darcy damping that otherwise stabilises high-Re flow in
the confined geometry.

**Neither h- nor p-refinement rescues this cell:**

| Refinement | k | n | $\|R(u_{\mathrm{ex}})\|_\infty$ at iter 0 | Outcome |
|---|---|---|---|---|
| (baseline) | 1 | 10  | 2.15e-1 | ❌ |
| h-refine | 1 | 40  | 1.13e-2 | ❌ |
| h-refine | 1 | 80  | 1.88e-3 | ❌ |
| h-refine | 1 | 160 | 1.95e-4 | ❌ |
| p-refine | 2 | 10  | 2.18e-3 | ❌ |
| p-refine | 3 | 10  | 1.27e-3 | ❌ |

The FE consistency residual at $u_{\mathrm{ex}}$ shrinks at the expected
$\mathcal{O}(h^{k+1})$ rate, but Newton/Picard cannot find any nearby
discrete root at any combination tried — including `eps_pert = 0`
(exact MMS solution as initial guess) at n=160 where the consistency
residual is only ~$10^{-4}$.

##### Implications for §2.1

The corner cell that motivated §2.1 turns out **not** to be a
basin-of-attraction failure. It's not rescued by:
- Better initial guess (`eps_pert = 0` at any tested resolution fails);
- More mesh resolution (h-sweep up to n=160 fails);
- Higher polynomial order (p-sweep up to k=3 fails).

§2.1 (Re/Da ramp continuation) produces *warm-starts*; the worst-case
warm-start it could possibly deliver is no better than $u_{\mathrm{ex}}$
itself, and $u_{\mathrm{ex}}$ as a warm-start has already been shown to
fail. So §2.1 cannot rescue this case. The general value proposition
of §2.1 (rescuing cases where the *discrete root exists and Newton
basin is just narrow*) is not falsified — but **no such case has been
identified in this codebase**. Every case in `small_test_config.json`
converges trivially; every case in the 2×2×2 isolation grid either
converges trivially or fails for the structural reason below.

§2.1 is therefore deferred until either (a) a case appears that
demonstrably has the basin-failure pathology continuation can fix, or
(b) the codebase is extended beyond MMS validation toward general
engineering use (where exact-solution warm-starts are unavailable and
continuation becomes the only realistic strategy).

##### What this case probably IS — open question

The failure pattern matches one of three live hypotheses, undiscriminated
by the experiments so far:

- **H2 — Jacobian ill-conditioning at the asymptotic corner.** With
  `σ ≈ 1e-6` and `ν ≈ 1e-6`, the discrete momentum block loses its
  dominant diagonal and the per-element Reynolds number is ~$10^5$.
- **H3 — discrete VMS has no fixed point with residual ≤ ftol nearby.**
  The stabilised system's coercivity proof in
  [article.tex](../../theory/paper/article.tex) degrades as `ν, σ → 0`; the discrete
  solution may not exist in the strong sense the test demands.
- **H4 — τ stabilisation breaks down asymptotically.** At
  `Da=1 + Re=1e6` both viscous and reactive contributions to `τ_1`
  vanish, leaving `τ_1 ≈ h/(c_2·|a|)` — pure convective stabilisation.
  Whether this provides sufficient crosswind dissipation as `Re → ∞`
  is exactly the regime the paper's analysis pushes against.

##### Where to look next (probes deliberately NOT run in this session)

Each probe targets one of H2/H3/H4 and would cost ~30 min of scripting
plus < 1 min of compute. A future session that wants to settle the
question should write a standalone Julia script in
[test/extended/ManufacturedSolutions/](../../test/extended/ManufacturedSolutions/)
that mirrors the failing case's FE setup (mesh, spaces, formulation,
$u_{\mathrm{ex}}$ interpolation) and performs:

- **Probe D — Jacobian condition number.** Assemble the exact-Newton
  Jacobian at $u_{\mathrm{ex}}$ via `jacobian!(A, op, x0)` where
  `op = FEOperator(res_fn, jac_newton, X, Y)` and `x0` is the
  interpolated exact solution. Compute `cond(Matrix(A))` or
  `svdvals(Matrix(A))[end] / svdvals(Matrix(A))[1]`. A value $> 10^{14}$
  is a smoking gun for H2.
- **Probe C — Picard-only iteration from $u_{\mathrm{ex}}$.** Construct
  the cascade's Picard `FESolver` (`SafeNewtonSolver(...; mode=:picard)`)
  and run `solve!(x0, fe_solver_picard, op_picard)` from
  $u_{\mathrm{ex}}$. If Picard converges where Newton fails, the
  iteration map has a fixed point and Newton's Jacobian is the obstacle
  (H2 confirmed). If Picard also wanders, the discrete fixed point
  doesn't exist in the contraction-mapping sense (H3 / H4).
- **Probe E — residual landscape line probe.** Compute the Newton
  step `δ = -J\b` at $u_{\mathrm{ex}}$, then evaluate
  `‖R(u_ex + s·δ)‖_∞` for `s ∈ [-1, 1.5]` at ~40 sample points. Plot.
  A clear minimum near `s = 1` indicates a usable Newton direction
  Newton's step-size policy isn't taking; a featureless / oscillatory
  landscape supports H3 (no nearby root) or H4 (stabilisation pathology).

The four probe configs listed above stay committed as scaffolding so
the next session can reuse the exact same `(Re, Da, α₀, n, k)`
combinations.

#### Phase 7 — Quality-of-life

Low-risk, low-effort polish:

- §3.4 element-wise `freeze_jacobian_cusp`
- §4.2 cross-parameter validation (some already added in §1.4)
- §4.3 partial export on failure
- §4.4 configurable Picard variants (research feature; defer)

---

### Bridge state — things a future session must be aware of

#### Floor overrides in `small_test_config.json` are regime-specific

The file currently has:
```
ftol = xtol = 1e-3
dynamic_ftol_ceiling = 1e-3
dynamic_ftol_spatial_safety_factor = 1e-1
stagnation_noise_floor = 1e-3
condition_noise_floor_absolute_min = 1e-4
newton_iterations = 10
```

These are sized for the `Re=1e-6, Da=1e6, P_c ≈ 1e12` regime, where the
machine-residual floor is `eps_mach × P_c ≈ 1e-4`. **Do not copy these
into other configs.** In a milder regime they would make Newton declare
success at the initial guess.

The floors are temporary bridge values pending Phase 5 (§5.1 h-scaling MMS
floors). After P5 lands, the floors should be reverted to base_config
defaults; rates and errors should then be at least as good across all
regimes without per-regime tuning.

A note documenting this lives at
`predictions_small_test.md §0`.

#### Picard-side payoff of §1.1, §1.3, §1.4 is unverified

The current regime keeps Picard from firing (Newton always succeeds), so
the only check on these three items is "doesn't break Newton" (verified).
The actual Picard correctness gains require a stiff config exercising the
Picard fallback paths:

- §1.1: case where Stage-I previously hit `linesearch_failed` on Picard.
- §1.3: case where OSGS inner Newton previously aborted with
  `linesearch_failed` / `merit_divergence_escaped` / `linear_solve_nan`.
- §1.4: case where Picard previously hit `picard_iterations` cap.

The plan suggests "high Re *and* high Da" as the test point. A
`probe_stiff.json` would need regime-specific floors (~`P_c × eps_mach`
for that case's `P_c`). Designing this is a small task; it would slot
naturally into a Phase 6 (continuation) test bench since the continuation
driver targets exactly these cases.

#### Stage I cascade was *not* refactored

§1.3 inlined the Picard fallback at the OSGS inner site rather than
extracting a reusable function. Stage I's existing cascade (Newton →
Picard → Newton at [porous_solver.jl:168-244](../../src/solvers/porous_solver.jl#L168-L244))
is **left untouched** to bound regression risk. A future refactor can
factor both into a shared `picard_fallback_cascade!` helper. Until then,
the cascade logic is duplicated — keep both copies in sync if you change
the cascade semantics.

#### Pre-existing tier warnings (not introduced by this session)

- `formulation_consistency_blitz_test.jl` runs in 5.0–5.3 s (Blitz limit
  is 5.0 s). Should be promoted to Quick eventually.
- `osgs_orthogonality_quick_test.jl` runs in 150–160 s (Quick limit is
  120 s). Should be promoted to Extended eventually.

Neither blocks correctness; both pre-date this session's work.

---

### How to resume

#### Picking the next phase

The plan's own ordering says: P1 → P2 → P3 → P4 → P5 → P6 → P7. We did
P1, P3, P4 (skipping P2 because P3 was the single highest-impact item).
**P2 §3.3 (Anderson hardening) is the natural next move** — small, low-risk,
high-value, and independent of P5/P6.

If you want to push directly to P5 instead, see the "Phase 5" section
above and budget for at least one full MMS sweep re-baseline (~18 min
post-P4, but P5 may shift it).

#### Verification cadence per phase

Per [CLAUDE.md](../../CLAUDE.md):
1. Edits in `src/formulations/`, `src/stabilization/tau.jl`,
   `src/models/reaction.jl`, or `src/solvers/nonlinear.jl` → run **Blitz**.
2. Edits to assembly / residual / Jacobian / solver orchestration → run
   **Quick** after Blitz.
3. MMS-touching or rate-affecting changes → run **Extended** after Quick;
   for Phase 5 specifically, run small_test_config.json and compare
   against the baseline tables above.

#### Useful invocations

```bash
## Blitz
julia -O0 -t 1 test/run_blitz_tests.jl

## Quick
julia --project=. test/run_quick_tests.jl

## probe_k1 (fast regression check, ~1-2 min, exercises the OSGS path)
cd test/extended/ManufacturedSolutions && julia --project=../../.. run_test.jl probe_k1.json

## small_test_config full sweep (~18 min post-P4; expect changes after P5)
cd test/extended/ManufacturedSolutions && julia --project=../../.. run_test.jl small_test_config.json

## Inspect an h5
python3 test/extended/ManufacturedSolutions/print_h5.py test/extended/ManufacturedSolutions/results/<name>.h5
```

The four probe configs (`probe_k1.json`, `probe_k2.json`, `probe_k3.json`,
`probe_k2_tri.json`) in `test/extended/ManufacturedSolutions/data/` are
scaffolding for per-phase regression tests. `probe_k1.json` is the
cheapest (~1-2 min) and exercises both ASGS and OSGS paths; it's the
default fast smoke test.

Four additional stiffness-frontier probes were added in the 2026-05-18
investigation that deferred Phase 6 §2.1:

- `probe_stiff.json` — single stiff cell `(Re=1e6, Da=1e6, α=0.05)`.
- `probe_stiff_diag.json` — 2×2×2 isolation grid over `(Re, Da, α)`.
- `probe_stiff_n80.json` — h-refinement sweep `n ∈ {40, 80, 160}` on the
  failing cell.
- `probe_stiff_psweep.json` — p-refinement sweep `k ∈ {1, 2, 3}` on the
  failing cell.

These are not regression tests — they're scaffolding for future
diagnostic work. See the "Phase 6 §2.1 — DEFERRED" section above for
how to use them and what probes (C/D/E) remain unanswered.

#### Per-commit conventions used this session

- One commit per plan sub-item (or pair, when items are tightly coupled).
- Commit message includes: what changed, why (referencing the plan item),
  verification done, and any caveats (e.g., "Picard-side payoff
  unverified in this regime").
- All commits sign with `Co-Authored-By: Claude Opus 4.7 (1M context)`.

---

### Necessity for the full-range MMS sweep

The full-range MMS test
([test_config.json](../../test/extended/ManufacturedSolutions/data/test_config.json))
sweeps `Re ∈ {1e-6, 1, 1e6}` × `Da ∈ {1e-6, 1, 1e6}` × `α₀ ∈ {1, 0.5, 0.05}`,
`k ∈ {1, 2}`, `n ∈ {10, 20, 40, 80, 160, 320}`, both element types and both
methods — roughly 1300 individual cases spanning nine `(Re, Da)` corners
with `P_c` varying ~12 orders of magnitude. Honest assessment of which
pending items are essential vs optional:

#### Necessary

- ~~**Phase 5 §5.1 + §5.2.**~~ **Done** in `af0bf07` (h-scaling) and
  `af05173` (rate-aware plateau). The plateau verifier is now
  discretization-budget-scaled; bridge floors in `small_test_config.json`
  can be revisited once Phase 6 has been validated.
- ~~**Phase 6 §2.1 (continuation driver).**~~ **Deferred** as of 2026-05-18.
  The "highest-impact remaining item" framing was overturned by direct
  empirical investigation: the corner cell that motivated §2.1
  (`Re=1e6, Da=1, α₀=0.05` — note: low Da, not high Da as originally
  framed) fails for a reason continuation cannot fix — it fails even
  with the exact MMS solution as initial guess, at every tested mesh up
  to n=160, and at every polynomial order up to k=3. See the "Phase 6
  §2.1 — DEFERRED" section above for the full investigation,
  alternative hypotheses, and the C/D/E probes that would settle which
  hypothesis is correct.

#### Strongly recommended (needed for some cells, not all)

- **Phase 5 §3.6 (pressure mean removal).** Constant-mode pollution at
  `α₀ = 0.05` corrupts pressure rate measurements. **Deferred** with full
  rationale in [paper-code-divergences.md §6](paper-code-divergences.md);
  re-evaluation triggers documented there.
- ~~**Phase 6 §2.4 (!isfinite guards).**~~ **Done** in `41c55ec`.
- ~~**Phase 2 §3.3 (Anderson hardening).**~~ **Done** in `e702c16`.

#### Nice to have

- **Phase 2 §3.2 (Cholesky).** Pure speed.
- ~~**Phase 5 §3.1, §3.5.**~~ **Done** in `f72c819` and `a307bcd`.
- **Phase 7 §3.4, §4.3.** Debugging aids.

#### Not necessary

- **Phase 7 §4.4 (Picard variants), §4.2 (extra cross-param validation).**

#### Caveat — Picard payoff is untested

P3 (§1.1) + P4 (§1.3 + §1.4) all assume the Picard branch fires correctly
in stiff cases. The current Re=1e-6 regime keeps Picard idle, so these
paths are verified only as "doesn't break Newton". **The high-Re corners
of the full sweep will be the first time they actually fire.** Budget
time after P5 + P6 to inspect diagnostics there before trusting them.

#### Minimum order to attempt the full sweep

**P5 → P6 §2.1 + §2.4 → P2 §3.3.** Roughly 8–12 days per the plan's own
estimates. Skipping P5 means manual per-regime tuning. Skipping P6's
continuation means corner-cell failures (acceptable if you accept ~80%
coverage). Skipping P2 §3.3 means occasional Anderson stalls.

If "most cells converge with reasonable rates" is the goal, P5 alone
covers ~80%. If "every cell converges within ±0.15 of theoretical rate"
is the goal, P5 + P6 are both essential.

---

### Open questions worth a future session's attention

1. **Should we extract the Picard cascade into a reusable helper?**
   See "Stage I cascade was not refactored" above. Pro: removes
   duplication. Con: touches working Stage I code, regression risk.
   Defer until there's a third call site that would benefit.

2. **Do the bridge floors need to be reverted before P5, or after?**
   They're declared as bridge in `predictions_small_test.md`. Reverting
   before P5 would make small_test_config.json fail (Newton can't reach
   the tight `ftol`). After P5's h-scaling, the base defaults should work.
   So: revert as part of P5.

3. **The k=2 sub-optimal velocity rate (~2.5 instead of 3) seen
   pre-P5.** §5.2 is designed to *flag* this case (`mms_plateau_at_suboptimal_rate`).
   §5.1's h-scaling floors should *fix* it by pushing Newton further
   only when the FE error budget warrants it. Verify both behaviours
   when P5 lands.

4. **Pressure rates are super-optimal** (h^{k+1} instead of h^k) in the
   current data. The plan predicts §3.1 + §3.6 will improve them
   "slightly". If they get worse, that's a regression bug.

---

### Required next steps for robust optimal convergence (post-audit, 2026-05-20)

**Goal.** Reach a state where every cell of the paper's manufactured-solutions
parameter space converges at the theoretical FE rate without manual per-regime
tuning. The audit-response bundle landed the last batch of paper-vs-code
honesty items; what remains is *validation* (running the sweep and seeing what
falls out) and a small set of *coverage extensions* (3D, Forchheimer-σ MMS,
open-outlet MMS) plus the still-deferred §3.6.

#### Tier 1 — Run the full-range sweep on the post-audit baseline (load-bearing)

This is the single largest unknown right now. The full
[test_config.json](../../test/extended/ManufacturedSolutions/data/test_config.json)
sweep (~1300 cases over `Re ∈ {1e-6, 1, 1e6} × Da ∈ {1e-6, 1, 1e6} × α₀ ∈
{1, 0.5, 0.05} × k ∈ {1, 2} × n ∈ {10, 20, 40, 80, 160, 320}`, both element
types, both ASGS/OSGS) has **not** been executed end-to-end since `fa8aaec`
unblocked the stiff cell and the audit-response bundle re-baselined `osgs_iterations`
everywhere.

The output we need is a triage of: (a) cells that converge cleanly at
theoretical rate; (b) cells that converge but at a sub-optimal rate
(`mms_stop_reason = "mms_plateau_at_suboptimal_rate"`); (c) cells that
exhaust the MMS budget (`mms_plateau_success = false`, exposed by Fix 6 Commit 2);
(d) cells that don't converge at all. Without this triage, the items below
are speculative.

**Action.**
```bash
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl test_config.json
```
Expected wall-clock: hours, possibly ~1 day depending on machine — the
`small_test_config.json` sub-sweep takes ~19 minutes, and `test_config.json`
is order-of-magnitude larger.

**Expected output to scrutinise.**
1. The `[⚠]` flag now printed when `solver_success = true` but
   `mms_plateau_success = false` (Fix 6 Commit 2) — count these and
   classify by `(Re, Da, α₀, k)` corner.
2. The rate sub-optimality flag `mms_plateau_at_suboptimal_rate`
   (Phase 5 §5.2) — was previously dead code; will now fire for cells
   that plateau above the FE budget.
3. Convergence-rate plots per `(method, k, etype)` — should show
   `O(h^(kv+1))` velocity L², `O(h^kv)` velocity H¹, `O(h^(kv+1))` pressure
   L² (paper §6). Deviations beyond ±0.15 of theoretical are the failure
   class to fix.

#### Tier 2 — Coverage extensions the paper exercises but the suite doesn't

##### 2a. 3D MMS test bench (unlocks Fix 4a verification)

Fix 4a made the viscous adjoint paper-faithful in 3D, but **the entire MMS
suite is 2D** so the new term `(0.5 − 1/d)·∇(∇·v)` is exercised at coefficient
zero. The fix is theoretically correct and compiles cleanly, but is empirically
unverified.

**Minimum sufficient.**
- A 3D analytical manufactured solution (paper §6 — confirm whether it gives
  an explicit 3D `u_ex(x, y, z)`, `p_ex(x, y, z)`, `α(x, y, z)`; if not,
  construct a polynomial one of comparable structure to the 2D MMS).
- Extend [`Paper2DMMS`](../../src/PorousNSSolver.jl) to a `Paper3DMMS` (or
  generalise) with the corresponding `f`, `g`, `Δu`, `∇∇u` analytics.
- One small 3D config (`probe_3d.json` with `bounding_box` length-6 and
  `partition = [n, n, n]`), small enough to run in `< 5` min.
- A 3D adjoint-identity blitz test would be cheaper still: build a tiny 3D
  Lagrangian space, interpolate smooth polynomial fields with zero trace,
  evaluate `∫ (L_visc u)·v` and `∫ u·(L*_visc v)` numerically, assert
  agreement to quadrature tolerance. **This is the cheapest empirical
  verification of Fix 4a** and should land before any 3D rate sweep.

##### 2b. Forchheimer-σ MMS variant (unlocks Phase 5 §3.5)

Every MMS config currently builds `ConstantSigmaLaw` via `build_mms_formulation`.
The Forchheimer-aware quadrature bump landed in `a307bcd` (`min_quadrature_degree(::ForchheimerErgunLaw, k_v)` adds `⌊k_v/2⌋`) is therefore dead code in
the regression suite. Without exercising it the §3.5 path is unverified for
correctness.

**Minimum sufficient.** A `probe_forchheimer.json` config that swaps the MMS
oracle's σ-law to `ForchheimerErgunLaw`. The oracle's `f` source term must be
re-derived against the new reaction so the MMS is exact; this is a small
analytics change in the oracle module. One k=2 mesh-refinement series
suffices to confirm optimal rates.

##### 2c. Open-outlet MMS variant (unlocks pressure-space regime branch + P-009 re-evaluation)

All current MMS configs are all-Dirichlet on velocity. The paper's continuous
pressure space switches at the BC regime ([article.tex L407](../../theory/paper/article.tex#L407)):
all-Dirichlet → `L²₀`, mixed BCs → `L²`. The deferred §3.6 (pressure mean
removal) is BC-regime-conditional — currently we can't even test the conditional
because no open-outlet MMS exists.

**Minimum sufficient.** A `probe_outlet.json` MMS variant with one boundary
patch as an open outlet. Re-derive the MMS source so the analytical solution
satisfies the outlet condition. If this triggers the previously-undocumented
constant-mode pollution at low `α₀`, that's the re-evaluation trigger for
P-009 / §3.6.

#### Tier 3 — Items that may move rates (already-planned, still open)

These are inherited from the original `algorithm-improvement-plan.md`. Run
Tier 1 first; the sweep results will tell us which are load-bearing.

##### 3a. §3.6 — Pressure mean removal in OSGS projection (DEFERRED)

Constant-mode pollution at small `α₀` may inflate the projection-drift
metric `d_π^m` and corrupt pressure-rate measurements. See
[paper-code-divergences.md §6](paper-code-divergences.md) for the deferred
patch (Option A: post-hoc mean removal; Option B: constrained projection
space). The re-evaluation triggers documented there:

1. A non-all-Dirichlet MMS config appears in the suite (covered by Tier 2c).
2. Post-sweep results show pressure-rate sub-optimality traceable to the
   constant mode (revealed by Tier 1).
3. Narrow-channel runs at `α₀ = 0.05` show corner-cell pressure-rate
   degradation (also revealed by Tier 1).
4. Another mass-side divergence-ledger entry interacts with the pressure
   space (no current pressure).

If Tier 1 shows pressure-rate degradation at `α₀ = 0.05` traceable to the
constant mode, land §3.6 **as Option B with the BC-regime conditional**
(Option A alone would silently degrade Tier 2c).

##### 3b. §2.3 — Best-state retention across cascade stages

[algorithm-improvement-plan.md §2.3](algorithm-improvement.md) — record
`best_residual_so_far` / `best_x0` after each cascade stage, restore on
fall-through failure. Cheap and obviously correct. Helpful for corner cells
where Stage I succeeds in an intermediate Newton pass but a later cascade step
diverges before exit. **Recommended once Tier 1 identifies cells that fail
this way.**

##### 3c. §2.1 — Automatic `(Re, Da)` continuation (deferred → still deferred)

The 2026-05-19 resolution overturned the previous "must-have" framing: the
sole cell that motivated §2.1 now converges without continuation. Keep §2.1
on the long-term ledger; it is justified only if Tier 1 surfaces a corner
cell where (i) the discrete root exists, (ii) no warm-start within Newton's
basin can be constructed from existing initial-guess machinery, AND (iii) the
cell can't be unlocked by larger solver budgets. None of those conditions is
currently demonstrated.

##### 3d. §3.2 — Cholesky factorisation of the mass matrices

[algorithm-improvement-plan.md §3.2](algorithm-improvement.md) — pure
speed (LU on SPD is 2× the work). Does not affect correctness or rates.
Suggested only if profiling after Tier 1 identifies mass-matrix factorisation
as a hot path; given OSGS pre-factorises both matrices once per
mesh ([porous_solver.jl:409-415](../../src/solvers/porous_solver.jl#L409-L415)),
the speed-up is bounded by the number of distinct `n` values in the sweep,
i.e. small.

#### Tier 4 — Empirical anchors for assumptions documented but not stress-tested

##### 4a. 3D viscous-adjoint identity blitz test

Already mentioned in Tier 2a. Single small blitz test:
```julia
## build small 3D Lagrangian space, k_v = 2
## interpolate smooth polynomial u, v with zero trace on ∂Ω
## assert ∫ (L_visc u) ⋅ v ≈ ∫ u ⋅ (L*_visc v) to quadrature tolerance
```
This is the load-bearing empirical check for Fix 4a's correctness.
Recommended **before** any 3D rate sweep — if the identity fails the
sweep results will be uninterpretable.

##### 4b. Steep-porosity-gradient stress test

The simplified `eq:Tau1Final` form is justified by `h|∇α|/α ≲ α` per
[article.tex L765](../../theory/paper/article.tex#L765). The current MMS suite satisfies this
with worst-case `~0.5` (recorded in
[paper-code-divergences.md §4b](paper-code-divergences.md)). A stress probe
with a narrow transition layer (`r₁ = 0.2, r₂ = 0.201` instead of `r₂ = 0.4`)
would violate the assumption and either:
- confirm the simplified form still gives optimal rates (the assumption is
  more permissive than the inequality suggests), or
- show rate degradation that triggers implementing the full `eq:Tau1` with
  `C_α = α + (h/|k_0|)|∇α|` (the `grad_alpha` field is already plumbed in
  `MediumState`, so the switch is local).

This is **not** required for the paper's MMS as written but is the natural
empirical check that pins down where the simplification breaks. Low priority.

##### 4c. Equal-order LBB stability

The paper allows equal-order interpolation `k_v = k_p`. The schema has an
`equal_order_only` field but the current test suite uses Taylor-Hood-like
`k_v > k_p` by default. Equal-order with VMS stabilisation is exactly what
the τ₁/τ₂ machinery is supposed to make work — it should be tested at least
once. **Recommended as a single config (`probe_equal_order_k2.json`).** Low
priority unless Tier 1 surfaces a paper-relevant equal-order claim that's
not covered.

#### Tier 5 — Validation against the paper's Cocquet reference experiment

[test/extended/CocquetExperiment](../../test/extended/CocquetExperiment) exists
but uses `α ≡ 1` (constant porosity), so it doesn't exercise the porous side
of the formulation. Per the audit-response findings the Cocquet driver is
plumbed but the actual paper-faithful Cocquet run hasn't been re-validated
since Phase 5 + the audit-response bundle. A confirmation run with the new
code state would be a useful end-to-end sanity check.

#### Recommended order of execution

1. **Tier 1** — run the full-range sweep, get triage data. **One commit.**
2. **Tier 4a** — 3D adjoint-identity blitz test. **One small commit.**
3. **Tier 2a** — 3D MMS oracle + driver + small rate config. **Largest single
   item.** Skip if no 3D claim needs to be made for the paper.
4. **Tier 2b** — Forchheimer-σ MMS. **One commit.**
5. **Tier 2c** — open-outlet MMS. **One commit.** This is the natural
   precursor to (6).
6. **Tier 3a** — §3.6 pressure mean removal, conditional on BC regime. Land
   only after Tier 2c provides the test config that distinguishes regimes.
7. **Tier 3b** — §2.3 best-state retention. Cheap; can land anytime after
   Tier 1 surfaces the failure mode it addresses.
8. **Tier 5** — Cocquet re-run. **One commit.**
9. **Tier 4b, 4c, Tier 3d** — only if Tier 1 / Tier 5 surface a need; otherwise
   defer indefinitely.

#### Honest assessment of the gap remaining

After this bundle the code is **paper-faithful at the formulation level**
in every respect we know about (tau formulas, viscous adjoint, OSGS projection,
config strictness, MMS plateau semantics). The remaining gap to "robust
optimal convergence across the paper's full parameter range" is largely
**empirical validation work**, not new algorithmic content. The biggest
unknowns are:

- Does the post-audit baseline give optimal rates across all ~1300 cells of
  the full sweep? Unknown until Tier 1 runs.
- Does the paper-faithful 3D adjoint actually integrate correctly in Gridap
  0.18.6 for `k_v = 2` test functions on hex/tet meshes? Unknown until Tier
  4a runs.
- Does the constant-mode pressure pollution at `α₀ = 0.05` actually visibly
  degrade rates? Unknown until Tier 1 + Tier 2c run.

If all three answer favourably with no new defects, the code is in shape.
If any one surfaces a new failure mode, the diagnostic harness built in
2026-05-19 (`probe_stiff_diagnose.jl`) is the template for the next round.

