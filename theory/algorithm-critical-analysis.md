# Critical Analysis of the Stabilized Porous Navier–Stokes Nonlinear Solver

**Scope.** This report evaluates the orchestrated solver cascade documented in
[theory/osgs_algorithm.tex](osgs_algorithm.tex) against the standard
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

## 1. Theoretically problematic items (highest priority)

### 1.1 The line search applies the Newton Gauss-Newton identity to Picard steps

**Where.** [src/solvers/nonlinear.jl:216](../src/solvers/nonlinear.jl#L216),
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

### 1.2 The OSGS short-circuit silently breaks the `Mode_stop = "both"` semantics

**Where.** [src/solvers/porous_solver.jl:531-534](../src/solvers/porous_solver.jl#L531-L534):

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

### 1.3 The OSGS inner Newton has no Picard fallback

**Where.** [src/solvers/porous_solver.jl:449-453](../src/solvers/porous_solver.jl#L449-L453):

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

### 1.4 Picard runs to the same tight `ftol` as Newton

**Where.** [src/run_simulation.jl:174](../src/run_simulation.jl#L174):

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

### 1.5 The bootstrap projection $\boldsymbol{\pi}_h^0$ uses the ASGS state

**Where.** [src/solvers/porous_solver.jl:400-412](../src/solvers/porous_solver.jl#L400-L412):

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

## 2. Robustness gaps

### 2.1 No automatic homotopy in $\mathrm{Re}$, $\mathrm{Da}$, or $\alpha$

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
([test/extended/ManufacturedSolutions/run_test.jl:499](../test/extended/ManufacturedSolutions/run_test.jl#L499))
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

### 2.2 No fallback for `xtol_stagnation` / `max_iters_stagnation`

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

### 2.3 No best-state retention across cascade stages

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

### 2.4 The catastrophic-state guard is asymmetric

After Picard, the code checks
[`any(!isfinite, x0)`](../src/solvers/porous_solver.jl#L218). This same
check is *not* applied after the first Newton attempt. If the linear
solver in Newton produces a non-NaN but Inf-contaminated step (this can
happen with marginally singular matrices and certain BLAS code paths),
the Picard pass starts with garbage.

**Recommendation.** Promote the `any(!isfinite)` check to a wrapper
applied after *every* stage of the cascade.

---

## 3. Numerical-quality concerns (FEM theory)

### 3.1 Merit-function equilibration is diagonal, not block-diagonal

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

### 3.2 Mass matrices are LU-factored even though they are SPD

[src/solvers/porous_solver.jl:374-379](../src/solvers/porous_solver.jl#L374-L379):

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

### 3.3 Anderson's hardcoded safety factor and lack of restart

[src/solvers/accelerators.jl:83](../src/solvers/accelerators.jl#L83):

```julia
if norm(x_next .- g_k, Inf) > 10.0 * norm(f_k, Inf)
```

The factor of `10.0` is a magic number. Worse, when this safety check
fires, the history is *not cleared*: the very next iteration will try the
same near-singular least-squares problem with one more column appended.
The standard Anderson-with-restart pattern would empty `history_X` and
`history_F` on triggering, so the accelerator rebuilds from scratch.

Additionally, the least-squares solve
[`A_ls \ b_ls`](../src/solvers/accelerators.jl#L69) has no
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

### 3.4 `freeze_jacobian_cusp` is opaque

The flag at
[src/stabilization/tau.jl:18-19](../src/stabilization/tau.jl#L18-L19)
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

### 3.5 Quadrature degree is not adaptive to nonlinear terms

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

### 3.6 Pressure is unpinned in the OSGS projection

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

## 4. Architectural observations

### 4.1 OSGS warmup damps the inner Newton but not the projection update

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

### 4.2 No cross-parameter consistency validation

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

### 4.3 The cascade does not export partial results on failure

Stage-I failure at `solve_system` ends the run without writing fields.
For long-running simulations, exporting the best-seen state with a
`_failed.vtu` suffix would dramatically help post-mortem diagnosis. The
necessary state (`best_x` inside `_safe_solve_inner!`) is already kept in
memory; only the wiring to `export_results` is missing.

### 4.4 The `Picard` Jacobian is a single hard-coded variant

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

## 5. The MMS plateau verifier (Algorithm D)

### 5.1 Per-norm floors do not scale with $h$

The floors $\varepsilon_{u,L^2}$, $\varepsilon_{u,H^1}$, $\varepsilon_{p,L^2}$
are static constants. On fine meshes the FE error itself drops below the
floors, and the relative-change ratio becomes meaningless. The plateau
test then either trivially passes (the floors dominate every denominator)
or oscillates (the floors are crossed in some iterations and not others).

**Recommendation.** Make the floors $h$-dependent: $\varepsilon_\bullet =
\varepsilon^{\mathrm{base}}_\bullet \cdot h^{p_\bullet}$ where
$p_{L^2} = k+1$ and $p_{H^1} = k$. This tracks the discretisation scale
naturally.

### 5.2 The plateau test is not rate-aware

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

## 6. Recommended priority

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

## 7. Things the cascade gets right

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

## 8. Closing thought

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
