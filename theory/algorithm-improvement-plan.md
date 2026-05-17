# Implementation Plan — Algorithm Improvements

**Companion to** [algorithm-critical-analysis.md](algorithm-critical-analysis.md).

This plan translates each finding of the critical analysis into a concrete
code change, sequenced so that each phase is self-contained: it can be
implemented, tested against Blitz / Quick / Extended, and merged before the
next phase begins. Phases are ordered by a combination of (a) impact on
robustness, (b) dependency between items, and (c) likelihood of shifting MMS
convergence rates — phases that *will* shift numbers come together, so that
re-baselining only happens once.

The numbering matches the section numbers of the critical-analysis report
so cross-reference is easy.

## Strategy and ordering

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

## Phase 1 — Silent-bug fixes

### Item §1.2 — Fix the OSGS short-circuit's `Mode_stop` interaction

**What.** The unconditional override
([porous_solver.jl:531-534](../src/solvers/porous_solver.jl#L531-L534))
can declare convergence while the projection is still drifting in
`Mode_stop = "both"` mode.

**Where.** [src/solvers/porous_solver.jl:531-534](../src/solvers/porous_solver.jl#L531-L534).

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

### Item §1.5 — Initialise `π_h^0 = 0` and let the first iteration do the bootstrap

**What.** Bootstrapping `π_h^0` from the ASGS state injects an
off-distribution projection that the warmup phase then has to unwind.
Replacing it with `π_h^0 = 0` makes the cascade a clean ASGS → OSGS
continuation.

**Where.** [src/solvers/porous_solver.jl:399-412](../src/solvers/porous_solver.jl#L399-L412).

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

## Phase 2 — Numerical infrastructure

### Item §3.2 — Switch mass-matrix factorization to Cholesky

**Where.** [src/solvers/porous_solver.jl:374-379](../src/solvers/porous_solver.jl#L374-L379).

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

### Item §3.3 — Anderson regularization, restart, and configurable safety factor

**Where.** [src/solvers/accelerators.jl](../src/solvers/accelerators.jl).

**How.** Three independent fixes:

1. **Configurable safety factor.** Move the hard-coded `10.0` to a new
   field of `AcceleratorConfig` in [src/config.jl](../src/config.jl):

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
   [accelerators.jl:83-85](../src/solvers/accelerators.jl#L83-L85):

   ```julia
   if norm(x_next .- g_k, Inf) > acc.safety_factor * norm(f_k, Inf)
       empty!(acc.history_X)
       empty!(acc.history_F)
       acc.iter = 0
       return x_k .+ acc.relaxation_factor .* f_k
   end
   ```

3. **Regularized least-squares.** In
   [accelerators.jl:67-69](../src/solvers/accelerators.jl#L67-L69):

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

## Phase 3 — Picard line-search correctness

### Item §1.1 — Mode-aware line search

**What.** Newton uses `D = -2Φ` for the Armijo predicate (correct).
Picard uses the same value (incorrect; the cancellation requires the true
Jacobian). The line search must know which Jacobian fed the step.

**Where.**
- [src/solvers/nonlinear.jl:7-19](../src/solvers/nonlinear.jl#L7-L19): `SafeNewtonSolver` struct.
- [src/solvers/nonlinear.jl:82-116](../src/solvers/nonlinear.jl#L82-L116): `eval_armijo_linesearch_pass!`.
- [src/solvers/nonlinear.jl:165-249](../src/solvers/nonlinear.jl#L165-L249): `_safe_solve_inner!`.
- [src/run_simulation.jl:174,177](../src/run_simulation.jl#L174-L177): solver construction.

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
[porous_solver.jl:425](../src/solvers/porous_solver.jl#L425).

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

## Phase 4 — OSGS inner robustness

### Item §1.4 — Loosen Picard's `ftol`

**Where.** [src/config.jl:58-84](../src/config.jl#L58-L84), [src/run_simulation.jl:174](../src/run_simulation.jl#L174).

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

### Item §1.3 — Picard fallback inside OSGS inner Newton

**Where.** [src/solvers/porous_solver.jl:425-463](../src/solvers/porous_solver.jl#L425-L463).

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
[porous_solver.jl:435-463](../src/solvers/porous_solver.jl#L435-L463) with
a call to `picard_fallback_solve!`. The cascade then uses the same
robustness machinery at every layer.

You'll need to construct an OSGS-local Picard solver alongside the
existing OSGS-local Newton solver
([porous_solver.jl:425](../src/solvers/porous_solver.jl#L425)). The
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

## Phase 5 — Convergence-rate-affecting changes (re-baseline once)

This phase bundles five items that each will shift MMS error magnitudes
slightly. Do them together so you re-baseline the convergence sweep once.

### Item §3.1 — Block-equilibrated merit function

**Where.** [src/solvers/nonlinear.jl:172-202](../src/solvers/nonlinear.jl#L172-L202).

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

### Item §3.5 — Adaptive quadrature for Forchheimer

**Where.** Wherever `get_quadrature_degree` is defined (likely
[src/formulations/](../src/formulations/)).

**How.** Bump the degree by `floor(k/2)` for the Forchheimer reaction
case. For linear elements this is a no-op; for cubic and above this adds
a few quadrature points per cell.

Document the trade-off in the function's docstring.

### Item §3.6 — Pressure mean removal in projection — **DEFERRED**

**Status.** Excluded from the Phase 5 batch by user direction during planning;
the deferral with rationale, regime dependence, two implementation options
(post-hoc mean removal vs. constrained projection space), and re-evaluation
triggers is captured in
[paper-code-divergences.md §6](paper-code-divergences.md). When this item
is taken up again, that ledger entry is the canonical source for what
needs to happen, not the recipe below.

**Where.** [src/solvers/porous_solver.jl:485-488](../src/solvers/porous_solver.jl#L485-L488)
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

### Item §5.1 — `h`-scaling MMS floors

**Where.** The MMS config-handling code (search for `eps_u_l2`,
`eps_p_l2`, `eps_u_h1` use sites in
[src/solvers/porous_solver.jl](../src/solvers/porous_solver.jl)).

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

### Item §5.2 — Rate-aware plateau verification

**Where.** Same code as §5.1.

**How.** Add a sanity check after the plateau-pass count test:

```julia
# After: if pass_count >= mms_cfg.require_consecutive_passes ...
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

## Phase 6 — Higher-level robustness

### Item §2.1 — Automatic `(Re, Da)` continuation

**What.** When Stage I fails, wrap it in a continuation driver that ramps
`Re` and `Da` from tame values to the target.

**Where.** New module `src/solvers/continuation.jl`, called from
[src/solvers/porous_solver.jl:solve_system](../src/solvers/porous_solver.jl#L140)
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

### Item §2.3 — Best-state retention across cascade stages

**Where.** [src/solvers/porous_solver.jl:169-246](../src/solvers/porous_solver.jl#L169-L246).

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

### Item §2.4 — Symmetric `!isfinite` guard

**Where.** [src/solvers/porous_solver.jl:218-220](../src/solvers/porous_solver.jl#L218-L220)
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

## Phase 7 — Quality-of-life

These items are low-risk and low-effort. I recommend doing them in a
single PR.

### Item §3.4 — Element-wise `freeze_jacobian_cusp`

**Where.** [src/stabilization/tau.jl:18-50](../src/stabilization/tau.jl#L18-L50).

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

### Item §4.2 — Cross-parameter validation

**Where.** [src/config.jl:118-150](../src/config.jl#L118-L150) (`validate!`).

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

### Item §4.3 — Partial export on failure

**Where.** [src/run_simulation.jl:185-198](../src/run_simulation.jl#L185-L198).

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

### Item §4.4 — Configurable Picard linearisation variants

**Where.** New abstract type `PicardVariant` in
[src/formulations/](../src/formulations/), with subtypes
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

## Cross-cutting concerns

### Documentation

After each phase: update
[theory/osgs_algorithm.tex](osgs_algorithm.tex) to reflect any structural
change. Specifically:

- Phase 1: update Algorithm C bootstrap and short-circuit text.
- Phase 3: add a note to Algorithm A.2 distinguishing Newton vs Picard
  line search.
- Phase 4: update Algorithm C to embed PicardFallback inside the inner
  Newton, and add `picard_handoff_ftol` to `tab:params`.
- Phase 5: update Algorithm D for `h`-scaling floors and rate check.
- Phase 6: add Algorithm O' (the continuation wrapper) before
  Algorithm O, and discuss in §3.

### Testing protocol

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

### Suggested commit/PR pattern

One PR per phase, with the phase number in the title (e.g.
`feat(solver): phase 1 — silent-bug fixes`). The PR description
references the relevant `algorithm-critical-analysis.md` items and
includes:

- Summary of changes (1–3 sentences).
- The acceptance criteria from this plan.
- The actual Blitz/Quick/Extended results.
- For Phase 5: side-by-side MMS rate tables.

### Lessons-learned ledger

Per [CLAUDE.md](../CLAUDE.md): append to
[docs/lessons_learned.md](../docs/lessons_learned.md) whenever a change
fixes a previous regression or surfaces a non-obvious invariant. Phase 1
and Phase 4 in particular are likely candidates.

---

## Summary timing estimate

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

## What to defer or skip

A few items in the original critique are deliberately *not* in this plan:

- **§4.4 — Configurable Picard variants**: research flexibility, not
  correctness. Defer unless you have an active research need.
- **§3.5 — Adaptive quadrature**: included in P5 but only as a `floor(k/2)`
  bump. A more elaborate scheme is over-engineering for now.
- **Anderson-with-Type-II / NEPv variants**: out of scope. The current
  Anderson with the P2 fixes is sufficient.

These can be revisited if Phases 1–6 prove insufficient for some specific
case class, but I would not invest in them up-front.
