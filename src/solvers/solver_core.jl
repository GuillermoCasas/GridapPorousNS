# src/solvers/solver_core.jl
#=
    solver_core.jl

# Role
The **shared solver core + the orchestrator** for the Porous Navier-Stokes system — i.e. everything
the two stabilization methods (ASGS, OSGS) have in *common*. It bridges the continuous operators,
stabilized forms, and Jacobians (`viscous_operators.jl`) with the iterative execution (`nonlinear.jl`)
needed to reach discrete convergence, and dispatches to the per-method solves.

This file deliberately names BOTH methods, because it is method-neutral infrastructure: the orchestrator
(`solve_system`) must select between them, and the verifier seam carries one hook per method. The
method-specific algorithm boxes live in their own files:

1. **ASGS (Algebraic Sub-Grid Scale)** — `asgs_solver.jl`: the SGS space is the space of finite element
   residuals; the projection onto it is the identity ``\widetilde{\Pi} = \mathcal{I}``, so the FE
   projection ``\boldsymbol{\pi}_h = \boldsymbol{0}``. Owns the Stage-I cascade `_initialize_asgs_state!`
   and the `STAGE_I_*` cascade policies.
2. **OSGS (Orthogonal Sub-Grid Scale)** — `osgs_solver.jl`: the SGS space is orthogonal to the FE space
   (``\mathcal{X}_{h0}^{\perp}``); it computes the ``L^2``-projection ``\boldsymbol{\pi}_h`` of the
   residual and subtracts it in the sub-scale equation. Owns the coupled solve `solve_osgs_stage!`,
   the L²-projection helpers, and the `OSGS_INNER_POLICY` cascade policy.

# What lives here (the commonalities)
- `FETopology`, `VMSFormulation`, `StageSolvers` — the method-neutral FE/solver containers every driver builds.
- `check_porous_solver_parameters` — validates the (ASGS|OSGS) method selector + solver tolerances.
- The `SolutionVerifier` seam (abstract type + `NoVerification` no-op default + the per-method hooks) —
  the observer the optional MMS plateau verification (Algorithm D, `mms_verification.jl`) plugs into.
- `safe_fe_solve!`, `_solve_one_step!`, `_record_stage!` — the FE-solve plumbing both methods reuse.
- `CascadePolicy` (the vocabulary) + `cascade_step_outcome` (the interpreter) + `_pingpong_cascade!`
  (the adaptive Newton↔Picard scheduler) — the shared Algorithm-B cascade. Each method supplies its
  own `CascadePolicy` *value* (declared in its own file); the *machinery* is shared and lives here.
- `solve_system` — the orchestrator (Algorithm O): Stage-I ASGS boot → method dispatch → optional verification.

# Methodological Background
Following the formulation in the companion article, the Galerkin finite element discretization of the generalized Navier-Stokes equations suffers from LBB condition violations for equal-order spaces and instabilities in convection- or reaction-dominated flows. We use VMS to approximate the unresolved sub-grid scales (SGS) ``\widetilde{U}`` in terms of the resolved scales' residual ``\mathcal{R}(U_h)`` and a local element matrix of stabilization parameters ``\boldsymbol{\tau}_K``.

# Relations to other files
- `src/formulations/viscous_operators.jl`: translates the abstract operators (e.g. ``\mathcal{L}_{\nu}, \mathcal{L}_c, \mathcal{L}_b``) into Gridap closures; this file delegates `eval_strong_residual`, `build_stabilized_weak_form_jacobian`, etc. to it.
- `src/solvers/nonlinear.jl`: the shared `SafeNewtonSolver` kernel (`solver_picard`, `solver_newton`) that converges the ASGS/OSGS algebraic problems, plus `_with_overrides`.
- `src/solvers/asgs_solver.jl`: the ASGS Stage-I cascade (`_initialize_asgs_state!`) + `STAGE_I_*` policies.
- `src/solvers/osgs_solver.jl`: the OSGS L²-projection helpers + coupled solve (`solve_osgs_stage!`) + `OSGS_INNER_POLICY`.

# Algorithm-to-code mapping
See `docs/solver/algorithm-code-mapping.md` for the full table. In short, `solve_system` mirrors
Algorithm O (SimulationOrchestration) of `theory/osgs_algorithm/osgs_algorithm.tex`; it delegates to
`_initialize_asgs_state!` (Stage-I ASGS boot, Algorithm B; in `asgs_solver.jl`) and `solve_osgs_stage!`
(the coupled OSGS solve, Algorithm C; in `osgs_solver.jl`) — plus the shared Newton↔Picard cascade
(`_pingpong_cascade!` / `cascade_step_outcome`, here). The optional MMS plateau verification (Algorithm D)
is decoupled behind the `SolutionVerifier` seam (`on_asgs_converged!` / `on_osgs_converged!`); the
concrete `MMSPlateauVerifier` lives in `src/solvers/mms_verification.jl`, and production uses the no-op
`NoVerification`.

# Include order
Included AFTER `nonlinear.jl`/`accelerators.jl` (it uses `_with_overrides`/`FESolver` in `solve_system`)
and BEFORE `asgs_solver.jl`/`osgs_solver.jl`/`mms_verification.jl` (those name `FETopology`/
`VMSFormulation`/`SolutionVerifier` in function signatures, resolved at parse time). `solve_system`'s
calls to `_initialize_asgs_state!`/`solve_osgs_stage!` are in its BODY, so they resolve at call time —
the two method files may be included after this one.
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

"""
    check_porous_solver_parameters(stab_cfg, sol_cfg)

Enforces strict parameter boundaries for the top-level VMS solver configurations,
preventing unbounded numerical cascades resulting from malformed JSON properties.
"""
function check_porous_solver_parameters(stab_cfg::StabilizationConfig, sol_cfg::SolverConfig)
    if stab_cfg.method != "ASGS" && stab_cfg.method != "OSGS"
        throw(ArgumentError("Stabilization method must be strictly 'ASGS' or 'OSGS'. Passed: $(stab_cfg.method)"))
    end
    if sol_cfg.ftol <= 0.0 || sol_cfg.stagnation_noise_floor <= 0.0
        throw(ArgumentError("Solver outer tolerances (ftol, stagnation_noise_floor) must be strictly positive floats."))
    end
end

# The discrete FE world the solver operates on: the trial/test spaces, the mesh + integration measure,
# the unconstrained spaces used for the OSGS L²-projection, and the data CellFields evaluated in the
# weak form. Every driver builds one of these and threads it through the formulation and solvers.
struct FETopology
    X            # trial MultiFieldFESpace for (u, p) (Dirichlet-constrained)
    Y            # test  MultiFieldFESpace for (v, q)
    model        # CartesianDiscreteModel (QUAD or simplexified TRI)
    Ω            # the triangulation (interior domain)
    dΩ           # Lebesgue measure / quadrature on Ω used by all ∫…dΩ integrals
    V_free       # UNCONSTRAINED velocity space (no Dirichlet) — domain of the OSGS π_h projection
    Q_free       # UNCONSTRAINED pressure space — projecting on the constrained space would inject an
                 #   O(1) boundary residual and destroy the O(h^{k+1}) MMS convergence. [known-fragility]
    h_cf         # element-size CellField h_K, the local length scale feeding τ₁/τ₂
    f_cf         # momentum forcing/source term f as a CellField
    alpha_cf     # porosity field α(x) driving the reaction tensor σ(α, u)
    g_cf         # mass-equation source (divergence forcing) as a CellField
end

# The stabilized weak form plus the equal-order stabilization constants. Built by `build_formulation`.
struct VMSFormulation
    form         # the Gridap weak-form closure (residual/Jacobian assembly delegate)
    c_1          # τ₁ constant (= 4k⁴ for equal-order, paper eq:conditions_on_num_param)
    c_2          # τ₂ constant (= 2k²)
end

# The two algebraic solvers the cascade ping-pongs between (both wrap SafeNewtonSolver kernels).
struct StageSolvers
    picard       # frozen-derivative Picard solve — the robust globalizer
    newton       # Exact-Newton solve (includes ∂τ/∂u, ∂Π/∂u) — the fast local converger
end

# Build the read-only convergence probe (the scale-free criterion of `convergence_criterion.jl`) for one
# cell, capturing its FE topology + formulation. The returned closure is what the inner Newton kernel
# calls once per accepted iteration (see `SafeNewtonSolver.conv_probe`):
#     (x, b, field_blocks) -> (eps_M, eps_C, degenerate)
# `x` is the current free-DOF iterate; `b` is the assembled stabilized residual the kernel already holds,
# so the momentum numerator ‖r_M‖ (Philosophy A) is the Euclidean norm of its velocity block
# (`field_blocks[1]`) — free. The closure reconstructs (uh, ph) from `x`, rebuilds the reaction field
# σ(α, u) exactly as the OSGS solve does, then evaluates the dynamic momentum envelope D_M and the mass
# ratio ε_C at THIS iterate (so the normalisation tracks the iterate, never frozen per stage). It is a
# read-only observer of the iterate (it allocates its own scratch and mutates nothing the solver owns),
# but it IS the authoritative success decision: `solve_system` attaches it UNCONDITIONALLY to both stage
# solvers (see ~L462), so in production AND the harness the converged verdict is ε_M ≤ tol_M ∧ ε_C ≤ tol_C,
# and the per-iteration field assembly (the D_M load vectors + the σ rebuild) is paid on every accepted
# iterate. [future] If that cost matters, gate the attachment behind a config flag (opt-out); the
# `conv_probe === nothing` path then reverts to the scalar-ftol fallback gate.
function build_convergence_probe(setup::FETopology, formulation::VMSFormulation,
                                 tol_M::Float64, tol_C::Float64)
    X        = setup.X
    Vvel     = setup.Y[1]                 # velocity TEST space — same DOFs as b's velocity block
    dΩ       = setup.dΩ
    α        = setup.alpha_cf
    f        = setup.f_cf                 # momentum body force (D_M envelope term)
    g        = setup.g_cf                 # mass source (subtracted in R_p; magnitude enters the ε_C denominator)
    h        = setup.h_cf
    grad_α   = ∇(α)                       # porosity gradient: fixed across iterations, captured once
    form     = formulation.form
    ν        = form.ν
    eps_val  = form.eps_val
    visc_op  = form.viscous_operator
    rxn_law  = form.reaction_law
    reg      = form.regularization
    c_1, c_2 = formulation.c_1, formulation.c_2
    d        = num_cell_dims(setup.model)
    sig_op   = SigOp(rxn_law, reg, ν, c_1, c_2)
    # The closure is the AUTHORITATIVE convergence evaluator the kernel gates on (not merely a
    # diagnostic): it re-assembles the genuine nonlinear residual's velocity block `r_M` at the
    # current iterate and defers the whole verdict (ε_M, ε_C, converged, degenerate) to the criterion
    # module — keeping the convergence RULES (convergence_criterion.jl) separate from the iteration
    # ALGORITHM (nonlinear.jl). Returns the full `ConvergenceMeasure`.
    return function (x, b, field_blocks)
        vblock = field_blocks === nothing ? (firstindex(b):lastindex(b)) : field_blocks[1]
        r_M = norm(view(b, vblock))                       # ‖r_M‖ — velocity block (Philosophy A numerator)
        uh, ph = FEFunction(X, x)
        σ = Operation(sig_op)(uh, ∇(uh), α, grad_α, h)    # σ(α, u) at this iterate
        return evaluate_convergence(r_M, uh, ph, α, ν, visc_op, σ, f, eps_val, g, Vvel, dΩ, d;
                                    tol = tol_M, tol_C = tol_C)
    end
end

# ==============================================================================
# Post-convergence verification seam (observer; production default is a no-op)
# ==============================================================================
# The core solver is verification-blind: at each convergence point it invokes a hook on a
# `SolutionVerifier`. Production passes `NoVerification` (multiple dispatch resolves the hooks to
# no-ops, so it costs nothing). The MMS harness passes an `MMSPlateauVerifier`
# (`src/solvers/mms_verification.jl`) that owns the manufactured-solution oracle and the plateau loop.
# The core never names MMS, reads an oracle, or writes an `mms_*` diagnostic key — that is the
# verifier's concern. This keeps the optional Algorithm-D verification out of the main solver body.
abstract type SolutionVerifier end

"""Production default: the solver runs no post-convergence verification."""
struct NoVerification <: SolutionVerifier end

# The core calls ONLY these three; concrete verifiers (e.g. MMSPlateauVerifier) override them.
#   on_asgs_converged! — ASGS plateaus by extra 1-iteration Newton cycles, so it gets `step_once!`.
#   on_osgs_converged! — the OSGS coupled solve is a single solve (no stepping), so its hook is stateless.
on_asgs_converged!(::NoVerification, x, step_once!, diag, iter_count_ref) = nothing
on_osgs_converged!(::NoVerification, x, diag) = nothing
verification_result(::SolutionVerifier) = nothing   # 2nd return element of solve_system; MMS overrides

"""
    safe_fe_solve!(x, fesolver, op; backup=nothing)

Single-attempt FE-solver wrapper that performs the bookkeeping every cascade
site in `solve_system` shares: tuple unwrapping of Gridap's `solve!` return
value, a non-finite-DOF guard with optional backup restoration, and exception
classification (Gridap's "Reached maximum iterations" string vs other
exceptions).

This helper does not decide success vs failure and does no logging — those
criteria differ across the call sites (Stage-I Newton 1 / Picard / Newton 2,
and OSGS inner Newton 1 / Picard / Newton 2 retry), so each caller pairs the
returned status with its own `CascadePolicy` verdict. The helper's job is to
turn a raw `solve!` into a uniform, finite-checked status `NamedTuple`.

Returns a `NamedTuple` with field `state`:
- `:ok` — finite state; also carries `iterations`, `residual_norm`,
  `initial_residual_norm`, `step_norm`, `stop_reason` from the result cache.
- `:nonfinite` — at least one DOF was non-finite; if `backup !== nothing`,
  the state was restored from `backup` before returning.
- `:max_iters_caught` — `solve!` threw a "Reached max(imum) iterations"
  exception. `SafeNewtonSolver` instead reports `stop_reason =
  "max_iters_stagnation"` on the `:ok` path; this branch is the defence for
  any solver instance that still raises the exception. [debugging-lore]
- `:exception` — `solve!` threw any other exception. Field `exc` holds it.

The backup restoration on `:nonfinite` is the only side effect beyond
`solve!`'s in-place mutation of `x`.
"""
function safe_fe_solve!(x, fesolver, op; backup=nothing)
    try
        res = solve!(x, fesolver, op)
        cache = res isa Tuple ? res[2] : res
        nls_cache = cache isa Tuple ? cache[2] : cache
        if any(!isfinite, get_free_dof_values(x))
            if backup !== nothing
                get_free_dof_values(x) .= backup
            end
            return (state = :nonfinite,)
        end
        r = nls_cache.result
        return (state = :ok,
                iterations = r.iterations,
                residual_norm = r.residual_norm,
                initial_residual_norm = r.initial_residual_norm,
                step_norm = r.step_norm,
                stop_reason = r.stop_reason,
                iteration_history = r.iteration_history,
                initial_residual_normalized = r.initial_residual_normalized,
                residual_normalized = r.residual_normalized)
    catch e
        msg = string(e)
        if occursin("Reached maximum iterations", msg) || occursin("Reached max iterations", msg)
            return (state = :max_iters_caught,)
        else
            return (state = :exception, exc = e)
        end
    end
end

# One raw Newton step on `op`: unwraps Gridap's `solve!` return and returns the result cache
# (`.iterations`, `.residual_norm`, …). Carries no try/catch — a verifier that must tolerate the
# "Reached maximum iterations" exception owns that try/catch itself.
# Used by `on_asgs_converged!` for the plateau loop's single-iteration re-solves.
function _solve_one_step!(x, fesolver, op)
    res = solve!(x, fesolver, op)
    cache = res isa Tuple ? res[2] : res
    nls_cache = cache isa Tuple ? cache[2] : cache
    return nls_cache.result
end

# [trajectory] Append one algorithm-stage record to diag_cache["trajectory"], capturing a case's exact
# path through the nonlinear orchestration (Algorithm O → B/C → A) for later analysis. `stage` is a
# documentation-aligned code (e.g. "B:StageI:N1", "C:OSGS[3]:Picard"). `res` is a `safe_fe_solve!`
# NamedTuple; non-:ok early-return states carry only `state`, so the numeric fields fall back to defaults.
function _record_stage!(diag_cache, stage::String, res)
    haskey(diag_cache, "trajectory") || (diag_cache["trajectory"] = Any[])
    push!(diag_cache["trajectory"], (
        stage       = stage,
        state       = String(res.state),
        stop_reason = get(res, :stop_reason, ""),
        iters       = get(res, :iterations, 0),
        res_in      = get(res, :initial_residual_norm, NaN),
        res_out     = get(res, :residual_norm, NaN),
        # [trajectory] normalized residuals (per-field convergence quantity, ≤1 ⟺ converged); the
        # plotter prefers these and falls back to the inf-norms above when they are absent.
        res_in_norm  = get(res, :initial_residual_normalized, NaN),
        res_out_norm = get(res, :residual_normalized, NaN),
        history     = get(res, :iteration_history, NamedTuple[]),
    ))
end

# ==============================================================================
# Shared cascade success-policy (Algorithm B) — the COMMONALITY between methods
# ==============================================================================
# The ASGS Stage-I cascade (`_initialize_asgs_state!`, asgs_solver.jl) and the coupled OSGS solve
# (`solve_osgs_stage!`, osgs_solver.jl) both reuse this Algorithm-B cascade. Each classifies its own
# stop-reason, which is the *only* place they legitimately differ — expressed as one `CascadePolicy`
# VALUE per method, declared in that method's OWN file (STAGE_I_POLICY/STAGE_I_N2_POLICY in
# asgs_solver.jl; OSGS_INNER_POLICY in osgs_solver.jl). `CascadePolicy` (the shared vocabulary) +
# `cascade_step_outcome` (the shared interpreter) make that per-method difference one explicit decision
# instead of duplicated control flow, and give the Newton↔Picard ping-pong a single reusable verdict
# primitive.
#
# THREE policy flags are required (not two): Stage-I Newton-2 accepts a noise-floor
# finish but rejects a soft stall (xtol/max-iters/no-progress), whereas the OSGS inner
# accepts both — so `accept_noise_floor` and `accept_soft_stall` must be independent.
#
# The full per-method truth table (pinned by test/blitz/cascade_policy_symmetry_blitz_test.jl;
# each row's `const` is defined in the file noted in brackets):
#                       accept_noise_floor | accept_soft_stall | max_iters_caught_is_failure
#   STAGE_I_POLICY      false                false               true   (N1: only ftol → success; else → Picard)             [asgs_solver.jl]
#   STAGE_I_N2_POLICY   true                 false               true   (N2: ftol+noise_floor → success; soft/struct → fail) [asgs_solver.jl]
#   OSGS_INNER_POLICY   true                 true                false  (accept ftol/noise/soft; struct → Picard; max_iters_caught → 1-iter) [osgs_solver.jl]
#
# Verdict vocabulary (caller owns logging, diagnostics, iter accounting, backup restore):
#   :success           — accept this Newton/Picard finish.
#   :one_iter_success  — a `:max_iters_caught` counted as a 1-iteration success (OSGS only).
#   :reject            — not a success; the caller decides (Stage-I N1 → Picard; Stage-I N2 → fail;
#                        OSGS N1 → Picard; OSGS N2 → abort outer loop).
struct CascadePolicy
    accept_noise_floor::Bool
    accept_soft_stall::Bool
    max_iters_caught_is_failure::Bool
end

"""
    cascade_step_outcome(res, policy::CascadePolicy) -> Symbol

Pure verdict for one Algorithm-B cascade segment from a `safe_fe_solve!` result + a
`CascadePolicy`. Returns `:success`, `:one_iter_success`, or `:reject`. The caller keeps
all side effects (logging, diagnostic pushes, iter-count accounting, backup restoration).
"""
function cascade_step_outcome(res, policy::CascadePolicy)
    st = res.state
    if st === :nonfinite || st === :exception
        return :reject
    elseif st === :max_iters_caught
        return policy.max_iters_caught_is_failure ? :reject : :one_iter_success
    else  # :ok — branch on the per-field solver verdict
        sr = res.stop_reason
        if sr == "ftol_reached" || sr == "initial_ftol"
            return :success
        elseif sr == "stagnation_noise_floor_reached"
            return policy.accept_noise_floor ? :success : :reject
        elseif sr == "linesearch_failed" || sr == "merit_divergence_escaped" || sr == "residual_divergence_escaped" || sr == "linear_solve_failed"
            return :reject  # structural failures are never accepted (incl. a failed/non-converged linear solve, C.1)
        else  # xtol_stagnation / max_iters_stagnation / no_progress_stall — a "soft stall"
            return policy.accept_soft_stall ? :success : :reject
        end
    end
end

# ==============================================================================
# _pingpong_cascade!  (adaptive Newton↔Picard scheduling, opt-in)
# ==============================================================================
"""
    _pingpong_cascade!(final_x0, restore_vec, op_newton, op_picard, solver_newton,
                       solver_picard_gain, policy::CascadePolicy, diag_cache, iter_count_ref;
                       stage_prefix::String, max_swaps::Int) -> Symbol

Adaptive Newton↔Picard ping-pong shared by Stage I (ASGS) and the OSGS inner cascade. Runs Newton
until it stalls (a non-success `cascade_step_outcome` under `policy`), then a Picard segment that stops
the moment it has driven ‖R‖∞ down `picard_gain_target` orders (`solver_picard_gain` carries the gain
target → stop_reason "picard_gain_reached") — just enough to re-enter Newton's basin — then back to
Newton, bounded by `max_swaps`. Each segment is tagged honestly via `_record_stage!`
(`<stage_prefix>:PP[swap]:N` / `:P`) — never relabel a Picard step as Newton.

Returns `:success` (a Newton segment was accepted), `:structural_abort` (a non-finite/exception/merit-
divergence segment, OR a Picard segment that fails to reduce ‖R‖ — neither solver can advance, so the
caller fails and the homotopy drops eps_pert; restore done), or `:reject` (swaps exhausted). This routine
only schedules *between* solves: the safeguards inside each segment (merit/line-search/divergence within
a Newton segment, monotone-residual within a Picard segment) belong to the solvers themselves. Inert
unless the caller opts in (pingpong_enabled).
"""
function _pingpong_cascade!(final_x0, restore_vec, op_newton, op_picard, solver_newton,
                            solver_picard_gain, policy::CascadePolicy, diag_cache, iter_count_ref;
                            stage_prefix::String, max_swaps::Int)
    # Account a Newton segment's iters + diagnostics (mirrors the cascade bookkeeping) and return its verdict.
    function _account_newton!(res, label)
        _record_stage!(diag_cache, label, res)
        if res.state == :ok
            iter_count_ref[] += res.iterations
            diag_cache["final_residual_norm"] = res.residual_norm
            get!(diag_cache, "initial_residual_norm", res.initial_residual_norm)
        elseif res.state == :max_iters_caught
            iter_count_ref[] += 1
        end
        return cascade_step_outcome(res, policy)
    end

    # Segment 0 — Newton from the entry iterate. (backup=restore_vec ⇒ helper auto-restores on non-finite.)
    res = safe_fe_solve!(final_x0, solver_newton, op_newton; backup=restore_vec)
    outcome = _account_newton!(res, "$(stage_prefix):PP[0]:N")
    if outcome == :success || outcome == :one_iter_success
        return :success
    end

    swap = 0
    while swap < max_swaps
        # Picard segment — globalize until the gain target re-enters Newton's basin.
        res_p = safe_fe_solve!(final_x0, solver_picard_gain, op_picard; backup=restore_vec)
        _record_stage!(diag_cache, "$(stage_prefix):PP[$(swap)]:P", res_p)
        if res_p.state == :ok
            iter_count_ref[] += res_p.iterations
            diag_cache["final_residual_norm"] = res_p.residual_norm
        elseif res_p.state == :max_iters_caught
            iter_count_ref[] += 1
        end
        # [ping-pong] A Picard segment that itself REACHES the convergence gate is a success — honor the
        # same stop-reason→success mapping the Newton branch uses (`cascade_step_outcome`). This MUST be
        # tested before the no-progress guard below: a 0-iteration `initial_ftol` exit (the Picard entry
        # already satisfies the gate) has residual_norm == initial_residual_norm, which the raw `≥` test
        # would otherwise misread as no-progress and abort — discarding a converged state (the historical
        # re-anchoring failure mode). [known-fragility]
        if res_p.state == :ok && (res_p.stop_reason == "ftol_reached" ||
                                  res_p.stop_reason == "initial_ftol" ||
                                  res_p.stop_reason == "stagnation_noise_floor_reached")
            return :success
        end
        # The Picard segment is the globalizer after Newton diverged. If it ALSO fails to make progress —
        # a structural failure (non-finite / exception / merit-divergence), a depleted line search, or
        # simply no reduction in ‖R‖ (and it did NOT converge, handled above) — then NEITHER solver can
        # advance from this iterate. Deem the attempt failed (restore the entry iterate) so the caller
        # fails and the OUTER homotopy loop drops eps_pert, instead of repeating the cycle to max_swaps.
        picard_no_progress = res_p.state == :ok &&
            (res_p.stop_reason == "merit_divergence_escaped" ||
             res_p.stop_reason == "linesearch_failed" ||
             res_p.residual_norm >= res_p.initial_residual_norm)
        if res_p.state == :nonfinite || res_p.state == :exception ||
           any(!isfinite, get_free_dof_values(final_x0)) || picard_no_progress
            get_free_dof_values(final_x0) .= restore_vec
            return :structural_abort
        end

        # Newton segment on the Picard-smoothed iterate.
        res_n = safe_fe_solve!(final_x0, solver_newton, op_newton; backup=restore_vec)
        outcome = _account_newton!(res_n, "$(stage_prefix):PP[$(swap+1)]:N")
        if outcome == :success || outcome == :one_iter_success
            return :success
        elseif res_n.state == :nonfinite || res_n.state == :exception
            return :structural_abort  # helper restored; cannot recover
        end
        swap += 1
    end
    return :reject  # swaps exhausted without success → caller falls through to terminal logic
end

"""
    solve_system(...)

Algorithm O (SimulationOrchestration) of `theory/osgs_algorithm/osgs_algorithm.tex`. Orchestrates
the nonlinear solver iteration loop over the chosen Variational Multiscale space: Stage-I ASGS
algebraic initialisation (`_initialize_asgs_state!`, asgs_solver.jl) → method dispatch (ASGS, or the
OSGS coupled solve `solve_osgs_stage!` in `osgs_solver.jl`), with optional post-convergence
verification through the `SolutionVerifier` hooks.

See the module-level "Algorithm-to-code mapping" docstring at the top of this file, and
`docs/solver/algorithm-code-mapping.md`.
"""
function solve_system(setup::FETopology, formulation::VMSFormulation, iter_solvers::StageSolvers,
                      config::PorousNSConfig, x0;
                      diagnostics_cache=nothing, verifier::SolutionVerifier=NoVerification())

    # The orchestrator only needs the trial/test spaces (for the init operators) and the solver pair;
    # the OSGS-specific fields (projection spaces, h/forcing CellFields) are unpacked from `setup` inside
    # `solve_osgs_stage!`, and the init operators take `setup`/`formulation` whole.
    X, Y = setup.X, setup.Y
    solver_picard, solver_newton = iter_solvers.picard, iter_solvers.newton

    phys_cfg = config.physical_properties
    stab_cfg = config.numerical_method.stabilization
    sol_cfg  = config.numerical_method.solver
    diag_cache = isnothing(diagnostics_cache) ? Dict{String, Any}() : diagnostics_cache
    pi_u = nothing
    pi_p = nothing

    # [Fix A — scale-free convergence gate] Build the authoritative convergence evaluator ONCE and inject
    # it into BOTH stage solvers. Every downstream segment then shares ONE physically-anchored success
    # test (ε_M ≤ tol_M, ε_C ≤ tol_C): the Stage-I Newton↔Picard cascade (which derives from these via
    # `base_nls_global`/`solver_picard`) and the OSGS coupled re-wraps inside `solve_osgs_stage!` (which
    # call `_with_overrides`, and that preserves `conv_probe`). Because D_M is read from the iterate — not
    # a per-segment-frozen ‖R₀‖ — the gate is identical across ping-pong segments, removing the
    # re-anchoring that made each segment demand another ×(1/ftol) drop. (convergence_criterion.jl)
    conv_eval = build_convergence_probe(setup, formulation, sol_cfg.eps_tol_momentum, sol_cfg.eps_tol_mass)
    solver_picard = FESolver(_with_overrides(solver_picard.nls; conv_probe=conv_eval))
    solver_newton = FESolver(_with_overrides(solver_newton.nls; conv_probe=conv_eval))

    local final_x0 = x0
    local success = false
    local eval_time = 0.0
    iter_count_ref = Ref(0)

    check_porous_solver_parameters(stab_cfg, sol_cfg)

    method = stab_cfg.method
    ftol = solver_newton.nls.ftol
    freeze_cusp = sol_cfg.freeze_jacobian_cusp

    # ==============================================================================
    # ALGEBRAIC INITIALIZATION (Stage I)
    # Even for OSGS, we first boot the global PDE with the zero-projection (ASGS) formulation, to map
    # the nonlinear fields into the exact-Newton quadratic basin before the OSGS coupled solve runs.
    # ==============================================================================
    # [iterative-penalty] OFF by default ⇒ penalty_on=false ⇒ p_prev=nothing everywhere ⇒ byte-identical
    # legacy path. When ON (and ε_num>0), the mass residual carries ε_num·(pⁿ − p_prev) with p_prev the
    # previous-pass pressure held FIXED within the pass; an OUTER loop (below) updates p_prev between passes.
    penalty_on = sol_cfg.iterative_penalty_enabled && phys_cfg.numerical_epsilon > 0.0
    p_prev_ref = Ref{Any}(nothing)
    function _pressure_copy(xh)
        _u, _p = xh
        return FEFunction(X.spaces[2], copy(get_free_dof_values(_p)))
    end
    function _pressure_rel_drift(p_new, p_old)
        num = sqrt(abs(sum(∫((p_new - p_old) * (p_new - p_old))setup.dΩ)))
        den = sqrt(abs(sum(∫(p_new * p_new)setup.dΩ)))
        # eps(Float64) is a machine-epsilon underflow guard (the codebase convention for relative-norm
        # denominators, cf. convergence_criterion.jl): it keeps the ratio finite when ‖pⁿ‖ → 0 WITHOUT
        # injecting a problem scale. It never activates in a well-posed solve (pressure is O(1) here).
        return num / max(den, eps(Float64))
    end

    res_fn_init(x, y) = build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, p_prev=(penalty_on ? p_prev_ref[] : nothing))
    jac_picard_init(x, dx, y) = build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    jac_newton_init(x, dx, y) = sol_cfg.ablation_mode == "picard_only" ?
        build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0) :
        build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=nothing, pi_p=nothing)

    op_picard_init = FEOperator(res_fn_init, jac_picard_init, X, Y)
    op_newton_init = FEOperator(res_fn_init, jac_newton_init, X, Y)

    base_nls_global = solver_newton.nls
    # Verbatim re-wrap of the configured Newton solver (no field overrides) —
    # the explicit FESolver(_with_overrides(base_nls_global)) form expresses
    # "use the configured solver verbatim" without depending on `solver_newton`'s
    # FESolver wrapper identity.
    solver_newton_asgs = FESolver(_with_overrides(base_nls_global))

    x0_backup = copy(get_free_dof_values(x0))

    # Ping-pong scheduling (opt-in; default off ⇒ bit-identical one-way cascade). When enabled, the
    # Stage-I Picard segment uses a gain-then-return target so it stops the moment it re-enters Newton's
    # basin. The gain target is applied here (single point) so build_iter_solvers / call sites need not.
    pingpong_enabled = sol_cfg.pingpong_enabled
    pingpong_max_swaps = sol_cfg.pingpong_max_swaps
    solver_picard_gain = pingpong_enabled ?
        FESolver(_with_overrides(solver_picard.nls; picard_gain_target=sol_cfg.pingpong_picard_gain_orders)) :
        solver_picard

    # One full solve pass (Stage-I ASGS boot → method dispatch → verification), returning the
    # solve_system 5-tuple. With penalty OFF this runs exactly ONCE (byte-identical legacy path); with
    # penalty ON the outer loop below calls it repeatedly, holding p_prev fixed within each pass.
    function _one_pass()
        local pass_success
        pass_backup = copy(get_free_dof_values(x0))
        # Stage I — Algorithm B (RobustNonlinearCascade), ASGS operators, restore=pass_backup.
        # [osgs_skip_asgs_boot] For OSGS with the boot skipped, run the OSGS staggered/coupled solve DIRECTLY
        # from the (warm) guess x0 — paper-faithful (alg:StationarySystem has no ASGS pre-boot), avoiding the
        # boot converging ASGS to its root (a different fixed point) and ill-conditioning the OSGS step. The
        # eps_pert homotopy provides the cold-start globalization the boot otherwise gave. ASGS-method always boots.
        skip_boot = (method == "OSGS") && sol_cfg.osgs_skip_asgs_boot
        pass_time = @elapsed begin
            pass_success = skip_boot ? true :
                _initialize_asgs_state!(x0, pass_backup, op_newton_init, op_picard_init,
                                              solver_newton_asgs, solver_picard, ftol,
                                              diag_cache, iter_count_ref;
                                              pingpong_enabled=pingpong_enabled,
                                              pingpong_max_swaps=pingpong_max_swaps,
                                              solver_picard_gain=solver_picard_gain)
        end
        skip_boot && println("\n      [+] OSGS: skipping ASGS Stage-I boot — solving directly from the initial guess.")

        # The 2nd return element reports the verifier's outcome (`verification_result`): `nothing` for
        # production (NoVerification), or the MMS plateau verdict for the MMS harness. It is independent
        # of the inner-solver `success` flag, so callers can distinguish "solver converged but
        # verification failed (budget exhausted)" from "solver failed".
        if !pass_success
            diag_cache["pi_u"] = pi_u
            diag_cache["pi_p"] = pi_p
            return (false, verification_result(verifier), x0, iter_count_ref[], pass_time)
        end

        pass_final_x0 = x0

        if method == "ASGS"
            println("\n      [+] ASGS base convergence reached.")
            # Post-convergence verification hook (Algorithm D). Production (NoVerification) is a no-op;
            # the MMS harness runs the plateau loop. `step_once!` advances the converged state by one
            # Newton iteration on the SAME ASGS operator the boot used — built here because operator
            # construction lives in the core, but the verifier owns the loop.
            local_solver = FESolver(_with_overrides(solver_newton.nls; max_iters=1))
            step_once! = () -> _solve_one_step!(pass_final_x0, local_solver, op_newton_init)
            pass_time += @elapsed on_asgs_converged!(verifier, pass_final_x0, step_once!, diag_cache, iter_count_ref)
            diag_cache["base_convergence_reached"] = true   # core key (set regardless of verification)
            diag_cache["pi_u"] = pi_u
            diag_cache["pi_p"] = pi_p
            return (pass_success, verification_result(verifier), pass_final_x0, iter_count_ref[], pass_time)
        end

        # method == "OSGS"
        # Algorithm C — the coupled OSGS solve. All OSGS-specific setup (unconstrained projection
        # spaces, factored L² mass matrices) and the solve itself live in `solve_osgs_stage!`
        # (osgs_solver.jl); the orchestrator stays OSGS-agnostic apart from this dispatch. The
        # iterative-penalty p_prev (fixed within this pass) is threaded through to its residual closures.
        (osgs_success, _osgs_pi_u, _osgs_pi_p, osgs_elapsed) = solve_osgs_stage!(
            pass_success, pass_final_x0, setup, formulation, config, solver_newton, verifier,
            diag_cache, iter_count_ref; p_prev=(penalty_on ? p_prev_ref[] : nothing))
        pass_time += osgs_elapsed
        # solve_osgs_stage! already wrote diag_cache["pi_u"] and diag_cache["pi_p"].
        return (osgs_success, verification_result(verifier), pass_final_x0, iter_count_ref[], pass_time)
    end

    # Penalty OFF: the legacy single pass, byte-identical.
    penalty_on || return _one_pass()

    # [iterative-penalty] OUTER fixed-point loop (Codina, article.tex §5.2). Each pass solves the full
    # system with pⁿ⁻¹ held fixed (via p_prev_ref); between passes we update pⁿ⁻¹ ← pⁿ and stop once the
    # relative pressure drift < xtol, at which point ε_num·(pⁿ − pⁿ⁻¹) → 0 and the solution is unaltered.
    p_prev_ref[] = _pressure_copy(x0)
    local ip_result = (false, verification_result(verifier), x0, iter_count_ref[], 0.0)
    local ip_time = 0.0
    for ip_pass in 1:sol_cfg.iterative_penalty_max_iters
        ip_result = _one_pass()
        ip_time += ip_result[5]
        ip_result[1] || break          # solve failed this pass → stop (keep the failed result)
        p_new = _pressure_copy(ip_result[3])
        drift = _pressure_rel_drift(p_new, p_prev_ref[])
        p_prev_ref[] = p_new
        println("      [iter-penalty] pass $(ip_pass): relative pressure drift = $(drift) (xtol=$(sol_cfg.xtol))"); flush(stdout)
        drift < sol_cfg.xtol && break
    end
    return (ip_result[1], ip_result[2], ip_result[3], ip_result[4], ip_time)
end
