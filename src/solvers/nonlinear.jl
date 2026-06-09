# src/solvers/nonlinear.jl
using Gridap
using Gridap.Algebra
using Gridap.FESpaces: get_test, num_free_dofs
using Gridap.MultiField: MultiFieldFESpace
using LinearAlgebra
using SparseArrays: diag

"""
    SafeNewtonSolver <: NonlinearSolver

A globalized nonlinear solver for the monolithic (u, p) stabilized system. Each iteration
assembles the residual `R` and Jacobian `J`, solves `J·dx = R` with `ls`, and accepts a
damped step `x ← x - α·dx` via an Armijo line search on a row-equilibrated merit function.
A cascade of safeguards (divergence guard, stagnation/noise-floor stops, iteration cap)
keeps the iteration from running away. The same struct serves two roles selected by `mode`:
the Exact-Newton step (`:newton`) and the Picard fixed-point fallback (`:picard`); the
orchestrator in `asgs_solver.jl` chains them. Implements Algorithm A (the per-attempt inner
solve) of `alg:StationarySystem`.

Fields:
- `ls` — the inner linear solver applied to each Newton/Picard linear system.
- `max_iters` — cap on nonlinear iterations.
- `max_increases` — number of consecutive merit increases tolerated before declaring divergence.
- `xtol` — step-norm floor: a step shorter than this is treated as coordinate stagnation.
- `ftol` — absolute per-field residual inf-norm convergence threshold (the hard floor).
- `linesearch_alpha_min` — smallest step fraction α the line search will try before giving up.
- `c1` — Armijo sufficient-decrease constant, in (0, 1).
- `divergence_merit_factor` — factor (≥ 1) by which the merit may grow per step before it counts
   as a divergence increment.
- `stagnation_noise_floor` — residual level below which further descent is treated as numerical noise.
- `max_linesearch_iterations` — backtracking budget per nonlinear step.
- `linesearch_contraction_factor` — geometric factor α is multiplied by on each backtrack.
"""
struct SafeNewtonSolver <: NonlinearSolver
    ls::LinearSolver
    max_iters::Int
    max_increases::Int
    xtol::Float64
    ftol::Float64
    linesearch_alpha_min::Float64
    c1::Float64
    divergence_merit_factor::Float64
    stagnation_noise_floor::Float64
    max_linesearch_iterations::Int
    linesearch_contraction_factor::Float64
    mode::Symbol  # :newton (Armijo on Jacobian-scaled merit) or :picard (monotone residual)
    # [honest-exit] Gate on whether a noise-floor stop (‖R‖∞ ≤ stagnation_noise_floor) is allowed
    # to report CONVERGED: it is, only if ‖R‖∞ ≤ noise_floor_success_max_ftol_multiple · ftol.
    # Inf ⇒ gate disabled (any noise-floor stop counts as success). A finite value (e.g. 10.0)
    # rejects high-Re fold stalls — where ‖R‖≈1e-5 sits ~10²–10³× above ftol — from masquerading
    # as a true root.
    noise_floor_success_max_ftol_multiple::Float64
    # Per-field RELATIVE convergence tolerances (one entry per (u, p) field block). When set, the
    # solver turns each dimensionless target into an absolute per-field threshold at iteration 0 by
    # scaling it by the INITIAL RESIDUAL norm of that field block, ‖R₀_k‖:
    #     effective_ftol_per_field[k] = max(ftol, relative_ftol_per_field[k] * ‖R₀_k‖)
    # All termination checks then use those per-field thresholds, so the gate is the DIMENSIONLESS
    # relative reduction `‖R_k‖ ≤ rel·‖R₀_k‖`. The scalar `ftol` is the per-field absolute FLOOR;
    # `relative_ftol_per_field[k]` is the dimensionless target (e.g. c_sf·h^(k+1), the MMS plateau).
    # `nothing` ⇒ the solver uses the scalar `ftol`/`stagnation_noise_floor` globally.
    #
    # [known-fragility] Scale by the RESIDUAL norm, not the solution magnitude ‖x₀‖: ‖R_k‖ and its
    # scale ‖R₀_k‖ share units, so the ratio is dimensionless and encoding-invariant. Scaling by
    # ‖x₀‖ is dimensional (velocity ∝ U, pressure ∝ U²) and makes the converged result non-covariant
    # across fields (worst in pressure) under OSGS's single-step Picard-type coupling. A true-zero initial
    # residual on a field collapses that field's threshold to the scalar `ftol` floor.
    relative_ftol_per_field::Union{Nothing, Vector{Float64}}
    relative_stagnation_noise_floor_per_field::Union{Nothing, Vector{Float64}}
    # [no-progress stall guard] If `stall_window > 0`, the solve aborts with stop_reason
    # "no_progress_stall" once the BEST residual inf-norm fails to improve by the relative
    # factor `stall_min_rel_improvement` over `stall_window` consecutive iterations. This lets a
    # doomed attempt (e.g. a stiff high-Re guess grinding the full Newton budget with the merit
    # stuck) bail in a handful of iterations so the caller can fall back to a milder homotopy
    # perturbation instead of exhausting max_iters. `stall_window = 0` ⇒ disabled.
    stall_window::Int
    stall_min_rel_improvement::Float64
    # [P4 ping-pong] Picard gain-then-return stop. In :picard mode with a FINITE value, the solve stops
    # with stop_reason "picard_gain_reached" as soon as ‖R‖∞ ≤ ‖R₀‖∞ · 10^(-picard_gain_target) — i.e.
    # once Picard has driven the residual `picard_gain_target` orders below its starting value, just enough
    # to re-enter Newton's quadratic basin. The Newton↔Picard ping-pong orchestrator (asgs_solver.jl)
    # then hands back to Newton instead of running Picard all the way to its ftol/cap. `Inf` ⇒ disabled
    # (Picard runs to ftol/cap). Ignored in :newton mode.
    picard_gain_target::Float64
    # [residual-divergence guard, :newton] The Φ-based divergence check is masked by the Armijo line
    # search (Φ strictly decreases on every accepted step), so it cannot catch a Newton solve whose
    # residual ‖R‖∞ is climbing. When `> 0`, abort (→ Picard via the cascade) after this many consecutive
    # ‖R‖∞ increases beyond `divergence_merit_factor`. `0` ⇒ disabled (Newton runs its full budget).
    newton_residual_divergence_patience::Int
    # [diagnostic, read-only] Optional convergence probe `(x, b, field_blocks) -> (eps_M, eps_C, degenerate)`
    # implementing the scale-free criterion (convergence_criterion.jl). When set, `_safe_solve_inner!`
    # calls it once per ACCEPTED iteration and records ε_M/ε_C in the iteration history. It is a PURE
    # OBSERVER: it never mutates the iterate, residual, step, or stopping logic (and is wrapped in a
    # try/catch), so the solve is byte-identical whether it is set or `nothing`. `nothing` ⇒ no
    # convergence-norm tracing — the default, and zero cost. Gated on by the harness only for trajectory
    # diagnostics, never in production (per-iteration field assembly is too costly for the sweep).
    conv_probe::Union{Nothing,Function}
end

# Positional constructor for the eleven core controls, with every safeguard exposed as a keyword
# (each defaulting to its disabled/inert value, so a caller opts in only to the guards it needs).
function SafeNewtonSolver(ls::LinearSolver, max_iters::Int, max_increases::Int,
                          xtol::Float64, ftol::Float64, linesearch_alpha_min::Float64,
                          c1::Float64, divergence_merit_factor::Float64,
                          stagnation_noise_floor::Float64, max_linesearch_iterations::Int,
                          linesearch_contraction_factor::Float64; mode::Symbol = :newton,
                          noise_floor_success_max_ftol_multiple::Float64 = Inf,
                          relative_ftol_per_field::Union{Nothing, Vector{Float64}} = nothing,
                          relative_stagnation_noise_floor_per_field::Union{Nothing, Vector{Float64}} = nothing,
                          stall_window::Int = 0,
                          stall_min_rel_improvement::Float64 = 0.0,
                          picard_gain_target::Float64 = Inf,
                          newton_residual_divergence_patience::Int = 0,
                          conv_probe::Union{Nothing,Function} = nothing)
    return SafeNewtonSolver(ls, max_iters, max_increases, xtol, ftol, linesearch_alpha_min,
                            c1, divergence_merit_factor, stagnation_noise_floor,
                            max_linesearch_iterations, linesearch_contraction_factor, mode,
                            noise_floor_success_max_ftol_multiple,
                            relative_ftol_per_field, relative_stagnation_noise_floor_per_field,
                            stall_window, stall_min_rel_improvement, picard_gain_target,
                            newton_residual_divergence_patience, conv_probe)
end

function SafeNewtonSolver(ls::LinearSolver, cfg::SolverConfig; mode::Symbol = :newton)
    return SafeNewtonSolver(
        ls, cfg.newton_iterations, cfg.max_increases, cfg.xtol, cfg.ftol,
        cfg.linesearch_alpha_min, cfg.armijo_c1, cfg.divergence_merit_factor, cfg.stagnation_noise_floor,
        cfg.max_linesearch_iterations, cfg.linesearch_contraction_factor; mode=mode,
        noise_floor_success_max_ftol_multiple=cfg.noise_floor_success_max_ftol_multiple,
        newton_residual_divergence_patience=cfg.newton_residual_divergence_patience
    )
end

# Copy-with-overrides: returns a clone of `nls` with selected fields replaced. Only the fields
# the orchestrator actually retunes at call sites are exposed (`max_iters`, `ftol`, `mode`,
# `stall_window`, `picard_gain_target`); all others are inherited from `nls` verbatim. `nothing`
# for any kwarg means "keep `nls`'s value". Cloning through one helper guarantees no struct field
# is silently dropped when a new safeguard is added.
function _with_overrides(nls::SafeNewtonSolver; max_iters=nothing, ftol=nothing, mode=nothing,
                         picard_gain_target=nothing, stall_window=nothing, conv_probe=nothing)
    return SafeNewtonSolver(
        nls.ls,
        isnothing(max_iters) ? nls.max_iters : max_iters,
        nls.max_increases,
        nls.xtol,
        isnothing(ftol) ? nls.ftol : ftol,
        nls.linesearch_alpha_min,
        nls.c1,
        nls.divergence_merit_factor,
        nls.stagnation_noise_floor,
        nls.max_linesearch_iterations,
        nls.linesearch_contraction_factor,
        isnothing(mode) ? nls.mode : mode,
        nls.noise_floor_success_max_ftol_multiple,
        nls.relative_ftol_per_field,
        nls.relative_stagnation_noise_floor_per_field,
        # `stall_window` override: the cold-start (Stage-I) boot wants the early-bail stall sensor
        # (oscillation/divergence → Picard), but the OSGS Stage-II coupled solve converges slowly and
        # monotonically (inexact-Newton linear rate); there the stall sensor would mistake slow progress
        # for a stall and bail, silently degenerating OSGS into ASGS. The coupled solve therefore passes
        # stall_window=0 to disable it.
        isnothing(stall_window) ? nls.stall_window : stall_window,
        nls.stall_min_rel_improvement,
        # `picard_gain_target` override (the ping-pong Picard segment): nothing ⇒ keep base (Inf = inert).
        # [known-fragility] MUST be threaded here — the OSGS-inner Picard is rebuilt via this helper, so
        # dropping it would silently disable ping-pong on the OSGS path.
        isnothing(picard_gain_target) ? nls.picard_gain_target : picard_gain_target,
        nls.newton_residual_divergence_patience,
        # read-only diagnostic probe: inherited verbatim unless explicitly set here (the harness attaches
        # it once, after `setup` exists; the OSGS coupled re-wraps then inherit it through this `nothing`).
        isnothing(conv_probe) ? nls.conv_probe : conv_probe,
    )
end

# Single construction point for the (picard, newton) FESolver pair the orchestrator chains. Used by
# both the production path (`src/run_simulation.jl`) and the MMS harness (`run_test.jl`), so any new
# scheduling control is threaded here once and both callers get it. The Picard solver is built in
# :picard mode, the Newton solver in :newton mode; everything else (max_increases, xtol, linesearch
# params, armijo_c1, divergence_merit_factor) is read from `sol_cfg` verbatim. Two distinct ftols
# (`newton_ftol`/`picard_ftol`) because the two paths want different convergence targets for the
# Newton solve and the Picard handoff. Keyword defaults reproduce the bare production construction;
# the harness overrides the per-field and stall/ping-pong kwargs.
function build_iter_solvers(sol_cfg::SolverConfig, ls::LinearSolver;
                            newton_max_iters::Int, picard_max_iters::Int,
                            newton_ftol::Float64, picard_ftol::Float64,
                            stagnation_noise_floor::Float64 = sol_cfg.stagnation_noise_floor,
                            noise_floor_success_max_ftol_multiple::Float64 = sol_cfg.noise_floor_success_max_ftol_multiple,
                            relative_ftol_per_field::Union{Nothing, Vector{Float64}} = nothing,
                            relative_stagnation_noise_floor_per_field::Union{Nothing, Vector{Float64}} = nothing,
                            stall_window::Int = 0,
                            stall_min_rel_improvement::Float64 = 0.0,
                            picard_gain_target::Float64 = Inf,
                            conv_probe::Union{Nothing,Function} = nothing)
    nls_picard = SafeNewtonSolver(
        ls, picard_max_iters, sol_cfg.max_increases, sol_cfg.xtol, picard_ftol,
        sol_cfg.linesearch_alpha_min, sol_cfg.armijo_c1, sol_cfg.divergence_merit_factor,
        stagnation_noise_floor, sol_cfg.max_linesearch_iterations, sol_cfg.linesearch_contraction_factor;
        mode = :picard,
        noise_floor_success_max_ftol_multiple = noise_floor_success_max_ftol_multiple,
        relative_ftol_per_field = relative_ftol_per_field,
        relative_stagnation_noise_floor_per_field = relative_stagnation_noise_floor_per_field,
        stall_window = stall_window, stall_min_rel_improvement = stall_min_rel_improvement,
        picard_gain_target = picard_gain_target, conv_probe = conv_probe)
    nls_newton = SafeNewtonSolver(
        ls, newton_max_iters, sol_cfg.max_increases, sol_cfg.xtol, newton_ftol,
        sol_cfg.linesearch_alpha_min, sol_cfg.armijo_c1, sol_cfg.divergence_merit_factor,
        stagnation_noise_floor, sol_cfg.max_linesearch_iterations, sol_cfg.linesearch_contraction_factor;
        mode = :newton,
        noise_floor_success_max_ftol_multiple = noise_floor_success_max_ftol_multiple,
        relative_ftol_per_field = relative_ftol_per_field,
        relative_stagnation_noise_floor_per_field = relative_stagnation_noise_floor_per_field,
        stall_window = stall_window, stall_min_rel_improvement = stall_min_rel_improvement,
        newton_residual_divergence_patience = sol_cfg.newton_residual_divergence_patience,
        conv_probe = conv_probe)
    return (picard = FESolver(nls_picard), newton = FESolver(nls_newton))
end

# Validates the solver's numeric controls at the start of every solve, turning an out-of-range
# tolerance, factor, or mode into a loud ArgumentError instead of silent misbehaviour. Each check's
# message names the field and the admissible range; the disabled sentinels (Inf for the gates,
# 0 for the patience/window counters) are accepted as valid.
function check_solver_parameters(s::SafeNewtonSolver)
    if !(0.0 < s.c1 < 1.0) throw(ArgumentError("Armijo 'c1' must be in (0, 1).")) end
    if s.divergence_merit_factor < 1.0 throw(ArgumentError("Divergence merit factor must be >= 1.0.")) end
    if s.xtol <= 0.0 || s.ftol <= 0.0 || s.stagnation_noise_floor <= 0.0 throw(ArgumentError("Tolerances must be > 0.")) end
    if s.linesearch_alpha_min <= 0.0 || s.linesearch_alpha_min > 1.0 throw(ArgumentError("alpha_min must be in (0, 1].")) end
    if !(s.mode in (:newton, :picard)) throw(ArgumentError("SafeNewtonSolver mode must be :newton or :picard, got $(s.mode).")) end
    if s.noise_floor_success_max_ftol_multiple < 1.0 throw(ArgumentError("noise_floor_success_max_ftol_multiple must be >= 1.0 (Inf disables the honest-exit gate).")) end
    if !(s.picard_gain_target > 0.0) throw(ArgumentError("picard_gain_target must be > 0 (Inf disables the P4 Picard gain-then-return stop).")) end
    if s.newton_residual_divergence_patience < 0 throw(ArgumentError("newton_residual_divergence_patience must be >= 0 (0 disables the residual-divergence → Picard handoff).")) end
    if s.relative_ftol_per_field !== nothing
        if isempty(s.relative_ftol_per_field)
            throw(ArgumentError("relative_ftol_per_field must be a non-empty Vector or nothing."))
        end
        for (k, v) in enumerate(s.relative_ftol_per_field)
            if !(v > 0.0) || !isfinite(v)
                throw(ArgumentError("relative_ftol_per_field[$k] = $v must be finite and > 0."))
            end
        end
    end
    if s.relative_stagnation_noise_floor_per_field !== nothing
        if isempty(s.relative_stagnation_noise_floor_per_field)
            throw(ArgumentError("relative_stagnation_noise_floor_per_field must be a non-empty Vector or nothing."))
        end
        for (k, v) in enumerate(s.relative_stagnation_noise_floor_per_field)
            if !(v > 0.0) || !isfinite(v)
                throw(ArgumentError("relative_stagnation_noise_floor_per_field[$k] = $v must be finite and > 0."))
            end
        end
    end
    # If both per-field vectors are present, their lengths must agree (same field count).
    if s.relative_ftol_per_field !== nothing && s.relative_stagnation_noise_floor_per_field !== nothing &&
       length(s.relative_ftol_per_field) != length(s.relative_stagnation_noise_floor_per_field)
        throw(ArgumentError("relative_ftol_per_field and relative_stagnation_noise_floor_per_field must have the same length (one entry per field)."))
    end
    if s.stall_window < 0 throw(ArgumentError("stall_window must be >= 0 (0 disables the no-progress stall guard).")) end
    if !(0.0 <= s.stall_min_rel_improvement < 1.0) throw(ArgumentError("stall_min_rel_improvement must be in [0, 1).")) end
end

# Outcome of one inner solve (one invocation of `_safe_solve_inner!`). Carries both the final state
# and a record of how the iteration got there, so the orchestrator can decide whether to accept,
# fall back to Picard, or dilute the homotopy.
struct SafeSolverResult
    iterations::Int               # nonlinear iterations actually taken
    residual_norm::Float64        # final residual inf-norm ‖R‖_∞ (field-blind)
    initial_residual_norm::Float64 # residual inf-norm at iteration 0
    step_norm::Float64            # last accepted step inf-norm (α·‖dx‖_∞)
    stop_reason::String           # why the loop exited (e.g. "ftol_reached", "linesearch_failed")
    # [trajectory] Per-iteration record of this inner run: a Vector of NamedTuples
    # (i, f_inf, f_norm, merit, step_inf, alpha, accepted). Empty for the initial-ftol short-circuit.
    # The orchestrator tags each invocation with a stage code (e.g. "B:StageI:N1", "C:OSGS[k]:Picard")
    # and assembles diag_cache["trajectory"] from these.
    iteration_history::Vector
    # [trajectory] Normalized residuals: the scalar actually compared against the convergence threshold,
    # f_norm = maxₖ(‖R[block_k]‖_∞ / effective_ftol_per_field[k]) (≤ 1 ⟺ converged). The `residual_norm`
    # above is field-blind and measured against the machine ftol floor; these track the per-field gate.
    initial_residual_normalized::Float64
    residual_normalized::Float64
end

# Reusable scratch for a solve: the residual vector `b`, Jacobian `A`, step `dx`, the linear-solver
# symbolic/numeric cache, and the `result`. Passing this back into `solve!` lets the orchestrator
# reuse the allocations and factorization workspace across the Newton/Picard/homotopy cascade.
struct SafeSolverCache
    b::AbstractVector
    A::AbstractMatrix
    dx::AbstractVector
    ls_cache::Any
    result::SafeSolverResult
end

# Algorithm A.1: assemble and solve the linear system `A·dx = b` with the inner linear solver.
# Reuses the symbolic factorization in `ls_cache` if present, builds a fresh numeric setup for the
# current `A`, and writes the step into `dx`. Returns `(failed, ls_cache)`: `failed` is true if the
# solve threw or produced NaNs, so the caller can roll back to the best-so-far iterate rather than
# propagate garbage.
function eval_linear_system_resolution!(dx, A, b, solver::SafeNewtonSolver, ls_cache)
    solve_failed = false
    try
        ls_cache = ls_cache === nothing ? symbolic_setup(solver.ls, A) : ls_cache
        ns = numerical_setup(ls_cache, A)
        solve!(dx, ns, b)
    catch e
        println("  [Linear Solve Exception] ", e)
        solve_failed = true
    end
    
    if solve_failed || any(isnan, dx)
        return true, ls_cache # Failed
    end
    return false, ls_cache # Success
end

# Mutable scratch threaded through one inner solve, holding the current/next iterate's residual and
# merit so the line search, divergence guard, and termination checks all read the same numbers.
mutable struct SafeIterationState
    i::Int               # current iteration index
    inc_count::Int       # consecutive merit (Φ in :newton, ‖R‖∞ in :picard) increases
    res_inc_count::Int   # consecutive ‖R‖∞ increases (Newton residual-divergence guard)
    ls_success::Bool     # did the last Armijo line search find an acceptable step
    alpha::Float64       # accepted step fraction α
    step_norm::Float64   # α·‖dx‖_∞ of the accepted step
    norm_b_inf::Float64      # residual inf-norm at the current iterate
    norm_b_new_inf::Float64  # residual inf-norm at the trial (post-step) iterate
    phi_x::Float64           # merit Φ at the current iterate
    phi_x_new::Float64       # merit Φ at the trial iterate
    # Per-field inf-norms of the residual, kept in lock-step with the global `norm_b_*` scalars.
    # A single-field operator (`field_blocks === nothing`) leaves these as 1-element vectors matching
    # the global value, so downstream code can index `[k]` uniformly.
    norm_b_per_field::Vector{Float64}
    norm_b_new_per_field::Vector{Float64}
    # Per-field initial-residual inf-norms ‖R₀_k‖, frozen at iteration 0. They convert the solver's
    # dimensionless `relative_*_per_field` targets into absolute per-field thresholds, so the
    # convergence gate is a dimensionless relative reduction `‖R_k‖ ≤ rel·‖R₀_k‖`. [known-fragility]
    # The scale is the residual norm (same units as `R_k`), not the solution magnitude `‖x‖`, which is
    # what keeps the gate encoding-invariant. A true-zero initial residual on every field collapses the
    # thresholds to the scalar `ftol`/`stagnation_noise_floor` floors.
    initial_residual_scale_per_field::Vector{Float64}
    # Per-field absolute thresholds, derived once from `initial_residual_scale_per_field` and the solver's
    # `relative_*_per_field` (or the scalar `ftol`/`stagnation_noise_floor` floors when no relative
    # tolerances are set). These are the numbers the termination checks compare residuals against.
    effective_ftol_per_field::Vector{Float64}
    effective_noise_floor_per_field::Vector{Float64}
end

"""
    _initialize_effective_thresholds(solver, initial_residual_scale_per_field) -> (ftols, noise_floors)

Derives the per-field absolute thresholds the solver uses throughout the run.

`initial_residual_scale_per_field[k]` is the per-field initial-residual norm ‖R₀_k‖, frozen at iteration 0
(see `SafeIterationState`). Scaling the gate by the residual norm keeps it encoding-invariant.

The user's `relative_*_per_field[k]` is the dimensionless target (e.g. `c_sf·h^(k+1)`).
The effective absolute threshold is `max(scalar_floor, relative_target * scale)` so
the scalar value remains a hard floor (the solver never tries to go tighter than
the user's absolute ftol, regardless of what relative×scale produces).
"""
function _initialize_effective_thresholds(solver::SafeNewtonSolver,
                                          initial_residual_scale_per_field::Vector{Float64})
    nfields = length(initial_residual_scale_per_field)
    ftols = Vector{Float64}(undef, nfields)
    noise_floors = Vector{Float64}(undef, nfields)

    if solver.relative_ftol_per_field !== nothing
        @assert length(solver.relative_ftol_per_field) == nfields "relative_ftol_per_field length mismatch"
        @inbounds for k in 1:nfields
            ftols[k] = max(solver.ftol, solver.relative_ftol_per_field[k] * initial_residual_scale_per_field[k])
        end
    else
        @inbounds for k in 1:nfields
            ftols[k] = solver.ftol
        end
    end

    if solver.relative_stagnation_noise_floor_per_field !== nothing
        @assert length(solver.relative_stagnation_noise_floor_per_field) == nfields "relative_stagnation_noise_floor_per_field length mismatch"
        @inbounds for k in 1:nfields
            noise_floors[k] = max(solver.stagnation_noise_floor,
                                  solver.relative_stagnation_noise_floor_per_field[k] * initial_residual_scale_per_field[k])
        end
    else
        @inbounds for k in 1:nfields
            noise_floors[k] = solver.stagnation_noise_floor
        end
    end

    return ftols, noise_floors
end

"""
    _per_field_inf_norms!(out, b, field_blocks)

In-place fill of `out` with the per-field inf-norms of the residual `b`. When
`field_blocks === nothing` (single-field operator), `out` is a 1-element vector
containing `norm(b, Inf)`. Otherwise `out[k] = maximum(abs, view(b, field_blocks[k]))`,
one entry per (u, p) field block.
"""
function _per_field_inf_norms!(out::Vector{Float64}, b::AbstractVector,
                                field_blocks::Union{Nothing, Vector{UnitRange{Int}}})
    if field_blocks === nothing
        @assert length(out) == 1 "single-field fallback requires a 1-element output buffer"
        out[1] = norm(b, Inf)
    else
        @assert length(out) == length(field_blocks) "out and field_blocks must match in length"
        @inbounds for k in eachindex(field_blocks)
            m = 0.0
            for j in field_blocks[k]
                a = abs(b[j])
                if a > m
                    m = a
                end
            end
            out[k] = m
        end
    end
    return out
end

"""
    _residual_meets_per_field_ftol(per_field_norms, effective_ftols)

Returns true iff every field's residual inf-norm is at or below its effective absolute
threshold — the convergence test. When `solver.relative_*_per_field === nothing`,
`effective_ftols[k]` equals the solver's scalar `ftol` for every k, so this collapses to
a single global inf-norm check.
"""
function _residual_meets_per_field_ftol(per_field_norms::Vector{Float64},
                                          effective_ftols::Vector{Float64})
    @assert length(per_field_norms) == length(effective_ftols) "field count mismatch"
    @inbounds for k in eachindex(effective_ftols)
        per_field_norms[k] <= effective_ftols[k] || return false
    end
    return true
end

"""
    _residual_meets_per_field_noise_floor(per_field_norms, effective_noise_floors)

Same pattern as `_residual_meets_per_field_ftol` but for the stagnation noise floor.
"""
function _residual_meets_per_field_noise_floor(per_field_norms::Vector{Float64},
                                                 effective_noise_floors::Vector{Float64})
    @assert length(per_field_norms) == length(effective_noise_floors) "field count mismatch"
    @inbounds for k in eachindex(effective_noise_floors)
        per_field_norms[k] <= effective_noise_floors[k] || return false
    end
    return true
end

"""
    _residual_meets_per_field_honest_exit_gate(per_field_norms, effective_ftols, k_nf)

The "honest-exit" gate (see `noise_floor_success_max_ftol_multiple`): a noise-floor stop
counts as a genuine convergence only if every field's residual is also within
`k_nf × effective_ftols[k]`. With `k_nf = Inf` the gate always passes (disabled); a finite
`k_nf` rejects high-Re stalls that sit far above ftol from being reported as success.
"""
function _residual_meets_per_field_honest_exit_gate(per_field_norms::Vector{Float64},
                                                      effective_ftols::Vector{Float64},
                                                      k_nf::Float64)
    @assert length(per_field_norms) == length(effective_ftols) "field count mismatch"
    @inbounds for k in eachindex(effective_ftols)
        per_field_norms[k] <= k_nf * effective_ftols[k] || return false
    end
    return true
end

# Block-equilibrated merit Φ(b) = ½ Σ (b_i / w_i)², the scalar the Armijo line search drives down.
# The weights `w` equilibrate the per-field-block row scales (so a saddle-point pressure block's
# near-zero diagonal can't dominate the velocity rows); they come from `_update_merit_weights!`,
# read off the Jacobian diagonal. Used by both the line-search pass and the main loop.
_merit_W(b, w) = 0.5 * sum(idx -> (b[idx] / w[idx])^2, eachindex(b))

# Algorithm A.2: backtracking line search. Starting from α = 1, it trial-steps x ← x_old - α·dx,
# re-evaluates the residual, and accepts the first α that satisfies the mode's descent test;
# otherwise it contracts α geometrically until it dips below `linesearch_alpha_min`. On return,
# `state` carries the accepted α, step norm, trial residual norms, and merit. The acceptance test
# branches on `solver.mode` (see inline note below).
function eval_armijo_linesearch_pass!(x, x_old, b, dx, op, dir_deriv, solver::SafeNewtonSolver, w, state::SafeIterationState,
                                       field_blocks::Union{Nothing, Vector{UnitRange{Int}}})
    alpha = 1.0
    step_norm_base = norm(dx, Inf)
    state.ls_success = false
    state.norm_b_new_inf = norm(b, Inf)
    _per_field_inf_norms!(state.norm_b_new_per_field, b, field_blocks)
    state.phi_x_new = state.phi_x

    for ls_iter in 1:solver.max_linesearch_iterations
        x .= x_old .- alpha .* dx
        residual!(b, op, x)
        state.phi_x_new = _merit_W(b, w)
        state.norm_b_new_inf = norm(b, Inf)
        _per_field_inf_norms!(state.norm_b_new_per_field, b, field_blocks)

        if isnan(state.phi_x_new) || isnan(state.norm_b_new_inf)
            alpha *= solver.linesearch_contraction_factor
            continue
        end

        # Acceptance test by mode:
        #   :newton — Armijo on Jacobian-diagonal-scaled merit Φ with D = -2Φ.
        #             Valid because exact-Newton direction makes the cancellation hold.
        #   :picard — monotone residual ‖b_new‖_∞ ≤ (1 - c1·α) ‖b_old‖_∞.
        #             Removes the merit function from Picard's accept/reject loop,
        #             which is required because the true Picard D ≠ -2Φ (the Jacobian
        #             approximation drops cross-terms). Bank & Rose 1980 reference.
        if solver.mode === :newton
            if state.phi_x_new <= state.phi_x + solver.c1 * alpha * dir_deriv
                state.ls_success = true
                break
            end
        else  # :picard
            if state.norm_b_new_inf <= (1.0 - solver.c1 * alpha) * state.norm_b_inf
                state.ls_success = true
                break
            end
        end
        if alpha <= solver.linesearch_alpha_min
            break
        end
        alpha *= solver.linesearch_contraction_factor
    end
    
    state.step_norm = alpha * step_norm_base
    state.alpha = alpha
    return state.ls_success, state.alpha, state.step_norm, state.norm_b_new_inf, state.phi_x_new
end

# Algorithm A.3: the per-iteration safeguard cascade run after each accepted step. It decides whether
# to stop and whether the stop counts as convergence, returning `(stop_reason, converged, should_break)`.
# The checks fire in priority order: per-field ftol convergence; line-search failure (with an
# honest-exit noise-floor escape); merit divergence; Newton residual divergence; step-norm (xtol)
# stagnation; and the iteration cap. Each non-convergent stop logs a diagnostic line.
function eval_safeguard_termination_bounds!(solver::SafeNewtonSolver, state::SafeIterationState,
                                            scale_free::Bool=false)
    stop_reason = ""
    converged = false
    should_break = false

    k_nf = solver.noise_floor_success_max_ftol_multiple

    # [Fix A] Under the scale-free gate, SUCCESS is owned by the caller's ε_M/ε_C verdict (checked before
    # this function): reaching here means ε has NOT converged, so the per-segment-re-anchored per-field
    # ftol and the noise-floor "honest-exit" success branches are disabled — they would otherwise grant a
    # spurious success on a re-anchored threshold the scale-free gate just rejected. The genuine FAILURE
    # guards (line-search depletion, merit/residual divergence, xtol/iteration-cap stagnation) stay live.
    if !scale_free && _residual_meets_per_field_ftol(state.norm_b_new_per_field, state.effective_ftol_per_field)
        return "ftol_reached", true, true
    end

    if !state.ls_success
        if !scale_free &&
           _residual_meets_per_field_noise_floor(state.norm_b_new_per_field, state.effective_noise_floor_per_field) &&
           _residual_meets_per_field_honest_exit_gate(state.norm_b_new_per_field, state.effective_ftol_per_field, k_nf)
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Linesearch Depleted] Geometric step fraction alpha vanished below minimum limits without sufficient mathematical descent. Aborting local sequence.")
        return "linesearch_failed", false, true
    end
    
    # Divergence safeguard. [known-fragility] The metric MUST match the line search's accept metric:
    # in :newton mode compare Φ_new to Φ_old (the Armijo search drives Φ down, so Φ growing by factor
    # `divergence_merit_factor` signals something structurally wrong); in :picard mode compare ‖b‖_∞
    # to its previous value, because Picard's line search accepts on ‖b‖_∞ and Φ is meaningless across
    # Picard iterations — the merit weights `w = diag(J)` are refreshed each iteration from Picard's
    # Jacobian (which differs from Newton's), so Φ can grow even while ‖b‖_∞ shrinks monotonically.
    # Using Φ in :picard mode would trigger premature `merit_divergence_escaped` exits.
    diverged = if solver.mode === :picard
        state.norm_b_new_inf > state.norm_b_inf * solver.divergence_merit_factor
    else
        state.phi_x_new > state.phi_x * solver.divergence_merit_factor
    end
    if diverged
        state.inc_count += 1
        if state.inc_count >= solver.max_increases
            println("  [Merit Divergence] Sequence catastrophically expanded beyond allowed bounds. Extracted state exhibits unbounded divergence.")
            return "merit_divergence_escaped", false, true
        end
    else
        state.inc_count = 0
    end

    # [residual-divergence guard, :newton] The Φ check above is driven down by the Armijo line search,
    # so it cannot detect a Newton solve whose residual ‖R‖∞ is climbing. This tracks consecutive
    # ‖R‖∞ increases (beyond `divergence_merit_factor`) and hands off to Picard (a structural reject the
    # orchestrator catches) after `newton_residual_divergence_patience` of them. `= 0` ⇒ disabled.
    if solver.mode === :newton && solver.newton_residual_divergence_patience > 0
        if state.norm_b_new_inf > state.norm_b_inf * solver.divergence_merit_factor
            state.res_inc_count += 1
            if state.res_inc_count >= solver.newton_residual_divergence_patience
                println("  [Residual Divergence] ‖R‖∞ rose for $(state.res_inc_count) consecutive steps beyond the divergence factor; handing off to Picard.")
                return "residual_divergence_escaped", false, true
            end
        else
            state.res_inc_count = 0
        end
    end

    if state.step_norm <= solver.xtol
        if !scale_free &&
           _residual_meets_per_field_noise_floor(state.norm_b_new_per_field, state.effective_noise_floor_per_field) &&
           _residual_meets_per_field_honest_exit_gate(state.norm_b_new_per_field, state.effective_ftol_per_field, k_nf)
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Coordinate Stagnation] Step update magnitude collapsed below relative machine tracking limits (xtol = $(solver.xtol)). Algebraic progress saturated.")
        return "xtol_stagnation", false, true
    end

    if state.i == solver.max_iters
        if !scale_free &&
           _residual_meets_per_field_noise_floor(state.norm_b_new_per_field, state.effective_noise_floor_per_field) &&
           _residual_meets_per_field_honest_exit_gate(state.norm_b_new_per_field, state.effective_ftol_per_field, k_nf)
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Iteration Cap] Sequence hit maximum bounded loops ($(state.i)) without geometrically saturating non-linear constraints.")
        return "max_iters_stagnation", false, true
    end
    
    return stop_reason, converged, should_break
end

"""
    _detect_field_blocks(op) -> Union{Nothing, Vector{UnitRange{Int}}}

Returns the per-field free-DOF index ranges of the operator's test space when that space is a
`MultiFieldFESpace` — e.g. `[1:n_u, n_u+1:n_u+n_p]` for the monolithic (u, p) system. Assumes
`ConsecutiveMultiFieldStyle` (the only style this codebase uses), so the blocks are contiguous.
Returns `nothing` for single-field spaces or when the test space cannot be introspected, in which
case the merit equilibration falls back to its single global block.

[debugging-lore] Gridap may wrap an `FEOperatorFromWeakForm` inside `AlgebraicOpFromFEOp` before it
reaches `Gridap.Algebra.solve!`; the wrapper exposes only a `.feop` field, while the bare
`FEOperatorFromWeakForm` has `.test` directly. This helper checks for both.

Used by `_update_merit_weights!` (§3.1, block-equilibrated merit).
"""
function _detect_field_blocks(op::NonlinearOperator)
    test_space = nothing
    if hasproperty(op, :feop)
        test_space = get_test(op.feop)
    elseif hasproperty(op, :test)
        test_space = op.test
    end

    if test_space isa MultiFieldFESpace
        offsets = Int[0]
        for field in test_space.spaces
            push!(offsets, offsets[end] + num_free_dofs(field))
        end
        return [(offsets[i] + 1):offsets[i + 1] for i in 1:length(offsets) - 1]
    end
    return nothing
end

"""
    _update_merit_weights!(w, d_A, field_blocks)

Refreshes the row-equilibration weight vector `w` used by the merit function `Φ = ½ ‖b./w‖²`,
reading the Jacobian diagonal `d_A`. With a single block (`field_blocks === nothing`) it uses the
global rule `w_k = max(|J_kk|, eps·‖diag J‖_∞, eps)`. With multiple blocks it applies the same rule
independently per field block (§3.1, block-equilibrated merit): velocity rows are scaled by the
velocity-block diagonal magnitude, pressure rows by the pressure-block diagonal magnitude. Without
this, the velocity scale dominates the merit (in the saddle-point case the pressure-pressure block
has essentially zero diagonal apart from the stabilization term), biasing the line search to
under-weight mass-residual decrease relative to momentum-residual decrease.
"""
function _update_merit_weights!(w, d_A, field_blocks)
    if field_blocks === nothing
        w .= max.(abs.(d_A), eps(Float64) * norm(d_A, Inf), eps(Float64))
    else
        for rng in field_blocks
            block_max = 0.0
            @inbounds for j in rng
                a = abs(d_A[j])
                if a > block_max
                    block_max = a
                end
            end
            scale = max(eps(Float64) * block_max, eps(Float64))
            @inbounds for j in rng
                w[j] = max(abs(d_A[j]), scale)
            end
        end
    end
end

# The inner solve (Algorithm A). Runs the safeguarded Newton/Picard loop on the operator `op`,
# starting from `x` (overwritten in place with the result). Detects the field blocks, freezes the
# per-field convergence thresholds from the initial residual, then iterates assemble → linear solve →
# Armijo line search → safeguard cascade until a stop fires. On any non-convergent structural exit it
# rolls `x` back to the best-residual iterate seen. Returns `(x, SafeSolverCache)`; the cache's
# `result` records the stop reason and the per-iteration trajectory.
function _safe_solve_inner!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, b, A, dx, ls_cache)
    check_solver_parameters(solver)

    # Per-field DOF index ranges for block-equilibrated merit (§3.1). `nothing` for a single-field
    # test space, in which case a single global equilibration block is used.
    field_blocks = _detect_field_blocks(op)
    # If the solver carries per-field tolerances, the field count must match.
    nfields = field_blocks === nothing ? 1 : length(field_blocks)
    if solver.relative_ftol_per_field !== nothing && length(solver.relative_ftol_per_field) != nfields
        throw(ArgumentError("relative_ftol_per_field has length $(length(solver.relative_ftol_per_field)) but the operator's test space has $nfields field(s)."))
    end
    if solver.relative_stagnation_noise_floor_per_field !== nothing && length(solver.relative_stagnation_noise_floor_per_field) != nfields
        throw(ArgumentError("relative_stagnation_noise_floor_per_field has length $(length(solver.relative_stagnation_noise_floor_per_field)) but the operator's test space has $nfields field(s)."))
    end

    println("    [+] Evaluating initial PDE residual...")
    residual!(b, op, x)
    norm_b_inf = norm(b, Inf)
    norm_b_per_field = Vector{Float64}(undef, nfields)
    norm_b_new_per_field = Vector{Float64}(undef, nfields)
    _per_field_inf_norms!(norm_b_per_field, b, field_blocks)
    copyto!(norm_b_new_per_field, norm_b_per_field)

    # [code-actual][known-fragility] Per-field convergence gate: the DIMENSIONLESS relative reduction
    # `‖R_k‖ ≤ rel·‖R₀_k‖`. The scale is the INITIAL per-field residual norm ‖R₀_k‖ (same units as
    # `R_k`), frozen at iter 0. Scaling by the residual — not the solution magnitude ‖x‖ (∝U velocity,
    # ∝U² pressure) — keeps the gate encoding-invariant: under OSGS's single-step Picard-type coupling a
    # solution-scaled gate is dimensional and makes the converged result non-covariant (worst in
    # pressure). Guarded by encoding_invariance_test.jl.
    initial_residual_scale_per_field = copy(norm_b_per_field)   # ‖R₀_k‖ (the residual scale), frozen
    effective_ftol_per_field, effective_noise_floor_per_field =
        _initialize_effective_thresholds(solver, initial_residual_scale_per_field)

    # Merit row-equilibration weights, refreshed from the Jacobian diagonal each iteration.
    w = ones(length(b))

    phi_x = _merit_W(b, w)
    println("Iter 0: f(x) inf-norm = $norm_b_inf | Merit Φ = $phi_x")

    # [trajectory] Initial normalized residual: maxₖ(‖R₀[block_k]‖_∞ / effective_ftol_per_field[k]),
    # the scalar the per-field convergence gate compares to 1.
    f_norm0 = maximum(norm_b_new_per_field ./ effective_ftol_per_field)

    # [Fix A] When the scale-free convergence evaluator is attached (`conv_probe`), it is the
    # AUTHORITATIVE success gate (ε_M ≤ tol_M, ε_C ≤ tol_C), superseding the per-segment-RE-ANCHORED
    # per-field ftol gate below — D_M is read from the iterate, not a frozen ‖R₀_k‖, so the gate is the
    # SAME across ping-pong segments. The f_norm machinery still runs (merit normalization + the trace
    # diagnostic), but no longer decides success.
    scale_free = solver.conv_probe !== nothing

    # Initial short-circuit: if the entry iterate already meets the convergence gate, return immediately
    # (zero iterations, stop_reason "initial_ftol"). Under the scale-free gate the test is the criterion
    # verdict with the `degenerate` flag standing in for the spec's k≥1 rule — it rejects the trivial
    # all-zero/roundoff entry (where ε is meaningless) while ACCEPTING a developed, already-converged one.
    # This is load-bearing for the OSGS coupled solve, whose entry is the (already-converged) Stage-I
    # state: without it the solve is forced to take a step on a solved state and the line search depletes
    # trying to improve an already-minimal merit — a spurious `linesearch_failed`. (`b = residual(x)` here,
    # so the probe reads a self-consistent (iterate, residual) pair.)
    if scale_free
        cm0 = nothing
        try
            cm0 = solver.conv_probe(x, b, field_blocks)
        catch
            cm0 = nothing
        end
        if cm0 !== nothing && !cm0.degenerate && cm0.converged
            res = SafeSolverResult(0, norm_b_inf, norm_b_inf, 0.0, "initial_ftol", NamedTuple[], f_norm0, f_norm0)
            return x, SafeSolverCache(b, A, dx, ls_cache, res)
        end
    elseif _residual_meets_per_field_ftol(norm_b_new_per_field, effective_ftol_per_field)
        res = SafeSolverResult(0, norm_b_inf, norm_b_inf, 0.0, "initial_ftol", NamedTuple[], f_norm0, f_norm0)
        return x, SafeSolverCache(b, A, dx, ls_cache, res)
    end

    initial_b_inf = norm_b_inf

    state = SafeIterationState(
        0, 0, 0, false, 1.0, 0.0, norm_b_inf, norm_b_inf, phi_x, phi_x,
        norm_b_per_field, norm_b_new_per_field,
        initial_residual_scale_per_field, effective_ftol_per_field, effective_noise_floor_per_field,
    )
    
    x_old = similar(x)
    best_x = copy(x)            # best-residual iterate, restored on non-convergent structural exits
    best_b_inf = norm_b_inf
    stop_reason = "max_iters"
    # [trajectory] per-iteration record of this inner run; [stall guard] iter of last meaningful improvement.
    iteration_history = NamedTuple[]
    last_improve_iter = 0
    # [trajectory] last recorded normalized residual (the per-field convergence quantity); seeded to the
    # initial value so a 0-iteration exit still reports a sensible final normalized residual.
    last_f_norm = f_norm0

    for i in 1:solver.max_iters
        state.i = i
        
        jacobian!(A, op, x)
        
        # Re-equilibrate the merit weights from the current Jacobian diagonal. Block-equilibrated
        # when the test space is multi-field (§3.1); a single global block otherwise.
        d_A = diag(A)
        _update_merit_weights!(w, d_A, field_blocks)
        state.phi_x = _merit_W(b, w)
        
        solve_failed, ls_cache = eval_linear_system_resolution!(dx, A, b, solver, ls_cache)
        
        if solve_failed
            println("  [Fatal] Linear solve failed or yielded NaNs. Aborting safely.")
            x .= best_x
            state.norm_b_inf = best_b_inf
            stop_reason = "linear_solve_nan"
            break
        end
        
        # Directional derivative of Φ along the Newton step. For the exact-Newton direction the
        # algebra collapses to D = -2Φ; this is the slope the Armijo sufficient-decrease test uses.
        dir_deriv = -2.0 * state.phi_x
        
        x_old .= x
        eval_armijo_linesearch_pass!(x, x_old, b, dx, op, dir_deriv, solver, w, state, field_blocks)
        
        if state.ls_success
            println("Iter $i: f(x) inf-norm = $(state.norm_b_new_inf) | Merit Φ = $(state.phi_x_new) | Step inf-norm = $(state.step_norm) | alpha = $(state.alpha)")
        end

        # [trajectory] record this iteration's residual / normalized residual / merit / step / alpha
        # (and whether the Armijo line search accepted) for later analysis of the exact nonlinear path.
        # f_norm = maxₖ(‖R[block_k]‖_∞ / effective_ftol_per_field[k]) is the scalar compared to 1.
        f_norm = maximum(state.norm_b_new_per_field ./ state.effective_ftol_per_field)
        last_f_norm = f_norm
        # [diagnostic] Scale-free convergence norms ε_M, ε_C at this iterate (see `conv_probe`). A PURE
        # read-only observer: evaluated only on an ACCEPTED step, where `b = residual(x)` so the
        # (iterate, residual) pair the probe reads is self-consistent; wrapped in try/catch so a probe
        # failure degrades to NaN and can never break or perturb the solve. NaN ⇒ not traced this step.
        eps_M_it, eps_C_it = NaN, NaN
        cm = nothing
        if solver.conv_probe !== nothing && state.ls_success
            try
                cm = solver.conv_probe(x, b, field_blocks)
                eps_M_it, eps_C_it = cm.eps_M, cm.eps_C
            catch
                cm = nothing
                eps_M_it, eps_C_it = NaN, NaN
            end
        end
        push!(iteration_history, (i = i, f_inf = state.norm_b_new_inf, f_norm = f_norm,
                                  eps_M = eps_M_it, eps_C = eps_C_it,
                                  merit = state.phi_x_new, step_inf = state.step_norm,
                                  alpha = state.alpha, accepted = state.ls_success))

        # [Fix A] Authoritative scale-free success gate. Checked BEFORE the stall / Picard-gain / safeguard
        # logic so a converged iterate is never misread as a no-progress stall when its final residual
        # improvements fall below the stall threshold (i.e. it has reached the floor). The just-accepted
        # trial iterate `x` is the answer; a degenerate verdict (denominator at the underflow floor) is
        # never accepted. The cascade detects this `ftol_reached` exit as success.
        if scale_free && cm !== nothing && !cm.degenerate && cm.converged
            state.norm_b_inf = state.norm_b_new_inf
            stop_reason = "ftol_reached"
            break
        end

        if state.norm_b_new_inf < best_b_inf
            # [stall guard] only a MEANINGFUL improvement of the best residual resets the window.
            if state.norm_b_new_inf < best_b_inf * (1.0 - solver.stall_min_rel_improvement)
                last_improve_iter = i
            end
            best_x .= x
            best_b_inf = state.norm_b_new_inf
        end

        # [stall guard] bail a doomed attempt early (e.g. a stiff high-Re guess grinding the full
        # Newton budget with the merit stuck) so the caller falls back to a milder homotopy step
        # rather than exhausting max_iters. Disabled when stall_window == 0.
        if solver.stall_window > 0 && (i - last_improve_iter) >= solver.stall_window
            x .= best_x
            state.norm_b_inf = best_b_inf
            stop_reason = "no_progress_stall"
            println("  [No-Progress Stall] best residual ($best_b_inf) failed to improve by ≥$(solver.stall_min_rel_improvement) over $(solver.stall_window) iterations — bailing for caller fallback.")
            break
        end

        # [P4 ping-pong] Picard gain-then-return stop. In :picard mode with a finite gain target, stop
        # as soon as ‖R‖∞ has dropped `picard_gain_target` orders below the initial residual ‖R₀‖∞
        # (initial_b_inf) — just enough to re-enter Newton's basin, so the ping-pong orchestrator hands
        # back to Newton without running Picard to its full ftol/cap. Inert when picard_gain_target == Inf.
        # Uses the frozen global ‖R₀‖∞ (a re-entry heuristic), independent of the per-field convergence gate.
        if solver.mode === :picard && isfinite(solver.picard_gain_target) &&
           state.norm_b_new_inf <= initial_b_inf * 10.0^(-solver.picard_gain_target)
            x .= best_x
            state.norm_b_inf = best_b_inf
            stop_reason = "picard_gain_reached"
            break
        end

        reason, converged, should_break = eval_safeguard_termination_bounds!(solver, state, scale_free)

        if should_break
            # On a genuine divergence/failure exit, roll back to the best-residual iterate; on a
            # convergence or near-convergence stop, keep the just-computed trial iterate as the answer.
            if !converged && reason != "ftol_reached" && reason != "stagnation_noise_floor_reached" && reason != "xtol_stagnation"
                 x .= best_x
                 state.norm_b_inf = best_b_inf
            else
                 state.norm_b_inf = state.norm_b_new_inf
            end
            stop_reason = reason
            break
        end

        state.phi_x = state.phi_x_new
        state.norm_b_inf = state.norm_b_new_inf
        copyto!(state.norm_b_per_field, state.norm_b_new_per_field)
    end
    
    res = SafeSolverResult(state.i, state.norm_b_inf, initial_b_inf, state.step_norm, stop_reason,
                           iteration_history, f_norm0, last_f_norm)
    return x, SafeSolverCache(b, A, dx, ls_cache, res)
end

# Gridap entry point, cold start: allocate the residual/Jacobian/step workspace, then run the inner
# solve. This is the dispatch Gridap calls when no cache exists yet.
function Gridap.Algebra.solve!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, cache::Nothing)
    b = allocate_residual(op, x)
    A = allocate_jacobian(op, x)
    dx = similar(b, axes(A, 2))
    fill!(dx, zero(eltype(dx)))
    return _safe_solve_inner!(x, solver, op, b, A, dx, nothing)
end

# Gridap entry point, warm start: reuse the workspace and linear-solver cache from a previous solve.
# The orchestrator threads this through the Newton/Picard/homotopy cascade to avoid reallocation.
function Gridap.Algebra.solve!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, cache::SafeSolverCache)
    return _safe_solve_inner!(x, solver, op, cache.b, cache.A, cache.dx, cache.ls_cache)
end
