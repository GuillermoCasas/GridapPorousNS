# src/solvers/nonlinear.jl
using Gridap
using Gridap.Algebra
using Gridap.FESpaces: get_test, num_free_dofs
using Gridap.MultiField: MultiFieldFESpace
using LinearAlgebra
using SparseArrays: diag

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
    # [honest-exit] A noise-floor stop (‖R‖∞ ≤ stagnation_noise_floor) is reported as CONVERGED
    # only if ‖R‖∞ ≤ noise_floor_success_max_ftol_multiple · ftol. Default Inf ⇒ gate DISABLED
    # (legacy: any noise-floor stop is "success"). A finite value (e.g. 10.0) rejects high-Re
    # fold stalls — where ‖R‖≈1e-5 sits ~10²–10³× above ftol — from masquerading as a true root.
    noise_floor_success_max_ftol_multiple::Float64
    # Per-field RELATIVE convergence tolerances. When set, the solver derives absolute
    # per-field thresholds at iter 0 from the SOLUTION-FIELD SCALE — the inf-norm of
    # the initial guess restricted to each field block:
    #     effective_ftol_per_field[k] = max(ftol, relative_ftol_per_field[k] * ‖x₀[block_k]‖_∞)
    # All termination checks then use those per-field absolute thresholds. The scalar
    # `ftol` is the per-field absolute FLOOR; `relative_ftol_per_field[k]` is the
    # dimensionless target (e.g., c_sf·h^(k+1) ≈ 1e-4 — "1% of the FE discretization
    # error level relative to the solution field's magnitude"). When `nothing` (default),
    # the solver uses the scalar `ftol` and `stagnation_noise_floor` globally —
    # bit-identical to legacy.
    #
    # Why "intrinsic": `‖x₀[block_k]‖_∞` is the magnitude of the solution field we are
    # converging TOWARDS (modulo perturbations the harness/orchestrator may have
    # introduced; for MMS this is ≈ ‖u_ex‖, for production it reflects the user's
    # initial guess or Dirichlet BC). It does NOT depend on knowing the analytic exact
    # solution or any other problem-specific external scale. When ‖x₀[block_k]‖ is zero
    # (e.g., a production solve starting from a true-zero IC with all-Dirichlet
    # boundary conditions also at zero), the scale collapses and effective_ftol_per_field[k]
    # falls back to the scalar `ftol` floor — legacy behavior.
    relative_ftol_per_field::Union{Nothing, Vector{Float64}}
    relative_stagnation_noise_floor_per_field::Union{Nothing, Vector{Float64}}
    # [no-progress stall guard] If `stall_window > 0`, the solve aborts with stop_reason
    # "no_progress_stall" once the BEST residual inf-norm fails to improve by the relative
    # factor `stall_min_rel_improvement` over `stall_window` consecutive iterations. This bails
    # doomed attempts — e.g. a high-Re eps_pert=1.0 guess that grinds the full Newton budget with
    # merit stuck ~1e19 then linesearch-depletes — in a handful of iterations, so the caller falls
    # back to a milder eps_pert instead of exhausting max_iters. `stall_window = 0` ⇒ DISABLED
    # (the default; the production single-run path constructs the solver without it and is unchanged).
    stall_window::Int
    stall_min_rel_improvement::Float64
end

# Convenience constructor: keeps existing positional call sites untouched (the honest-exit gate
# defaults to Inf = disabled, so legacy behaviour is bit-identical unless a caller opts in).
function SafeNewtonSolver(ls::LinearSolver, max_iters::Int, max_increases::Int,
                          xtol::Float64, ftol::Float64, linesearch_alpha_min::Float64,
                          c1::Float64, divergence_merit_factor::Float64,
                          stagnation_noise_floor::Float64, max_linesearch_iterations::Int,
                          linesearch_contraction_factor::Float64; mode::Symbol = :newton,
                          noise_floor_success_max_ftol_multiple::Float64 = Inf,
                          relative_ftol_per_field::Union{Nothing, Vector{Float64}} = nothing,
                          relative_stagnation_noise_floor_per_field::Union{Nothing, Vector{Float64}} = nothing,
                          stall_window::Int = 0,
                          stall_min_rel_improvement::Float64 = 0.0)
    return SafeNewtonSolver(ls, max_iters, max_increases, xtol, ftol, linesearch_alpha_min,
                            c1, divergence_merit_factor, stagnation_noise_floor,
                            max_linesearch_iterations, linesearch_contraction_factor, mode,
                            noise_floor_success_max_ftol_multiple,
                            relative_ftol_per_field, relative_stagnation_noise_floor_per_field,
                            stall_window, stall_min_rel_improvement)
end

function SafeNewtonSolver(ls::LinearSolver, cfg::SolverConfig; mode::Symbol = :newton)
    return SafeNewtonSolver(
        ls, cfg.newton_iterations, cfg.max_increases, cfg.xtol, cfg.ftol,
        cfg.linesearch_alpha_min, cfg.armijo_c1, cfg.divergence_merit_factor, cfg.stagnation_noise_floor,
        cfg.max_linesearch_iterations, cfg.linesearch_contraction_factor; mode=mode,
        noise_floor_success_max_ftol_multiple=cfg.noise_floor_success_max_ftol_multiple
    )
end

# Copy-with-overrides for the three fields ever overridden at call sites in
# `porous_solver.jl` (`max_iters`, `ftol`, `mode`). All other fields are
# inherited from `nls` verbatim. `nothing` for any kwarg means "keep `nls`'s
# value". Replaces the previous verbose field-by-field SafeNewtonSolver()
# rebuilds, which were vulnerable to silently dropping new struct fields.
function _with_overrides(nls::SafeNewtonSolver; max_iters=nothing, ftol=nothing, mode=nothing,
                         relative_ftol_per_field=nothing)
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
        # `relative_ftol_per_field` override (covariant OSGS warmup relaxation): nothing ⇒ keep base.
        isnothing(relative_ftol_per_field) ? nls.relative_ftol_per_field : relative_ftol_per_field,
        nls.relative_stagnation_noise_floor_per_field,
        nls.stall_window,
        nls.stall_min_rel_improvement,
    )
end

function check_solver_parameters(s::SafeNewtonSolver)
    if !(0.0 < s.c1 < 1.0) throw(ArgumentError("Armijo 'c1' must be in (0, 1).")) end
    if s.divergence_merit_factor < 1.0 throw(ArgumentError("Divergence merit factor must be >= 1.0.")) end
    if s.xtol <= 0.0 || s.ftol <= 0.0 || s.stagnation_noise_floor <= 0.0 throw(ArgumentError("Tolerances must be > 0.")) end
    if s.linesearch_alpha_min <= 0.0 || s.linesearch_alpha_min > 1.0 throw(ArgumentError("alpha_min must be in (0, 1].")) end
    if !(s.mode in (:newton, :picard)) throw(ArgumentError("SafeNewtonSolver mode must be :newton or :picard, got $(s.mode).")) end
    if s.noise_floor_success_max_ftol_multiple < 1.0 throw(ArgumentError("noise_floor_success_max_ftol_multiple must be >= 1.0 (Inf disables the honest-exit gate).")) end
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

struct SafeSolverResult
    iterations::Int
    residual_norm::Float64
    initial_residual_norm::Float64
    step_norm::Float64
    stop_reason::String
    # [trajectory] Per-iteration record of this Algorithm-A (ExactNewtonPipeline) run: a Vector of
    # NamedTuples (i, f_inf, f_norm, merit, step_inf, alpha, accepted). Empty for the initial-ftol
    # short-circuit. The orchestrator (Algorithm O) tags each invocation with a stage code
    # (e.g. "B:StageI:N1", "C:OSGS[k]:Picard") and assembles diag_cache["trajectory"] from these.
    iteration_history::Vector
    # [trajectory] Normalized residuals: the scalar actually compared to the convergence threshold,
    # f_norm = maxₖ(‖R[block_k]‖_∞ / effective_ftol_per_field[k]) (≤ 1 ⟺ converged). `‖R‖_∞`
    # above is field-blind and uses the machine ftol floor; these track the per-field gate instead.
    initial_residual_normalized::Float64
    residual_normalized::Float64
end

struct SafeSolverCache
    b::AbstractVector
    A::AbstractMatrix
    dx::AbstractVector
    ls_cache::Any
    result::SafeSolverResult
end

# Algorithm A.1: Linear Assembly and Resolution
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

mutable struct SafeIterationState
    i::Int
    inc_count::Int
    ls_success::Bool
    alpha::Float64
    step_norm::Float64
    norm_b_inf::Float64
    norm_b_new_inf::Float64
    phi_x::Float64
    phi_x_new::Float64
    # Per-field inf-norms of the residual, computed in lock-step with the global
    # `norm_b_*` scalars. `field_blocks === nothing` (single-field operator) leaves
    # these as a 1-element vector matching the global value, so downstream code
    # uniformly indexes `[k]` regardless.
    norm_b_per_field::Vector{Float64}
    norm_b_new_per_field::Vector{Float64}
    # Per-field initial-residual inf-norms ‖R₀_k‖, frozen at iteration 0. They convert the solver's
    # dimensionless `relative_*_per_field` targets into absolute per-field thresholds, so the
    # convergence gate is a dimensionless relative reduction `‖R_k‖ ≤ rel·‖R₀_k‖`. Scaling by the
    # residual norm (same units as `R_k`), not the solution magnitude `‖x‖`, is what keeps the gate
    # encoding-invariant. A true-zero initial residual on every field collapses the thresholds to the
    # scalar `ftol`/`stagnation_noise_floor` floors. (Name kept for history; it now holds ‖R₀‖.)
    solution_scale_per_field::Vector{Float64}
    # Per-field absolute thresholds, derived once from `solution_scale_per_field` and the solver's
    # `relative_*_per_field` (or the scalar `ftol`/`stagnation_noise_floor` floors when no relative
    # tolerances are set).
    effective_ftol_per_field::Vector{Float64}
    effective_noise_floor_per_field::Vector{Float64}
end

"""
    _resolve_solution_scale_per_field(raw_scales) -> Vector{Float64}

When one field's initial-guess inf-norm is zero (typical production case: zero
pressure IC even when velocity has a Dirichlet-imposed magnitude), fall back to
a dimensionally consistent proxy derived from the other field. For the standard
(u, p) saddle-point case at ρ = 1:

  • ‖u‖ has units of velocity [L/T]
  • ‖p‖ has units of ρ·u² (Euler pressure scaling) so ‖p‖^(1/2) has units of u

so if ‖u‖₀ = 0 we use `sqrt(‖p‖₀)` as the velocity-equivalent scale, and if
‖p‖₀ = 0 we use `‖u‖₀²` as the pressure-equivalent scale. This is the
"Bernoulli/total-head" combined scale heuristic and removes the per-field
zero-fallback hole that would otherwise force the solver back to legacy.

When ALL fields are zero (true-zero initial guess on every field), the resolved
scales stay zero and the effective thresholds collapse to the scalar `solver.ftol`
/ `solver.stagnation_noise_floor` floors — recovering legacy behavior.

For other field-count configurations (single field, or n > 2 multi-physics), no
saddle-point heuristic applies and the raw scales are returned unchanged.

The ρ = 1 assumption matches this codebase's convention (`PaperGeneralFormulation`
does not carry a ρ parameter; the dimensionless equation uses ρ = 1 throughout).
"""
function _resolve_solution_scale_per_field(raw_scales::Vector{Float64})
    if length(raw_scales) == 2
        u_scale, p_scale = raw_scales[1], raw_scales[2]
        if u_scale == 0.0 && p_scale > 0.0
            return [sqrt(p_scale), p_scale]
        elseif p_scale == 0.0 && u_scale > 0.0
            return [u_scale, u_scale^2]
        end
    end
    return copy(raw_scales)
end

"""
    _initialize_effective_thresholds(solver, solution_scale_per_field) -> (ftols, noise_floors)

Derives the per-field absolute thresholds the solver uses throughout the run.

`solution_scale_per_field[k]` is the per-field initial-residual norm ‖R₀_k‖, frozen at iteration 0
(see `SafeIterationState`). Scaling the gate by the residual norm keeps it encoding-invariant.

The user's `relative_*_per_field[k]` is the dimensionless target (e.g. `c_sf·h^(k+1)`).
The effective absolute threshold is `max(scalar_floor, relative_target * scale)` so
the scalar value remains a hard floor (the solver never tries to go tighter than
the user's absolute ftol, regardless of what relative×scale produces).
"""
function _initialize_effective_thresholds(solver::SafeNewtonSolver,
                                          solution_scale_per_field::Vector{Float64})
    nfields = length(solution_scale_per_field)
    ftols = Vector{Float64}(undef, nfields)
    noise_floors = Vector{Float64}(undef, nfields)

    if solver.relative_ftol_per_field !== nothing
        @assert length(solver.relative_ftol_per_field) == nfields "relative_ftol_per_field length mismatch"
        @inbounds for k in 1:nfields
            ftols[k] = max(solver.ftol, solver.relative_ftol_per_field[k] * solution_scale_per_field[k])
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
                                  solver.relative_stagnation_noise_floor_per_field[k] * solution_scale_per_field[k])
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

In-place fill of `out` with the per-field inf-norms of `b`. When
`field_blocks === nothing`, `out` is a 1-element vector containing `norm(b, Inf)`
(legacy/single-field path). Otherwise `out[k] = maximum(abs, view(b, field_blocks[k]))`.
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

Returns true iff every field's inf-norm is below its effective absolute threshold.
When `solver.relative_*_per_field === nothing`, `effective_ftols[k]` equals the
solver's scalar `ftol` for every k, so this collapses to a global inf-norm check —
bit-identical to legacy.
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

The "honest-exit" gate: a noise-floor stop counts as success only if every
field's residual is also within `k_nf × effective_ftols[k]`. Legacy collapses
to the global check when `effective_ftols` is uniform.
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

# Algorithm A.2: Armijo Linesearch Pass
function eval_armijo_linesearch_pass!(x, x_old, b, dx, op, dir_deriv, solver::SafeNewtonSolver, w, state::SafeIterationState,
                                       field_blocks::Union{Nothing, Vector{UnitRange{Int}}})
    alpha = 1.0
    step_norm_base = norm(dx, Inf)
    state.ls_success = false
    state.norm_b_new_inf = norm(b, Inf)
    _per_field_inf_norms!(state.norm_b_new_per_field, b, field_blocks)
    state.phi_x_new = state.phi_x

    eval_merit_W(b_vec) = 0.5 * sum(idx -> (b_vec[idx] / w[idx])^2, eachindex(b_vec))

    for ls_iter in 1:solver.max_linesearch_iterations
        x .= x_old .- alpha .* dx
        residual!(b, op, x)
        state.phi_x_new = eval_merit_W(b)
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

# Algorithm A.3: Safeguard Termination Bounds
function eval_safeguard_termination_bounds!(solver::SafeNewtonSolver, state::SafeIterationState)
    stop_reason = ""
    converged = false
    should_break = false

    k_nf = solver.noise_floor_success_max_ftol_multiple

    if _residual_meets_per_field_ftol(state.norm_b_new_per_field, state.effective_ftol_per_field)
        return "ftol_reached", true, true
    end

    if !state.ls_success
        if _residual_meets_per_field_noise_floor(state.norm_b_new_per_field, state.effective_noise_floor_per_field) &&
           _residual_meets_per_field_honest_exit_gate(state.norm_b_new_per_field, state.effective_ftol_per_field, k_nf)
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Linesearch Depleted] Geometric step fraction alpha vanished below minimum limits without sufficient mathematical descent. Aborting local sequence.")
        return "linesearch_failed", false, true
    end
    
    # Divergence safeguard. In :newton mode we compare Φ_new to Φ_old because the
    # Armijo line search is itself driving Φ down; if Φ grows by factor
    # `divergence_merit_factor`, something is structurally wrong.
    # In :picard mode the line search accepts on ‖b‖_∞ (line ~132), so we must
    # use the same metric here — Φ is meaningless across Picard iterations because
    # the merit weights `w = diag(J)` are refreshed each iter from a different Jacobian
    # (Picard's, which differs from Newton's), and Φ can grow even when ‖b‖_∞ shrinks
    # monotonically. Using Φ here causes premature `merit_divergence_escaped` exits
    # — see test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md.
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
    
    if state.step_norm <= solver.xtol
        if _residual_meets_per_field_noise_floor(state.norm_b_new_per_field, state.effective_noise_floor_per_field) &&
           _residual_meets_per_field_honest_exit_gate(state.norm_b_new_per_field, state.effective_ftol_per_field, k_nf)
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Coordinate Stagnation] Step update magnitude collapsed below relative machine tracking limits (xtol = $(solver.xtol)). Algebraic progress saturated.")
        return "xtol_stagnation", false, true
    end

    if state.i == solver.max_iters
        if _residual_meets_per_field_noise_floor(state.norm_b_new_per_field, state.effective_noise_floor_per_field) &&
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

Returns the per-field free-DOF index ranges of the operator's test space
when the space is a `MultiFieldFESpace` (assumes `ConsecutiveMultiFieldStyle`,
the only style this codebase uses). Returns `nothing` for single-field
spaces or when the test space cannot be introspected, in which case the
merit equilibration falls back to its legacy global rule.

Gridap may wrap an `FEOperatorFromWeakForm` inside `AlgebraicOpFromFEOp`
before it reaches `Gridap.Algebra.solve!`; the wrapper exposes only a
`.feop` field, while the bare `FEOperatorFromWeakForm` has `.test`
directly. This helper handles both.

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

Refreshes the row-equilibration weight vector `w` used by the Armijo merit
function `Φ = ½ ‖b./w‖²`. When `field_blocks === nothing`, falls back to the
legacy single-block rule `w_k = max(|J_kk|, eps·‖diag J‖_∞, eps)`. Otherwise
applies the same rule independently per field block (§3.1, block-
equilibrated merit): the velocity rows are equilibrated against the
velocity-block diagonal scale, the pressure rows against the pressure-
block diagonal scale. This prevents the velocity scale from dominating the
merit (the typical saddle-point case where pressure-pressure block has
zero diagonal except for the stabilization term) and removes the line-
search bias that under-weights mass-residual decrease relative to
momentum-residual decrease.
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

function _safe_solve_inner!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, b, A, dx, ls_cache)
    check_solver_parameters(solver)

    # Per-field DOF index ranges for block-equilibrated merit (§3.1). `nothing`
    # if the test space is single-field, in which case the legacy global
    # equilibration is used.
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

    # [code-actual] Per-field convergence gate: the DIMENSIONLESS relative reduction `‖R_k‖ ≤ rel·‖R₀_k‖`.
    # The scale is the INITIAL per-field residual norm ‖R₀_k‖ (same units as `R_k`), frozen at iter 0.
    # Scaling by the residual — not the solution magnitude ‖x‖ (∝U velocity, ∝U² pressure) — is what
    # keeps the gate encoding-invariant: under OSGS's single-inner-step staggering a solution-scaled gate
    # is dimensional and makes the converged result non-covariant (worst in pressure). Guarded by
    # encoding_invariance_test.jl.
    x_per_field_raw = Vector{Float64}(undef, nfields)   # retained for buffer parity; unused by the gate
    solution_scale_per_field = copy(norm_b_per_field)   # ‖R₀_k‖ (the residual scale), frozen
    effective_ftol_per_field, effective_noise_floor_per_field =
        _initialize_effective_thresholds(solver, solution_scale_per_field)

    # Preallocate zero-allocation weights
    w = ones(length(b))
    eval_merit_W(b_vec) = 0.5 * sum(idx -> (b_vec[idx] / w[idx])^2, eachindex(b_vec))

    phi_x = eval_merit_W(b)
    println("Iter 0: f(x) inf-norm = $norm_b_inf | Merit Φ = $phi_x")

    # [trajectory] Initial normalized residual: maxₖ(‖R₀[block_k]‖_∞ / effective_ftol_per_field[k]),
    # the scalar the per-field convergence gate compares to 1.
    f_norm0 = maximum(norm_b_new_per_field ./ effective_ftol_per_field)

    # Initial-ftol short-circuit using the per-field thresholds derived from x₀'s scale.
    if _residual_meets_per_field_ftol(norm_b_new_per_field, effective_ftol_per_field)
        res = SafeSolverResult(0, norm_b_inf, norm_b_inf, 0.0, "initial_ftol", NamedTuple[], f_norm0, f_norm0)
        return x, SafeSolverCache(b, A, dx, ls_cache, res)
    end

    initial_b_inf = norm_b_inf

    state = SafeIterationState(
        0, 0, false, 1.0, 0.0, norm_b_inf, norm_b_inf, phi_x, phi_x,
        norm_b_per_field, norm_b_new_per_field,
        solution_scale_per_field, effective_ftol_per_field, effective_noise_floor_per_field,
    )
    
    x_old = similar(x)
    best_x = copy(x)
    best_b_inf = norm_b_inf
    stop_reason = "max_iters"
    # [trajectory] per-iteration record of this Algorithm-A run; [stall guard] iter of last meaningful improvement.
    iteration_history = NamedTuple[]
    last_improve_iter = 0
    # [trajectory] last recorded normalized residual (the per-field convergence quantity); seeds to the
    # initial value so a 0-iteration exit still reports a sensible final normalized residual.
    last_f_norm = f_norm0

    for i in 1:solver.max_iters
        state.i = i
        
        jacobian!(A, op, x)
        
        # Dynamically scale the merit function using the exact Jacobian diagonal.
        # Block-equilibrated when the test space is multi-field (§3.1); falls back
        # to the legacy global rule for single-field problems.
        d_A = diag(A)
        _update_merit_weights!(w, d_A, field_blocks)
        state.phi_x = eval_merit_W(b)
        
        solve_failed, ls_cache = eval_linear_system_resolution!(dx, A, b, solver, ls_cache)
        
        if solve_failed
            println("  [Fatal] Linear solve failed or yielded NaNs. Aborting safely.")
            x .= best_x
            state.norm_b_inf = best_b_inf
            stop_reason = "linear_solve_nan"
            break
        end
        
        # Exact Newton step is natively affine invariant. Directional derivative simplifies algebraically perfectly.
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
        push!(iteration_history, (i = i, f_inf = state.norm_b_new_inf, f_norm = f_norm,
                                  merit = state.phi_x_new, step_inf = state.step_norm,
                                  alpha = state.alpha, accepted = state.ls_success))

        if state.norm_b_new_inf < best_b_inf
            # [stall guard] only a MEANINGFUL improvement of the best residual resets the window.
            if state.norm_b_new_inf < best_b_inf * (1.0 - solver.stall_min_rel_improvement)
                last_improve_iter = i
            end
            best_x .= x
            best_b_inf = state.norm_b_new_inf
        end

        # [stall guard] bail a doomed attempt early (e.g. a high-Re eps_pert=1.0 guess grinding the
        # full Newton budget with merit stuck) so the caller falls back to a milder eps_pert, rather
        # than exhausting max_iters. Disabled when stall_window == 0 (production single-run default).
        if solver.stall_window > 0 && (i - last_improve_iter) >= solver.stall_window
            x .= best_x
            state.norm_b_inf = best_b_inf
            stop_reason = "no_progress_stall"
            println("  [No-Progress Stall] best residual ($best_b_inf) failed to improve by ≥$(solver.stall_min_rel_improvement) over $(solver.stall_window) iterations — bailing for caller fallback.")
            break
        end

        reason, converged, should_break = eval_safeguard_termination_bounds!(solver, state)
        
        if should_break
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

function Gridap.Algebra.solve!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, cache::Nothing)
    b = allocate_residual(op, x)
    A = allocate_jacobian(op, x)
    dx = similar(b, axes(A, 2))
    fill!(dx, zero(eltype(dx)))
    return _safe_solve_inner!(x, solver, op, b, A, dx, nothing)
end

function Gridap.Algebra.solve!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, cache::SafeSolverCache)
    return _safe_solve_inner!(x, solver, op, cache.b, cache.A, cache.dx, cache.ls_cache)
end
