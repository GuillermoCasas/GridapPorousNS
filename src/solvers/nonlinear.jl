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
end

# Convenience constructor: keeps existing positional call sites untouched (the honest-exit gate
# defaults to Inf = disabled, so legacy behaviour is bit-identical unless a caller opts in).
function SafeNewtonSolver(ls::LinearSolver, max_iters::Int, max_increases::Int,
                          xtol::Float64, ftol::Float64, linesearch_alpha_min::Float64,
                          c1::Float64, divergence_merit_factor::Float64,
                          stagnation_noise_floor::Float64, max_linesearch_iterations::Int,
                          linesearch_contraction_factor::Float64; mode::Symbol = :newton,
                          noise_floor_success_max_ftol_multiple::Float64 = Inf)
    return SafeNewtonSolver(ls, max_iters, max_increases, xtol, ftol, linesearch_alpha_min,
                            c1, divergence_merit_factor, stagnation_noise_floor,
                            max_linesearch_iterations, linesearch_contraction_factor, mode,
                            noise_floor_success_max_ftol_multiple)
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
function _with_overrides(nls::SafeNewtonSolver; max_iters=nothing, ftol=nothing, mode=nothing)
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
    )
end

function check_solver_parameters(s::SafeNewtonSolver)
    if !(0.0 < s.c1 < 1.0) throw(ArgumentError("Armijo 'c1' must be in (0, 1).")) end
    if s.divergence_merit_factor < 1.0 throw(ArgumentError("Divergence merit factor must be >= 1.0.")) end
    if s.xtol <= 0.0 || s.ftol <= 0.0 || s.stagnation_noise_floor <= 0.0 throw(ArgumentError("Tolerances must be > 0.")) end
    if s.linesearch_alpha_min <= 0.0 || s.linesearch_alpha_min > 1.0 throw(ArgumentError("alpha_min must be in (0, 1].")) end
    if !(s.mode in (:newton, :picard)) throw(ArgumentError("SafeNewtonSolver mode must be :newton or :picard, got $(s.mode).")) end
    if s.noise_floor_success_max_ftol_multiple < 1.0 throw(ArgumentError("noise_floor_success_max_ftol_multiple must be >= 1.0 (Inf disables the honest-exit gate).")) end
end

struct SafeSolverResult
    iterations::Int
    residual_norm::Float64
    initial_residual_norm::Float64
    step_norm::Float64
    stop_reason::String
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
end

# Algorithm A.2: Armijo Linesearch Pass
function eval_armijo_linesearch_pass!(x, x_old, b, dx, op, dir_deriv, solver::SafeNewtonSolver, w, state::SafeIterationState)
    alpha = 1.0
    step_norm_base = norm(dx, Inf)
    state.ls_success = false
    state.norm_b_new_inf = norm(b, Inf)
    state.phi_x_new = state.phi_x
    
    eval_merit_W(b_vec) = 0.5 * sum(idx -> (b_vec[idx] / w[idx])^2, eachindex(b_vec))
    
    for ls_iter in 1:solver.max_linesearch_iterations
        x .= x_old .- alpha .* dx
        residual!(b, op, x)
        state.phi_x_new = eval_merit_W(b)
        state.norm_b_new_inf = norm(b, Inf)

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
    
    if state.norm_b_new_inf <= solver.ftol
        return "ftol_reached", true, true
    end

    if !state.ls_success
        if state.norm_b_new_inf <= solver.stagnation_noise_floor &&
           state.norm_b_new_inf <= solver.noise_floor_success_max_ftol_multiple * solver.ftol
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
        if state.norm_b_new_inf <= solver.stagnation_noise_floor &&
           state.norm_b_new_inf <= solver.noise_floor_success_max_ftol_multiple * solver.ftol
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Coordinate Stagnation] Step update magnitude collapsed below relative machine tracking limits (xtol = $(solver.xtol)). Algebraic progress saturated.")
        return "xtol_stagnation", false, true
    end
    
    if state.i == solver.max_iters
        if state.norm_b_new_inf <= solver.stagnation_noise_floor &&
           state.norm_b_new_inf <= solver.noise_floor_success_max_ftol_multiple * solver.ftol
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

    println("    [+] Evaluating initial PDE residual...")
    residual!(b, op, x)
    norm_b_inf = norm(b, Inf)

    # Preallocate zero-allocation weights
    w = ones(length(b))
    eval_merit_W(b_vec) = 0.5 * sum(idx -> (b_vec[idx] / w[idx])^2, eachindex(b_vec))
    
    phi_x = eval_merit_W(b)
    println("Iter 0: f(x) inf-norm = $norm_b_inf | Merit Φ = $phi_x")
    
    if norm_b_inf <= solver.ftol
        res = SafeSolverResult(0, norm_b_inf, norm_b_inf, 0.0, "initial_ftol")
        return x, SafeSolverCache(b, A, dx, ls_cache, res)
    end
    
    initial_b_inf = norm_b_inf
    
    state = SafeIterationState(
        0, 0, false, 1.0, 0.0, norm_b_inf, norm_b_inf, phi_x, phi_x
    )
    
    x_old = similar(x)
    best_x = copy(x)
    best_b_inf = norm_b_inf
    stop_reason = "max_iters"
    
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
        eval_armijo_linesearch_pass!(x, x_old, b, dx, op, dir_deriv, solver, w, state)
        
        if state.ls_success
            println("Iter $i: f(x) inf-norm = $(state.norm_b_new_inf) | Merit Φ = $(state.phi_x_new) | Step inf-norm = $(state.step_norm) | alpha = $(state.alpha)")
        end
        
        if state.norm_b_new_inf < best_b_inf
            best_x .= x
            best_b_inf = state.norm_b_new_inf
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
    end
    
    res = SafeSolverResult(state.i, state.norm_b_inf, initial_b_inf, state.step_norm, stop_reason)
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
