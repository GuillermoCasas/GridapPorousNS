# src/solvers/nonlinear.jl
using Gridap
using Gridap.Algebra
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
end
function SafeNewtonSolver(ls::LinearSolver, cfg::SolverConfig)
    return SafeNewtonSolver(
        ls, cfg.newton_iterations, cfg.max_increases, cfg.xtol, cfg.ftol,
        cfg.linesearch_alpha_min, cfg.armijo_c1, cfg.divergence_merit_factor, cfg.stagnation_noise_floor,
        cfg.max_linesearch_iterations, cfg.linesearch_contraction_factor
    )
end

function check_solver_parameters(s::SafeNewtonSolver)
    if !(0.0 < s.c1 < 1.0) throw(ArgumentError("Armijo 'c1' must be in (0, 1).")) end
    if s.divergence_merit_factor < 1.0 throw(ArgumentError("Divergence merit factor must be >= 1.0.")) end
    if s.xtol <= 0.0 || s.ftol <= 0.0 || s.stagnation_noise_floor <= 0.0 throw(ArgumentError("Tolerances must be > 0.")) end
    if s.linesearch_alpha_min <= 0.0 || s.linesearch_alpha_min > 1.0 throw(ArgumentError("alpha_min must be in (0, 1].")) end
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
        
        if isnan(state.phi_x_new)
            alpha *= solver.linesearch_contraction_factor
            continue
        end
        
        # Armijo Condition
        if state.phi_x_new <= state.phi_x + solver.c1 * alpha * dir_deriv
            state.ls_success = true
            state.norm_b_new_inf = norm(b, Inf)
            break
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
        if state.norm_b_new_inf <= solver.stagnation_noise_floor
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Linesearch Depleted] Geometric step fraction alpha vanished below minimum limits without sufficient mathematical descent. Aborting local sequence.")
        return "linesearch_failed", false, true
    end
    
    if state.phi_x_new > state.phi_x * solver.divergence_merit_factor
        state.inc_count += 1
        if state.inc_count >= solver.max_increases
            println("  [Merit Divergence] Sequence catastrophically expanded beyond allowed bounds. Extracted state exhibits unbounded divergence.")
            return "merit_divergence_escaped", false, true
        end
    else
        state.inc_count = 0
    end
    
    if state.step_norm <= solver.xtol
        if state.norm_b_new_inf <= solver.stagnation_noise_floor
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Coordinate Stagnation] Step update magnitude collapsed below relative machine tracking limits (xtol = $(solver.xtol)). Algebraic progress saturated.")
        return "xtol_stagnation", false, true
    end
    
    if state.i == solver.max_iters
        if state.norm_b_new_inf <= solver.stagnation_noise_floor
            return "stagnation_noise_floor_reached", true, true
        end
        println("  [Iteration Cap] Sequence hit maximum bounded loops ($(state.i)) without geometrically saturating non-linear constraints.")
        return "max_iters_stagnation", false, true
    end
    
    return stop_reason, converged, should_break
end

function _safe_solve_inner!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, b, A, dx, ls_cache)
    check_solver_parameters(solver)
    
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
        
        # Dynamically scale the merit function using the exact Jacobian diagonal
        d_A = diag(A)
        w .= max.(abs.(d_A), eps(Float64) * norm(d_A, Inf), eps(Float64))
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
