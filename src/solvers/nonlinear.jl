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
end

function SafeNewtonSolver(ls::LinearSolver, cfg::SolverConfig)
    return SafeNewtonSolver(
        ls, cfg.newton_iterations, cfg.max_increases, cfg.xtol, cfg.ftol,
        cfg.linesearch_alpha_min, cfg.armijo_c1, cfg.divergence_merit_factor, cfg.stagnation_noise_floor
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
    final_step_norm = 0.0
    final_i = 0
    inc_count = 0
    
    x_old = similar(x)
    best_x = copy(x)
    best_b_inf = norm_b_inf
    stop_reason = "max_iters"
    
    for i in 1:solver.max_iters
        final_i = i
        
        jacobian!(A, op, x)
        
        # Dynamically scale the merit function using the exact Jacobian diagonal
        d_A = diag(A)
        w .= max.(abs.(d_A), eps(Float64) * norm(d_A, Inf), eps(Float64))
        phi_x = eval_merit_W(b)
        
        local solve_failed = false
        try
            ls_cache = ls_cache === nothing ? symbolic_setup(solver.ls, A) : ls_cache
            ns = numerical_setup(ls_cache, A)
            solve!(dx, ns, b)
        catch e
            println("  [Linear Solve Exception] ", e)
            solve_failed = true
        end
        
        if solve_failed || any(isnan, dx)
            println("  [Fatal] Linear solve failed or yielded NaNs. Aborting safely.")
            x .= best_x
            norm_b_inf = best_b_inf
            stop_reason = "linear_solve_nan"
            break
        end
        
        # Exact Newton step is natively affine invariant. Directional derivative simplifies algebraically perfectly.
        dir_deriv = -2.0 * phi_x
        
        alpha = 1.0
        x_old .= x
        step_norm_base = norm(dx, Inf)
        ls_success = false
        norm_b_new_inf = norm_b_inf
        phi_x_new = phi_x
        
        for ls_iter in 1:15
            x .= x_old .- alpha .* dx
            residual!(b, op, x)
            phi_x_new = eval_merit_W(b)
            
            if isnan(phi_x_new)
                alpha *= 0.5
                continue
            end
            
            # Armijo Condition
            if phi_x_new <= phi_x + solver.c1 * alpha * dir_deriv
                ls_success = true
                norm_b_new_inf = norm(b, Inf)
                break
            end
            if alpha <= solver.linesearch_alpha_min
                break
            end
            alpha *= 0.5 
        end
        
        if !ls_success
            println("  [Fatal] Linesearch failed to find descent. Aborting Newton loop.")
            x .= best_x
            norm_b_inf = best_b_inf
            stop_reason = "linesearch_failed"
            break
        end
        
        step_norm = alpha * step_norm_base
        final_step_norm = step_norm
        println("Iter $i: f(x) inf-norm = $norm_b_new_inf | Merit Φ = $phi_x_new | Step inf-norm = $step_norm | alpha = $alpha")
        
        if norm_b_new_inf < best_b_inf
            best_x .= x
            best_b_inf = norm_b_new_inf
        end
        
        if norm_b_new_inf <= solver.ftol
            norm_b_inf = norm_b_new_inf
            stop_reason = "ftol_reached"
            break
        end
        
        if norm_b_new_inf <= solver.stagnation_noise_floor
            x .= best_x
            norm_b_inf = best_b_inf
            stop_reason = "stagnation_noise_floor_reached"
            break
        end
        
        if phi_x_new > phi_x * solver.divergence_merit_factor
            inc_count += 1
            if inc_count >= solver.max_increases
                println("  [Fatal] Divergence merit limit exceeded.")
                x .= best_x
                norm_b_inf = best_b_inf
                stop_reason = "merit_divergence_escaped"
                break
            end
        else
            inc_count = 0
        end
        
        if step_norm <= solver.xtol
            println("  [Stagnation] Step norm vanished below xtol.")
            x .= best_x
            norm_b_inf = best_b_inf
            stop_reason = "xtol_stagnation"
            break
        end
        
        if i == solver.max_iters
            x .= best_x
            norm_b_inf = best_b_inf
            stop_reason = "max_iters_stagnation"
        end
        
        phi_x = phi_x_new
        norm_b_inf = norm_b_new_inf
    end
    
    res = SafeSolverResult(final_i, norm_b_inf, initial_b_inf, final_step_norm, stop_reason)
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
