# src/safesolver.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

struct SafeNewtonSolver <: NonlinearSolver
    ls::LinearSolver
    max_iters::Int
    max_increases::Int
    xtol::Float64
    stagnation_tol::Float64
    ftol::Float64
    linesearch_tolerance::Float64
    linesearch_alpha_min::Float64
end

struct SafeSolverResult
    iterations::Int
    residual_norm::Float64
end

struct SafeSolverCache
    b::AbstractVector
    A::AbstractMatrix
    dx::AbstractVector
    ls_cache::Any
    result::SafeSolverResult
end

# Inner helper to keep code DRY and prevent memory leaks
function _safe_solve_inner!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, b, A, dx, ls_cache)
    inc_count = 0
    
    residual!(b, op, x)
    norm_b_inf = norm(b, Inf)
    norm_b_l2  = norm(b, 2)
    println("Iter 0: f(x) inf-norm = $norm_b_inf | L2-norm = $norm_b_l2")
    last_norm_l2 = norm_b_l2
    
    if norm_b_inf <= solver.ftol
        println("  [Convergence] Initial residual is below tolerance ($(solver.ftol)).")
        res = SafeSolverResult(0, norm_b_inf)
        return x, SafeSolverCache(b, A, dx, ls_cache, res)
    end
    
    final_i = 0
    x_old = similar(x)
    
    for i in 1:solver.max_iters
        final_i = i
        
        jacobian!(A, op, x)
        if ls_cache === nothing
            ls_cache = symbolic_setup(solver.ls, A)
        end
        ns = numerical_setup(ls_cache, A)
        solve!(dx, ns, b)
        
        alpha = 1.0
        x_old .= x
        step_norm_base = norm(dx, Inf)
        
        norm_b_new_inf = norm_b_inf
        norm_b_new_l2  = norm_b_l2
        
        ls_success = false
        for ls_iter in 1:15
            x .= x_old .- alpha .* dx
            residual!(b, op, x)
            norm_b_new_inf = norm(b, Inf)
            norm_b_new_l2  = norm(b, 2)
            
            if norm_b_new_l2 <= last_norm_l2 * solver.linesearch_tolerance 
                ls_success = true
                break
            end
            if alpha <= solver.linesearch_alpha_min
                break
            end
            alpha *= 0.5 
        end
        
        if !ls_success
            println("  [Stagnation] Linesearch failed to find a descent direction. Aborting Newton loop.")
            x .= x_old
            residual!(b, op, x) # Restore residual
            norm_b_inf = norm(b, Inf)
            break
        end
        
        step_norm = alpha * step_norm_base
        println("Iter $i: f(x) inf-norm = $norm_b_new_inf | L2-norm = $norm_b_new_l2 | Step inf-norm = $step_norm | alpha = $alpha")
        
        if norm_b_new_inf <= solver.ftol
            println("  [Convergence] Residual inf-norm ($norm_b_new_inf) is below tolerance ($(solver.ftol)).")
            norm_b_inf = norm_b_new_inf
            break
        end
        
        # SMART DIVERGENCE GUARD: Check L2 norm explosion (>5% growth to ignore noise)
        if isnan(norm_b_new_l2) || norm_b_new_l2 > last_norm_l2 * 1.05
            inc_count += 1
            if inc_count >= solver.max_increases
                if norm_b_new_inf > 1e-2
                    error("Solver Diverged: L2 Residual exploded or hit NaN for $(inc_count) iterations.")
                else
                    println("  [Stagnation] Solver hit the numerical noise floor. Stopping safely.")
                    norm_b_inf = norm_b_new_inf
                    break
                end
            end
        else
            inc_count = 0
        end
        last_norm_l2 = norm_b_new_l2
        
        # SMART STAGNATION GUARD: Graceful exit to OSGS stage if stuck near the root
        if step_norm <= solver.xtol
            if norm_b_new_inf > 1e-2
                error("Solver Stalled: Step jump vanished below xtol ($(solver.xtol)).")
            else
                println("  [Stagnation] Step jump vanished below xtol in the noise floor. Stopping safely.")
                norm_b_inf = norm_b_new_inf
                break
            end
        end
        
        # SMART MAX ITERS GUARD
        if i == solver.max_iters
            if norm_b_new_inf > 1e-2
                error("Solver Failed: Reached maximum iterations ($(solver.max_iters)).")
            else
                println("  [Max Iters] Reached max iterations in the noise floor. Stopping safely.")
                norm_b_inf = norm_b_new_inf
            end
        end
        
        norm_b_inf = norm_b_new_inf
    end
    
    res = SafeSolverResult(final_i, norm_b_inf)
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
