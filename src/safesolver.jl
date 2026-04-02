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

function Gridap.Algebra.solve!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, cache::Nothing)
    b = allocate_residual(op, x)
    A = allocate_jacobian(op, x)
    dx = similar(b, axes(A, 2))
    fill!(dx, zero(eltype(dx)))
    
    ls_cache = nothing
    inc_count = 0
    
    residual!(b, op, x)
    norm_b = norm(b, Inf)
    println("Iter 0: f(x) inf-norm = ", norm_b)
    last_norm = norm_b
    
    if norm_b <= solver.stagnation_tol
        println("  [Convergence] Initial residual is below tolerance ($(solver.stagnation_tol)).")
        res = SafeSolverResult(0, norm_b)
        return x, SafeSolverCache(b, A, dx, ls_cache, res)
    end
    
    final_i = 0
    
    for i in 1:solver.max_iters
        final_i = i
        
        jacobian!(A, op, x)
        if ls_cache === nothing
            ls_cache = symbolic_setup(solver.ls, A)
        end
        ns = numerical_setup(ls_cache, A)
        solve!(dx, ns, b)
        
        step_norm = norm(dx, Inf)
        x .= x .- dx
        
        residual!(b, op, x)
        norm_b = norm(b, Inf)
        
        println("Iter $i: f(x) inf-norm = $norm_b | Step inf-norm = $step_norm")
        
        if norm_b <= solver.stagnation_tol
            println("  [Convergence] Residual inf-norm ($norm_b) is below tolerance ($(solver.stagnation_tol)).")
            break
        end
        
        if norm_b > last_norm
            inc_count += 1
            if inc_count >= solver.max_increases
                println("  [Divergence Guard] Residual monotonically increased for $(inc_count) iterations! Aborting.")
                break
            end
        else
            inc_count = 0
        end
        last_norm = norm_b
        
        if step_norm <= solver.xtol
            println("  [Stagnation Guard] Step jump vanished below xtol ($(solver.xtol)). Aborting.")
            break
        end
        
        if i == solver.max_iters
            println("  [Max Iterations] Reached maximum iterations ($(solver.max_iters)).")
        end
    end
    
    res = SafeSolverResult(final_i, norm_b)
    return x, SafeSolverCache(b, A, dx, ls_cache, res)
end

function Gridap.Algebra.solve!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, cache::SafeSolverCache)
    b = cache.b
    A = cache.A
    dx = cache.dx
    ls_cache = cache.ls_cache
    
    inc_count = 0
    
    residual!(b, op, x)
    norm_b = norm(b, Inf)
    println("Iter 0: f(x) inf-norm = ", norm_b)
    last_norm = norm_b
    
    if norm_b <= solver.stagnation_tol
        println("  [Convergence] Initial residual is below tolerance ($(solver.stagnation_tol)).")
        res = SafeSolverResult(0, norm_b)
        return x, SafeSolverCache(b, A, dx, ls_cache, res)
    end
    
    final_i = 0
    
    for i in 1:solver.max_iters
        final_i = i
        
        jacobian!(A, op, x)
        if ls_cache === nothing
            ls_cache = symbolic_setup(solver.ls, A)
        end
        ns = numerical_setup(ls_cache, A)
        solve!(dx, ns, b)
        
        step_norm = norm(dx, Inf)
        x .= x .- dx
        
        residual!(b, op, x)
        norm_b = norm(b, Inf)
        
        println("Iter $i: f(x) inf-norm = $norm_b | Step inf-norm = $step_norm")
        
        if norm_b <= solver.stagnation_tol
            println("  [Convergence] Residual inf-norm ($norm_b) is below tolerance ($(solver.stagnation_tol)).")
            break
        end
        
        if norm_b > last_norm
            inc_count += 1
            if inc_count >= solver.max_increases
                println("  [Divergence Guard] Residual monotonically increased for $(inc_count) iterations! Aborting.")
                break
            end
        else
            inc_count = 0
        end
        last_norm = norm_b
        
        if step_norm <= solver.xtol
            println("  [Stagnation Guard] Step jump vanished below xtol ($(solver.xtol)). Aborting.")
            break
        end
        
        if i == solver.max_iters
            println("  [Max Iterations] Reached maximum iterations ($(solver.max_iters)).")
        end
    end
    
    res = SafeSolverResult(final_i, norm_b)
    return x, SafeSolverCache(b, A, dx, ls_cache, res)
end
