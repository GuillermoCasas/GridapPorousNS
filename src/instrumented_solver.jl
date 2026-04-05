# src/instrumented_solver.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

struct InstrumentedNewtonSolver <: NonlinearSolver
    ls::LinearSolver
    max_iters::Int
    max_increases::Int
    xtol::Float64
    stagnation_tol::Float64
    ftol::Float64
    linesearch_tolerance::Float64
    linesearch_alpha_min::Float64
    reporter::Any # For writing to the report text file
end

struct InstrumentedSolverResult
    iterations::Int
    residual_norm::Float64
end

struct InstrumentedSolverCache
    b::AbstractVector
    A::AbstractMatrix
    dx::AbstractVector
    ls_cache::Any
    result::InstrumentedSolverResult
end

function log_to_reporter(reporter, msg)
    if reporter !== nothing && hasmethod(reporter, Tuple{String})
        reporter(msg)
    else
        println(msg)
    end
end

function _instrumented_solve_inner!(x::AbstractVector, solver::InstrumentedNewtonSolver, op::NonlinearOperator, b, A, dx, ls_cache)
    inc_count = 0
    log_cb = (msg) -> log_to_reporter(solver.reporter, msg)
    
    log_cb("--- InstrumentedNewtonSolver ---")
    residual!(b, op, x)
    norm_b_inf = norm(b, Inf)
    norm_b_l2  = norm(b, 2)
    log_cb("Iter 0: f(x) inf-norm = $norm_b_inf | L2-norm = $norm_b_l2")
    last_norm_l2 = norm_b_l2
    
    # [DIAGNOSTICS C1: Residual Decomposition at Init]
    # To do this correctly, we need the `X` and `Y` spaces inside the operator, which we don't have direct access to here.
    # Actually, the user script will call the residual decomposition itself at specific points. We just provide hooks if possible.
    # However, for C4 (Merit-function descent test), we can do it here!
    
    if norm_b_inf <= solver.ftol
        log_cb("  [Convergence] Initial residual is below tolerance ($(solver.ftol)).")
        res = InstrumentedSolverResult(0, norm_b_inf)
        return x, InstrumentedSolverCache(b, A, dx, ls_cache, res)
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
        
        # [DIAGNOSTICS C4: Merit Descent Test]
        Adx = A * dx
        phi_prime = dot(b, -Adx)
        merit_phi = 0.5 * dot(b, b)
        is_descent = phi_prime < 0
        log_cb("  [C4 Merit] Iter $i base phi(x) = $(merit_phi)")
        log_cb("  [C4 Merit] Iter $i phi'(0) = $(phi_prime) (Descent? $(is_descent))")
        
        alpha = 1.0
        x_old .= x
        step_norm_base = norm(dx, Inf)
        
        norm_b_new_inf = norm_b_inf
        norm_b_new_l2  = norm_b_l2
        
        ls_success = false
        merit_phi_new = 0.0
        for ls_iter in 1:15
            x .= x_old .- alpha .* dx
            residual!(b, op, x)
            norm_b_new_inf = norm(b, Inf)
            norm_b_new_l2  = norm(b, 2)
            merit_phi_new = 0.5 * dot(b, b)
            
            if norm_b_new_l2 <= last_norm_l2 * solver.linesearch_tolerance 
                ls_success = true
                break
            end
            if alpha <= solver.linesearch_alpha_min
                break
            end
            alpha *= 0.5 
        end
        
        real_descent = merit_phi_new < merit_phi
        log_cb("  [C4 Merit] Iter $i accepted alpha = $(alpha)")
        log_cb("  [C4 Merit] Iter $i accepted phi(x_new) = $(merit_phi_new) (Actual decrease? $(real_descent))")
        
        if !ls_success
            log_cb("  [Stagnation] Linesearch failed to find a descent direction. Aborting Newton loop.")
            x .= x_old
            residual!(b, op, x) # Restore residual
            norm_b_inf = norm(b, Inf)
            break
        end
        
        step_norm = alpha * step_norm_base
        log_cb("Iter $i: f(x) inf-norm = $norm_b_new_inf | L2-norm = $norm_b_new_l2 | Step inf-norm = $step_norm | alpha = $alpha")
        
        if norm_b_new_inf <= solver.ftol
            log_cb("  [Convergence] Residual inf-norm ($norm_b_new_inf) is below tolerance ($(solver.ftol)).")
            norm_b_inf = norm_b_new_inf
            break
        end
        
        if isnan(norm_b_new_l2) || norm_b_new_l2 > last_norm_l2 * 1.05
            inc_count += 1
            if inc_count >= solver.max_increases
                if norm_b_new_inf > 1e-2
                    error("Solver Diverged: L2 Residual exploded or hit NaN for $(inc_count) iterations.")
                else
                    log_cb("  [Stagnation] Solver hit the numerical noise floor. Stopping safely.")
                    norm_b_inf = norm_b_new_inf
                    break
                end
            end
        else
            inc_count = 0
        end
        last_norm_l2 = norm_b_new_l2
        
        if step_norm <= solver.xtol
            if norm_b_new_inf > 1e-2
                log_cb("  [Stagnation] Solver Stalled: Step jump vanished below xtol ($(solver.xtol)). Stopping.")
                norm_b_inf = norm_b_new_inf
                break
            else
                log_cb("  [Stagnation] Step jump vanished below xtol in the noise floor. Stopping safely.")
                norm_b_inf = norm_b_new_inf
                break
            end
        end
        
        if i == solver.max_iters
            if norm_b_new_inf > 1e-2
                log_cb("  [Max Iters] Reached maximum iterations ($(solver.max_iters)) without convergence.")
                norm_b_inf = norm_b_new_inf
            else
                log_cb("  [Max Iters] Reached max iterations in the noise floor. Stopping safely.")
                norm_b_inf = norm_b_new_inf
            end
        end
        
        norm_b_inf = norm_b_new_inf
    end
    
    res = InstrumentedSolverResult(final_i, norm_b_inf)
    return x, InstrumentedSolverCache(b, A, dx, ls_cache, res)
end

function Gridap.Algebra.solve!(x::AbstractVector, solver::InstrumentedNewtonSolver, op::NonlinearOperator, cache::Nothing)
    b = allocate_residual(op, x)
    A = allocate_jacobian(op, x)
    dx = similar(b, axes(A, 2))
    fill!(dx, zero(eltype(dx)))
    return _instrumented_solve_inner!(x, solver, op, b, A, dx, nothing)
end

function Gridap.Algebra.solve!(x::AbstractVector, solver::InstrumentedNewtonSolver, op::NonlinearOperator, cache::InstrumentedSolverCache)
    return _instrumented_solve_inner!(x, solver, op, cache.b, cache.A, cache.dx, cache.ls_cache)
end
