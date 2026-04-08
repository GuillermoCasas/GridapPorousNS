# src/solvers/nonlinear.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

"""
    SafeNewtonSolver <: NonlinearSolver

An inexact Newton solver structurally fortified with Armijo backtracking linesearch 
and strict topological bounds checks to guarantee stability upon floating-point non-linear systems.

# JSON Configuration Mapping
- `max_iters`: Absolute iteration limit per full Newton homotopy solve.
- `max_increases`: Bound on permissible consecutive iterations where the merit function diverges.
- `xtol`: Acceptable jump size limit, preventing the solver from stalling on singular topologies.
- `stagnation_tol`: Evaluates local relative noise bounds when tracking divergence states.
- `ftol`: Absolute threshold on `Inf`-norm residual. Signals continuous mathematical convergence.
- `linesearch_alpha_min`: The minimum geometric step size (alpha) reduction permitted during linesearch backtracking.
- `c1` (Armijo parameter): Specifies absolute threshold for sufficient residual descent. Higher values demand stricter gradient conformity.
- `divergence_merit_factor`: Relative expansion threshold. e.g., 1.05 permits the merit function to grow temporarily up to 5% before triggering divergence bounds. Allows tunneling noisy manifolds.
- `stagnation_noise_floor`: Lowest numerical bound (`Inf`-norm) realistically achievable before machine epsilon limits induce spurious divergence logic.
"""
struct SafeNewtonSolver <: NonlinearSolver
    ls::LinearSolver
    max_iters::Int
    max_increases::Int
    xtol::Float64
    stagnation_tol::Float64
    ftol::Float64
    linesearch_alpha_min::Float64
    c1::Float64 # Armijo parameter
    divergence_merit_factor::Float64
    stagnation_noise_floor::Float64
end

"""
    check_solver_parameters(solver::SafeNewtonSolver)

Validates the SafeNewtonSolver parameter space mathematically, preventing unphysical bounds or 
pathological backtracking thresholds inherited from the JSON interface.
"""
function check_solver_parameters(s::SafeNewtonSolver)
    if !(0.0 < s.c1 < 1.0)
        throw(ArgumentError("Armijo backtracking parameter 'c1' must be strictly bounded in (0, 1). Passed: \$(s.c1)"))
    end
    if s.divergence_merit_factor < 1.0
        throw(ArgumentError("Divergence merit factor must realistically be >= 1.0 to permit temporary topology steps. Passed: \$(s.divergence_merit_factor)"))
    end
    if s.xtol <= 0.0 || s.ftol <= 0.0 || s.stagnation_noise_floor <= 0.0
        throw(ArgumentError("Solver criteria (xtol, ftol, stagnation_noise_floor) must be strictly positive floating bounds."))
    end
    if s.linesearch_alpha_min <= 0.0 || s.linesearch_alpha_min > 1.0
        throw(ArgumentError("Linesearch bounds 'linesearch_alpha_min' must be precisely within (0, 1]. Passed: \$(s.linesearch_alpha_min)"))
    end
    if s.max_iters < 1 || s.max_increases < 1
        throw(ArgumentError("Maximum solver loops must strictly be integers >= 1."))
    end
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

function _safe_solve_inner!(x::AbstractVector, solver::SafeNewtonSolver, op::NonlinearOperator, b, A, dx, ls_cache)
    # Strictly validate solver boundaries dynamically parsed from runtime JSON configuration
    check_solver_parameters(solver)
    
    inc_count = 0
    
    println("    => Assembling initial non-linear residual...")
    residual!(b, op, x)
    
    # Track L_∞ norm of continuous spatial residual formulation for absolute convergence
    norm_b_inf = norm(b, Inf)
    
    # Establish base scalar optimization constraint: `Φ(x) = (1/2) * ||F(x)||^2_2`
    phi_x = 0.5 * (norm(b, 2)^2)
    last_phi = phi_x
    
    println("Iter 0: f(x) inf-norm = $norm_b_inf | Merit Φ(x) = $phi_x")
    
    if norm_b_inf <= solver.ftol
        println("  [Convergence] Initial residual is below tolerance ($(solver.ftol)).")
        res = SafeSolverResult(0, norm_b_inf)
        return x, SafeSolverCache(b, A, dx, ls_cache, res)
    end
    
    final_i = 0
    x_old = similar(x)
    # Tracking monotonic behavior recursively independent of step limit bounds
    best_x = copy(x)
    best_b_inf = norm_b_inf
    best_phi = phi_x
    
    inc_count = 0
    
    for i in 1:solver.max_iters
        final_i = i
        
        # Gridap directly links analytical mathematical Jacobian functions mapping onto explicit matrix blocks
        println("    => Assembling Jacobian matrix (Iteration $i)...")
        jacobian!(A, op, x)
        
        println("    => Setting up numerical factorization...")
        if ls_cache === nothing
            ls_cache = symbolic_setup(solver.ls, A)
        end
        ns = numerical_setup(ls_cache, A)
        
        println("    => Solving linear system...")
        solve!(dx, ns, b)
        
        # If linear solve produced NaNs, matrix is singular or completely broken
        if any(isnan, dx)
            println("  [Linear Solve Failed] Newton direction contains NaNs. Matrix likely singular. Aborting safely.")
            x .= best_x
            norm_b_inf = best_b_inf
            break
        end
        
        alpha = 1.0
        x_old .= x
        step_norm_base = norm(dx, Inf)
        
        # =========================================================================================
        # EXACT DIRECTIONAL DERIVATIVE
        # Rather than enforcing native ∇Φ·p = -||b||², we explicitly evaluate Adx = A(x) * dx.
        # Functionally, this identically aligns sequentially into the actual descent defined by the 
        # computed direction `dx`, perfectly stabilizing inexact sub-solvers (e.g., iterative GMRES).
        # =========================================================================================
        Adx = A * dx
        dir_deriv = - dot(b, Adx)
        
        ls_success = false
        norm_b_new_inf = norm_b_inf
        phi_x_new = phi_x
        
        for ls_iter in 1:15
            x .= x_old .- alpha .* dx
            residual!(b, op, x)
            norm_b_new_inf = norm(b, Inf)
            phi_x_new = 0.5 * (norm(b, 2)^2)
            
            if isnan(phi_x_new)
                # If we stepped into a NaN domain, strictly reject this alpha
                alpha *= 0.5
                continue
            end
            
            # Armijo condition ensuring strictly bounded sufficient decrease mapping along step path:
            # `Φ(x_{k} - α * dx) <= Φ(x_{k}) + c1 * α * ∇Φ(x_{k}) * p_{k}`.
            if phi_x_new <= phi_x + solver.c1 * alpha * dir_deriv
                ls_success = true
                break
            end
            if alpha <= solver.linesearch_alpha_min
                break
            end
            alpha *= 0.5 
        end
        
        if isnan(phi_x_new) || !ls_success
            println("  [Stagnation] Linesearch failed to find a valid descent direction or hit pure NaNs. Aborting Newton loop safely.")
            x .= best_x
            norm_b_inf = best_b_inf
            break
        end
        
        step_norm = alpha * step_norm_base
        println("Iter $i: f(x) inf-norm = $norm_b_new_inf | Merit Φ = $phi_x_new | Step inf-norm = $step_norm | alpha = $alpha | dir_deriv = $dir_deriv")
        
        # Track best state monotonically across the Newton homotopy!
        if norm_b_new_inf < best_b_inf
            best_x .= x
            best_b_inf = norm_b_new_inf
            best_phi = phi_x_new
        end
        
        if norm_b_new_inf <= solver.ftol
            println("  [Convergence] Residual inf-norm ($norm_b_new_inf) is below tolerance ($(solver.ftol)).")
            norm_b_inf = norm_b_new_inf
            break
        end
        
        # Divergence guard based on merit.
        if phi_x_new > last_phi * solver.divergence_merit_factor
            inc_count += 1
            if inc_count >= solver.max_increases
                if best_b_inf > solver.stagnation_noise_floor
                    error("Solver Diverged: Merit function exploded for $(inc_count) iterations.")
                else
                    println("  [Stagnation] Solver hit the numerical noise floor. Stopping safely.")
                    x .= best_x
                    norm_b_inf = best_b_inf
                    break
                end
            end
        else
            inc_count = 0
        end
        
        last_phi = phi_x_new
        phi_x = phi_x_new
        
        if step_norm <= solver.xtol
            if best_b_inf > solver.stagnation_noise_floor
                error("Solver Stalled: Step jump vanished below xtol ($(solver.xtol)).")
            else
                println("  [Stagnation] Step jump vanished below xtol in the noise floor. Stopping safely.")
                x .= best_x
                norm_b_inf = best_b_inf
                break
            end
        end
        
        if i == solver.max_iters
            if best_b_inf > solver.stagnation_noise_floor
                error("Solver Failed: Reached maximum iterations ($(solver.max_iters)).")
            else
                println("  [Max Iters] Reached max iterations in the noise floor. Stopping safely.")
                x .= best_x
                norm_b_inf = best_b_inf
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
