# src/instrumented_solver.jl
#=
A diagnostic Newton solver that mirrors the damped-Newton loop of the production
`SafeNewtonSolver` (src/solvers/nonlinear.jl) but emits a verbose, per-iteration
trace of the convergence behaviour through a pluggable `reporter` sink. It plugs
into Gridap as a `NonlinearSolver` (via `Gridap.Algebra.solve!`), so it can be
dropped into the same call site as the real solver to study why a particular
(Re, Da, h) case stalls, diverges, or fails the merit-descent test.

The diagnostic checkpoints are labelled with a "C<n>" scheme; this file implements
the C4 (merit-function descent) check inline. The C1 residual decomposition is run
externally by the caller's script because it needs the X/Y FE spaces, which are
not visible inside the algebraic solver layer.
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

"""
    InstrumentedNewtonSolver <: NonlinearSolver

Damped-Newton driver with instrumentation. Each field is a stopping/safeguard
control threaded explicitly from config (no implicit defaults):

- `ls`: inner linear solver used to solve the Newton system `A dx = b`.
- `max_iters`: cap on outer Newton iterations.
- `max_increases`: how many consecutive L²-residual increases (or NaNs) are
  tolerated before declaring divergence / noise-floor stagnation.
- `xtol`: step-size floor — Newton stops when the accepted step inf-norm drops below it.
- `stagnation_tol`: stagnation threshold passed through for parity with the
  production solver's contract.
- `ftol`: residual inf-norm convergence tolerance.
- `linesearch_tolerance`: required relative L²-residual reduction for a line-search
  step to be accepted (Armijo-style sufficient-decrease factor).
- `linesearch_alpha_min`: smallest step length the backtracking line search will try.
- `reporter`: callable sink for trace lines (e.g. writes to a report text file);
  `nothing` falls back to `println`.
"""
struct InstrumentedNewtonSolver <: NonlinearSolver
    ls::LinearSolver
    max_iters::Int
    max_increases::Int
    xtol::Float64
    stagnation_tol::Float64
    ftol::Float64
    linesearch_tolerance::Float64
    linesearch_alpha_min::Float64
    reporter::Any # callable that writes a trace line to the report sink
end

"""
    InstrumentedSolverResult

Summary of a finished solve: how many Newton iterations ran and the final residual
inf-norm that was reached.
"""
struct InstrumentedSolverResult
    iterations::Int
    residual_norm::Float64
end

"""
    InstrumentedSolverCache

Reusable workspace returned by `solve!` so a subsequent call can avoid
reallocating: the residual vector `b`, Jacobian `A`, Newton step `dx`, the cached
linear-solver setup `ls_cache`, and the `result` summary of the last solve.
"""
struct InstrumentedSolverCache
    b::AbstractVector
    A::AbstractMatrix
    dx::AbstractVector
    ls_cache::Any
    result::InstrumentedSolverResult
end

# Route one trace line to the configured sink: call `reporter(msg)` when a
# String-accepting reporter is present, otherwise print to stdout.
function log_to_reporter(reporter, msg)
    if reporter !== nothing && hasmethod(reporter, Tuple{String})
        reporter(msg)
    else
        println(msg)
    end
end

# Core damped-Newton loop, in-place on `x`. `op` is the Gridap nonlinear operator
# defining `residual!`/`jacobian!`; `b`, `A`, `dx`, `ls_cache` are reused workspace.
# Each Newton step solves `A dx = b`, then backtracks the step length `alpha` until
# the L²-residual drops by the required factor, while logging the C4 merit check and
# the convergence / divergence / stagnation verdict at every iteration.
function _instrumented_solve_inner!(x::AbstractVector, solver::InstrumentedNewtonSolver, op::NonlinearOperator, b, A, dx, ls_cache)
    inc_count = 0  # consecutive L²-residual increases / NaNs (divergence counter)
    log_cb = (msg) -> log_to_reporter(solver.reporter, msg)
    
    log_cb("--- InstrumentedNewtonSolver ---")
    residual!(b, op, x)
    norm_b_inf = norm(b, Inf)
    norm_b_l2  = norm(b, 2)
    log_cb("Iter 0: f(x) inf-norm = $norm_b_inf | L2-norm = $norm_b_l2")
    last_norm_l2 = norm_b_l2

    # [DIAGNOSTICS] The C1 residual decomposition needs the X/Y FE spaces (not visible in this
    # algebraic layer), so the caller's script runs it externally at chosen points; only the C4
    # merit-function descent check is performed inline here.

    if norm_b_inf <= solver.ftol
        log_cb("  [Convergence] Initial residual is below tolerance ($(solver.ftol)).")
        res = InstrumentedSolverResult(0, norm_b_inf)
        return x, InstrumentedSolverCache(b, A, dx, ls_cache, res)
    end
    
    final_i = 0
    x_old = similar(x)
    
    for i in 1:solver.max_iters
        final_i = i
        
        # Assemble the Jacobian and solve the Newton system A dx = b for the step dx.
        jacobian!(A, op, x)
        if ls_cache === nothing
            ls_cache = symbolic_setup(solver.ls, A)
        end
        ns = numerical_setup(ls_cache, A)
        solve!(dx, ns, b)

        # [DIAGNOSTICS C4: Merit Descent Test]
        # Merit function phi(x) = ½‖b‖² with b = residual(x). The directional derivative
        # along the Newton step (x ← x − α·dx) is phi'(0) = −bᵀ(A·dx); a strictly negative value
        # certifies that the (undamped) Newton step is a genuine descent direction for phi.
        Adx = A * dx
        phi_prime = dot(b, -Adx)
        merit_phi = 0.5 * dot(b, b)
        is_descent = phi_prime < 0
        log_cb("  [C4 Merit] Iter $i base phi(x) = $(merit_phi)")
        log_cb("  [C4 Merit] Iter $i phi'(0) = $(phi_prime) (Descent? $(is_descent))")

        alpha = 1.0           # line-search step length, halved each backtrack
        x_old .= x            # save the pre-step iterate so a failed step can be rolled back
        step_norm_base = norm(dx, Inf)
        
        norm_b_new_inf = norm_b_inf
        norm_b_new_l2  = norm_b_l2
        
        # Backtracking line search: trial x_old - alpha·dx, accept the first step whose
        # L²-residual meets the relative-decrease target, else halve alpha and retry until
        # the step length falls below `linesearch_alpha_min`.
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
        
        # No acceptable step found: roll x back to the pre-step iterate, recompute its
        # residual so the returned state is consistent, and abort the Newton loop.
        if !ls_success
            log_cb("  [Stagnation] Linesearch failed to find a descent direction. Aborting Newton loop.")
            x .= x_old
            residual!(b, op, x) # recompute residual at the restored iterate
            norm_b_inf = norm(b, Inf)
            break
        end
        
        step_norm = alpha * step_norm_base
        log_cb("Iter $i: f(x) inf-norm = $norm_b_new_inf | L2-norm = $norm_b_new_l2 | Step inf-norm = $step_norm | alpha = $alpha")
        
        # Primary convergence test: residual inf-norm below ftol.
        if norm_b_new_inf <= solver.ftol
            log_cb("  [Convergence] Residual inf-norm ($norm_b_new_inf) is below tolerance ($(solver.ftol)).")
            norm_b_inf = norm_b_new_inf
            break
        end

        # Divergence guard: a NaN residual or a >5% rise in the L²-residual counts as a
        # non-improving step. After `max_increases` such steps, decide between true
        # divergence (residual still large) and benign noise-floor stagnation (residual
        # already tiny) using the 1e-2 inf-norm cutoff.
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
            inc_count = 0  # reset the streak once a step improves the residual
        end
        last_norm_l2 = norm_b_new_l2
        
        # Step-stagnation test: the accepted step has shrunk below `xtol`, so no further
        # progress is expected. The 1e-2 cutoff only changes the wording of the log line
        # (genuine stall vs. acceptable noise-floor stop); the solver halts either way.
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

        # Final iteration exhausted without convergence; record the residual and (again)
        # distinguish a real non-convergence from a noise-floor finish for the log.
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

# Gridap entry point, cold start: allocate the residual/Jacobian/step workspace from
# the operator and run the Newton loop with a fresh (nothing) linear-solver cache.
function Gridap.Algebra.solve!(x::AbstractVector, solver::InstrumentedNewtonSolver, op::NonlinearOperator, cache::Nothing)
    b = allocate_residual(op, x)
    A = allocate_jacobian(op, x)
    dx = similar(b, axes(A, 2))
    fill!(dx, zero(eltype(dx)))
    return _instrumented_solve_inner!(x, solver, op, b, A, dx, nothing)
end

# Gridap entry point, warm start: reuse the workspace and symbolic linear-solver setup
# carried in `cache` so a repeated solve (e.g. homotopy step) skips reallocation.
function Gridap.Algebra.solve!(x::AbstractVector, solver::InstrumentedNewtonSolver, op::NonlinearOperator, cache::InstrumentedSolverCache)
    return _instrumented_solve_inner!(x, solver, op, cache.b, cache.A, cache.dx, cache.ls_cache)
end
