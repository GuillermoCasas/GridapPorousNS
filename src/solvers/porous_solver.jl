# src/solvers/porous_solver.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

function inner_projection_u(u, p, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
    # Define L2 projection for momentum residual
    R_u = x -> eval_strong_residual_u(form, u, p, h_cf, alpha_cf, f_cf, c_1, c_2)(x)
    return R_u
end

function inner_projection_p(u, p, form, dΩ, h_cf, alpha_cf, g_cf)
    # Define L2 projection for mass residual
    R_p = x -> eval_strong_residual_p(form, u, p, alpha_cf, g_cf)(x)
    return R_p
end

function solve_system(
    X, Y, model, dΩ, Ω, h_cf, 
    f_cf, alpha_cf, g_cf, form, 
    c_1, c_2, tau_reg_lim, freeze_cusp, 
    solver_picard, solver_newton, 
    x0, method, ftol, stagnation_tol, max_osgs_iters, osgs_tol
)
    u_h, p_h = x0
    pi_u = nothing
    pi_p = nothing

    local final_x0 = x0
    local success = false
    local iter_count = 0
    local eval_time = 0.0
    
    if method == "ASGS"
        # Pure decoupled, single iteration without iterative projections
        res_fn(x, y) = build_stabilized_weak_form_residual(x, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, pi_u, pi_p, c_1, c_2, tau_reg_lim)
        jac_picard(x, dx, y) = build_picard_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, pi_u, pi_p, c_1, c_2, tau_reg_lim; mult_mom=1.0, mult_mass=1.0)
        jac_newton(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, pi_u, pi_p, c_1, c_2, tau_reg_lim, freeze_cusp, ExactNewtonMode())

        op_picard = FEOperator(res_fn, jac_picard, X, Y)
        op_newton = FEOperator(res_fn, jac_newton, X, Y)

        # Attempt Picard
        println("      -> ASGS: Attempting Picard solver...")
        x0_backup = copy(get_free_dof_values(x0))
        try
            solve!(x0, solver_picard, op_picard)
            if norm(get_free_dof_values(x0), Inf) > 1e12 || any(isnan, get_free_dof_values(x0))
                println("      -> Picard diverged. Reverting.")
                get_free_dof_values(x0) .= x0_backup
            end
        catch
            println("      -> Picard threw error. Reverting.")
            get_free_dof_values(x0) .= x0_backup
        end

        println("      -> ASGS: Solving Exact Newton...")
        eval_time = @elapsed begin
            try
                res_solve = solve!(x0, solver_newton, op_newton)
                cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                iter_count = nls_cache.result.iterations
                final_res = nls_cache.result.residual_norm
                if final_res <= ftol || (final_res < stagnation_tol && final_res > 0.0)
                    success = true
                end
            catch
                println("      -> Newton ConvergenceError.")
                success = false
            end
        end
        return success, x0, iter_count, eval_time
    end

    if method == "OSGS"
        # OSGS iterative projection loop
        println("      -> OSGS Iterative Loop Start")
        
        # Need projection spaces (V_proj, etc.)
        # Wait, instead of setting up a global FESpace for projections, 
        # Gridap L2 projections can be done natively if we just interpolate!
        # But wait, true L2 projection requires a solve.
        # For OSGS, typically we just project the CellField into the TestFESpace.
        # Gridap's FESpaces for u and p:
        U, P = X
        V, Q = Y

        eval_time = @elapsed begin
            for osgs_iter in 1:max_osgs_iters
                println("        [OSGS Iter $osgs_iter]")
                
                res_fn_osgs(x, y) = build_stabilized_weak_form_residual(x, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, pi_u, pi_p, c_1, c_2, tau_reg_lim)
                jac_newton_osgs(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, pi_u, pi_p, c_1, c_2, tau_reg_lim, freeze_cusp, ExactNewtonMode())
                
                op_newton_osgs = FEOperator(res_fn_osgs, jac_newton_osgs, X, Y)
                
                x_prev = copy(get_free_dof_values(final_x0))
                
                try
                    res_solve = solve!(final_x0, solver_newton, op_newton_osgs)
                    cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                    nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                    iter_count += nls_cache.result.iterations
                catch e
                    println("        -> OSGS Newton failed. Aborting OSGS loop.")
                    success = false
                    break
                end
                
                x_diff = norm(get_free_dof_values(final_x0) - x_prev, Inf)
                println("        -> Diff: $x_diff")
                
                if x_diff < osgs_tol
                    println("        -> OSGS Converged!")
                    success = true
                    break
                end
                
                # Update projections for next step
                u_h, p_h = final_x0
                
                R_u = inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
                R_p = inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf)
                
                pi_u_op = AffineFEOperator((u,v) -> ∫(v ⋅ u)dΩ, v -> ∫(v ⋅ R_u)dΩ, U, V)
                pi_p_op = AffineFEOperator((p,q) -> ∫(q * p)dΩ, q -> ∫(q * R_p)dΩ, P, Q)
                
                pi_u = solve(pi_u_op)
                pi_p = solve(pi_p_op)
            end
        end
        return success, final_x0, iter_count, eval_time
    end
end
