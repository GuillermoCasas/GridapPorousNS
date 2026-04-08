# src/solvers/porous_solver.jl
"""
    porous_solver.jl

# Role 
This module acts as the overall Variational Multiscale (VMS) orchestrator for the Porous Navier-Stokes system. It bridges the pure algebraic continuous operators, variational stabilized forms, and Jacobians (defined in `viscous_operators.jl`) with the iterative numerical execution and topological logic needed to achieve discrete convergence (defined in `nonlinear.jl`).

# Methodological Background
Following the formulation in the companion article, the Galerkin finite element discretization of the generalized Navier-Stokes equations suffers from LBB condition violations for equal-order spaces and instabilities in convection- or reaction-dominated flows. We use VMS to approximate the unresolved sub-grid scales (SGS) ``\\widetilde{U}`` in terms of the resolved scales' residual ``\\mathcal{R}(U_h)`` and a local element matrix of stabilization parameters ``\\boldsymbol{\\tau}_K``. 

Depending on the chosen space ``\\widetilde{\\mathcal{X}}`` for the sub-grid scales, two different stabilized methods are implemented here:
1. **ASGS (Algebraic Sub-Grid Scale)**: The SGS space is taken as the space of finite element residuals. The projection operator onto the SGS space is the identity ``\\widetilde{\\Pi} = \\mathcal{I}``, rendering the FE projection ``\\boldsymbol{\\pi}_h = \\boldsymbol{0}``.
2. **OSGS (Orthogonal Sub-Grid Scale)**: The SGS space is strictly orthogonal to the finite element space (``\\mathcal{X}_{h0}^{\\perp}``). This requires actively computing the ``L^2``-projection of the residual ``\\boldsymbol{\\pi}_h`` iteratively and subtracting it from the strong residual in the sub-scale equation.

# Relations to other files
- `src/formulations/viscous_operators.jl`: Contains the translation of the abstract operators (e.g. ``\\mathcal{L}_{\\nu}, \\mathcal{L}_c, \\mathcal{L}_b``) into Gridap closures. `porous_solver.jl` delegates the assembly of `eval_strong_residual`, `build_stabilized_weak_form_jacobian`, etc., to this file.
- `src/solvers/nonlinear.jl`: Defines the exact solver execution schemes (`solver_picard`, `solver_newton`) required to converge the coupled nonlinear ASGS/OSGS algebraic problems.
"""
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

"""
    check_porous_solver_parameters(method, ftol, stagnation_tol, max_osgs_iters, osgs_tol)

Enforces strict parameter boundaries for the top-level VMS solver configurations, 
preventing unbounded numerical cascades resulting from malformed JSON properties.
"""
function check_porous_solver_parameters(method, ftol, stagnation_tol, max_osgs_iters, osgs_tol)
    if method != "ASGS" && method != "OSGS"
        throw(ArgumentError("Stabilization method must be strictly 'ASGS' or 'OSGS'. Passed: $method"))
    end
    if ftol <= 0.0 || stagnation_tol <= 0.0 || osgs_tol <= 0.0
        throw(ArgumentError("Solver outer tolerances (ftol, stagnation_tol, osgs_tol) must be strictly positive floats."))
    end
    if max_osgs_iters < 1
        throw(ArgumentError("OSGS maximum iterations must strictly be integers >= 1."))
    end
end

"""
    solve_system(...)

Orchestrates the nonlinear solver iteration loops over a defined Variational Multiscale space.
"""
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
    
    check_porous_solver_parameters(method, ftol, stagnation_tol, max_osgs_iters, osgs_tol)
    
    if method == "ASGS"
        # ==============================================================================
        # ALGEBRAIC SUBGRID SCALE (ASGS)
        # Bypasses iterative projection evaluations entirely. According to the VMS theory 
        # outlined in the companion article, the ASGS method maps the SGS space to the space 
        # of finite element residuals, meaning the SGS projection operator is the identity 
        # ($\\widetilde{\\Pi} = \\mathcal{I}$) and therefore the FE residual projection is exactly zero
        # ($\\boldsymbol{\\pi}_h = \\boldsymbol{0}$). The unresolvable sub-grid physics are mathematically 
        # modeled as directly proportional to the local strong residual (`pi_u = nothing`, 
        # `pi_p = nothing`), mapping cleanly into a single Newton-homotopy execution without
        # iterative, staggered updates.
        # ==============================================================================
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
        # ==============================================================================
        # ORTHOGONAL SUBGRID SCALE (OSGS)
        # Iteratively tracks and orthogonalizes the sub-grid unresolvable scales against 
        # the finite element basis explicitly by isolating $(I - \\Pi_{h})(\\mathcal{R})$.
        # In this formulation, the space for the SGS is taken as $\\mathcal{X}_{h0}^{\\perp}$.
        # Solving the globally-coupled, monolithic nonlinear system with the projection operations
        # is computationally prohibitive. Instead, we solve the OSGS system in a staggered, iterative 
        # scheme: the global FE residual projection ($\\boldsymbol{\\pi}_h$) from the previous 
        # iteration is held fixed during the underlying nonlinear Newton homotopy to compute 
        # the updated primary fields ($U_h$), which are then used to re-project the continuous 
        # residual until discrete equilibrium ($x_{diff} < osgs_{tol}$) is achieved.
        # ==============================================================================
        println("      -> OSGS Iterative Loop Start")
        
        # Gridap L2 Projections inherit boundary conditions from the FESpace they map onto.
        # Note: Projecting directly onto TrialFESpace `U` will inherently zero out the residual 
        # errors exactly on Dirichlet boundary layers (since `U` enforces fixed walls/inflow). 
        # While mathematically imperfect (the continuous residual is not zero on boundary nodes),
        # this topological choice stabilizes the sub-grid field interior natively without requiring 
        # a fully disconnected Discontinuous Galerkin L2 field construction loop.
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
                
                # Extract updated local solution for projection mapping
                u_h, p_h = final_x0
                
                # Continuous representations of the local analytical residual
                R_u = inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
                R_p = inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf)
                
                # Explicit L2 continuous projection operator mappings: `∫(v ⋅ pi_u) = ∫(v ⋅ R_u)`
                # As noted in the article (around Eq. 3.20b), this relies on a simplified standard L2 
                # inner product approximation (∫(u ⋅ v)dΩ) of the theoretical tau-weighted inner product 
                # projection (∫(v ⋅ τ ⋅ u)dΩ). This theoretically mild concession favors highly efficient, 
                # symmetric (often diagonal or easily invertible) mass-matrix based L2 projections while 
                # preserving analogous subgrid stabilization and resolving properties.
                pi_u_op = AffineFEOperator((u,v) -> ∫(v ⋅ u)dΩ, v -> ∫(v ⋅ R_u)dΩ, U, V)
                pi_p_op = AffineFEOperator((p,q) -> ∫(q * p)dΩ, q -> ∫(q * R_p)dΩ, P, Q)
                
                pi_u = solve(pi_u_op)
                pi_p = solve(pi_p_op)
            end
        end
        return success, final_x0, iter_count, eval_time
    end
end
