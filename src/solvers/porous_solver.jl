# src/solvers/porous_solver.jl
#=
    porous_solver.jl

# Role 
This module acts as the overall Variational Multiscale (VMS) orchestrator for the Porous Navier-Stokes system. It bridges the pure algebraic continuous operators, variational stabilized forms, and Jacobians (defined in `viscous_operators.jl`) with the iterative numerical execution and topological logic needed to achieve discrete convergence (defined in `nonlinear.jl`).

# Methodological Background
Following the formulation in the companion article, the Galerkin finite element discretization of the generalized Navier-Stokes equations suffers from LBB condition violations for equal-order spaces and instabilities in convection- or reaction-dominated flows. We use VMS to approximate the unresolved sub-grid scales (SGS) ``\widetilde{U}`` in terms of the resolved scales' residual ``\mathcal{R}(U_h)`` and a local element matrix of stabilization parameters ``\boldsymbol{\tau}_K``. 

Depending on the chosen space ``\widetilde{\mathcal{X}}`` for the sub-grid scales, two different stabilized methods are implemented here:
1. **ASGS (Algebraic Sub-Grid Scale)**: The SGS space is taken as the space of finite element residuals. The projection operator onto the SGS space is the identity ``\widetilde{\Pi} = \mathcal{I}``, rendering the FE projection ``\boldsymbol{\pi}_h = \boldsymbol{0}``.
2. **OSGS (Orthogonal Sub-Grid Scale)**: The SGS space is strictly orthogonal to the finite element space (``\mathcal{X}_{h0}^{\perp}``). This requires actively computing the ``L^2``-projection of the residual ``\boldsymbol{\pi}_h`` iteratively and subtracting it from the strong residual in the sub-scale equation.

# Relations to other files
- `src/formulations/viscous_operators.jl`: Contains the translation of the abstract operators (e.g. ``\mathcal{L}_{\nu}, \mathcal{L}_c, \mathcal{L}_b``) into Gridap closures. `porous_solver.jl` delegates the assembly of `eval_strong_residual`, `build_stabilized_weak_form_jacobian`, etc., to this file.
- `src/solvers/nonlinear.jl`: Defines the exact solver execution schemes (`solver_picard`, `solver_newton`) required to converge the coupled nonlinear ASGS/OSGS algebraic problems.
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

function inner_projection_u(u, p, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
    # Define L2 projection for momentum residual natively as a CellField
    return eval_strong_residual_u(form, u, p, h_cf, alpha_cf, f_cf, c_1, c_2)
end

function inner_projection_p(u, p, form, dΩ, h_cf, alpha_cf, g_cf)
    # Define L2 projection for mass residual natively as a CellField
    return eval_strong_residual_p(form, u, p, alpha_cf, g_cf)
end

"""
    check_porous_solver_parameters(method, ftol, stagnation_tol, max_osgs_iters, osgs_tol)

Enforces strict parameter boundaries for the top-level VMS solver configurations, 
preventing unbounded numerical cascades resulting from malformed JSON properties.
"""
function check_porous_solver_parameters(stab_cfg::StabilizationConfig, sol_cfg::SolverConfig)
    if stab_cfg.method != "ASGS" && stab_cfg.method != "OSGS"
        throw(ArgumentError("Stabilization method must be strictly 'ASGS' or 'OSGS'. Passed: $(stab_cfg.method)"))
    end
    if sol_cfg.ftol <= 0.0 || sol_cfg.stagnation_noise_floor <= 0.0 || stab_cfg.osgs_tolerance <= 0.0
        throw(ArgumentError("Solver outer tolerances (ftol, stagnation_noise_floor, osgs_tol) must be strictly positive floats."))
    end
    if stab_cfg.osgs_iterations < 1
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
    solver_picard, solver_newton, 
    x0, c_1, c_2,
    phys_cfg::PhysicalProperties, stab_cfg::StabilizationConfig, sol_cfg::SolverConfig;
    V_free=nothing, Q_free=nothing
)
    u_h, p_h = x0
    pi_u = nothing
    pi_p = nothing

    local final_x0 = x0
    local success = false
    local iter_count = 0
    local eval_time = 0.0
    
    check_porous_solver_parameters(stab_cfg, sol_cfg)
    
    # Primitive unpacking (Critical for Gridap performance)
    method = stab_cfg.method
    max_osgs_iters = stab_cfg.osgs_iterations
    osgs_tol = stab_cfg.osgs_tolerance
    ftol = sol_cfg.ftol
    stagnation_tol = sol_cfg.stagnation_noise_floor
    tau_reg_lim = phys_cfg.tau_regularization_limit
    freeze_cusp = sol_cfg.freeze_jacobian_cusp
    
    # ==============================================================================
    # ALGEBRAIC INITIALIZATION BOUNDS
    # Even for OSGS, we universally boot up the global PDE using the zero-projection 
    # (ASGS) formulation to securely map the non-linear fields into the exact Newton 
    # quadratic basin before fragmenting the state via staggered projections.
    # ==============================================================================
    res_fn_init(x, y) = build_stabilized_weak_form_residual(x, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, nothing, nothing, c_1, c_2, tau_reg_lim)
    jac_picard_init(x, dx, y) = build_picard_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, nothing, nothing, c_1, c_2, tau_reg_lim; mult_mom=1.0, mult_mass=1.0)
    jac_newton_init(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, nothing, nothing, c_1, c_2, tau_reg_lim, freeze_cusp, ExactNewtonMode())

    op_picard_init = FEOperator(res_fn_init, jac_picard_init, X, Y)
    op_newton_init = FEOperator(res_fn_init, jac_newton_init, X, Y)

    x0_backup = copy(get_free_dof_values(x0))
    
    eval_time = @elapsed begin
        println("      -> ASGS Initializer: Attempting Exact Newton solver initially...")
        local newton_success = false
        try
            res_newton = solve!(x0, solver_newton, op_newton_init)
            cache_newton = res_newton isa Tuple ? res_newton[2] : res_newton
            nls_cache = cache_newton isa Tuple ? cache_newton[2] : cache_newton
            final_res = nls_cache.result.residual_norm
            local_iters = nls_cache.result.iterations
            if final_res <= ftol || (final_res < stagnation_tol && final_res > 0.0)
                newton_success = true
                iter_count += local_iters
            else
                iter_count += local_iters
            end
        catch e
            println("      -> ASGS Initializer: Newton ConvergenceError. Exception: ", e)
            newton_success = false
        end
        
        if newton_success
            println("      -> ASGS Initializer: Exact Newton converged initially! Skipping Picard homotopy.")
            success = true
        else
            println("      -> ASGS Initializer: Exact Newton aborted. Orchestrating Picard Homotopy fallback...")
            get_free_dof_values(x0) .= x0_backup
            
            try
                res_picard = solve!(x0, solver_picard, op_picard_init)
                cache_picard = res_picard isa Tuple ? res_picard[2] : res_picard
                nls_cache_picard = cache_picard isa Tuple ? cache_picard[2] : cache_picard
                
                final_res_picard = nls_cache_picard.result.residual_norm
                iter_count += nls_cache_picard.result.iterations
                if final_res_picard <= ftol
                    println("      -> ASGS Initializer: Picard fully converged! Escaping to evaluation.")
                    success = true
                end
            catch e
                if occursin("Reached maximum iterations", string(e)) || occursin("Reached max iterations", string(e))
                    println("      -> ASGS Initializer: Picard concluded initial smoothing loops.")
                else
                    println("      -> ASGS Initializer: Picard threw unrecoverable error. Exception: ", e)
                    get_free_dof_values(x0) .= x0_backup
                end
            end
            
            if norm(get_free_dof_values(x0), Inf) > 1e12 || any(isnan, get_free_dof_values(x0))
                println("      -> ASGS Initializer: Picard catastrophically diverged. Evaluation impossible.")
                success = false
            elseif !success
                println("      -> ASGS Initializer: Picard smoothing finalized. Re-engaging Exact Newton...")
                try
                    res_solve = solve!(x0, solver_newton, op_newton_init)
                    cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                    nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                    iter_count += nls_cache.result.iterations
                    final_res = nls_cache.result.residual_norm
                    if final_res <= ftol || (final_res < stagnation_tol && final_res > 0.0)
                        success = true
                    end
                catch e
                     println("      -> ASGS Initializer: Newton ConvergenceError on Homotopy Pass. Exception: ", e)
                     success = false
                end
            end
        end
    end
    
    if !success
        return false, x0, iter_count, eval_time
    end

    final_x0 = x0

    if method == "ASGS"
        println("\n      [+] ASGS Formulation exclusively resolved. Exiting solver module.")
        return success, final_x0, iter_count, eval_time
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
        println("\n      [+] Commencing Orthogonal Subgrid Scale (OSGS) fixed-point recursive relaxation loop (allocating for max iterations: $max_osgs_iters)...")
        
        # Gridap L2 Projections inherit boundary conditions from the FESpace they map onto.
        # Stabilizing the sub-grid field natively without requiring full DG looping. We enforce 
        # completely unconstrained spaces explicitly to prevent physical wall velocities from crushing
        # orthogonal projection boundary convergence rates.
        U, P = X
        V, Q = Y
        
        V_proj = V_free !== nothing ? V_free : V
        Q_proj = Q_free !== nothing ? Q_free : Q
        U_proj = TrialFESpace(V_proj)
        P_proj = TrialFESpace(Q_proj)
        
        # --- Precompute and Cache L2 Mass Matrices ---
        # The left-hand side mass-matrices for the L2 projections remain functionally invariant 
        # across the non-linear OSGS scheme, allowing us to factorize them exactly once.
        println("\n      [+] Initiating pre-assembly & structural factorization for static OSGS L2 Mass Matrices...")
        M_u = assemble_matrix((u,v) -> ∫(v ⋅ u)dΩ, U_proj, V_proj)
        M_p = assemble_matrix((p,q) -> ∫(q * p)dΩ, P_proj, Q_proj)
        
        ls_u = LUSolver()
        ls_p = LUSolver()
        num_u_fac = numerical_setup(symbolic_setup(ls_u, M_u), M_u)
        num_p_fac = numerical_setup(symbolic_setup(ls_p, M_p), M_p)
        
        # Override the Exact Newton solver dynamically for OSGS to perform only 1 monolithic iteration 
        # per sub-scale projection, mapping identically to classic literature practice.
        base_nls = solver_newton.nls
        local_osgs_nls = SafeNewtonSolver(base_nls.ls, 1, base_nls.max_increases, base_nls.xtol, base_nls.ftol, base_nls.linesearch_alpha_min, base_nls.c1, base_nls.divergence_merit_factor, base_nls.stagnation_noise_floor)
        local_fesolver = FESolver(local_osgs_nls)
        
        # Expand maximum bounds to absorb the monolithic 1-step iteration workflow
        eff_osgs_iters = max(max_osgs_iters, sol_cfg.newton_iterations + 5)

        eval_time += @elapsed begin
            local accelerator = nothing
            if sol_cfg.accelerator.type == "Anderson"
                println("      [+] Initializing Anderson Acceleration (m = $(sol_cfg.accelerator.m), damping = $(sol_cfg.accelerator.relaxation_factor))")
                accelerator = AndersonAccelerator(sol_cfg.accelerator.m, sol_cfg.accelerator.relaxation_factor)
            end

            for osgs_iter in 1:eff_osgs_iters
                println("        [OSGS Iter $osgs_iter]")
                
                res_fn_osgs(x, y) = build_stabilized_weak_form_residual(x, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, pi_u, pi_p, c_1, c_2, tau_reg_lim)
                jac_newton_osgs(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, pi_u, pi_p, c_1, c_2, tau_reg_lim, freeze_cusp, ExactNewtonMode())
                
                op_newton_osgs = FEOperator(res_fn_osgs, jac_newton_osgs, X, Y)
                
                x_prev = copy(get_free_dof_values(final_x0))
                
                try
                    res_solve = solve!(final_x0, local_fesolver, op_newton_osgs)
                    cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                    nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                    iter_count += nls_cache.result.iterations
                catch e
                    println("        -> OSGS Newton failed. Aborting OSGS loop.")
                    success = false
                    break
                end
                
                if !isnothing(accelerator)
                    g_k = get_free_dof_values(final_x0)
                    x_mixed = update!(accelerator, x_prev, g_k)
                    get_free_dof_values(final_x0) .= x_mixed
                end
                
                x_diff = norm(get_free_dof_values(final_x0) - x_prev, Inf)
                
                # Geometrically bind orthogonality tolerance to the exact numerical limitations of the evaluated noise floor
                dynamic_osgs_tol = max(osgs_tol, stagnation_tol)
                
                println("        * Comparing L^inf projection difference against previous relaxation step: \n          x_diff_norm = $x_diff (Dynamic OSGS Tol: $dynamic_osgs_tol)")
                
                if x_diff <= dynamic_osgs_tol
                    println("        [+] OSGS Staggered Fixed-Point loop successfully converged structurally within mathematical boundaries!")
                    success = true
                    break
                end
                
                u_h, p_h = final_x0
                R_u = inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
                R_p = inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf)
                
                # Exclusively assemble dynamic residual right-hand side tracking updated non-linear field
                println("        * Executing staggered global domain numerical integrals to map abstract analytical residuals onto L2 discrete Right-Hand Side vectors (R_u, R_p)...")
                b_u = assemble_vector(v -> ∫(v ⋅ R_u)dΩ, V_proj)
                b_p = assemble_vector(q -> ∫(q * R_p)dΩ, Q_proj)
                
                println("        * Projecting analytical algebraic residuals onto discrete sub-grid scale finite element meshes (pi_u, pi_p) via back-substitution...")
                x_u = allocate_in_domain(M_u); solve!(x_u, num_u_fac, b_u)
                x_p = allocate_in_domain(M_p); solve!(x_p, num_p_fac, b_p)
                
                pi_u = FEFunction(U_proj, x_u)
                pi_p = FEFunction(P_proj, x_p)
            end
        end
        return success, final_x0, iter_count, eval_time
    end
end
