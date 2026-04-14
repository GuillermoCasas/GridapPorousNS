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
    R_u = eval_strong_residual_u(form, u, p, h_cf, alpha_cf, f_cf, c_1, c_2)
    sig_op = SigOp(form.reaction_law, form.regularization, form.ν, c_1, c_2)
    σ = Operation(sig_op)(u, ∇(u), alpha_cf, ∇(alpha_cf), h_cf)
    return apply_projectable_residual_u(form.projection_policy, R_u, σ, u)
end

function inner_projection_p(u, p, form, dΩ, h_cf, alpha_cf, g_cf)
    # Define L2 projection for mass residual natively as a CellField
    R_p = eval_strong_residual_p(form, u, p, alpha_cf, g_cf)
    return apply_projectable_residual_p(form.projection_policy, R_p, form.eps_val, p)
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
    discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)

Generic high-efficiency continuous mapping tool. Evaluates the explicit discrete L2 analytical 
projection of a given analytical function (`field`) specifically onto a constructed target algebraic 
vector domain (`U_proj`, `V_proj`). Requires the pre-allocated topological mass matrix bounds 
(`M_mat`, `num_fac`) strictly to prevent invariant compilation allocations dynamically inside deep solver iterators.
"""
function discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)
    b_vec = assemble_vector(v -> ∫(v ⋅ field)dΩ, V_proj)
    x_solve = allocate_in_domain(M_mat)
    solve!(x_solve, num_fac, b_vec)
    return FEFunction(U_proj, x_solve)
end

"""
    discrete_l2_projection(field, U_proj, V_proj, dΩ)

Convenience wrapper for single-pass diagnostics. Dynamically derives the underlying finite 
element mass matrices mapped to an explicit `LUSolver()`. Highly accurate but algebraically 
unsuitable for deep staggered execution loops.
"""
function discrete_l2_projection(field, U_proj, V_proj, dΩ)
    p_ls = LUSolver()
    M_mat = assemble_matrix((u,v) -> ∫(u ⋅ v)dΩ, U_proj, V_proj)
    num_fac = numerical_setup(symbolic_setup(p_ls, M_mat), M_mat)
    return discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)
end

struct FETopology
    X
    Y
    model
    Ω
    dΩ
    V_free
    Q_free
    h_cf
    f_cf
    alpha_cf
    g_cf
end

struct VMSFormulation
    form
    c_1
    c_2
end

struct IterativeSolvers
    picard
    newton
end

"""
    solve_system(...)

Orchestrates the nonlinear solver iteration loops over a defined Variational Multiscale space.
"""
function solve_system(setup::FETopology, formulation::VMSFormulation, iter_solvers::IterativeSolvers,
                      config::PorousNSConfig, x0;
                      diagnostics_cache=nothing, mms_cfg=nothing)
                      
    X, Y, model, dΩ, Ω = setup.X, setup.Y, setup.model, setup.dΩ, setup.Ω
    h_cf, f_cf, alpha_cf, g_cf = setup.h_cf, setup.f_cf, setup.alpha_cf, setup.g_cf
    V_free, Q_free = setup.V_free, setup.Q_free
    
    form, c_1, c_2 = formulation.form, formulation.c_1, formulation.c_2
    solver_picard, solver_newton = iter_solvers.picard, iter_solvers.newton
    
    phys_cfg = config.physical_properties
    stab_cfg = config.numerical_method.stabilization
    sol_cfg  = config.numerical_method.solver
    diag_cache = isnothing(diagnostics_cache) ? Dict{String, Any}() : diagnostics_cache
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
    res_fn_init(x, y) = build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing)
    jac_picard_init(x, dx, y) = build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    jac_newton_init(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=nothing, pi_p=nothing)

    op_picard_init = FEOperator(res_fn_init, jac_picard_init, X, Y)
    op_newton_init = FEOperator(res_fn_init, jac_newton_init, X, Y)
    
    base_nls_global = solver_newton.nls
    solver_newton_asgs = FESolver(SafeNewtonSolver(base_nls_global.ls, base_nls_global.max_iters, base_nls_global.max_increases, base_nls_global.xtol, base_nls_global.ftol, base_nls_global.linesearch_alpha_min, base_nls_global.c1, base_nls_global.divergence_merit_factor, base_nls_global.stagnation_noise_floor, base_nls_global.max_linesearch_iterations, base_nls_global.linesearch_contraction_factor))

    x0_backup = copy(get_free_dof_values(x0))
    
    eval_time = @elapsed begin
        println("      -> ASGS Initializer: Attempting Exact Newton solver initially...")
        local newton_success = false
        try
            res_newton = solve!(x0, solver_newton_asgs, op_newton_init)
            cache_newton = res_newton isa Tuple ? res_newton[2] : res_newton
            nls_cache = cache_newton isa Tuple ? cache_newton[2] : cache_newton
            final_res = nls_cache.result.iterations > 0 ? nls_cache.result.residual_norm : 0.0
            local_iters = nls_cache.result.iterations
            if final_res <= ftol
                newton_success = true
                iter_count += local_iters
                println("      -> ASGS Initializer: Exact Newton converged vigorously to absolute continuous limits ($ftol)! Bypassing Picard.")
            else
                # When initializing, if it hits noise floor but not ftol, we still allow Picard to attempt homotopy
                # to strictly guarantee we enter the quadratic basin, so we leave newton_success = false.
                iter_count += local_iters
            end
        catch e
            println("      -> ASGS Initializer: Newton ConvergenceError. Exception: ", e)
            newton_success = false
        end
        
        if newton_success
            success = true
        else
            println("      -> ASGS Initializer: Exact Newton structurally aborted loop without geometric saturation. Orchestrating Picard Homotopy fallback...")
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
            
            if any(!isfinite, get_free_dof_values(x0))
                println("      -> ASGS Initializer: Picard catastrophically diverged. Evaluation impossible.")
                success = false
            elseif !success
                println("      -> ASGS Initializer: Picard smoothing finalized. Re-engaging Exact Newton...")
                try
                    res_solve = solve!(x0, solver_newton_asgs, op_newton_init)
                    cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                    nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                    iter_count += nls_cache.result.iterations
                    final_res = nls_cache.result.residual_norm
                    
                    stop_reason = nls_cache.result.stop_reason
                    if stop_reason == "ftol_reached" || stop_reason == "initial_ftol"
                        println("      -> ASGS Initializer: Newton Homotopy Pass achieved exact theoretical tolerance ($ftol).")
                        success = true
                    elseif stop_reason == "stagnation_noise_floor_reached"
                        println("      -> ASGS Initializer: Newton Homotopy Pass cleanly saturated at numerical noise floor ($final_res). Approving.")
                        success = true
                    else
                        println("      -> ASGS Initializer: Newton Homotopy Pass structurally failed (Reason: $stop_reason). Bounding collapse.")
                        success = false
                    end
                catch e
                     println("      -> ASGS Initializer: Newton ConvergenceError on Homotopy Pass. Exception: ", e)
                     success = false
                end
            end
        end
    end
    
    if !success
        diag_cache["pi_u"] = pi_u
        diag_cache["pi_p"] = pi_p
        return false, x0, iter_count, eval_time
    end

    final_x0 = x0

    if method == "ASGS"
        if !isnothing(mms_cfg) && mms_cfg.enabled
            println("\n      [!] Commencing ASGS MMS Error Verification Plateau Loop...")
            
            diag_cache["base_convergence_reached"] = true
            diag_cache["mms_plateau_reached"] = false
            mms_err_hist = []
            mms_rc_hist = []
            
            E_u2_0, E_p2_0, E_u1_0, E_p1_0 = mms_cfg.oracle(final_x0...)
            push!(mms_err_hist, (E_u2_0, E_p2_0, E_u1_0, E_p1_0))
            
            consecutive_passes = 0
            
            base_nls = solver_newton.nls
            local_asgs_nls = SafeNewtonSolver(base_nls.ls, 1, base_nls.max_increases, base_nls.xtol, base_nls.ftol, base_nls.linesearch_alpha_min, base_nls.c1, base_nls.divergence_merit_factor, base_nls.stagnation_noise_floor, base_nls.max_linesearch_iterations, base_nls.linesearch_contraction_factor)
            local_fesolver = FESolver(local_asgs_nls)
            
            eval_time += @elapsed begin
                for cycle in 1:mms_cfg.max_extra_cycles
                    try
                        res_solve = solve!(final_x0, local_fesolver, op_newton_init)
                        cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                        nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                        iter_count += nls_cache.result.iterations
                    catch e
                        if occursin("Reached maximum iterations", string(e)) || occursin("Reached max iterations", string(e))
                            iter_count += 1
                        else
                            println("      -> ASGS MMS Verification Newton failed. Aborting extension. Exception: ", e)
                            diag_cache["mms_stop_reason"] = "nonlinear_failure"
                            break
                        end
                    end
                    
                    E_u2_k, E_p2_k, E_u1_k, E_p1_k = mms_cfg.oracle(final_x0...)
                    push!(mms_err_hist, (E_u2_k, E_p2_k, E_u1_k, E_p1_k))
                    
                    E_u2_prev, E_p2_prev, E_u1_prev, E_p1_prev = mms_err_hist[end-1]
                    
                    r_u2 = abs(E_u2_k - E_u2_prev) / max(E_u2_k, E_u2_prev, mms_cfg.eps_u_l2)
                    r_u1 = abs(E_u1_k - E_u1_prev) / max(E_u1_k, E_u1_prev, mms_cfg.eps_u_h1)
                    r_p2 = abs(E_p2_k - E_p2_prev) / max(E_p2_k, E_p2_prev, mms_cfg.eps_p_l2)
                    
                    push!(mms_rc_hist, (r_u2, r_p2, r_u1))
                    max_r = max(r_u2, r_u1, r_p2)
                    
                    println("        * [Verification Cycle $cycle] Plateau max ratio: $max_r (Target: $(mms_cfg.tau_err))")
                    
                    if max_r < mms_cfg.tau_err
                        consecutive_passes += 1
                    else
                        consecutive_passes = 0
                    end
                    
                    if consecutive_passes >= mms_cfg.require_consecutive_passes
                        println("        [+] ASGS MMS Plateau formally established natively ($consecutive_passes consecutive < $(mms_cfg.tau_err)).")
                        diag_cache["mms_plateau_reached"] = true
                        diag_cache["mms_stop_reason"] = "mms_plateau_satisfied"
                        break
                    end
                end
            end
            
            if !get(diag_cache, "mms_plateau_reached", false) && !haskey(diag_cache, "mms_stop_reason")
                diag_cache["mms_stop_reason"] = "mms_budget_exhausted"
            end
            
            diag_cache["mms_error_history"] = mms_err_hist
            diag_cache["mms_relative_change_history"] = mms_rc_hist
            diag_cache["pi_u"] = pi_u
            diag_cache["pi_p"] = pi_p
            return success, final_x0, iter_count, eval_time
        else
            println("\n      [+] ASGS Formulation exclusively resolved. Exiting solver module.")
            diag_cache["base_convergence_reached"] = true
            diag_cache["mms_stop_reason"] = "base_convergence_only"
            diag_cache["pi_u"] = pi_u
            diag_cache["pi_p"] = pi_p
            return success, final_x0, iter_count, eval_time
        end
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
        
        # Override the Exact Newton solver dynamically for OSGS to perform only a few monolithic iterations
        # per sub-scale projection (e.g., 3 inner steps) rather than unconditionally clamping to 1.
        base_nls = solver_newton.nls
        # Inner tolerancing is determined dynamically in loop
        
        # Expand maximum bounds to absorb the monolithic 1-step iteration workflow
        eff_osgs_iters = max(max_osgs_iters, sol_cfg.newton_iterations + 5)
        if !isnothing(mms_cfg) && mms_cfg.enabled
            eff_osgs_iters += mms_cfg.max_extra_cycles
        end

        eval_time += @elapsed begin
            local accel_u = nothing
            local accel_p = nothing
            if sol_cfg.accelerator.type == "Anderson"
                println("      [+] Initializing Blocked Anderson Acceleration (m = $(sol_cfg.accelerator.m), damping = $(sol_cfg.accelerator.relaxation_factor))")
                accel_u = AndersonAccelerator(sol_cfg.accelerator.m, sol_cfg.accelerator.relaxation_factor, M_u)
                accel_p = AndersonAccelerator(sol_cfg.accelerator.m, sol_cfg.accelerator.relaxation_factor, M_p)
            end
            
            # Map the global ASGS converged solution into the first orthogonal projection cleanly before beginning decoupled evaluation
            println("        * Bootstrapping initial structural orthogonal projection onto converged internal bounds limit...")
            u_h, p_h = final_x0
            R_u = inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
            R_p = inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf)
            
            b_u = assemble_vector(v -> ∫(v ⋅ R_u)dΩ, V_proj)
            b_p = assemble_vector(q -> ∫(q * R_p)dΩ, Q_proj)
            
            x_u = allocate_in_domain(M_u); solve!(x_u, num_u_fac, b_u)
            x_p = allocate_in_domain(M_p); solve!(x_p, num_p_fac, b_p)
            
            pi_u = FEFunction(U_proj, x_u)
            pi_p = FEFunction(P_proj, x_p)
            
            diag_cache["inner_osgs_diagnostics"] = []
            diag_cache["outer_osgs_diagnostics"] = []
            
            prev_pi_drift = 1.0
            prev_x_diff = 1.0

            for osgs_iter in 1:eff_osgs_iters
                println("        [OSGS Iter $osgs_iter]")
                
                tau_inner_m = osgs_iter <= stab_cfg.osgs_warmup_iterations ? max(ftol, stab_cfg.osgs_warmup_tolerance) : ftol
                
                local_osgs_nls = SafeNewtonSolver(base_nls.ls, stab_cfg.osgs_inner_newton_iters, base_nls.max_increases, base_nls.xtol, tau_inner_m, base_nls.linesearch_alpha_min, base_nls.c1, base_nls.divergence_merit_factor, base_nls.stagnation_noise_floor, base_nls.max_linesearch_iterations, base_nls.linesearch_contraction_factor)
                local_fesolver = FESolver(local_osgs_nls)
                
                res_fn_osgs(x, y) = build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p)
                jac_newton_osgs(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=pi_u, pi_p=pi_p)
                
                op_newton_osgs = FEOperator(res_fn_osgs, jac_newton_osgs, X, Y)
                
                x_prev = copy(get_free_dof_values(final_x0))
                
                try
                    res_solve = solve!(final_x0, local_fesolver, op_newton_osgs)
                    cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                    nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                    iter_count += nls_cache.result.iterations
                    
                    push!(diag_cache["inner_osgs_diagnostics"], (
                        iterations = nls_cache.result.iterations,
                        initial_res = nls_cache.result.initial_residual_norm,
                        final_res = nls_cache.result.residual_norm,
                        step_norm = nls_cache.result.step_norm,
                        stop_reason = nls_cache.result.stop_reason
                    ))
                    
                    if nls_cache.result.stop_reason == "linesearch_failed" || nls_cache.result.stop_reason == "merit_divergence_escaped" || nls_cache.result.stop_reason == "linear_solve_nan"
                        println("        -> OSGS Inner Newton sweep failed algebraically (Reason: $(nls_cache.result.stop_reason)). Aborting OSGS nested sequence.")
                        success = false
                        break
                    end
                catch e
                    if occursin("Reached maximum iterations", string(e)) || occursin("Reached max iterations", string(e))
                        println("        -> OSGS Newton 1-step interior pass dynamically executed.")
                        iter_count += 1
                    else
                        println("        -> OSGS Newton failed. Aborting OSGS loop. Exception: ", e)
                        success = false
                        break
                    end
                end
                

                
                x_diff_inf = norm(get_free_dof_values(final_x0) - x_prev, Inf)
                
                # Split separation L2 mapping norms native evaluation
                if stab_cfg.osgs_state_drift_scale == "L2_mass"
                    u_prev, p_prev = FEFunction(X, x_prev)
                    u_h, p_h = final_x0
                    e_u = u_h - u_prev
                    e_p = p_h - p_prev
                    x_diff_u_l2 = sqrt(abs(sum(∫(e_u ⋅ e_u)dΩ)))
                    x_diff_p_l2 = sqrt(abs(sum(∫(e_p * e_p)dΩ)))
                    x_diff = max(x_diff_u_l2, x_diff_p_l2)
                else
                    x_diff = x_diff_inf
                end
                
                # Pre-map the sub-grid scales natively before bounding loop state
                u_h, p_h = final_x0
                R_u = inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
                R_p = inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf)
                
                pi_u_next = discrete_l2_projection(R_u, U_proj, V_proj, dΩ, M_u, num_u_fac)
                pi_p_next = discrete_l2_projection(R_p, P_proj, Q_proj, dΩ, M_p, num_p_fac)
                
                if !isnothing(accel_u) && !isnothing(accel_p)
                    pi_u_mixed = update!(accel_u, get_free_dof_values(pi_u), get_free_dof_values(pi_u_next))
                    pi_p_mixed = update!(accel_p, get_free_dof_values(pi_p), get_free_dof_values(pi_p_next))
                    
                    get_free_dof_values(pi_u_next) .= pi_u_mixed
                    get_free_dof_values(pi_p_next) .= pi_p_mixed
                end
                
                dpi_u_vec = get_free_dof_values(pi_u_next) - get_free_dof_values(pi_u)
                dpi_p_vec = get_free_dof_values(pi_p_next) - get_free_dof_values(pi_p)
                
                pi_u_drift = sqrt(abs(dot(dpi_u_vec, M_u * dpi_u_vec)))
                pi_p_drift = sqrt(abs(dot(dpi_p_vec, M_p * dpi_p_vec)))
                
                b_u_R = assemble_vector(v -> ∫(v ⋅ R_u)dΩ, V_proj)
                R_u_algebraic_norm = norm(b_u_R, 2)
                
                push!(diag_cache["outer_osgs_diagnostics"], (
                    x_diff_inf = x_diff_inf,
                    x_diff_resolved = x_diff,
                    pi_u_drift = pi_u_drift,
                    pi_p_drift = pi_p_drift,
                    R_u_norm = R_u_algebraic_norm
                ))
                
                prev_x_diff = x_diff
                prev_pi_drift = max(pi_u_drift, pi_p_drift)
                
                # Geometrically bind orthogonality tolerance to the exact numerical limitations
                dynamic_osgs_tol = max(osgs_tol, stagnation_tol)
                state_converged = x_diff <= dynamic_osgs_tol
                proj_converged = max(pi_u_drift, pi_p_drift) <= max(stab_cfg.osgs_projection_tolerance, stagnation_tol)
                
                if stab_cfg.osgs_stopping_mode == "state_drift"
                    overall_converged = state_converged
                elseif stab_cfg.osgs_stopping_mode == "projection_drift"
                    overall_converged = proj_converged
                else
                    overall_converged = state_converged && proj_converged
                end
                
                if x_diff_inf <= stagnation_tol && osgs_iter >= 2
                    # Stagnation noise floor exclusively hit uniformly, structurally breaking limits avoids infinite tracking
                    overall_converged = true
                end
                
                println("        * Comparing limits against previous relaxation step: \n          x_diff_inf = $x_diff_inf | x_diff_mode_$(stab_cfg.osgs_state_drift_scale) = $x_diff \n          pi_u_drift = $pi_u_drift | pi_p_drift = $pi_p_drift \n          overall_converged = $overall_converged")
                
                if overall_converged
                    pi_u = pi_u_next
                    pi_p = pi_p_next
                    
                    if !get(diag_cache, "base_convergence_reached", false)
                        println("        [+] OSGS Base Staggered Fixed-Point nominally mathematically converged natively!")
                        diag_cache["base_convergence_reached"] = true
                        
                        if !isnothing(mms_cfg) && mms_cfg.enabled
                            diag_cache["mms_error_history"] = []
                            diag_cache["mms_relative_change_history"] = []
                            diag_cache["mms_consecutive_passes"] = 0
                            
                            E_u2_0, E_p2_0, E_u1_0, E_p1_0 = mms_cfg.oracle(final_x0...)
                            push!(diag_cache["mms_error_history"], (E_u2_0, E_p2_0, E_u1_0, E_p1_0))
                        end
                    else
                        if !isnothing(mms_cfg) && mms_cfg.enabled
                            E_u2_k, E_p2_k, E_u1_k, E_p1_k = mms_cfg.oracle(final_x0...)
                            mms_err_hist = diag_cache["mms_error_history"]
                            push!(mms_err_hist, (E_u2_k, E_p2_k, E_u1_k, E_p1_k))
                            
                            E_u2_prev, E_p2_prev, E_u1_prev, E_p1_prev = mms_err_hist[end-1]
                            r_u2 = abs(E_u2_k - E_u2_prev) / max(E_u2_k, E_u2_prev, mms_cfg.eps_u_l2)
                            r_u1 = abs(E_u1_k - E_u1_prev) / max(E_u1_k, E_u1_prev, mms_cfg.eps_u_h1)
                            r_p2 = abs(E_p2_k - E_p2_prev) / max(E_p2_k, E_p2_prev, mms_cfg.eps_p_l2)
                            
                            mms_rc_hist = diag_cache["mms_relative_change_history"]
                            push!(mms_rc_hist, (r_u2, r_p2, r_u1))
                            max_r = max(r_u2, r_u1, r_p2)
                            
                            println("        * [Verification Cycle] Plateau max ratio: $max_r (Target: $(mms_cfg.tau_err))")
                            
                            pass_count = diag_cache["mms_consecutive_passes"]
                            if max_r < mms_cfg.tau_err
                                pass_count += 1
                            else
                                pass_count = 0
                            end
                            diag_cache["mms_consecutive_passes"] = pass_count
                            
                            if pass_count >= mms_cfg.require_consecutive_passes
                                println("        [+] OSGS MMS Plateau formally established natively ($pass_count consecutive < $(mms_cfg.tau_err)).")
                                diag_cache["mms_plateau_reached"] = true
                                diag_cache["mms_stop_reason"] = "mms_plateau_satisfied"
                                success = true
                                break
                            end
                        end
                    end
                    
                    if isnothing(mms_cfg) || !mms_cfg.enabled
                        println("        [+] OSGS Staggered Fixed-Point loop successfully converged structurally within mathematical boundaries!")
                        diag_cache["mms_stop_reason"] = "base_convergence_only"
                        success = true
                        break
                    end
                else
                    if !isnothing(mms_cfg) && mms_cfg.enabled
                        diag_cache["mms_consecutive_passes"] = 0
                    end
                end
                
                pi_u = pi_u_next
                pi_p = pi_p_next
            end
        end
        if !isnothing(mms_cfg) && mms_cfg.enabled
            if get(diag_cache, "base_convergence_reached", false) && !get(diag_cache, "mms_plateau_reached", false)
                println("        [!] OSGS hit maximum extended cycle boundaries without formal plateau verification. Assuming base convergence.")
                diag_cache["mms_stop_reason"] = "mms_budget_exhausted"
                success = true
            end
        end
        
        diag_cache["pi_u"] = pi_u
        diag_cache["pi_p"] = pi_p
        return success, final_x0, iter_count, eval_time
    end
end
