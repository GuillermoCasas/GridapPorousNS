function execute_outer_homotopy_perturbation_loop!(
    setup::PorousNSSolver.FETopology, formulation::PorousNSSolver.VMSFormulation,
    iter_solvers::PorousNSSolver.IterativeSolvers, config::PorousNSConfig,
    method::String, dynamic_ftol::Float64, mms_setup::MMSSetup, pert_cfg::PerturbationConfig,
    mms_verification_enabled::Bool, mms_tau_err, mms_eps_u_l2, mms_eps_u_h1, mms_eps_p_l2,
    mms_max_extra_cycles, mms_require_consecutive_passes
)
    success = false
    successful_eps = -1.0
    eval_time = 0.0
    iter_count_attempt = 0
    final_x0 = nothing
    final_residual_attempt = NaN
    
    for attempt in 0:(pert_cfg.max_n_pert + 1)
        eps_p = attempt <= pert_cfg.max_n_pert ? pert_cfg.eps_pert_base / (10.0^attempt) : 0.0
        
        u_0_func = PerturbationFunc(mms_setup.u_final, mms_setup.h_raw_func, eps_p * (mms_setup.u_ex_L2 / mms_setup.norm_h))
        x0 = interpolate_everywhere([u_0_func, mms_setup.p_final], setup.X)
        
        println("\n    ==================================================")
        println("    [Attempt $(attempt+1)/$(pert_cfg.max_n_pert + 2)] Homotopy Perturbation Scale: eps_pert = $eps_p")
        println("    [!] Delegating orchestration to PDE assembly module via `src/solvers/porous_solver.jl` (Mode: $method)")
        println("    ==================================================")
        
        local_stab_cfg = PorousNSSolver.StabilizationConfig(
            method=method,
            osgs_iterations=config.numerical_method.stabilization.osgs_iterations,
            osgs_inner_newton_iters=config.numerical_method.stabilization.osgs_inner_newton_iters,
            osgs_tolerance=dynamic_ftol,
            osgs_stopping_mode=config.numerical_method.stabilization.osgs_stopping_mode,
            osgs_projection_tolerance=dynamic_ftol,
            osgs_state_drift_scale=config.numerical_method.stabilization.osgs_state_drift_scale
        )
        
        local_diagnostics_cache = Dict{String, Any}()
        mms_cfg = nothing
        if mms_verification_enabled
            mms_cfg = (
                enabled = true, tau_err = mms_tau_err, eps_u_l2 = mms_eps_u_l2, eps_u_h1 = mms_eps_u_h1,
                eps_p_l2 = mms_eps_p_l2, max_extra_cycles = mms_max_extra_cycles,
                require_consecutive_passes = mms_require_consecutive_passes,
                oracle = (uh, ph) -> calculate_normalized_errors(uh, ph, mms_setup.u_final, mms_setup.p_final, mms_setup.U_c, mms_setup.P_c, mms_setup.L, setup.dΩ)
            )
        end
        
        sys_success, sys_final_x0, sys_iter_count, sys_eval_time = PorousNSSolver.solve_system(
            setup, formulation, iter_solvers,
            config, x0;
            diagnostics_cache=local_diagnostics_cache, mms_cfg=mms_cfg
        )
        
        if haskey(local_diagnostics_cache, "mms_stop_reason")
            println("    -> MMS Plateau Reason: ", local_diagnostics_cache["mms_stop_reason"])
            println("    -> Base Convergence Reached: ", get(local_diagnostics_cache, "base_convergence_reached", false))
            println("    -> MMS Plateau Reached: ", get(local_diagnostics_cache, "mms_plateau_reached", false))
        end
        
        if sys_success
            println("\n      [✅] Full non-linear algebraic system converged gracefully mathematically! Escaping constraint loop.")
            success = true
            successful_eps = eps_p
            final_x0 = sys_final_x0
            eval_time = sys_eval_time
            iter_count_attempt = sys_iter_count
            final_residual_attempt = get(local_diagnostics_cache, "final_residual_norm", NaN)
            break
        else
            println("\n      [❌] Outer loop execution completely stalled structurally above convergence tolerance (\`$(local_stab_cfg.osgs_tolerance)\`) or system fully diverged.")
            final_residual_attempt = get(local_diagnostics_cache, "final_residual_norm", NaN)
        end
    end
    
    if !success
         println("    [WARNING] Completely failed to find root basin. Returning NaN.")
    end
    
    return success, successful_eps, final_x0, eval_time, iter_count_attempt, final_residual_attempt
end

struct PerturbationFunc{F1, F2} <: Function
    u_base::F1
    h_func::F2
    scale::Float64
end
