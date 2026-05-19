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
    p_ls = CholeskySolver()  # SPD mass matrix — see `CholeskySolver` in `linear_solvers.jl`.
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
    safe_fe_solve!(x, fesolver, op; backup=nothing)

Single-attempt FE-solver wrapper that absorbs the boilerplate every cascade
site in `solve_system` repeats: tuple unwrapping of Gridap's `solve!` return
value, non-finite-DOF guard with optional backup restoration, and exception
classification (Gridap's old "Reached maximum iterations" string vs other
exceptions).

Does NOT classify success/failure or do logging — the criteria differ across
the six call sites (Stage I Newton 1 / Picard / Newton 2 and OSGS inner
Newton 1 / Picard / Newton 2 retry), so each caller owns its own decision
logic. This helper exists purely to eliminate the duplicated try/catch +
tuple-unwrap + finite-check boilerplate.

Returns a `NamedTuple` with field `state`:
- `:ok` — finite state; also carries `iterations`, `residual_norm`,
  `initial_residual_norm`, `step_norm`, `stop_reason` from the result cache.
- `:nonfinite` — at least one DOF was non-finite; if `backup !== nothing`,
  the state was restored from `backup` before returning.
- `:max_iters_caught` — `solve!` threw a "Reached max(imum) iterations"
  exception. This is a legacy Gridap contract; `SafeNewtonSolver` uses
  `stop_reason = "max_iters_stagnation"` on the `:ok` path instead. Kept
  for defence against legacy or non-`SafeNewtonSolver` solver instances.
- `:exception` — `solve!` threw any other exception. Field `exc` holds it.

The backup restoration on `:nonfinite` is the only side effect beyond
`solve!`'s in-place mutation of `x`.
"""
function safe_fe_solve!(x, fesolver, op; backup=nothing)
    try
        res = solve!(x, fesolver, op)
        cache = res isa Tuple ? res[2] : res
        nls_cache = cache isa Tuple ? cache[2] : cache
        if any(!isfinite, get_free_dof_values(x))
            if backup !== nothing
                get_free_dof_values(x) .= backup
            end
            return (state = :nonfinite,)
        end
        r = nls_cache.result
        return (state = :ok,
                iterations = r.iterations,
                residual_norm = r.residual_norm,
                initial_residual_norm = r.initial_residual_norm,
                step_norm = r.step_norm,
                stop_reason = r.stop_reason)
    catch e
        msg = string(e)
        if occursin("Reached maximum iterations", msg) || occursin("Reached max iterations", msg)
            return (state = :max_iters_caught,)
        else
            return (state = :exception, exc = e)
        end
    end
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
    ftol = solver_newton.nls.ftol
    stagnation_tol = solver_newton.nls.stagnation_noise_floor
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
        res = safe_fe_solve!(x0, solver_newton_asgs, op_newton_init; backup=x0_backup)
        if res.state == :nonfinite
            println("      -> ASGS Initializer: Exact Newton produced non-finite state. Restoring backup, falling through to Picard.")
            newton_success = false
        elseif res.state == :exception
            println("      -> ASGS Initializer: Newton ConvergenceError. Exception: ", res.exc)
            newton_success = false
        elseif res.state == :max_iters_caught
            # Preserve pre-refactor behavior: Stage I Newton 1's old `catch` block
            # logged any exception (including the legacy "Reached max iterations"
            # path) as "ConvergenceError" and set newton_success = false. Kept
            # bit-identical here.
            println("      -> ASGS Initializer: Newton ConvergenceError. Exception: Reached maximum iterations")
            newton_success = false
        else  # :ok
            final_res = res.iterations > 0 ? res.residual_norm : 0.0
            diag_cache["final_residual_norm"] = final_res
            local_iters = res.iterations
            if final_res <= ftol
                newton_success = true
                iter_count += local_iters
                println("      -> ASGS Initializer: Exact Newton converged vigorously to absolute continuous limits ($ftol)! Bypassing Picard.")
            else
                # When initializing, if it hits noise floor but not ftol, we still allow Picard to attempt homotopy
                # to strictly guarantee we enter the quadratic basin, so we leave newton_success = false.
                iter_count += local_iters
            end
        end
        
        if newton_success
            success = true
        else
            println("      -> ASGS Initializer: Exact Newton structurally aborted loop without geometric saturation. Orchestrating Picard Homotopy fallback...")
            get_free_dof_values(x0) .= x0_backup

            # NOTE: helper's backup-restore-on-nonfinite is intentionally NOT used
            # here. The pre-refactor code performed an *explicit* secondary `any(!isfinite, …)`
            # check after the catch block to handle catastrophic Picard divergence
            # (Picard can produce non-finite mid-iteration even without throwing).
            # That secondary check is preserved verbatim below.
            res_p = safe_fe_solve!(x0, solver_picard, op_picard_init)
            if res_p.state == :ok
                final_res_picard = res_p.residual_norm
                diag_cache["final_residual_norm"] = final_res_picard
                iter_count += res_p.iterations
                if final_res_picard <= ftol
                    println("      -> ASGS Initializer: Picard fully converged! Escaping to evaluation.")
                    success = true
                end
            elseif res_p.state == :max_iters_caught
                println("      -> ASGS Initializer: Picard concluded initial smoothing loops.")
            elseif res_p.state == :exception
                println("      -> ASGS Initializer: Picard threw unrecoverable error. Exception: ", res_p.exc)
                get_free_dof_values(x0) .= x0_backup
            end
            # `:nonfinite` is handled by the secondary `any(!isfinite, …)` check
            # below (preserved from pre-refactor); the helper does not restore
            # backup automatically here because we did not pass `backup=`.

            if any(!isfinite, get_free_dof_values(x0))
                println("      -> ASGS Initializer: Picard catastrophically diverged. Restoring backup, evaluation impossible.")
                get_free_dof_values(x0) .= x0_backup
                success = false
            elseif !success
                println("      -> ASGS Initializer: Picard smoothing finalized. Re-engaging Exact Newton...")
                res2 = safe_fe_solve!(x0, solver_newton_asgs, op_newton_init; backup=x0_backup)
                if res2.state == :nonfinite
                    println("      -> ASGS Initializer: Newton Homotopy Pass produced non-finite state. Restoring backup, marking failure.")
                    success = false
                elseif res2.state == :exception
                    println("      -> ASGS Initializer: Newton ConvergenceError on Homotopy Pass. Exception: ", res2.exc)
                    success = false
                elseif res2.state == :max_iters_caught
                    # Preserve pre-refactor behavior: lumped with generic ConvergenceError.
                    println("      -> ASGS Initializer: Newton ConvergenceError on Homotopy Pass. Exception: Reached maximum iterations")
                    success = false
                else  # :ok
                    iter_count += res2.iterations
                    final_res = res2.residual_norm
                    diag_cache["final_residual_norm"] = final_res

                    stop_reason = res2.stop_reason
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
                end
            end
        end
    end
    
    # `mms_plateau_success::Union{Bool,Nothing}` is `nothing` when MMS verification is
    # disabled; otherwise it reports whether the MMS plateau was formally established
    # (independent of the inner-solver `success` flag). Splitting these allows callers
    # to detect the "solver converged but MMS budget exhausted" regression case
    # explicitly. See plan Fix 6 / P-007.
    mms_enabled = !isnothing(mms_cfg) && mms_cfg.enabled
    _mms_plateau() = mms_enabled ? get(diag_cache, "mms_plateau_reached", false) : nothing

    if !success
        diag_cache["pi_u"] = pi_u
        diag_cache["pi_p"] = pi_p
        return false, _mms_plateau(), x0, iter_count, eval_time
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
                        diag_cache["final_residual_norm"] = nls_cache.result.residual_norm
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
                    
                    # §5.1: scale plateau floors by the FE discretization budget.
                    # L² convergence rate is h^(kv+1); H¹ rate is h^kv. The config field
                    # `mms_cfg.eps_*` is now a *baseline* (dimensionless) coefficient;
                    # the effective floor is baseline × h^(rate).
                    eps_u_l2_eff = mms_cfg.eps_u_l2 * mms_cfg.h_local^(mms_cfg.kv + 1)
                    eps_u_h1_eff = mms_cfg.eps_u_h1 * mms_cfg.h_local^mms_cfg.kv
                    eps_p_l2_eff = mms_cfg.eps_p_l2 * mms_cfg.h_local^(mms_cfg.kv + 1)

                    r_u2 = abs(E_u2_k - E_u2_prev) / max(E_u2_k, E_u2_prev, eps_u_l2_eff)
                    r_u1 = abs(E_u1_k - E_u1_prev) / max(E_u1_k, E_u1_prev, eps_u_h1_eff)
                    r_p2 = abs(E_p2_k - E_p2_prev) / max(E_p2_k, E_p2_prev, eps_p_l2_eff)

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

                        # §5.2: rate-aware sanity check. If the converged L² velocity error
                        # exceeds the FE accuracy budget by more than `rate_check_factor`,
                        # the iteration plateaued at a non-FE-optimal state. Flag via the
                        # stop reason; keep success in the calling context (the plateau IS
                        # established, just at a sub-optimal level).
                        budget = mms_cfg.h_local^(mms_cfg.kv + 1)
                        if E_u2_k > mms_cfg.rate_check_factor * budget
                            diag_cache["mms_stop_reason"] = "mms_plateau_at_suboptimal_rate"
                            println("        [!] Plateau at sub-optimal rate: E_u2=$E_u2_k > $(mms_cfg.rate_check_factor) × h^(kv+1)=$(mms_cfg.rate_check_factor * budget). Flagged for inspection.")
                        end

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
            return success, _mms_plateau(), final_x0, iter_count, eval_time
        else
            println("\n      [+] ASGS Formulation exclusively resolved. Exiting solver module.")
            diag_cache["base_convergence_reached"] = true
            diag_cache["mms_stop_reason"] = "base_convergence_only"
            diag_cache["pi_u"] = pi_u
            diag_cache["pi_p"] = pi_p
            return success, _mms_plateau(), final_x0, iter_count, eval_time
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

        # The L² mass matrices are symmetric positive-definite by construction. Cholesky
        # via CHOLMOD is the mathematically honest factorization here: it's faster than
        # the previous `LUSolver()` (no partial pivoting overhead), uses less memory,
        # and preserves the symmetry of operations downstream. The factor is reused
        # across every OSGS outer iteration and every MMS-plateau re-projection.
        ls_u = CholeskySolver()
        ls_p = CholeskySolver()
        num_u_fac = numerical_setup(symbolic_setup(ls_u, M_u), M_u)
        num_p_fac = numerical_setup(symbolic_setup(ls_p, M_p), M_p)
        
        # Override the Exact Newton solver dynamically for OSGS to perform only a few monolithic iterations
        # per sub-scale projection (e.g., 3 inner steps) rather than unconditionally clamping to 1.
        base_nls = solver_newton.nls
        # Inner tolerancing is determined dynamically in loop

        # OSGS outer-iteration budget is exactly the configured `stab_cfg.osgs_iterations`.
        # When MMS plateau verification is active, the explicit `mms_cfg.max_extra_cycles`
        # extension is added on top so the verifier has its own clearly-named budget.
        eff_osgs_iters = max_osgs_iters
        if !isnothing(mms_cfg) && mms_cfg.enabled
            eff_osgs_iters += mms_cfg.max_extra_cycles
        end

        eval_time += @elapsed begin
            local accel_u = nothing
            local accel_p = nothing
            if sol_cfg.accelerator.type == "Anderson"
                println("      [+] Initializing Blocked Anderson Acceleration (m = $(sol_cfg.accelerator.m), damping = $(sol_cfg.accelerator.relaxation_factor), safety = $(sol_cfg.accelerator.safety_factor))")
                accel_u = AndersonAccelerator(sol_cfg.accelerator.m, sol_cfg.accelerator.relaxation_factor, sol_cfg.accelerator.safety_factor, M_u)
                accel_p = AndersonAccelerator(sol_cfg.accelerator.m, sol_cfg.accelerator.relaxation_factor, sol_cfg.accelerator.safety_factor, M_p)
            end
            
            # Initialize the orthogonal projection π_h^0 = 0. The first OSGS outer
            # iteration then re-solves U_h against π_h = 0 — which IS the ASGS problem,
            # already converged — and produces the first non-trivial projection naturally.
            # Avoids the off-distribution bootstrap that the warmup phase had to unwind.
            println("        * Initializing orthogonal projection π_h^0 = 0 (first iter re-derives from ASGS state)...")
            x_u = allocate_in_domain(M_u); fill!(x_u, 0.0)
            x_p = allocate_in_domain(M_p); fill!(x_p, 0.0)
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

                local_osgs_nls_picard = SafeNewtonSolver(base_nls.ls, stab_cfg.osgs_inner_newton_iters, base_nls.max_increases, base_nls.xtol, sol_cfg.picard_handoff_ftol, base_nls.linesearch_alpha_min, base_nls.c1, base_nls.divergence_merit_factor, base_nls.stagnation_noise_floor, base_nls.max_linesearch_iterations, base_nls.linesearch_contraction_factor; mode=:picard)
                local_fesolver_picard = FESolver(local_osgs_nls_picard)

                res_fn_osgs(x, y) = build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p)
                jac_newton_osgs(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=pi_u, pi_p=pi_p)
                jac_picard_osgs(x, dx, y) = build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p, mult_mom=1.0, mult_mass=1.0)

                op_newton_osgs = FEOperator(res_fn_osgs, jac_newton_osgs, X, Y)
                op_picard_osgs = FEOperator(res_fn_osgs, jac_picard_osgs, X, Y)

                x_prev = copy(get_free_dof_values(final_x0))

                # --- Attempt 1: OSGS Inner Newton ---
                newton_failed = false
                res_n1 = safe_fe_solve!(final_x0, local_fesolver, op_newton_osgs; backup=x_prev)
                if res_n1.state == :nonfinite
                    println("        -> OSGS Inner Newton produced non-finite state. Restoring previous iterate, engaging Picard fallback.")
                    newton_failed = true
                elseif res_n1.state == :max_iters_caught
                    println("        -> OSGS Newton 1-step interior pass dynamically executed.")
                    iter_count += 1
                elseif res_n1.state == :exception
                    println("        -> OSGS Newton threw unrecoverable exception: ", res_n1.exc, ". Attempting Picard fallback.")
                    newton_failed = true
                else  # :ok
                    iter_count += res_n1.iterations

                    push!(diag_cache["inner_osgs_diagnostics"], (
                        iterations = res_n1.iterations,
                        initial_res = res_n1.initial_residual_norm,
                        final_res = res_n1.residual_norm,
                        step_norm = res_n1.step_norm,
                        stop_reason = res_n1.stop_reason
                    ))
                    diag_cache["final_residual_norm"] = res_n1.residual_norm

                    # P-003 back-pointer: only the three structural-failure stop reasons below
                    # mark the inner Newton as failed. `xtol_stagnation` and `max_iters_stagnation`
                    # (IterationCap) are accepted as outer-loop progress by design — see
                    # `theory/osgs_algorithm.tex` §1.2.4 L1118. Do not add IterationCap to this
                    # set without first revising the algorithm document; tightening here would
                    # cause OSGS to reject converged-but-not-tight inner iterates.
                    if res_n1.stop_reason in ("linesearch_failed", "merit_divergence_escaped", "linear_solve_nan")
                        newton_failed = true
                    end
                end

                # --- Attempt 2 (only if Newton failed): Picard smoother, then Attempt 3: Newton retry ---
                if newton_failed
                    println("        -> OSGS Inner Newton failed. Restoring previous iterate and engaging Picard smoother...")
                    get_free_dof_values(final_x0) .= x_prev
                    picard_completed = false
                    # Helper's backup is NOT passed: pre-refactor behavior performed a
                    # secondary explicit `any(!isfinite, …)` check below to catch
                    # catastrophic divergence in either the OK or the exception path.
                    # Preserve verbatim.
                    res_p = safe_fe_solve!(final_x0, local_fesolver_picard, op_picard_osgs)
                    if res_p.state == :ok
                        iter_count += res_p.iterations
                        push!(diag_cache["inner_osgs_diagnostics"], (
                            iterations = res_p.iterations,
                            initial_res = res_p.initial_residual_norm,
                            final_res = res_p.residual_norm,
                            step_norm = res_p.step_norm,
                            stop_reason = "picard:" * res_p.stop_reason
                        ))
                        diag_cache["final_residual_norm"] = res_p.residual_norm
                        picard_completed = true
                    elseif res_p.state == :max_iters_caught
                        println("        -> OSGS Picard concluded smoothing loops.")
                        picard_completed = true
                    elseif res_p.state == :exception
                        println("        -> OSGS Picard threw unrecoverable exception: ", res_p.exc, ". Restoring backup and aborting OSGS.")
                        get_free_dof_values(final_x0) .= x_prev
                    end
                    # `:nonfinite` falls through to the explicit check below (helper did
                    # not auto-restore because `backup=` was not passed).

                    if any(!isfinite, get_free_dof_values(final_x0))
                        println("        -> OSGS Picard catastrophically diverged. Restoring and aborting OSGS.")
                        get_free_dof_values(final_x0) .= x_prev
                        success = false
                        break
                    end

                    if !picard_completed
                        success = false
                        break
                    end

                    # --- Attempt 3: re-engage Newton on the Picard-smoothed iterate ---
                    println("        -> OSGS Picard smoothing finalized. Re-engaging Newton...")
                    res_n2 = safe_fe_solve!(final_x0, local_fesolver, op_newton_osgs; backup=x_prev)
                    if res_n2.state == :nonfinite
                        println("        -> OSGS Inner Newton retry produced non-finite state. Restoring previous iterate, aborting OSGS nested sequence.")
                        success = false
                        break
                    elseif res_n2.state == :max_iters_caught
                        iter_count += 1
                    elseif res_n2.state == :exception
                        println("        -> OSGS Newton retry threw unrecoverable exception: ", res_n2.exc, ". Aborting OSGS loop.")
                        success = false
                        break
                    else  # :ok
                        iter_count += res_n2.iterations
                        push!(diag_cache["inner_osgs_diagnostics"], (
                            iterations = res_n2.iterations,
                            initial_res = res_n2.initial_residual_norm,
                            final_res = res_n2.residual_norm,
                            step_norm = res_n2.step_norm,
                            stop_reason = "retry:" * res_n2.stop_reason
                        ))
                        diag_cache["final_residual_norm"] = res_n2.residual_norm

                        if res_n2.stop_reason in ("linesearch_failed", "merit_divergence_escaped", "linear_solve_nan")
                            println("        -> OSGS Inner Newton retry also failed (Reason: $(res_n2.stop_reason)). Aborting OSGS nested sequence.")
                            success = false
                            break
                        end
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
                    if stab_cfg.osgs_stopping_mode == "state_drift"
                        # Stagnation noise floor on ℓ∞ acts as a safety net only when state drift
                        # is the binding criterion. In "projection_drift" and "both" modes, the
                        # projection may still be drifting legitimately — don't short-circuit.
                        overall_converged = true
                    end
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

                            # §5.1: h-scaled plateau floors (see ASGS site for full comment).
                            eps_u_l2_eff = mms_cfg.eps_u_l2 * mms_cfg.h_local^(mms_cfg.kv + 1)
                            eps_u_h1_eff = mms_cfg.eps_u_h1 * mms_cfg.h_local^mms_cfg.kv
                            eps_p_l2_eff = mms_cfg.eps_p_l2 * mms_cfg.h_local^(mms_cfg.kv + 1)

                            r_u2 = abs(E_u2_k - E_u2_prev) / max(E_u2_k, E_u2_prev, eps_u_l2_eff)
                            r_u1 = abs(E_u1_k - E_u1_prev) / max(E_u1_k, E_u1_prev, eps_u_h1_eff)
                            r_p2 = abs(E_p2_k - E_p2_prev) / max(E_p2_k, E_p2_prev, eps_p_l2_eff)
                            
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

                                # §5.2: rate-aware sanity check (see ASGS site for full comment).
                                budget = mms_cfg.h_local^(mms_cfg.kv + 1)
                                if E_u2_k > mms_cfg.rate_check_factor * budget
                                    diag_cache["mms_stop_reason"] = "mms_plateau_at_suboptimal_rate"
                                    println("        [!] Plateau at sub-optimal rate: E_u2=$E_u2_k > $(mms_cfg.rate_check_factor) × h^(kv+1)=$(mms_cfg.rate_check_factor * budget). Flagged for inspection.")
                                end

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
        return success, _mms_plateau(), final_x0, iter_count, eval_time
    end
end
