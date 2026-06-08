# src/solvers/osgs_solver.jl
#
# OSGS (Orthogonal Sub-Grid Scale) — the coupled Stage-II solve (Algorithm C of
# `theory/osgs_algorithm/osgs_algorithm.tex`). The SGS space is taken strictly orthogonal to the FE
# space (X_{h0}^⊥): we compute the L²-projection π_h = Π(R(U_h)) of the strong residual and subtract it
# in the sub-scale equation. ASGS recovers π_h = 0 (identity projection) and lives in asgs_solver.jl.
#
# Included AFTER asgs_solver.jl, which defines the shared core this file uses: the structs
# FETopology/VMSFormulation (named in `solve_osgs_stage!`'s signature, so they must exist at parse
# time — hence the include-order constraint), plus safe_fe_solve!, _record_stage!, cascade_step_outcome,
# _pingpong_cascade!, OSGS_INNER_POLICY, the SolutionVerifier hooks, and _with_overrides (nonlinear.jl).
# CholeskySolver lives in linear_solvers.jl.
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
element mass matrices mapped to an explicit `CholeskySolver()`. Highly accurate but algebraically
unsuitable for deep execution loops.
"""
function discrete_l2_projection(field, U_proj, V_proj, dΩ)
    p_ls = CholeskySolver()  # SPD mass matrix — see `CholeskySolver` in `linear_solvers.jl`.
    M_mat = assemble_matrix((u,v) -> ∫(u ⋅ v)dΩ, U_proj, V_proj)
    num_fac = numerical_setup(symbolic_setup(p_ls, M_mat), M_mat)
    return discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)
end

# ==============================================================================
# solve_osgs_stage!  (Algorithm C — the coupled OSGS solve)
# ==============================================================================
"""
    solve_osgs_stage!(success, final_x0, setup::FETopology, formulation::VMSFormulation,
                      config::PorousNSConfig, solver_newton, verifier, diag_cache, iter_count_ref)
        → (success::Bool, pi_u, pi_p, elapsed::Float64)

The single OSGS entry point the orchestrator dispatches to. Builds the unconstrained projection spaces
and the Cholesky-factored L² mass matrices, then runs the coupled solve: ONE Newton solve whose residual
recomputes `π = Π(R(u))` from the current iterate at every evaluation (local frozen-π Jacobian — sparse,
NOT the prohibitive monolithic ∂π/∂u), so it is a Picard-type coupling, linearly convergent to the
discrete OSGS fixed point `R̃ = R − Π(R)`. The shared Newton↔Picard cascade is the fallback
(`pingpong_enabled`); the stall sensor is disabled for this solve (it converges slowly-monotone).

`success` carries Stage I's flag forward: it must survive a budget-exhausted, verification-disabled exit,
so it is reassigned only by an explicit structural-failure / converged verdict below. After convergence,
`on_osgs_converged!(verifier, …)` runs the optional verification (NoVerification ⇒ no-op).

Returns the elapsed wall time of the timed coupled solve only; the mass-matrix assembly/factorization
runs outside the `@elapsed` region (it is setup, not solve).
"""
function solve_osgs_stage!(success, final_x0, setup::FETopology, formulation::VMSFormulation,
                           config::PorousNSConfig, solver_newton, verifier, diag_cache, iter_count_ref)
    initial_success = success

    # Local unpacking (cheap pointer aliases for Gridap performance)
    X, Y = setup.X, setup.Y
    V_free, Q_free = setup.V_free, setup.Q_free
    dΩ = setup.dΩ
    h_cf, f_cf, alpha_cf, g_cf = setup.h_cf, setup.f_cf, setup.alpha_cf, setup.g_cf
    form, c_1, c_2 = formulation.form, formulation.c_1, formulation.c_2
    phys_cfg = config.physical_properties
    sol_cfg  = config.numerical_method.solver
    freeze_cusp = sol_cfg.freeze_jacobian_cusp
    base_nls = solver_newton.nls

    println("\n      [+] Commencing Orthogonal Subgrid Scale (OSGS) coupled solve...")

    # Gridap L² projections inherit the boundary conditions of the FESpace they map onto. Project on the
    # UNCONSTRAINED V_free/Q_free: projecting on the Dirichlet-constrained space injects an O(1) boundary
    # residual that destroys the O(h^{k+1}) MMS convergence rate.
    U, P = X
    V, Q = Y
    V_proj = V_free !== nothing ? V_free : V
    Q_proj = Q_free !== nothing ? Q_free : Q
    U_proj = TrialFESpace(V_proj)
    P_proj = TrialFESpace(Q_proj)

    # L² mass matrices are SPD and functionally invariant across the solve, so assemble + Cholesky-factor
    # them exactly once (CHOLMOD; reused by every per-evaluation projection). This is setup, not solve, so
    # it lives outside the timed region.
    M_u = assemble_matrix((u,v) -> ∫(v ⋅ u)dΩ, U_proj, V_proj)
    M_p = assemble_matrix((p,q) -> ∫(q * p)dΩ, P_proj, Q_proj)
    num_u_fac = numerical_setup(symbolic_setup(CholeskySolver(), M_u), M_u)
    num_p_fac = numerical_setup(symbolic_setup(CholeskySolver(), M_p), M_p)

    pi_u = nothing
    pi_p = nothing

    coupled_elapsed = @elapsed begin
        println("      [+] OSGS COUPLED mode: single Newton solve; π = Π(R(u)) recomputed each nonlinear iteration (local frozen-π Jacobian; non-monolithic).")
        diag_cache["inner_osgs_diagnostics"] = []
        diag_cache["outer_osgs_diagnostics"] = []

        # Residual: recompute π from the CURRENT iterate every evaluation (no lag).
        res_fn_coupled = (x, y) -> begin
            u_x, p_x = x
            R_u_cf = inner_projection_u(u_x, p_x, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
            R_p_cf = inner_projection_p(u_x, p_x, form, dΩ, h_cf, alpha_cf, g_cf)
            pi_u_x = discrete_l2_projection(R_u_cf, U_proj, V_proj, dΩ, M_u, num_u_fac)
            pi_p_x = discrete_l2_projection(R_p_cf, P_proj, Q_proj, dΩ, M_p, num_p_fac)
            build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u_x, pi_p=pi_p_x)
        end
        # Jacobian: local frozen-π form. The π VALUE is irrelevant to the Jacobian (ProjectFullResidual
        # ignores it; the reaction-trim uses σ/∂σ); a zero π just selects the OSGS (is_osgs=true) branch.
        _zu = allocate_in_domain(M_u); fill!(_zu, 0.0); _pi_u0 = FEFunction(U_proj, _zu)
        _zp = allocate_in_domain(M_p); fill!(_zp, 0.0); _pi_p0 = FEFunction(P_proj, _zp)
        jac_fn_coupled = (x, dx, y) -> build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=_pi_u0, pi_p=_pi_p0)

        op_coupled = FEOperator(res_fn_coupled, jac_fn_coupled, X, Y)
        x_backup = copy(get_free_dof_values(final_x0))
        # The coupled inexact-Newton converges SLOWLY-MONOTONE (the dropped ∂π/∂U gives a linear rate);
        # DISABLE the stall sensor for it (a slow step is not a stall). With it on, the high-Da/fine-mesh
        # coupled solve fails the "improve by stall_min_rel_improvement over stall_window steps" test and
        # bails after ~2 steps, silently leaving U at the ASGS state — OSGS degenerates into ASGS and reports
        # ASGS's (optimal) rate under the OSGS label. Its genuine failures (line-search failure, divergence,
        # max-iters) are still caught. The Stage-I boot keeps the stall sensor (cold-start oscillation → Picard).
        coupled_nls = _with_overrides(base_nls; stall_window=0)
        if sol_cfg.pingpong_enabled
            # [coupled cascade] Give the coupled solve the SAME Newton↔Picard fallback as the Stage-I boot:
            # if Newton's line search fails or it diverges → hand to Picard (frozen-advection Oseen
            # linearisation, wider basin) → back to Newton once Picard has cleared `picard_gain_orders`.
            # Converging cells run Newton straight to ftol (Picard stays inert); the `linesearch_failed` cells
            # (steep under-resolved porosity layers) now get a real fallback instead of quitting after one
            # step. The Picard Jacobian uses the same zero-π placeholder (the π value is irrelevant to the
            # local frozen-π tangent — only is_osgs is selected).
            jac_picard_coupled = (x, dx, y) -> build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=_pi_u0, pi_p=_pi_p0, mult_mom=1.0, mult_mass=1.0)
            op_picard_coupled  = FEOperator(res_fn_coupled, jac_picard_coupled, X, Y)
            solver_picard_gain_c = FESolver(_with_overrides(coupled_nls; ftol=sol_cfg.picard_handoff_ftol, mode=:picard, picard_gain_target=sol_cfg.pingpong_picard_gain_orders))
            v_c = _pingpong_cascade!(final_x0, x_backup, op_coupled, op_picard_coupled,
                                     FESolver(coupled_nls), solver_picard_gain_c, OSGS_INNER_POLICY,
                                     diag_cache, iter_count_ref; stage_prefix="C:OSGS", max_swaps=sol_cfg.pingpong_max_swaps)
            success = initial_success && (v_c == :success)
        else
            res_c = safe_fe_solve!(final_x0, FESolver(coupled_nls), op_coupled; backup=x_backup)
            _record_stage!(diag_cache, "C:OSGS:Coupled", res_c)
            if res_c.state == :ok
                iter_count_ref[] += res_c.iterations
                diag_cache["final_residual_norm"] = res_c.residual_norm
                get!(diag_cache, "initial_residual_norm", res_c.initial_residual_norm)
            elseif res_c.state == :max_iters_caught
                iter_count_ref[] += 1
            end
            outcome_c = cascade_step_outcome(res_c, OSGS_INNER_POLICY)
            success = initial_success && (outcome_c == :success || outcome_c == :one_iter_success)
        end

        # Final self-consistent projection (for π export / diagnostics).
        u_h, p_h = final_x0
        pi_u = discrete_l2_projection(inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2), U_proj, V_proj, dΩ, M_u, num_u_fac)
        pi_p = discrete_l2_projection(inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf), P_proj, Q_proj, dΩ, M_p, num_p_fac)

        diag_cache["base_convergence_reached"] = true   # core key (set regardless of verification)
        diag_cache["base_convergence_outer_iter"] = 1
        on_osgs_converged!(verifier, final_x0, diag_cache)   # NoVerification ⇒ no-op; MMS ⇒ oracle eval + rate flag
    end
    diag_cache["pi_u"] = pi_u
    diag_cache["pi_p"] = pi_p
    return success, pi_u, pi_p, coupled_elapsed
end
