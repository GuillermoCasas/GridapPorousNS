# test/extended/CocquetExperimentLiteralPicard/literal_picard_driver.jl
# ==============================================================================================
# [H-B diagnostic] Pure-Galerkin (UNSTABILIZED) execution driver that runs ONLY the Picard
# linearization at a fixed it_max cap — the exact protocol of Cocquet et al. (2020) page 30
# (Eq. 41 plus the per-paragraph it_max stopping rule).
#
# Differs from CocquetExperiment/galerkin_driver.jl::execute_solver_galerkin in exactly one way:
# the orchestration "Exact Newton → Picard fallback → Newton polish" is REPLACED by a single
# Picard solve with picard_iterations as a hard cap. We accept BOTH :ok (Picard converged within
# the cap) and :max_iters_caught (Picard hit the it_max ceiling) as valid termination — in both
# cases x0 holds the iterate-at-it_max which is exactly what Cocquet's protocol records.
#
# The same it_max ceiling is enforced on BOTH the N=200 reference run AND the coarse N<=100
# runs by run_convergence.jl, so any iteration-truncation bias is shared symmetrically and the
# reported errors are apples-to-apples (just as Cocquet's are).
#
# Lives outside src/ for the same reason CocquetExperiment/galerkin_driver.jl does: validation
# tooling, not part of the production solver. No src/ files were changed for H-B.
# ==============================================================================================

function execute_solver_galerkin_literal_picard(model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, config)
    mult_mom = 0.0
    mult_mass = 0.0

    form = PorousNSSolver.build_formulation(config.physical_properties, config.numerical_method)
    f_cf = VectorValue(config.physical_properties.f_x, config.physical_properties.f_y)
    g_cf = 0.0

    k = config.numerical_method.element_spaces.k_velocity
    c_1, c_2 = PorousNSSolver.get_c1_c2(typeof(form), k)

    ls = LUSolver()
    div_fac = config.numerical_method.solver.divergence_merit_factor
    n_floor = config.numerical_method.solver.stagnation_noise_floor
    ar_c1 = config.numerical_method.solver.armijo_c1
    ls_alpha = config.numerical_method.solver.linesearch_alpha_min
    xtol = config.numerical_method.solver.xtol
    ftol = config.numerical_method.solver.ftol
    max_inc = config.numerical_method.solver.max_increases
    ls_contract = config.numerical_method.solver.linesearch_contraction_factor
    max_ls = config.numerical_method.solver.max_linesearch_iterations

    # The Cocquet it_max cap: picard_iterations from config (set to 10 by the literal-Picard config).
    # newton_iterations is unused — required >= 1 by schema validation but never constructed here.
    it_max = config.numerical_method.solver.picard_iterations
    nls_picard = PorousNSSolver.SafeNewtonSolver(ls, it_max, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor, max_ls, ls_contract; mode=:picard)
    solver_picard = FESolver(nls_picard)

    x0 = FEFunction(X, zeros(num_free_dofs(X)))

    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)

    setup = PorousNSSolver.FETopology(X, Y, model, Triangulation(model), dΩ, V_free, Q_free, h_cf, f_cf, alpha_h, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)

    res_fn(x, y) = PorousNSSolver.build_stabilized_weak_form_residual(x, y, setup, formulation, config.physical_properties; pi_u=nothing, pi_p=nothing, mult_mom=mult_mom, mult_mass=mult_mass)
    jac_picard(x, dx, y) = PorousNSSolver.build_picard_jacobian(x, dx, y, setup, formulation, config.physical_properties; pi_u=nothing, pi_p=nothing, mult_mom=mult_mom, mult_mass=mult_mass)

    op_picard = FEOperator(res_fn, jac_picard, X, Y)

    x0_backup = copy(get_free_dof_values(x0))
    iter_count = 0

    eval_time = @elapsed begin
        res_p = PorousNSSolver.safe_fe_solve!(x0, solver_picard, op_picard; backup=x0_backup)
        if res_p.state == :ok
            iter_count = res_p.iterations
            println("      [LiteralPicard] converged in $iter_count iterations (state=:ok, residual=$(res_p.residual_norm), reason=$(res_p.stop_reason))")
        elseif res_p.state == :max_iters_caught
            iter_count = it_max
            println("      [LiteralPicard] HIT it_max=$it_max cap (state=:max_iters_caught) — iterate-at-cap recorded as Cocquet's protocol prescribes")
        else
            error("LiteralPicard solver failed unexpectedly (state $(res_p.state)).")
        end
    end

    return x0, eval_time, iter_count
end
