# test/extended/CocquetExperiment/galerkin_driver.jl
# ==============================================================================================
# Pure-Galerkin (UNSTABILIZED) execution driver — the exact discrete method of Cocquet et al.
# (inf-sup-stable Taylor–Hood, NO VMS stabilization).
#
# This lives OUTSIDE src/ on purpose: it is validation tooling, not part of the production solver.
# It does NOT duplicate the formulation — it reuses the same public weak-form builders the production
# solver uses (`build_stabilized_weak_form_residual`, `build_picard_jacobian`,
# `build_stabilized_weak_form_jacobian`), called with `mult_mom = mult_mass = 0`. Those multipliers
# scale the VMS stabilization terms, so 0 leaves exactly the bare Galerkin Darcy–Brinkman–Forchheimer
# system (conv + viscous + pressure + reaction + mass − source), with the εp pressure penalty intact.
# It therefore bypasses `solve_system`'s ASGS/OSGS machinery entirely (there is no τ-stabilization and
# no orthogonal-projection stage in the Cocquet method).
#
# `execute_solver_galerkin` mirrors the signature/return of `execute_solver` in run_convergence.jl
# (returns `(final_x0, eval_time, iter_count)`) so the convergence driver can dispatch on the method
# label with no other changes. Reuses `build_solver` from run_convergence.jl (shared, not duplicated).
#
# Intended for inf-sup-stable pairs (Taylor–Hood P2/P1). Equal-order Galerkin (P1/P1, P2/P2) would be
# inf-sup unstable and is not used here.
# ==============================================================================================

function execute_solver_galerkin(model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, config)
    # mult_mom = mult_mass = 0 ⇒ pure Galerkin (no VMS stabilization). This is the mathematical
    # definition of the Cocquet (Taylor–Hood Galerkin) method, not a tunable parameter.
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
    freeze_cusp = config.numerical_method.solver.freeze_jacobian_cusp

    nls_picard = PorousNSSolver.SafeNewtonSolver(ls, config.numerical_method.solver.picard_iterations, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor, max_ls, ls_contract; mode=:picard)
    nls_newton = PorousNSSolver.SafeNewtonSolver(ls, config.numerical_method.solver.newton_iterations, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor, max_ls, ls_contract)
    solver_picard = FESolver(nls_picard)
    solver_newton = FESolver(nls_newton)

    x0 = FEFunction(X, zeros(num_free_dofs(X)))

    # Unconstrained spaces are unused by the Galerkin assembly (no projection), but FETopology needs them.
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)

    setup = PorousNSSolver.FETopology(X, Y, model, Triangulation(model), dΩ, V_free, Q_free, h_cf, f_cf, alpha_h, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)

    res_fn(x, y) = PorousNSSolver.build_stabilized_weak_form_residual(x, y, setup, formulation, config.physical_properties; pi_u=nothing, pi_p=nothing, mult_mom=mult_mom, mult_mass=mult_mass)
    jac_picard(x, dx, y) = PorousNSSolver.build_picard_jacobian(x, dx, y, setup, formulation, config.physical_properties; pi_u=nothing, pi_p=nothing, mult_mom=mult_mom, mult_mass=mult_mass)
    jac_newton(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, config.physical_properties, freeze_cusp, PorousNSSolver.ExactNewtonMode(); pi_u=nothing, pi_p=nothing, mult_mom=mult_mom, mult_mass=mult_mass)

    op_picard = FEOperator(res_fn, jac_picard, X, Y)
    op_newton = FEOperator(res_fn, jac_newton, X, Y)

    x0_backup = copy(get_free_dof_values(x0))
    iter_count = 0

    eval_time = @elapsed begin
        # Exact Newton first; Picard globalization fallback, then a final Newton polish.
        res = PorousNSSolver.safe_fe_solve!(x0, solver_newton, op_newton; backup=x0_backup)
        if res.state == :ok
            iter_count = res.iterations
        else
            println("      [Galerkin] Newton state $(res.state); falling back to Picard...")
            get_free_dof_values(x0) .= x0_backup
            res_p = PorousNSSolver.safe_fe_solve!(x0, solver_picard, op_picard; backup=x0_backup)
            res_p.state == :ok || error("Galerkin Picard fallback failed (state $(res_p.state)).")
            iter_count += res_p.iterations
            res_n = PorousNSSolver.safe_fe_solve!(x0, solver_newton, op_newton; backup=x0_backup)
            res_n.state == :ok && (iter_count += res_n.iterations)
        end
    end

    return x0, eval_time, iter_count
end
