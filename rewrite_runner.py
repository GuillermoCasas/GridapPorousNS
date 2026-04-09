import re

with open('test/extended/CocquetExperiment/run_convergence.jl', 'r') as f:
    text = f.read()

# 1. Replace degree calculation
text = text.replace(
    '    degree = 2 * local_config.numerical_method.element_spaces.k_velocity + 4',
    '    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, local_config.numerical_method.element_spaces.k_velocity)'
)

# 2. Add build_cocquet_formulation below u_wall
form_code = """u_wall(x) = VectorValue(0.0, 0.0)

function build_cocquet_formulation(config)
    rxn = PorousNSSolver.ForchheimerErgunLaw(config.physical_properties.sigma_linear, config.physical_properties.sigma_nonlinear)
    reg = PorousNSSolver.SmoothVelocityFloor(config.physical_properties.u_base_floor_ref, config.physical_properties.h_floor_weight, config.physical_properties.epsilon_floor)
    proj = PorousNSSolver.ProjectFullResidual()
    return PorousNSSolver.PaperGeneralFormulation(PorousNSSolver.DeviatoricSymmetricViscosity(), rxn, proj, reg, config.physical_properties.nu, config.physical_properties.eps_val, config.physical_properties.eps_floor)
end"""
text = text.replace('u_wall(x) = VectorValue(0.0, 0.0)', form_code)

# 3. Replace execute_solver with a wrapper that delegates to solve_system
exec_repl = """function execute_solver(model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, config)
    form = build_cocquet_formulation(config)
    f_cf = VectorValue(config.physical_properties.f_x, config.physical_properties.f_y)
    g_cf = 0.0
    
    k = config.numerical_method.element_spaces.k_velocity
    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, k)
    
    ls = LUSolver()
    div_fac = config.numerical_method.solver.divergence_merit_factor
    n_floor = config.numerical_method.solver.stagnation_noise_floor
    ar_c1 = config.numerical_method.solver.armijo_c1
    ls_alpha = config.numerical_method.solver.linesearch_alpha_min
    xtol = config.numerical_method.solver.xtol
    ftol = config.numerical_method.solver.ftol
    max_inc = config.numerical_method.solver.max_increases
    
    nls_picard = PorousNSSolver.SafeNewtonSolver(ls, config.numerical_method.solver.picard_iterations, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor)
    nls_newton = PorousNSSolver.SafeNewtonSolver(ls, config.numerical_method.solver.newton_iterations, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor)
    
    solver_picard = FESolver(nls_picard)
    solver_newton = FESolver(nls_newton)
    
    x0 = FEFunction(X, zeros(num_free_dofs(X)))
    
    stab_cfg = PorousNSSolver.StabilizationConfig(
        method=config.numerical_method.stabilization.method,
        osgs_iterations=config.numerical_method.stabilization.osgs_iterations,
        osgs_tolerance=config.numerical_method.stabilization.osgs_tolerance
    )
    
    success, final_x0, iter_count, eval_time = PorousNSSolver.solve_system(
        X, Y, model, dΩ, Ω, h_cf, f_cf, alpha_h, g_cf, form,
        solver_picard, solver_newton,
        x0, c_1, c_2,
        config.physical_properties, stab_cfg, config.numerical_method.solver
    )
    
    return final_x0, eval_time, iter_count
end"""

text = re.sub(r'function execute_solver\(model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, config\).*?end\n', exec_repl + '\n', text, flags=re.DOTALL)

with open('test/extended/CocquetExperiment/run_convergence.jl', 'w') as f:
    f.write(text)
