# src/run_simulation.jl
using LineSearches

function build_formulation(config::PorousNSConfig)
    rxn_mode = config.phys.reaction_model
    if rxn_mode == "Constant_Sigma"
        reaction_law = ConstantSigmaLaw(config.phys.sigma_constant)
    else
        reaction_law = ForchheimerErgunLaw(config.phys.sigma_linear, config.phys.sigma_nonlinear)
    end

    reg = SmoothVelocityFloor(config.phys.u_base_floor_ref, config.phys.h_floor_weight, config.phys.epsilon_floor)

    proj = ProjectFullResidual()
    if config.numerical_method.solver.experimental_reaction_mode == "standard" && rxn_mode == "Constant_Sigma"
        proj = ProjectResidualWithoutReactionWhenConstantSigma()
    end

    eps_val = config.phys.eps_val
    eps_floor = config.phys.eps_floor
    nu = config.phys.nu
    
    # We choose DeviatoricSymmetricViscosity as requested for the exact D_Pi S_Pi mapping
    form = PaperGeneralFormulation(DeviatoricSymmetricViscosity(), reaction_law, proj, reg, nu, eps_val, eps_floor)
    return form
end

function run_simulation(config_path::String; 
                        dirichlet_tags=["walls"], 
                        dirichlet_masks=[(true,true)],
                        dirichlet_values=[VectorValue(0.0,0.0)])
    
    config = load_config(config_path)
    model = create_mesh(config)
    
    kv = config.numerical_method.element_spaces.k_velocity
    kp = config.numerical_method.element_spaces.k_pressure
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    
    V = TestFESpace(model, refe_u, conformity=:H1, labels=get_face_labeling(model), 
                    dirichlet_tags=dirichlet_tags, dirichlet_masks=dirichlet_masks)
    Q = TestFESpace(model, refe_p, conformity=:H1)
    
    U = TrialFESpace(V, dirichlet_values)
    P = TrialFESpace(Q)
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    degree = get_quadrature_degree(typeof(form), kv)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    is_tri = config.numerical_method.mesh.element_type == "TRI"
    cell_measures = get_cell_measure(Ω)
    h_array = lazy_map(v -> is_tri ? sqrt(2.0 * abs(v)) : sqrt(abs(v)), cell_measures)
    h = CellField(h_array, Ω)
    
    alpha_fn(x) = config.domain.alpha_0
    alpha_cf = CellField(alpha_fn, Ω)
    f_fn(x) = VectorValue(config.phys.f_x, config.phys.f_y)
    f_cf = CellField(f_fn, Ω)
    
    form = build_formulation(config)
    
    c_1 = 4.0 * kv^4
    c_2 = 2.0 * kv^2
    tau_reg_lim = config.phys.tau_regularization_limit
    
    res(x, y) = build_stabilized_weak_form_residual(x, y, form, dΩ, h, f_cf, alpha_cf, 0.0, nothing, nothing, c_1, c_2, tau_reg_lim)
    jac(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h, f_cf, alpha_cf, 0.0, nothing, nothing, c_1, c_2, tau_reg_lim, config.numerical_method.solver.freeze_jacobian_cusp; is_picard=false)
    
    op = FEOperator(res, jac, X, Y)
    
    nls = SafeNewtonSolver(LUSolver(), config.numerical_method.solver.newton_iterations, config.numerical_method.solver.max_increases, config.numerical_method.solver.xtol, config.numerical_method.solver.stagnation_tol, config.numerical_method.solver.ftol, config.numerical_method.solver.linesearch_alpha_min, config.numerical_method.solver.armijo_c1)
    solver = FESolver(nls)
    
    println("Solving non-linear system...")
    x_h = solve(solver, op)
    
    u_h, p_h = x_h
    
    export_results(config, model, u_h, p_h)
    
    return u_h, p_h, model, Ω, dΩ
end

function run_simulation(op_newton::FEOperator, op_picard::FEOperator, config::PorousNSConfig, model, X)
    nls_picard = NLSolver(show_trace=true, method=:newton, iterations=config.numerical_method.solver.picard_iterations)
    solver_picard = FESolver(nls_picard)
    println("Solving Picard Initialization...")
    x_picard = solve(solver_picard, op_picard)
    
    nls_newton = SafeNewtonSolver(LUSolver(), config.numerical_method.solver.newton_iterations, config.numerical_method.solver.max_increases, config.numerical_method.solver.xtol, config.numerical_method.solver.stagnation_tol, config.numerical_method.solver.ftol, config.numerical_method.solver.linesearch_alpha_min, config.numerical_method.solver.armijo_c1)
    solver_newton = FESolver(nls_newton)
    println("Solving Newton-Raphson...")
    solve!(x_picard, solver_newton, op_newton)
    
    u_h, p_h = x_picard
    export_results(config, model, u_h, p_h)
    
    return u_h, p_h
end