# src/run_simulation.jl
using LineSearches

function build_formulation(phys::PhysicalProperties, solver_cfg::SolverConfig)
    rxn_mode = phys.reaction_model
    if rxn_mode == "Constant_Sigma"
        reaction_law = ConstantSigmaLaw(phys.sigma_constant)
    else
        reaction_law = ForchheimerErgunLaw(phys.sigma_linear, phys.sigma_nonlinear)
    end

    reg = SmoothVelocityFloor(phys.u_base_floor_ref, phys.h_floor_weight, phys.epsilon_floor)

    proj = ProjectFullResidual()
    if solver_cfg.experimental_reaction_mode == "standard" && rxn_mode == "Constant_Sigma"
        proj = ProjectResidualWithoutReactionWhenConstantSigma()
    end

    eps_val = phys.eps_val
    eps_floor = phys.eps_floor
    nu = phys.nu
    
    form = PaperGeneralFormulation(DeviatoricSymmetricViscosity(), reaction_law, proj, reg, nu, eps_val, eps_floor)
    return form
end

function build_fe_spaces(model, elem::ElementSpacesConfig, dirichlet_tags, dirichlet_masks, dirichlet_values)
    kv = elem.k_velocity
    kp = elem.k_pressure
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    
    V = TestFESpace(model, refe_u, conformity=:H1, labels=get_face_labeling(model), 
                    dirichlet_tags=dirichlet_tags, dirichlet_masks=dirichlet_masks)
    Q = TestFESpace(model, refe_p, conformity=:H1)
    
    U = TrialFESpace(V, dirichlet_values)
    P = TrialFESpace(Q)
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    return X, Y, kv, kp
end

function run_simulation(config_path::String; 
                        dirichlet_tags=["walls"], 
                        dirichlet_masks=[(true,true)],
                        dirichlet_values=[VectorValue(0.0,0.0)])
    
    cfg = load_config_with_defaults(config_path)
    model = create_mesh(cfg.domain, cfg.numerical_method.mesh)
    
    X, Y, kv, kp = build_fe_spaces(model, cfg.numerical_method.element_spaces, dirichlet_tags, dirichlet_masks, dirichlet_values)
    
    # Bugfix: Build formulation BEFORE querying the typeof(form)
    form = build_formulation(cfg.phys, cfg.numerical_method.solver)
    
    degree = get_quadrature_degree(typeof(form), kv)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    is_tri_val = (cfg.numerical_method.mesh.element_type == "TRI")
    cell_measures = get_cell_measure(Ω)
    h_array = lazy_map(v -> is_tri_val ? sqrt(2.0 * abs(v)) : sqrt(abs(v)), cell_measures)
    h = CellField(h_array, Ω)
    
    alpha_0_val = cfg.domain.alpha_0
    alpha_fn(x) = alpha_0_val
    alpha_cf = CellField(alpha_fn, Ω)
    
    f_x_val = cfg.phys.f_x
    f_y_val = cfg.phys.f_y
    f_fn(x) = VectorValue(f_x_val, f_y_val)
    f_cf = CellField(f_fn, Ω)
    
    c_1, c_2 = get_c1_c2(typeof(form), kv)
    
    # Critical extraction for performance (avoids struct boxing in closure)
    tau_reg_lim = cfg.phys.tau_regularization_limit
    freeze_cusp = cfg.numerical_method.solver.freeze_jacobian_cusp
    
    res(x, y) = build_stabilized_weak_form_residual(x, y, form, dΩ, h, f_cf, alpha_cf, 0.0, nothing, nothing, c_1, c_2, tau_reg_lim)
    # Bugfix: Use ExactNewtonMode() positionally instead of is_picard
    jac(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h, f_cf, alpha_cf, 0.0, nothing, nothing, c_1, c_2, tau_reg_lim, freeze_cusp, ExactNewtonMode())
    
    op = FEOperator(res, jac, X, Y)
    
    nls = SafeNewtonSolver(LUSolver(), cfg.numerical_method.solver)
    solver = FESolver(nls)
    
    println("Solving non-linear system...")
    x_h = solve(solver, op)
    
    u_h, p_h = x_h
    
    export_results(cfg, model, u_h, p_h)
    
    return u_h, p_h, model, Ω, dΩ
end

function run_simulation(op_newton::FEOperator, op_picard::FEOperator, cfg::PorousNSConfig, model, X)
    nls_picard = NLSolver(show_trace=true, method=:newton, iterations=cfg.numerical_method.solver.picard_iterations)
    solver_picard = FESolver(nls_picard)
    println("Solving Picard Initialization...")
    x_picard = solve(solver_picard, op_picard)
    
    nls_newton = SafeNewtonSolver(LUSolver(), cfg.numerical_method.solver)
    solver_newton = FESolver(nls_newton)
    println("Solving Newton-Raphson...")
    solve!(x_picard, solver_newton, op_newton)
    
    u_h, p_h = x_picard
    export_results(cfg, model, u_h, p_h)
    
    return u_h, p_h
end