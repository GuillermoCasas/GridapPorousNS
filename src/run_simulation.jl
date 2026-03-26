# src/run_simulation.jl

function run_simulation(config_path::String; 
                        dirichlet_tags=["walls"], 
                        dirichlet_masks=[(true,true)],
                        dirichlet_values=[VectorValue(0.0,0.0)])
    
    config = load_config(config_path)
    model = create_mesh(config)
    
    kv = config.phys.k_velocity
    kp = config.phys.k_pressure
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    
    V = TestFESpace(model, refe_u, conformity=:H1, labels=get_face_labeling(model), 
                    dirichlet_tags=dirichlet_tags, dirichlet_masks=dirichlet_masks)
    Q = TestFESpace(model, refe_p, conformity=:H1) # Pressure usually has no Dirichlet BCs (unless specified)
    
    U = TrialFESpace(V, dirichlet_values)
    P = TrialFESpace(Q) # We might need to pin pressure if no outflow BC is present
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    degree = 2 * kv
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    # Calculate element size h
    # Using cell measures (area in 2D) and taking sqrt
    cell_measures = get_cell_measure(Ω)
    h_array = lazy_map(v -> sqrt(abs(v)), cell_measures)
    h = CellField(h_array, Ω)
    
    # Define residual operator
    res(x, y) = weak_form_residual(x, y, config, dΩ, h)
    
    # Assemble non-linear system
    op = FEOperator(res, X, Y)
    
    # Non-linear solver (Newton-Raphson)
    nls = NLSolver(show_trace=true, method=:newton, iterations=10)
    solver = FESolver(nls)
    
    println("Solving non-linear system...")
    x_h = solve(solver, op)
    
    u_h, p_h = x_h
    
    # Export
    export_results(config, model, u_h, p_h)
    
    return u_h, p_h, model, Ω, dΩ
end

# An alternative for when X and Y and op are constructed externally (like Manufactured Solutions)
function run_simulation(op::FEOperator, config::PorousNSConfig, model, X)
    nls = NLSolver(show_trace=true, method=:newton, iterations=10)
    solver = FESolver(nls)
    
    println("Solving non-linear system...")
    x_h = solve(solver, op)
    
    u_h, p_h = x_h
    
    export_results(config, model, u_h, p_h)
    
    return u_h, p_h
end
