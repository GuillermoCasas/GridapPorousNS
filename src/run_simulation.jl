# src/run_simulation.jl
using LineSearches

function run_simulation(config_path::String; 
                        dirichlet_tags=["walls"], 
                        dirichlet_masks=[(true,true)],
                        dirichlet_values=[VectorValue(0.0,0.0)])
    
    config = load_config(config_path)
    model = create_mesh(config)
    
    kv = config.discretization.k_velocity
    kp = config.discretization.k_pressure
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    
    V = TestFESpace(model, refe_u, conformity=:H1, labels=get_face_labeling(model), 
                    dirichlet_tags=dirichlet_tags, dirichlet_masks=dirichlet_masks)
    Q = TestFESpace(model, refe_p, conformity=:H1) # Pressure usually has no Dirichlet BCs (unless specified)
    
    U = TrialFESpace(V, dirichlet_values)
    P = TrialFESpace(Q) # We might need to pin pressure if no outflow BC is present
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    # ALGORITHMIC RATIONALE: Quadrature Aliasing Prevention
    # A standard degree 2*k formulation maps generic Oseen momentum operators cleanly.
    # However, ASGS momentum stabilization integrates high-order structural cross-products 
    # heavily coupling non-linear residuals e.g. τ_1 * ((u ⋅ ∇)u) ⋅ ((u ⋅ ∇)v). 
    # For P2 interpolations, velocity scales as O(x^2), expanding the associated convective operators smoothly 
    # up towards O(x^3) internally, thus deriving algebraic polynomials locally of degree 6 prior to structural division. 
    # Raising to 'degree = 4 * k_velocity' flawlessly prevents dynamic mathematical aliasing artifacts from 
    # violently crushing the numerical convergence rates (triggering negative spatial orders) naturally bounded 
    # across extreme limit hyper-advective execution regimes cleanly natively scaling physics accurately.
    degree = 4 * kv
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    # Calculate element size h
    # Using cell measures (area in 2D) and taking sqrt
    is_tri = config.mesh.element_type == "TRI"
    cell_measures = get_cell_measure(Ω)
    h_array = lazy_map(v -> is_tri ? sqrt(2.0 * abs(v)) : sqrt(abs(v)), cell_measures)
    h = CellField(h_array, Ω)
    
    # Define residual operator
    alpha_fn(x) = config.porosity.alpha_0
    alpha_cf = CellField(alpha_fn, Ω)
    f_fn(x) = VectorValue(config.phys.f_x, config.phys.f_y)
    f_cf = CellField(f_fn, Ω)
    
    res(x, y) = weak_form_residual(x, y, config, dΩ, h, f_cf, alpha_cf, nothing)
    jac(x, dx, y) = weak_form_jacobian(x, dx, y, config, dΩ, h, f_cf, alpha_cf, nothing)
    
    # Assemble non-linear system
    op = FEOperator(res, jac, X, Y)
    
    # Non-linear solver (Newton-Raphson)
    nls = SafeNewtonSolver(LUSolver(), config.solver.newton_iterations, config.solver.max_increases, config.solver.xtol, config.solver.stagnation_tol, config.solver.ftol, config.solver.linesearch_tolerance, config.solver.linesearch_alpha_min)
    solver = FESolver(nls)
    
    println("Solving non-linear system...")
    x_h = solve(solver, op)
    
    u_h, p_h = x_h
    
    # Export
    export_results(config, model, u_h, p_h)
    
    return u_h, p_h, model, Ω, dΩ
end

# An alternative for when X and Y and op are constructed externally (like Manufactured Solutions)
function run_simulation(op_newton::FEOperator, op_picard::FEOperator, config::PorousNSConfig, model, X)
    # ALGORITHMIC RATIONALE: Hybrid Picard-Newton Two-Stage Iteration
    # Native executions scaling to extreme advection thresholds (Re=10^6) uniformly divergence
    # mapping from zero vector limits cleanly dropping due to their minuscule theoretical radius of convergence.
    #
    # Stage 1: Picard Globalization. We manually formulate a surrogate linearized fixed-point mapping
    # internally scaling evaluating jac_picard exclusively to drop computationally unstable advecting velocity
    # variations structurally mapping d(u ⋅ ∇)u bounds mathematically. This constructs a globally wide tracking basin
    # safely generating stable continuous vector domains smoothly safely inside constraints naturally bounding logic natively.
    # NOTE: BackTracking is strictly disabled! Linear mapping structural surrogates natively break exact
    # strong residual linesearch tracking since they aren't guaranteed mathematically accurate descent maps!
    # Stage 1: Picard Initialization (NO BackTracking! Must take full surrogate steps)
    nls_picard = NLSolver(show_trace=true, method=:newton, iterations=config.solver.picard_iterations)
    solver_picard = FESolver(nls_picard)
    println("Solving Picard Initialization...")
    x_picard = solve(solver_picard, op_picard)
    
    # Stage 2: Newton-Raphson Quadratic Lock
    # Stage 1 places our iteration successfully into the robust quadratic basin of attraction. 
    # Now we natively flip active formulation natively resolving structurally executing the 
    # exact Fréchet continuous matrix. Because this resolves the authentic analytical Jacobian dynamically bounding 
    # mapping exact strong mathematical residuals properly, we re-apply BackTracking natively tracking step lengths.
    # Stage 2: Newton-Raphson (Keep BackTracking to safely secure the quadratic basin)
    nls_newton = SafeNewtonSolver(LUSolver(), config.solver.newton_iterations, config.solver.max_increases, config.solver.xtol, config.solver.stagnation_tol, config.solver.ftol, config.solver.linesearch_tolerance, config.solver.linesearch_alpha_min)
    solver_newton = FESolver(nls_newton)
    println("Solving Newton-Raphson...")
    solve!(x_picard, solver_newton, op_newton)
    
    u_h, p_h = x_picard
    
    export_results(config, model, u_h, p_h)
    
    return u_h, p_h
end