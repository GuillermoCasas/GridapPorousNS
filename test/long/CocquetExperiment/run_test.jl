# test/long/CocquetExperiment/run_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON
using LineSearches

# Section 4.2 Smooth Porosity Field
# ε(y) = 0.45 + 0.55 * exp(y - 1.0)
alpha_func(x) = 0.45 + 0.55 * exp(x[2] - 1.0)
Gridap.∇(::typeof(alpha_func)) = x -> VectorValue(0.0, 0.55 * exp(x[2] - 1.0))

# Inflow profile for (Re=500, c_in=0.5)
c_in = 0.5
u_in(x) = VectorValue(c_in * x[2] * (1.0 - x[2]), 0.0)
u_wall(x) = VectorValue(0.0, 0.0)

function run_cocquet()
    println("Running Cocquet Experiment...")
    config_path = joinpath(@__DIR__, "data", "test_config.json")
    config = PorousNSSolver.load_config(config_path)
    
    model = PorousNSSolver.create_mesh(config)
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, config.discretization.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, config.discretization.k_pressure)
    
    labels = get_face_labeling(model)
    
    # Velocity spaces: Driven inlet, no-slip walls. Outlet is left free.
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["inlet", "walls"])
    U = TrialFESpace(V, [u_in, u_wall])
    
    # Pressure spaces: fix p at outlet to 0 to prevent nullspace
    # In standard Navier-Stokes, this implies an outflow boundary condition when velocity is free.
    Q = TestFESpace(model, refe_p, conformity=:H1, labels=labels, dirichlet_tags=["outlet"])
    P = TrialFESpace(Q, x -> 0.0)
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    degree = 2 * config.discretization.k_velocity
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h = CellField(h_array, Ω)
    
    # Interpolate alpha_func to a boundary-unconstrained space matching Q so it avoids zeroing out the outlet Dirichlet nodes
    Q_alpha = TestFESpace(model, refe_p, conformity=:H1)
    alpha_h = interpolate_everywhere(alpha_func, Q_alpha)
    res(x, y) = PorousNSSolver.weak_form_residual(x, y, config, dΩ, h, nothing, alpha_h)
    jac(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, config, dΩ, h, nothing, alpha_h)
    jac_picard(x, dx, y) = PorousNSSolver.weak_form_jacobian_picard(x, dx, y, config, dΩ, h, nothing, alpha_h)
    
    op_picard = FEOperator(res, jac_picard, X, Y)
    op_newton = FEOperator(res, jac, X, Y)
    
    if config.solver.use_linesearch
        nls_newton = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=config.solver.newton_iterations)
        nls_picard = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=config.solver.picard_iterations)
    else
        nls_newton = NLSolver(show_trace=true, method=:newton, iterations=config.solver.newton_iterations)
        nls_picard = NLSolver(show_trace=true, method=:newton, iterations=config.solver.picard_iterations)
    end
    solver_newton = FESolver(nls_newton)
    solver_picard = FESolver(nls_picard)
    
    if config.solver.picard_iterations > 0
        println("Solving the uncoupled/porous DBF equations (Picard stage, $(config.solver.picard_iterations) iters)...")
        x_h = solve(solver_picard, op_picard)
        println("Solving the uncoupled/porous DBF equations (Newton stage)...")
        solve!(x_h, solver_newton, op_newton)
    else
        println("Solving the uncoupled/porous DBF equations (Newton stage directly)...")
        x_h = solve(solver_newton, op_newton)
    end
    u_h, p_h = x_h
    
    # Export fields out to VTK
    PorousNSSolver.export_results(config, model, u_h, p_h)
    println("Cocquet Experiment Complete. Results in $(config.output.directory)")
end

run_cocquet()
