# tests/CocquetExperiment/run_test.jl

using PorousNSSolver
using Gridap
using JSON

# Alpha varies: 1.0 in free flow (y >= 0.5), 0.4 in porous (y < 0.5)
# This mimics the Section 4.2 inhomogeneous porous media experiment with a free stream
alpha_func(x) = x[2] >= 0.5 ? 1.0 : 0.4

# Inflow profile
u_in(x) = VectorValue(4.0 * x[2] * (1.0 - x[2]), 0.0)
u_wall(x) = VectorValue(0.0, 0.0)

function run_cocquet()
    println("Running Cocquet Experiment...")
    config_path = joinpath(@__DIR__, "data", "test_config.json")
    config = PorousNSSolver.load_config(config_path)
    
    model = PorousNSSolver.create_mesh(config)
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, config.phys.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, config.phys.k_pressure)
    
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
    
    degree = 2 * config.phys.k_velocity
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h = CellField(h_array, Ω)
    
    # The true formulation residual will receive alpha_func and automatically evaluate it element-wise
    res(x, y) = PorousNSSolver.weak_form_residual(x, y, config, dΩ, h, nothing, alpha_func)
    
    op = FEOperator(res, X, Y)
    
    nls = NLSolver(show_trace=true, method=:newton, iterations=15)
    solver = FESolver(nls)
    
    println("Solving the uncoupled/porous DBF equations...")
    x_h = solve(solver, op)
    u_h, p_h = x_h
    
    # Export fields out to VTK
    PorousNSSolver.export_results(config, model, u_h, p_h)
    println("Cocquet Experiment Complete. Results in $(config.output.directory)")
end

run_cocquet()
