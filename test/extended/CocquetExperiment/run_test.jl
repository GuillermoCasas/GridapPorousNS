# test/long/CocquetExperiment/run_test.jl
# ==============================================================================================
# Nature & Intent:
# A basic driver for the Cocquet benchmark setup. Implements a fluid flow passing through a smooth 
# transversal continuous porosity variation $\epsilon(y)$. Used to simply execute a single simulation 
# run on the physics geometry described in Section 4.2 of the referenced continuum mechanics paper.
#
# Mathematical Formulation Alignment:
# Exact mapping of physical boundary and boundary initial fluid definitions. Guarantees 
# mathematical translation of the domain bounds and inflow polynomials to the solver framework.
#
# Associated Files / Functions:
# - `test/extended/CocquetExperiment/run_convergence.jl` (Generalizes this base)
# ==============================================================================================


using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON
using LineSearches

function _build_local_mesh(domain_cfg, mesh_cfg)
    domain = Tuple(domain_cfg.bounding_box)
    partition = Tuple(mesh_cfg.partition)
    if mesh_cfg.element_type == "TRI"
        model = CartesianDiscreteModel(domain, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain, partition)
    end
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [7])
    add_tag_from_tags!(labels, "outlet", [8])
    add_tag_from_tags!(labels, "walls", [1, 2, 3, 4, 5, 6])
    return model
end

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
    
    model = _build_local_mesh(config.domain, config.numerical_method.mesh)
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, config.numerical_method.element_spaces.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, config.numerical_method.element_spaces.k_pressure)
    
    labels = get_face_labeling(model)
    
    # Velocity spaces: Driven inlet, no-slip walls. Outlet is left free.
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["inlet", "walls"])
    U = TrialFESpace(V, [u_in, u_wall])
    
    # Pressure is left completely free (Neumann outlet provides pressure pinning naturally)
    Q = TestFESpace(model, refe_p, conformity=:H1)
    P = TrialFESpace(Q)
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    degree = 2 * config.numerical_method.element_spaces.k_velocity
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h = CellField(h_array, Ω)
    
    # Nodal interpolation degree must rigorously match the velocity field polynomial to preserve superconvergent bounds!
    k_v = config.numerical_method.element_spaces.k_velocity
    refe_alpha = ReferenceFE(lagrangian, Float64, k_v)
    Q_alpha = TestFESpace(model, refe_alpha, conformity=:H1)
    alpha_h = interpolate_everywhere(alpha_func, Q_alpha)
    res(x, y) = PorousNSSolver.weak_form_residual(x, y, config, dΩ, h, nothing, alpha_h)
    jac(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, config, dΩ, h, nothing, alpha_h)
    jac_picard(x, dx, y) = PorousNSSolver.weak_form_jacobian_picard(x, dx, y, config, dΩ, h, nothing, alpha_h)
    
    op_picard = FEOperator(res, jac_picard, X, Y)
    op_newton = FEOperator(res, jac, X, Y)
    
    if config.numerical_method.solver.use_linesearch
        nls_newton = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=config.numerical_method.solver.newton_iterations)
        nls_picard = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=config.numerical_method.solver.picard_iterations)
    else
        nls_newton = NLSolver(show_trace=true, method=:newton, iterations=config.numerical_method.solver.newton_iterations)
        nls_picard = NLSolver(show_trace=true, method=:newton, iterations=config.numerical_method.solver.picard_iterations)
    end
    solver_newton = FESolver(nls_newton)
    solver_picard = FESolver(nls_picard)
    
    if config.numerical_method.solver.picard_iterations > 0
        println("Solving the uncoupled/porous DBF equations (Picard stage, $(config.numerical_method.solver.picard_iterations) iters)...")
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
