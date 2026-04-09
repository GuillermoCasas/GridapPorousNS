using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Gridap
using PorousNSSolver
using JSON3

base_config_path = joinpath(@__DIR__, "..", "extended", "CocquetExperiment", "data", "test_config.json")
base_config_dict = JSON3.read(read(base_config_path, String), Dict{String, Any})

# Run a smaller resolution that still clearly shows the 1.8 slope limit, say N=100.
N_ref = 100  
println("Solving N_ref = $N_ref...")
base_config_dict["numerical_method"]["mesh"]["partition"] = [2*N_ref, N_ref]
base_config_dict["numerical_method"]["element_spaces"]["k_velocity"] = 2
base_config_dict["numerical_method"]["element_spaces"]["k_pressure"] = 2

cfg = PorousNSSolver.load_config_from_dict(base_config_dict)
model = PorousNSSolver._build_default_mesh(cfg.domain, cfg.numerical_method.mesh)

refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
refe_p = ReferenceFE(lagrangian, Float64, 2)
labels = get_face_labeling(model)
u_in(x) = VectorValue(0.5 * x[2] * (1.0 - x[2]), 0.0)
u_wall(x) = VectorValue(0.0, 0.0)

V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["inlet", "walls"])
U = TrialFESpace(V, [u_in, u_wall])
Q = TestFESpace(model, refe_p, conformity=:H1)
P = TrialFESpace(Q)
Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

degree = 4
Ω = Triangulation(model)
dΩ_ref = Measure(Ω, degree)
h_cf = CellField(lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω)), Ω)

alpha_func(x) = 0.45 + 0.55 * exp(x[2] - 1.0)
alpha_h = CellField(alpha_func, Ω)

form = PorousNSSolver.build_formulation(cfg.physical_properties, cfg.numerical_method.solver)
solver_picard = FESolver(PorousNSSolver.SafeNewtonSolver(LUSolver(), 30, 3, 1e-8, 1e-8, 1e-3, 1e-4, 1e6, 1e-12))
solver_newton = FESolver(PorousNSSolver.SafeNewtonSolver(LUSolver(), 10, 3, 1e-8, 1e-8, 1e-3, 1e-4, 1e6, 1e-12))

x0 = FEFunction(X, zeros(num_free_dofs(X)))
success, x_h, iters, t = PorousNSSolver.solve_system(X, Y, model, dΩ_ref, Ω, h_cf, VectorValue(0.0,0.0), alpha_h, 0.0, form, solver_picard, solver_newton, x0, 1.0, 1.0, cfg.physical_properties, PorousNSSolver.StabilizationConfig(), cfg.numerical_method.solver)

println("Solver returned Success: $success in $iters iterations.")

u_ref, p_ref = x_h
iu_ref = Gridap.FESpaces.Interpolable(u_ref)

println("Interpolating and measuring...")
for N in [25, 50]
    model_h = PorousNSSolver._build_default_mesh(cfg.domain, Dict("partition"=>[2*N, N], "element_type"=>"TRI"))
    # Wait: U_h needs dirichlet labels natively
    labels_h = get_face_labeling(model_h)
    V_h = TestFESpace(model_h, refe_u, conformity=:H1, labels=labels_h, dirichlet_tags=["inlet", "walls"])
    U_h = TrialFESpace(V_h, [u_in, u_wall])
    
    u_h = interpolate(iu_ref, U_h)
    res = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h, Measure(Triangulation(model_h),4), dΩ_ref)
    println("N=$N, L2 = $(res[3])")
end

