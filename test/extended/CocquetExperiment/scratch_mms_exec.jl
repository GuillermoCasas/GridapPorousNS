using Pkg
Pkg.activate("../../..")
using PorousNSSolver

include("../ManufacturedSolutions/run_test.jl")

println("Testing MMS P2P2 convergence rate explicitly!")
# Construct a clean dictionary for P2P2
config_dict = Dict(
    "physical_properties" => Dict("nu" => 1.0, "eps_val" => 1e-8, "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0, "tau_regularization_limit" => 1e-12),
    "domain" => Dict("alpha_0" => 0.4, "bounding_box" => [0.0, 1.0, 0.0, 1.0], "r_1" => 0.1, "r_2" => 0.4),
    "numerical_method" => Dict(
        "element_spaces" => Dict("k_velocity" => 2, "k_pressure" => 2),
        "mesh" => Dict("element_type" => "QUAD", "partition" => [10, 10]),
        "stabilization" => Dict("method" => "ASGS", "osgs_iterations" => 3, "osgs_tolerance" => 1e-8),
        "solver" => Dict("newton_iterations" => 10, "picard_iterations" => 5, "max_increases" => 5, "xtol" => 1e-8, "ftol" => 1e-8, "linesearch_alpha_min" => 0.1, "armijo_c1" => 1e-4, "divergence_merit_factor" => 10.0, "stagnation_noise_floor" => 1e-12, "freeze_jacobian_cusp" => false)
    )
)

using Gridap
results = []
for n in [10, 20]
    local_cfg = copy(config_dict)
    local_cfg["numerical_method"]["mesh"]["partition"] = [n, n]
    cfg = PorousNSSolver.load_config_from_dict(local_cfg)
    
    model = CartesianDiscreteModel((0,1,0,1), (n, n))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
    refe_p = ReferenceFE(lagrangian, Float64, 2)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
    Q = TestFESpace(model, refe_p, conformity=:H1)
    
    degree = 8
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree + 4)
    h_cf = CellField(lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω)), Ω)
    Y = MultiFieldFESpace([V, Q])
    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, 2)
    
    alpha_infty = 1.0
    alpha_field = PorousNSSolver.SmoothRadialPorosity(0.4, 1.0, 0.1, 0.4)
    alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)
    
    form = build_mms_formulation(cfg, 1.0, 1.0, 1.0, 1.0, 1.0)
    mms = PorousNSSolver.Paper2DMMS(form, 1.0, alpha_field; L=1.0, alpha_infty=1.0)
    u_final = PorousNSSolver.get_u_ex(mms)
    p_final = PorousNSSolver.get_p_ex(mms)
    
    U = TrialFESpace(V, u_final)  
    P = TrialFESpace(Q, p_final)
    X = MultiFieldFESpace([U, P])
    
    f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, 1e-12)
    
    nls_picard = PorousNSSolver.SafeNewtonSolver(LUSolver(), 5, 5, 1e-8, 1e-8, 0.1, 1e-4, 10.0, 1e-12)
    nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), 10, 5, 1e-8, 1e-8, 0.1, 1e-4, 10.0, 1e-12)
    solver_picard = FESolver(nls_picard)
    solver_newton = FESolver(nls_newton)
    
    x0 = interpolate_everywhere([u_final, p_final], X)
    
    sys_success, sys_final_x0 = PorousNSSolver.solve_system(
        X, Y, model, dΩ, Ω, h_cf, f_cf, alpha_cf, g_cf, form, 
        solver_picard, solver_newton, 
        x0, c_1, c_2, 
        cfg.physical_properties, PorousNSSolver.StabilizationConfig(method="ASGS", osgs_iterations=3, osgs_tolerance=1e-8), cfg.numerical_method.solver
    )
    u_h, p_h = sys_final_x0
    el2_u, el2_p, eh1_u, eh1_p = calculate_normalized_errors(u_h, p_h, u_final, p_final, 1.0, 1.0, 1.0, dΩ)
    push!(results, (n, el2_u, el2_p))
end

println("MMS P2P2 Convergence:")
println("N=10: ", results[1])
println("N=20: ", results[2])
rate = log(results[1][2]/results[2][2])/log(20/10)
println("L2 VELOCITY RATE = ", rate)
