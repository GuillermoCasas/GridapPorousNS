using Pkg
Pkg.activate("../../..")
using PorousNSSolver

include("../ManufacturedSolutions/run_test.jl")

println("Testing MMS P2P2 vs P2P1")
config_path = "../ManufacturedSolutions/data/small_test_config.json"

# We run it manually overriding equal order
using JSON3
test_dict = JSON3.read(read(config_path, String), Dict{String, Any})

methods = ["ASGS"]
# We simulate N=5 and N=10 for fast check
conv_parts = [5, 10]
results = []
Re = 1.0; Da = 1.0; alpha_0 = 0.4
etype = "TRI"
for (kv, kp) in [(1,1), (2,2), (2,1)]
    for n in conv_parts
        config_dict = Dict(
            "physical_properties" => Dict("nu" => 1.0, "eps_val" => 1e-8, "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
            "domain" => Dict("alpha_0" => alpha_0, "bounding_box" => [0.0, 1.0, 0.0, 1.0], "r_1" => 0.0, "r_2" => 0.5),
            "numerical_method" => Dict(
                "element_spaces" => Dict("k_velocity" => kv, "k_pressure" => kp),
                "mesh" => Dict("element_type" => etype, "partition" => [n, n]),
                "stabilization" => Dict("method" => "ASGS"),
                "solver" => Dict("newton_iterations" => 10, "picard_iterations" => 5, "max_increases" => 5, "xtol" => 1e-6, "ftol" => 1e-6, "linesearch_alpha_min" => 0.1, "armijo_c1" => 1e-4, "divergence_merit_factor" => 10.0, "stagnation_noise_floor" => 1e-12, "freeze_jacobian_cusp" => false)
            )
        )
        config = PorousNSSolver.load_config_from_dict(config_dict)
        model = CartesianDiscreteModel((0,1,0,1), (n, n))
        model = simplexify(model)
        labels = get_face_labeling(model)
        add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])
        refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
        refe_p = ReferenceFE(lagrangian, Float64, kp)
        V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
        Q = TestFESpace(model, refe_p, conformity=:H1)
        
        degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv)
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree + 4)
        h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
        h_cf = CellField(h_array, Ω)
        Y = MultiFieldFESpace([V, Q])
        c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, kv)
        
        alpha_infty = 1.0
        alpha_field = PorousNSSolver.SmoothRadialPorosity(alpha_0, alpha_infty, 0.1, 0.4)
        alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)
        
        form = build_mms_formulation(config, Da, Re, 1.0, 1.0, alpha_infty)
        mms = PorousNSSolver.Paper2DMMS(form, 1.0, alpha_field; L=1.0, alpha_infty=alpha_infty)
        U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)
        u_final = PorousNSSolver.get_u_ex(mms)
        p_final = PorousNSSolver.get_p_ex(mms)
        
        U = TrialFESpace(V, u_final)  
        P = TrialFESpace(Q, p_final)
        X = MultiFieldFESpace([U, P])
        
        f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, 1e-12)
        nls_picard = PorousNSSolver.SafeNewtonSolver(LUSolver(), config.numerical_method.solver.picard_iterations, config.numerical_method.solver.max_increases, config.numerical_method.solver.xtol, config.numerical_method.solver.ftol, config.numerical_method.solver.linesearch_alpha_min, config.numerical_method.solver.armijo_c1, config.numerical_method.solver.divergence_merit_factor, config.numerical_method.solver.stagnation_noise_floor)
        nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), config.numerical_method.solver.newton_iterations, config.numerical_method.solver.max_increases, config.numerical_method.solver.xtol, config.numerical_method.solver.ftol, config.numerical_method.solver.linesearch_alpha_min, config.numerical_method.solver.armijo_c1, config.numerical_method.solver.divergence_merit_factor, config.numerical_method.solver.stagnation_noise_floor)
        solver_picard = FESolver(nls_picard)
        solver_newton = FESolver(nls_newton)
        
        x0 = interpolate_everywhere([u_final, p_final], X)
        
        sys_success, sys_final_x0, sys_iter_count, sys_eval_time = PorousNSSolver.solve_system(
            X, Y, model, dΩ, Ω, h_cf, f_cf, alpha_cf, g_cf, form, 
            solver_picard, solver_newton, 
            x0, c_1, c_2, 
            config.physical_properties, PorousNSSolver.StabilizationConfig(method="ASGS", osgs_iterations=3, osgs_tolerance=1e-8), config.numerical_method.solver
        )
        u_h, p_h = sys_final_x0
        el2_u, el2_p, eh1_u, eh1_p = calculate_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, 1.0, dΩ)
        push!(results, (kv, kp, n, el2_u, el2_p))
    end
end
println("P1P1:")
println("N=5: ", results[1])
println("N=10: ", results[2])
println("RATE: ", log(results[1][4]/results[2][4])/log(10/5))
println("\nP2P2:")
println("N=5: ", results[3])
println("N=10: ", results[4])
println("RATE: ", log(results[3][4]/results[4][4])/log(10/5))
println("\nP2P1:")
println("N=5: ", results[5])
println("N=10: ", results[6])
println("RATE: ", log(results[5][4]/results[6][4])/log(10/5))
