using Pkg
Pkg.activate(".")
using PorousNSSolver
using Gridap

include("test/extended/CocquetExperiment/run_convergence.jl")

function run_quick_p2()
    base_config_dict = Dict(
        "domain" => Dict("alpha_infty" => 1.0, "bounding_box" => [0.0, 2.0, 0.0, 1.0], "l_char" => 1.0, "tau_y" => 1.0),
        "physical_properties" => Dict("eps_val" => 1e-8, "nu" => 0.002, "rho" => 1.0, "sigma_linear" => 0.3, "sigma_nonlinear" => 1.75, "reaction_model" => "Forchheimer_Ergun", "tau_regularization_limit" => 1e-12),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity" => 2, "k_pressure" => 2),
            "solver" => Dict("armijo_c1" => 0.0001, "divergence_merit_factor" => 10.0, "freeze_jacobian_cusp" => false, "ftol" => 1e-6, "linesearch_alpha_min" => 0.1, "max_increases" => 3, "newton_iterations" => 15, "picard_iterations" => 5, "stagnation_noise_floor" => 1e-10, "xtol" => 1e-6),
            "stabilization" => Dict("method" => "ASGS", "osgs_iterations" => 3, "osgs_tolerance" => 1.0e-8)
        ),
        "outlet_truncation_delta" => 0.0,
        "interpolate_solution_to_coarser_meshes" => false
    )

    Re = 500.0
    c_in = 1.0

    for N in [10]
        println("\n>>> Solver N = $N")
        model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, local_config = build_solver(N, base_config_dict, Re, c_in)

        solver_picard, solver_newton, c_1, c_2 = build_nonlinear_solver(local_config, 2)
        println("c_1 = $c_1, c_2 = $c_2")
        
        form_dict = Dict("ASGS" => PorousNSSolver.PaperGeneralFormulation())
        form = form_dict[local_config.numerical_method.stabilization.method]
        
        ru_h = interpolate_everywhere(VectorValue(0.0, 0.0), X.spaces[1])
        rp_h = interpolate_everywhere(0.0, X.spaces[2])
        
        sys_success, sys_final_x0, sys_iter_count, sys_eval_time = PorousNSSolver.solve_system(
            X, Y, model, dΩ, Triangulation(model), h_cf, VectorValue(0.0, 0.0), alpha_h, 0.0, form,
            solver_picard, solver_newton,
            (ru_h, rp_h), c_1, c_2,
            local_config.physical_properties, local_config.numerical_method.stabilization, local_config.numerical_method.solver
        )
        println("Success? ", sys_success, " in ", sys_iter_count, " iterations.")
    end
end

run_quick_p2()
