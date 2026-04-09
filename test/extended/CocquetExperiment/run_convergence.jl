# test/long/CocquetExperiment/run_convergence.jl
# ==============================================================================================
# Nature & Intent:
# The definitive physical benchmark test for solver exactness on non-synthetic data. Unlike MMS, 
# there is no analytical "exact" polynomial solution, so convergence rates $O(h^{k+1})$ are extracted 
# by using the highest resolution grid as a surrogate mathematically true baseline.
# Exact grid prolongation maps coarse fields to fine mesh measure spaces $d\Omega_{ref}$ to eliminate
# cross-grid interpolation geometry noise and precisely extract the algebraic convergence slope.
#
# Mathematical Formulation Alignment:
# Proves that the VMS scheme yields dimensionally optimal $L_2$ and semi-$H_1$ continuum 
# approximation bounds in the presence of physical varying media (non-constant coefficients).
# It uses an explicit exponential porosity profile ε(y) = 0.45 + 0.55 * exp(y - 1.0) and validates
# structural bounds via orthogonal projections dynamically evaluated against fine-domain integrals.
#
# Associated Data & Endpoints:
# - `data/test_config.json`: Dictates entirely mathematical scales, topological boundaries, `Re`, `c_in`, and interpolation ranges.
# - Outputs exact HDF5 convergence rates for Python plotting `plot_convergence.py`.
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON3
using HDF5
using LinearAlgebra


# Section 4.2 Smooth Porosity Field
# ε(y) = 0.45 + 0.55 * exp(y - 1.0)
alpha_func(x) = 0.45 + 0.55 * exp(x[2] - 1.0)
Gridap.∇(::typeof(alpha_func)) = x -> VectorValue(0.0, 0.55 * exp(x[2] - 1.0))

function get_inflow_profile(Re::Float64, c_in::Float64)
    return x -> VectorValue(c_in * x[2] * (1.0 - x[2]), 0.0)
end
u_wall(x) = VectorValue(0.0, 0.0)


function build_solver(N::Int, config_dict, Re::Float64, c_in::Float64)
    # The paper uses domain [0, 2]x[0, 1] mapped up to N nodes on bounds
    # For a Cartesian grid, a grid of 2N x N elements maintains square shape (h = 1/N).
    # Since the paper specifies parameter N, we use partition = [2*N, N].
    local_config_dict = deepcopy(config_dict)
    if !haskey(local_config_dict["numerical_method"], "mesh")
        local_config_dict["numerical_method"]["mesh"] = Dict()
    end
    local_config_dict["numerical_method"]["mesh"]["partition"] = [2*N, N]
    local_config = PorousNSSolver.load_config_from_dict(local_config_dict)
    

    u_in = get_inflow_profile(Re, c_in)
    
    model = PorousNSSolver._build_default_mesh(local_config.domain, local_config.numerical_method.mesh)
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, local_config.numerical_method.element_spaces.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, local_config.numerical_method.element_spaces.k_pressure)
    
    labels = get_face_labeling(model)
    
    # Inlet [7], Outlet [8] left traction-free (Neumann), Walls [1,2,3,4,5,6] (Dirichlet 0)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["inlet", "walls"])
    U = TrialFESpace(V, [u_in, u_wall])
    
    # Pressure is left completely free (Neumann outlet provides pressure pinning naturally)
    Q = TestFESpace(model, refe_p, conformity=:H1)
    P = TrialFESpace(Q)
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, local_config.numerical_method.element_spaces.k_velocity)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h_cf = CellField(h_array, Ω)
    
    # Emulate Cocquet et al. exactly by nodally interpolating alpha on a linear (P1) element regardless of solver k
    refe_alpha = ReferenceFE(lagrangian, Float64, 1)
    V_alpha = TestFESpace(model, refe_alpha, conformity=:H1)
    alpha_h = interpolate(alpha_func, V_alpha)
    
    # Return raw geometric constructs instead of operators to enable explicit OSGS projections dynamically
    return model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, local_config
end

function execute_solver(model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, config)
    form = PorousNSSolver.build_formulation(config.physical_properties, config.numerical_method.solver)
    f_cf = VectorValue(config.physical_properties.f_x, config.physical_properties.f_y)
    g_cf = 0.0
    
    k = config.numerical_method.element_spaces.k_velocity
    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, k)
    
    ls = LUSolver()
    div_fac = config.numerical_method.solver.divergence_merit_factor
    n_floor = config.numerical_method.solver.stagnation_noise_floor
    ar_c1 = config.numerical_method.solver.armijo_c1
    ls_alpha = config.numerical_method.solver.linesearch_alpha_min
    xtol = config.numerical_method.solver.xtol
    ftol = config.numerical_method.solver.ftol
    max_inc = config.numerical_method.solver.max_increases
    
    nls_picard = PorousNSSolver.SafeNewtonSolver(ls, config.numerical_method.solver.picard_iterations, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor)
    nls_newton = PorousNSSolver.SafeNewtonSolver(ls, config.numerical_method.solver.newton_iterations, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor)
    
    solver_picard = FESolver(nls_picard)
    solver_newton = FESolver(nls_newton)
    
    x0 = FEFunction(X, zeros(num_free_dofs(X)))
    
    stab_cfg = PorousNSSolver.StabilizationConfig(
        method=config.numerical_method.stabilization.method,
        osgs_iterations=config.numerical_method.stabilization.osgs_iterations,
        osgs_tolerance=config.numerical_method.stabilization.osgs_tolerance
    )
    
    success, final_x0, iter_count, eval_time = PorousNSSolver.solve_system(
        X, Y, model, dΩ, Triangulation(model), h_cf, f_cf, alpha_h, g_cf, form,
        solver_picard, solver_newton,
        x0, c_1, c_2,
        config.physical_properties, stab_cfg, config.numerical_method.solver
    )
    
    return final_x0, eval_time, iter_count
end
function run_convergence()
    println("--- Cocquet Convergence Analysis (N=200 Reference) ---")
    
    base_config_path = joinpath(@__DIR__, "data", "test_config.json")
    base_config_dict = JSON3.read(read(base_config_path, String), Dict{String, Any})
    
    # Store dynamic script parameters before strict schema parsing drops them or warns
    Re = Float64(base_config_dict["Re"])
    c_in = Float64(base_config_dict["c_in"])
    k_list = convert(Vector{Int}, base_config_dict["k_convergence_list"])
    delta = Float64(get(base_config_dict, "outlet_truncation_delta", 0.0))
    interpolate_bypass = get(base_config_dict, "interpolate_solution_to_coarser_meshes", false)
    
    # Remove to prevent loud strict-schema warnings
    delete!(base_config_dict, "Re")
    delete!(base_config_dict, "c_in")
    delete!(base_config_dict, "k_convergence_list")
    delete!(base_config_dict, "outlet_truncation_delta")
    delete!(base_config_dict, "interpolate_solution_to_coarser_meshes")
    
    # All physical schemas and geometrical limits are universally driven by the native test JSON payload
    base_config = PorousNSSolver.load_config_from_dict(base_config_dict)
    
    L_max = base_config.domain.bounding_box[2]
    bounding_rule = x -> x[1] <= (L_max - delta)
    
    # Extract the target refinement nodes mathematically from the parsed structs.
    # partition limits are defined as [2*N, N], so N = partition[2] (the vertical subdivision)
    N_ref = base_config.numerical_method.mesh.partition[2]
    N_list = base_config.numerical_method.mesh.convergence_partitions
    
    results_dir = joinpath(@__DIR__, "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    h5_path = joinpath(results_dir, "convergence_cocquet.h5")
    
    # Initialize the file struct once and close it immediately to free lock
    h5open(h5_path, "w") do file
        file["N_list"] = collect(N_list)
    end
    
    as_list(x) = x isa Vector ? convert(Vector{String}, x) : [String(x)]
    nm_dict = get(base_config_dict, "numerical_method", Dict())
    stab_dict = get(nm_dict, "stabilization", Dict())
    methods = as_list(get(stab_dict, "method", ["ASGS", "OSGS"]))
    
    for method in methods
        for k in k_list
            println("\n\n==========================================================================================")
            println("[!] INITIATING BENCHMARK SEQUENCE | INTERPOLATION: P$(k)/P$(k) | STABILIZATION: $method")
            println("==========================================================================================")
            
            println("\n   [+] Assembling High-Fidelity Reference Mesh Solution (N = $N_ref) natively...")
            if !haskey(base_config_dict, "numerical_method")
                base_config_dict["numerical_method"] = Dict()
            end
            base_config_dict["numerical_method"]["stabilization"] = Dict("method" => method, "osgs_iterations" => 3)
            
            if !haskey(base_config_dict["numerical_method"], "element_spaces")
                base_config_dict["numerical_method"]["element_spaces"] = Dict()
            end
            base_config_dict["numerical_method"]["element_spaces"]["k_velocity"] = k
            base_config_dict["numerical_method"]["element_spaces"]["k_pressure"] = k
            
            println("Solving Reference Grid (N = $N_ref) -> [$(2*N_ref) x $N_ref] Elements...")
            base_config_dict["output"]["basename"] = "cocquet_ref_$(method)_P$(k)P$(k)_N$(N_ref)"
            mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref = build_solver(N_ref, base_config_dict, Re, c_in)
            xh_ref, time_ref, iters_ref = execute_solver(mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref)
            u_ref, p_ref = xh_ref
            
            # Build unbound free spaces on the reference mesh for exact subspace prolongation
            V_ref_free = TestFESpace(mod_ref, ru_ref, conformity=:H1)
            Q_ref_free = TestFESpace(mod_ref, rp_ref, conformity=:H1)
            
            # Build interpolable wrappers for the highly-resolved reference fields once
            iu_ref = Gridap.FESpaces.Interpolable(u_ref)
            ip_ref = Gridap.FESpaces.Interpolable(p_ref)
            
            # Export exhaustive baseline .vtu metrics mathematically mapped on the highest resolution geometry dynamically tracked
            PorousNSSolver.export_results(cfg_ref, mod_ref, u_ref, p_ref, "alpha" => alpha_ref)
            
            errors_l2_u = Float64[]
            errors_h1_u = Float64[]
            errors_l2_p = Float64[]
            errors_h1_p = Float64[]
            eval_times = Float64[]
            eval_iters = Int[]
            
            for N in N_list
                println("\n   ==============================================================")
                println("   [+] Launching Coarse Grid Algebraic Evaluation Space for N = $N")
                println("   ==============================================================")
                base_config_dict["output"]["basename"] = "cocquet_$(method)_P$(k)P$(k)_N$(N)"
                mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h = build_solver(N, base_config_dict, Re, c_in)
                
                if interpolate_bypass
                    # TEMPORARY EXPERIMENT: Bypass Formulation Solver entirely.
                    time_h = 0.0
                    iters_h = 0
                    
                    U_h = X_h[1]
                    P_h = X_h[2]
                    
                    # Purely map the exact N_ref evaluated reference bounds down into the coarse algebraic topological space
                    u_h = interpolate(iu_ref, U_h)
                    p_h = interpolate(ip_ref, P_h)
                else
                    # STANDARD EXECUTION: Solve the actual FEM numerical formulation on the coarse grid natively
                    xh_h, time_h, iters_h = execute_solver(mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h)
                    u_h, p_h = xh_h
                end
                
                # ---------------------------------------------------------------------
                # EXACT ERROR EVALUATION 
                # Encapsulated cleanly in metrics.jl to standardize consistency 
                # cross-checks against the fine reference representations natively.
                # ---------------------------------------------------------------------
                
                V_h_free = TestFESpace(mod_h, ru_h, conformity=:H1)
                Q_h_free = TestFESpace(mod_h, rp_h, conformity=:H1)
                
                res_u = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref; filter_func=bounding_rule)
                res_p = PorousNSSolver.compute_reference_errors(p_h, p_ref, ip_ref, Q_h_free, dΩ_h, dΩ_ref; filter_func=bounding_rule)
                
                l2_eu_nested, h1_eu_nested, l2_eu, h1_eu, eu_nested, eu_cons = res_u
                l2_ep_nested, h1_ep_nested, l2_ep, h1_ep, ep_nested, ep_cons = res_p
                
                # Export native coarse fields alongside exactly projected errors for visual inspection
                PorousNSSolver.export_results(cfg_h, mod_h, u_h, p_h, "alpha" => alpha_h, "e_u" => eu_nested, "e_p" => ep_nested)
                
                println("   [+] L2-norms of error globally mapping consistent reference integration fields | L2(u): ", l2_eu, " | L2(p): ", l2_ep)
                println("   [+] H1-seminorms evaluating spatial gradients mapped on topological exactness  | semiH1(u): ", h1_eu, " | semiH1(p): ", h1_ep)
                
                push!(errors_l2_u, l2_eu)
                push!(errors_h1_u, h1_eu)
                push!(errors_l2_p, l2_ep)
                push!(errors_h1_p, h1_ep)
                push!(eval_times, time_h)
                push!(eval_iters, iters_h)
            end
            
            println("Writing $(method)/P$(k)P$(k) slice to HDF5...")
            # Automatically open the file in read-write (r+) without erasing the file layout
            h5open(h5_path, "r+") do file
                group_name = "$method/P$(k)P$(k)"
                # if exists delete to overwrite successfully
                if haskey(file, group_name)
                    delete_object(file, group_name)
                end
                g = create_group(file, group_name)
                g["errors_l2_u"] = errors_l2_u
                g["errors_h1_u"] = errors_h1_u
                g["errors_l2_p"] = errors_l2_p
                g["errors_h1_p"] = errors_h1_p
                g["eval_times"] = eval_times
                g["eval_iters"] = eval_iters
                
                attributes(g)["total_time_s"] = sum(eval_times)
                attributes(g)["total_iters"] = sum(eval_iters)
                attributes(g)["outlet_truncation_delta"] = delta
            end
        end
    end
    println("\nConvergence Data Generated and Exported to convergence_cocquet.h5")
end

run_convergence()
