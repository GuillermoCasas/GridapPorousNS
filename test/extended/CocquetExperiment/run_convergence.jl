# test/long/CocquetExperiment/run_convergence.jl
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

# Inflow profile for (Re=500, c_in=0.5)
c_in = 0.5
u_in(x) = VectorValue(c_in * x[2] * (1.0 - x[2]), 0.0)
u_wall(x) = VectorValue(0.0, 0.0)

function build_solver(N::Int, config_dict)
    # The paper uses domain [0, 2]x[0, 1] mapped up to N nodes on bounds
    # For a Cartesian grid, a grid of 2N x N elements maintains square shape (h = 1/N).
    # Since the paper specifies parameter N, we use partition = [2*N, N].
    local_config_dict = deepcopy(config_dict)
    if !haskey(local_config_dict["numerical_method"], "mesh")
        local_config_dict["numerical_method"]["mesh"] = Dict()
    end
    local_config_dict["numerical_method"]["mesh"]["partition"] = [2*N, N]
    local_config = PorousNSSolver.load_config_from_dict(local_config_dict)
    
    model = PorousNSSolver.create_mesh(local_config)
    
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
    
    degree = 2 * local_config.numerical_method.element_spaces.k_velocity
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h_cf = CellField(h_array, Ω)
    
    alpha_h = interpolate_everywhere(alpha_func, Q)
    # Return raw geometric constructs instead of operators to enable explicit OSGS projections dynamically
    return model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, local_config
end

function execute_solver(model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, config)
    form = PorousNSSolver.build_formulation(config)
    f_cf = VectorValue(config.phys.f_x, config.phys.f_y)
    
    c_1 = 4.0 * config.numerical_method.element_spaces.k_velocity^4
    c_2 = 2.0 * config.numerical_method.element_spaces.k_velocity^2
    tau_reg_lim = config.phys.tau_regularization_limit
    
    res_asgs(x, y) = PorousNSSolver.build_stabilized_weak_form_residual(x, y, form, dΩ, h_cf, f_cf, alpha_h, 0.0, nothing, nothing, c_1, c_2, tau_reg_lim)
    jac_asgs(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_h, 0.0, nothing, nothing, c_1, c_2, tau_reg_lim, config.numerical_method.solver.freeze_jacobian_cusp, PorousNSSolver.ExactNewtonMode())
    
    op_asgs = FEOperator(res_asgs, jac_asgs, X, Y)
    nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), 12, config.numerical_method.solver.max_increases, config.numerical_method.solver.xtol, config.numerical_method.solver.ftol, config.numerical_method.solver.linesearch_alpha_min, 1e-4, config.numerical_method.solver.divergence_merit_factor, config.numerical_method.solver.stagnation_noise_floor)
    solver_newton = FESolver(nls_newton)
    
    eval_iters = 0
    eval_time = @elapsed begin
        # Extract native empty initialization exactly structured to Gridap arrays
        xh0 = FEFunction(X, zeros(num_free_dofs(X)))
        
        # Base initialization identically solves ASGS
        res_solve = solve!(xh0, solver_newton, op_asgs)
        cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
        nls_cache_asgs = cache_solve isa Tuple ? cache_solve[2] : cache_solve
        eval_iters += nls_cache_asgs.result.iterations
        x_sol = res_solve isa Tuple ? res_solve[1] : res_solve
    end
    return x_sol, eval_time, eval_iters
end

function run_convergence()
    println("--- Cocquet Convergence Analysis (N=200 Reference) ---")
    
    base_config_path = joinpath(@__DIR__, "data", "test_config.json")
    base_config_dict = JSON3.read(read(base_config_path, String), Dict{String, Any})
    
    # All physical schemas and geometrical limits are universally driven by the native test JSON payload
    base_config = PorousNSSolver.load_config_from_dict(base_config_dict)
    
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
    
    for method in ["ASGS", "OSGS"]
        println("\n====== STARTING METHOD $method ======")
        if !haskey(base_config_dict, "numerical_method")
            base_config_dict["numerical_method"] = Dict()
        end
        base_config_dict["numerical_method"]["stabilization"] = Dict("method" => method, "osgs_iterations" => 3)
        
        for k in [1, 2]
            println("\n   --- EQUAL-ORDER k=$k ---")
            if !haskey(base_config_dict["numerical_method"], "element_spaces")
                base_config_dict["numerical_method"]["element_spaces"] = Dict()
            end
            base_config_dict["numerical_method"]["element_spaces"]["k_velocity"] = k
            base_config_dict["numerical_method"]["element_spaces"]["k_pressure"] = k
            
            println("Solving Reference Grid (N = $N_ref) -> [$(2*N_ref) x $N_ref] Elements...")
            base_config_dict["output"]["basename"] = "cocquet_ref_$(method)_P$(k)P$(k)_N$(N_ref)"
            mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref = build_solver(N_ref, base_config_dict)
            xh_ref, time_ref, iters_ref = execute_solver(mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref)
            u_ref, p_ref = xh_ref
            
            # Build unbound free spaces on the reference mesh for exact subspace prolongation
            V_ref_free = TestFESpace(mod_ref, ru_ref, conformity=:H1)
            Q_ref_free = TestFESpace(mod_ref, rp_ref, conformity=:H1)
            
            # Export exhaustive baseline .vtu metrics mathematically mapped on the highest resolution geometry dynamically tracked
            PorousNSSolver.export_results(cfg_ref, mod_ref, u_ref, p_ref, "alpha" => alpha_ref)
            
            errors_l2_u = Float64[]
            errors_h1_u = Float64[]
            errors_l2_p = Float64[]
            errors_h1_p = Float64[]
            errors_l2_alpha = Float64[]
            errors_h1_alpha = Float64[]
            eval_times = Float64[]
            eval_iters = Int[]
            
            for N in N_list
                println("\n   Evaluating Coarse Grid (N = $N)")
                base_config_dict["output"]["basename"] = "cocquet_$(method)_P$(k)P$(k)_N$(N)"
                mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h = build_solver(N, base_config_dict)
                xh_h, time_h, iters_h = execute_solver(mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h)
                u_h, p_h = xh_h
                
                # ---------------------------------------------------------------------
                # EXACT SUBSPACE PROLONGATION
                # Because the uniform Cartesian grids are perfectly nested, V_h ⊂ V_ref. 
                # We losslessly prolongate the coarse solution onto the fine mesh topology.
                # This perfectly preserves the exact continuous error while bypassing dense 
                # in-loop geometric searches. Integrating on dΩ_ref eliminates quadrature 
                # aliasing by resolving sub-grid derivative discontinuities exactly.
                # ---------------------------------------------------------------------
                println("   [Speedup & Exactness] Prolongating Coarse Solution to Fine Reference Space natively...")
                
                # Wrap coarse fields to evaluate them at the fine nodes
                u_h_eval(x) = u_h(x)
                p_h_eval(x) = p_h(x)
                alpha_h_eval(x) = alpha_h(x)
                
                # Interpolate exactly onto the fine topology (cross-mesh search happens only at nodes)
                u_h_prol = interpolate_everywhere(u_h_eval, V_ref_free)
                p_h_prol = interpolate_everywhere(p_h_eval, Q_ref_free)
                alpha_h_prol = interpolate_everywhere(alpha_h_eval, Q_ref_free)
                
                # Compute error fields natively on the fine mesh memory
                eu = u_ref - u_h_prol
                ep = p_ref - p_h_prol
                e_alpha = alpha_ref - alpha_h_prol
                
                # Export native coarse fields only (avoids triggering geometric search 
                # of the fine error fields onto the coarse VTK mesh)
                PorousNSSolver.export_results(cfg_h, mod_h, u_h, p_h, "alpha" => alpha_h)
                
                # Integrate instantly without cross-mesh search overhead using the fine measure
                l2_eu = sqrt(sum(∫( eu ⋅ eu ) * dΩ_ref))
                l2_ep = sqrt(sum(∫( ep * ep ) * dΩ_ref))
                l2_ealpha = sqrt(sum(∫( e_alpha * e_alpha ) * dΩ_ref))
                
                # Explicitly distribute the gradient operator over the core FEFunctions 
                # before subtraction to safely bypass Gridap OperationCellField AD limits
                grad_eu = ∇(u_ref) - ∇(u_h_prol)
                grad_ep = ∇(p_ref) - ∇(p_h_prol)
                grad_ealpha = ∇(alpha_ref) - ∇(alpha_h_prol)
                
                h1_eu = sqrt(sum(∫( grad_eu ⊙ grad_eu ) * dΩ_ref))
                h1_ep = sqrt(sum(∫( grad_ep ⋅ grad_ep ) * dΩ_ref))
                h1_ealpha = sqrt(sum(∫( grad_ealpha ⋅ grad_ealpha ) * dΩ_ref))
                
                println("   => L2(u): ", l2_eu, " | L2(p): ", l2_ep, " | L2(alpha): ", l2_ealpha)
                println("   => semiH1(u): ", h1_eu, " | semiH1(p): ", h1_ep, " | semiH1(alpha): ", h1_ealpha)
                
                push!(errors_l2_u, l2_eu)
                push!(errors_h1_u, h1_eu)
                push!(errors_l2_p, l2_ep)
                push!(errors_h1_p, h1_ep)
                push!(errors_l2_alpha, l2_ealpha)
                push!(errors_h1_alpha, h1_ealpha)
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
                g["errors_l2_alpha"] = errors_l2_alpha
                g["errors_h1_alpha"] = errors_h1_alpha
                g["eval_times"] = eval_times
                g["eval_iters"] = eval_iters
                
                attributes(g)["total_time_s"] = sum(eval_times)
                attributes(g)["total_iters"] = sum(eval_iters)
            end
        end
    end
    println("\nConvergence Data Generated and Exported to convergence_cocquet.h5")
end

run_convergence()
