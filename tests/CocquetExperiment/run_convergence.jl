# tests/CocquetExperiment/run_convergence.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PorousNSSolver
using Gridap
using JSON
using HDF5
using LinearAlgebra

# Section 4.2 Smooth Porosity Field
# ε(y) = 0.45 + 0.55 * exp(y - 1.0)
alpha_func(x) = 0.45 + 0.55 * exp(x[2] - 1.0)

# Inflow profile for (Re=500, c_in=0.5)
c_in = 0.5
u_in(x) = VectorValue(c_in * x[2] * (1.0 - x[2]), 0.0)
u_wall(x) = VectorValue(0.0, 0.0)

function build_solver(N::Int, config_dict)
    # The paper uses domain [0, 2]x[0, 1] mapped up to N nodes on bounds
    # For a Cartesian grid, a grid of 2N x N elements maintains square shape (h = 1/N).
    # Since the paper specifies parameter N, we use partition = [2*N, N].
    local_config_dict = deepcopy(config_dict)
    local_config_dict["mesh"]["partition"] = [2*N, N]
    local_config = PorousNSSolver.load_config_from_dict(local_config_dict)
    
    model = PorousNSSolver.create_mesh(local_config)
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, local_config.discretization.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, local_config.discretization.k_pressure)
    
    labels = get_face_labeling(model)
    
    # Inlet [7], Outlet [8] left traction-free (Neumann), Walls [1,2,3,4,5,6] (Dirichlet 0)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["inlet", "walls"])
    U = TrialFESpace(V, [u_in, u_wall])
    
    # Pressure is left completely free (Neumann outlet provides pressure pinning naturally)
    Q = TestFESpace(model, refe_p, conformity=:H1)
    P = TrialFESpace(Q)
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    degree = 2 * local_config.discretization.k_velocity
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h_cf = CellField(h_array, Ω)
    
    alpha_h = interpolate_everywhere(alpha_func, Q)
    res(x, y) = PorousNSSolver.weak_form_residual(x, y, local_config, dΩ, h_cf, nothing, alpha_h)
    jac(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, local_config, dΩ, h_cf, nothing, alpha_h)
    
    op = FEOperator(res, jac, X, Y)
    
    return op, model, X, Y, dΩ
end

function run_convergence()
    println("--- Cocquet Convergence Analysis (N=100 Reference) ---")
    
    base_config_path = joinpath(@__DIR__, "data", "test_config.json")
    base_config_dict = JSON.parsefile(base_config_path)
    
    # Force Parameters for Figure 2: Re=500, Da=1.0, epsilon=1e-7
    base_config_dict["physical_parameters"]["Re"] = 500.0
    base_config_dict["physical_parameters"]["Da"] = 1.0
    base_config_dict["physical_parameters"]["epsilon"] = 1e-7
    base_config_dict["mesh"]["domain"] = [0.0, 2.0, 0.0, 1.0]
    base_config = PorousNSSolver.load_config_from_dict(base_config_dict)
    
    N_ref = 100
    N_list = [10, 20, 30]
    
    results_dir = joinpath(@__DIR__, "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Store multi-config arrays under HDF5
    h5open(joinpath(results_dir, "convergence_cocquet.h5"), "w") do file
        file["N_list"] = collect(N_list)
        
        for k in [1, 2]
            println("\n====== STARTING EQUAL-ORDER k=$k ======")
            base_config_dict["discretization"]["k_velocity"] = k
            base_config_dict["discretization"]["k_pressure"] = k
            
            println("Solving Reference Grid (N = $N_ref) -> [$(2*N_ref) x $N_ref] Elements...")
            op_ref, model_ref, X_ref, Y_ref, dΩ_ref = build_solver(N_ref, base_config_dict)
            nls_ref = NLSolver(show_trace=true, method=:newton, iterations=12, ftol=1e-13)
            solver_ref = FESolver(nls_ref)
            xh_ref = solve(solver_ref, op_ref)
            u_ref, p_ref = xh_ref
            
            errors_l2_u = Float64[]
            errors_l2_p = Float64[]
            
            for N in N_list
                println("\n--- Evaluating Coarse Grid (N = $N) ---")
                op_h, model_h, X_h, Y_h, dΩ_h = build_solver(N, base_config_dict)
                nls_h = NLSolver(show_trace=true, method=:newton, iterations=12, ftol=1e-13)
                solver_h = FESolver(nls_h)
                xh_h = solve(solver_h, op_h)
                u_h, p_h = xh_h
                
                # Gridap natively supports point-evaluation mapping from h-triangulations!
                # Thus, we calculate the error natively integrating coarse uh onto the fine reference geometry!
                u_ref_eval(x) = u_ref(x)
                p_ref_eval(x) = p_ref(x)
                
                eu = u_h - u_ref_eval
                ep = p_h - p_ref_eval
                
                l2_eu = sqrt(sum(∫( eu ⋅ eu ) * dΩ_h))
                l2_ep = sqrt(sum(∫( ep * ep ) * dΩ_h))
                
                println("N = $N => L2(u): ", l2_eu, " | L2(p): ", l2_ep)
                push!(errors_l2_u, l2_eu)
                push!(errors_l2_p, l2_ep)
            end
            
            group_name = "P$(k)P$(k)"
            g = create_group(file, group_name)
            g["errors_l2_u"] = errors_l2_u
            g["errors_l2_p"] = errors_l2_p
        end
    end
    println("\nConvergence Data Generated and Exported to convergence_cocquet.h5")
end

run_convergence()
