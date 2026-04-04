# tests/ManufacturedSolutions/run_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PorousNSSolver
using Gridap
using JSON
using DelimitedFiles
using HDF5
using LineSearches
using Printf
using Random

function alpha_field(x, config::PorousNSSolver.PorousNSConfig)
    alpha_0 = config.porosity.alpha_0
    r1 = config.porosity.r_1
    r2 = config.porosity.r_2
    dx = x[1] - 0.0
    dy = x[2] - 0.0
    r_sq = dx^2 + dy^2
    r = sqrt(r_sq)
    if r <= r1
        return alpha_0
    elseif r >= r2
        return 1.0
    else
        eta = (r_sq - r1^2) / (r2^2 - r1^2)
        gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
        if gamma_val > 100.0
            return 1.0
        elseif gamma_val < -100.0
            return alpha_0
        end
        return 1.0 - (1.0 - alpha_0) / (1.0 + exp(gamma_val))
    end
end

# Global closures strictly preventing recreation within quadrature loops
function get_force_scale(config)
    Re = config.phys.Re; Da = config.phys.Da
    a0 = config.porosity.alpha_0
    
    geom_a = ((1.0 - a0) / a0)^2
    geom_b = (1.0 - a0) / a0
    
    visc_scale = 1.0 / Re
    drag_scale = (150.0 / (Da * Re)) * geom_a + 1.75 * geom_b
    
    return 1.0 + visc_scale + drag_scale
end

function u_ex(x, config)
    f_scale = get_force_scale(config)
    return (1.0 / f_scale) * VectorValue(sin(pi*x[1])*sin(pi*x[2]), cos(pi*x[1])*cos(pi*x[2]))
end

# DO NOT SCALE PRESSURE. Leave it strictly O(1)
p_ex(x, config) = cos(pi*x[1])*sin(pi*x[2])

function g_ex(x, config::PorousNSSolver.PorousNSConfig)
    _, grad_alpha_val, _ = analyze_alpha(x, config)
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_num = config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da))
    
    # MUST EXACTLY MATCH formulation clamping to prevent O(10^-8) mass residual drifts
    eps_val = max(config.phys.physical_epsilon + eps_num, 1e-8)
    
    return eps_val * p_ex(x, config) + (u_ex(x, config) ⋅ grad_alpha_val)
end

function analyze_alpha(x, config)
    alpha_0 = config.porosity.alpha_0
    r1 = config.porosity.r_1
    r2 = config.porosity.r_2
    dx = x[1] - 0.0
    dy = x[2] - 0.0
    r_sq = dx^2 + dy^2
    r = sqrt(r_sq)
    
    if r <= r1 || r >= r2
        return alpha_field(x, config), VectorValue(0.0, 0.0), 0.0
    end
    
    eta = (r_sq - r1^2) / (r2^2 - r1^2)
    gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
    
    # Float64 double-precision asymptote cutoff to prevent NaN from Inf/Inf
    if gamma_val > 100.0 || gamma_val < -100.0
        return alpha_field(x, config), VectorValue(0.0, 0.0), 0.0
    end
    
    exp_g = exp(gamma_val)
    alpha_val = 1.0 - (1.0 - alpha_0) / (1.0 + exp_g)
    
    deta_dr = 2.0 * r / (r2^2 - r1^2)
    d2eta_dr2 = 2.0 / (r2^2 - r1^2)
    
    dgamma_deta = (2.0*eta^2 - 2.0*eta + 1.0) / (eta^2 * (1.0 - eta)^2)
    d2gamma_deta2 = (4.0*eta - 2.0) / (eta^2 * (1.0 - eta)^2) - 2.0 * (2.0*eta^2 - 2.0*eta + 1.0)*(1.0 - 2.0*eta) / (eta^3 * (1.0 - eta)^3)
    
    da_dg = (1.0 - alpha_0) * exp_g / (1.0 + exp_g)^2
    d2a_dg2 = (1.0 - alpha_0) * exp_g * (1.0 - exp_g) / (1.0 + exp_g)^3
    
    da_dr = da_dg * dgamma_deta * deta_dr
    d2a_dr2 = d2a_dg2 * (dgamma_deta * deta_dr)^2 + da_dg * (d2gamma_deta2 * deta_dr^2 + dgamma_deta * d2eta_dr2)
    
    lap_alpha = d2a_dr2 + (1.0/r) * da_dr
    grad_alpha = VectorValue(da_dr * (dx/r), da_dr * (dy/r))
    return alpha_val, grad_alpha, lap_alpha
end

function grad_u_ex(x, config)
    f_scale = get_force_scale(config)
    return (1.0 / f_scale) * TensorValue(
        pi * cos(pi*x[1])*sin(pi*x[2]), pi * sin(pi*x[1])*cos(pi*x[2]),
        -pi * sin(pi*x[1])*cos(pi*x[2]), -pi * cos(pi*x[1])*sin(pi*x[2])
    )
end

lap_u_ex(x, config) = -2.0 * pi^2 * u_ex(x, config)

grad_p_ex(x, config) = VectorValue(-pi * sin(pi*x[1])*sin(pi*x[2]), pi * cos(pi*x[1])*cos(pi*x[2]))

function f_ex(x, config::PorousNSSolver.PorousNSConfig)
    α_val, grad_alpha_val, lap_alpha_val = analyze_alpha(x, config)
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    
    u_val = u_ex(x, config)
    grad_u_val = grad_u_ex(x, config)
    lap_u_val = lap_u_ex(x, config)
    grad_p_val = grad_p_ex(x, config)
    
    # Exact advection & viscous pseudo-traction perfectly matching your Gridap transpose syntax
    conv_u_val = transpose(grad_u_val) ⋅ u_val
    div_stress = α_val * ν * lap_u_val + ν * (transpose(grad_u_val) ⋅ grad_alpha_val) 
    
    a_term = PorousNSSolver.a_resistance(α_val, Re, Da)
    b_term = PorousNSSolver.b_resistance(α_val)
    σ_val = a_term + b_term * norm(u_val)
    
    return α_val * conv_u_val - div_stress + α_val * grad_p_val + σ_val * u_val
end

struct ExactAlpha <: Function
    cfg::PorousNSSolver.PorousNSConfig
end
(a::ExactAlpha)(x) = alpha_field(x, a.cfg)
Gridap.∇(a::ExactAlpha) = x -> analyze_alpha(x, a.cfg)[2]

struct ExactF <: Function
    cfg::PorousNSSolver.PorousNSConfig
end
(f::ExactF)(x) = f_ex(x, f.cfg)

struct ExactG <: Function
    cfg::PorousNSSolver.PorousNSConfig
end
(g::ExactG)(x) = g_ex(x, g.cfg)


function run_mms(config_file="test_config.json")
    config_path = joinpath(@__DIR__, "data", config_file)
    test_dict = JSON.parsefile(config_path)
    
    # Helper to enforce arrays
    as_list(x) = x isa Vector ? x : [x]
    
    Re_list = as_list(get(test_dict["physical_parameters"], "Re", [10.0]))
    Da_list = as_list(get(test_dict["physical_parameters"], "Da", [1.0]))
    eps_list = as_list(get(test_dict["physical_parameters"], "epsilon", [0.0]))
    alpha_list = as_list(get(test_dict["porosity_field"], "alpha_0", [0.4]))
    kv_list = as_list(get(test_dict["discretization"], "k_velocity", [1]))
    kp_list = as_list(get(test_dict["discretization"], "k_pressure", [1]))
    etype_list = as_list(get(test_dict["mesh"], "element_type", ["QUAD"]))
    equal_order_only = get(test_dict, "equal_order_only", false)
    if haskey(test_dict, "discretization")
        equal_order_only = get(test_dict["discretization"], "equal_order_only", equal_order_only)
    end
    
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    h5_path = joinpath(results_dir, "convergence_data.h5")
    
    erase_past = get(test_dict, "erase_past_results", false)
    h5_mode = erase_past ? "w" : "cw"
    
    h5open(h5_path, h5_mode) do h5f
        # Pre-scan existing HDF5 to seamlessly append newly requested combinations dynamically mapping to identical c_idx schemas
        max_idx = 0
        existing_signatures = Dict()
        if !erase_past
            for gname in keys(h5f)
                parts = split(gname, "_")
                if length(parts) >= 3 && parts[1] == "config"
                    idx = parse(Int, parts[2])
                    max_idx = max(max_idx, idx)
                    
                    g = h5f[gname]
                    att = attributes(g)
                    try
                        r = Float64(read(att["Re"]))
                        d = Float64(read(att["Da"]))
                        a = Float64(read(att["alpha_0"]))
                        kv_v = Int(read(att["k_velocity"]))
                        kp_v = Int(read(att["k_pressure"]))
                        et = String(read(att["element_type"]))
                        # Identify core physical signature mapping dynamically 
                        sig_base = (r, d, a, kv_v, kp_v, et)
                        existing_signatures[sig_base] = idx
                    catch
                    end
                end
            end
        end
        
        total_runs = sum(1 for (Re, Da, alpha_0, kv, kp, etype) in Iterators.product(Re_list, Da_list, alpha_list, kv_list, kp_list, etype_list) if !(equal_order_only && kv != kp)) * 2
        run_idx = 0

        for (Re, Da, alpha_0, kv, kp, etype) in Iterators.product(Re_list, Da_list, alpha_list, kv_list, kp_list, etype_list)
            if equal_order_only && kv != kp
                continue
            end
            
            sig_base = (Float64(Re), Float64(Da), Float64(alpha_0), Int(kv), Int(kp), String(etype))
            
            if haskey(existing_signatures, sig_base)
                c_idx = existing_signatures[sig_base]
            else
                max_idx += 1
                c_idx = max_idx
                existing_signatures[sig_base] = c_idx
            end
            
            for method in ["ASGS", "OSGS"]
            run_idx += 1
            println("\n========================================")
            println("Running Config $run_idx (of $total_runs) [ID: $c_idx]: Re=$Re, Da=$Da, alpha_0=$alpha_0, k=$kv, type=$etype, method=$method")
            println("========================================")
            
            conv_parts = get(test_dict["mesh"], "convergence_partitions", [10, 20, 30])
            
            # Extract physically constant values assuming array of length 1 for tests keeping defaults intact
            phys_eps = get(test_dict["physical_parameters"], "physical_epsilon", [0.0])[1]
            num_eps_coeff = get(test_dict["physical_parameters"], "numerical_epsilon_coefficient", [0.0001])[1]
            
            override_dict = Dict(
                "physical_parameters" => Dict("Re" => Float64(Re), "Da" => Float64(Da), "physical_epsilon" => Float64(phys_eps), "numerical_epsilon_coefficient" => Float64(num_eps_coeff)),
                "porosity_field" => Dict("alpha_0" => Float64(alpha_0)),
                "discretization" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
                "mesh" => Dict("element_type" => String(etype), "convergence_partitions" => Int.(conv_parts), "domain" => test_dict["mesh"]["domain"])
            )
            
            base_config = PorousNSSolver.load_config_from_dict(override_dict)
            partitions = base_config.mesh.convergence_partitions
            hs = Float64[]
            err_u_L2 = Float64[]
            err_p_L2 = Float64[]
            err_u_H1 = Float64[]
            err_p_H1 = Float64[]
            eval_times = Float64[]
            eval_iters = Int[]
            eval_eps = Float64[]
    
            for n in partitions
                println("Running MMS for partition $n x $n")
        
        vtu_name = "mms_Re$(Re)_Da$(Da)_a$(alpha_0)_$(etype)_n$(n)_$(method)"
        config = PorousNSSolver.PorousNSConfig(
            phys=base_config.phys,
            porosity=base_config.porosity,
            discretization=base_config.discretization,
            mesh=PorousNSSolver.MeshConfig(domain=base_config.mesh.domain, partition=[n, n], element_type=String(etype)),
            output=PorousNSSolver.OutputConfig(directory=joinpath(@__DIR__, "results"), basename=vtu_name),
            solver=PorousNSSolver.SolverConfig(method=method, osgs_iterations=3)
        )
        
        model = PorousNSSolver.create_mesh(config)
        
        # For MMS, all boundaries are Dirichlet for velocity
        labels = get_face_labeling(model)
        add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])
        
        refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, config.discretization.k_velocity)
        refe_p = ReferenceFE(lagrangian, Float64, config.discretization.k_pressure)
        V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
        Q = TestFESpace(model, refe_p, conformity=:H1)
        
        u_final(x) = u_ex(x, config)
        U = TrialFESpace(V, u_final)  
        
        p_final(x) = p_ex(x, config)
        P = TrialFESpace(Q, p_final)
        
        Y = MultiFieldFESpace([V, Q])
        X = MultiFieldFESpace([U, P])
        
        Ω = Triangulation(model)
        
        # ALGORITHMIC RATIONALE: High-Order Dynamic Integration
        # We fundamentally bypass Gridap's JIT AST compilation lock by using plain black-box closures inside CellField. 
        # By elevating the geometric global `dΩ` limit proportionally (+4), we structurally resolve steep geometric integrals
        # perfectly matching $P_{k+2}$ stability, while evaluating natively instantly without allocating 360k-DOF interpolation matrices!
        degree = 4 * config.discretization.k_velocity
        dΩ = Measure(Ω, degree + 4)
        

        
        if config.mesh.element_type == "TRI"
            h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
        else
            h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
        end
        h = CellField(h_array, Ω)
        
        f_fn(x) = f_ex(x, config)
        alpha_fn_wrapper(x) = alpha_field(x, config)
        g_fn(x) = g_ex(x, config)
        
        f_cf = CellField(ExactF(config), Ω)
        alpha_cf = CellField(ExactAlpha(config), Ω)
        g_cf = CellField(ExactG(config), Ω)
        
        res_fn(x, y) = PorousNSSolver.weak_form_residual(x, y, config, dΩ, h, f_cf, alpha_cf, g_cf)
        jac_fn(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, config, dΩ, h, f_cf, alpha_cf, g_cf)
        
        println("Creating FEOperators...")
        # Only the exact Newton operator is needed for MMS spatial error tracking
        op_newton = FEOperator(res_fn, jac_fn, X, Y)
        
        # ALGORITHMIC RATIONALE: Initialize exactly at the manufactured root.
        # This isolates pure spatial FME truncation errors without triggering nonlinear globalization failures.
        println("Interpolating exact initial guess...")
        x0_exact = interpolate_everywhere([u_final, p_final], X)
        free_exact = copy(get_free_dof_values(x0_exact))
        
        nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), 15, config.solver.max_increases, config.solver.xtol, config.solver.stagnation_tol, config.solver.ftol, config.solver.linesearch_tolerance, config.solver.linesearch_alpha_min)
        solver_newton = FESolver(nls_newton)
        
        println("Solving Newton-Raphson from perturbed initialization...")
        iter_count = 0
        eps_pert_base = Float64(get(test_dict, "epsilon_pert", [0.1])[1])
        max_n_pert = Int(get(test_dict, "max_n_pert", 5))
        
        success = false
        successful_eps = eps_pert_base
        final_x0 = x0_exact
        iter_count_attempt = 0
        
        xmin, xmax, ymin, ymax = config.mesh.domain[1], config.mesh.domain[2], config.mesh.domain[3], config.mesh.domain[4]
        
        Random.seed!(42)
        kx, ky = Tuple(rand(1:3, 3)), Tuple(rand(1:3, 3))
        A, ph = Tuple(rand(3) .* 2.0 .- 1.0), Tuple(rand(3) .* 2π)

        B_fn(x) = (x[1] - xmin)^2 * (xmax - x[1])^2 * (x[2] - ymin)^2 * (ymax - x[2])^2

        function psi_raw_fn(x)
            return A[1] * sin(kx[1]*π*x[1] + ph[1]) * cos(ky[1]*π*x[2] + ph[1]) +
                   A[2] * sin(kx[2]*π*x[1] + ph[2]) * cos(ky[2]*π*x[2] + ph[2]) +
                   A[3] * sin(kx[3]*π*x[1] + ph[3]) * cos(ky[3]*π*x[2] + ph[3])
        end

        psi(x) = B_fn(x) * psi_raw_fn(x)
        
        function h_raw_func(x)
            grad_psi = ∇(psi)(x) 
            alpha = alpha_fn_wrapper(x)
            return (1.0 / alpha) * VectorValue(grad_psi[2], -grad_psi[1])
        end
        
        h_cf = CellField(h_raw_func, Ω)
        norm_h = sqrt(abs(sum(∫( h_cf ⋅ h_cf )dΩ))) + 1e-14
        u_h_exact, p_h_exact = x0_exact
        u_ex_L2 = sqrt(abs(sum(∫(u_h_exact ⋅ u_h_exact)dΩ)))
        
        # Check normalization and exact mass conservation of analytical formulation
        test_h_norm = sqrt(sum(∫( h_cf ⋅ h_cf )dΩ)) / (norm_h - 1e-14)
        div_alpha_h = sqrt(abs(sum(∫( (∇⋅(alpha_cf * h_cf)) * (∇⋅(alpha_cf * h_cf)) )dΩ)))
        
        Γ = BoundaryTriangulation(model)
        dΓ = Measure(Γ, degree+4)
        boundary_h_norm = sqrt(abs(sum(∫( h_cf ⋅ h_cf )dΓ)))
        
        println("  [Test] Perturbation intrinsic L2 norm base    (expected ~1.0): ", test_h_norm)
        println("  [Test] Perturbation mass residual ||∇⋅(α h)|| (expected ~0.0): ", div_alpha_h)
        println("  [Test] Perturbation boundary norm ||h||_∂Ω    (expected ~0.0): ", boundary_h_norm)
        
        @assert abs(test_h_norm - 1.0) < 1e-10 "Perturbation normalization failed!"
        @assert div_alpha_h < 1e-10 "Perturbation mass conservation failed!"
        @assert boundary_h_norm < 1e-10 "Perturbation no-slip Dirichlet boundary constraint failed!"
        
        eval_time = @elapsed begin
            for attempt in 0:(max_n_pert + 1)
                eps_p = attempt <= max_n_pert ? eps_pert_base / (10.0^attempt) : 0.0
                
                function u_0_func(x)
                    return u_final(x) + eps_p * (u_ex_L2 / norm_h) * h_raw_func(x)
                end
                
                x0 = interpolate_everywhere([u_0_func, p_final], X)
                
                println("Attempting nonlinear solve with eps_pert = $eps_p ...")
                iter_count_attempt = 0
                attempt_success = true
                
                # --- Safe Picard Globalization Pass ---
                jac_picard(x, dx, y) = PorousNSSolver.weak_form_jacobian_picard(x, dx, y, config, dΩ, h, f_cf, alpha_cf, g_cf, nothing, nothing)
                op_picard = FEOperator(res_fn, jac_picard, X, Y)
                nls_picard = NLSolver(show_trace=false, method=:newton, iterations=config.solver.picard_iterations)
                solver_picard = FESolver(nls_picard)
                
                x0_backup = copy(get_free_dof_values(x0))
                try
                    println("    -> Running Picard Globalization (Max $(config.solver.picard_iterations) iters)...")
                    solve!(x0, solver_picard, op_picard)
                catch e
                    println("    -> Picard phase hit limit/error. Proceeding to exact Newton.")
                end
                if any(isnan, get_free_dof_values(x0))
                    println("    -> [Warning] Picard generated NaNs. Reverted to pristine initial guess.")
                    get_free_dof_values(x0) .= x0_backup
                end
                # --------------------------------------
                
                try
                    # 1. ALWAYS execute ASGS first to find a smooth, physical root
                    res_solve = solve!(x0, solver_newton, op_newton)
                    cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                    nls_cache_asgs = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                    
                    iter_count_attempt = nls_cache_asgs.result.iterations
                    final_res = nls_cache_asgs.result.residual_norm
                    
                    # Ensure script evaluates against ftol (with a safety floor for Float64 noise)
                    if iter_count_attempt >= config.solver.newton_iterations || final_res > max(config.solver.ftol, 1e-12)
                        attempt_success = false
                    end
                    
                    # 2. ONLY if OSGS is requested, apply it using the pristine ASGS root
                    if attempt_success && method == "OSGS"
                        osgs_iters = config.solver.osgs_iterations
                        
                        V_pi = TestFESpace(model, refe_u, conformity=:H1) 
                        Q_pi = TestFESpace(model, refe_p, conformity=:H1)
                        U_pi = TrialFESpace(V_pi)
                        P_pi = TrialFESpace(Q_pi)
                        
                        for osgs_idx in 1:osgs_iters
                            println("OSGS Iteration $osgs_idx/$osgs_iters")
                            u_h_prev, p_h_prev = x0
                            
                            println("    [Timing] Starting project_residuals computation...")
                            pi_u_raw, pi_p_raw = @time PorousNSSolver.project_residuals(u_h_prev, p_h_prev, config, dΩ, h, f_cf, alpha_cf, g_cf, V_pi, Q_pi, U_pi, P_pi)
                            println("    [Timing] project_residuals completed.")
                            
                            let pi_u = pi_u_raw, pi_p = pi_p_raw
                                res_osgs(x, y) = PorousNSSolver.weak_form_residual(x, y, config, dΩ, h, f_cf, alpha_cf, g_cf, pi_u, pi_p)
                                jac_osgs(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, config, dΩ, h, f_cf, alpha_cf, g_cf, pi_u, pi_p)
                                
                                op_osgs = FEOperator(res_osgs, jac_osgs, X, Y)
                                nls_osgs = PorousNSSolver.SafeNewtonSolver(LUSolver(), config.solver.newton_iterations, config.solver.max_increases, config.solver.xtol, config.solver.stagnation_tol, config.solver.ftol, config.solver.linesearch_tolerance, config.solver.linesearch_alpha_min)
                                solver_osgs = FESolver(nls_osgs)
                                
                                res_solve_osgs = solve!(x0, solver_osgs, op_osgs)
                                cache_osgs = res_solve_osgs isa Tuple ? res_solve_osgs[2] : res_solve_osgs
                                nls_cache_osgs = cache_osgs isa Tuple ? cache_osgs[2] : cache_osgs
                                
                                iter_count_attempt += nls_cache_osgs.result.iterations
                                final_res = nls_cache_osgs.result.residual_norm
                                
                                if nls_cache_osgs.result.iterations >= config.solver.newton_iterations || final_res > max(config.solver.ftol, 1e-12)
                                    attempt_success = false
                                    break
                                end
                            end
                        end
                    end
                catch e
                    println("Nonlinear solver crashed: ", typeof(e), " - ", e)
                    attempt_success = false
                end
                
                if attempt_success
                    success = true
                    successful_eps = eps_p
                    iter_count = iter_count_attempt
                    final_x0 = x0
                    println("Convergence achieved with eps_pert = $eps_p in $iter_count iterations.")
                    break
                else
                    println("Diverged with eps_pert = $eps_p.")
                end
            end
            if !success
                println("Convergence NOT achieved after $max_n_pert retries. Using the last state to preserve errors.")
                iter_count = iter_count_attempt
                final_x0 = x0
                successful_eps = eps_pert_base / (10.0^max_n_pert)
            end
        end
        x0 = final_x0
        u_h, p_h = x0
        
        u_init_func(x) = u_final(x) + successful_eps * (u_ex_L2 / norm_h) * h_raw_func(x)
        
        # Compute errors
        e_u = u_final - u_h
        e_p = p_final - p_h
                PorousNSSolver.export_results(config, model, u_h, p_h,
                    "u_ex" => u_final,
                    "p_ex" => p_final,
                    "e_u" => e_u,
                    "e_p" => e_p,
                    "alpha" => alpha_fn_wrapper,
                    "u_init" => u_init_func,
                    "p_init" => p_final)
                
                el2_u = sqrt(sum(∫(e_u ⋅ e_u)dΩ))
                
                # Center pressure to isolate mean-fixing errors introduced by nullspace boundaries
                area = sum(∫(1.0)dΩ)
                mean_e_p = sum(∫(e_p)dΩ) / area
                e_p_centered = e_p - mean_e_p
                el2_p = sqrt(sum(∫(e_p_centered * e_p_centered)dΩ))
                
                eh1_semi_u = sqrt(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ))
                eh1_semi_p = sqrt(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ))
                eh1_u = sqrt(el2_u^2 + eh1_semi_u^2)
                eh1_p = sqrt(el2_p^2 + eh1_semi_p^2)
                
                push!(hs, 1.0 / n)
                push!(err_u_L2, el2_u)
                push!(err_p_L2, el2_p)
                push!(err_u_H1, eh1_u)
                push!(err_p_H1, eh1_p)
                push!(eval_times, eval_time)
                push!(eval_iters, iter_count)
                push!(eval_eps, successful_eps)
                println("L2 Error u: $el2_u, p: $el2_p")
                println("H1 Error u: $eh1_u, p: $eh1_p")
                
                # Prevent caching leaks
                GC.gc()
            end
            
            # Save combination to HDF5
            grp_name = "config_$(c_idx)_$(method)"
            
            # Helper to enforce vector type from HDF5 scalar readings
            as_vec(x) = x isa Vector ? x : [x]

            # Non-destructively merge with existing datasets to avoid nullifying prior partition sweeps!
            if haskey(h5f, grp_name) && !erase_past
                println("Merging preexisting result tuple globally within HDF5 namespace for $(grp_name)...")
                g_old = h5f[grp_name]
                old_hs = as_vec(read(g_old["h"]))
                old_u_L2 = as_vec(read(g_old["err_u_l2"]))
                old_p_L2 = as_vec(read(g_old["err_p_l2"]))
                old_u_H1 = as_vec(read(g_old["err_u_h1"]))
                old_p_H1 = as_vec(read(g_old["err_p_h1"]))
                old_times = zeros(Float64, length(old_hs))
                old_iters = zeros(Int, length(old_hs))
                old_eps = zeros(Float64, length(old_hs))
                try
                    if haskey(g_old, "eval_times")
                        old_times = as_vec(read(g_old["eval_times"]))
                    end
                    if haskey(g_old, "eval_iters")
                        old_iters = as_vec(read(g_old["eval_iters"]))
                    end
                    if haskey(g_old, "eval_eps")
                        old_eps = as_vec(read(g_old["eval_eps"]))
                    end
                catch e
                    println("Warning: Corrupted performance tracking nodes detected in H5 (likely from a force-killed run). Overwriting legacy metrics with zero.")
                end
                
                # Zip old values into dictionary keyed by grid interval h
                combined = Dict{Float64, Tuple}()
                for i in 1:length(old_hs)
                    combined[old_hs[i]] = (old_u_L2[i], old_p_L2[i], old_u_H1[i], old_p_H1[i], old_times[i], old_iters[i], old_eps[i])
                end
                
                # Overlay newly computed resolutions
                for i in 1:length(hs)
                    combined[hs[i]] = (err_u_L2[i], err_p_L2[i], err_u_H1[i], err_p_H1[i], eval_times[i], eval_iters[i], eval_eps[i])
                end
                
                # Sort strictly by descending h = mathematically ascending N refinement
                sorted_hs = sort(collect(keys(combined)), rev=true)
                
                hs = sorted_hs
                err_u_L2 = Float64[combined[h][1] for h in sorted_hs]
                err_p_L2 = Float64[combined[h][2] for h in sorted_hs]
                err_u_H1 = Float64[combined[h][3] for h in sorted_hs]
                err_p_H1 = Float64[combined[h][4] for h in sorted_hs]
                eval_times = Float64[combined[h][5] for h in sorted_hs]
                eval_iters = Int[combined[h][6] for h in sorted_hs]
                eval_eps = Float64[combined[h][7] for h in sorted_hs]
                
                # Nuke group structure safely to allow continuous reallocation
                delete_object(h5f, grp_name)
            elseif haskey(h5f, grp_name)
                delete_object(h5f, grp_name)
                println("Overwriting preexisting result tuple globally within HDF5 namespace for $(grp_name)...")
            end
            
            g = create_group(h5f, grp_name)
            
            # Write datasets
            g["h"] = hs
            g["err_u_l2"] = err_u_L2
            g["err_p_l2"] = err_p_L2
            g["err_u_h1"] = err_u_H1
            g["err_p_h1"] = err_p_H1
            g["eval_times"] = eval_times
            g["eval_iters"] = eval_iters
            g["eval_eps"] = eval_eps
            
            # Write metadata attributes
            attributes(g)["total_time_s"] = sum(eval_times)
            attributes(g)["total_iters"] = sum(eval_iters)
            attributes(g)["Re"] = Float64(Re)
            attributes(g)["Da"] = Float64(Da)
            attributes(g)["alpha_0"] = Float64(alpha_0)
            attributes(g)["physical_epsilon"] = Float64(phys_eps)
            attributes(g)["numerical_epsilon_coeff"] = Float64(num_eps_coeff)
            attributes(g)["k_velocity"] = Int(kv)
            attributes(g)["k_pressure"] = Int(kp)
            attributes(g)["element_type"] = String(etype)
            attributes(g)["method"] = String(method)
            
            # Calculate rates
            compute_slope(x, y) = sum((x .- sum(x)/length(x)) .* (y .- sum(y)/length(y))) / sum((x .- sum(x)/length(x)).^2)
            log_h = log.(hs)
            rate_u_l2 = compute_slope(log_h, log.(err_u_L2))
            rate_u_h1 = compute_slope(log_h, log.(err_u_H1))
            rate_p_l2 = compute_slope(log_h, log.(err_p_L2))
            rate_p_h1 = compute_slope(log_h, log.(err_p_H1))
            
            attributes(g)["rate_u_l2"] = rate_u_l2
            attributes(g)["rate_u_h1"] = rate_u_h1
            attributes(g)["rate_p_l2"] = rate_p_l2
            attributes(g)["rate_p_h1"] = rate_p_h1
            
            end
        end
    end
    
    println("MMS setup complete. Parametric sweep written to convergence_data.h5")
    println("Please run `python3 plot_results.py` to compile the colored convergence_report.md tables.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    config_file = length(ARGS) > 0 ? ARGS[1] : "test_config.json"
    run_mms(config_file)
end
