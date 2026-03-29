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
function u_ex(x, config)
    Re = config.phys.Re; Da = config.phys.Da
    # ALGORITHMIC RATIONALE: Universal scaling to trap numerical floating-point limits natively.
    # At extreme viscosity limits (Re=10^-6), standard u_ex generates f_ex ~ 10^12.
    # In standard Float64 precision, adding O(1) fields to 10^12 introduces a 10^-4 truncation baseline,
    # mapping false '0.00' spatial convergence slopes. Scaling velocity explicitly bounds maximum stress inside O(1) limits.
    force_scale = 1.0 + (1.0 / Re) + (1.0 / (Da * Re))
    return (1.0 / force_scale) * VectorValue(sin(pi*x[1])*sin(pi*x[2]), cos(pi*x[1])*cos(pi*x[2]))
end

# DO NOT SCALE PRESSURE. Leave it strictly O(1)
p_ex(x, config) = cos(pi*x[1])*sin(pi*x[2])

function g_ex(x, config::PorousNSSolver.PorousNSConfig)
    _, grad_alpha_val, _ = analyze_alpha(x, config)
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_val = config.phys.physical_epsilon + (config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)))
    
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
    Re = config.phys.Re; Da = config.phys.Da
    force_scale = 1.0 + (1.0 / Re) + (1.0 / (Da * Re))
    return (1.0 / force_scale) * TensorValue(
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

function run_mms()
    config_path = joinpath(@__DIR__, "data", "test_config.json")
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
    
    h5open(h5_path, "w") do h5f
        # Precompute total valid configurations
        total_configs = 0
        for (Re, Da, alpha_0, kv, kp, etype) in Iterators.product(Re_list, Da_list, alpha_list, kv_list, kp_list, etype_list)
            if equal_order_only && kv != kp
                continue
            end
            total_configs += 1
        end
        
        config_idx = 1
        for (Re, Da, alpha_0, kv, kp, etype) in Iterators.product(Re_list, Da_list, alpha_list, kv_list, kp_list, etype_list)
            if equal_order_only && kv != kp
                continue
            end
            println("\n========================================")
            println("Running Config $config_idx of $total_configs: Re=$Re, Da=$Da, alpha_0=$alpha_0, k=$kv, type=$etype")
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
    
            for n in partitions
                println("Running MMS for partition $n x $n")
        
        vtu_name = "mms_Re$(Re)_Da$(Da)_a$(alpha_0)_$(etype)_n$(n)"
        config = PorousNSSolver.PorousNSConfig(
            phys=base_config.phys,
            porosity=base_config.porosity,
            discretization=base_config.discretization,
            mesh=PorousNSSolver.MeshConfig(domain=base_config.mesh.domain, partition=[n, n], element_type=String(etype)),
            output=PorousNSSolver.OutputConfig(directory=joinpath(@__DIR__, "results"), basename=vtu_name)
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
        
        # ALGORITHMIC RATIONALE: Adjust the degree calculation to properly integrate ASGS non-linear bounds
        # ASGS formulation cross-multiplies convective structures tau_1 * (u*grad_v) * (u*grad_u).
        # For P2 interpolations, velocity is O(x^2), convective derivative is O(x^3).
        # Integration of the ASGS cross terms requires evaluating O(x^6) polynomials. Degree 4*k prevents aliasing.
        degree = 4 * config.discretization.k_velocity
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree)
        
        is_tri = config.mesh.element_type == "TRI"
        h_array = lazy_map(v -> is_tri ? sqrt(2.0 * abs(v)) : sqrt(abs(v)), get_cell_measure(Ω))
        h = CellField(h_array, Ω)
        
        f(x) = f_ex(x, config)
        alpha_fn(x) = alpha_field(x, config)
        
        println("Nodally interpolating alpha and f...")
        alpha_h = interpolate_everywhere(alpha_fn, Q)
        f_h = interpolate_everywhere(f, V)
        
        g_fn(x) = g_ex(x, config)
        g_h = interpolate_everywhere(g_fn, Q)
        
        println("Defining formulation closures...")
        res(x, y) = PorousNSSolver.weak_form_residual(x, y, config, dΩ, h, f_h, alpha_h, g_h)
        jac(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, config, dΩ, h, f_h, alpha_h, g_h)
        jac_picard(x, dx, y) = PorousNSSolver.weak_form_jacobian_picard(x, dx, y, config, dΩ, h, f_h, alpha_h, g_h)
        
        println("Creating FEOperators...")
        op_picard = FEOperator(res, jac_picard, X, Y)
        op_newton = FEOperator(res, jac, X, Y)
        
        # ALGORITHMIC RATIONALE: Hybrid Picard-Newton Initialization
        # At Re=10^6, standard Newton-Raphson advective Jacobians diverge immediately from zero.
        # Stage 1 (Picard): drops the unstable d(u \cdot grad)u velocity adjoint variation natively forming a continuous bounded quadratic basin.
        nls_picard = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=5)
        solver_picard = FESolver(nls_picard)
        println("Solving Picard Initialization...")
        x_picard = solve(solver_picard, op_picard)
        
        # Stage 2 (Newton-Raphson): executes the rigorous analytical Fréchet exact Jacobian from the Picard initialization.
        nls_newton = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=15)
        solver_newton = FESolver(nls_newton)
        println("Solving Newton-Raphson...")
        x_h = solve!(x_picard, solver_newton, op_newton) # Start from Picard guess!
        
        u_h, p_h = x_picard
        
        # Compute errors
        e_u = u_final - u_h
        e_p = p_final - p_h
                PorousNSSolver.export_results(config, model, u_h, p_h,
                    "u_ex" => u_final,
                    "p_ex" => p_final,
                    "e_u" => e_u,
                    "e_p" => e_p,
                    "alpha" => alpha_fn)
                
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
                println("L2 Error u: $el2_u, p: $el2_p")
                println("H1 Error u: $eh1_u, p: $eh1_p")
                
                # Prevent caching leaks
                GC.gc()
            end
            
            # Save combination to HDF5
            grp_name = "config_$config_idx"
            g = create_group(h5f, grp_name)
            
            # Write datasets
            g["h"] = hs
            g["err_u_l2"] = err_u_L2
            g["err_p_l2"] = err_p_L2
            g["err_u_h1"] = err_u_H1
            g["err_p_h1"] = err_p_H1
            
            # Write metadata attributes
            attributes(g)["Re"] = Float64(Re)
            attributes(g)["Da"] = Float64(Da)
            attributes(g)["alpha_0"] = Float64(alpha_0)
            attributes(g)["physical_epsilon"] = Float64(phys_eps)
            attributes(g)["numerical_epsilon_coeff"] = Float64(num_eps_coeff)
            attributes(g)["k_velocity"] = Int(kv)
            attributes(g)["k_pressure"] = Int(kp)
            attributes(g)["element_type"] = String(etype)
            
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
            
            config_idx += 1
        end
    end
    
    # Generate Markdown Report
    report_file = joinpath(results_dir, "convergence_report.md")
    open(report_file, "w") do io
        println(io, "# Convergence Rate and FME Table\n")
        println(io, "| Config | Re | Da | α_0 | k | Elem | rate_u_L2 | opt_u_L2 | rate_p_L2 | opt_p_L2 | err_p_L2 (fine) |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
        
        h5open(h5_path, "r") do h5f
            for group_name in sort(keys(h5f), by=x->parse(Int, split(x, "_")[2]))
                g = h5f[group_name]
                attr = attributes(g)
                c_idx = split(group_name, "_")[2]
                re = read(attr["Re"])
                da = read(attr["Da"])
                a0 = read(attr["alpha_0"])
                kv = read(attr["k_velocity"])
                etype = read(attr["element_type"])
                
                rate_u = read(attr["rate_u_l2"])
                rate_p = read(attr["rate_p_l2"])
                err_p_last = read(g["err_p_l2"])[end]
                
                opt_u = kv + 1.0
                opt_p = Float64(kv)
                
                Printf.@printf(io, "| C%s | %.0e | %.0e | %.2f | %d | %s | %5.2f | %5.2f | %5.2f | %5.2f | %.4e |\n", 
                        c_idx, re, da, a0, kv, etype, rate_u, opt_u, rate_p, opt_p, err_p_last)
            end
        end
    end
    
    println("MMS setup complete. Parametric sweep written to convergence_data.h5")
    println("Markdown convergence report generated at $(report_file)")
end

run_mms()
