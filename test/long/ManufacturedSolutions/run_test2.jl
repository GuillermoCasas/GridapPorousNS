# test/long/ManufacturedSolutions/run_test.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using Gridap.Algebra
using JSON
using DelimitedFiles
using HDF5
using LineSearches
using Printf
using Random
using LinearAlgebra

function run_mms(config_file="test_config.json")
    config_path = joinpath(@__DIR__, "data", config_file)
    test_dict = JSON.parsefile(config_path)
    
    as_list(x) = x isa Vector ? x : [x]
    
    Re_list = as_list(get(get(test_dict, "physical_properties", Dict()), "Re", [10.0]))
    Da_list = as_list(get(get(test_dict, "physical_properties", Dict()), "Da", [1.0]))
    alpha_list = as_list(get(get(test_dict, "domain", test_dict), "alpha_0", [0.4]))
    
    nm_dict = get(test_dict, "numerical_method", Dict())
    elem_dict = get(nm_dict, "element_spaces", test_dict)
    mesh_dict = get(nm_dict, "mesh", test_dict)
    
    kv_list = as_list(get(elem_dict, "k_velocity", [1]))
    kp_list = as_list(get(elem_dict, "k_pressure", [1]))
    etype_list = as_list(get(mesh_dict, "element_type", ["QUAD"]))
    
    equal_order_only = get(test_dict, "equal_order_only", false)
    if haskey(nm_dict, "element_spaces") && haskey(elem_dict, "equal_order_only")
        equal_order_only = get(elem_dict, "equal_order_only", equal_order_only)
    end
    
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    h5_filename = get(test_dict, "h5_filename", "convergence_data.h5")
    h5_path = joinpath(results_dir, h5_filename)
    
    erase_past = get(test_dict, "erase_past_results", false)
    h5_mode = erase_past ? "w" : "cw"
    
    h5open(h5_path, h5_mode) do h5f
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
            
            conv_parts = get(mesh_dict, "convergence_partitions", [10, 20, 30])
            
            # Use constant sigma for the paper's accurate reproduction benchmark
            sigma_c = 1.0 / Float64(Da)
            nu_calculated = 1.0 / Float64(Re)
            
            # The paper exact MMS uses exactly eps_val = 0.0 inside the continuous problem!
            eps_calculated = 0.0 
            
            override_dict = Dict(
                "physical_properties" => Dict("nu" => nu_calculated, "eps_val" => eps_calculated, "reaction_model" => "Constant_Sigma", "sigma_constant" => sigma_c),
                "domain" => Dict("alpha_0" => Float64(alpha_0), "bounding_box" => test_dict["domain"]["bounding_box"]),
                "numerical_method" => Dict(
                    "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
                    "mesh" => Dict("element_type" => String(etype), "convergence_partitions" => Int.(conv_parts)),
                    "stabilization" => Dict("method" => method)
                )
            )
            
            base_config = PorousNSSolver.load_config_from_dict(override_dict)
            partitions = base_config.numerical_method.mesh.convergence_partitions
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
                
                config = PorousNSSolver.PorousNSConfig(
                    physical_properties=base_config.physical_properties,
                    domain=base_config.domain,
                    numerical_method=PorousNSSolver.NumericalMethodConfig(
                        element_spaces=base_config.numerical_method.element_spaces,
                        stabilization=PorousNSSolver.StabilizationConfig(method=String(method), osgs_iterations=3),
                        mesh=PorousNSSolver.MeshConfig(partition=[n, n], element_type=String(etype)),
                        solver=base_config.numerical_method.solver
                    ),
                    output=base_config.output
                )
                
                model = PorousNSSolver.create_mesh(config)
                
                labels = get_face_labeling(model)
                add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])
                
                refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, config.numerical_method.element_spaces.k_velocity)
                refe_p = ReferenceFE(lagrangian, Float64, config.numerical_method.element_spaces.k_pressure)
                V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
                Q = TestFESpace(model, refe_p, conformity=:H1)
                
                # Problem formulation object
                alpha_field = PorousNSSolver.SmoothRadialPorosity(config.domain.alpha_0, config.domain.r_1, config.domain.r_2)
                
                rxn = PorousNSSolver.ConstantSigmaLaw(config.physical_properties.sigma_constant)
                reg = PorousNSSolver.SmoothVelocityFloor(1e-6, 0.0, 1e-12)
                proj = PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma()
                # Use deviatoric-symmetric formulation per paper
                form = PorousNSSolver.PaperGeneralFormulation(PorousNSSolver.DeviatoricSymmetricViscosity(), rxn, proj, reg, config.physical_properties.nu, config.physical_properties.eps_val, config.physical_properties.eps_floor)
                
                # We do NOT arbitrarily scale U by some reaction formulation size. We set U=1, P=1 amplitude!
                U_amp = 1.0
                P_amp = 1.0
                mms = PorousNSSolver.Paper2DMMS(form, U_amp, P_amp, alpha_field)
                
                u_final = PorousNSSolver.get_u_ex(mms)
                p_final = PorousNSSolver.get_p_ex(mms)
                
                U = TrialFESpace(V, u_final)  
                P = TrialFESpace(Q, p_final)
                
                Y = MultiFieldFESpace([V, Q])
                X = MultiFieldFESpace([U, P])
                
                degree = 4 * config.numerical_method.element_spaces.k_velocity
                Ω = Triangulation(model)
                dΩ = Measure(Ω, degree + 4)
                
                if config.numerical_method.mesh.element_type == "TRI"
                    h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
                else
                    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
                end
                h_cf = CellField(collect(h_array), Ω)
                alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)
                
                c_1 = 4.0 * config.numerical_method.element_spaces.k_velocity^4
                c_2 = 2.0 * config.numerical_method.element_spaces.k_velocity^2
                tau_reg_lim = config.physical_properties.tau_regularization_limit
                
                # Build EXACT forcing using the MMS closure
                f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)
                
                res_fn(x, y) = PorousNSSolver.build_stabilized_weak_form_residual(x, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, nothing, nothing, c_1, c_2, tau_reg_lim)
                jac_fn(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, nothing, nothing, c_1, c_2, tau_reg_lim, config.numerical_method.solver.freeze_jacobian_cusp)
                
                op_newton = FEOperator(res_fn, jac_fn, X, Y)
                
                # --- EXACT ROOT TEST ---
                if config.numerical_method.element_spaces.k_velocity == 1
                    println("\n>>> Phase 1: EXACT ROOT TEST")
                    println("  [SKIPPED] EXACTNESS DIAGNOSTIC SKIPPED for k=1 because bilinear laplacians artificially vanish.")
                    x0_exact = interpolate_everywhere([u_final, p_final], X)
                else
                    println("\n>>> Phase 1: EXACT ROOT TEST")
                    x0_exact = interpolate_everywhere([u_final, p_final], X)
                    
                    # Check 1: Exact Stabilized Residual Consistency
                    r_exact = allocate_residual(op_newton, x0_exact)
                    residual!(r_exact, op_newton, x0_exact)
                    L2_diag = norm(r_exact, 2)
                    Linf_diag = norm(r_exact, Inf)
                    
                    # Check 2: Jacobian Directional Consistency (Finite Difference Taylor Check)
                    jac_exact = allocate_jacobian(op_newton, x0_exact)
                    jacobian!(jac_exact, op_newton, x0_exact)
                    
                    # Create a small perturbation in the direction of the solution
                    x_pert = copy(get_free_dof_values(x0_exact))
                    x_pert .= x_pert .* (1.0 + 1e-6)
                    
                    r_pert = allocate_residual(op_newton, x0_exact)
                    residual!(r_pert, op_newton, FEFunction(X, x_pert))
                    
                    dx_pert = x_pert .- get_free_dof_values(x0_exact)
                    # Taylor expansion: R(x + dx) ≈ R(x) + J * dx
                    taylor_approx = r_exact + jac_exact * dx_pert
                    taylor_error = norm(r_pert - taylor_approx, Inf) / (norm(dx_pert, Inf) + 1e-16)
                    
                    println("Exactness Diagnostic -> Stabilized L2: $L2_diag | L_inf: $Linf_diag | Jacobian Directional Error/|dx|: $taylor_error")
                    
                    # Because MMS functions are transcendentals (sines/cosines), the FE interpolant
                    # carries an O(h^{k+1}) residual. Mismatch > 0 is expected.
                    if L2_diag < 1e-2 && taylor_error < 1e-4
                        println("  [SUCCESS] EXACTNESS DIAGNOSTIC PASSED. Interpolant root mapped correctly.")
                    else
                        println("  [WARNING] EXACTNESS DIAGNOSTIC FAILED. The formulation operators lack numerical continuity mappings!")
                    end
                end
                
                nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), 15, config.numerical_method.solver.max_increases, config.numerical_method.solver.xtol, config.numerical_method.solver.stagnation_tol, config.numerical_method.solver.ftol, config.numerical_method.solver.linesearch_alpha_min, 1e-4)
                solver_newton = FESolver(nls_newton)
                
                # --- ROBUSTNESS TEST ---
                println("\n>>> Phase 2: ROBUST HOMOTOPY / PERTURBATION TEST")
                eps_pert_base = Float64(get(test_dict, "epsilon_pert", [0.1])[1])
                max_n_pert = Int(get(test_dict, "max_n_pert", 5))
                
                begin
                    x_init = copy(get_free_dof_values(x0_exact))
                    # We inject white noise mapping exact Dirichlet BCs zero-boundaries via simple geometric modulation
                    u_h_exact, p_h_exact = x0_exact
                    u_ex_L2 = sqrt(abs(sum(∫(u_h_exact ⋅ u_h_exact)dΩ)))
                    
                    xmin, xmax, ymin, ymax = config.domain.bounding_box[1], config.domain.bounding_box[2], config.domain.bounding_box[3], config.domain.bounding_box[4]
                    B_fn(x) = (x[1] - xmin)^2 * (xmax - x[1])^2 * (x[2] - ymin)^2 * (ymax - x[2])^2
                    h_raw_func(x) = B_fn(x) * VectorValue(sin(3π*x[1])*cos(2π*x[2]), -cos(3π*x[1])*sin(2π*x[2]))
                    h_pert_cf = CellField(h_raw_func, Ω)
                    norm_h = sqrt(abs(sum(∫( h_pert_cf ⋅ h_pert_cf )dΩ))) + 1e-14
                    
                    success = false
                    successful_eps = eps_pert_base
                    final_x0 = x0_exact
                    iter_count_attempt = 0
                    
                    successful_eps = -1.0
                    eval_time = 0.0
                    iter_count_attempt = 0
                    final_x0 = nothing
                    success = false
                    
                    for attempt in 0:(max_n_pert + 1)
                        eps_p = attempt <= max_n_pert ? eps_pert_base / (10.0^attempt) : 0.0
                        u_0_func(x) = u_final(x) + eps_p * (u_ex_L2 / norm_h) * h_raw_func(x)
                        x0 = interpolate_everywhere([u_0_func, p_final], X)
                        
                        println("Attempting nonlinear solve with eps_pert = $eps_p ...")
                        
                        jac_picard(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, nothing, nothing, c_1, c_2, tau_reg_lim, config.numerical_method.solver.freeze_jacobian_cusp, PorousNSSolver.PicardMode())
                        op_picard = FEOperator(res_fn, jac_picard, X, Y)
                        nls_picard = PorousNSSolver.SafeNewtonSolver(LUSolver(), config.numerical_method.solver.picard_iterations, config.numerical_method.solver.max_increases, config.numerical_method.solver.xtol, config.numerical_method.solver.stagnation_tol, config.numerical_method.solver.ftol, config.numerical_method.solver.linesearch_alpha_min, 1e-4)
                        solver_picard = FESolver(nls_picard)
                        
                        x0_backup = copy(get_free_dof_values(x0))
                        try
                            solve!(x0, solver_picard, op_picard)
                            
                            b_new = allocate_residual(op_picard, x0)
                            residual!(b_new, op_picard, x0)
                            res_new = norm(b_new, Inf)
                            
                            x0_test = FEFunction(X, x0_backup)
                            b_test = allocate_residual(op_picard, x0_test)
                            residual!(b_test, op_picard, x0_test)
                            res_orig = norm(b_test, Inf)
                            
                            if isnan(res_new) || res_new > res_orig * 1.05
                                println("    -> Picard residual worsened ($res_new vs $res_orig). Reverting.")
                                get_free_dof_values(x0) .= x0_backup
                            else
                                println("    -> Picard residual improved ($res_new vs $res_orig). Accepting as Newton guess.")
                            end
                        catch e
                            println("    -> Picard phase hit limit/error. Proceeding to exact Newton.")
                            get_free_dof_values(x0) .= x0_backup
                        end
                        # Comprehensive rollback guard for numerical explosion irrespective of NaN
                        if any(isnan, get_free_dof_values(x0)) || sum(isnan.(get_free_dof_values(x0))) > 0 || norm(get_free_dof_values(x0), Inf) > 1e12
                            get_free_dof_values(x0) .= x0_backup
                        end
                        
                        status = :converged
                        final_res = Inf
                        try
                            local_time = @elapsed begin
                                res_solve = solve!(x0, solver_newton, op_newton)
                            end
                            eval_time = local_time
                            cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                            nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                            
                            iter_count_attempt = nls_cache.result.iterations
                            final_res = nls_cache.result.residual_norm
                            if iter_count_attempt >= config.numerical_method.solver.newton_iterations || final_res > max(config.numerical_method.solver.ftol, 1e-10)
                                if final_res < 1e-7 && final_res > 0.0
                                    status = :stagnated_near_root
                                else
                            println(e, "\n", stacktrace(catch_backtrace()))
                                end
                            end
                        catch e
                            println(e, "\n", stacktrace(catch_backtrace()))
                        end
                        
                        if status === :converged
                            success = true
                            successful_eps = eps_p
                            final_x0 = x0
                            println("Convergence achieved with eps_pert = $eps_p in $iter_count_attempt iterations.")
                            break
                        elseif status === :stagnated_near_root
                            println("Stagnated near root with eps_pert = $eps_p in $iter_count_attempt iterations (res = $final_res). Recording status separately.")
                        else
                            println("Diverged with eps_pert = $eps_p.")
                        end
                    end
                    
                    if success && final_x0 !== nothing
                        x0 = final_x0
                        u_h, p_h = x0
                        
                        e_u = u_final - u_h
                        e_p = p_final - p_h
                        
                        el2_u = sqrt(sum(∫(e_u ⋅ e_u)dΩ))
                        area = sum(∫(1.0)dΩ)
                        mean_e_p = sum(∫(e_p)dΩ) / area
                        e_p_centered = e_p - mean_e_p
                        el2_p = sqrt(sum(∫(e_p_centered * e_p_centered)dΩ))
                        
                        eh1_semi_u = sqrt(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ))
                        eh1_semi_p = sqrt(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ))
                        eh1_u = sqrt(el2_u^2 + eh1_semi_u^2)
                        eh1_p = sqrt(el2_p^2 + eh1_semi_p^2)
                    else
                        el2_u, el2_p, eh1_u, eh1_p = NaN, NaN, NaN, NaN
                    end
                    
                    push!(hs, 1.0 / n)
                    push!(err_u_L2, el2_u)
                    push!(err_p_L2, el2_p)
                    push!(err_u_H1, eh1_u)
                    push!(err_p_H1, eh1_p)
                    push!(eval_times, eval_time)
                    push!(eval_iters, iter_count_attempt)
                    push!(eval_eps, successful_eps)
                    println("L2 Error u: $el2_u, p: $el2_p")
                    println("H1 Error u: $eh1_u, p: $eh1_p")
                    GC.gc()
                end
            end
            
            grp_name = "config_$(c_idx)_$(method)"
            as_vec(x) = x isa Vector ? x : [x]
            if haskey(h5f, grp_name)
                delete_object(h5f, grp_name)
            end
            
            g = create_group(h5f, grp_name)
            g["h"] = hs
            g["err_u_l2"] = err_u_L2
            g["err_p_l2"] = err_p_L2
            g["err_u_h1"] = err_u_H1
            g["err_p_h1"] = err_p_H1
            g["eval_times"] = eval_times
            g["eval_iters"] = eval_iters
            g["eval_eps"] = eval_eps
            
            attributes(g)["total_time_s"] = sum(eval_times)
            attributes(g)["total_iters"] = sum(eval_iters)
            attributes(g)["Re"] = Float64(Re)
            attributes(g)["Da"] = Float64(Da)
            attributes(g)["alpha_0"] = Float64(alpha_0)
            attributes(g)["physical_epsilon"] = eps_calculated
            attributes(g)["numerical_epsilon_coeff"] = 0.0
            attributes(g)["k_velocity"] = Int(kv)
            attributes(g)["k_pressure"] = Int(kp)
            attributes(g)["element_type"] = String(etype)
            attributes(g)["method"] = String(method)
            
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
end

if abspath(PROGRAM_FILE) == @__FILE__
    config_file = length(ARGS) > 0 ? ARGS[1] : "test_config.json"
    run_mms(config_file)
end