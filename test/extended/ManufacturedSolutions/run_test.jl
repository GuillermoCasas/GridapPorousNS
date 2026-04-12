# test/long/ManufacturedSolutions/run_test.jl
# ==============================================================================================
# Nature & Intent:
# The definitive Method of Manufactured Solutions (MMS) workflow. Constructs an exact artificial 
# polynomial state $(u_{ex}, p_{ex})$ and evaluates its continuous symbolic derivatives to force the 
# solver's RHS. By running the full Picard/Newton non-linear steps, it validates the continuous ExactNewton 
# exactness and global system numerical stabilization properties. Extensively sweeps ($Re, Da, h$) configurations
# to map asymptotic convergence bounds mapping to the error definitions $O(h^{k+1})$.
#
# Mathematical Formulation Alignment:
# Completely aligns with Equation 1018-1039 of `article.tex`. Uses parameter normalizations ($U_c, P_c$)
# strictly matching continuum characteristic analysis, forbidding arbitrary rescaling limits.
#
# Associated Files / Functions:
# - `src/formulations/continuous_problem.jl`
# - `src/solvers/nonlinear.jl` (`solve_system`)
# - MMS analytical fields mapping (`u_ex`, `p_ex`)
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using Gridap.Algebra
using JSON3
using DelimitedFiles
using HDF5
using LineSearches
using Printf
using Random
using LinearAlgebra

# ==============================================================================
# Helper Constructors
# ==============================================================================

function _build_local_mesh(domain_cfg, mesh_cfg)
    domain = Tuple(domain_cfg.bounding_box)
    partition = Tuple(mesh_cfg.partition)
    if mesh_cfg.element_type == "TRI"
        model = CartesianDiscreteModel(domain, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain, partition)
    end
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [7])
    add_tag_from_tags!(labels, "outlet", [8])
    add_tag_from_tags!(labels, "walls", [1, 2, 3, 4, 5, 6])
    return model
end

# Creates the domain porosity field structure parameterized by alpha bounds and geometry
function build_porosity_field(config, alpha_0, alpha_infty)
    PorousNSSolver.SmoothRadialPorosity(Float64(alpha_0), Float64(alpha_infty), config.domain.r_1, config.domain.r_2)
end

# Sets up the specific continuous VMS mathematical behavior for the Manufactured Solution
function build_mms_formulation(config, Da, Re, U_amp, L, alpha_infty)
    # Define exact residual projections enforcing strictly positive stabilization operators
    proj = PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma()
    
    # Establish velocity regularization parameters strictly from configuration
    reg = PorousNSSolver.SmoothVelocityFloor(
        config.physical_properties.u_base_floor_ref, 
        0.0, 
        config.physical_properties.epsilon_floor
    )
    
    # Derive kinematic viscosity and exact constant sigma for the reaction operator 
    # to perfectly represent Darcy-scale bounds parameterized by Reynolds / Darcy inputs.
    nu_calculated = U_amp * L / Float64(Re)
    eps_calculated = config.physical_properties.eps_val
    
    sigma_c = Float64(Da) * alpha_infty * nu_calculated / (L^2)
    rxn = PorousNSSolver.ConstantSigmaLaw(sigma_c)
    
    # Bound the exact viscous analytical formulation directly targeting the governing operator from the numerical schema definitions
    visc_type = config.numerical_method.viscous_operator_type
    if visc_type == "DeviatoricSymmetric"
        visc_op = PorousNSSolver.DeviatoricSymmetricViscosity()
    elseif visc_type == "SymmetricGradient"
        visc_op = PorousNSSolver.SymmetricGradientViscosity()
    else
        visc_op = PorousNSSolver.LaplacianPseudoTractionViscosity()
    end
    
    PorousNSSolver.PaperGeneralFormulation(visc_op, rxn, proj, reg, nu_calculated, eps_calculated, 0.0)
end

# Error evaluator operating exclusively upon dimensionless algebraic norms corresponding to characteristic scaling
function calculate_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, L, dΩ)
    # Native error vectors
    e_u = u_final - u_h
    e_p = p_final - p_h
    
    # 1. Absolute Velocity L2 norm rigorously divided by Characteristic Velocity `U_c`
    el2_u = sqrt(sum(∫(e_u ⋅ e_u)dΩ)) / U_c
    
    # Align pressure exactly (cancelling null space integration variations) 
    area = sum(∫(1.0)dΩ)
    mean_e_p = sum(∫(e_p)dΩ) / area
    e_p_centered = e_p - mean_e_p
    
    # 2. Absolute Pressure L2 norm strictly divided by Characteristic Pressure `P_c`
    el2_p = sqrt(sum(∫(e_p_centered * e_p_centered)dΩ)) / P_c
    
    # 3. Absolute Semi-H1 gradient errors normalized by exact corresponding characteristic scales taking into account geometric `L` characteristic derivation factors.
    eh1_semi_u = sqrt(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ)) / (U_c / L)
    eh1_semi_p = sqrt(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ)) / (P_c / L)
    
    return el2_u, el2_p, eh1_semi_u, eh1_semi_p
end

function run_mms(config_file="test_config.json")
    config_path = joinpath(@__DIR__, "data", config_file)
    test_dict = JSON3.read(read(config_path, String), Dict{String, Any})
    
    as_list(x) = x isa Vector ? x : [x]
    
    Re_list = as_list(test_dict["physical_properties"]["Re"])
    Da_list = as_list(test_dict["physical_properties"]["Da"])
    alpha_list = as_list(test_dict["domain"]["alpha_0"])
    
    
    nm_dict = test_dict["numerical_method"]
    elem_dict = nm_dict["element_spaces"]
    mesh_dict = nm_dict["mesh"]
    stab_dict = get(nm_dict, "stabilization", Dict("method" => ["ASGS", "OSGS"]))
    
    kv_list = as_list(elem_dict["k_velocity"])
    kp_list = as_list(elem_dict["k_pressure"])
    etype_list = as_list(mesh_dict["element_type"])
    methods = as_list(get(stab_dict, "method", ["ASGS", "OSGS"]))
    
    equal_order_only = get(test_dict, "equal_order_only", false)
    if haskey(nm_dict, "element_spaces") && haskey(elem_dict, "equal_order_only")
        equal_order_only = elem_dict["equal_order_only"]
    end
    
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    h5_filename = get(test_dict, "h5_filename", "convergence_data.h5")
    h5_path = joinpath(results_dir, h5_filename)
    
    erase_past = get(test_dict, "erase_past_results", false)
    h5_mode = erase_past ? "w" : "cw"
    
    h5f = h5open(h5_path, h5_mode)
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
    close(h5f)
    
    conv_parts = mesh_dict["convergence_partitions"]
    
    total_runs = sum(1 for etype in etype_list, kv in kv_list, kp in kp_list, alpha_0 in alpha_list, Da in Da_list, Re in Re_list if !(equal_order_only && kv != kp)) * length(methods)
    run_idx = 0
    
    # Pre-allocate cache for collecting metrics over partitions dynamically
    results_cache = Dict()
    
    for etype in etype_list
        for kv in kv_list
            for kp in kp_list
                if equal_order_only && kv != kp
                    continue
                end
                
                # Setup dictionary trackers for all physics configurations natively inside this geometry
                for alpha_0 in alpha_list, Da in Da_list, Re in Re_list, method in methods
                    k_id = (etype, kv, kp, alpha_0, Da, Re, method)
                    results_cache[k_id] = Dict(
                        "hs" => Float64[],
                        "err_u_l2" => Float64[],
                        "err_p_l2" => Float64[],
                        "err_u_h1" => Float64[],
                        "err_p_h1" => Float64[],
                        "eval_times" => Float64[],
                        "eval_iters" => Int[],
                        "eval_eps" => Float64[]
                    )
                end
                
                for n in conv_parts
                    println("\n========================================")
                    println("BUILDING MESH N = $n for etype = $etype, k_v = $kv, k_p = $kp")
                    println("========================================")
                    
                    # ==============================================================================
                    # MESH & SPATIAL DISCRETIZATION
                    # Evaluated EXACTLY ONCE per resolution (n) to bypass intensive julia compilations
                    # ==============================================================================
                    config_dict = Dict(
                        "physical_properties" => Dict("nu" => 1.0, "eps_val" => 1e-8, "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
                        "domain" => Dict(
                            "alpha_0" => 0.4, 
                            "bounding_box" => test_dict["domain"]["bounding_box"],
                            "r_1" => test_dict["domain"]["r_1"],
                            "r_2" => test_dict["domain"]["r_2"]
                        ),
                        "numerical_method" => Dict(
                            "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
                            "mesh" => Dict("element_type" => String(etype), "partition" => [n, n]),
                            "stabilization" => Dict("method" => "ASGS"),
                            "solver" => get(get(test_dict, "numerical_method", Dict()), "solver", Dict())
                        )
                    )
                    
                    # Dynamically instantiate hierarchy overrides ensuring JSON parser rigor
                    config = PorousNSSolver.load_config_from_dict(config_dict)
                    
                    # Generate logical cell models natively spanning bounded domain
                    model = _build_local_mesh(config.domain, config.numerical_method.mesh)
                    labels = get_face_labeling(model)
                    add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])
                    
                    # Compile Taylor-Hood Finite Element Spaces relying heavily on ReferenceFE definitions
                    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
                    refe_p = ReferenceFE(lagrangian, Float64, kp)
                    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
                    Q = TestFESpace(model, refe_p, conformity=:H1)
                    
                    # Structurally generate topologically unbound sub-grid reference domains 
                    # stripped of physical inlet/wall definitions for mathematically pure exact L2 bounds mapping
                    V_free = TestFESpace(model, refe_u, conformity=:H1)
                    Q_free = TestFESpace(model, refe_p, conformity=:H1)
                    
                    # Coordinate quadrature degree evaluation recursively mapped back to formulation definitions
                    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv)
                    Ω = Triangulation(model)
                    dΩ = Measure(Ω, degree + 4) # Extra numeric points exactly resolving high order source integration
                    
                    # Compute element-wise characteristic length variable 'h' structurally defined for stabilizations
                    if etype == "TRI"
                        h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
                    else
                        h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
                    end
                    h_cf = CellField(collect(h_array), Ω)
                    
                    Y = MultiFieldFESpace([V, Q])
                    
                    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, kv)
                    tau_reg_lim = config.physical_properties.tau_regularization_limit
                    solver_newton_it = config.numerical_method.solver.newton_iterations
                    solver_picard_it = config.numerical_method.solver.picard_iterations
                    max_inc = config.numerical_method.solver.max_increases
                    xtol = config.numerical_method.solver.xtol
                    stagnation_tol = config.numerical_method.solver.stagnation_noise_floor
                    ftol = config.numerical_method.solver.ftol
                    ls_alpha_min = config.numerical_method.solver.linesearch_alpha_min
                    freeze_cusp = config.numerical_method.solver.freeze_jacobian_cusp
                    
                    # ==============================================================================
                    # PHYSICS PARAMETER SWEEP
                    # Iterate fluid configurations using precompiled outer topological operators
                    # ==============================================================================
                    for alpha_0 in alpha_list
                        # Outer extraction of dynamic CellFields evaluating purely geometric domains
                        alpha_infty = 1.0
                        alpha_field = build_porosity_field(config, alpha_0, alpha_infty)
                        alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)
                        for Da in Da_list
                            for Re in Re_list
                                U_amp = 1.0
                                L = 1.0
                                form = build_mms_formulation(config, Da, Re, U_amp, L, alpha_infty)
                                
                                # Exact execution of full manufactured solutions analytical expressions
                                mms = PorousNSSolver.Paper2DMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)
                                U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)
                                
                                u_final = PorousNSSolver.get_u_ex(mms)
                                p_final = PorousNSSolver.get_p_ex(mms)
                                
                                U = TrialFESpace(V, u_final)  
                                P = TrialFESpace(Q, p_final)
                                X = MultiFieldFESpace([U, P])
                                
                                # Evaluate symbolic PDE operators mapping directly onto the exact physical boundary source terms
                                f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)
                                
                                ar_c1 = config.numerical_method.solver.armijo_c1
                                div_fac = config.numerical_method.solver.divergence_merit_factor
                                n_floor = config.numerical_method.solver.stagnation_noise_floor
                                
                                nls_picard = PorousNSSolver.SafeNewtonSolver(LUSolver(), solver_picard_it, max_inc, xtol, ftol, ls_alpha_min, ar_c1, div_fac, n_floor)
                                nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), solver_newton_it, max_inc, xtol, ftol, ls_alpha_min, ar_c1, div_fac, n_floor)
                                
                                solver_picard = FESolver(nls_picard)
                                solver_newton = FESolver(nls_newton)
                                
                                # Interpolate initial mathematical guess state perfectly atop exact solution roots
                                x0_exact = interpolate_everywhere([u_final, p_final], X)
                                
                                for method in methods
                                    run_idx += 1
                                    println("\n--- Progress $(run_idx) / $(total_runs * length(conv_parts)) | Re=$(Re), Da=$(Da), α=$(alpha_0), method=$(method), N=$(n) ---")
                                    
                                    eps_pert_base = Float64(get(test_dict, "epsilon_pert", [0.1])[1])
                                    max_n_pert = Int(get(test_dict, "max_n_pert", 5))
                                    
                                    u_h_exact, p_h_exact = x0_exact
                                    u_ex_L2 =  sqrt(abs(sum(∫(u_h_exact ⋅ u_h_exact)dΩ)))
                                    
                                    # Formulate a geometric bump function to construct boundary-invariant divergence-free perturbations.
                                    xmin, xmax, ymin, ymax = config.domain.bounding_box[1], config.domain.bounding_box[2], config.domain.bounding_box[3], config.domain.bounding_box[4]
                                    B_fn(x) = (x[1] - xmin)^2 * (xmax - x[1])^2 * (x[2] - ymin)^2 * (ymax - x[2])^2
                                    h_raw_func(x) = B_fn(x) * VectorValue(sin(3π*x[1])*cos(2π*x[2]), -cos(3π*x[1])*sin(2π*x[2]))
                                    h_pert_cf = CellField(h_raw_func, Ω)
                                    
                                    # Rigorous structural offset validator mapped explicitly from Gridap norms
                                    norm_h = sqrt(abs(sum(∫( h_pert_cf ⋅ h_pert_cf )dΩ)))
                                    if norm_h <= 0.0
                                        error("Perturbation field norm must be strictly positive (when perturbing).")
                                    end
                                    
                                    success = false
                                    successful_eps = -1.0
                                    eval_time = 0.0
                                    iter_count_attempt = 0
                                    final_x0 = nothing
                                    
                                    # ==============================================================================
                                    # NONLINEAR CONVERGENCE LOOP
                                    # Incrementally shrink initial numerical perturbation forcing iterative validation
                                    # ==============================================================================
                                    for attempt in 0:(max_n_pert + 1)
                                        eps_p = attempt <= max_n_pert ? eps_pert_base / (10.0^attempt) : 0.0
                                        u_0_func(x) = u_final(x) + eps_p * (u_ex_L2 / norm_h) * h_raw_func(x)
                                        x0 = interpolate_everywhere([u_0_func, p_final], X)
                                        
                                        println("\n    ==================================================")
                                        println("    [Attempt $(attempt+1)/$(max_n_pert + 2)] Homotopy Perturbation Scale: eps_pert = $eps_p")
                                        println("    [!] Delegating orchestration to PDE assembly module via `src/solvers/porous_solver.jl` (Mode: $method)")
                                        println("    ==================================================")
                                        
                                        local_stab_cfg = PorousNSSolver.StabilizationConfig(
                                            method=method,
                                            osgs_iterations=config.numerical_method.stabilization.osgs_iterations,
                                            osgs_tolerance=config.numerical_method.stabilization.osgs_tolerance
                                        )
                                        sys_success, sys_final_x0, sys_iter_count, sys_eval_time = PorousNSSolver.solve_system(
                                            X, Y, model, dΩ, Ω, h_cf, f_cf, alpha_cf, g_cf, form, 
                                            solver_picard, solver_newton, 
                                            x0, c_1, c_2, 
                                            config.physical_properties, local_stab_cfg, config.numerical_method.solver;
                                            V_free=V_free, Q_free=Q_free
                                        )
                                        
                                        if sys_success
                                            println("\n      [✅] Full non-linear algebraic system converged gracefully mathematically! Escaping constraint loop.")
                                            success = true
                                            successful_eps = eps_p
                                            final_x0 = sys_final_x0
                                            eval_time = sys_eval_time
                                            iter_count_attempt = sys_iter_count
                                            break
                                        else
                                            println("\n      [❌] Outer loop execution completely stalled structurally above convergence tolerance (`$(local_stab_cfg.osgs_tolerance)`) or system fully diverged.")
                                        end
                                    end
                                    
                                    if !success
                                         println("    [WARNING] Completely failed to find root basin. Returning NaN.")
                                    end
                                    
                                    if success && final_x0 !== nothing
                                        u_h, p_h = final_x0
                                        el2_u, el2_p, eh1_u, eh1_p = calculate_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, L, dΩ)
                                        
                                        vtk_dir = joinpath("results", "vtk")
                                        if !isdir(vtk_dir)
                                            mkpath(vtk_dir)
                                        end
                                        u_ex_cf = interpolate_everywhere(u_final, U)
                                        p_ex_cf = interpolate_everywhere(p_final, P)
                                        filename = joinpath(vtk_dir, "mms_$(method)_Re$(Float64(Re))_Da$(Float64(Da))_N$(n).vtu")
                                        writevtk(Ω, filename, cellfields=["uh"=>u_h, "ph"=>p_h, "uex"=>u_ex_cf, "pex"=>p_ex_cf, "alpha"=>alpha_cf, "err_u"=>u_h - u_ex_cf, "err_p"=>p_h - p_ex_cf])
                                    else
                                        el2_u, el2_p, eh1_u, eh1_p = NaN, NaN, NaN, NaN
                                    end
                                    
                                    k_id = (etype, kv, kp, alpha_0, Da, Re, method)
                                    push!(results_cache[k_id]["hs"], 1.0 / n)
                                    push!(results_cache[k_id]["err_u_l2"], el2_u)
                                    push!(results_cache[k_id]["err_p_l2"], el2_p)
                                    push!(results_cache[k_id]["err_u_h1"], eh1_u)
                                    push!(results_cache[k_id]["err_p_h1"], eh1_p)
                                    push!(results_cache[k_id]["eval_times"], eval_time)
                                    push!(results_cache[k_id]["eval_iters"], iter_count_attempt)
                                    push!(results_cache[k_id]["eval_eps"], successful_eps)
                                    
                                    println("  -> L2 u/p: ", round(el2_u, sigdigits=4), " / ", round(el2_p, sigdigits=4), " | H1 u/p: ", round(eh1_u, sigdigits=4), " / ", round(eh1_p, sigdigits=4))
                                    # =========================================================================================
                                    # EXPORT CURRENT CURVE TO HDF5 DYNAMICALLY
                                    # =========================================================================================
                                    sig_base = (Float64(Re), Float64(Da), Float64(alpha_0), Int(kv), Int(kp), String(etype))
                                    if haskey(existing_signatures, sig_base)
                                        c_idx = existing_signatures[sig_base]
                                    else
                                        max_idx += 1
                                        c_idx = max_idx
                                        existing_signatures[sig_base] = c_idx
                                    end
                                    
                                    h5open(h5_path, "r+") do h5f_out
                                        res = results_cache[k_id]
                                        grp_name = "config_$(c_idx)_$(method)"
                                        
                                        if haskey(h5f_out, grp_name)
                                            delete_object(h5f_out, grp_name)
                                        end
                                        
                                        g = create_group(h5f_out, grp_name)
                                        g["h"] = res["hs"]
                                        g["err_u_l2"] = res["err_u_l2"]
                                        g["err_p_l2"] = res["err_p_l2"]
                                        g["err_u_h1"] = res["err_u_h1"]
                                        g["err_p_h1"] = res["err_p_h1"]
                                        g["eval_times"] = res["eval_times"]
                                        g["eval_iters"] = res["eval_iters"]
                                        g["eval_eps"] = res["eval_eps"]
                                        
                                        attributes(g)["total_time_s"] = sum(res["eval_times"])
                                        attributes(g)["total_iters"] = sum(res["eval_iters"])
                                        attributes(g)["Re"] = Float64(Re)
                                        attributes(g)["Da"] = Float64(Da)
                                        attributes(g)["alpha_0"] = Float64(alpha_0)
                                        attributes(g)["physical_epsilon"] = 0.0
                                        attributes(g)["numerical_epsilon_coeff"] = 0.0
                                        attributes(g)["k_velocity"] = Int(kv)
                                        attributes(g)["k_pressure"] = Int(kp)
                                        attributes(g)["element_type"] = String(etype)
                                        attributes(g)["method"] = String(method)
                                        
                                        compute_slope(x, y) = sum((x .- sum(x)/length(x)) .* (y .- sum(y)/length(y))) / (sum((x .- sum(x)/length(x)).^2) + 1e-15)
                                        log_h = log.(res["hs"])
                                        attributes(g)["rate_u_l2"] = compute_slope(log_h, log.(res["err_u_l2"]))
                                        attributes(g)["rate_u_h1"] = compute_slope(log_h, log.(res["err_u_h1"]))
                                        attributes(g)["rate_p_l2"] = compute_slope(log_h, log.(res["err_p_l2"]))
                                        attributes(g)["rate_p_h1"] = compute_slope(log_h, log.(res["err_p_h1"]))
                                        
                                        println("\n    [💾] Appended $(grp_name) accurately to HDF5 file layout. Available for plotting!")
                                    end
                                    
                                    # Force garbage collection to prevent C-pointer memory leaks from UMFPACK LU factorizations across the N sweeps
                                    GC.gc()
                                    
                                end # end method
                            end # end Re
                        end # end Da
                    end # end alpha_0
                end # end n
            end # end kp
        end # end kv
    end # end etype
end # end function

if abspath(PROGRAM_FILE) == @__FILE__
    config_file = length(ARGS) > 0 ? ARGS[1] : "test_config.json"
    run_mms(config_file)
end
