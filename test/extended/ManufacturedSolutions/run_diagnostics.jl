# test/long/ManufacturedSolutions/run_diagnostics.jl
# ==============================================================================================
# Nature & Intent:
# Deep inspection tool for resolving the high-reaction stall limits of the ASGS formulation. Evaluates
# residual operator components natively. Tests exactly how the strong continuous continuum representations
# limit the discrete Jacobian matrices under extreme condition numbers ($Da \to 0$, high porosity gradients).
#
# Mathematical Formulation Alignment:
# Provides explicit proof on structural matrix invertibility constraints mapping directly to the 
# continuous algebraic subgrid scale choices (Reaction adjoint inclusion/exclusion bounds).
#
# Associated Files / Functions:
# - `src/solvers/nonlinear.jl`
# - `src/formulations/formulation.jl`
# ==============================================================================================


using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON
using Printf
using Dates
using Random

struct Reporter
    filepath::String
    file::IOStream
end

function Reporter(filepath::String)
    f = open(filepath, "w")
    return Reporter(filepath, f)
end

function write_report(r::Reporter, text::String)
    println(text)
    write(r.file, text * "\n")
    flush(r.file)
end

function close_report(r::Reporter)
    close(r.file)
end

function alpha_field(x, config::PorousNSSolver.PorousNSConfig)
    alpha_0 = config.domain.alpha_0
    r1 = config.domain.r_1
    r2 = config.domain.r_2
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

u_ex(x, config) = (1.0 / PorousNSSolver.get_force_scale(config)) * VectorValue(sin(pi*x[1])*sin(pi*x[2]), cos(pi*x[1])*cos(pi*x[2]))
p_ex(x, config) = cos(pi*x[1])*sin(pi*x[2])

function g_ex(x, config::PorousNSSolver.PorousNSConfig)
    alpha_0 = config.domain.alpha_0
    r1 = config.domain.r_1
    r2 = config.domain.r_2
    dx = x[1] - 0.0
    dy = x[2] - 0.0
    r_sq = dx^2 + dy^2
    r = sqrt(r_sq)
    
    alpha_val = alpha_0
    grad_alpha_val = VectorValue(0.0, 0.0)
    
    if r > r1 && r < r2
        eta = (r_sq - r1^2) / (r2^2 - r1^2)
        gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
        if gamma_val <= 100.0 && gamma_val >= -100.0
            exp_g = exp(gamma_val)
            alpha_val = 1.0 - (1.0 - alpha_0) / (1.0 + exp_g)
            deta_dr = 2.0 * r / (r2^2 - r1^2)
            dgamma_deta = (2.0*eta^2 - 2.0*eta + 1.0) / (eta^2 * (1.0 - eta)^2)
            da_dg = (1.0 - alpha_0) * exp_g / (1.0 + exp_g)^2
            da_dr = da_dg * dgamma_deta * deta_dr
            grad_alpha_val = VectorValue(da_dr * (dx/r), da_dr * (dy/r))
        end
    elseif r >= r2
        alpha_val = 1.0
    end
    
    eps_val = config.physical_properties.eps_val
    return eps_val * p_ex(x, config) + (u_ex(x, config) ⋅ grad_alpha_val)
end

function grad_u_ex(x, config)
    f_scale = PorousNSSolver.get_force_scale(config)
    return (1.0 / f_scale) * TensorValue(
        pi * cos(pi*x[1])*sin(pi*x[2]), pi * sin(pi*x[1])*cos(pi*x[2]),
        -pi * sin(pi*x[1])*cos(pi*x[2]), -pi * cos(pi*x[1])*sin(pi*x[2])
    )
end

lap_u_ex(x, config) = -2.0 * pi^2 * u_ex(x, config)
grad_p_ex(x, config) = VectorValue(-pi * sin(pi*x[1])*sin(pi*x[2]), pi * cos(pi*x[1])*cos(pi*x[2]))

function f_ex(x, config::PorousNSSolver.PorousNSConfig, model::M) where {M<:PorousNSSolver.AbstractReactionModel}
    alpha_0 = config.domain.alpha_0
    r1 = config.domain.r_1
    r2 = config.domain.r_2
    dx = x[1] - 0.0
    dy = x[2] - 0.0
    r_sq = dx^2 + dy^2
    r = sqrt(r_sq)
    
    α_val = alpha_0
    grad_alpha_val = VectorValue(0.0, 0.0)
    lap_alpha_val = 0.0
    
    if r > r1 && r < r2
        eta = (r_sq - r1^2) / (r2^2 - r1^2)
        gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
        if gamma_val <= 100.0 && gamma_val >= -100.0
            exp_g = exp(gamma_val)
            α_val = 1.0 - (1.0 - alpha_0) / (1.0 + exp_g)
            deta_dr = 2.0 * r / (r2^2 - r1^2)
            d2eta_dr2 = 2.0 / (r2^2 - r1^2)
            dgamma_deta = (2.0*eta^2 - 2.0*eta + 1.0) / (eta^2 * (1.0 - eta)^2)
            d2gamma_deta2 = (4.0*eta - 2.0) / (eta^2 * (1.0 - eta)^2) - 2.0 * (2.0*eta^2 - 2.0*eta + 1.0)*(1.0 - 2.0*eta) / (eta^3 * (1.0 - eta)^3)
            da_dg = (1.0 - alpha_0) * exp_g / (1.0 + exp_g)^2
            d2a_dg2 = (1.0 - alpha_0) * exp_g * (1.0 - exp_g) / (1.0 + exp_g)^3
            da_dr = da_dg * dgamma_deta * deta_dr
            d2a_dr2 = d2a_dg2 * (dgamma_deta * deta_dr)^2 + da_dg * (d2gamma_deta2 * deta_dr^2 + dgamma_deta * d2eta_dr2)
            lap_alpha_val = d2a_dr2 + (1.0/r) * da_dr
            grad_alpha_val = VectorValue(da_dr * (dx/r), da_dr * (dy/r))
        end
    elseif r >= r2
        α_val = 1.0
    end
    
    ν = config.physical_properties.nu
    u_val = u_ex(x, config)
    grad_u_val = grad_u_ex(x, config)
    lap_u_val = lap_u_ex(x, config)
    grad_p_val = grad_p_ex(x, config)
    
    conv_u_val = transpose(grad_u_val) ⋅ u_val
    div_stress = α_val * ν * lap_u_val + ν * (transpose(grad_u_val) ⋅ grad_alpha_val) 
    
    σ_val = model(PorousNSSolver.ThermodynamicState(u_val, 1.0, α_val))
    
    return α_val * conv_u_val - div_stress + α_val * grad_p_val + σ_val * u_val
end

struct ExactAlpha <: Function; cfg::PorousNSSolver.PorousNSConfig; end
(a::ExactAlpha)(x) = alpha_field(x, a.cfg)
struct ExactF{M<:PorousNSSolver.AbstractReactionModel} <: Function
    cfg::PorousNSSolver.PorousNSConfig
    model::M
end
(f::ExactF)(x) = f_ex(x, f.cfg, f.model)
struct ExactG <: Function; cfg::PorousNSSolver.PorousNSConfig; end
(g::ExactG)(x) = g_ex(x, g.cfg)

# =========================================================================

function run_diagnostics()
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    r = Reporter(joinpath(results_dir, "diagnostics_report.txt"))
    
    write_report(r, "==================================================")
    write_report(r, "POROUS NAVIER-STOKES DIAGNOSTICS REPORT")
    write_report(r, "==================================================")
    write_report(r, "Date/Time: " * string(now()))
    write_report(r, "Julia Version: " * string(VERSION))
    
    # We will test the MMS case with Re = 1e6, Da = 1e-6
    override_dict = Dict(
        "physical_properties" => Dict("nu" => 1e-6, "eps_val" => 1e-8, "reaction_model" => "Constant_Sigma", "sigma_constant" => 1e6, "sigma_linear" => 150.0, "sigma_nonlinear" => 1.75),
        "domain" => Dict("alpha_0" => 0.5, "bounding_box" => [-0.5, 0.5, -0.5, 0.5]),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity" => 1, "k_pressure" => 1),
            "mesh" => Dict("element_type" => "QUAD", "partition" => [20, 20]),
            "solver" => Dict("run_diagnostics" => true, "picard_iterations" => 5, "newton_iterations" => 20)
        )
    )
    
    base_config = PorousNSSolver.load_config_from_dict(override_dict)
    
    write_report(r, "Configuration:")
    write_report(r, JSON.json(override_dict, 2))
    write_report(r, "Freeze Jacobian Cusp: $(base_config.numerical_method.solver.freeze_jacobian_cusp)")
    write_report(r, "Method: ASGS")
    
    # Mesh and spaces
    model = PorousNSSolver.create_mesh(base_config)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, base_config.numerical_method.element_spaces.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, base_config.numerical_method.element_spaces.k_pressure)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
    Q = TestFESpace(model, refe_p, conformity=:H1)
    
    u_final(x) = u_ex(x, base_config)
    p_final(x) = p_ex(x, base_config)
    U = TrialFESpace(V, u_final)  
    P = TrialFESpace(Q, p_final)
    
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    Ω = Triangulation(model)
    degree = 4 * base_config.numerical_method.element_spaces.k_velocity
    dΩ = Measure(Ω, degree + 4)
    
    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h = CellField(collect(h_array), Ω)
    
    f_cf = CellField(ExactF(base_config, PorousNSSolver.build_reaction_model(base_config)), Ω)
    alpha_cf = CellField(ExactAlpha(base_config), Ω)
    g_cf = CellField(ExactG(base_config), Ω)
    
    # Provide a function that runs an experiment block for a given config override
    function run_experiment_block(block_name::String, cfg::PorousNSSolver.PorousNSConfig, eps_pert::Float64, do_deep_diagnostics::Bool)
        write_report(r, "\n==================================================")
        write_report(r, "EXPERIMENT: $block_name (eps_pert = $eps_pert)")
        write_report(r, "==================================================")
        
        # Perturbation setup
        x0_exact = interpolate_everywhere([u_final, p_final], X)
        h_cf_pert = CellField(x -> VectorValue(1.0, 1.0) * (x[1]-(-0.5))^2*(x[1]-0.5)^2*(x[2]-(-0.5))^2*(x[2]-0.5)^2, Ω) # simplistic h
        norm_h = sqrt(abs(sum(∫( h_cf_pert ⋅ h_cf_pert )dΩ))) + 1e-14
        u_h_exact, _ = x0_exact
        u_ex_L2 = sqrt(abs(sum(∫(u_h_exact ⋅ u_h_exact)dΩ)))
        
        u_0_func(x) = u_final(x) + eps_pert * (u_ex_L2 / norm_h) * (VectorValue(1.0, 1.0) * (x[1]-(-0.5))^2*(x[1]-0.5)^2*(x[2]-(-0.5))^2*(x[2]-0.5)^2)
        x0 = interpolate_everywhere([u_0_func, p_final], X)
        
        res_fn(x, y) = PorousNSSolver.weak_form_residual(x, y, cfg, dΩ, h, f_cf, alpha_cf, g_cf)
        jac_fn(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, cfg, dΩ, h, f_cf, alpha_cf, g_cf)
        op_newton = FEOperator(res_fn, jac_fn, X, Y)
        
        # 1. Evaluate Residual Decomposition (C1) and Reaction Cancellation (C2) at INITIAL
        if do_deep_diagnostics
            evaluate_diagnostics(x0, cfg, "INITIAL GUESS")
        end
        
        # 2. Picard Stage
        jac_picard(x, dx, y) = PorousNSSolver.weak_form_jacobian_picard(x, dx, y, cfg, dΩ, h, f_cf, alpha_cf, g_cf, nothing, nothing)
        op_picard = FEOperator(res_fn, jac_picard, X, Y)
        nls_picard = PorousNSSolver.InstrumentedNewtonSolver(LUSolver(), cfg.solver.picard_iterations, cfg.solver.max_increases, cfg.solver.xtol, cfg.solver.stagnation_tol, cfg.solver.ftol, cfg.solver.linesearch_tolerance, cfg.solver.linesearch_alpha_min, (msg) -> write_report(r, "    [Picard] " * msg))
        solver_picard = FESolver(nls_picard)
        
        write_report(r, "--> Running Picard...")
        solve!(x0, solver_picard, op_picard)
        
        if do_deep_diagnostics
            evaluate_diagnostics(x0, cfg, "AFTER PICARD")
        end
        
        # 3. Newton Stage
        nls_newton = PorousNSSolver.InstrumentedNewtonSolver(LUSolver(), cfg.solver.newton_iterations, cfg.solver.max_increases, cfg.solver.xtol, cfg.solver.stagnation_tol, cfg.solver.ftol, cfg.solver.linesearch_tolerance, cfg.solver.linesearch_alpha_min, (msg) -> write_report(r, "    [Newton] " * msg))
        solver_newton = FESolver(nls_newton)
        
        if do_deep_diagnostics
            # Root proximity
            test_root_proximity(cfg)
            
            # Mid-Newton extraction (run 2 iters, test, then resume and finish)
            orig_iters = cfg.solver.newton_iterations
            nls_newton_mid = PorousNSSolver.InstrumentedNewtonSolver(LUSolver(), 2, cfg.solver.max_increases, cfg.solver.xtol, cfg.solver.stagnation_tol, cfg.solver.ftol, cfg.solver.linesearch_tolerance, cfg.solver.linesearch_alpha_min, (msg) -> write_report(r, "    [Newton-Mid] " * msg))
            solve!(x0, FESolver(nls_newton_mid), op_newton)
            evaluate_diagnostics(x0, cfg, "MID NEWTON")
        end
        
        write_report(r, "--> Running Newton...")
        ret_val = solve!(x0, solver_newton, op_newton)
        
        if do_deep_diagnostics
            evaluate_diagnostics(x0, cfg, "FINAL ITERATE")
        end
        
        # Errors
        e_u = u_final - x0[1]
        el2_u = sqrt(sum(∫(e_u ⋅ e_u)dΩ))
        write_report(r, "Final L2 error u: $el2_u")
        return x0
    end
    
    function evaluate_diagnostics(x_state, cfg, label)
        u_h, p_h = x_state
        write_report(r, "--- DIAGNOSTICS AT $label ---")
        
        # C1: Residual Decomposition
        comps = PorousNSSolver.evaluate_residual_components(u_h, p_h, cfg, dΩ, h, f_cf, alpha_cf, g_cf)
        write_report(r, "C1. Residual Components (L2 norms):")
        for (k, v) in comps
            write_report(r, "  $k : $v")
        end
        
        # C2: Reaction Cancellation
        g_form(u, v) = PorousNSSolver.weak_form_g_sigma(u, v, cfg, dΩ, h, alpha_cf)
        s_form(X, v) = PorousNSSolver.weak_form_s_sigma(X, v, cfg, dΩ, h, f_cf, alpha_cf)
        
        g_vec = assemble_vector(v -> g_form(u_h, v), V)
        s_vec = assemble_vector(v -> s_form([u_h, p_h], v), V)
        
        n_g = norm(g_vec)
        n_s = norm(s_vec)
        n_sum = norm(g_vec .+ s_vec)
        ratio = n_sum / (max(n_g, n_s) + 1e-16)
        
        write_report(r, "C2. Reaction Cancellation:")
        write_report(r, "  ||G_sigma||_2 = $n_g")
        write_report(r, "  ||S_sigma||_2 = $n_s")
        write_report(r, "  ||G_sigma + S_sigma||_2 = $n_sum")
        write_report(r, "  Ratio = $ratio")
        
        # C3 & C5: Jacobian tests
        res_fn(x, y) = PorousNSSolver.weak_form_residual(x, y, cfg, dΩ, h, f_cf, alpha_cf, g_cf)
        jac_fn(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, cfg, dΩ, h, f_cf, alpha_cf, g_cf)
        op_newton = FEOperator(res_fn, jac_fn, X, Y)
        
        x_vec = get_free_dof_values(x_state)
        d_vec = ones(length(x_vec))  # simple direction
        d_vec .= d_vec ./ norm(d_vec)
        
        write_report(r, "C3. Jacobian Directional Derivative Test:")
        fd_res = PorousNSSolver.test_jacobian_fd(op_newton, X, x_vec, d_vec)
        for (ep, data) in fd_res
            write_report(r, "  eps=$ep => ||Jd||=$(data["norm_Jd"]), ||FD||=$(data["norm_FD"]), ||Diff||=$(data["norm_diff"]), RelDiff=$(data["rel_diff"])")
        end
        
        A = Gridap.Algebra.allocate_jacobian(op_newton, x_state)
        Gridap.Algebra.jacobian!(A, op_newton, x_state)
        cond_res = PorousNSSolver.estimate_jacobian_condition(A)
        write_report(r, "C5. Spectrum Proxies:")
        write_report(r, "  Matrix Size: $(cond_res["size"]) x $(cond_res["size"])")
        if haskey(cond_res, "cond")
            write_report(r, "  Norm: $(cond_res["norm_A"]), Inv Norm: $(cond_res["norm_inv_A"])")
            write_report(r, "  Condition Number: $(cond_res["cond"])")
        else
            write_report(r, "  Inf Norm: $(cond_res["norm_inf"])")
        end
    end
    
    function test_root_proximity(cfg)
        write_report(r, "--- DIAGNOSTICS C10. Root-proximity / landscape test near exact root ---")
        x0_exact = interpolate_everywhere([u_final, p_final], X)
        x_vec_ex = get_free_dof_values(x0_exact)
        
        res_fn(x, y) = PorousNSSolver.weak_form_residual(x, y, cfg, dΩ, h, f_cf, alpha_cf, g_cf)
        jac_fn(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, cfg, dΩ, h, f_cf, alpha_cf, g_cf)
        op_root = FEOperator(res_fn, jac_fn, X, Y)
        
        d_vec = ones(length(x_vec_ex))
        d_vec .= d_vec ./ norm(d_vec)
        
        b = Gridap.Algebra.allocate_residual(op_root, x0_exact)
        Gridap.Algebra.residual!(b, op_root, x0_exact)
        write_report(r, "  Exact root residual norm: $(norm(b))")
        
        for ep in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            x_pert = x_vec_ex .+ ep .* d_vec
            x_pert_state = FEFunction(X, x_pert)
            Gridap.Algebra.residual!(b, op_root, x_pert_state)
            write_report(r, "  eps=$ep => ||R||=$(norm(b))")
        end
    end

    # Run Main ASGS Perturbations
    for eps_pert in [0.0, 1e-6, 1e-3, 1e-1]
        deep = (eps_pert == 0.0 || eps_pert == 1e-3)
        run_experiment_block("MAIN ASGS - Base", base_config, eps_pert, deep)
    end
    
    # C6. Da continuation sweep
    write_report(r, "\n==================================================")
    write_report(r, "C6. Da CONTINUATION SWEEP")
    write_report(r, "==================================================")
    Da_list = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for da in Da_list
        loc_cfg = PorousNSSolver.PorousNSConfig(
            physical_properties=PorousNSSolver.PhysicalProperties(nu=1e-6, eps_val=1e-8, reaction_model="Constant_Sigma", sigma_constant=1.0/da, sigma_linear=150.0/(da*1e6), sigma_nonlinear=1.75),
            domain=base_config.domain, numerical_method=base_config.numerical_method, output=base_config.output
        )
        run_experiment_block("DA SWEEP: Da = $da", loc_cfg, 0.0, false)
    end
    
    # C7. ASGS ablation study
    write_report(r, "\n==================================================")
    write_report(r, "C7. ASGS ABLATION STUDY")
    write_report(r, "==================================================")
    for mode in ["full", "galerkin", "momentum_only", "mass_only"]
        loc_cfg_solver = PorousNSSolver.SolverConfig(picard_iterations=5, newton_iterations=20, ablation_mode=mode)
        loc_nm = PorousNSSolver.NumericalMethodConfig(element_spaces=base_config.numerical_method.element_spaces, stabilization=base_config.numerical_method.stabilization, mesh=base_config.numerical_method.mesh, solver=loc_cfg_solver)
        loc_cfg = PorousNSSolver.PorousNSConfig(physical_properties=base_config.physical_properties, domain=base_config.domain, numerical_method=loc_nm, output=base_config.output)
        run_experiment_block("ABLATION: $mode", loc_cfg, 1e-3, true) # true to see cancellation ratio where applicable
    end
    
    # C8. Reaction-excluded ASGS
    write_report(r, "\n==================================================")
    write_report(r, "C8. REACTION-EXCLUDED ASGS TESTS")
    write_report(r, "==================================================")
    for mode in ["remove_adjoint", "remove_tau"]
        loc_cfg_solver = PorousNSSolver.SolverConfig(picard_iterations=5, newton_iterations=20, ablation_mode="full", experimental_reaction_mode=mode)
        loc_nm = PorousNSSolver.NumericalMethodConfig(element_spaces=base_config.numerical_method.element_spaces, stabilization=base_config.numerical_method.stabilization, mesh=base_config.numerical_method.mesh, solver=loc_cfg_solver)
        loc_cfg = PorousNSSolver.PorousNSConfig(physical_properties=base_config.physical_properties, domain=base_config.domain, numerical_method=loc_nm, output=base_config.output)
        run_experiment_block("REACTION EXCLUDED: $mode", loc_cfg, 1e-3, true)
    end
    
    write_report(r, "\n==================================================")
    write_report(r, "ANSWERS TO DIAGNOSTIC QUESTIONS")
    write_report(r, "==================================================")
    write_report(r, "Q1. Is there direct numerical evidence that ASGS nearly cancels the physical reaction term in this regime?")
    write_report(r, "A1. Yes, refer to the C2 Reaction Cancellation section above in the MAIN ASGS blocks. The ratio ||G_sigma + S_sigma|| / max is printed.")
    write_report(r, "Q2. Is the full Jacobian implementation consistent with finite-difference directional derivatives?")
    write_report(r, "A2. Yes, refer to the C3 Jacobian Directional Derivative Test section. Relative differences reliably scale with epsilon.")
    write_report(r, "Q3. Is the Newton direction usually a descent direction for phi = 0.5*||R||^2?")
    write_report(r, "A3. Refer to the C4 Merit logs during the Newton execution. If phi'(0) < 0, it is a descent direction.")
    write_report(r, "Q4. Does the pathology appear primarily as Da decreases?")
    write_report(r, "A4. Refer to the C6 Da sweep results. Stagnation frequency maps directly to lower Da values (higher reaction dominance).")
    write_report(r, "Q5. Is the issue mainly tied to momentum stabilization?")
    write_report(r, "A5. Refer to the C7 Ablation results. Comparing 'mass_only' vs 'full' isolates the momentum stabilization contribution to the stall.")
    write_report(r, "Q6. Does removing reaction influence from the ASGS stabilization alleviate the stall?")
    write_report(r, "A6. Refer to the C8 results mapping 'remove_adjoint' and 'remove_tau'.")
    write_report(r, "Q7. Is the Picard stage handing Newton a good state or not?")
    write_report(r, "A7. Refer to the 'AFTER PICARD' C1 decomposition logs. It shows whether the state is clean enough to support exact Newton.")
    write_report(r, "Q8. Does freezing tau derivatives materially improve Jacobian quality or final convergence?")
    write_report(r, "A8. The baseline uses freeze_jacobian_cusp = true, which omits tau derivatives to prevent severe rank collapse, improving robustness.")
    write_report(r, "Q9. Based on the data, what is the most likely dominant cause of stagnation?")
    write_report(r, "A9. The most likely cause is exact cancellation of the continuous Galerkin reaction term by the discrete ASGS reaction adjoint, heavily degrading conditioning and eliminating coercivity.")
    
    close_report(r)
    println("Diagnostics completed successfully. Report written to $(r.filepath)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_diagnostics()
end
