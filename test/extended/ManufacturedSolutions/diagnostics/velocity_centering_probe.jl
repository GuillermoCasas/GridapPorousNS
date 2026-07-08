#!/usr/bin/env julia
# diagnostics/velocity_centering_probe.jl — Step 1a of the convergence-failure plan.
#
# Question: For config_18 (Re=1e6, Da=1e6, α=0.5, P1/P1 QUAD ASGS) the L²(u) error grows from
# N=160 (9.49e-5) to N=320 (1.07e-4) while H¹(u) keeps halving optimally. The harness's pressure
# error metric already centres p, so the L²(p) growth is NOT a pressure-mean drift.
# This probe checks whether the velocity error has a growing constant-spatial-mean component.
#
# Method: solve config_18 at N=80 (the last "good slope" mesh) and N=160 (where collapse starts)
# using the EXACT same harness machinery (build_cell mirrors run_test.jl). Compute:
#   - raw    L²(u) = ‖u_h - u_ex‖_L²
#   - mean   = ∫(u_h - u_ex) / |Ω|     (vector-valued)
#   - centered L²(u) = ‖u_h - u_ex - mean‖_L²
# If centered slope (N=80 → N=160) recovers to ≥ 1.7 while raw stays at ~1.3, the spurious
# constant-velocity-mode hypothesis is confirmed and the fix is in the formulation.
# If both stay sub-optimal, the cause is elsewhere (Step 1b is decisive).
#
# Cost: ~10-30 min per N (Re=1e6 is the slowest regime). Total ~30-60 min.
# Saves a markdown report at diagnostics/velocity_centering_probe.md.
#
# Run: cd test/extended/ManufacturedSolutions && julia --project=../../.. diagnostics/velocity_centering_probe.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using PorousNSSolver
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Printf
using Dates

# [harness-frame] Re/Da iteration-budget knobs (relocated out of production SolverConfig — audit §A.1/F1).
@isdefined(read_mms_dynamic_budget) || include(joinpath(@__DIR__, "..", "..", "harness_dynamic_budget.jl"))

# Mirror probe_stiff_diagnose.jl::build_cell — minimal version targeted at our cell.
function setup_cell(; Re::Float64, Da::Float64, alpha_0::Float64, n::Int,
                    kv::Int=1, element_type::String="QUAD")
    alpha_infty = 1.0
    U_amp = 1.0
    L = 1.0
    kp = kv

    config_dict = Dict(
        "physical_properties" => Dict("nu" => 1.0, "physical_epsilon" => 1e-8,
                                      "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
        "domain" => Dict("alpha_0" => alpha_0,
                         "bounding_box" => [-0.5, 0.5, -0.5, 0.5],
                         "r_1" => 0.2, "r_2" => 0.4),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
            "mesh" => Dict("element_type" => element_type, "partition" => [n, n]),
            "stabilization" => Dict("method" => "ASGS"),
            "solver" => Dict(),
        ),
    )
    config = PorousNSSolver.load_config_from_dict(config_dict)

    # Mesh + tags (mirrors run_test.jl _build_local_mesh)
    domain_tuple = Tuple(config.domain.bounding_box)
    partition = Tuple(config.numerical_method.mesh.partition)
    if config.numerical_method.mesh.element_type == "TRI"
        model = CartesianDiscreteModel(domain_tuple, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain_tuple, partition)
    end
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "all_boundaries", [1, 2, 3, 4, 5, 6, 7, 8])

    refe_u = ReferenceFE(lagrangian, VectorValue{2, Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
    Q = TestFESpace(model, refe_p, conformity=:H1)
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)

    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv, PorousNSSolver.ConstantSigmaLaw(0.0))
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree + 4)

    h_array = config.numerical_method.mesh.element_type == "TRI" ?
        lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω)) :
        lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h_cf = CellField(collect(h_array), Ω)

    Y = MultiFieldFESpace([V, Q])
    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, kv)
    tau_reg_lim = config.physical_properties.tau_regularization_limit

    alpha_field = PorousNSSolver.SmoothRadialPorosity(Float64(alpha_0), alpha_infty, config.domain.r_1, config.domain.r_2)
    alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)

    nu_calc = U_amp * L / Float64(Re)
    sigma_c = Float64(Da) * alpha_infty * nu_calc / L^2
    form = PorousNSSolver.PaperGeneralFormulation(
        PorousNSSolver.DeviatoricSymmetricViscosity(),
        PorousNSSolver.ConstantSigmaLaw(sigma_c),
        PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma(),
        PorousNSSolver.SmoothVelocityFloor(config.physical_properties.u_base_floor_ref, 0.0,
                                            config.physical_properties.epsilon_floor,
                                            config.physical_properties.velocity_magnitude_derivative_floor),
        nu_calc, config.physical_properties.physical_epsilon,
    )
    mms = PorousNSSolver.PaperMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)
    u_final = PorousNSSolver.get_u_ex(mms)
    p_final = PorousNSSolver.get_p_ex(mms)
    U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)

    U = TrialFESpace(V, u_final)
    P = TrialFESpace(Q, p_final)
    X = MultiFieldFESpace([U, P])

    f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)

    # Solver tolerances (mirror run_test.jl logic for k_v=1)
    h_scale = 1.0 / n
    spatial_err_est = h_scale^(kv + 1)
    budget = read_mms_dynamic_budget()   # [harness-frame] programmatic cell: inherits the defaults
    c_ceil = budget.ftol_ceiling
    c_sf   = budget.ftol_spatial_safety_factor
    dynamic_ftol = max(config.numerical_method.solver.ftol, min(c_ceil, c_sf * spatial_err_est))
    condition_scaling = Float64(n)^2 * max(1.0, Float64(Re))
    n_base = config.numerical_method.solver.condition_noise_floor_baseline
    n_min  = config.numerical_method.solver.condition_noise_floor_absolute_min
    n_sf   = config.numerical_method.solver.condition_noise_floor_safety_factor
    dynamic_noise_floor = min(config.numerical_method.solver.stagnation_noise_floor,
                              max(n_min, n_base * condition_scaling))
    dynamic_noise_floor = max(dynamic_noise_floor, dynamic_ftol * n_sf)
    freeze_cusp = config.numerical_method.solver.freeze_jacobian_cusp
    k_nf = config.numerical_method.solver.noise_floor_success_max_ftol_multiple

    nls_newton = PorousNSSolver.SafeNewtonSolver(
        LUSolver(), 150, config.numerical_method.solver.max_increases,
        config.numerical_method.solver.xtol, dynamic_ftol,
        config.numerical_method.solver.linesearch_alpha_min,
        config.numerical_method.solver.armijo_c1,
        config.numerical_method.solver.divergence_merit_factor,
        dynamic_noise_floor, config.numerical_method.solver.max_linesearch_iterations,
        config.numerical_method.solver.linesearch_contraction_factor;
        noise_floor_success_max_ftol_multiple=k_nf,
    )
    solver_newton = FESolver(nls_newton)

    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)

    res_fn(x, y)        = PorousNSSolver.build_stabilized_weak_form_residual(x, y, setup, formulation, config.physical_properties; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    jac_newton(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, config.physical_properties, freeze_cusp, PorousNSSolver.ExactNewtonMode(); pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    op_newton = FEOperator(res_fn, jac_newton, X, Y)

    x0_exact = interpolate_everywhere([u_final, p_final], X)

    return (config=config, setup=setup, formulation=formulation,
            X=X, Y=Y, Ω=Ω, dΩ=dΩ, op_newton=op_newton, solver_newton=solver_newton,
            x0_exact=x0_exact, u_final=u_final, p_final=p_final,
            U_c=U_c, P_c=P_c, L=L)
end

# Solve one cell and return (u_h, p_h, residual_info)
function solve_cell(cell)
    x = FEFunction(cell.X, copy(get_free_dof_values(cell.x0_exact)))
    x_backup = copy(get_free_dof_values(x))
    res = PorousNSSolver.safe_fe_solve!(x, cell.solver_newton, cell.op_newton; backup=x_backup)
    return x, res
end

# Compute (a) raw L²(u), (b) mean of (u_h - u_ex), (c) centered L²(u). Also report
# the harness's standard scalars for sanity.
function diagnose_errors(cell, x_solved)
    u_h, p_h = x_solved
    dΩ = cell.dΩ
    e_u_func = cell.u_final - u_h
    e_p_func = cell.p_final - p_h
    area = sum(∫(1.0)dΩ)
    # Vector-valued mean of e_u
    mean_e_u_x = sum(∫(e_u_func ⋅ VectorValue(1.0, 0.0))dΩ) / area
    mean_e_u_y = sum(∫(e_u_func ⋅ VectorValue(0.0, 1.0))dΩ) / area
    mean_e_p   = sum(∫(e_p_func)dΩ) / area
    e_u_centered = e_u_func - VectorValue(mean_e_u_x, mean_e_u_y)
    e_p_centered = e_p_func - mean_e_p
    raw_L2u      = sqrt(sum(∫(e_u_func ⋅ e_u_func)dΩ)) / cell.U_c
    centered_L2u = sqrt(sum(∫(e_u_centered ⋅ e_u_centered)dΩ)) / cell.U_c
    raw_L2p      = sqrt(sum(∫(e_p_func * e_p_func)dΩ)) / cell.P_c
    centered_L2p = sqrt(sum(∫(e_p_centered * e_p_centered)dΩ)) / cell.P_c
    H1u          = sqrt(sum(∫(∇(e_u_func) ⊙ ∇(e_u_func))dΩ)) / (cell.U_c / cell.L)
    return (raw_L2u=raw_L2u, centered_L2u=centered_L2u,
            raw_L2p=raw_L2p, centered_L2p=centered_L2p,
            H1u=H1u, mean_e_u=(mean_e_u_x, mean_e_u_y), mean_e_p=mean_e_p)
end

function main()
    Ns = [80, 160]
    println("============================================================")
    println("Velocity-centering probe for config_18 (Re=1e6, Da=1e6, α=0.5)")
    println("Run start: ", now())
    println("============================================================\n")
    results = Dict{Int, NamedTuple}()
    for n in Ns
        t0 = time()
        println("\n>>> Building + solving N=$n ...")
        cell = setup_cell(Re=1.0e6, Da=1.0e6, alpha_0=0.5, n=n, kv=1, element_type="QUAD")
        x_solved, res = solve_cell(cell)
        @printf("    Solver state: %s | iters=%d | final ‖R‖=%.4e\n",
                res.state, res.iterations, res.residual_norm)
        if res.state != :ok
            @warn "Solver did not converge cleanly at N=$n; proceeding with current iterate."
        end
        diag = diagnose_errors(cell, x_solved)
        @printf("    raw      L²(u) = %.6e   (harness-standard)\n", diag.raw_L2u)
        @printf("    centered L²(u) = %.6e   (after subtracting mean velocity error)\n", diag.centered_L2u)
        @printf("    mean(e_u_x)    = %.6e   mean(e_u_y) = %.6e\n", diag.mean_e_u[1], diag.mean_e_u[2])
        @printf("    raw      L²(p) = %.6e\n", diag.raw_L2p)
        @printf("    centered L²(p) = %.6e   (harness reports this)\n", diag.centered_L2p)
        @printf("    mean(e_p)      = %.6e\n", diag.mean_e_p)
        @printf("    H¹(u)          = %.6e\n", diag.H1u)
        @printf("    elapsed        = %.1fs\n", time() - t0)
        results[n] = diag
    end

    println("\n============================================================")
    println("Slope comparison (N=80 → N=160):")
    if length(Ns) >= 2
        a, b = results[Ns[1]], results[Ns[2]]
        slope_raw = log(a.raw_L2u / b.raw_L2u) / log(2.0)
        slope_cen = log(a.centered_L2u / b.centered_L2u) / log(2.0)
        @printf("  raw      L²(u) slope = %+.3f  (expected ≈ 2; HDF5 showed 1.34 at this pair)\n", slope_raw)
        @printf("  centered L²(u) slope = %+.3f\n", slope_cen)
        println()
        if slope_cen > slope_raw + 0.5
            println("  >>> Centering RECOVERS the slope: spurious constant-velocity-mode hypothesis CONFIRMED.")
            println("      Next: production change to center velocity error in calculate_normalized_errors,")
            println("            or root-cause the formulation term that injects the mode.")
        elseif slope_cen < slope_raw + 0.2
            println("  >>> Centering does NOT recover the slope: constant-mode hypothesis FALSIFIED.")
            println("      Next: Step 1b (forcing accuracy) and/or stabilization-scaling investigation.")
        else
            println("  >>> Partial improvement: mode contribution is present but not dominant.")
            println("      Constant-velocity-mode is PART of the story but more refinement points needed.")
        end
    end
    println("============================================================")
end

isinteractive() || main()
