#!/usr/bin/env julia
# diagnostics/jacobian_equilibration_osgs_probe.jl
#
# Goal: scout whether picking different DIMENSIONAL parameters (U_amp, L) for the SAME
# dimensionless regime (Re, Da, α) changes OSGS convergence behaviour. The harness
# currently hardcodes U_amp=L=1, which produces extreme dimensional ν, σ at high-Br cells.
# Centered encoding sets U_amp=Re/√(α·Da), L=√(α·Da) so ν_dim=σ_dim=1 exactly.
#
# Test cell: C16 OSGS (Re=1e-6, Da=1e6, α=0.5).
# Encoding A (default):  U_amp=1, L=1            → ν=1e6, σ=5e11
# Encoding B (centered): U_amp=√2·1e-9, L=√(5e5) → ν=σ=1
#
# Run: cd test/extended/ManufacturedSolutions && julia --project=../../.. diagnostics/jacobian_equilibration_osgs_probe.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using PorousNSSolver
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using LinearAlgebra
using Printf
using Dates

# [harness-frame] Re/Da iteration-budget knobs (relocated out of production SolverConfig — audit §A.1/F1).
@isdefined(read_mms_dynamic_budget) || include(joinpath(@__DIR__, "..", "..", "harness_dynamic_budget.jl"))

function setup_cell(; Re::Float64, Da::Float64, alpha_0::Float64, n::Int,
                    U_amp::Float64, L::Float64,
                    kv::Int=1, element_type::String="QUAD")
    alpha_infty = 1.0
    kp = kv

    bb = [-L/2, L/2, -L/2, L/2]
    config_dict = Dict(
        "physical_properties" => Dict("nu" => 1.0, "eps_val" => 1e-8,
                                      "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
        "domain" => Dict("alpha_0" => alpha_0,
                         "bounding_box" => bb,
                         "r_1" => 0.2 * L, "r_2" => 0.4 * L),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
            "mesh" => Dict("element_type" => element_type, "partition" => [n, n]),
            "stabilization" => Dict("method" => "OSGS"),
            "solver" => Dict("newton_iterations" => 150),
        ),
    )
    config = PorousNSSolver.load_config_from_dict(config_dict)

    domain_tuple = Tuple(config.domain.bounding_box)
    partition = Tuple(config.numerical_method.mesh.partition)
    model = CartesianDiscreteModel(domain_tuple, partition)
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

    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
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
                                            config.physical_properties.epsilon_floor),
        nu_calc, config.physical_properties.eps_val,
    )
    mms = PorousNSSolver.Paper2DMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)
    u_final = PorousNSSolver.get_u_ex(mms)
    p_final = PorousNSSolver.get_p_ex(mms)
    U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)

    U = TrialFESpace(V, u_final)
    P = TrialFESpace(Q, p_final)
    X = MultiFieldFESpace([U, P])

    f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)

    h_scale = L / n
    spatial_err_est = (h_scale)^(kv + 1)
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
    k_nf = config.numerical_method.solver.noise_floor_success_max_ftol_multiple
    nit_p = config.numerical_method.solver.picard_iterations
    nit_n = config.numerical_method.solver.newton_iterations

    nls_picard = PorousNSSolver.SafeNewtonSolver(
        LUSolver(), nit_p, config.numerical_method.solver.max_increases,
        config.numerical_method.solver.xtol, config.numerical_method.solver.picard_ftol,
        config.numerical_method.solver.linesearch_alpha_min,
        config.numerical_method.solver.armijo_c1,
        config.numerical_method.solver.divergence_merit_factor,
        dynamic_noise_floor, config.numerical_method.solver.max_linesearch_iterations,
        config.numerical_method.solver.linesearch_contraction_factor;
        mode=:picard, noise_floor_success_max_ftol_multiple=k_nf,
    )
    nls_newton = PorousNSSolver.SafeNewtonSolver(
        LUSolver(), nit_n, config.numerical_method.solver.max_increases,
        config.numerical_method.solver.xtol, dynamic_ftol,
        config.numerical_method.solver.linesearch_alpha_min,
        config.numerical_method.solver.armijo_c1,
        config.numerical_method.solver.divergence_merit_factor,
        dynamic_noise_floor, config.numerical_method.solver.max_linesearch_iterations,
        config.numerical_method.solver.linesearch_contraction_factor;
        noise_floor_success_max_ftol_multiple=k_nf,
    )
    fe_picard = FESolver(nls_picard)
    fe_newton = FESolver(nls_newton)
    iter_solvers = PorousNSSolver.StageSolvers(fe_picard, fe_newton)

    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)

    x0 = interpolate_everywhere([u_final, p_final], X)

    return (config=config, setup=setup, formulation=formulation, iter_solvers=iter_solvers,
            X=X, Y=Y, Ω=Ω, dΩ=dΩ, x0=x0, u_final=u_final, p_final=p_final,
            U_c=U_c, P_c=P_c, L=L, U_amp=U_amp, nu_dim=nu_calc, sigma_dim=sigma_c)
end

function solve_cell(cell)
    diag_cache = Dict{String, Any}()
    success, mms_plateau_success, final_x0, iters, eval_time =
        PorousNSSolver.solve_system(cell.setup, cell.formulation, cell.iter_solvers,
                                     cell.config, cell.x0; diagnostics_cache=diag_cache)
    final_res = get(diag_cache, "final_residual_norm", NaN)
    return (success=success, x_solved=final_x0, iters=iters, residual=final_res,
            elapsed=eval_time, diag=diag_cache)
end

function diagnose_errors(cell, x_solved)
    u_h, p_h = x_solved
    dΩ = cell.dΩ
    e_u_func = cell.u_final - u_h
    e_p_func = cell.p_final - p_h
    area = sum(∫(1.0)dΩ)
    mean_e_p = sum(∫(e_p_func)dΩ) / area
    e_p_centered = e_p_func - mean_e_p
    L2u  = sqrt(sum(∫(e_u_func ⋅ e_u_func)dΩ)) / cell.U_c
    L2p  = sqrt(sum(∫(e_p_centered * e_p_centered)dΩ)) / cell.P_c
    H1u  = sqrt(sum(∫(∇(e_u_func) ⊙ ∇(e_u_func))dΩ)) / (cell.U_c / cell.L)
    return (L2u=L2u, L2p=L2p, H1u=H1u)
end

function run_one(; Re, Da, alpha_0, n, U_amp, L, label)
    t0 = time()
    println("\n  [$label  |  U_amp=$U_amp, L=$L  |  N=$n]")
    cell = setup_cell(Re=Re, Da=Da, alpha_0=alpha_0, n=n, kv=1,
                      element_type="QUAD", U_amp=U_amp, L=L)
    @printf("    ν_dim=%.3e, σ_dim=%.3e\n", cell.nu_dim, cell.sigma_dim)
    r = solve_cell(cell)
    @printf("    success=%s | iters=%d | ‖R‖_final=%.4e | elapsed=%.1fs\n",
            r.success, r.iters, r.residual, time() - t0)
    diag = diagnose_errors(cell, r.x_solved)
    @printf("    L²(u)=%.4e  L²(p)=%.4e  H¹(u)=%.4e\n", diag.L2u, diag.L2p, diag.H1u)
    return (success=r.success, iters=r.iters, residual=r.residual,
            L2u=diag.L2u, L2p=diag.L2p, H1u=diag.H1u,
            nu_dim=cell.nu_dim, sigma_dim=cell.sigma_dim)
end

function main()
    Re, Da, alpha_0 = 1.0, 1e-6, 1.0  # C2 OSGS — easy regime; U_centered=1000 differs from default U=1
    n = 80
    sigma_check = Da * 1.0 * (1.0 / Re) / 1.0
    println("============================================================")
    println("Centered-encoding scout (OSGS, C16 cell)")
    println("Cell: Re=$Re, Da=$Da, α=$alpha_0 — σ(default-encoding)≈$(sigma_check)")
    println("Run start: ", now())
    println("============================================================")

    # Encoding A: legacy (U_amp = L = 1)  →  extreme dimensional ν, σ
    r_A = run_one(Re=Re, Da=Da, alpha_0=alpha_0, n=n,
                  U_amp=1.0, L=1.0, label="A: default (U=L=1)")

    # Encoding B: L=1 fixed, U_amp chosen to centre geom-mean(ν,σ) at 1.
    # This avoids the MMS shape becoming high-frequency (Paper2DMMS hardcodes sin(πx),
    # which is one half-period on L=1 but ~L/2 oscillations on L≠1).
    # Spread max/min stays the same — but the absolute magnitudes shift toward 1.
    U_centered = Re / sqrt(alpha_0 * Da)
    r_B = run_one(Re=Re, Da=Da, alpha_0=alpha_0, n=n,
                  U_amp=U_centered, L=1.0, label="B: centered (geom-mean(ν,σ)=1, L=1)")

    println("\n============================================================")
    @printf("Default encoding:  ν=%.2e  σ=%.2e  ⇒  L²(u)=%.4e, iters=%d, ‖R‖=%.2e\n",
            r_A.nu_dim, r_A.sigma_dim, r_A.L2u, r_A.iters, r_A.residual)
    @printf("Centered encoding: ν=%.2e  σ=%.2e  ⇒  L²(u)=%.4e, iters=%d, ‖R‖=%.2e\n",
            r_B.nu_dim, r_B.sigma_dim, r_B.L2u, r_B.iters, r_B.residual)
    println()
    ratio = r_B.L2u / r_A.L2u
    if abs(ratio - 1.0) < 1e-3
        println("  >>> L²(u) MATCHES between encodings (ratio $(round(ratio,digits=5))).")
        println("      The two encodings agree on the discrete solution → harness is consistent.")
        println("      No improvement from centering at this cell.")
    elseif ratio < 0.5
        println("  >>> Centered encoding produced LOWER L²(u) (ratio $(round(ratio,digits=4)))")
        println("      Worth pursuing the full harness reparametrization.")
    elseif ratio > 2.0
        println("  >>> Centered encoding produced HIGHER L²(u) (ratio $(round(ratio,digits=4)))")
        println("      Investigate: P_c / U_c normalization may not be regime-invariant.")
    else
        println("  >>> Modest change (ratio $(round(ratio,digits=4))). Marginal evidence either way.")
    end
    println("============================================================")
end

isinteractive() || main()
