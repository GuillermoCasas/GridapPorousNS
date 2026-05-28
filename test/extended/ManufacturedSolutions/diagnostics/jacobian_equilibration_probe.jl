#!/usr/bin/env julia
# diagnostics/jacobian_equilibration_probe.jl
#
# Question: Does symmetric-Jacobi diagonal scaling of the Newton Jacobian recover
# optimal MMS slopes in the high-σ regime (σ = Da·α/Re ≫ 1)?
#
# Test cells (all P1/P1 QUAD ASGS — orchestration kept minimal):
#   - C23-like: Re=1e-6, Da=1e6, α=0.05, σ ≈ 5e10 → known SUBOPTIMAL slope (1.74)
#
# For each mesh N ∈ {80, 160}, solve the cell TWICE:
#   1. equilibration_mode = :none           (legacy path; bit-identical to production)
#   2. equilibration_mode = :symmetric_jacobi (new path)
# Report: per-mesh L²(u), final ‖R‖, Newton iters, and the N=80 → N=160 slope.
#
# Run: cd test/extended/ManufacturedSolutions && julia --project=../../.. diagnostics/jacobian_equilibration_probe.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using PorousNSSolver
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Printf
using Dates

function setup_cell(; Re::Float64, Da::Float64, alpha_0::Float64, n::Int,
                    equilibration::String,
                    kv::Int=1, element_type::String="QUAD")
    alpha_infty = 1.0
    U_amp = 1.0
    L = 1.0
    kp = kv

    config_dict = Dict(
        "physical_properties" => Dict("nu" => 1.0, "eps_val" => 1e-8,
                                      "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
        "domain" => Dict("alpha_0" => alpha_0,
                         "bounding_box" => [-0.5, 0.5, -0.5, 0.5],
                         "r_1" => 0.2, "r_2" => 0.4),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
            "mesh" => Dict("element_type" => element_type, "partition" => [n, n]),
            "stabilization" => Dict("method" => "ASGS"),
            "solver" => Dict("jacobian_equilibration" => equilibration),
        ),
    )
    config = PorousNSSolver.load_config_from_dict(config_dict)

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

    h_scale = 1.0 / n
    spatial_err_est = h_scale^(kv + 1)
    c_ceil = config.numerical_method.solver.dynamic_ftol_ceiling
    c_sf   = config.numerical_method.solver.dynamic_ftol_spatial_safety_factor
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
    eq_mode = equilibration == "symmetric_jacobi" ? :symmetric_jacobi : :none

    nls_newton = PorousNSSolver.SafeNewtonSolver(
        LUSolver(), 150, config.numerical_method.solver.max_increases,
        config.numerical_method.solver.xtol, dynamic_ftol,
        config.numerical_method.solver.linesearch_alpha_min,
        config.numerical_method.solver.armijo_c1,
        config.numerical_method.solver.divergence_merit_factor,
        dynamic_noise_floor, config.numerical_method.solver.max_linesearch_iterations,
        config.numerical_method.solver.linesearch_contraction_factor;
        noise_floor_success_max_ftol_multiple=k_nf,
        equilibration_mode=eq_mode,
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
            U_c=U_c, P_c=P_c, L=L, sigma_c=sigma_c)
end

function solve_cell(cell)
    x = FEFunction(cell.X, copy(get_free_dof_values(cell.x0_exact)))
    x_backup = copy(get_free_dof_values(x))
    res = PorousNSSolver.safe_fe_solve!(x, cell.solver_newton, cell.op_newton; backup=x_backup)
    return x, res
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

function run_one(; Re, Da, alpha_0, Ns, equilibration)
    label = equilibration
    results = Dict{Int, NamedTuple}()
    for n in Ns
        t0 = time()
        println("\n  [N=$n | equilibration=$label]")
        cell = setup_cell(Re=Re, Da=Da, alpha_0=alpha_0, n=n, kv=1,
                          element_type="QUAD", equilibration=equilibration)
        x_solved, res = solve_cell(cell)
        @printf("    state=%s iters=%d ‖R‖_final=%.4e elapsed=%.1fs\n",
                res.state, res.iterations, res.residual_norm, time() - t0)
        diag = diagnose_errors(cell, x_solved)
        @printf("    L²(u) = %.6e | L²(p) = %.6e | H¹(u) = %.6e\n",
                diag.L2u, diag.L2p, diag.H1u)
        results[n] = (state=res.state, iters=res.iterations,
                      residual=res.residual_norm,
                      L2u=diag.L2u, L2p=diag.L2p, H1u=diag.H1u)
    end
    return results
end

function main()
    Re, Da, alpha_0 = 1e-6, 1e6, 0.05  # C23 — known SUBOPTIMAL ASGS (rate 1.74)
    Ns = [80, 160]
    sigma_c = Da * 1.0 * (1.0 / Re) / 1.0
    println("============================================================")
    println("Jacobian-equilibration probe")
    println("Cell: Re=$Re, Da=$Da, α₀=$alpha_0 (C23 ASGS — high σ ≈ $(sigma_c))")
    println("Run start: ", now())
    println("============================================================")

    r_none = run_one(Re=Re, Da=Da, alpha_0=alpha_0, Ns=Ns, equilibration="none")
    r_eq   = run_one(Re=Re, Da=Da, alpha_0=alpha_0, Ns=Ns, equilibration="symmetric_jacobi")

    println("\n============================================================")
    println("Slope comparison (N=$(Ns[1]) → N=$(Ns[end])):")
    function slope(r)
        a, b = r[Ns[1]], r[Ns[end]]
        sL2u = log(a.L2u / b.L2u) / log(2.0)
        sH1u = log(a.H1u / b.H1u) / log(2.0)
        return sL2u, sH1u
    end
    sLn, sHn = slope(r_none)
    sLe, sHe = slope(r_eq)
    @printf("  equilibration=none              L²(u) slope = %+.3f | H¹(u) slope = %+.3f\n", sLn, sHn)
    @printf("  equilibration=symmetric_jacobi  L²(u) slope = %+.3f | H¹(u) slope = %+.3f\n", sLe, sHe)
    println()
    if sLe >= sLn + 0.2
        println("  >>> Equilibration RECOVERS slope (Δ = $(round(sLe - sLn, digits=3)))")
        println("      Confirms conditioning hypothesis for this regime.")
    elseif sLe <= sLn - 0.2
        println("  >>> Equilibration WORSE — investigate (Δ = $(round(sLe - sLn, digits=3)))")
    else
        println("  >>> No meaningful difference (Δ = $(round(sLe - sLn, digits=3)))")
        println("      Either both already optimal, or the conditioning is not the binding constraint here.")
    end
    println("============================================================")
end

isinteractive() || main()
