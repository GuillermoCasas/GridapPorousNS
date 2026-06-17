# test/extended/ManufacturedSolutions/probe_stiff_diagnose.jl
# ==============================================================================================
# [diagnostic-tool] Low-level cell primitives for the stiff MMS corner.
#
# MINIMAL RESTORE: this file provides exactly the primitives that `run_continuation.jl` reuses
# via `include` — `build_cell` (mirrors run_test.jl's FE setup for ONE cell) and
# `probe_a2_heavy_solve` (a tight-tolerance, false-success-disabled heavy solve to a TRUE root),
# plus `CellArtifacts` and `calc_normalized_errors`. The full diagnostic battery (A1 Jacobian-FD,
# A3 τ-landscape, condition probes, report writers) lived in the original file removed in
# commit 3c66edd ("lean OSGS to a single coupled route"); only the continuation primitives are
# reconstructed here, verbatim from 3c66edd~1, since `run_continuation.jl` still depends on them.
# All referenced PorousNSSolver/Gridap symbols and config fields were verified to survive that
# refactor. No top-level executable code runs on `include`.
# ==============================================================================================

using PorousNSSolver
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using LinearAlgebra
using Printf

# ==============================================================================
# Cell construction — one self-contained struct + function mirroring run_test.jl
# ==============================================================================

struct CellArtifacts
    label::String
    config
    setup
    formulation
    phys_cfg
    freeze_cusp::Bool
    x0_exact                # interpolated MultiFieldFEFunction at u_ex, p_ex
    op_newton
    op_picard
    fe_solver_newton
    fe_solver_picard
    n::Int
    kv::Int
    Re::Float64
    Da::Float64
    alpha_0::Float64
    # --- extra read-only handles ---
    alpha_field             # SmoothRadialPorosity
    u_final                 # u_ex callable
    p_final                 # p_ex callable
    U_c::Float64            # characteristic velocity scale
    P_c::Float64            # characteristic pressure scale
    L::Float64
    nu::Float64             # ν = U·L/Re used by the formulation
    c_1::Float64
    c_2::Float64
    reg                     # SmoothVelocityFloor
    law                     # ConstantSigmaLaw
    tau_reg_lim::Float64
end

function build_cell(label::String; Re, Da, alpha_0, kv=1, n=10, element_type="QUAD")
    alpha_infty = 1.0
    U_amp = 1.0
    L = 1.0
    kp = kv

    config_dict = Dict(
        "physical_properties" => Dict("nu" => 1.0, "eps_val" => 1e-8, "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
        "domain" => Dict(
            "alpha_0" => alpha_0,
            "bounding_box" => [-0.5, 0.5, -0.5, 0.5],
            "r_1" => 0.2,
            "r_2" => 0.4,
        ),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
            "mesh" => Dict("element_type" => element_type, "partition" => [n, n]),
            "stabilization" => Dict("method" => "ASGS"),
            "solver" => Dict(),
        ),
    )
    config = PorousNSSolver.load_config_from_dict(config_dict)

    # Mesh + labels (mirror run_test.jl _build_local_mesh)
    domain_tuple = Tuple(config.domain.bounding_box)
    partition = Tuple(config.numerical_method.mesh.partition)
    if config.numerical_method.mesh.element_type == "TRI"
        model = CartesianDiscreteModel(domain_tuple, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain_tuple, partition)
    end
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [7])
    add_tag_from_tags!(labels, "outlet", [8])
    add_tag_from_tags!(labels, "walls", [1, 2, 3, 4, 5, 6])
    add_tag_from_tags!(labels, "all_boundaries", [1, 2, 3, 4, 5, 6, 7, 8])

    # FE spaces
    refe_u = ReferenceFE(lagrangian, VectorValue{2, Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
    Q = TestFESpace(model, refe_p, conformity=:H1)
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)

    # Quadrature
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv, PorousNSSolver.ConstantSigmaLaw(0.0))
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree + 4)

    # Element-wise characteristic length
    if config.numerical_method.mesh.element_type == "TRI"
        h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
    else
        h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    end
    h_cf = CellField(collect(h_array), Ω)

    Y = MultiFieldFESpace([V, Q])

    # c_1, c_2 + tau regularization
    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, kv)
    tau_reg_lim = config.physical_properties.tau_regularization_limit
    freeze_cusp = config.numerical_method.solver.freeze_jacobian_cusp

    # Porosity field
    alpha_field = PorousNSSolver.SmoothRadialPorosity(Float64(alpha_0), alpha_infty, config.domain.r_1, config.domain.r_2)
    alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)

    # Formulation + MMS oracle
    form = PorousNSSolver.PaperGeneralFormulation(
        PorousNSSolver.DeviatoricSymmetricViscosity(),
        PorousNSSolver.ConstantSigmaLaw(Float64(Da) * alpha_infty * (U_amp * L / Float64(Re)) / L^2),
        PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma(),
        PorousNSSolver.SmoothVelocityFloor(
            config.physical_properties.u_base_floor_ref,
            0.0,
            config.physical_properties.epsilon_floor,
        ),
        U_amp * L / Float64(Re),
        config.physical_properties.eps_val,
    )
    mms = PorousNSSolver.Paper2DMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)
    u_final = PorousNSSolver.get_u_ex(mms)
    p_final = PorousNSSolver.get_p_ex(mms)
    U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)
    nu_local = U_amp * L / Float64(Re)   # the ν the formulation was built with

    U = TrialFESpace(V, u_final)
    P = TrialFESpace(Q, p_final)
    X = MultiFieldFESpace([U, P])

    # Forcing
    f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)

    # Solver-tolerance scaling (mirror run_test.jl)
    h_scale = 1.0 / n
    spatial_err_est = h_scale^(kv + 1)
    ar_c1 = config.numerical_method.solver.armijo_c1
    div_fac = config.numerical_method.solver.divergence_merit_factor
    c_ceil = config.numerical_method.solver.dynamic_ftol_ceiling
    c_sf = config.numerical_method.solver.dynamic_ftol_spatial_safety_factor
    dynamic_ftol = max(config.numerical_method.solver.ftol, min(c_ceil, c_sf * spatial_err_est))
    condition_scaling = Float64(n)^2 * max(1.0, Float64(Re))
    n_base = config.numerical_method.solver.condition_noise_floor_baseline
    n_min = config.numerical_method.solver.condition_noise_floor_absolute_min
    n_sf = config.numerical_method.solver.condition_noise_floor_safety_factor
    dynamic_noise_floor = min(config.numerical_method.solver.stagnation_noise_floor, max(n_min, n_base * condition_scaling))
    dynamic_noise_floor = max(dynamic_noise_floor, dynamic_ftol * n_sf)
    max_inc = config.numerical_method.solver.max_increases
    xtol = config.numerical_method.solver.xtol
    ls_alpha_min = config.numerical_method.solver.linesearch_alpha_min
    max_ls_iters = config.numerical_method.solver.max_linesearch_iterations
    ls_contract = config.numerical_method.solver.linesearch_contraction_factor
    solver_newton_it = config.numerical_method.solver.newton_iterations
    solver_picard_it = config.numerical_method.solver.picard_iterations

    local_picard_it = solver_picard_it
    if Re >= config.numerical_method.solver.dynamic_picard_re_threshold
        local_picard_it = max(local_picard_it, config.numerical_method.solver.dynamic_picard_re_iterations)
    end
    if Da >= config.numerical_method.solver.dynamic_picard_da_threshold
        local_picard_it = max(local_picard_it, config.numerical_method.solver.dynamic_picard_da_iterations)
    end

    nls_picard = PorousNSSolver.SafeNewtonSolver(LUSolver(), local_picard_it, max_inc, xtol, dynamic_ftol, ls_alpha_min, ar_c1, div_fac, dynamic_noise_floor, max_ls_iters, ls_contract; mode=:picard)
    nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), solver_newton_it, max_inc, xtol, dynamic_ftol, ls_alpha_min, ar_c1, div_fac, dynamic_noise_floor, max_ls_iters, ls_contract)
    fe_solver_picard = FESolver(nls_picard)
    fe_solver_newton = FESolver(nls_newton)

    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
    phys_cfg = config.physical_properties

    res_fn(x, y) = PorousNSSolver.build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing)
    jac_picard(x, dx, y) = PorousNSSolver.build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    jac_newton(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, PorousNSSolver.ExactNewtonMode(); pi_u=nothing, pi_p=nothing)
    op_picard = FEOperator(res_fn, jac_picard, X, Y)
    op_newton = FEOperator(res_fn, jac_newton, X, Y)

    x0_exact = interpolate_everywhere([u_final, p_final], X)

    return CellArtifacts(
        label, config, setup, formulation, phys_cfg, freeze_cusp,
        x0_exact, op_newton, op_picard, fe_solver_newton, fe_solver_picard,
        n, kv, Float64(Re), Float64(Da), Float64(alpha_0),
        alpha_field, u_final, p_final, U_c, P_c, L,
        nu_local, c_1, c_2, form.regularization, form.reaction_law, tau_reg_lim,
    )
end

# ==============================================================================
# Error norms — same normalization as run_test.jl calculate_normalized_errors
# ==============================================================================

const _TINY = 1e-300

function calc_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, L, dΩ)
    e_u = u_final - u_h
    e_p = p_final - p_h
    el2_u = sqrt(abs(sum(∫(e_u ⋅ e_u)dΩ))) / U_c
    area = sum(∫(1.0)dΩ)
    mean_e_p = sum(∫(e_p)dΩ) / area
    e_p_c = e_p - mean_e_p
    el2_p = sqrt(abs(sum(∫(e_p_c * e_p_c)dΩ))) / P_c
    eh1_u = sqrt(abs(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ))) / (U_c / L)
    eh1_p = sqrt(abs(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ))) / (P_c / L)
    return el2_u, el2_p, eh1_u, eh1_p
end

# ==============================================================================
# Heavy solve to a TRUE root — tight ftol, false-success disabled (noise_floor=1e-12)
# ==============================================================================

function probe_a2_heavy_solve(cell::CellArtifacts, mode::Symbol; max_iters=500, verbose=true, x0_dofs=nothing)
    verbose && println(@sprintf("\n  >>> A2 — heavy %s from u_ex (budget=%d, ftol=1e-8, noise_floor=1e-12)", mode, max_iters))
    X = cell.setup.X
    op = mode === :picard ? cell.op_picard : cell.op_newton
    init_dofs = x0_dofs === nothing ? get_free_dof_values(cell.x0_exact) : x0_dofs
    x = FEFunction(X, copy(init_dofs))
    nls = PorousNSSolver.SafeNewtonSolver(LUSolver(), max_iters, 10^6, 1e-14, 1e-8, 1e-8,
                                          1e-4, 1e10, 1e-12, 60, 0.5; mode=mode)
    solver = FESolver(nls)
    do_solve = () -> begin
        res = solve!(x, solver, op)
        cache = res isa Tuple ? res[2] : res
        nls_cache = cache isa Tuple ? cache[2] : cache
        hasproperty(nls_cache, :result) ? nls_cache.result : nothing
    end
    r = nothing
    try
        r = verbose ? do_solve() : redirect_stdout(do_solve, devnull)
    catch e
        verbose && println("      threw: ", e)
        return Dict("error" => string(e))
    end
    uh, ph = x
    el2_u, el2_p, eh1_u, eh1_p = calc_normalized_errors(uh, ph, cell.u_final, cell.p_final,
                                                         cell.U_c, cell.P_c, cell.L, cell.setup.dΩ)
    final_R = r === nothing ? NaN : r.residual_norm
    if verbose && r !== nothing
        println(@sprintf("      iters=%d  stop=%s  ‖R‖_∞: %.3e -> %.3e",
            r.iterations, r.stop_reason, r.initial_residual_norm, final_R))
    end
    verbose && println(@sprintf("      L2 u=%.4e  L2 p=%.4e  H1 u=%.4e", el2_u, el2_p, eh1_u))
    reached_root = isfinite(final_R) && final_R <= 1e-7
    return Dict("iters" => r === nothing ? -1 : r.iterations,
                "stop" => r === nothing ? "?" : r.stop_reason,
                "final_R" => final_R, "l2_u" => el2_u, "l2_p" => el2_p, "h1_u" => eh1_u,
                "reached_root" => reached_root, "any_nonfinite" => any(!isfinite, get_free_dof_values(x)),
                "dofs" => copy(get_free_dof_values(x)))
end
