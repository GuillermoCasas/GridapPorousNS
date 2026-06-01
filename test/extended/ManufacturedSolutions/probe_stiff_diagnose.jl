# test/extended/ManufacturedSolutions/probe_stiff_diagnose.jl
# ==============================================================================================
# [diagnostic-tool] Manually-run investigation driver — NOT part of the automated sweep
# (run_test.jl) or the test tiers. Also provides the low-level primitives (`build_cell`,
# `probe_a2_heavy_solve`) that run_continuation.jl reuses via `include`.
#
# Nature & Intent:
# Standalone diagnostic for the stiff MMS corner cell C24 (Re=1e6, Da=1, α=0.05, k=1, n=10,
# QUAD, ASGS) — re-diagnosing, with TRUSTWORTHY probes, why it converges to a wrong solution.
#
# Decisive probes (see theory/.../in-a-previous-session plan):
#   - A0: benign-baseline GATE (Re=1, Da=1, α=0.5). The harness must behave textbook here
#         (A1 consistent + heavy Picard reaches the root) or we abort before interpreting C24.
#   - A1: Jacobian-vs-finite-difference consistency at u_ex (THE decisive test, never cleanly
#         run before). Compares assembled ExactNewton J·v to centered FD of the residual, per
#         field block; cross-checks the Picard Jacobian. A non-descending error floor localizes
#         an inconsistent (wrong) ExactNewton derivative term.
#   - A2: heavy solve from u_ex (Picard AND Newton) with huge budget + disabled false-success
#         (noise_floor=1e-12). Picard's direction is uncorrupted by ExactNewton-only τ terms, so
#         Picard→ftol ⇒ a discrete root exists. Reports the L2 error of the reached iterate.
#   - A3: τ / convective-dominance landscape along the radial porosity layer (effective_speed,
#         compute_tau_1/2, cell-Péclet) to locate any non-smooth thresholds.
#   - Supplementary: cond(J) (Probe D) and the Newton-step line probe (Probe E).
#
# Run as:   cd test/extended/ManufacturedSolutions && julia --project=../../.. probe_stiff_diagnose.jl
#   (args: "easy" runs only the gate; "stiff" runs only C24 and skips the gate.)
#
# Mirrors the FE-setup call path of run_test.jl for ONE cell at a time (build_cell). The
# resulting fresh report is diagnostics/probe_diag_v2.md; it does NOT overwrite the historical
# (self-flagged unreliable) probe_stiff_raw.md / probe_stiff_findings.md.
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using LinearAlgebra
using SparseArrays
using Printf
using Dates
using Random

# ==============================================================================
# Helpers
# ==============================================================================

# Mirrors src/diagnostics_helpers.jl:166-185 (that file is orphaned in PorousNSSolver.jl).
function estimate_condition(A)
    n = size(A, 1)
    if n > 5000
        norm_inf = opnorm(A, Inf)
        return Dict{String,Any}("size" => n, "norm_inf" => norm_inf, "cond_estimate" => -1.0)
    else
        try
            Adense = Matrix(A)
            S = svdvals(Adense)
            cnd = S[1] / S[end]
            return Dict{String,Any}("size" => n, "cond" => cnd, "min_sv" => S[end], "max_sv" => S[1])
        catch e
            return Dict{String,Any}("size" => n, "norm_inf" => opnorm(A, Inf), "cond_estimate" => -1.0, "svd_error" => string(e))
        end
    end
end

# ==============================================================================
# Cell construction — one self-contained function that mirrors run_test.jl
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
    # --- extra handles for the A1/A2/A3 probes (everything below is read-only) ---
    alpha_field             # SmoothRadialPorosity (for A3 τ-landscape)
    u_final                 # u_ex callable (for A2 error norms, A3 evaluation)
    p_final                 # p_ex callable
    U_c::Float64            # characteristic velocity scale (error normalization)
    P_c::Float64            # characteristic pressure scale
    L::Float64
    nu::Float64             # ν = U·L/Re used by the formulation (A3)
    c_1::Float64
    c_2::Float64
    reg                     # SmoothVelocityFloor (A3 effective_speed)
    law                     # ConstantSigmaLaw (A3 σ, τ₁)
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

    # Mesh + labels (mirror run_test.jl:38-52 _build_local_mesh)
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

    # FE spaces (run_test.jl:378-412)
    refe_u = ReferenceFE(lagrangian, VectorValue{2, Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
    Q = TestFESpace(model, refe_p, conformity=:H1)
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)

    # Quadrature (run_test.jl:393-395)
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv, PorousNSSolver.ConstantSigmaLaw(0.0))
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree + 4)

    # Element-wise characteristic length (run_test.jl:398-403)
    if config.numerical_method.mesh.element_type == "TRI"
        h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
    else
        h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    end
    h_cf = CellField(collect(h_array), Ω)

    Y = MultiFieldFESpace([V, Q])

    # c_1, c_2 + tau regularization (run_test.jl:414-423)
    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, kv)
    tau_reg_lim = config.physical_properties.tau_regularization_limit
    freeze_cusp = config.numerical_method.solver.freeze_jacobian_cusp

    # Porosity field (run_test.jl:431-433)
    alpha_field = PorousNSSolver.SmoothRadialPorosity(Float64(alpha_0), alpha_infty, config.domain.r_1, config.domain.r_2)
    alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)

    # Formulation + MMS oracle (run_test.jl:438-449)
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

    # Forcing (run_test.jl:452)
    f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)

    # Solver-tolerance scaling (run_test.jl:454-485) — adopt full run_test.jl logic so the
    # diagnostic mirrors what the failing cell actually saw.
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
# Probes
# ==============================================================================

# Probe D — Jacobian conditioning at u_ex.
function probe_d(cell::CellArtifacts)
    println("\n  >>> Probe D — Jacobian condition number at u_ex")
    A = allocate_jacobian(cell.op_newton, cell.x0_exact)
    jacobian!(A, cell.op_newton, cell.x0_exact)
    b = allocate_residual(cell.op_newton, cell.x0_exact)
    residual!(b, cell.op_newton, cell.x0_exact)

    norm_b_inf = norm(b, Inf)
    norm_b_2 = norm(b, 2)
    info = estimate_condition(A)

    println(@sprintf("      ‖R(u_ex)‖_∞     = %.6e", norm_b_inf))
    println(@sprintf("      ‖R(u_ex)‖_2     = %.6e", norm_b_2))
    println(@sprintf("      n_dofs          = %d", info["size"]))
    if haskey(info, "cond")
        println(@sprintf("      cond(J)         = %.6e", info["cond"]))
        println(@sprintf("      σ_max(J)        = %.6e", info["max_sv"]))
        println(@sprintf("      σ_min(J)        = %.6e", info["min_sv"]))
    else
        println(@sprintf("      cond(J)         = (n>5000, estimate skipped)"))
        println(@sprintf("      ‖J‖_∞ (opnorm)  = %.6e", info["norm_inf"]))
    end

    return Dict(
        "norm_b_inf" => norm_b_inf,
        "norm_b_2" => norm_b_2,
        "info" => info,
        "A" => A,
        "b" => b,
    )
end

# ------------------------------------------------------------------------------
# Helpers shared by the A1/A2/A3 probes
# ------------------------------------------------------------------------------
const _TINY = 1e-300
_relerr(diff, ref) = norm(diff) / max(norm(ref), _TINY)

# Velocity/pressure free-DOF index ranges (ConsecutiveMultiFieldStyle: u first, then p).
function field_ranges(X)
    nu = num_free_dofs(X.spaces[1])
    np = num_free_dofs(X.spaces[2])
    return (u = 1:nu, p = (nu + 1):(nu + np), nu = nu, np = np)
end

# Same normalization as run_test.jl:93 calculate_normalized_errors (replicated to keep
# this script self-contained — no include of run_test.jl's HDF5/JSON3 deps).
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

# A1 — THE decisive test: does the assembled ExactNewton Jacobian act like the true
# directional derivative of the residual at u_ex? Compare J·v to the centered FD
# (R(x0+εv) − R(x0−εv))/(2ε) over an ε-sweep, per field block. A correct derivative
# V-converges (∝ ε²) to ~0; an inconsistent term leaves a non-descending floor.
# Cross-checks the Picard Jacobian action too (Picard drops the ExactNewton-only terms).
function probe_a1_jacobian_fd(cell::CellArtifacts; n_random=3, seed=20260521, pass_tol=1e-5)
    println("\n  >>> A1 — Jacobian-vs-FD consistency at u_ex (centered difference)")
    X = cell.setup.X
    x0 = cell.x0_exact
    rng = field_ranges(X)
    ndof = num_free_dofs(X)

    A = allocate_jacobian(cell.op_newton, x0); jacobian!(A, cell.op_newton, x0)
    Ap = allocate_jacobian(cell.op_picard, x0); jacobian!(Ap, cell.op_picard, x0)
    x0d = get_free_dof_values(x0)
    bp = allocate_residual(cell.op_newton, x0)
    bm = allocate_residual(cell.op_newton, x0)
    eps_list = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    Random.seed!(seed)
    dirs = Tuple{String, Vector{Float64}}[]
    for k in 1:n_random
        v = randn(ndof); v ./= norm(v); push!(dirs, ("random_$k", v))
    end
    vu = zeros(ndof); vu[rng.u] .= randn(rng.nu); vu ./= norm(vu); push!(dirs, ("velocity_only", vu))
    vp = zeros(ndof); vp[rng.p] .= randn(rng.np); vp ./= norm(vp); push!(dirs, ("pressure_only", vp))

    best_newton_full = Inf
    worst_blocks = (u = Inf, p = Inf)
    per_dir = Dict{String, Any}()
    for (name, v) in dirs
        Jv = A * v
        Jpv = Ap * v
        println("    direction: $name")
        println("        ε         relerr(N,full)  relerr(N,u-blk) relerr(N,p-blk) relerr(Picard,full)")
        bestf = Inf; bestu = Inf; bestp = Inf
        for ε in eps_list
            residual!(bp, cell.op_newton, FEFunction(X, x0d .+ ε .* v))
            residual!(bm, cell.op_newton, FEFunction(X, x0d .- ε .* v))
            gfd = (bp .- bm) ./ (2ε)
            d = Jv .- gfd
            rf = _relerr(d, gfd)
            ru = _relerr(d[rng.u], gfd[rng.u])
            rp = _relerr(d[rng.p], gfd[rng.p])
            rpic = _relerr(Jpv .- gfd, gfd)
            println(@sprintf("        %.0e   %.6e    %.6e    %.6e    %.6e", ε, rf, ru, rp, rpic))
            bestf = min(bestf, rf); bestu = min(bestu, ru); bestp = min(bestp, rp)
        end
        per_dir[name] = (full = bestf, u = bestu, p = bestp)
        best_newton_full = min(best_newton_full, bestf)
        # Track the worst per-block min across directions where that block is excited.
        worst_blocks = (u = min(worst_blocks.u, bestu), p = min(worst_blocks.p, bestp))
        println(@sprintf("        -> best over ε: full=%.3e  u-blk=%.3e  p-blk=%.3e", bestf, bestu, bestp))
    end
    consistent = best_newton_full <= pass_tol
    println(@sprintf("    A1 verdict: ExactNewton Jacobian %s  (best full relerr = %.3e, pass ≤ %.0e)",
        consistent ? "CONSISTENT ✓" : "INCONSISTENT ✗", best_newton_full, pass_tol))
    return Dict("consistent" => consistent, "best_full" => best_newton_full, "per_dir" => per_dir)
end

# A2 — Does a discrete root exist near u_ex? Heavy solve (Picard OR Newton) from u_ex with a
# huge budget and disabled false-success (noise_floor=1e-12, big divergence tolerance), so the
# only way it stops "converged" is genuinely reaching ftol. Picard's direction is uncorrupted by
# the ExactNewton-only τ/convective terms, so Picard→ftol ⇒ the root exists. Reports the L2 error
# of the reached iterate: ‖R‖≈1e-3 with L2≈5e-3 means "near root"; L2≈0.12 means "not the root".
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
    verbose && println(@sprintf("      L2 u=%.4e  L2 p=%.4e  H1 u=%.4e   (correct α=0.05 cells: L2 u≈5e-3..7e-3)",
        el2_u, el2_p, eh1_u))
    reached_root = isfinite(final_R) && final_R <= 1e-7
    return Dict("iters" => r === nothing ? -1 : r.iterations,
                "stop" => r === nothing ? "?" : r.stop_reason,
                "final_R" => final_R, "l2_u" => el2_u, "l2_p" => el2_p, "h1_u" => eh1_u,
                "reached_root" => reached_root, "any_nonfinite" => any(!isfinite, get_free_dof_values(x)),
                "dofs" => copy(get_free_dof_values(x)))
end

# Da-continuation experiment: from the Da=1e6 root (the C27 seed, reachable from u_ex), step Da
# DOWN a geometric ramp, warm-starting each solve from the previous converged state. Tells us
# whether the discrete root can be TRACKED to the target Da=1 (⇒ Path 2 continuation is the fix)
# or whether it folds/vanishes at some Da_crit (⇒ the C24 root genuinely doesn't exist at this N).
# Generalized natural-parameter continuation toward the C24 target (Re=1e6, Da=1, α=0.05, N).
# Ramps ONE axis from a converged-neighbor seed to the target, warm-starting each step. A fold
# (root lost far from target) ⇒ that axis can't reach C24; reaching the target with ‖R‖→ftol
# ⇒ the discrete root EXISTS and that axis is a viable Path-2 continuation.
function probe_continuation(axis::Symbol; nsteps=40, N=10, mode=:newton, max_iters=200,
                            adaptive=false, min_step_ratio=1e-3)
    Re_t, Da_t, a_t = 1e6, 1.0, 0.05   # the C24 target point
    if axis === :Da          # seed C27 (Da=1e6)
        start = 1e6; target = Da_t; setp = v -> (Re=Re_t, Da=v, a=a_t); pname = "Da"
    elseif axis === :Re      # seed C23 (Re=1, Da=1, α=0.05)
        start = 1.0; target = Re_t; setp = v -> (Re=v, Da=Da_t, a=a_t); pname = "Re"
    elseif axis === :alpha   # seed C6 (Re=1e6, Da=1, α=1.0)
        start = 1.0; target = a_t; setp = v -> (Re=Re_t, Da=Da_t, a=v); pname = "α"
    else
        error("axis must be :Da, :Re, or :alpha")
    end
    ramp = [start * (target / start)^(k / nsteps) for k in 0:nsteps]

    println("\n", "="^78)
    println(@sprintf(" %s-CONTINUATION toward C24 (Re=1e6, Da=1, α=0.05, N=%d, %s): %s=%.0e → %.0e in %d steps",
        pname, N, mode, pname, start, target, nsteps))
    println("="^78)
    p0 = setp(start)
    seed_cell = build_cell("seed"; Re=p0.Re, Da=p0.Da, alpha_0=p0.a, kv=1, n=N, element_type="QUAD")
    s = probe_a2_heavy_solve(seed_cell, mode; verbose=false, max_iters=500)
    println(@sprintf("  seed %s=%.3e: reached_root=%s  ‖R‖=%.3e  L2u=%.3e",
        pname, start, get(s, "reached_root", false), get(s, "final_R", NaN), get(s, "l2_u", NaN)))
    if !get(s, "reached_root", false)
        println("  [!] seed did not converge — abort continuation."); return
    end
    x_prev = s["dofs"]; last_good = start
    println(@sprintf("  %-12s %-7s %-11s %-11s %-6s", pname, "root?", "‖R‖", "L2u", "iters"))

    try_step(v) = begin
        p = setp(v)
        cell = build_cell("cont"; Re=p.Re, Da=p.Da, alpha_0=p.a, kv=1, n=N, element_type="QUAD")
        probe_a2_heavy_solve(cell, mode; verbose=false, max_iters=max_iters, x0_dofs=x_prev)
    end

    if !adaptive
        for v in ramp[2:end]
            r = try_step(v)
            ok = get(r, "reached_root", false)
            println(@sprintf("  %-12.4e %-7s %.3e   %.3e   %d",
                v, ok, get(r, "final_R", NaN), get(r, "l2_u", NaN), get(r, "iters", -1)))
            ok || (println(@sprintf("  [!] root LOST at %s=%.4e (last good %s=%.4e). Branch folds (fixed steps).",
                pname, v, pname, last_good)); return)
            x_prev = r["dofs"]; last_good = v
        end
    else
        # Adaptive log-space stepping with halving on failure — distinguishes a TRUE fold
        # (root genuinely gone) from a fixed-step artifact (basin shrank faster than the step).
        logc = log(start); logt = log(target)
        step = (logt - logc) / nsteps                  # signed log-step
        base = abs(step); min_step = base * min_step_ratio
        while abs(logc - logt) > 1e-9
            trial = logc + step
            ((step > 0) == (trial > logt)) && (trial = logt)   # clamp to target
            v = exp(trial)
            r = try_step(v)
            if get(r, "reached_root", false)
                println(@sprintf("  %-12.4e %-7s %.3e   %.3e   %d",
                    v, true, get(r, "final_R", NaN), get(r, "l2_u", NaN), get(r, "iters", -1)))
                x_prev = r["dofs"]; last_good = v; logc = trial
                step = sign(step) * min(base, abs(step) * 2.0)   # grow back toward base
            else
                step /= 2
                if abs(step) < min_step
                    println(@sprintf("  [!] TRUE FOLD near %s=%.4e: min step (ratio %.0e) reached and the root is still lost.",
                        pname, last_good, min_step_ratio))
                    println(@sprintf("      ⇒ the discrete branch genuinely turns back; %s=%.3e is unreachable by %s-continuation at N=%d.",
                        pname, target, pname, N))
                    return
                end
                println(@sprintf("      [halve] step→%.3e (log) after fail near %s=%.4e", abs(step), pname, v))
            end
        end
    end
    println(@sprintf("  [✓] Tracked a TRUE root to the target %s=%.3e (L2u=%.3e). %s-continuation REACHES C24.",
        pname, target, get(try_step(target), "l2_u", NaN), pname))
end

# Root-existence map: from u_ex, run heavy Picard+Newton across (Da, N) at Re=1e6, α=0.05.
# reached_root ⇒ a discrete root with ‖R‖≤ftol exists and is reachable. Tells us whether a
# continuation seed exists at high Da, and whether mesh refinement recovers a root at Da=1.
function probe_root_map()
    println("\n", "="^78)
    println(" ROOT-EXISTENCE MAP — heavy Picard+Newton from u_ex (Re=1e6, α=0.05)")
    println("   reached_root ⇒ discrete root exists & reachable; correct L2 u ≈ 5e-3..7e-3")
    println("="^78)
    println(@sprintf("  %-9s %-4s | %-7s %-10s %-10s | %-7s %-10s %-10s",
        "Da", "N", "Pic?", "Pic‖R‖", "Pic L2u", "Nwt?", "Nwt‖R‖", "Nwt L2u"))
    function row(Da, N)
        cell = build_cell("map"; Re=1e6, Da=Da, alpha_0=0.05, kv=1, n=N, element_type="QUAD")
        p = probe_a2_heavy_solve(cell, :picard; verbose=false)
        w = probe_a2_heavy_solve(cell, :newton; verbose=false)
        println(@sprintf("  %-9.0e %-4d | %-7s %.3e  %.3e | %-7s %.3e  %.3e",
            Da, N, get(p, "reached_root", false), get(p, "final_R", NaN), get(p, "l2_u", NaN),
            get(w, "reached_root", false), get(w, "final_R", NaN), get(w, "l2_u", NaN)))
    end
    println("  -- Da sweep at N=10 (find where the root vanishes; Da=1e6 is the C27 seed) --")
    for Da in [1e6, 1e4, 1e2, 1e1, 1e0]
        row(Da, 10)
    end
    println("  -- h sweep at Da=1 (does refinement recover a root?) --")
    for N in [10, 20, 40]
        row(1.0, N)
    end
end

# A3 — τ / convective-dominance landscape along a radial line crossing the porosity layer
# (r ∈ [0.1, 0.5], r₁=0.2, r₂=0.4). Evaluates the EXACT u_ex pointwise (sidesteps CellField
# point-eval) and feeds the real code paths (effective_speed, compute_tau_1/2). Shows that at
# C24 the convective term c₂|u|/h dominates A_NS by orders of magnitude (cell-Péclet), and
# whether effective_speed / porosity clamps introduce non-smoothness.
function probe_a3_tau_landscape(cell::CellArtifacts; npts=41)
    println("\n  >>> A3 — τ / convective-dominance landscape along radial line (r∈[0.1,0.5])")
    reg = cell.reg; ν = cell.nu; c1 = cell.c_1; c2 = cell.c_2; law = cell.law
    af = cell.alpha_field; h = 1.0 / cell.n; treg = cell.tau_reg_lim
    u_ex = cell.u_final
    println("        r      |u|_raw    eff_speed   alpha      viscΑ=c1ν/h²  convA=c2|u|/h  sigma      Pe_cell    tau1       tau2")
    rows = []
    s = 1.0 / sqrt(2.0)
    for r in range(0.1, 0.5; length=npts)
        x = Point(r * s, r * s)
        u = u_ex(x); magu = sqrt(u ⋅ u)
        es = PorousNSSolver.effective_speed(reg, u, ν, h, c1, c2)
        a = PorousNSSolver.alpha(af, x)
        ga = PorousNSSolver.grad_alpha(af, x)
        gu = ∇(u_ex)(x)
        kin = PorousNSSolver.KinematicState(u, gu, es)
        med = PorousNSSolver.MediumState(a, ga, h)
        sig = PorousNSSolver.sigma(law, kin, med, es)
        viscA = c1 * ν / h^2
        convA = c2 * es / h
        pe = convA / max(viscA, _TINY)
        t1 = PorousNSSolver.compute_tau_1(kin, med, ν, c1, c2, treg, law)
        t2 = PorousNSSolver.compute_tau_2(kin, med, ν, c1, c2, treg)
        println(@sprintf("        %.3f  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.3e  %.4e  %.4e",
            r, magu, es, a, viscA, convA, sig, pe, t1, t2))
        push!(rows, (r = r, magu = magu, es = es, a = a, pe = pe, t1 = t1))
    end
    return Dict("rows" => rows)
end

# Probe E — Newton-step line probe at u_ex.
# Gridap's convention is J·dx = b with update x ← x − α·dx; so the Newton step direction is −dx
# and a parameter s ∈ [-1, 1.5] traces x_s = x0 − s·dx. At s=1 we expect convergence; at s=0 we
# recover x0; at s<0 we deliberately go uphill.
function probe_e(cell::CellArtifacts, A, b)
    println("\n  >>> Probe E — Newton-step line probe at u_ex")
    Adense = nothing
    delta = nothing
    try
        Adense = Matrix(A)
        delta = Adense \ b
    catch e
        println("      Linear solve failed: ", e)
        return Dict("error" => string(e))
    end

    x_dofs = copy(get_free_dof_values(cell.x0_exact))
    sample_s = collect(range(-1.0, 1.5; length=51))  # 51 sample points; includes 0 and 1
    rows = Vector{NamedTuple{(:s, :rinf, :phi),Tuple{Float64,Float64,Float64}}}()
    b_s = allocate_residual(cell.op_newton, cell.x0_exact)
    for s in sample_s
        dofs_s = x_dofs .- s .* delta
        x_s = FEFunction(cell.setup.X, dofs_s)
        try
            residual!(b_s, cell.op_newton, x_s)
            r_inf = norm(b_s, Inf)
            phi = 0.5 * dot(b_s, b_s)
            push!(rows, (s=s, rinf=r_inf, phi=phi))
        catch e
            push!(rows, (s=s, rinf=NaN, phi=NaN))
        end
    end

    # Find s* minimising ‖R‖_∞
    finite_rows = [r for r in rows if isfinite(r.rinf)]
    if isempty(finite_rows)
        println("      All samples diverged. Featureless / catastrophic landscape.")
        return Dict("rows" => rows, "min_s" => NaN, "min_rinf" => NaN, "delta_norm" => norm(delta))
    end
    min_row = argmin(r -> r.rinf, finite_rows)
    println(@sprintf("      ‖δ‖_2 (Newton step magnitude) = %.6e", norm(delta)))
    println(@sprintf("      s*  = %+.4f   ‖R(x0 - s* δ)‖_∞ = %.6e", min_row.s, min_row.rinf))
    println(@sprintf("      s=0:  ‖R‖_∞ = %.6e", first(r.rinf for r in rows if r.s == 0.0 || isapprox(r.s, 0.0; atol=1e-12))))
    s1_row = findfirst(r -> isapprox(r.s, 1.0; atol=1e-6), rows)
    if s1_row !== nothing
        println(@sprintf("      s=1:  ‖R‖_∞ = %.6e", rows[s1_row].rinf))
    end
    println("      Full landscape (s, ‖R‖_∞, Φ):")
    for r in rows
        rinf_str = isfinite(r.rinf) ? @sprintf("%.4e", r.rinf) : "NaN     "
        phi_str = isfinite(r.phi) ? @sprintf("%.4e", r.phi) : "NaN     "
        println("        ", @sprintf("s=%+.4f   ‖R‖_∞=%s   Φ=%s", r.s, rinf_str, phi_str))
    end

    return Dict(
        "rows" => rows,
        "min_s" => min_row.s,
        "min_rinf" => min_row.rinf,
        "delta_norm" => norm(delta),
    )
end

# ==============================================================================
# Findings report — write a markdown summary
# ==============================================================================

function classify(stiff::Dict)
    a1 = stiff["A1"]; pic = stiff["Picard"]; nwt = stiff["Newton"]
    jac = get(a1, "consistent", false)
    pic_root = get(pic, "reached_root", false)
    nwt_root = get(nwt, "reached_root", false)
    pic_l2 = get(pic, "l2_u", NaN); nwt_l2 = get(nwt, "l2_u", NaN)

    v = String[]
    push!(v, @sprintf("A1: ExactNewton Jacobian is %s at u_ex (best full relerr %.2e).",
        jac ? "CONSISTENT" : "INCONSISTENT", get(a1, "best_full", NaN)))
    push!(v, @sprintf("A2 heavy Picard: reached_root=%s, final ‖R‖=%.2e, L2 u=%.3e (stop=%s).",
        pic_root, get(pic, "final_R", NaN), pic_l2, get(pic, "stop", "?")))
    push!(v, @sprintf("A2 heavy Newton: reached_root=%s, final ‖R‖=%.2e, L2 u=%.3e (stop=%s).",
        nwt_root, get(nwt, "final_R", NaN), nwt_l2, get(nwt, "stop", "?")))

    # FE-correct L2 for an α=0.05 cell is ≈5e-3..7e-3; treat ≤1e-2 as "near the true root".
    root_l2_ok = (isfinite(pic_l2) && pic_l2 <= 1e-2) || (isfinite(nwt_l2) && nwt_l2 <= 1e-2)
    if !jac && (pic_root || nwt_root)
        push!(v, "VERDICT → Path 1: J INCONSISTENT but a discrete root is reachable. Fix the offending " *
                 "ExactNewton derivative term (localize u-block vs p-block from the A1 table).")
    elseif !jac
        push!(v, "VERDICT → Path 1 then Path 2: J INCONSISTENT and no nearby root reached. Fix the " *
                 "derivative first; add continuation if it stays far.")
    elseif (pic_root || nwt_root) && root_l2_ok
        push!(v, "VERDICT → Path 3: J CONSISTENT and the true root is reachable at FE-correct L2 — the " *
                 "production wrong-answer is the loose noise-floor declaring false success. Tighten the " *
                 "noise-floor success classification (and give Picard/Newton honest budget).")
    elseif pic_root || nwt_root
        push!(v, "VERDICT → Path 3 + check: root reached by residual but L2 still large — re-examine the " *
                 "error norm / mesh marginality before concluding.")
    else
        push!(v, "VERDICT → Path 2: J CONSISTENT but NO nearby discrete root at ftol — landscape genuinely " *
                 "hard. Da-continuation from C27 and/or smooth the τ/velocity-floor (see A3 dominance/threshold).")
    end
    return v
end

_fmt(x) = (x isa Real && isfinite(x)) ? @sprintf("%.6e", x) : string(x)

function _dump_section(io::IO, r::Dict)
    d = r["D"]; info = d["info"]
    write(io, "### Supplementary — cond(J), residual at u_ex\n\n")
    write(io, "- ‖R(u_ex)‖_∞ = $(_fmt(d["norm_b_inf"])),  ‖R(u_ex)‖_2 = $(_fmt(d["norm_b_2"])),  n_dofs = $(info["size"])\n")
    if haskey(info, "cond")
        write(io, "- cond(J) = $(_fmt(info["cond"])),  σ_max = $(_fmt(info["max_sv"])),  σ_min = $(_fmt(info["min_sv"]))\n\n")
    else
        write(io, "- cond(J) skipped (n>5000),  ‖J‖_∞ = $(_fmt(info["norm_inf"]))\n\n")
    end

    a1 = r["A1"]
    write(io, "### A1 — Jacobian-vs-FD consistency (decisive)\n\n")
    write(io, "ExactNewton Jacobian **$(get(a1, "consistent", false) ? "CONSISTENT" : "INCONSISTENT")** ")
    write(io, "(best full relerr $(_fmt(get(a1, "best_full", NaN)))).\n\n")
    write(io, "| direction | best full relerr | u-block | p-block |\n|---|---|---|---|\n")
    for (name, pd) in get(a1, "per_dir", Dict())
        write(io, "| $name | $(_fmt(pd.full)) | $(_fmt(pd.u)) | $(_fmt(pd.p)) |\n")
    end
    write(io, "\n")

    for (key, title) in (("Picard", "A2 — heavy Picard from u_ex"), ("Newton", "A2 — heavy Newton from u_ex"))
        s = r[key]
        write(io, "### $title\n\n")
        if haskey(s, "error")
            write(io, "threw: `$(s["error"])`\n\n")
        else
            write(io, "- iters=$(s["iters"])  stop=`$(s["stop"])`  reached_root=$(s["reached_root"])\n")
            write(io, "- final ‖R‖ = $(_fmt(s["final_R"])),  L2 u = $(_fmt(s["l2_u"])),  L2 p = $(_fmt(s["l2_p"])),  H1 u = $(_fmt(s["h1_u"]))\n\n")
        end
    end

    e = r["E"]
    write(io, "### Supplementary — Newton-step line probe\n\n")
    if haskey(e, "error")
        write(io, "line probe failed: `$(e["error"])`\n\n")
    else
        write(io, "‖δ‖_2 = $(_fmt(e["delta_norm"])),  min at s* = $(@sprintf("%+.3f", e["min_s"])),  ‖R‖_∞(s*) = $(_fmt(e["min_rinf"]))\n\n")
    end
end

function write_report(easy::Union{Dict,Nothing}, stiff::Dict, report_path::String)
    mkpath(dirname(report_path))
    open(report_path, "w") do io
        write(io, "# probe_stiff_diagnose.jl — findings (v2: A1 J–FD consistency, A2 heavy solves, A3 τ-landscape)\n\n")
        write(io, "Run: $(string(now()))\n\n")
        write(io, "Decisive re-diagnosis of C24 (`Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS`). ")
        write(io, "Supersedes probe_stiff_raw.md / probe_stiff_findings.md (whose harness was self-flagged unreliable).\n\n")
        if easy !== nothing
            gate_ok = get(easy["A1"], "consistent", false) && get(easy["Picard"], "reached_root", false)
            write(io, "## A0 benign gate — `Re=1, Da=1, α=0.5, k=1, n=10, QUAD, ASGS`\n\n")
            write(io, gate_ok ? "**GATE PASSED.**\n\n" : "**GATE FAILED — do not interpret stiff results.**\n\n")
            _dump_section(io, easy)
        end
        write(io, "## Stiff cell C24 — `Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS`\n\n")
        _dump_section(io, stiff)
        write(io, "## Verdict\n\n")
        for line in classify(stiff)
            write(io, "- " * line * "\n")
        end
        write(io, "\n")
    end
    println("\n[report] wrote $report_path")
end

# ==============================================================================
# Main
# ==============================================================================

function run_diagnostic_for(label::String, cell_kwargs::NamedTuple)
    println("\n========================================================================")
    println(" CELL: $label")
    println("   Re=$(cell_kwargs.Re), Da=$(cell_kwargs.Da), α=$(cell_kwargs.alpha_0), kv=$(cell_kwargs.kv), n=$(cell_kwargs.n)")
    println("========================================================================")
    cell = build_cell(label; cell_kwargs...)
    println("   built. n_dofs = $(num_free_dofs(cell.setup.X))")

    d   = probe_d(cell)                          # supplementary: cond(J), ‖R(u_ex)‖
    a1  = probe_a1_jacobian_fd(cell)             # A1 (decisive): Jacobian-vs-FD consistency
    pic = probe_a2_heavy_solve(cell, :picard)    # A2: uncorrupted heavy Picard from u_ex
    nwt = probe_a2_heavy_solve(cell, :newton)    # A2: heavy Newton from u_ex (current code)
    tau = probe_a3_tau_landscape(cell)           # A3: τ / convective-dominance landscape
    e   = probe_e(cell, d["A"], d["b"])          # supplementary: Newton-step line probe
    return Dict("D" => d, "A1" => a1, "Picard" => pic, "Newton" => nwt, "Tau" => tau, "E" => e, "label" => label)
end

function main()
    args = ARGS

    if "rootmap" in args
        probe_root_map()
        return
    end

    if "continuation" in args || "da_cont" in args
        probe_continuation(:Da)
        return
    end
    if "re_cont" in args
        probe_continuation(:Re)
        return
    end
    if "alpha_cont" in args
        probe_continuation(:alpha)
        return
    end
    if "allcont" in args
        probe_continuation(:Re)
        probe_continuation(:alpha)
        return
    end
    if "alpha_meshsweep" in args
        # Does the α-fold move past the target α=0.05 as the mesh resolves the layer?
        for n in [20, 40, 80]
            probe_continuation(:alpha; N=n)
        end
        return
    end
    if "alpha_adaptive" in args
        # Decisive: TRUE fold vs fixed-step artifact. Adaptive halving down to a tiny step.
        for n in [40, 80]
            probe_continuation(:alpha; N=n, adaptive=true, nsteps=40, min_step_ratio=1e-3)
        end
        return
    end
    if "alpha_fineN" in args
        # Find the minimum N at which α-continuation reaches the C24 target α=0.05 (fold < 0.05).
        for n in [160, 320]
            probe_continuation(:alpha; N=n, adaptive=true, nsteps=48, min_step_ratio=2e-3)
        end
        return
    end

    do_easy = isempty(args) || "easy" in args
    do_stiff = isempty(args) || "stiff" in args

    easy_results = nothing
    stiff_results = nothing

    if do_easy
        # Benign baseline: everything O(1), well-conditioned. The harness MUST behave
        # textbook here (A1 consistent + heavy Picard reaches the root) before any
        # stiff-cell number is trustworthy.
        easy_results = run_diagnostic_for(
            "benign baseline (Re=1, Da=1, α=0.5, k=1, n=10, QUAD, ASGS)",
            (Re=1.0, Da=1.0, alpha_0=0.5, kv=1, n=10, element_type="QUAD"),
        )
        gate_ok = get(easy_results["A1"], "consistent", false) && get(easy_results["Picard"], "reached_root", false)
        println("\n", "="^72)
        if gate_ok
            println(" A0 GATE PASSED — harness validated on a benign cell. Proceeding to the stiff cell.")
        else
            println(" A0 GATE FAILED — the diagnostic harness misbehaves on a benign cell.")
            println("   A1 consistent = ", get(easy_results["A1"], "consistent", false),
                    " ; heavy-Picard reached root = ", get(easy_results["Picard"], "reached_root", false))
            println("   Do NOT interpret stiff-cell results. Aborting.")
        end
        println("="^72)
        gate_ok || return
    end

    if do_stiff
        stiff_results = run_diagnostic_for(
            "stiff C24 (Re=1e6, Da=1, α=0.05, k=1, n=10, QUAD, ASGS)",
            (Re=1e6, Da=1.0, alpha_0=0.05, kv=1, n=10, element_type="QUAD"),
        )
    end

    if stiff_results !== nothing
        println("\n", "="^72)
        println(" DECISION-MATRIX VERDICT (C24)")
        println("="^72)
        for line in classify(stiff_results)
            println("  - ", line)
        end
        # Fresh trustworthy report; does NOT overwrite the historical probe_stiff_raw.md.
        report_path = joinpath(@__DIR__, "diagnostics", "probe_diag_v2.md")
        write_report(easy_results, stiff_results, report_path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
