# test/extended/ManufacturedSolutions/probe_stiff_diagnose.jl
# ==============================================================================================
# Nature & Intent:
# Standalone diagnostic for the stiff MMS corner cell (Re=1e6, Da=1, α=0.05, k=1, n=10, QUAD, ASGS)
# that the previous-session investigation deferred Phase 6 §2.1 over.
#
# Runs three probes named in theory/algorithm-improvement-progress.md §"Phase 6 §2.1 — DEFERRED":
#   - Probe D: Jacobian condition number at u_ex (targets H2 — ill-conditioning).
#   - Probe C: Picard iteration from u_ex (targets H2 vs H3/H4 — does ANY iteration map have a
#              fixed point here?).
#   - Probe E: Newton-step line probe at u_ex (targets line-search policy vs H3/H4 —
#              is the Newton direction usable?).
#
# Run as:
#   cd test/extended/ManufacturedSolutions
#   julia --project=../../.. probe_stiff_diagnose.jl
#
# Mirrors the FE-setup call path of run_test.jl:368-516 for ONE cell at a time. Sanity-checks
# the probes against a known-easy cell first; if the easy-cell probes don't behave (Probe D
# cond < 1e10, Probe C converges in ≤ 3 iters, Probe E has a clean minimum near s=1), the
# script is wrong, not the solver — abort before drawing conclusions about the stiff cell.
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
        0.0,
    )
    mms = PorousNSSolver.Paper2DMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)
    u_final = PorousNSSolver.get_u_ex(mms)
    p_final = PorousNSSolver.get_p_ex(mms)

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

# Probe C — Picard from u_ex. Builds a fresh FEFunction from u_ex and calls the Picard solver.
function probe_c(cell::CellArtifacts)
    println("\n  >>> Probe C — Picard iteration from u_ex")

    # Fresh FEFunction at u_ex via DOF copy — don't mutate cell.x0_exact (probes D and E reuse it).
    x_picard = FEFunction(cell.setup.X, copy(get_free_dof_values(cell.x0_exact)))
    initial_norm = norm(begin
        bb = allocate_residual(cell.op_picard, x_picard)
        residual!(bb, cell.op_picard, x_picard)
        bb
    end, Inf)
    println(@sprintf("      initial ‖R(u_ex)‖_∞ = %.6e", initial_norm))

    try
        res = solve!(x_picard, cell.fe_solver_picard, cell.op_picard)
        cache = res isa Tuple ? res[2] : res
        nls_cache = cache isa Tuple ? cache[2] : cache
        if hasproperty(nls_cache, :result)
            r = nls_cache.result
            println(@sprintf("      iters           = %d", r.iterations))
            println(@sprintf("      stop_reason     = %s", r.stop_reason))
            println(@sprintf("      final ‖R‖       = %.6e (initial %.6e)", r.residual_norm, r.initial_residual_norm))
            return Dict(
                "iters" => r.iterations,
                "stop_reason" => r.stop_reason,
                "final_residual_norm" => r.residual_norm,
                "initial_residual_norm" => r.initial_residual_norm,
                "any_nonfinite" => any(!isfinite, get_free_dof_values(x_picard)),
            )
        else
            println("      (no result struct returned)")
            return Dict("error" => "no_result", "any_nonfinite" => any(!isfinite, get_free_dof_values(x_picard)))
        end
    catch e
        println("      Picard threw: ", e)
        return Dict("error" => string(e))
    end
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

function classify_findings(easy::Dict, stiff::Dict)
    # Easy-cell sanity: NOT cond-based (cond is dominated by P_c scaling, ~1e18 even on
    # easy cells in this codebase). Use the real discriminators: Picard converges and
    # the line probe minimum is near s=1.
    easy_sane = true
    easy_sane &= !haskey(easy["C"], "error") && get(easy["C"], "stop_reason", "") in ("ftol_reached", "stagnation_noise_floor_reached", "initial_ftol")
    easy_sane &= isfinite(easy["E"]["min_s"]) && abs(easy["E"]["min_s"] - 1.0) < 0.3

    # Stiff diagnosis
    stiff_cond = haskey(stiff["D"]["info"], "cond") ? stiff["D"]["info"]["cond"] : NaN
    stiff_picard_ok = !haskey(stiff["C"], "error") &&
                       isfinite(get(stiff["C"], "final_residual_norm", NaN)) &&
                       get(stiff["C"], "final_residual_norm", Inf) < 1e-1 * get(stiff["C"], "initial_residual_norm", 1.0)
    stiff_line_min_near_1 = haskey(stiff["E"], "min_s") && isfinite(stiff["E"]["min_s"]) && abs(stiff["E"]["min_s"] - 1.0) < 0.3
    stiff_line_uphill = haskey(stiff["E"], "min_s") && isfinite(stiff["E"]["min_s"]) && stiff["E"]["min_s"] < 0.0

    verdict = String[]
    if !easy_sane
        push!(verdict, "**Easy-cell sanity FAILED.** Diagnostic script is wrong, not the solver. Do NOT interpret stiff results.")
    end

    # Note: cond(J) is dominated by P_c scaling in this codebase (~1e18 on the easy
    # cell at α₀=0.05). It is not a useful H2 discriminator on its own — compare against
    # the easy-cell reference if available.
    if isfinite(stiff_cond) && haskey(easy["D"]["info"], "cond")
        easy_cond = easy["D"]["info"]["cond"]
        if isfinite(easy_cond) && stiff_cond > easy_cond * 1e3
            push!(verdict, "**H2 (Jacobian ill-conditioning) candidate**: stiff cond(J) = $(@sprintf("%.2e", stiff_cond)) exceeds easy-cell cond(J) = $(@sprintf("%.2e", easy_cond)) by >3 orders of magnitude. Cross-check with Probe E line-probe shape before committing.")
        else
            push!(verdict, "**H2 (Jacobian ill-conditioning) ruled out**: stiff cond(J) = $(@sprintf("%.2e", stiff_cond)) is comparable to or better than easy-cell cond(J) = $(@sprintf("%.2e", easy_cond)). Look elsewhere (line-probe spikes, safeguard misfires, iteration budget).")
        end
    end
    if stiff_picard_ok
        push!(verdict, "**Iteration map has a usable fixed point**: Picard reduced residual where Newton fails. → H2 confirmed (Newton-specific obstacle); H3 ruled out. Fix path: Branch H2.")
    elseif haskey(stiff["C"], "final_residual_norm")
        push!(verdict, "**Picard ALSO fails to reduce residual** (final $(@sprintf("%.2e", get(stiff["C"], "final_residual_norm", NaN))) vs initial $(@sprintf("%.2e", get(stiff["C"], "initial_residual_norm", NaN)))). → H2 weakened; H3 or H4 elevated.")
    end
    if stiff_line_min_near_1
        push!(verdict, "**Newton direction is valid**: line probe minimum at s ≈ $(@sprintf("%.2f", stiff["E"]["min_s"])) shows the s=1 Newton step IS a descent direction. → SafeNewtonSolver line-search policy is the obstacle (plan §Phase 2 sub-branch). Fix path: tune c1 / linesearch_alpha_min / linesearch_contraction_factor.")
    elseif stiff_line_uphill
        push!(verdict, "**Newton direction is uphill**: line probe minimum at s < 0 means the computed J⁻¹R points the wrong way. → H2 (J pathological) or H4 (τ-stabilisation pathology). Fix path: Branch H2 or Branch H4.")
    elseif haskey(stiff["E"], "min_s") && isfinite(stiff["E"]["min_s"])
        push!(verdict, "**Newton direction is partial**: line probe minimum at s ≈ $(@sprintf("%.2f", stiff["E"]["min_s"])), neither uphill nor full-step. Featureless landscape supports H3 (no nearby discrete root).")
    end
    if isempty(verdict)
        push!(verdict, "(No clear pattern — manual interpretation required.)")
    end
    return verdict
end

function write_report(easy::Union{Dict,Nothing}, stiff::Dict, report_path::String)
    mkpath(dirname(report_path))
    open(report_path, "w") do io
        ts = string(now())
        write(io, "# probe_stiff_diagnose.jl — findings\n\n")
        write(io, "Run: $(ts)\n\n")
        write(io, "Mirrors the FE setup of run_test.jl for the deferred Phase 6 §2.1 cell ")
        write(io, "(`Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS`). Runs Probes C/D/E from ")
        write(io, "theory/algorithm-improvement-progress.md §\"Phase 6 §2.1 — DEFERRED\".\n\n")

        if easy !== nothing
            write(io, "## Sanity check — easy cell (`Re=1e-6, Da=1e6, α₀=0.05, k=1, n=10, QUAD, ASGS`)\n\n")
            _dump_section(io, easy)
        end
        write(io, "## Stiff cell — `Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS`\n\n")
        _dump_section(io, stiff)

        write(io, "## Verdict\n\n")
        if easy !== nothing
            for line in classify_findings(easy, stiff)
                write(io, "- " * line * "\n")
            end
        else
            write(io, "(Easy-cell sanity was skipped — verdict reduced.)\n")
        end
        write(io, "\n")
    end
    println("\n[report] wrote $report_path")
end

function _dump_section(io::IO, results::Dict)
    write(io, "### Probe D — Jacobian condition number at u_ex\n\n")
    info = results["D"]["info"]
    write(io, "| Metric | Value |\n|---|---|\n")
    write(io, "| ‖R(u_ex)‖_∞ | $(@sprintf("%.6e", results["D"]["norm_b_inf"])) |\n")
    write(io, "| ‖R(u_ex)‖_2 | $(@sprintf("%.6e", results["D"]["norm_b_2"])) |\n")
    write(io, "| n_dofs | $(info["size"]) |\n")
    if haskey(info, "cond")
        write(io, "| cond(J) | $(@sprintf("%.6e", info["cond"])) |\n")
        write(io, "| σ_max(J) | $(@sprintf("%.6e", info["max_sv"])) |\n")
        write(io, "| σ_min(J) | $(@sprintf("%.6e", info["min_sv"])) |\n")
    else
        write(io, "| cond(J) | (n>5000, skipped) |\n")
        write(io, "| ‖J‖_∞ (opnorm) | $(@sprintf("%.6e", info["norm_inf"])) |\n")
    end
    write(io, "\n")

    write(io, "### Probe C — Picard iteration from u_ex\n\n")
    c = results["C"]
    if haskey(c, "error")
        write(io, "Picard threw / failed: `$(c["error"])`\n\n")
    else
        write(io, "| Metric | Value |\n|---|---|\n")
        write(io, "| iters | $(c["iters"]) |\n")
        write(io, "| stop_reason | `$(c["stop_reason"])` |\n")
        write(io, "| initial ‖R‖ | $(@sprintf("%.6e", c["initial_residual_norm"])) |\n")
        write(io, "| final ‖R‖ | $(@sprintf("%.6e", c["final_residual_norm"])) |\n")
        write(io, "| any non-finite DOF | $(c["any_nonfinite"]) |\n\n")
    end

    write(io, "### Probe E — Newton-step line probe at u_ex\n\n")
    e = results["E"]
    if haskey(e, "error")
        write(io, "Line probe failed: `$(e["error"])`\n\n")
    else
        write(io, "‖δ‖_2 (Newton step magnitude) = $(@sprintf("%.6e", e["delta_norm"]))\n\n")
        write(io, "Minimum at s* = $(@sprintf("%+.4f", e["min_s"])), ‖R(x0 − s* δ)‖_∞ = $(@sprintf("%.6e", e["min_rinf"]))\n\n")
        write(io, "| s | ‖R‖_∞ | Φ |\n|---|---|---|\n")
        for r in e["rows"]
            rinf_s = isfinite(r.rinf) ? @sprintf("%.4e", r.rinf) : "NaN"
            phi_s = isfinite(r.phi) ? @sprintf("%.4e", r.phi) : "NaN"
            write(io, "| $(@sprintf("%+.4f", r.s)) | $rinf_s | $phi_s |\n")
        end
        write(io, "\n")
    end
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

    d = probe_d(cell)
    c = probe_c(cell)
    e = probe_e(cell, d["A"], d["b"])
    return Dict("D" => d, "C" => c, "E" => e, "label" => label)
end

function main()
    args = ARGS
    do_easy = isempty(args) || "easy" in args
    do_stiff = isempty(args) || "stiff" in args

    easy_results = nothing
    stiff_results = nothing

    if do_easy
        easy_results = run_diagnostic_for(
            "easy (Re=1e-6, Da=1e6, α=0.05, k=1, n=10, QUAD, ASGS)",
            (Re=1e-6, Da=1e6, alpha_0=0.05, kv=1, n=10, element_type="QUAD"),
        )
    end
    if do_stiff
        stiff_results = run_diagnostic_for(
            "stiff (Re=1e6, Da=1, α=0.05, k=1, n=10, QUAD, ASGS)",
            (Re=1e6, Da=1.0, alpha_0=0.05, kv=1, n=10, element_type="QUAD"),
        )
    end

    if stiff_results !== nothing
        # Write to *_raw.md so the curated interpretive probe_stiff_findings.md isn't overwritten.
        report_path = joinpath(@__DIR__, "diagnostics", "probe_stiff_raw.md")
        write_report(easy_results, stiff_results, report_path)
    end
end

main()
