# test/long/CocquetExperiment/run_convergence.jl
# ==============================================================================================
# Nature & Intent:
# The definitive physical benchmark test for solver exactness on non-synthetic data. Unlike MMS, 
# there is no analytical "exact" polynomial solution, so convergence rates $O(h^{k+1})$ are extracted 
# by using the highest resolution grid as a surrogate mathematically true baseline.
# Exact grid prolongation maps coarse fields to fine mesh measure spaces $d\Omega_{ref}$ to eliminate
# cross-grid interpolation geometry noise and precisely extract the algebraic convergence slope.
#
# Mathematical Formulation Alignment:
# Proves that the VMS scheme yields dimensionally optimal $L_2$ and semi-$H_1$ continuum 
# approximation bounds in the presence of physical varying media (non-constant coefficients).
# It uses an explicit exponential porosity profile ε(y) = 0.45 + 0.55 * exp(y - 1.0) and validates
# structural bounds via orthogonal projections dynamically evaluated against fine-domain integrals.
#
# Associated Data & Endpoints:
# - `data/paper_comparison.json`: Dictates entirely mathematical scales, topological boundaries, `Re`, `c_in`, and interpolation ranges.
# - Outputs exact HDF5 convergence rates for Python plotting `plot_convergence.py`.
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON3
using HDF5
using LinearAlgebra

# Pure-Galerkin (unstabilized) execution driver for the exact Cocquet (Taylor–Hood) comparison.
# Kept out of src/ — validation tooling that reuses the production weak-form builders with mult=0.
include(joinpath(@__DIR__, "galerkin_driver.jl"))


# Section 4.2 Smooth Porosity Field
# ε(y) = 0.45 + 0.55 * exp(y - 1.0)
alpha_func(x) = 0.45 + 0.55 * exp(x[2] - 1.0)
Gridap.∇(::typeof(alpha_func)) = x -> VectorValue(0.0, 0.55 * exp(x[2] - 1.0))

function get_inflow_profile(Re::Float64, c_in::Float64)
    return x -> VectorValue(c_in * x[2] * (1.0 - x[2]), 0.0)
end
u_wall(x) = VectorValue(0.0, 0.0)


function build_solver(N::Int, config_dict, Re::Float64, c_in::Float64; porosity_order::Union{Int,Nothing}=nothing)
    # The paper uses domain [0, 2]x[0, 1] mapped up to N nodes on bounds
    # For a Cartesian grid, a grid of 2N x N elements maintains square shape (h = 1/N).
    # Since the paper specifies parameter N, we use partition = [2*N, N].
    local_config_dict = deepcopy(config_dict)
    if !haskey(local_config_dict["numerical_method"], "mesh")
        local_config_dict["numerical_method"]["mesh"] = Dict()
    end
    local_config_dict["numerical_method"]["mesh"]["partition"] = [2*N, N]
    local_config = PorousNSSolver.load_config_from_dict(local_config_dict)
    

    u_in = get_inflow_profile(Re, c_in)
    
    model = PorousNSSolver._build_default_mesh(local_config.domain, local_config.numerical_method.mesh)
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, local_config.numerical_method.element_spaces.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, local_config.numerical_method.element_spaces.k_pressure)
    
    # Generate Trial and Test spaces, dynamically inheriting boundary limits
    X, Y, kv, kp = PorousNSSolver.build_fe_spaces(
        model, 
        local_config.numerical_method.element_spaces, 
        ["inlet", "walls"], 
        [(true,true), (true,true)],
        [u_in, u_wall]
    )
    
    U, P = X
    V, Q = Y
    
    # Build a type-representative reaction law from the config so the §3.5 quadrature
    # decision sees Forchheimer's non-polynomial bump when applicable. Coefficients are
    # irrelevant for the dispatch — only the law's type drives `min_quadrature_degree`.
    local_rxn_law = local_config.physical_properties.reaction_model == "Constant_Sigma" ?
        PorousNSSolver.ConstantSigmaLaw(0.0) :
        PorousNSSolver.ForchheimerErgunLaw(0.0, 0.0)
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, local_config.numerical_method.element_spaces.k_velocity, local_rxn_law)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    # Element characteristic length for the τ stabilization. Match the manufactured-solution
    # driver (test/extended/CocquetFormMMS/run_test.jl) EXACTLY so the two tests use the same
    # convention: for the simplexified mesh, √(2·area) = the structured mesh parameter 1/N (the
    # square's side / triangle leg). The previous √(area) here was 1/(N√2) — a factor-√2 too small,
    # which only rescaled τ (rate-neutral) but made the Cocquet and MMS drivers inconsistent.
    # (Cocquet et al. report the triangle DIAMETER h=√2/N; that is √2× this leg-based value and,
    # being a constant factor, does not affect slopes — which are measured against N.)
    if local_config.numerical_method.mesh.element_type == "TRI"
        h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
    else
        h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    end
    h_cf = CellField(h_array, Ω)
    
    # Porosity interpolation order. By default it matches the velocity order `k_v` (otherwise a P1
    # porosity injects an O(h^2) modeling error that caps P2 superconvergence). `porosity_order` lets
    # a run override this — e.g. the Cocquet/Galerkin run uses P1 porosity regardless of velocity order,
    # to reproduce Cocquet et al. exactly.
    k_v = local_config.numerical_method.element_spaces.k_velocity
    porder = porosity_order === nothing ? k_v : porosity_order
    refe_alpha = ReferenceFE(lagrangian, Float64, porder)
    V_alpha = TestFESpace(model, refe_alpha, conformity=:H1)
    alpha_h = interpolate(alpha_func, V_alpha)
    
    # Return raw geometric constructs instead of operators to enable explicit OSGS projections dynamically
    return model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, local_config
end

function execute_solver(model, X, Y, dΩ, h_cf, alpha_h, refe_u, refe_p, config)
    form = PorousNSSolver.build_formulation(config.physical_properties, config.numerical_method)
    f_cf = VectorValue(config.physical_properties.f_x, config.physical_properties.f_y)
    g_cf = 0.0
    
    k = config.numerical_method.element_spaces.k_velocity
    c_1, c_2 = PorousNSSolver.get_c1_c2(typeof(form), k)
    
    ls = LUSolver()
    div_fac = config.numerical_method.solver.divergence_merit_factor
    n_floor = config.numerical_method.solver.stagnation_noise_floor
    ar_c1 = config.numerical_method.solver.armijo_c1
    ls_alpha = config.numerical_method.solver.linesearch_alpha_min
    xtol = config.numerical_method.solver.xtol
    ftol = config.numerical_method.solver.ftol
    max_inc = config.numerical_method.solver.max_increases
    
    ls_contract = config.numerical_method.solver.linesearch_contraction_factor
    max_ls = config.numerical_method.solver.max_linesearch_iterations
    
    nls_picard = PorousNSSolver.SafeNewtonSolver(ls, config.numerical_method.solver.picard_iterations, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor, max_ls, ls_contract; mode=:picard)
    nls_newton = PorousNSSolver.SafeNewtonSolver(ls, config.numerical_method.solver.newton_iterations, max_inc, xtol, ftol, ls_alpha, ar_c1, div_fac, n_floor, max_ls, ls_contract)
    
    solver_picard = FESolver(nls_picard)
    solver_newton = FESolver(nls_newton)
    
    x0 = FEFunction(X, zeros(num_free_dofs(X)))
    
    stab_cfg = config.numerical_method.stabilization
    
    # Generate unbound functional spaces specifically for pure mapping projections structurally independent of physical walls
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)
    
    setup = PorousNSSolver.FETopology(X, Y, model, Triangulation(model), dΩ, V_free, Q_free, h_cf, f_cf, alpha_h, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
    iter_solvers = PorousNSSolver.IterativeSolvers(solver_picard, solver_newton)

    success, _mms_plateau_unused, final_x0, iter_count, eval_time = PorousNSSolver.solve_system(
        setup, formulation, iter_solvers, config, x0
    )
    
    return final_x0, eval_time, iter_count
end
function run_convergence()
    println("--- Cocquet Convergence Analysis (N=200 Reference) ---")
    
    # Config filename may be passed as the first CLI argument (default: paper_comparison.json,
    # the equal-order P1/P1 + P2/P2 VMS study).
    config_file = length(ARGS) > 0 ? ARGS[1] : "paper_comparison.json"
    base_config_path = joinpath(@__DIR__, "data", config_file)
    base_config_dict = JSON3.read(read(base_config_path, String), Dict{String, Any})

    # Store dynamic script parameters before strict schema parsing drops them or warns
    Re = Float64(base_config_dict["Re"])
    c_in = Float64(base_config_dict["c_in"])
    delta = Float64(get(base_config_dict, "outlet_truncation_delta", 0.0))

    # Support mixed-order native schemas (e.g. Taylor-Hood P2/P1) directly from the JSON without equal-order script clamping
    k_v = base_config_dict["numerical_method"]["element_spaces"]["k_velocity"]
    k_p = base_config_dict["numerical_method"]["element_spaces"]["k_pressure"]

    # Optional list of (k_velocity, k_pressure) element pairs to sweep — e.g. the
    # stabilized equal-order pairs [[1,1],[2,2]] plus Taylor-Hood [2,1]. Defaults to the
    # single pair from element_spaces above. Results are stored per pair as P<kv>P<kp>.
    element_pairs = get(base_config_dict, "element_pairs", [[k_v, k_p]])

    # Optional explicit per-run list overriding the element_pairs × methods cartesian product. Each
    # entry is {k_velocity, k_pressure, method, porosity_order?}; `method` may be "ASGS"/"OSGS" (our
    # VMS) or "Galerkin" (the unstabilized Cocquet/Taylor–Hood method, solved via galerkin_driver.jl).
    # This is how the CocquetExperiment compares VMS P1/P1, VMS P2/P2 and Cocquet Galerkin P2/P1 in one run.
    comparison_runs = get(base_config_dict, "comparison_runs", nothing)

    # Remove script variables to prevent strict-schema enforcement warnings during formal validation API loading
    delete!(base_config_dict, "Re")
    delete!(base_config_dict, "c_in")
    if haskey(base_config_dict, "k_convergence_list")
        delete!(base_config_dict, "k_convergence_list")
    end
    delete!(base_config_dict, "outlet_truncation_delta")
    delete!(base_config_dict, "element_pairs")
    delete!(base_config_dict, "comparison_runs")
    
    # All physical schemas and geometrical limits are universally driven by the native test JSON payload
    # Temporary override to prevent strict parser crashing on dynamic iteration array
    original_method = base_config_dict["numerical_method"]["stabilization"]["method"]
    base_config_dict["numerical_method"]["stabilization"]["method"] = "ASGS"
    
    base_config = PorousNSSolver.load_config_from_dict(base_config_dict)
    
    base_config_dict["numerical_method"]["stabilization"]["method"] = original_method
    
    L_max = base_config.domain.bounding_box[2]
    bounding_rule = x -> x[1] <= (L_max - delta)

    # Magnitude-gap diagnostic probes (S3 / corner-localised norm). The outlet-wall
    # corners are at (L_max, y_min) and (L_max, y_max) of the bounding box; derived,
    # not hard-coded. R_list = {0.05, 0.1, 0.2} per the diagnostic plan; R=0.1 is
    # the primary plotted value. See docs/cocquet/investigation-synthesis.md (S3a / S3b).
    bbox = base_config.domain.bounding_box
    outlet_corners = ((bbox[2], bbox[3]), (bbox[2], bbox[4]))
    corner_excl_radii = [0.05, 0.1, 0.2]
    
    # Extract the target refinement nodes mathematically from the parsed structs.
    # partition limits are defined as [2*N, N], so N = partition[2] (the vertical subdivision)
    N_ref = base_config.numerical_method.mesh.partition[2]
    N_list = base_config.numerical_method.mesh.convergence_partitions
    
    results_dir = joinpath(@__DIR__, "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Output file is named after the config (e.g. paper_comparison.json -> convergence_paper_comparison.h5).
    h5_name = "convergence_$(splitext(config_file)[1]).h5"
    h5_path = joinpath(results_dir, h5_name)
    
    # Initialize the file struct once and close it immediately to free lock.
    # Re / c_in / delta are stored as file-level attributes so the plotter's title is
    # fully data-driven (no hard-coded physical scales).
    h5open(h5_path, "w") do file
        file["N_list"] = collect(N_list)
        attributes(file)["Re"] = Re
        attributes(file)["c_in"] = c_in
        attributes(file)["outlet_truncation_delta"] = delta
    end
    
    as_list(x) = x isa Vector ? convert(Vector{String}, x) : [String(x)]
    nm_dict = get(base_config_dict, "numerical_method", Dict())
    stab_dict = get(nm_dict, "stabilization", Dict())
    methods = as_list(get(stab_dict, "method", ["ASGS", "OSGS"]))

    # One convergence run for a single (element pair, method). `method` is the HDF5 group LABEL
    # ("ASGS"/"OSGS"/"Galerkin"). For "Galerkin" the config carries a valid method ("ASGS", to satisfy
    # strict validation) and the solve is dispatched to the unstabilized galerkin_driver.jl; otherwise
    # solve_system runs the VMS method. `porosity_order=nothing` ⇒ build_solver defaults to k_v.
    function do_run(k_v::Int, k_p::Int, method::String, porosity_order)
        is_galerkin = method == "Galerkin"
        config_method = is_galerkin ? "ASGS" : method      # value parsed by load_config_from_dict
        exec_fn = is_galerkin ? execute_solver_galerkin : execute_solver

        base_config_dict["numerical_method"]["element_spaces"]["k_velocity"] = k_v
        base_config_dict["numerical_method"]["element_spaces"]["k_pressure"] = k_p
        base_config_dict["numerical_method"]["stabilization"]["method"] = config_method

        po_str = porosity_order === nothing ? "k_v" : string(porosity_order)
        println("\n##########################################################################################")
        println("[#] P$(k_v)/P$(k_p) | METHOD: $method$(is_galerkin ? " (unstabilized Galerkin / Cocquet)" : "") | porosity order: $po_str")
        println("##########################################################################################")

        println("\n   [+] Assembling High-Fidelity Reference Mesh Solution (N = $N_ref) natively...")
        base_config_dict["output"]["basename"] = "cocquet_ref_$(method)_P$(k_v)P$(k_p)_N$(N_ref)"
        mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref = build_solver(N_ref, base_config_dict, Re, c_in; porosity_order=porosity_order)
        xh_ref, time_ref, iters_ref = exec_fn(mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref)
        u_ref, p_ref = xh_ref

        iu_ref = Gridap.FESpaces.Interpolable(u_ref)
        ip_ref = Gridap.FESpaces.Interpolable(p_ref)

        PorousNSSolver.export_results(cfg_ref, mod_ref, u_ref, p_ref, "alpha" => alpha_ref)

        errors_l2_u = Float64[]; errors_h1_u = Float64[]
        errors_l2_p = Float64[]; errors_h1_p = Float64[]
        eval_times = Float64[]; eval_iters = Int[]

        # Diagnostic probes for the magnitude-gap investigation
        # (docs/cocquet/investigation-synthesis.md, S3).
        #   fraction_cellavg = ‖P₀ e_h‖² / ‖e_h‖² — share captured by piecewise constants
        #                      on the coarse mesh. Tends to 1 generically as h → 0 for any
        #                      smooth converging error; useful for within-cell-oscillation
        #                      diagnostics (P₁ vs P₂) but NOT a discriminator for a globally-
        #                      applied gauge/BC bias (self-reference cancels such modes).
        #   chi_Omega        = |Ω|·‖ē_Ω‖² / ‖e_h‖² — share carried by the single domain-wide
        #                      mean ē_Ω = (∫_Ω e)/|Ω|. Self-reference-invariant: χ_Ω → 0 for
        #                      any zero-mean error; only a true rigid offset of e_h produces
        #                      χ_Ω → 1. This is the actual S3 discriminator.
        cellavg_frac_u = Float64[]; cellavg_frac_p = Float64[]
        chi_Omega_u    = Float64[]; chi_Omega_p    = Float64[]
        l2_cellavg_u   = Float64[]; l2_cellavg_p   = Float64[]
        l2_domainmean_u = Float64[]; l2_domainmean_p = Float64[]
        # Corner-excluded norms (one row per N, one column per R in corner_excl_radii).
        n_radii = length(corner_excl_radii)
        l2_eu_corner_excl = Matrix{Float64}(undef, length(N_list), n_radii)
        h1_eu_corner_excl = Matrix{Float64}(undef, length(N_list), n_radii)
        l2_ep_corner_excl = Matrix{Float64}(undef, length(N_list), n_radii)
        h1_ep_corner_excl = Matrix{Float64}(undef, length(N_list), n_radii)

        for (n_idx, N) in enumerate(N_list)
            println("\n   ==============================================================")
            println("   [+] Launching Coarse Grid Algebraic Evaluation Space for N = $N")
            println("   ==============================================================")
            base_config_dict["output"]["basename"] = "cocquet_$(method)_P$(k_v)P$(k_p)_N$(N)"
            mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h = build_solver(N, base_config_dict, Re, c_in; porosity_order=porosity_order)

            xh_h, time_h, iters_h = exec_fn(mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h)
            u_h, p_h = xh_h

            V_h_free = TestFESpace(mod_h, ru_h, conformity=:H1)
            Q_h_free = TestFESpace(mod_h, rp_h, conformity=:H1)

            # Multi-mask call: cross-mesh interpolation built once per field, applied under
            # the full bounding rule and the three corner-exclusion radii in one pass. The
            # first mask is the full bounding rule (matches the legacy `compute_reference_errors`
            # output); the next |corner_excl_radii| are the radius-sweep variants.
            corner_masks = [let R2 = R*R, base = bounding_rule
                                x -> begin
                                    base(x) || return false
                                    @inbounds for c in outlet_corners
                                        d2 = (x[1] - c[1])^2 + (x[2] - c[2])^2
                                        d2 ≤ R2 && return false
                                    end
                                    return true
                                end
                            end for R in corner_excl_radii]
            mask_set = vcat([bounding_rule], corner_masks)

            res_u_mm = PorousNSSolver.compute_reference_errors_multimask(u_h, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref; masks=mask_set)
            res_p_mm = PorousNSSolver.compute_reference_errors_multimask(p_h, p_ref, ip_ref, Q_h_free, dΩ_h, dΩ_ref; masks=mask_set)

            # Index 1 is the bounding-rule (full) variant — same as the legacy call would have produced.
            full_u = res_u_mm.per_mask[1]; full_p = res_p_mm.per_mask[1]
            l2_eu_nested, h1_eu_nested = full_u.l2_nested, full_u.h1_nested
            l2_eu,        h1_eu        = full_u.l2_cons,   full_u.h1_cons
            l2_ep_nested, h1_ep_nested = full_p.l2_nested, full_p.h1_nested
            l2_ep,        h1_ep        = full_p.l2_cons,   full_p.h1_cons
            eu_nested, eu_cons = res_u_mm.e_nested, res_u_mm.e_cons
            ep_nested, ep_cons = res_p_mm.e_nested, res_p_mm.e_cons

            for (r_idx, _) in enumerate(corner_excl_radii)
                excl_u = res_u_mm.per_mask[r_idx + 1]
                excl_p = res_p_mm.per_mask[r_idx + 1]
                l2_eu_corner_excl[n_idx, r_idx] = excl_u.l2_cons; h1_eu_corner_excl[n_idx, r_idx] = excl_u.h1_cons
                l2_ep_corner_excl[n_idx, r_idx] = excl_p.l2_cons; h1_ep_corner_excl[n_idx, r_idx] = excl_p.h1_cons
            end

            # S3 probes: cell-average mode fraction AND domain-mean fraction χ_Ω.
            # The second is the actual discriminator for a globally-applied gauge/BC
            # bias surviving self-reference; the first is informative about cell-scale
            # smoothness only.
            md_u = PorousNSSolver.compute_mode_decomposition(u_h, iu_ref, V_h_free, dΩ_h)
            md_p = PorousNSSolver.compute_mode_decomposition(p_h, ip_ref, Q_h_free, dΩ_h)

            PorousNSSolver.export_results(cfg_h, mod_h, u_h, p_h, "alpha" => alpha_h, "e_u" => eu_nested, "e_p" => ep_nested)

            println("   [+] L2-norms | L2(u): ", l2_eu, " | L2(p): ", l2_ep)
            println("   [+] H1-seminorms | semiH1(u): ", h1_eu, " | semiH1(p): ", h1_ep)
            println("   [+] S3 χ_Ω (domain-mean share) | u: ", round(md_u.chi_Omega, sigdigits=4),
                    " | p: ", round(md_p.chi_Omega, sigdigits=4),
                    "   [cell-avg share | u: ", round(md_u.fraction_cellavg, sigdigits=4),
                    " p: ", round(md_p.fraction_cellavg, sigdigits=4), "]")
            r_show = corner_excl_radii[min(2, end)]  # R=0.1 by default
            r_idx_show = min(2, n_radii)
            println("   [+] Corner-excluded L²(u) (R=$(r_show)): ", l2_eu_corner_excl[n_idx, r_idx_show],
                    " | L²(p): ", l2_ep_corner_excl[n_idx, r_idx_show])

            push!(errors_l2_u, l2_eu); push!(errors_h1_u, h1_eu)
            push!(errors_l2_p, l2_ep); push!(errors_h1_p, h1_ep)
            push!(eval_times, time_h); push!(eval_iters, iters_h)
            push!(cellavg_frac_u, md_u.fraction_cellavg); push!(cellavg_frac_p, md_p.fraction_cellavg)
            push!(chi_Omega_u, md_u.chi_Omega);   push!(chi_Omega_p, md_p.chi_Omega)
            push!(l2_cellavg_u, md_u.l2_cellavg); push!(l2_cellavg_p, md_p.l2_cellavg)
            push!(l2_domainmean_u, md_u.l2_domainmean); push!(l2_domainmean_p, md_p.l2_domainmean)

            GC.gc()  # free UMFPACK C-pointer memory between solves
        end

        println("Writing $(method)/P$(k_v)P$(k_p) slice to HDF5...")
        h5open(h5_path, "r+") do file
            group_name = "$method/P$(k_v)P$(k_p)"
            if haskey(file, group_name)
                delete_object(file, group_name)
            end
            g = create_group(file, group_name)
            g["errors_l2_u"] = errors_l2_u
            g["errors_h1_u"] = errors_h1_u
            g["errors_l2_p"] = errors_l2_p
            g["errors_h1_p"] = errors_h1_p
            g["eval_times"] = eval_times
            g["eval_iters"] = eval_iters
            # S3 / corner-excluded probes for the magnitude-gap investigation.
            # `chi_Omega_*` is the self-reference-invariant domain-mean fraction;
            # `cellavg_frac_*` is retained for the within-cell-oscillation diagnostic.
            g["cellavg_frac_u"]    = cellavg_frac_u
            g["cellavg_frac_p"]    = cellavg_frac_p
            g["chi_Omega_u"]       = chi_Omega_u
            g["chi_Omega_p"]       = chi_Omega_p
            g["l2_cellavg_u"]      = l2_cellavg_u
            g["l2_cellavg_p"]      = l2_cellavg_p
            g["l2_domainmean_u"]   = l2_domainmean_u
            g["l2_domainmean_p"]   = l2_domainmean_p
            g["l2_eu_corner_excl"] = l2_eu_corner_excl
            g["h1_eu_corner_excl"] = h1_eu_corner_excl
            g["l2_ep_corner_excl"] = l2_ep_corner_excl
            g["h1_ep_corner_excl"] = h1_ep_corner_excl
            attributes(g)["corner_excl_radii"] = corner_excl_radii
            attributes(g)["outlet_corners_x"]  = collect(c[1] for c in outlet_corners)
            attributes(g)["outlet_corners_y"]  = collect(c[2] for c in outlet_corners)
            attributes(g)["total_time_s"] = sum(eval_times)
            attributes(g)["total_iters"] = sum(eval_iters)
            attributes(g)["outlet_truncation_delta"] = delta
            attributes(g)["porosity_order"] = porosity_order === nothing ? -1 : Int(porosity_order)
        end
    end

    if comparison_runs !== nothing
        # Explicit per-run comparison (e.g. VMS P1/P1, VMS P2/P2, Cocquet Galerkin P2/P1).
        for run in comparison_runs
            kv = Int(run["k_velocity"]); kp = Int(run["k_pressure"])
            m  = String(run["method"])
            po = haskey(run, "porosity_order") ? Int(run["porosity_order"]) : nothing
            do_run(kv, kp, m, po)
        end
    else
        # Backward-compatible element_pairs × methods cartesian product.
        for ep in element_pairs
            for method in methods
                do_run(Int(ep[1]), Int(ep[2]), method, nothing)
            end
        end
    end
    println("\nConvergence Data Generated and Exported to $h5_name")
end

run_convergence()
