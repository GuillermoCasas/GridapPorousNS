# test/extended/CocquetTubeTest/run_convergence.jl
# ==============================================================================================
# UNIFIED Cocquet tube-flow convergence driver.
#
# One config-driven driver for the whole family of Cocquet lid/tube-flow convergence studies that
# used to live as ~9 near-duplicate sibling directories. Every experimental variant is now a JSON
# config under `data/<name>/`; this driver reads one such config and writes its results to
# the parallel folder `results/<name>/` with a UNIFORM structure and error-diagnostic suite, so all
# variants are directly comparable.
#
# The variant axes are ALL config-driven (test-harness keys read with `get(dict, key, default)` and
# stripped before strict-schema parsing, exactly like Re / c_in):
#   * mesh_generator          "STRUCTURED" (default) | "UNSTRUCTURED" (gmsh Delaunay)
#   * wall_divisions          "uniform" (default)    | "freefem" (N-per-border, walls 2x coarser)
#   * freefem_mesh_dir        ""        | dir (relative to the config's folder) of FreeFem .msh files
#   * mesh_algorithm          gmsh meshing algorithm id (default 5)
#   * boundary_policy         "standard" (default: inlet+walls Dirichlet, outlet natural)
#                             | "all_dirichlet" (outlet also Dirichlet with the inlet parabola)
#                             | "modified_corner" (structured mesh with outlet-corner tags released)
#   * alpha_profile           "exponential" (default: 0.45+0.55*e^{y-1}) | "constant" (alpha == 1)
#   * quadrature_degree_bonus extra quadrature degree added to the base rule (default 0)
#   * viscous_operator_type / reaction  — already carried by the strict config
#   * method labels (via comparison_runs / stabilization.method): "ASGS" / "OSGS" (VMS), "Galerkin"
#     (unstabilized Taylor-Hood, galerkin_driver.jl) and "Galerkin_LiteralPicard" (capped Picard,
#     literal_picard_driver.jl).
#
# Run (from this directory):
#   julia --project=../../.. run_convergence.jl data/structured/paper_comparison.json
#   python plot_convergence.py structured
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON3
using HDF5
using LinearAlgebra

# Unstabilized-Galerkin (Cocquet Taylor-Hood) and literal-capped-Picard execution drivers. Kept out
# of src/ — validation tooling that reuses the production weak-form builders with mult=0.
include(joinpath(@__DIR__, "galerkin_driver.jl"))
include(joinpath(@__DIR__, "literal_picard_driver.jl"))


# --- Porosity profiles (Section 4.2). Two named functions with explicit gradient overloads so the
# stabilization sees the exact ∇alpha; select by the `alpha_profile` config key. -----------------
alpha_exponential(x) = 0.45 + 0.55 * exp(x[2] - 1.0)
Gridap.∇(::typeof(alpha_exponential)) = x -> VectorValue(0.0, 0.55 * exp(x[2] - 1.0))
alpha_constant(x) = 1.0
Gridap.∇(::typeof(alpha_constant)) = x -> VectorValue(0.0, 0.0)

function get_inflow_profile(Re::Float64, c_in::Float64)
    return x -> VectorValue(c_in * x[2] * (1.0 - x[2]), 0.0)
end
u_wall(x) = VectorValue(0.0, 0.0)


# Local mirror of PorousNSSolver._build_default_mesh with the one edit that entities 2 and 4 (the
# outlet-side corner vertices in Gridap's 2D Cartesian labeling) are OMITTED from the "walls" tag,
# so those DOFs become free — the `modified_corner` boundary policy (docs/cocquet, H1 falsification).
function _build_modified_corner_mesh(domain_cfg, mesh_cfg)
    domain = Tuple(domain_cfg.bounding_box)
    partition = Tuple(mesh_cfg.partition)

    if mesh_cfg.element_type == "TRI"
        model = CartesianDiscreteModel(domain, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain, partition)
    end

    labels = Gridap.Geometry.get_face_labeling(model)
    Gridap.Geometry.add_tag_from_tags!(labels, "inlet",  [7])
    Gridap.Geometry.add_tag_from_tags!(labels, "outlet", [8])
    Gridap.Geometry.add_tag_from_tags!(labels, "walls",  [1, 3, 5, 6])
    return model
end


function build_solver(N::Int, config_dict, Re::Float64, c_in::Float64;
                      porosity_order::Union{Int,Nothing}=nothing,
                      mesh_generator::String="STRUCTURED",
                      wall_divisions::Symbol=:uniform,
                      mesh_algorithm::Int=5,
                      mesh_file::String="",
                      boundary_policy::String="standard",
                      alpha_profile::String="exponential",
                      quadrature_degree_bonus::Int=0)
    # The paper uses domain [0,2]x[0,1]. STRUCTURED: partition [2N, N] keeps square cells (h = 1/N).
    # UNSTRUCTURED: partition is unused — build_unstructured_model(N) targets edge length 1/N.
    local_config_dict = deepcopy(config_dict)
    if !haskey(local_config_dict["numerical_method"], "mesh")
        local_config_dict["numerical_method"]["mesh"] = Dict()
    end
    local_config_dict["numerical_method"]["mesh"]["partition"] = [2*N, N]
    local_config = PorousNSSolver.load_config_from_dict(local_config_dict)

    u_in = get_inflow_profile(Re, c_in)

    # MESH SOURCE — resolution order: FreeFem .msh from disk > gmsh unstructured > modified-corner
    # structured > default structured. All expose the named tags "inlet"/"walls" (+ "outlet").
    is_unstructured = mesh_generator == "UNSTRUCTURED"
    model = if !isempty(mesh_file)
        PorousNSSolver.load_freefem_mesh(mesh_file)
    elseif is_unstructured
        PorousNSSolver.build_unstructured_model(N; domain=Tuple(local_config.domain.bounding_box),
                                                 wall_divisions=wall_divisions,
                                                 algorithm=mesh_algorithm)
    elseif boundary_policy == "modified_corner"
        _build_modified_corner_mesh(local_config.domain, local_config.numerical_method.mesh)
    else
        PorousNSSolver._build_default_mesh(local_config.domain, local_config.numerical_method.mesh)
    end

    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, local_config.numerical_method.element_spaces.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, local_config.numerical_method.element_spaces.k_pressure)

    # BOUNDARY POLICY — "all_dirichlet" additionally pins the outlet with the inlet parabola so the
    # wall/outlet corners become Dirichlet-Dirichlet (the mixed-BC-corner diagnostic). Otherwise the
    # outlet stays natural (Neumann) and only inlet+walls are Dirichlet.
    if boundary_policy == "all_dirichlet"
        dtags  = ["inlet", "walls", "outlet"]
        dmasks = [(true, true), (true, true), (true, true)]
        dfns   = [u_in, u_wall, u_in]
    else
        dtags  = ["inlet", "walls"]
        dmasks = [(true, true), (true, true)]
        dfns   = [u_in, u_wall]
    end
    X, Y, kv, kp = PorousNSSolver.build_fe_spaces(
        model,
        local_config.numerical_method.element_spaces,
        dtags, dmasks, dfns,
    )

    U, P = X
    V, Q = Y

    # Type-representative reaction law drives the §3.5 quadrature decision (coefficients irrelevant).
    local_rxn_law = local_config.physical_properties.reaction_model == "Constant_Sigma" ?
        PorousNSSolver.ConstantSigmaLaw(0.0) :
        PorousNSSolver.ForchheimerErgunLaw(0.0, 0.0)
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, local_config.numerical_method.element_spaces.k_velocity, local_rxn_law)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree + quadrature_degree_bonus)

    # Element characteristic length for τ. For triangles (simplexified-Cartesian, gmsh or FreeFem)
    # use √(2·area) = the structured leg / a per-cell length; for quads √(area). Matches CocquetFormMMS.
    is_tri = local_config.numerical_method.mesh.element_type == "TRI" || is_unstructured || !isempty(mesh_file)
    h_array = is_tri ?
        lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω)) :
        lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    h_cf = CellField(h_array, Ω)

    # Porosity interpolation. Defaults to velocity order k_v; `porosity_order` lets a run override
    # it (the Cocquet/Galerkin comparison uses P1 porosity regardless of velocity order).
    alpha_func = alpha_profile == "constant" ? alpha_constant : alpha_exponential
    k_v = local_config.numerical_method.element_spaces.k_velocity
    porder = porosity_order === nothing ? k_v : porosity_order
    refe_alpha = ReferenceFE(lagrangian, Float64, porder)
    V_alpha = TestFESpace(model, refe_alpha, conformity=:H1)
    alpha_h = interpolate(alpha_func, V_alpha)

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

    # Unbound spaces for pure projections (independent of physical walls).
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)

    setup = PorousNSSolver.FETopology(X, Y, model, Triangulation(model), dΩ, V_free, Q_free, h_cf, f_cf, alpha_h, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
    iter_solvers = PorousNSSolver.StageSolvers(solver_picard, solver_newton)

    success, _mms_plateau_unused, final_x0, iter_count, eval_time = PorousNSSolver.solve_system(
        setup, formulation, iter_solvers, config, x0
    )

    return final_x0, eval_time, iter_count
end

function run_convergence()
    # Config path is the CLI argument, resolved relative to this driver's directory if not absolute.
    # The variant NAME is the parent folder of the config (data/<name>/…), which also names
    # the parallel results folder results/<name>/.
    rel_config = length(ARGS) > 0 ? ARGS[1] : joinpath("data", "structured", "paper_comparison.json")
    base_config_path = isabspath(rel_config) ? rel_config : joinpath(@__DIR__, rel_config)
    config_dir = dirname(base_config_path)
    name = basename(config_dir)
    config_file = basename(base_config_path)

    println("--- Cocquet Tube-Flow Convergence :: variant '$name' ($(config_file)) ---")

    base_config_dict = JSON3.read(read(base_config_path, String), Dict{String, Any})

    # Store dynamic / test-harness parameters before strict schema parsing drops or warns on them.
    Re   = Float64(base_config_dict["Re"])
    c_in = Float64(base_config_dict["c_in"])
    delta = Float64(get(base_config_dict, "outlet_truncation_delta", 0.0))

    # Variant axes (all with defaults matching the canonical structured benchmark).
    mesh_generator = String(get(base_config_dict, "mesh_generator", "STRUCTURED"))
    wall_divisions = Symbol(String(get(base_config_dict, "wall_divisions", "uniform")))
    mesh_algorithm = Int(get(base_config_dict, "mesh_algorithm", 5))
    freefem_mesh_dir = String(get(base_config_dict, "freefem_mesh_dir", ""))
    boundary_policy = String(get(base_config_dict, "boundary_policy", "standard"))
    alpha_profile   = String(get(base_config_dict, "alpha_profile", "exponential"))
    quadrature_degree_bonus = Int(get(base_config_dict, "quadrature_degree_bonus", 0))

    k_v = base_config_dict["numerical_method"]["element_spaces"]["k_velocity"]
    k_p = base_config_dict["numerical_method"]["element_spaces"]["k_pressure"]

    element_pairs   = get(base_config_dict, "element_pairs", [[k_v, k_p]])
    comparison_runs = get(base_config_dict, "comparison_runs", nothing)

    # Remove script-only keys before strict validation.
    for scriptkey in ("Re", "c_in", "k_convergence_list", "outlet_truncation_delta",
                      "mesh_generator", "wall_divisions", "mesh_algorithm", "freefem_mesh_dir",
                      "boundary_policy", "alpha_profile", "quadrature_degree_bonus",
                      "element_pairs", "comparison_runs")
        haskey(base_config_dict, scriptkey) && delete!(base_config_dict, scriptkey)
    end

    # Temporarily coerce a valid single method so the strict parser accepts the payload.
    original_method = base_config_dict["numerical_method"]["stabilization"]["method"]
    base_config_dict["numerical_method"]["stabilization"]["method"] = "ASGS"
    base_config = PorousNSSolver.load_config_from_dict(base_config_dict)
    base_config_dict["numerical_method"]["stabilization"]["method"] = original_method

    L_max = base_config.domain.bounding_box[2]
    bounding_rule = x -> x[1] <= (L_max - delta)

    # Outlet-wall corners (derived from the bounding box) for the corner-localised norm probes.
    bbox = base_config.domain.bounding_box
    outlet_corners = ((bbox[2], bbox[3]), (bbox[2], bbox[4]))
    corner_excl_radii = [0.05, 0.1, 0.2]

    # partition = [2N, N] ⇒ N is partition[2]; reference N and the coarse sweep.
    N_ref  = base_config.numerical_method.mesh.partition[2]
    N_list = base_config.numerical_method.mesh.convergence_partitions

    # UNIFORM parallel results layout: results/<name>/{convergence.h5, config.json, vtk/}.
    results_dir = joinpath(@__DIR__, "results", name)
    vtk_dir = joinpath(results_dir, "vtk")
    isdir(results_dir) || mkpath(results_dir)
    isdir(vtk_dir) || mkpath(vtk_dir)
    # Self-describing: keep the exact config next to its results.
    cp(base_config_path, joinpath(results_dir, "config.json"); force=true)
    # Route VTU exports under results/<name>/vtk/ (absolute, so the run is cwd-independent).
    base_config_dict["output"]["directory"] = vtk_dir

    h5_path = joinpath(results_dir, "convergence.h5")

    # Initialise the file and record file-level provenance attributes.
    h5open(h5_path, "w") do file
        file["N_list"] = collect(N_list)
        attributes(file)["Re"] = Re
        attributes(file)["c_in"] = c_in
        attributes(file)["outlet_truncation_delta"] = delta
        attributes(file)["mesh_generator"] = mesh_generator
        attributes(file)["wall_divisions"] = String(wall_divisions)
        attributes(file)["boundary_policy"] = boundary_policy
        attributes(file)["alpha_profile"] = alpha_profile
        attributes(file)["config_file"] = config_file
    end

    println("    [variant] mesh=$mesh_generator/$wall_divisions  boundary=$boundary_policy  alpha=$alpha_profile  quad_bonus=$quadrature_degree_bonus",
            isempty(freefem_mesh_dir) ? "" : "  freefem_mesh_dir=$freefem_mesh_dir")

    # Resolve a FreeFem .msh path for a given N (empty ⇒ generate instead). freefem_mesh_dir is
    # relative to the config folder, so each data/<name>/ is self-contained.
    function freefem_path_for(N)
        isempty(freefem_mesh_dir) && return ""
        if N == N_ref
            p = joinpath(config_dir, freefem_mesh_dir, "mesh_fig2_N$(N)_reference.msh")
            isfile(p) && return p
        end
        return joinpath(config_dir, freefem_mesh_dir, "mesh_fig2_N$(N).msh")
    end

    # Shared kwargs for build_solver so the reference and coarse solves are always identical.
    bs_kwargs(N) = (mesh_generator=mesh_generator, wall_divisions=wall_divisions,
                    mesh_algorithm=mesh_algorithm, mesh_file=freefem_path_for(N),
                    boundary_policy=boundary_policy, alpha_profile=alpha_profile,
                    quadrature_degree_bonus=quadrature_degree_bonus)

    # One convergence run for a single (element pair, method). `method` is the HDF5 group LABEL and
    # selects the execution path: "Galerkin" ⇒ unstabilized Taylor-Hood; "Galerkin_LiteralPicard" ⇒
    # capped-Picard Cocquet protocol; anything else ("ASGS"/"OSGS") ⇒ VMS solve_system.
    function do_run(k_v::Int, k_p::Int, method::String, porosity_order)
        exec_fn = if method == "Galerkin"
            execute_solver_galerkin
        elseif method == "Galerkin_LiteralPicard"
            execute_solver_galerkin_literal_picard
        else
            execute_solver
        end
        is_galerkin_label = method == "Galerkin" || method == "Galerkin_LiteralPicard"
        config_method = is_galerkin_label ? "ASGS" : method   # value parsed by load_config_from_dict

        base_config_dict["numerical_method"]["element_spaces"]["k_velocity"] = k_v
        base_config_dict["numerical_method"]["element_spaces"]["k_pressure"] = k_p
        base_config_dict["numerical_method"]["stabilization"]["method"] = config_method

        po_str = porosity_order === nothing ? "k_v" : string(porosity_order)
        println("\n##########################################################################################")
        println("[#] P$(k_v)/P$(k_p) | METHOD: $method$(is_galerkin_label ? " (unstabilized Galerkin / Cocquet)" : "") | porosity order: $po_str")
        println("##########################################################################################")

        println("\n   [+] Assembling High-Fidelity Reference Mesh Solution (N = $N_ref) natively...")
        base_config_dict["output"]["basename"] = "cocquet_ref_$(name)_$(method)_P$(k_v)P$(k_p)_N$(N_ref)"
        mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref = build_solver(N_ref, base_config_dict, Re, c_in; porosity_order=porosity_order, bs_kwargs(N_ref)...)
        xh_ref, time_ref, iters_ref = exec_fn(mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref)
        u_ref, p_ref = xh_ref

        # Tolerant cross-mesh point-location. KDTree with a wide candidate set is a strict superset of
        # the default nearest-vertex search: it is REQUIRED for non-nested (unstructured/FreeFem)
        # meshes and harmless (rate-neutral) on structured ones, so every variant uses it uniformly.
        ksearch = Gridap.CellData.KDTreeSearch(num_nearest_vertices=10)
        iu_ref = Gridap.FESpaces.Interpolable(u_ref; searchmethod=ksearch)
        ip_ref = Gridap.FESpaces.Interpolable(p_ref; searchmethod=ksearch)

        PorousNSSolver.export_results(cfg_ref, mod_ref, u_ref, p_ref, "alpha" => alpha_ref)

        errors_l2_u = Float64[]; errors_h1_u = Float64[]
        errors_l2_p = Float64[]; errors_h1_p = Float64[]
        # Trial-projection metric (reference projected onto the coarse TRIAL space).
        errors_l2_u_trial = Float64[]; errors_h1_u_trial = Float64[]
        errors_l2_p_trial = Float64[]; errors_h1_p_trial = Float64[]
        eval_times = Float64[]; eval_iters = Int[]

        # S3 magnitude-gap probes (see docs/cocquet):
        #   cellavg_frac = ‖P₀ e‖²/‖e‖² (within-cell smoothness);  chi_Omega = |Ω|‖ē_Ω‖²/‖e‖²
        #   (self-reference-invariant domain-mean share — the actual S3 discriminator).
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
            base_config_dict["output"]["basename"] = "cocquet_$(name)_$(method)_P$(k_v)P$(k_p)_N$(N)"
            mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h = build_solver(N, base_config_dict, Re, c_in; porosity_order=porosity_order, bs_kwargs(N)...)

            xh_h, time_h, iters_h = exec_fn(mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h)
            u_h, p_h = xh_h
            U_h, P_h = X_h

            V_h_free = TestFESpace(mod_h, ru_h, conformity=:H1)
            Q_h_free = TestFESpace(mod_h, rp_h, conformity=:H1)

            # Multi-mask cross-mesh errors: full bounding rule (index 1) + one corner-exclusion radius
            # each. The heavy cross-mesh interpolation is built once and shared across masks.
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

            res_u_mm = PorousNSSolver.compute_reference_errors_multimask(u_h, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref; masks=mask_set, search_method=ksearch)
            res_p_mm = PorousNSSolver.compute_reference_errors_multimask(p_h, p_ref, ip_ref, Q_h_free, dΩ_h, dΩ_ref; masks=mask_set, search_method=ksearch)

            full_u = res_u_mm.per_mask[1]; full_p = res_p_mm.per_mask[1]
            l2_eu, h1_eu = full_u.l2_cons, full_u.h1_cons
            l2_ep, h1_ep = full_p.l2_cons, full_p.h1_cons
            eu_nested = res_u_mm.e_nested
            ep_nested = res_p_mm.e_nested

            for (r_idx, _) in enumerate(corner_excl_radii)
                excl_u = res_u_mm.per_mask[r_idx + 1]
                excl_p = res_p_mm.per_mask[r_idx + 1]
                l2_eu_corner_excl[n_idx, r_idx] = excl_u.l2_cons; h1_eu_corner_excl[n_idx, r_idx] = excl_u.h1_cons
                l2_ep_corner_excl[n_idx, r_idx] = excl_p.l2_cons; h1_ep_corner_excl[n_idx, r_idx] = excl_p.h1_cons
            end

            # Trial-projection errors (reference → coarse trial space, integrated on the coarse mesh).
            l2_eu_trial, h1_eu_trial = PorousNSSolver.compute_trial_projection_errors(u_h, iu_ref, U_h, dΩ_h; filter_func=bounding_rule)
            l2_ep_trial, h1_ep_trial = PorousNSSolver.compute_trial_projection_errors(p_h, ip_ref, P_h, dΩ_h; filter_func=bounding_rule)

            # S3 mode decomposition (uses iu_ref/ip_ref, which carry the KDTree search).
            md_u = PorousNSSolver.compute_mode_decomposition(u_h, iu_ref, V_h_free, dΩ_h)
            md_p = PorousNSSolver.compute_mode_decomposition(p_h, ip_ref, Q_h_free, dΩ_h)

            PorousNSSolver.export_results(cfg_h, mod_h, u_h, p_h, "alpha" => alpha_h, "e_u" => eu_nested, "e_p" => ep_nested)

            println("   [+] L2-norms | L2(u): ", l2_eu, " | L2(p): ", l2_ep)
            println("   [+] H1-seminorms | semiH1(u): ", h1_eu, " | semiH1(p): ", h1_ep)
            println("   [+] TRIAL | L2(u): ", l2_eu_trial, " | L2(p): ", l2_ep_trial)
            println("   [+] S3 χ_Ω | u: ", round(md_u.chi_Omega, sigdigits=4), " | p: ", round(md_p.chi_Omega, sigdigits=4))
            r_idx_show = min(2, n_radii)
            println("   [+] Corner-excluded L²(u) (R=$(corner_excl_radii[r_idx_show])): ", l2_eu_corner_excl[n_idx, r_idx_show],
                    " | L²(p): ", l2_ep_corner_excl[n_idx, r_idx_show])

            push!(errors_l2_u, l2_eu); push!(errors_h1_u, h1_eu)
            push!(errors_l2_p, l2_ep); push!(errors_h1_p, h1_ep)
            push!(errors_l2_u_trial, l2_eu_trial); push!(errors_h1_u_trial, h1_eu_trial)
            push!(errors_l2_p_trial, l2_ep_trial); push!(errors_h1_p_trial, h1_ep_trial)
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
            haskey(file, group_name) && delete_object(file, group_name)
            g = create_group(file, group_name)
            g["errors_l2_u"] = errors_l2_u
            g["errors_h1_u"] = errors_h1_u
            g["errors_l2_p"] = errors_l2_p
            g["errors_h1_p"] = errors_h1_p
            g["errors_l2_u_trial"] = errors_l2_u_trial
            g["errors_h1_u_trial"] = errors_h1_u_trial
            g["errors_l2_p_trial"] = errors_l2_p_trial
            g["errors_h1_p_trial"] = errors_h1_p_trial
            g["eval_times"] = eval_times
            g["eval_iters"] = eval_iters
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
            attributes(g)["ref_iters"] = iters_ref
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
        as_list(x) = x isa Vector ? convert(Vector{String}, x) : [String(x)]
        nm_dict = get(base_config_dict, "numerical_method", Dict())
        stab_dict = get(nm_dict, "stabilization", Dict())
        methods = as_list(get(stab_dict, "method", ["ASGS", "OSGS"]))
        for ep in element_pairs
            for method in methods
                do_run(Int(ep[1]), Int(ep[2]), method, nothing)
            end
        end
    end
    println("\nConvergence data for variant '$name' exported to ", relpath(h5_path, @__DIR__))
end

# Only auto-run when invoked as a script; allow `include` for reuse of build_solver/execute_solver.
if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence()
end
