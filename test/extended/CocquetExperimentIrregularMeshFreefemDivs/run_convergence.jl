# test/extended/CocquetExperimentIrregularMeshFreefemDivs/run_convergence.jl
# ==============================================================================================
# Nature & Intent:
# A single-variable FLIP off `test/extended/CocquetExperimentIrregularMesh`: same gmsh Delaunay
# unstructured mesh, same 3-way comparison (VMS P1/P1, VMS P2/P2, Cocquet Galerkin P2/P1), same
# self-reference convergence methodology. ONE difference: the boundary-divisions policy is
# `wall_divisions = :freefem`, which gives N segments per boundary part regardless of length —
# the LITERAL `buildmesh(a(N)+b(N)+c(N)+d(N))` behaviour of FreeFem++. On our [0,2]×[0,1] domain
# this leaves the walls (length 2) 2× coarser per edge than the inlet/outlet (length 1), an
# anisotropy that the standard `:uniform` IrregularMesh sibling deliberately avoided by giving
# walls 2N divisions.
#
# Purpose: complete the FreeFem-parity experiment chain. CocquetExperimentIrregularMesh proved
# the unstructured Delaunay topology recovers Cocquet's reported ~O(h²) slope on P2/P2. This
# sibling asks: does adopting FreeFem's literal (anisotropic) boundary-divisions recipe push the
# slope further toward Cocquet's reported values, or does the anisotropy at the walls actually
# hurt? Already-existing diagnostic `freefem_recipe_diag.jl` exercises the mode at a single
# (N_coarse, N_ref) pair; this driver runs the full convergence sweep in the same config-driven
# pipeline so the result can be plotted side-by-side with the `:uniform` benchmark.
#
# The wall_divisions choice is read from the config's top-level "wall_divisions" key
# ("freefem" here), stripped before strict schema parsing (exactly how mesh_generator/Re/c_in
# are handled), so no shared `src/config.jl` schema change is needed.
#
# Run: cd test/extended/CocquetExperimentIrregularMeshFreefemDivs && \
#      julia --project=../../.. run_convergence.jl paper_comparison_irregular_freefem_divs.json && \
#      python plot_convergence.py convergence_paper_comparison_irregular_freefem_divs.h5
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON3
using HDF5
using LinearAlgebra

# Pure-Galerkin (unstabilized) execution driver for the exact Cocquet (Taylor–Hood) comparison.
# Reuse the benchmark's driver unchanged — single source of truth, kept out of src/.
include(joinpath(@__DIR__, "..", "CocquetExperiment", "galerkin_driver.jl"))


# Section 4.2 Smooth Porosity Field
# ε(y) = 0.45 + 0.55 * exp(y - 1.0)
alpha_func(x) = 0.45 + 0.55 * exp(x[2] - 1.0)
Gridap.∇(::typeof(alpha_func)) = x -> VectorValue(0.0, 0.55 * exp(x[2] - 1.0))

function get_inflow_profile(Re::Float64, c_in::Float64)
    return x -> VectorValue(c_in * x[2] * (1.0 - x[2]), 0.0)
end
u_wall(x) = VectorValue(0.0, 0.0)


function build_solver(N::Int, config_dict, Re::Float64, c_in::Float64;
                      porosity_order::Union{Int,Nothing}=nothing,
                      mesh_generator::String="STRUCTURED",
                      wall_divisions::Symbol=:uniform,
                      mesh_algorithm::Int=5,
                      mesh_file::String="")
    # The paper uses domain [0, 2]x[0, 1] mapped up to N nodes on bounds. For the STRUCTURED mesh a
    # grid of 2N x N elements maintains square shape (h = 1/N). For the UNSTRUCTURED mesh the partition
    # is unused — `build_unstructured_model(N)` targets edge length 1/N (≈ N divisions per side).
    local_config_dict = deepcopy(config_dict)
    if !haskey(local_config_dict["numerical_method"], "mesh")
        local_config_dict["numerical_method"]["mesh"] = Dict()
    end
    local_config_dict["numerical_method"]["mesh"]["partition"] = [2*N, N]
    local_config = PorousNSSolver.load_config_from_dict(local_config_dict)

    u_in = get_inflow_profile(Re, c_in)

    # MESH SOURCE — three options, all returning a Gridap DiscreteModel with the named tags
    # "inlet" / "walls" (and "outlet" for the natural BC, no Dirichlet needed):
    #   1. mesh_file != ""  ⇒ load a FreeFem-generated .msh from disk (via load_freefem_mesh).
    #                         This is the only path that yields a literally paper-faithful mesh.
    #   2. mesh_generator == "UNSTRUCTURED" ⇒ generate via gmsh (transfinite boundary + Delaunay/BAMG).
    #   3. otherwise ⇒ structured Cartesian-simplexified (the original CocquetExperiment).
    is_unstructured = mesh_generator == "UNSTRUCTURED"
    model = if !isempty(mesh_file)
        PorousNSSolver.load_freefem_mesh(mesh_file)
    elseif is_unstructured
        PorousNSSolver.build_unstructured_model(N; domain=Tuple(local_config.domain.bounding_box),
                                                 wall_divisions=wall_divisions,
                                                 algorithm=mesh_algorithm)
    else
        PorousNSSolver._build_default_mesh(local_config.domain, local_config.numerical_method.mesh)
    end

    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, local_config.numerical_method.element_spaces.k_velocity)
    refe_p = ReferenceFE(lagrangian, Float64, local_config.numerical_method.element_spaces.k_pressure)

    # Generate Trial and Test spaces, dynamically inheriting boundary limits. The unstructured model
    # exposes the same named tags ("inlet","walls","outlet") as the structured one (see
    # build_unstructured_model), so this call is identical.
    X, Y, kv, kp = PorousNSSolver.build_fe_spaces(
        model,
        local_config.numerical_method.element_spaces,
        ["inlet", "walls"],
        [(true,true), (true,true)],
        [u_in, u_wall]
    )

    U, P = X
    V, Q = Y

    # Build a type-representative reaction law from the config so the §3.5 quadrature decision sees
    # Forchheimer's non-polynomial bump when applicable. Coefficients are irrelevant for the dispatch.
    local_rxn_law = local_config.physical_properties.reaction_model == "Constant_Sigma" ?
        PorousNSSolver.ConstantSigmaLaw(0.0) :
        PorousNSSolver.ForchheimerErgunLaw(0.0, 0.0)
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, local_config.numerical_method.element_spaces.k_velocity, local_rxn_law)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # Element characteristic length for the τ stabilization. For triangles (both simplexified-Cartesian
    # and unstructured) use √(2·area) = the structured leg / a per-cell length estimate. Matches the
    # benchmark and the CocquetFormMMS driver. The unstructured mesh is all-triangles, so route it here.
    is_tri = local_config.numerical_method.mesh.element_type == "TRI" || is_unstructured || !isempty(mesh_file)
    if is_tri
        h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
    else
        h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
    end
    h_cf = CellField(h_array, Ω)

    # Porosity interpolation order. By default it matches the velocity order `k_v`. `porosity_order`
    # lets a run override this — the Cocquet/Galerkin run uses P1 porosity regardless of velocity order.
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
    println("--- Cocquet Convergence Analysis (IRREGULAR/UNSTRUCTURED mesh, N=200 Reference) ---")

    config_file = length(ARGS) > 0 ? ARGS[1] : "paper_comparison_irregular_freefem_divs.json"
    base_config_path = joinpath(@__DIR__, "data", config_file)
    base_config_dict = JSON3.read(read(base_config_path, String), Dict{String, Any})

    Re = Float64(base_config_dict["Re"])
    c_in = Float64(base_config_dict["c_in"])
    delta = Float64(get(base_config_dict, "outlet_truncation_delta", 0.0))

    # Mesh choice: "STRUCTURED" (default) or "UNSTRUCTURED" (gmsh Delaunay). Stripped before strict parse.
    mesh_generator = String(get(base_config_dict, "mesh_generator", "STRUCTURED"))
    # Boundary-divisions policy for the gmsh path. "uniform" (default) gives edge length ≈ 1/N
    # everywhere (walls get 2N segments, inlet/outlet get N). "freefem" prescribes N segments per
    # boundary part regardless of length — the literal `buildmesh(a(N)+b(N)+…)` behavior, leaving
    # walls 2× coarser per edge than inlets. Sister-test `CocquetExperimentIrregularMeshFreefemDivs`
    # uses "freefem"; the irregular benchmark uses "uniform".
    wall_divisions = Symbol(String(get(base_config_dict, "wall_divisions", "uniform")))
    # Optional: load meshes from disk instead of generating. Expected filename convention (the user's
    # FreeFem output): `<dir>/mesh_fig2_N<N>.msh` for coarse N, `<dir>/mesh_fig2_N<N_ref>_reference.msh`
    # for the reference. Path is resolved relative to this driver's directory.
    freefem_mesh_dir = String(get(base_config_dict, "freefem_mesh_dir", ""))

    k_v = base_config_dict["numerical_method"]["element_spaces"]["k_velocity"]
    k_p = base_config_dict["numerical_method"]["element_spaces"]["k_pressure"]

    element_pairs = get(base_config_dict, "element_pairs", [[k_v, k_p]])
    comparison_runs = get(base_config_dict, "comparison_runs", nothing)

    # Remove script variables to prevent strict-schema enforcement warnings during formal validation.
    delete!(base_config_dict, "Re")
    delete!(base_config_dict, "c_in")
    if haskey(base_config_dict, "k_convergence_list")
        delete!(base_config_dict, "k_convergence_list")
    end
    delete!(base_config_dict, "outlet_truncation_delta")
    delete!(base_config_dict, "mesh_generator")
    delete!(base_config_dict, "wall_divisions")
    delete!(base_config_dict, "freefem_mesh_dir")
    delete!(base_config_dict, "element_pairs")
    delete!(base_config_dict, "comparison_runs")

    original_method = base_config_dict["numerical_method"]["stabilization"]["method"]
    base_config_dict["numerical_method"]["stabilization"]["method"] = "ASGS"
    base_config = PorousNSSolver.load_config_from_dict(base_config_dict)
    base_config_dict["numerical_method"]["stabilization"]["method"] = original_method

    L_max = base_config.domain.bounding_box[2]
    bounding_rule = x -> x[1] <= (L_max - delta)

    N_ref = base_config.numerical_method.mesh.partition[2]
    N_list = base_config.numerical_method.mesh.convergence_partitions

    results_dir = joinpath(@__DIR__, "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end

    h5_name = "convergence_$(splitext(config_file)[1]).h5"
    h5_path = joinpath(results_dir, h5_name)

    h5open(h5_path, "w") do file
        file["N_list"] = collect(N_list)
        attributes(file)["Re"] = Re
        attributes(file)["c_in"] = c_in
        attributes(file)["outlet_truncation_delta"] = delta
        attributes(file)["mesh_generator"] = mesh_generator
        attributes(file)["wall_divisions"] = String(wall_divisions)
    end

    println("    [mesh] generator = $mesh_generator  | wall_divisions = $wall_divisions",
            isempty(freefem_mesh_dir) ? "" : "  | freefem_mesh_dir = $freefem_mesh_dir")

    # Resolve a coarse-N or reference-N FreeFem mesh path (empty when generation should be used
    # instead). The user's naming convention has the reference mesh suffixed `_reference` — try that
    # first for N==N_ref, fall back to the plain `mesh_fig2_N<N>.msh` otherwise.
    function freefem_path_for(N)
        isempty(freefem_mesh_dir) && return ""
        if N == N_ref
            p = joinpath(@__DIR__, freefem_mesh_dir, "mesh_fig2_N$(N)_reference.msh")
            isfile(p) && return p
        end
        return joinpath(@__DIR__, freefem_mesh_dir, "mesh_fig2_N$(N).msh")
    end

    function do_run(k_v::Int, k_p::Int, method::String, porosity_order)
        is_galerkin = method == "Galerkin"
        config_method = is_galerkin ? "ASGS" : method
        exec_fn = is_galerkin ? execute_solver_galerkin : execute_solver

        base_config_dict["numerical_method"]["element_spaces"]["k_velocity"] = k_v
        base_config_dict["numerical_method"]["element_spaces"]["k_pressure"] = k_p
        base_config_dict["numerical_method"]["stabilization"]["method"] = config_method

        po_str = porosity_order === nothing ? "k_v" : string(porosity_order)
        println("\n##########################################################################################")
        println("[#] P$(k_v)/P$(k_p) | METHOD: $method$(is_galerkin ? " (unstabilized Galerkin / Cocquet)" : "") | porosity order: $po_str | mesh: $mesh_generator")
        println("##########################################################################################")

        println("\n   [+] Assembling High-Fidelity Reference Mesh Solution (N = $N_ref) natively...")
        base_config_dict["output"]["basename"] = "cocquet_irr_ffd_ref_$(method)_P$(k_v)P$(k_p)_N$(N_ref)"
        mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref = build_solver(N_ref, base_config_dict, Re, c_in; porosity_order=porosity_order, mesh_generator=mesh_generator, wall_divisions=wall_divisions, mesh_file=freefem_path_for(N_ref))
        xh_ref, time_ref, iters_ref = exec_fn(mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref)
        u_ref, p_ref = xh_ref

        # Tolerant cross-mesh point-location: the coarse and reference meshes are independent,
        # NON-NESTED unstructured triangulations, so near-boundary query points need a wider
        # candidate-cell set than the default (num_nearest_vertices=1) or KDTree location throws.
        ksearch = Gridap.CellData.KDTreeSearch(num_nearest_vertices=10)
        iu_ref = Gridap.FESpaces.Interpolable(u_ref; searchmethod=ksearch)
        ip_ref = Gridap.FESpaces.Interpolable(p_ref; searchmethod=ksearch)

        PorousNSSolver.export_results(cfg_ref, mod_ref, u_ref, p_ref, "alpha" => alpha_ref)

        errors_l2_u = Float64[]; errors_h1_u = Float64[]
        errors_l2_p = Float64[]; errors_h1_p = Float64[]
        errors_l2_u_trial = Float64[]; errors_h1_u_trial = Float64[]
        errors_l2_p_trial = Float64[]; errors_h1_p_trial = Float64[]
        eval_times = Float64[]; eval_iters = Int[]

        for N in N_list
            println("\n   ==============================================================")
            println("   [+] Launching Coarse Grid Algebraic Evaluation Space for N = $N")
            println("   ==============================================================")
            base_config_dict["output"]["basename"] = "cocquet_irr_ffd_$(method)_P$(k_v)P$(k_p)_N$(N)"
            mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h = build_solver(N, base_config_dict, Re, c_in; porosity_order=porosity_order, mesh_generator=mesh_generator, wall_divisions=wall_divisions, mesh_file=freefem_path_for(N))

            xh_h, time_h, iters_h = exec_fn(mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h)
            u_h, p_h = xh_h

            V_h_free = TestFESpace(mod_h, ru_h, conformity=:H1)
            Q_h_free = TestFESpace(mod_h, rp_h, conformity=:H1)

            res_u = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref; filter_func=bounding_rule, search_method=ksearch)
            res_p = PorousNSSolver.compute_reference_errors(p_h, p_ref, ip_ref, Q_h_free, dΩ_h, dΩ_ref; filter_func=bounding_rule, search_method=ksearch)

            l2_eu_nested, h1_eu_nested, l2_eu, h1_eu, eu_nested, eu_cons = res_u
            l2_ep_nested, h1_ep_nested, l2_ep, h1_ep, ep_nested, ep_cons = res_p

            # [H-C diagnostic] Project u_ref onto the Dirichlet-constrained TRIAL space U_h
            # (not the free-DOF test space V_h_free used above). Hypothesis: paper plots
            # ‖I_{U_h} u_ref − u_h‖, which differs from l2_nested only by how boundary nodes
            # are populated (V_h_free interpolates u_ref's boundary trace; U_h uses analytic
            # Dirichlet values). For P2-exact inlet `c_in y(1-y)` the algebraic difference is
            # zero at boundary nodes; any measurable gap signals cross-mesh interpolation
            # noise at coarse boundary nodes on unstructured meshes.
            U_h, P_h = X_h
            res_u_trial = PorousNSSolver.compute_trial_projection_errors(u_h, iu_ref, U_h, dΩ_h; filter_func=bounding_rule)
            res_p_trial = PorousNSSolver.compute_trial_projection_errors(p_h, ip_ref, P_h, dΩ_h; filter_func=bounding_rule)
            l2_eu_trial, h1_eu_trial = res_u_trial
            l2_ep_trial, h1_ep_trial = res_p_trial

            PorousNSSolver.export_results(cfg_h, mod_h, u_h, p_h, "alpha" => alpha_h, "e_u" => eu_nested, "e_p" => ep_nested)

            # Reported metric is the CONSISTENT one (integrated on the fine reference mesh). On
            # non-nested unstructured meshes this is the fragile direction (coarse field sampled at
            # fine boundary quadrature points). Print the NESTED metric (ref → coarse space, integrated
            # on the coarse mesh — robust direction) and the solution norm for a magnitude cross-check.
            l2_uref = sqrt(sum(∫(u_ref ⊙ u_ref) * dΩ_ref))
            l2_uh   = sqrt(sum(∫(u_h ⊙ u_h) * dΩ_h))
            println("   [+] L2(u) consistent: ", l2_eu, "  | L2(u) NESTED: ", l2_eu_nested,
                    "  | L2(u) TRIAL: ", l2_eu_trial,
                    "  | rel(consistent)=", l2_eu / l2_uref)
            println("   [+] H1(u) consistent: ", h1_eu, "  | H1(u) NESTED: ", h1_eu_nested,
                    "  | H1(u) TRIAL: ", h1_eu_trial)
            println("   [+] L2(p) consistent: ", l2_ep, "  | L2(p) NESTED: ", l2_ep_nested,
                    "  | L2(p) TRIAL: ", l2_ep_trial)
            println("   [+] ||u_ref||=", l2_uref, "  ||u_h||=", l2_uh)

            push!(errors_l2_u, l2_eu); push!(errors_h1_u, h1_eu)
            push!(errors_l2_p, l2_ep); push!(errors_h1_p, h1_ep)
            push!(errors_l2_u_trial, l2_eu_trial); push!(errors_h1_u_trial, h1_eu_trial)
            push!(errors_l2_p_trial, l2_ep_trial); push!(errors_h1_p_trial, h1_ep_trial)
            push!(eval_times, time_h); push!(eval_iters, iters_h)

            GC.gc()
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
            g["errors_l2_u_trial"] = errors_l2_u_trial
            g["errors_h1_u_trial"] = errors_h1_u_trial
            g["errors_l2_p_trial"] = errors_l2_p_trial
            g["errors_h1_p_trial"] = errors_h1_p_trial
            g["eval_times"] = eval_times
            g["eval_iters"] = eval_iters
            attributes(g)["total_time_s"] = sum(eval_times)
            attributes(g)["total_iters"] = sum(eval_iters)
            attributes(g)["outlet_truncation_delta"] = delta
            attributes(g)["porosity_order"] = porosity_order === nothing ? -1 : Int(porosity_order)
        end
    end

    if comparison_runs !== nothing
        for run in comparison_runs
            kv = Int(run["k_velocity"]); kp = Int(run["k_pressure"])
            m  = String(run["method"])
            po = haskey(run, "porosity_order") ? Int(run["porosity_order"]) : nothing
            do_run(kv, kp, m, po)
        end
    else
        for ep in element_pairs
            for method in ["ASGS"]
                do_run(Int(ep[1]), Int(ep[2]), method, nothing)
            end
        end
    end
    println("\nConvergence Data Generated and Exported to $h5_name")
end

# Only auto-run when invoked as a script; allow `include` for reuse of build_solver/execute_solver
# (e.g. cross-mesh-family reference diagnostics) without triggering the full study.
if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence()
end
