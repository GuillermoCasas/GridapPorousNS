# test/extended/CocquetExperimentLiteralPicard/run_convergence.jl
# ==============================================================================================
# [H-B test sibling] Convergence sweep where the Cocquet Galerkin solver is replaced by a custom
# LITERAL Picard driver (literal_picard_driver.jl): pure Picard iteration, no Newton at all, no
# Newton polish, capped at picard_iterations=10 — the exact protocol described in Cocquet et al.
# (2020) page 30. BOTH the N=200 reference and the N<=100 coarse runs use this same capped Picard,
# so the comparison vs the paper's Figure 2 is apples-to-apples.
#
# Adapted from CocquetExperimentIrregularMeshFreefemDivs/run_convergence.jl with two differences:
#  (i)  the "Galerkin_LiteralPicard" method label dispatches to execute_solver_galerkin_literal_picard
#  (ii) build_solver and execute_solver are reused unchanged from CocquetExperiment/run_convergence
#       via include — single source of truth.
#
# The reference solver MUST use the same it_max=10 Picard so the cross-mesh comparison sees the
# same iteration-truncation bias. This is enforced by reading picard_iterations from the same
# config for both reference and coarse runs.
#
# Run: cd test/extended/CocquetExperimentLiteralPicard && \
#      julia --project=../../.. run_convergence.jl paper_comparison_literal_picard.json
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON3
using HDF5
using LinearAlgebra

# Reuse the freefem-divs build_solver (mesh path + FE spaces) and the standard Newton-based
# execute_solver (used only as a fallback if a non-LiteralPicard method ever appears).
include(joinpath(@__DIR__, "..", "CocquetExperimentIrregularMeshFreefemDivs", "run_convergence.jl"))
# The literal-Picard execution function:
include(joinpath(@__DIR__, "literal_picard_driver.jl"))


function run_literal_picard_convergence()
    println("--- [H-B] Cocquet Literal Picard Convergence (it_max=10 cap, freefem-divs mesh) ---")

    config_file = length(ARGS) > 0 ? ARGS[1] : "paper_comparison_literal_picard.json"
    base_config_path = joinpath(@__DIR__, "data", config_file)
    base_config_dict = JSON3.read(read(base_config_path, String), Dict{String, Any})

    Re = Float64(base_config_dict["Re"])
    c_in = Float64(base_config_dict["c_in"])
    delta = Float64(get(base_config_dict, "outlet_truncation_delta", 0.0))

    mesh_generator = String(get(base_config_dict, "mesh_generator", "STRUCTURED"))
    wall_divisions = Symbol(String(get(base_config_dict, "wall_divisions", "uniform")))
    freefem_mesh_dir = String(get(base_config_dict, "freefem_mesh_dir", ""))

    k_v = base_config_dict["numerical_method"]["element_spaces"]["k_velocity"]
    k_p = base_config_dict["numerical_method"]["element_spaces"]["k_pressure"]

    element_pairs = get(base_config_dict, "element_pairs", [[k_v, k_p]])
    comparison_runs = get(base_config_dict, "comparison_runs", nothing)

    # Strip script-only keys before strict config validation.
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
    it_max = base_config.numerical_method.solver.picard_iterations

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
        attributes(file)["picard_it_max"] = it_max
    end

    println("    [mesh] generator = $mesh_generator  | wall_divisions = $wall_divisions  | picard_it_max = $it_max",
            isempty(freefem_mesh_dir) ? "" : "  | freefem_mesh_dir = $freefem_mesh_dir")

    function freefem_path_for(N)
        isempty(freefem_mesh_dir) && return ""
        if N == N_ref
            p = joinpath(@__DIR__, freefem_mesh_dir, "mesh_fig2_N$(N)_reference.msh")
            isfile(p) && return p
        end
        return joinpath(@__DIR__, freefem_mesh_dir, "mesh_fig2_N$(N).msh")
    end

    function do_run(k_v::Int, k_p::Int, method::String, porosity_order)
        # Dispatch: ONLY "Galerkin_LiteralPicard" supported here. Anything else is a config error.
        if method != "Galerkin_LiteralPicard"
            error("CocquetExperimentLiteralPicard only supports method=\"Galerkin_LiteralPicard\"; got $method")
        end
        exec_fn = execute_solver_galerkin_literal_picard
        config_method = "ASGS"  # stab method irrelevant — Galerkin path with mult_*=0 zeroes stabilization

        base_config_dict["numerical_method"]["element_spaces"]["k_velocity"] = k_v
        base_config_dict["numerical_method"]["element_spaces"]["k_pressure"] = k_p
        base_config_dict["numerical_method"]["stabilization"]["method"] = config_method

        po_str = porosity_order === nothing ? "k_v" : string(porosity_order)
        println("\n##########################################################################################")
        println("[#] P$(k_v)/P$(k_p) | METHOD: $method (LITERAL Picard, it_max=$it_max) | porosity order: $po_str | mesh: $mesh_generator/$wall_divisions")
        println("##########################################################################################")

        println("\n   [+] Assembling High-Fidelity Reference Mesh Solution (N = $N_ref) with LiteralPicard...")
        base_config_dict["output"]["basename"] = "cocquet_lp_ref_$(method)_P$(k_v)P$(k_p)_N$(N_ref)"
        mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref = build_solver(N_ref, base_config_dict, Re, c_in; porosity_order=porosity_order, mesh_generator=mesh_generator, wall_divisions=wall_divisions, mesh_file=freefem_path_for(N_ref))
        xh_ref, time_ref, iters_ref = exec_fn(mod_ref, X_ref, Y_ref, dΩ_ref, h_ref, alpha_ref, ru_ref, rp_ref, cfg_ref)
        u_ref, p_ref = xh_ref

        ksearch = Gridap.CellData.KDTreeSearch(num_nearest_vertices=10)
        iu_ref = Gridap.FESpaces.Interpolable(u_ref; searchmethod=ksearch)
        ip_ref = Gridap.FESpaces.Interpolable(p_ref; searchmethod=ksearch)

        PorousNSSolver.export_results(cfg_ref, mod_ref, u_ref, p_ref, "alpha" => alpha_ref)

        errors_l2_u = Float64[]; errors_h1_u = Float64[]
        errors_l2_p = Float64[]; errors_h1_p = Float64[]
        eval_times = Float64[]; eval_iters = Int[]

        for N in N_list
            println("\n   ==============================================================")
            println("   [+] Launching Coarse Grid Algebraic Evaluation Space for N = $N")
            println("   ==============================================================")
            base_config_dict["output"]["basename"] = "cocquet_lp_$(method)_P$(k_v)P$(k_p)_N$(N)"
            mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h = build_solver(N, base_config_dict, Re, c_in; porosity_order=porosity_order, mesh_generator=mesh_generator, wall_divisions=wall_divisions, mesh_file=freefem_path_for(N))

            xh_h, time_h, iters_h = exec_fn(mod_h, X_h, Y_h, dΩ_h, h_h, alpha_h, ru_h, rp_h, cfg_h)
            u_h, p_h = xh_h

            V_h_free = TestFESpace(mod_h, ru_h, conformity=:H1)
            Q_h_free = TestFESpace(mod_h, rp_h, conformity=:H1)

            res_u = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref; filter_func=bounding_rule, search_method=ksearch)
            res_p = PorousNSSolver.compute_reference_errors(p_h, p_ref, ip_ref, Q_h_free, dΩ_h, dΩ_ref; filter_func=bounding_rule, search_method=ksearch)

            l2_eu_nested, h1_eu_nested, l2_eu, h1_eu, eu_nested, eu_cons = res_u
            l2_ep_nested, h1_ep_nested, l2_ep, h1_ep, ep_nested, ep_cons = res_p

            PorousNSSolver.export_results(cfg_h, mod_h, u_h, p_h, "alpha" => alpha_h, "e_u" => eu_nested, "e_p" => ep_nested)

            l2_uref = sqrt(sum(∫(u_ref ⊙ u_ref) * dΩ_ref))
            l2_uh   = sqrt(sum(∫(u_h ⊙ u_h) * dΩ_h))
            println("   [+] L2(u) consistent: ", l2_eu, "  | L2(u) NESTED: ", l2_eu_nested,
                    "  | rel(consistent)=", l2_eu / l2_uref, "  | picard_iters=$iters_h")
            println("   [+] H1(u) consistent: ", h1_eu, "  | H1(u) NESTED: ", h1_eu_nested)
            println("   [+] L2(p) consistent: ", l2_ep, "  | L2(p) NESTED: ", l2_ep_nested)
            println("   [+] ||u_ref||=", l2_uref, "  ||u_h||=", l2_uh)

            push!(errors_l2_u, l2_eu); push!(errors_h1_u, h1_eu)
            push!(errors_l2_p, l2_ep); push!(errors_h1_p, h1_ep)
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
            g["eval_times"] = eval_times
            g["eval_iters"] = eval_iters
            attributes(g)["total_time_s"] = sum(eval_times)
            attributes(g)["total_iters"] = sum(eval_iters)
            attributes(g)["outlet_truncation_delta"] = delta
            attributes(g)["porosity_order"] = porosity_order === nothing ? -1 : Int(porosity_order)
            attributes(g)["picard_it_max"] = it_max
            attributes(g)["ref_iters"] = iters_ref
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
        error("CocquetExperimentLiteralPicard requires 'comparison_runs' in the JSON config.")
    end
    println("\nLiteral-Picard convergence data exported to $h5_name")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_literal_picard_convergence()
end
