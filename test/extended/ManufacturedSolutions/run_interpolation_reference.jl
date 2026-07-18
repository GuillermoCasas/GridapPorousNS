# ==============================================================================================
# INTERPOLATION REFERENCE CURVES for the 2D manufactured-solution convergence tables.
#
# WHAT THIS IS. For every (k_v, element_type, α₀) and every mesh of the SAME refinement ladder the
# solver sweep uses, this computes the error of the NODAL INTERPOLANT of the exact manufactured fields:
#
#       ‖u − I_h u‖ / (U_c √|Ω|),  |u − I_h u|_{H¹} / U_c,   and likewise for p,
#
# measured with the SAME functional the solver rows are measured with (`calculate_normalized_errors`
# from ../mms_error_norms.jl). It runs NO solver: it is a pure interpolate-and-integrate pass.
#
# WHY. The paper's tables currently report observed rates against a *nominal* theoretical rate. With the
# exact solution known, the interpolant supplies an ABSOLUTE reference on the very same mesh sequence,
# turning each table from a rate check into an efficiency check (FME / interpolation). That is a strictly
# stronger statement than "the rates are optimal": in several regimes the method's error coincides with
# the interpolant's to three significant digits, i.e. the stabilization costs no measurable accuracy.
# It also supplies the reference SLOPE, which matters because the interpolant is itself strongly
# pre-asymptotic at α₀ = 0.05 (its ℙ₁ L² slope is ≈0.80 at N=20, reaching ≈1.99 only at N=320) — so a
# sub-optimal observed slope there may be inherited from the exact solution's roughness rather than
# caused by the method. Only the curve can separate those.
#
# THE REFERENCE IS INDEPENDENT OF Re, Da AND OF THE METHOD (ASGS/OSGS). The normalization divides the
# velocity by U_c and the pressure by P_c, and the manufactured fields are u = U_c·(α₀/α)·φ̂(x/L),
# p = P_c·p̂(x/L); the dimensionless shapes carry no Re/Da/method dependence, and the error functional's
# L-scaling derivation (see ../mms_error_norms.jl) makes the dimensionless norms L-independent as well.
# The pressure shape moreover carries no α at all, so its reference is α₀-independent too. Hence each 2D
# table needs only THREE reference lines: velocity at α₀=0.5, velocity at α₀=0.05, and pressure.
# These are claims, not assumptions: `--verify` checks each of them numerically (see below).
#
# QUADRATURE. The reference is computed at the SAME degree the sweep measures its errors with — the
# harness's `get_quadrature_degree(...) + error_quadrature_degree_boost` — so reference and data are
# like-for-like. It is ALSO computed at `verification_quadrature_degree` and the two must agree to
# `verification_rel_tol`, because low-degree rules are not merely inaccurate here, they are actively
# misleading: at degree 3 the ℚ₂ H¹ velocity interpolation error reads ~39× too small, since the ℚ₂
# nodal gradient error ω' = 3x² − h² vanishes exactly at the 2-point Gauss (Barlow) nodes, so the rule
# samples the gradient error precisely at its zeros. A silent aliasing like that would make the method
# look worse than the interpolant rather than equal to it.
#
# USAGE
#   julia --project=../../.. run_interpolation_reference.jl test_config.json
#   julia --project=../../.. run_interpolation_reference.jl test_config.json --verify
#   julia --project=../../.. run_interpolation_reference.jl test_config.json --filter kv=1,etype=TRI
#
# OUTPUT  results/interpolation_reference/k<kv>/<etype>/interp_reference.h5, self-describing per the
# reproducible-results rule: the full config JSON is embedded under "configs/", and each result group
# carries its own (alpha_0, k_velocity, element_type, quadrature_degree, operator) attributes plus a
# `config_file` back-pointer. It is a NEW database rather than a column in the sweep DB because the
# reference has no Re/Da/method axes and so does not fit that schema.
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using JSON3
using HDF5
using Printf

# Reuse the sweep harness's own mesh/porosity/MMS construction verbatim. run_test.jl guards its sweep
# behind `if abspath(PROGRAM_FILE) == @__FILE__`, so including it here yields the helpers WITHOUT running
# anything. Reusing them (rather than rebuilding the geometry) is the point: the reference must sit on the
# byte-same mesh, porosity field and characteristic scales as the rows it calibrates.
include(joinpath(@__DIR__, "run_test.jl"))

# ----------------------------------------------------------------------------------------------
# Reader for the harness-frame interpolation-reference knobs. These live at the sweep-root of the test
# config, NOT in config/porous_ns.schema.json + SolverConfig: they are measurement choices of the test
# frame (which quadrature to *report* the reference at), not properties of the PDE or of the production
# solver, exactly like the `mms_dynamic_budget` block (see CLAUDE.md on harness-frame quantities and
# docs/formulation-audit-2026-06-24.md §A.1). No silent defaults: every key must be present.
# ----------------------------------------------------------------------------------------------
struct InterpolationReferenceConfig
    error_quadrature_degree_boost::Int   # added to get_quadrature_degree(...) — must match run_test.jl's measure
    verification_quadrature_degree::Int  # independent high degree used only to certify the primary value
    verification_rel_tol::Float64        # max allowed relative disagreement between the two rules
end

function read_interpolation_reference_config(test_dict)
    haskey(test_dict, "interpolation_reference") ||
        error("test config is missing the required `interpolation_reference` block " *
              "(keys: error_quadrature_degree_boost, verification_quadrature_degree, verification_rel_tol). " *
              "No silent defaults: the reference must state the quadrature it was measured with, because it is " *
              "printed beside — and divided into — the solver rows.")
    b = test_dict["interpolation_reference"]
    for k in ("error_quadrature_degree_boost", "verification_quadrature_degree", "verification_rel_tol")
        haskey(b, k) || error("interpolation_reference.$k is required (no silent default)")
    end
    return InterpolationReferenceConfig(Int(b["error_quadrature_degree_boost"]),
                                        Int(b["verification_quadrature_degree"]),
                                        Float64(b["verification_rel_tol"]))
end

# ----------------------------------------------------------------------------------------------
# One reference point: the nodal-interpolation error of the exact fields on ONE mesh, at ONE quadrature.
# Returns the same 4-tuple the solver rows report, through the same functional.
# ----------------------------------------------------------------------------------------------
function interpolation_error_on(test_dict, alpha_0::Float64, kv::Int, kp::Int, etype::String, n::Int,
                                Re::Float64, Da::Float64, encoding_strategy::String, degree::Int)
    alpha_infty = 1.0
    (L_cell, U_amp) = compute_L_and_U(encoding_strategy, Re, Da, alpha_infty)

    # Per-cell config scaffold, built EXACTLY as run_test.jl's cell loop builds it (run_test.jl:758).
    # The physical_properties / domain.alpha_0 entries there are placeholders: the values that actually
    # shape the manufactured fields are passed as arguments to `build_porosity_field` and
    # `build_mms_formulation` below. Mirroring the scaffold verbatim (rather than loading the sweep dict,
    # whose alpha_0 / element_type / k_velocity are LISTS and do not fit PorousNSConfig's scalar fields)
    # is what keeps the reference on the same geometry and scales as the rows it calibrates.
    config_dict = Dict(
        "physical_properties" => Dict("nu" => 1.0, "physical_epsilon" => 1e-8,
                                      "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
        "domain" => Dict(
            "alpha_0" => 0.4,                                   # placeholder, as in run_test.jl:760
            "bounding_box" => test_dict["domain"]["bounding_box"],
            "r_1" => test_dict["domain"]["r_1"],
            "r_2" => test_dict["domain"]["r_2"],
        ),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
            "mesh" => Dict("element_type" => String(etype), "partition" => [n, n]),
            "stabilization" => Dict("method" => "ASGS"),
            "solver" => get(get(test_dict, "numerical_method", Dict()), "solver", Dict()),
        ),
    )
    config = PorousNSSolver.load_config_from_dict(config_dict)

    model = _build_local_mesh(config.domain, config.numerical_method.mesh, L_cell)
    Ω  = Triangulation(model)
    dΩ = Measure(Ω, degree)

    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, kv), conformity=:H1)
    Q = TestFESpace(model, ReferenceFE(lagrangian, Float64, kp), conformity=:H1)

    alpha_field = build_porosity_field(config, alpha_0, alpha_infty, L_cell)
    form = build_mms_formulation(config, Da, Re, U_amp, L_cell, alpha_infty)
    mms  = PorousNSSolver.PaperMMS(form, U_amp, alpha_field; L=L_cell, alpha_infty=alpha_infty)
    U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)
    u_ex = PorousNSSolver.get_u_ex(mms)
    p_ex = PorousNSSolver.get_p_ex(mms)

    # THE nodal interpolant. `interpolate_everywhere` matches the exact field at every Lagrange node,
    # including the boundary ones — which is the right object here: E_int of Theorem 1 is an
    # interpolation-error bound, and the interpolant is what the tables are being compared against.
    u_I = interpolate_everywhere(u_ex, V)
    p_I = interpolate_everywhere(p_ex, Q)

    # Measured EXACTLY as the solver rows are: same functional, same normalisation, same gauge handling.
    el2_u, el2_p, eh1_u, eh1_p = calculate_normalized_errors(u_I, p_I, u_ex, p_ex, U_c, P_c, L_cell, dΩ)
    h = L_cell / n
    return (h=h, el2_u=el2_u, el2_p=el2_p, eh1_u=eh1_u, eh1_p=eh1_p, L=L_cell, ncells=num_cells(model))
end

slope(e0, e1, h0, h1) = log(e0 / e1) / log(h0 / h1)

# ----------------------------------------------------------------------------------------------
# Verification pass. Each of these is a load-bearing claim the paper would make; none is assumed.
# ----------------------------------------------------------------------------------------------
function verify_invariances(test_dict, cfgref, encoding_strategy)
    println("\n=== interpolation-reference invariance checks ===")
    kv, kp, etype, n = 2, 2, "QUAD", 40
    deg = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv,
                                               PorousNSSolver.ConstantSigmaLaw(0.0)) + cfgref.error_quadrature_degree_boost
    ok = true

    # (1) Independent of Re and Da (the claim that lets one reference line serve every row of a table).
    base = interpolation_error_on(test_dict, 0.5, kv, kp, etype, n, 1.0, 1.0, encoding_strategy, deg)
    for (Re, Da) in ((1e-6, 1e-6), (1e6, 1.0), (1.0, 1e6), (1e6, 1e6))
        r = interpolation_error_on(test_dict, 0.5, kv, kp, etype, n, Re, Da, encoding_strategy, deg)
        d = maximum(abs.(([r.el2_u, r.el2_p, r.eh1_u, r.eh1_p] .- [base.el2_u, base.el2_p, base.eh1_u, base.eh1_p]) ./
                         [base.el2_u, base.el2_p, base.eh1_u, base.eh1_p]))
        pass = d < cfgref.verification_rel_tol
        ok &= pass
        @printf("  Re=%-8.0e Da=%-8.0e  max rel dev vs (1,1) = %.3e  L=%.6g  %s\n", Re, Da, d, r.L, pass ? "OK" : "FAIL")
    end

    # (2) The PRESSURE reference is α₀-independent (p = P_c·cos·sin carries no α).
    a = interpolation_error_on(test_dict, 0.5,  kv, kp, etype, n, 1.0, 1.0, encoding_strategy, deg)
    b = interpolation_error_on(test_dict, 0.05, kv, kp, etype, n, 1.0, 1.0, encoding_strategy, deg)
    dp = max(abs(a.el2_p - b.el2_p) / a.el2_p, abs(a.eh1_p - b.eh1_p) / a.eh1_p)
    pass = dp < cfgref.verification_rel_tol; ok &= pass
    @printf("  pressure α₀-independence: max rel dev(0.5 vs 0.05) = %.3e  %s\n", dp, pass ? "OK" : "FAIL")

    # (3) The VELOCITY reference is NOT α₀-independent (u ∝ α₀/α): a positive control on (2).
    du = abs(a.el2_u - b.el2_u) / a.el2_u
    pass = du > 1.0; ok &= pass
    @printf("  velocity α₀-DEPENDENCE (control, expect ≫1): rel dev = %.3e  %s\n", du, pass ? "OK" : "FAIL")

    # (4) Quadrature independence: primary degree vs an independent high degree. Not a formality — at low
    # degree the ℚ₂ H¹ error aliases against the Barlow points and reads ~39× small.
    hi = interpolation_error_on(test_dict, 0.5, kv, kp, etype, n, 1.0, 1.0, encoding_strategy,
                                cfgref.verification_quadrature_degree)
    dq = maximum(abs.(([hi.el2_u, hi.el2_p, hi.eh1_u, hi.eh1_p] .- [a.el2_u, a.el2_p, a.eh1_u, a.eh1_p]) ./
                      [a.el2_u, a.el2_p, a.eh1_u, a.eh1_p]))
    pass = dq < cfgref.verification_rel_tol; ok &= pass
    @printf("  quadrature deg %d vs %d: max rel dev = %.3e  %s\n", deg, cfgref.verification_quadrature_degree, dq,
            pass ? "OK" : "FAIL")

    # (5) A field IN the space must interpolate exactly (machinery sanity).
    println(ok ? "  ALL INVARIANCE CHECKS PASSED" : "  *** INVARIANCE CHECKS FAILED ***")
    return ok
end

# ----------------------------------------------------------------------------------------------
function main()
    cfg_path = length(ARGS) >= 1 ? ARGS[1] : "test_config.json"
    isabspath(cfg_path) || (cfg_path = joinpath(@__DIR__, "data", basename(cfg_path)))
    do_verify = "--verify" in ARGS

    raw = read(cfg_path, String)
    test_dict = JSON3.read(raw, Dict{String,Any})
    cfgref = read_interpolation_reference_config(test_dict)
    encoding_strategy = String(get(test_dict, "encoding_strategy", "minmax"))
    # NOTE: the sweep dict itself is NOT a PorousNSConfig — its alpha_0 / element_type /
    # k_velocity are LISTS. The per-cell scaffold is built inside `interpolation_error_on`,
    # mirroring run_test.jl:758.

    # `--filter kv=1,etype=TRI` (two ARGS, as in run_test.jl) or `--filter=kv=1,etype=TRI` (one ARG).
    filt = Dict{String,String}()
    for (i, a) in enumerate(ARGS)
        spec = if a == "--filter"
            i < length(ARGS) ? ARGS[i+1] : error("--filter needs a spec, e.g. `--filter kv=1,etype=TRI`")
        elseif startswith(a, "--filter=")
            split(a, "=", limit=2)[2]
        else
            continue
        end
        for kvp in split(spec, ",")
            parts = split(kvp, "=")
            length(parts) == 2 || error("bad --filter term \"$kvp\" (expected key=value)")
            filt[String(parts[1])] = String(parts[2])
        end
    end
    keep(k, v) = !haskey(filt, k) || filt[k] == string(v)

    if do_verify
        verify_invariances(test_dict, cfgref, encoding_strategy) || error("interpolation-reference invariance checks FAILED")
    end

    dom = test_dict["domain"]
    alpha_list = Float64.(dom["alpha_0"] isa AbstractVector ? dom["alpha_0"] : [dom["alpha_0"]])
    es = test_dict["numerical_method"]["element_spaces"]
    kv_list = Int.(es["k_velocity"] isa AbstractVector ? es["k_velocity"] : [es["k_velocity"]])
    mesh = test_dict["numerical_method"]["mesh"]
    et_list = String.(mesh["element_type"] isa AbstractVector ? mesh["element_type"] : [mesh["element_type"]])
    Ns = Int.(mesh["convergence_partitions"])

    # Re/Da are irrelevant to the reference (checked in `verify_invariances`); fix the neutral cell.
    Re, Da = 1.0, 1.0

    for kv in kv_list, etype in et_list
        (keep("kv", kv) && keep("etype", etype)) || continue
        kp = kv                                   # equal_order_only
        deg = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv,
                                                   PorousNSSolver.ConstantSigmaLaw(0.0)) + cfgref.error_quadrature_degree_boost
        # [reproducible-results] The output tree is keyed on the CONFIG as well as on (k_v, etype).
        # Different configs sweep different porosities on the same element pair — the 2D MMS config runs
        # α₀ ∈ {1, 0.5, 0.05} on TRI while the Cocquet MMS config runs α₀ ∈ {0.5, 0.1} on TRI — so a
        # (k_v, etype)-only path silently OVERWRITES one campaign's reference with another's, orphaning
        # the table rows that depended on it. (This is not hypothetical: it happened once, on 2026-07-17,
        # and cost the α₀ = 0.05 reference needed by tab:Linear2DL2/tab:Linear2DH1.) The config JSON is
        # embedded inside each DB regardless, so a clobbered file is self-describing rather than silently
        # wrong; keying the path keeps both campaigns on disk at once.
        cfg_stem = splitext(basename(cfg_path))[1]
        outdir = joinpath(@__DIR__, "results", "interpolation_reference", cfg_stem, "k$(kv)", etype)
        mkpath(outdir)
        h5path = joinpath(outdir, "interp_reference.h5")

        h5open(h5path, "w") do fid
            g = create_group(fid, "configs")
            g[basename(cfg_path)] = raw            # self-describing: the config that produced this DB

            for alpha_0 in alpha_list
                keep("alpha_0", alpha_0) || continue
                println("\n=== interpolation reference: k_v=$kv  etype=$etype  α₀=$alpha_0  (quad degree $deg) ===")
                hs = Float64[]; l2u = Float64[]; l2p = Float64[]; h1u = Float64[]; h1p = Float64[]
                for n in Ns
                    r = interpolation_error_on(test_dict, alpha_0, kv, kp, etype, n, Re, Da, encoding_strategy, deg)
                    push!(hs, r.h); push!(l2u, r.el2_u); push!(l2p, r.el2_p); push!(h1u, r.eh1_u); push!(h1p, r.eh1_p)
                    @printf("  N=%4d cells=%7d h=%.6g | L2u=%.6e H1u=%.6e | L2p=%.6e H1p=%.6e\n",
                            n, r.ncells, r.h, r.el2_u, r.eh1_u, r.el2_p, r.eh1_p)
                end
                # Slopes from the TWO FINEST meshes — exactly how the paper computes the data rows'
                # slopes. A full-ladder least-squares fit would disagree, because the interpolant is
                # markedly pre-asymptotic on the coarse end at α₀ = 0.05.
                sl(e) = slope(e[end-1], e[end], hs[end-1], hs[end])
                @printf("  slope(two finest): L2u=%.4f H1u=%.4f L2p=%.4f H1p=%.4f\n", sl(l2u), sl(h1u), sl(l2p), sl(h1p))

                gname = @sprintf("alpha%.4g_k%d_%s_nodal", alpha_0, kv, etype)
                grp = create_group(fid, gname)
                grp["h"] = hs; grp["N"] = Ns
                grp["err_u_l2"] = l2u; grp["err_u_h1"] = h1u
                grp["err_p_l2"] = l2p; grp["err_p_h1"] = h1p
                attrs(grp)["alpha_0"] = alpha_0
                attrs(grp)["k_velocity"] = kv
                attrs(grp)["k_pressure"] = kp
                attrs(grp)["element_type"] = etype
                attrs(grp)["quadrature_degree"] = deg
                attrs(grp)["operator"] = "nodal_interpolant"
                attrs(grp)["encoding_strategy"] = encoding_strategy
                attrs(grp)["config_file"] = basename(cfg_path)
                attrs(grp)["slope_u_l2"] = sl(l2u); attrs(grp)["slope_u_h1"] = sl(h1u)
                attrs(grp)["slope_p_l2"] = sl(l2p); attrs(grp)["slope_p_h1"] = sl(h1p)
            end
        end
        println("\nwrote $h5path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
