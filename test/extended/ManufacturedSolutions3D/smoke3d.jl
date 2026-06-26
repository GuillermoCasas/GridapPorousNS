# test/extended/ManufacturedSolutions3D/smoke3d.jl
# Smoke / convergence check for the 3D MMS solve path (paper §5.2). Runs the full pipeline
# (gmsh tet mesh -> porosity -> ConstantSigma DeviatoricSymmetric formulation -> 3D FE spaces ->
# z-extruded MMS forcing oracle -> solve_system -> normalized errors) over a short mesh sequence
# and prints the convergence rate. Optimal velocity-L2 rate O(h^{kv+1}) validates the oracle. [must-test]
using Gridap
using Gridap.Algebra
using GridapGmsh
using JSON3
using Printf
using Dates
using PorousNSSolver
const PNS = PorousNSSolver
include("mesh3d.jl")
include("mms3d.jl")
# [harness-frame] Re/Da iteration-budget knobs (relocated out of production SolverConfig — audit §A.1/F1).
@isdefined(read_mms_dynamic_budget) || include(joinpath(@__DIR__, "..", "harness_dynamic_budget.jl"))

# ---- fixed paper §5.2 parameters ----
const DOMAIN = (0.0,1.0, 0.0,1.0, 0.0,0.3)
const ALPHA0 = 0.5
const ALPHAINF = 1.0
const RE = 1.0
const DA = 1.0
const R1 = 0.2
const R2 = 0.4
const L = 1.0
const U_AMP = 1.0

# Above this many monolithic (u,p) free DOFs, a direct sparse LU factorization risks exhausting a ~32GB
# machine on 3D meshes (empirical: LU is comfortable at ~2e4 DOF / ~8.5GB RSS, but the ~1.6e5-DOF fine
# OSGS systems OOM). solve_one's "auto" linsolver switches to the low-memory ILU-GMRES backend past this
# threshold — "choose whatever method fits". [code-actual]
const LU_DOF_LIMIT = 80_000

function build_config(kv::Int, method::String; eps_tol_m_over=nothing, ftol_over=nothing, eps_tol_mass_over=nothing,
                     numerical_epsilon::Float64=0.0)
    eps_tol_m = something(eps_tol_m_over, kv == 2 ? 1e-9 : 1e-6)   # k=2 tightened gate (MEMORY lesson)
    ftol_v    = something(ftol_over, kv == 2 ? 1e-12 : 1e-10)
    solver_dict = Dict{String,Any}("eps_tol_momentum"=>eps_tol_m, "ftol"=>ftol_v)
    eps_tol_mass_over === nothing || (solver_dict["eps_tol_mass"] = eps_tol_mass_over)
    cfg = Dict(
        "physical_properties" => Dict("nu"=>1.0, "eps_val"=>1e-8, "numerical_epsilon"=>numerical_epsilon,
                                      "reaction_model"=>"Constant_Sigma", "sigma_constant"=>1.0),
        "domain" => Dict("alpha_0"=>ALPHA0, "bounding_box"=>[0.0,1.0,0.0,1.0], "r_1"=>R1, "r_2"=>R2),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity"=>kv, "k_pressure"=>kv),
            "mesh" => Dict("element_type"=>"TRI", "partition"=>[1,1]),  # placeholders; gmsh model built directly
            "stabilization" => Dict("method"=>method),
            "solver" => solver_dict,
        ),
    )
    return PNS.load_config_from_dict(cfg)
end

# normalized errors (mirrors 2D calculate_normalized_errors; exact at L=1)
function calc_errors3d(u_h, p_h, u_ex, p_ex, U_c, P_c, dΩ)
    e_u = u_ex - u_h
    e_p = p_ex - p_h
    area = sum(∫(1.0)dΩ); sqrt_area = sqrt(abs(area))
    el2_u = sqrt(abs(sum(∫(e_u ⋅ e_u)dΩ))) / (U_c * sqrt_area)
    mean_e_p = sum(∫(e_p)dΩ) / area
    e_p_c = e_p - mean_e_p
    el2_p = sqrt(abs(sum(∫(e_p_c * e_p_c)dΩ))) / (P_c * sqrt_area)
    eh1_u = sqrt(abs(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ))) / U_c
    eh1_p = sqrt(abs(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ))) / P_c
    return el2_u, el2_p, eh1_u, eh1_p
end

# [trajectory] Persist one nonlinear-solver trajectory sidecar per (cell, method, mesh), in the SAME
# schema as the 2D harness (run_test.jl): run/cell/N/.../attempts[].stages, read by plot_trajectory.py
# via the shared tools/trajectory_viz renderer. The 3D solve is a single direct exact-guess solve, so
# there is ONE attempt with eps_pert = 0 whose stages are the orchestrator's diag["trajectory"]. N is
# the effective resolution round(1/h) (analogous to the 2D partition N=1/h). Best-effort: a write
# failure must never abort a sweep. Traces land under <outroot>/k<kv>/TET/<mesh_sequence>/traces/ — the
# mesh_sequence (e.g. "structured", "nested_red") is the LEAF folder under the element type, so each element
# type keeps a separate folder per mesh sequence and results from different sequences are conserved side-by-side.
# mesh_sequence="" falls back to the legacy <outroot>/k<kv>/TET/traces/ layout.
function _write_trajectory_sidecar(; outroot, run_name, kv, method, h_mean, ncells, success,
                                   trajectory, res_init, res_final, tol_M, tol_C, mesh_sequence::String="")
    try
        cell_rel = isempty(mesh_sequence) ? joinpath("k$(Int(kv))", "TET") :
                                            joinpath("k$(Int(kv))", "TET", mesh_sequence)
        traces_dir = joinpath(outroot, cell_rel, "traces")
        isdir(traces_dir) || mkpath(traces_dir)
        nlabel = max(1, round(Int, 1.0 / h_mean))
        trace_name = @sprintf("traj_Re%.0e_Da%.0e_a%.2f_kv%d_kp%d_TET_%s_N%d.json",
                              RE, DA, ALPHA0, Int(kv), Int(kv), String(method), nlabel)
        trace_obj = (
            run = run_name,
            cell = (Re=RE, Da=DA, alpha_0=ALPHA0, kv=Int(kv), kp=Int(kv), etype="TET",
                    method=String(method), encoding="exact_guess", ncells=Int(ncells), h=h_mean),
            N = nlabel, success = success, successful_eps = 0.0,
            final_residual = res_final, initial_residual = res_init,
            tol_M = tol_M, tol_C = tol_C,
            attempts = [(eps_pert = 0.0, success = success, stages = trajectory)],
        )
        open(joinpath(traces_dir, trace_name), "w") do io
            JSON3.write(io, trace_obj; allow_inf=true)   # allow_inf: stage records may default residuals to NaN
        end
        @printf("    [trace] %s\n", joinpath(cell_rel, "traces", trace_name)); flush(stdout)
    catch e
        @warn "Trajectory trace write failed (non-fatal)" exception=e
    end
end

function solve_one(kv::Int, method::String, model; visc::String="Deviatoric", eps_mult::Float64=1.0,
                   linsolver::String="auto", trace_dir::Union{Nothing,String}=nothing, run_name::String="",
                   c1_mult::Float64=1.0, eps_tol_m_over=nothing, ftol_over=nothing, eps_tol_mass_over=nothing,
                   eps_phys::Float64=0.0, mesh_sequence::String="")
    nu = U_AMP * L / RE
    # ε_num = the NUMERICAL penalty (Codina ITERATIVE penalty, paper ε = 1e-4·ε_ref). The equation is
    # INCOMPRESSIBLE: there is NO physical compressibility, so eps_phys MUST default to 0. ε_num lives ONLY
    # in the Jacobian pressure block (lagged εpⁿ⁻¹ on the RHS), so it CANCELS at convergence and does not
    # alter the manufactured (incompressible) solution — article.tex §2 (eq:StrongMassEquation, "mainly
    # include for numerical reasons … iterative penalty") and §5.2 (eq:EpsilonRef + the iterative-penalty
    # remark). A non-zero eps_phys would instead solve a genuinely COMPRESSIBLE problem (εp in the residual
    # AND εp_ex in the oracle g). Only override eps_phys for a deliberate compressible experiment.
    eps_num = eps_mult * 1e-4 * ALPHA0 / (nu * (1.0 + RE + DA))
    config = build_config(kv, method; numerical_epsilon=eps_num,
                          eps_tol_m_over=eps_tol_m_over, ftol_over=ftol_over, eps_tol_mass_over=eps_tol_mass_over)
    sol = config.numerical_method.solver

    sigma_c = DA * ALPHAINF * nu / L^2
    alpha_field = PNS.SmoothRadialPorosity(ALPHA0, ALPHAINF, R1, R2)

    proj = sol.experimental_reaction_mode == "standard" ?
           PNS.ProjectResidualWithoutReactionWhenConstantSigma() : PNS.ProjectFullResidual()
    reg  = PNS.SmoothVelocityFloor(config.physical_properties.u_base_floor_ref, 0.0,
                                   config.physical_properties.epsilon_floor,
                                   config.physical_properties.velocity_magnitude_derivative_floor)
    visc_op = visc == "SymmetricGradient" ? PNS.SymmetricGradientViscosity() :
              visc == "Laplacian" ? PNS.LaplacianPseudoTractionViscosity() :
              PNS.DeviatoricSymmetricViscosity()
    # eps_val = ε_phys (residual + Jacobian); ε_num is the Codina iterative penalty — it lives ONLY in the
    # Jacobian's pressure block (lagged to the iterate, it cancels in the residual) and vanishes at convergence.
    form = PNS.PaperGeneralFormulation(visc_op,
                                       PNS.ConstantSigmaLaw(sigma_c), proj, reg, nu, eps_phys;
                                       numerical_epsilon=eps_num)

    mms = Paper3DMMS(form, U_AMP, alpha_field, L, ALPHAINF, eps_phys)   # oracle g uses ε_phys (0 ⇒ incompressible source)
    U_c, P_c = characteristic_scales3d(mms)
    u_ex = get_u_ex3d(mms); p_ex = get_p_ex3d(mms)

    labels = get_face_labeling(model)
    refe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kv)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["boundary"])
    Q = TestFESpace(model, refe_p, conformity=:H1)
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)
    Ut = TrialFESpace(V, u_ex)
    Pt = TrialFESpace(Q, p_ex)
    X = MultiFieldFESpace([Ut, Pt]); Y = MultiFieldFESpace([V, Q])

    # Auto-pick the inner linear solver that FITS in memory for THIS mesh ("whatever method fits"): exact
    # sparse LU below LU_DOF_LIMIT (fast, the small meshes), ILU-GMRES above it (the fine 3D OSGS systems
    # whose LU fill-in would OOM a 32GB machine). The backend choice does not change the converged
    # solution. ILU parameters come from the config's linear_solver block (base_config defaults).
    # linsolver="auto" (default) selects by size; "LU"/"ILU_GMRES" force a backend (used for validation).
    ndof = num_free_dofs(X)
    lsc  = sol.linear_solver
    use_ilu = linsolver == "ILU_GMRES" || (linsolver == "auto" && ndof > LU_DOF_LIMIT)
    linear_solver = use_ilu ?
        PNS.ILUGMRESSolver(m=lsc.gmres_restart, drop_tolerance=lsc.ilu_drop_tolerance,
                           rel_tol=lsc.gmres_rel_tol, maxiter=lsc.gmres_maxiter,
                           allow_unpreconditioned_fallback=lsc.allow_unpreconditioned_fallback) :
        LUSolver()
    @printf("    [linsolver] ndof=%d -> %s\n", ndof, use_ilu ? "ILU_GMRES" : "LU"); flush(stdout)

    degree = PNS.get_quadrature_degree(PNS.PaperGeneralFormulation, kv, PNS.ConstantSigmaLaw(0.0))
    Ω = Triangulation(model); dΩ = Measure(Ω, degree + 4)
    c_1, c_2 = PNS.get_c1_c2(PNS.PaperGeneralFormulation, kv)
    c_1 *= c1_mult; c_2 *= c1_mult   # [diagnostic] scale stabilization constants — paper Remark (eq:conditions_on_num_param):
                                     # the coercivity bound needs c1 > 2ξ·C_inv², and the OPTIMAL c1 depends on element type.

    # tet element size from cell VOLUME: regular-tet edge = (6√2·V)^{1/3}
    h_array = collect(lazy_map(v -> (6.0*sqrt(2.0)*abs(v))^(1.0/3.0), get_cell_measure(Ω)))
    h_cf = CellField(h_array, Ω)
    h_mean = sum(h_array) / length(h_array)   # ACHIEVED mesh size — the correct convergence abscissa
    alpha_cf = CellField(x -> PNS.alpha(alpha_field, x), Ω)

    f_cf, g_cf, _, _ = evaluate_exactness_diagnostics3d(mms, Ω, dΩ, c_1, c_2)

    h_scale = h_mean   # key tolerances off the ACHIEVED mesh size (nested family halves h exactly)
    spatial_err_est = h_scale^(kv + 1)
    budget = read_mms_dynamic_budget()   # [harness-frame] programmatic cell: inherits the defaults
    dynamic_ftol = max(sol.ftol, min(budget.ftol_ceiling, budget.ftol_spatial_safety_factor * spatial_err_est))
    condition_scaling = (1.0/h_mean)^2 * max(1.0, RE)
    dnf = min(sol.stagnation_noise_floor, max(sol.condition_noise_floor_absolute_min,
                                              sol.condition_noise_floor_baseline * condition_scaling))
    dnf = max(dnf, dynamic_ftol * sol.condition_noise_floor_safety_factor)

    _iter = PNS.build_iter_solvers(sol, linear_solver;
        newton_max_iters = sol.newton_iterations,
        picard_max_iters = sol.picard_iterations,
        newton_ftol = dynamic_ftol, picard_ftol = dynamic_ftol,
        stagnation_noise_floor = dnf,
        noise_floor_success_max_ftol_multiple = sol.noise_floor_success_max_ftol_multiple,
        stall_window = 2, stall_min_rel_improvement = 0.01)
    iter_solvers = PNS.StageSolvers(_iter.picard, _iter.newton)

    setup = PNS.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    vmsform = PNS.VMSFormulation(form, c_1, c_2)
    x0 = interpolate_everywhere([u_ex, p_ex], X)
    diag = Dict{String,Any}()
    success, _, final_x0, iters, etime = PNS.solve_system(setup, vmsform, iter_solvers, config, x0;
                                                          diagnostics_cache=diag, verifier=PNS.NoVerification())
    u_h, p_h = final_x0
    el2_u, el2_p, eh1_u, eh1_p = calc_errors3d(u_h, p_h, u_ex, p_ex, U_c, P_c, dΩ)
    # Newton/Picard iteration split from the stage trajectory (stages tagged ":N"/":P"); the solve is
    # from the exact-solution guess, so eps_pert = 0 (a direct solve, like the 2D corner cells).
    traj = get(diag, "trajectory", Any[])
    n_ns  = sum(Int[Int(get(s, :iters, 0)) for s in traj if endswith(String(s.stage), ":N")]; init=0)
    n_pic = sum(Int[Int(get(s, :iters, 0)) for s in traj if endswith(String(s.stage), ":P")]; init=0)
    res_init  = isempty(traj) ? NaN : Float64(get(first(traj), :res_in,  NaN))
    res_final = isempty(traj) ? NaN : Float64(get(last(traj),  :res_out, NaN))
    if trace_dir !== nothing
        _write_trajectory_sidecar(; outroot=trace_dir, run_name=run_name, kv=kv, method=method,
                                  h_mean=h_mean, ncells=num_cells(model), success=success,
                                  trajectory=traj, res_init=res_init, res_final=res_final,
                                  tol_M=sol.eps_tol_momentum, tol_C=sol.eps_tol_mass, mesh_sequence=mesh_sequence)
    end
    return (success=success, ncells=num_cells(model), iters=iters, h_mean=h_mean,
            el2_u=el2_u, el2_p=el2_p, eh1_u=eh1_u, eh1_p=eh1_p, n_ns=n_ns, n_pic=n_pic)
end

slope(e0,e1,h0,h1) = log(e0/e1)/log(h0/h1)

# Convergence run over the NESTED red-refined family (one base mesh, recursive 1->8 subdivision).
function run_convergence(kv, method, visc, eps_mult, geom, nlevels, base_lc)
    domain = geom == "cube" ? (0.0,1.0, 0.0,1.0, 0.0,1.0) : DOMAIN
    println("=== 3D MMS: kv=$kv method=$method visc=$visc eps_mult=$eps_mult geom=$geom, base_lc=$base_lc, $nlevels refinements ===")
    fam = build_nested_family(nlevels; lc=base_lc, domain=domain)
    res = NamedTuple[]
    for (lvl, model) in enumerate(fam)
        r = solve_one(kv, method, model; visc=visc, eps_mult=eps_mult,
                      trace_dir=joinpath(@__DIR__, "results"), run_name="run_convergence_$(geom)")
        push!(res, r)
        println("  level=$(lvl-1): cells=$(r.ncells) hmean=$(round(r.h_mean,sigdigits=4)) success=$(r.success) | " *
                "L2u=$(round(r.el2_u,sigdigits=4)) L2p=$(round(r.el2_p,sigdigits=4)) H1u=$(round(r.eh1_u,sigdigits=4))")
    end
    println("--- per-segment rates vs MEASURED h [P$(kv): L2u opt $(kv+1), H1u opt $kv, L2p opt $kv] ---")
    for i in 1:length(res)-1
        a, b = res[i], res[i+1]
        hr = round(a.h_mean/b.h_mean, digits=3)
        println("  h $(round(a.h_mean,sigdigits=3))->$(round(b.h_mean,sigdigits=3)) (h↓×$hr): " *
                "L2u=$(round(slope(a.el2_u,b.el2_u,a.h_mean,b.h_mean),digits=2)) " *
                "H1u=$(round(slope(a.eh1_u,b.eh1_u,a.h_mean,b.h_mean),digits=2)) " *
                "L2p=$(round(slope(a.el2_p,b.el2_p,a.h_mean,b.h_mean),digits=2))")
    end
    return res
end

# Full paper §5.2 sweep over the NESTED red-refined (recursive-subdivision) family, persisted to JSON
# for the table generator + plotter. P1 uses 4 meshes (levels 0-3), P2 uses 3 (levels 0-2); both
# ASGS and OSGS at the fixed (α,Re,Da)=(0.5,1,1). One base mesh is built and refined once, reused
# across kv/method. Writes incrementally so a partial run is not lost.
function run_sweep_and_save(; outpath, base_lc=0.2, geom="box", visc="Deviatoric", eps_mult=1.0)
    domain = geom == "cube" ? (0.0,1.0, 0.0,1.0, 0.0,1.0) : DOMAIN
    println("=== 3D MMS SWEEP (nested red-refined family) geom=$geom base_lc=$base_lc visc=$visc ==="); flush(stdout)
    fam = build_nested_family(3; lc=base_lc, domain=domain)   # 4 meshes: levels 0..3
    results = Any[]
    for kv in (1, 2)
        nmesh = kv == 1 ? 4 : 3                                 # P1: 4 meshes, P2: 3 (paper §5.2)
        for method in ("ASGS", "OSGS")
            @printf("\n--- kv=%d (%s) method=%s : %d meshes ---\n", kv, kv==1 ? "P1" : "P2", method, nmesh); flush(stdout)
            hs=Float64[]; l2us=Float64[]; l2ps=Float64[]; h1us=Float64[]; h1ps=Float64[]; levels=Any[]
            for lvl in 1:nmesh
                t0 = time()
                r = solve_one(kv, method, fam[lvl]; visc=visc, eps_mult=eps_mult,
                              trace_dir=dirname(outpath), run_name=splitext(basename(outpath))[1])
                push!(hs, r.h_mean); push!(l2us, r.el2_u); push!(l2ps, r.el2_p); push!(h1us, r.eh1_u); push!(h1ps, r.eh1_p)
                push!(levels, Dict("level"=>lvl-1, "h"=>r.h_mean, "ncells"=>r.ncells, "success"=>r.success,
                                   "iters"=>r.iters, "n_ns"=>r.n_ns, "n_pic"=>r.n_pic,
                                   "l2u"=>r.el2_u, "l2p"=>r.el2_p, "h1u"=>r.eh1_u, "h1p"=>r.eh1_p))
                @printf("  level %d: h=%.4g cells=%d success=%s | L2u=%.4g H1u=%.4g L2p=%.4g H1p=%.4g  (%.0fs)\n",
                        lvl-1, r.h_mean, r.ncells, r.success, r.el2_u, r.eh1_u, r.el2_p, r.eh1_p, time()-t0); flush(stdout)
            end
            push!(results, Dict("kv"=>kv, "kp"=>kv, "method"=>method, "element_type"=>"TET",
                                "mesh"=>"nested_red", "alpha_0"=>ALPHA0, "Re"=>RE, "Da"=>DA,
                                "hs"=>hs, "l2u"=>l2us, "l2p"=>l2ps, "h1u"=>h1us, "h1p"=>h1ps, "levels"=>levels))
            open(outpath, "w") do io
                JSON3.write(io, [Dict(k => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k,v) in d) for d in results])
            end
            println("  [wrote incremental -> $outpath]"); flush(stdout)
        end
    end
    println("\nDONE. 3D sweep results -> $outpath"); flush(stdout)
end

# Memory-capped, RESUMABLE remainder sweep. ASGS keeps its full paper mesh count (P1=4, P2=3); OSGS
# is capped to the meshes a DIRECT solver fits in RAM (P1=3, P2=2). The dropped finest OSGS meshes
# (P1 level-3 = 223744 tets; P2 level-2 = 27968 P2-tets ≈ 150K DOF) exceed this machine's memory —
# a HARDWARE limit, not a method limitation. Seeds `results` from any existing JSON and skips the
# (kv,method) blocks already present, so prior data (and its provenance) is preserved untouched.
# Builds the family only to level 2, so the 223744-tet model is never constructed.
function run_sweep_capped(; outpath, base_lc=0.2, geom="box", visc="Deviatoric", eps_mult=1.0)
    domain = geom == "cube" ? (0.0,1.0, 0.0,1.0, 0.0,1.0) : DOMAIN
    nmesh_for(kv, method) = method == "ASGS" ? (kv == 1 ? 4 : 3) : (kv == 1 ? 3 : 2)  # OSGS capped
    plan = [(kv, m) for kv in (1, 2) for m in ("ASGS", "OSGS")]

    # seed from existing JSON (preserve provenance), record which (kv,method) are already done
    results = Any[]; done = Set{Tuple{Int,String}}()
    if isfile(outpath)
        for d in JSON3.read(read(outpath, String))
            push!(results, Dict(String(k) => v for (k, v) in d))
            push!(done, (Int(d.kv), String(d.method)))
        end
        println("[resume] preserving $(length(done)) existing block(s): $(sort(collect(done)))"); flush(stdout)
    end

    todo = [(kv, m) for (kv, m) in plan if !((kv, m) in done)]
    isempty(todo) && (println("[resume] nothing to do — all blocks present."); return)
    maxmesh = maximum(nmesh_for(kv, m) for (kv, m) in todo)
    nlevels_build = maxmesh - 1                                # build_nested_family(n) -> n+1 models
    println("=== 3D MMS CAPPED SWEEP geom=$geom base_lc=$base_lc visc=$visc | building $(maxmesh) meshes (no level-3) ===")
    println("[plan] to compute: $(todo)  (OSGS capped: P1→3, P2→2)"); flush(stdout)
    fam = build_nested_family(nlevels_build; lc=base_lc, domain=domain)

    for (kv, method) in todo
        nmesh = nmesh_for(kv, method)
        @printf("\n--- kv=%d (%s) method=%s : %d meshes%s ---\n", kv, kv==1 ? "P1" : "P2", method, nmesh,
                method == "OSGS" ? " [OSGS capped for memory]" : ""); flush(stdout)
        hs=Float64[]; l2us=Float64[]; l2ps=Float64[]; h1us=Float64[]; h1ps=Float64[]; levels=Any[]
        for lvl in 1:nmesh
            t0 = time()
            r = solve_one(kv, method, fam[lvl]; visc=visc, eps_mult=eps_mult)
            push!(hs, r.h_mean); push!(l2us, r.el2_u); push!(l2ps, r.el2_p); push!(h1us, r.eh1_u); push!(h1ps, r.eh1_p)
            push!(levels, Dict("level"=>lvl-1, "h"=>r.h_mean, "ncells"=>r.ncells, "success"=>r.success,
                               "iters"=>r.iters, "n_ns"=>r.n_ns, "n_pic"=>r.n_pic,
                               "l2u"=>r.el2_u, "l2p"=>r.el2_p, "h1u"=>r.eh1_u, "h1p"=>r.eh1_p))
            @printf("  level %d: h=%.4g cells=%d success=%s | L2u=%.4g H1u=%.4g L2p=%.4g H1p=%.4g  (%.0fs)\n",
                    lvl-1, r.h_mean, r.ncells, r.success, r.el2_u, r.eh1_u, r.el2_p, r.eh1_p, time()-t0); flush(stdout)
            GC.gc()   # release the factorization before the next (larger) solve — keep peak RAM down
        end
        push!(results, Dict("kv"=>kv, "kp"=>kv, "method"=>method, "element_type"=>"TET",
                            "mesh"=>"nested_red", "alpha_0"=>ALPHA0, "Re"=>RE, "Da"=>DA,
                            "osgs_mesh_capped"=>(method == "OSGS"),
                            "hs"=>hs, "l2u"=>l2us, "l2p"=>l2ps, "h1u"=>h1us, "h1p"=>h1ps, "levels"=>levels))
        open(outpath, "w") do io
            JSON3.write(io, [Dict(k => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k,v) in d) for d in results])
        end
        println("  [wrote incremental -> $outpath]"); flush(stdout)
    end
    println("\nDONE. capped 3D sweep results -> $outpath"); flush(stdout)
end

# Parse a cells spec "kv=2,method=ASGS,lo=2,n=4;..." -> [(2,"ASGS",2,4),...] as (kv, method, lo, n):
# run the nested family's levels lo..n-1 (0-indexed) for that (kv, method). `lo` (default 0) lets you run
# ONLY the finer meshes — skipping coarse levels that diverge — so a two-finest slope comes from converged
# meshes alone. n is the exclusive upper level+1 (so n=4 means up to level 3, the 223744-tet mesh).
function _parse_cells(spec::String)
    cells = Tuple{Int,String,Int,Int}[]
    for chunk in split(spec, ';'; keepempty=false)
        kv = 0; meth = ""; n = 0; lo = 0
        for kvpair in split(chunk, ','; keepempty=false)
            kvparts = split(kvpair, '='; limit=2)
            length(kvparts) == 2 || error("bad cell field \"$kvpair\" (want key=value)")
            k = strip(kvparts[1]); v = strip(kvparts[2])
            k == "kv"     && (kv = parse(Int, v))
            k == "method" && (meth = String(v))
            k == "n"      && (n = parse(Int, v))
            k == "lo"     && (lo = parse(Int, v))
        end
        (kv in (1, 2) && meth in ("ASGS", "OSGS") && n >= 2 && 0 <= lo <= n - 1) ||
            error("bad cell spec chunk \"$chunk\" (need kv=1|2, method=ASGS|OSGS, n>=2, 0<=lo<=n-1)")
        push!(cells, (kv, meth, lo, n))
    end
    isempty(cells) && error("empty cells spec")
    return cells
end

# Provenance: snapshot the pre-upgrade JSON into the TRACKED previous_results/ archive + a manifest line,
# BEFORE a cells run overwrites any memory-capped block in place. This keeps the capped (paper-committed)
# OSGS numbers reconstructable to the exact config that produced them, honoring .agents/rules/
# reproducible-results.md ("make provenance explicit before consolidating").
function _archive_snapshot(outpath, cells)
    archdir = joinpath(@__DIR__, "previous_results", "convergence3d")
    mkpath(archdir)
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    snap = joinpath(archdir, "convergence3d_results_pre_cells_$(stamp).json")
    cp(outpath, snap; force=true)
    githead = try strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String)) catch; "unknown" end
    open(joinpath(archdir, "MANIFEST.md"), "a") do io
        println(io, "- `$(basename(snap))` — snapshot $(stamp) before full-mesh cells upgrade $(cells); " *
                    "git HEAD $(githead). Holds the prior memory-capped OSGS results so they stay reconstructable.")
    end
    println("  [provenance] archived pre-upgrade JSON -> $snap"); flush(stdout)
end

# On-demand FULL-mesh runner. Runs an explicit list of (kv, method, n_meshes) cells at FULL mesh count —
# including the fine meshes the capped sweep skipped for memory (solve_one auto-picks ILU-GMRES there) —
# and merges them into the sweep JSON, REPLACING any capped same-(kv,method) block in place. The capped
# block is first archived (see _archive_snapshot), so nothing is orphaned. Resumable: existing blocks for
# other (kv,method) are preserved untouched. This is how you "run the larger mesh later".
function run_sweep_cells(; outpath, cells::Vector{Tuple{Int,String,Int,Int}}, base_lc=0.2, geom="box",
                         visc="Deviatoric", eps_mult=1.0, linsolver::String="auto")
    domain = geom == "cube" ? (0.0,1.0, 0.0,1.0, 0.0,1.0) : DOMAIN
    results = Any[]; idx_of = Dict{Tuple{Int,String},Int}()
    if isfile(outpath)
        for d in JSON3.read(read(outpath, String))
            push!(results, Dict(String(k) => v for (k, v) in d))
            idx_of[(Int(d.kv), String(d.method))] = length(results)
        end
        _archive_snapshot(outpath, cells)
    end
    maxn = maximum(n for (_, _, _, n) in cells)
    cfgid_date = Dates.format(Dates.now(), "yyyymmdd")
    println("=== 3D MMS CELLS run $(cells) | building $(maxn) meshes (auto LU/ILU per mesh) base_lc=$base_lc ==="); flush(stdout)
    fam = build_nested_family(maxn - 1; lc=base_lc, domain=domain)
    for (kv, method, lo, nmesh) in cells
        @printf("\n--- CELL kv=%d (%s) method=%s : levels %d..%d (auto LU/ILU) ---\n",
                kv, kv==1 ? "P1" : "P2", method, lo, nmesh-1); flush(stdout)
        hs=Float64[]; l2us=Float64[]; l2ps=Float64[]; h1us=Float64[]; h1ps=Float64[]; levels=Any[]
        for lvl in (lo+1):nmesh
            t0 = time()
            r = solve_one(kv, method, fam[lvl]; visc=visc, eps_mult=eps_mult, linsolver=linsolver,
                          trace_dir=dirname(outpath), run_name=splitext(basename(outpath))[1])
            push!(hs, r.h_mean); push!(l2us, r.el2_u); push!(l2ps, r.el2_p); push!(h1us, r.eh1_u); push!(h1ps, r.eh1_p)
            push!(levels, Dict("level"=>lvl-1, "h"=>r.h_mean, "ncells"=>r.ncells, "success"=>r.success,
                               "iters"=>r.iters, "n_ns"=>r.n_ns, "n_pic"=>r.n_pic,
                               "l2u"=>r.el2_u, "l2p"=>r.el2_p, "h1u"=>r.eh1_u, "h1p"=>r.eh1_p))
            @printf("  level %d: h=%.4g cells=%d success=%s | L2u=%.4g H1u=%.4g L2p=%.4g H1p=%.4g  (%.0fs)\n",
                    lvl-1, r.h_mean, r.ncells, r.success, r.el2_u, r.eh1_u, r.el2_p, r.eh1_p, time()-t0); flush(stdout)
            GC.gc()
        end
        block = Dict("kv"=>kv, "kp"=>kv, "method"=>method, "element_type"=>"TET", "mesh"=>"nested_red",
                     "alpha_0"=>ALPHA0, "Re"=>RE, "Da"=>DA, "osgs_mesh_capped"=>false,
                     "result_config_id"=>"$(lowercase(method))_full_P$(kv)_lv$(lo)to$(nmesh-1)_$(cfgid_date)",
                     "hs"=>hs, "l2u"=>l2us, "l2p"=>l2ps, "h1u"=>h1us, "h1p"=>h1ps, "levels"=>levels)
        if haskey(idx_of, (kv, method))
            println("  [merge] replacing prior (kv=$kv,$method) block in place (snapshot archived)")
            results[idx_of[(kv, method)]] = block
        else
            push!(results, block); idx_of[(kv, method)] = length(results)
        end
        open(outpath, "w") do io
            JSON3.write(io, [Dict(k => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k,v) in d) for d in results])
        end
        println("  [wrote -> $outpath]"); flush(stdout)
    end
    println("\nDONE. cells run merged -> $outpath"); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "cells"
        # smoke3d.jl cells [outpath] "kv=2,method=ASGS,lo=2,n=4;..." [base_lc] [geom] [visc] [linsolver]
        #   each chunk runs nested-family levels lo..n-1 (lo default 0); lo>0 runs ONLY finer meshes.
        #   linsolver: "auto" (LU below LU_DOF_LIMIT, ILU above), "LU" (force direct), or "ILU_GMRES".
        outpath = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results", "convergence3d_results.json")
        spec    = length(ARGS) >= 3 ? ARGS[3] :
                  error("cells mode needs a spec, e.g. \"kv=1,method=OSGS,n=4;kv=2,method=OSGS,n=3\"")
        base_lc = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.2
        geom    = length(ARGS) >= 5 ? ARGS[5] : "box"
        visc    = length(ARGS) >= 6 ? ARGS[6] : "Deviatoric"
        linsolver = length(ARGS) >= 7 ? ARGS[7] : "auto"
        mkpath(dirname(outpath))
        run_sweep_cells(; outpath=outpath, cells=_parse_cells(spec), base_lc=base_lc, geom=geom,
                        visc=visc, linsolver=linsolver)
    elseif length(ARGS) >= 1 && ARGS[1] == "sweep_capped"
        # smoke3d.jl sweep_capped [outpath] [base_lc] [geom] [visc]  (resumable; OSGS mesh-capped for RAM)
        outpath  = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results", "convergence3d_results.json")
        base_lc  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.2
        geom     = length(ARGS) >= 4 ? ARGS[4] : "box"
        visc     = length(ARGS) >= 5 ? ARGS[5] : "Deviatoric"
        mkpath(dirname(outpath))
        run_sweep_capped(; outpath=outpath, base_lc=base_lc, geom=geom, visc=visc)
    elseif length(ARGS) >= 1 && ARGS[1] == "sweep"
        # smoke3d.jl sweep [outpath] [base_lc] [geom] [visc]
        outpath  = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results", "convergence3d_results.json")
        base_lc  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.2
        geom     = length(ARGS) >= 4 ? ARGS[4] : "box"
        visc     = length(ARGS) >= 5 ? ARGS[5] : "Deviatoric"
        mkpath(dirname(outpath))
        run_sweep_and_save(; outpath=outpath, base_lc=base_lc, geom=geom, visc=visc)
    else
        kv = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
        method = length(ARGS) >= 2 ? ARGS[2] : "ASGS"
        visc = length(ARGS) >= 3 ? ARGS[3] : "Deviatoric"
        eps_mult = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
        geom = length(ARGS) >= 5 ? ARGS[5] : "box"      # "box" (paper) or "cube" (clean control)
        nlevels = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
        base_lc = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.2
        run_convergence(kv, method, visc, eps_mult, geom, nlevels, base_lc)
    end
end
