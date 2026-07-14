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
# Element-type-aware viscous stabilization constant c₁*(K) (theory/numerical_constants/
# c1_dimension_note.tex, transcribed + Table-1 validated). Guarded so the probes that include both
# this file and element_c1.jl do not re-`const` its Gauss nodes. [element-aware-c1]
@isdefined(element_c1_star) || include(joinpath(@__DIR__, "element_c1.jl"))
# The manufactured-solution oracle is the shared dimension-generic core in
# src/problems/mms_paper.jl (PNS.PaperMMS with dim=3 → the z-extruded 3D field). [unified 2026-07-08]
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

# Above this many monolithic (u,p) free DOFs, solve_one's "auto" linsolver switches to ILU-GMRES.
# [2026-07-13] RAISED 80_000 → 300_000 after MEASURING that direct sparse LU DOES fit the finest 3D cells
# on this machine: P2-L2 = 142706 DOF and P1-L3 ≈ 163000 DOF both factorize+solve under LU (~1–3h each,
# ~exact root). The old 80k cutoff forced those cells onto ILU-GMRES, which STAGNATES on the large
# saddle-point Jacobian (restart-30 GMRES plateaus at ~3% residual → step rejected → the cell returned the
# exact-guess interpolant with success=false). Root cause was the LINEAR SOLVER, not c₁/coercivity, ε, or a
# true hardware wall (disentangled 2026-07-13). A strengthened ILU (restart=200,drop=1e-6) matches the LU
# root for ASGS but NaN-breaks on the harder OSGS coupled/JFNK tangent — so direct LU is the robust choice
# for BOTH methods. [code-actual]. NOTE: the very finest OSGS cells are the memory risk; the sweep runs
# ASGS first so its ladders are safely written before any OSGS OOM.
const LU_DOF_LIMIT = 300_000
# [eps_pert] relative-L² gate classifying "converged to the SAME discrete root as the exact-guess reference".
# Same-root agreement is ~the solver's convergence tolerance (≈1e-6, the common discretization error cancels);
# a different (spurious) root is O(1) away. 1e-3 sits in that 3-order gap, so it rejects spurious roots without
# false-rejecting a genuine same-root start. Test-frame constant (not a production knob).
const ROOT_MATCH_TOL = 1e-3

# [F6 config-driven] The §5.2 PHYSICAL study parameters bundled as a NamedTuple, so a JSON config can
# override them (via `run_config`) while every existing CLI/sweep path keeps the committed defaults
# BYTE-IDENTICALLY. `domain` is the (x0,x1,y0,y1,z0,z1) slab. `solve_one` unpacks this into locals that
# SHADOW the module consts, so its body is unchanged; the default equals the consts, so no path drifts.
default_study3d() = (re=RE, da=DA, alpha0=ALPHA0, alphainf=ALPHAINF, r1=R1, r2=R2, Lc=L, u=U_AMP, domain=DOMAIN)

function build_config(kv::Int, method::String; eps_tol_m_over=nothing, ftol_over=nothing, eps_tol_mass_over=nothing,
                     numerical_epsilon::Float64=0.0, jfnk::Bool=false, anderson::Bool=false,
                     jfnk_maxiter=nothing, jfnk_restart=nothing, jfnk_reltol=nothing, jfnk_precond_c1_mult=nothing,
                     iterative_penalty::Bool=true, osgs_skip_boot::Bool=false, ablation::String="full",
                     recenter::Bool=false, alpha0::Float64=ALPHA0, r1::Float64=R1, r2::Float64=R2)
    @assert !(jfnk && anderson) "jfnk and anderson are mutually-exclusive OSGS paths"
    eps_tol_m    = something(eps_tol_m_over, kv == 2 ? 1e-9 : 1e-6)   # k=2 tightened gate (MEMORY lesson)
    # [Route B 2026-07-01] mass gate is now the Philosophy-A algebraic ‖r_C‖/D_C → 0, so default it
    # SYMMETRIC with the momentum gate (both residuals brought down the same way); overridable for A/B.
    eps_tol_mass = something(eps_tol_mass_over, eps_tol_m)
    ftol_v       = something(ftol_over, kv == 2 ? 1e-12 : 1e-10)
    solver_dict = Dict{String,Any}("eps_tol_momentum"=>eps_tol_m, "eps_tol_mass"=>eps_tol_mass, "ftol"=>ftol_v)
    # [iterative-penalty] Codina iterative penalty ε_num·(pⁿ−pⁿ⁻¹) in the mass residual (article.tex §5.2
    # line 1383) — REQUIRED for the 3D all-Dirichlet case (ill-posed at ε=0). ON by default here; acts only
    # because the harness sets numerical_epsilon = 1e-4·ε_ref > 0. Pass iterative_penalty=false for an A/B.
    solver_dict["iterative_penalty_enabled"] = iterative_penalty
    # [ablation] diagnostic: "picard_only" forces the Picard/Oseen Jacobian in every Newton slot (solver_core.jl),
    # so the CONVERGED solution is tested for solver-linearization independence. Default "full" = ExactNewton
    # (byte-identical to before). Used to test whether P2-3D "converged-but-wrong" is a Newton root-selection
    # artifact (paper used plain Picard) vs a discretization discrepancy (both Newton & Picard land on it).
    solver_dict["ablation_mode"] = ablation
    # [JFNK] opt in to the matrix-free full-tangent OSGS coupled solve (recovers the dropped ∂π/∂u
    # coupling). This is what the 2D k=2 OSGS recipe uses (data/phase1_quad_k2.json); the rest of the
    # osgs_jfnk_* params inherit from base_config.json via the deep-merge. No-op for ASGS.
    # The inner-GMRES budget is overridable for 3D, where each mat-vec re-projects the OSGS residual (so a
    # too-large maxiter is expensive) and the frozen-π preconditioner may need more Krylov vectors than 2D.
    if jfnk
        solver_dict["osgs_jfnk_enabled"] = true
        jfnk_maxiter === nothing || (solver_dict["osgs_jfnk_gmres_maxiter"] = jfnk_maxiter)
        jfnk_restart === nothing || (solver_dict["osgs_jfnk_gmres_restart"] = jfnk_restart)
        jfnk_reltol  === nothing || (solver_dict["osgs_jfnk_gmres_rel_tol"] = jfnk_reltol)
        jfnk_precond_c1_mult === nothing || (solver_dict["osgs_jfnk_precond_c1_mult"] = jfnk_precond_c1_mult)
    end
    # [Anderson] opt in to the STAGGERED OSGS outer fixed-point (freeze π → solve consistent frozen-π
    # system → re-project, Anderson-mixed). Each inner solve has a CONSISTENT tangent (no coupled-Newton
    # overshoot), and it is far cheaper than JFNK's matrix-free re-projecting GMRES at P2-3D scale. No-op for ASGS.
    anderson && (solver_dict["osgs_anderson_enabled"] = true)
    # [osgs_skip_asgs_boot] run OSGS directly from the (eps_pert) initial guess — no ASGS pre-boot (the boot
    # is a code-side safeguard, not in the paper; the eps_pert homotopy provides cold-start globalization).
    osgs_skip_boot && (solver_dict["osgs_skip_asgs_boot"] = true)
    # [pressure re-centering] Bound the all-Dirichlet pressure-mean drift −ρ/(ε_num·|Ω|) that the WEAK gauge
    # (ε_num pins pⁿ to pⁿ⁻¹, not to zero mean) accumulates each penalty pass — a single P2 pass can already
    # reach O(230) and dominate ‖p‖, corrupting the L²p error/rate and making the drift stopping-test spurious
    # (theory/pressure_recentering_note). Zero-meaning pⁿ each pass is EXACT (pure gauge; ε_phys=0 required, so
    # we drop the config's vestigial 1e-8 — the assembled form already uses eps_phys=0). Default off = legacy.
    recenter && (solver_dict["recenter_pressure_between_penalty_passes"] = true)
    cfg = Dict(
        "physical_properties" => Dict("nu"=>1.0, "physical_epsilon"=>(recenter ? 0.0 : 1e-8), "numerical_epsilon"=>numerical_epsilon,
                                      "reaction_model"=>"Constant_Sigma", "sigma_constant"=>1.0),
        "domain" => Dict("alpha_0"=>alpha0, "bounding_box"=>[0.0,1.0,0.0,1.0], "r_1"=>r1, "r_2"=>r2),
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
                                   trajectory, res_init, res_final, tol_M, tol_C, mesh_sequence::String="",
                                   re=RE, da=DA, alpha0=ALPHA0)
    try
        cell_rel = isempty(mesh_sequence) ? joinpath("k$(Int(kv))", "TET") :
                                            joinpath("k$(Int(kv))", "TET", mesh_sequence)
        traces_dir = joinpath(outroot, cell_rel, "traces")
        isdir(traces_dir) || mkpath(traces_dir)
        nlabel = max(1, round(Int, 1.0 / h_mean))
        trace_name = @sprintf("traj_Re%.0e_Da%.0e_a%.2f_kv%d_kp%d_TET_%s_N%d.json",
                              re, da, alpha0, Int(kv), Int(kv), String(method), nlabel)
        trace_obj = (
            run = run_name,
            cell = (Re=re, Da=da, alpha_0=alpha0, kv=Int(kv), kp=Int(kv), etype="TET",
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
                   eps_phys::Float64=0.0, mesh_sequence::String="", jfnk::Bool=false, anderson::Bool=false,
                   jfnk_maxiter=nothing, jfnk_restart=nothing, jfnk_reltol=nothing, jfnk_precond_c1_mult=nothing, iterative_penalty::Bool=true,
                   osgs_skip_boot::Bool=false, eps_pert_base::Float64=1.0, max_n_pert::Int=5,
                   ablation::String="full", h_conv::String="regular_tet", study=default_study3d(),
                   # [element-aware-c1] c1_mode="fixed" ⇒ legacy scalar c1_mult (and c2_mult, default = c1_mult).
                   # c1_mode="element_aware" ⇒ per-element coercivity floor c₁*(K) (element_c1.jl) reduced over
                   # the mesh by (c1_percentile, c1_safety), applied c₁-ONLY (c₂ stays at paper — only the
                   # viscous constant gates coercivity; c1_dimension_note.tex §, coercivity_probe.jl rationale).
                   c1_mode::String="fixed", c1_percentile::Float64=100.0, c1_safety::Float64=1.0,
                   c2_mult::Float64=NaN, recenter::Bool=false)
    # [F6 config-driven] Unpack the study into locals that SHADOW the module consts. Every reference below
    # (nu, sigma_c, the porosity, PaperMMS, the eps_pert domain bounds, the trace filename) now reads the
    # study; with the default study == the consts, this is byte-identical to the pre-config behaviour.
    RE = study.re; DA = study.da; ALPHA0 = study.alpha0; ALPHAINF = study.alphainf
    R1 = study.r1; R2 = study.r2; L = study.Lc; U_AMP = study.u; DOMAIN = study.domain
    nu = U_AMP * L / RE
    # ε_num = the NUMERICAL penalty (Codina ITERATIVE penalty, paper ε = 1e-4·ε_ref). The equation is
    # INCOMPRESSIBLE: there is NO physical compressibility, so eps_phys MUST default to 0. The iterative
    # penalty adds ε_num·pⁿ to the mass-equation LHS and ε_num·pⁿ⁻¹ (previous nonlinear iterate) to the RHS,
    # so the residual carries ε_num·(pⁿ−pⁿ⁻¹) (pinning the constant-pressure null mode — REQUIRED for the 3D
    # all-Dirichlet case, ill-posed at ε=0) and it CANCELS at convergence (pⁿ=pⁿ⁻¹), leaving the manufactured
    # (incompressible) solution UNALTERED — article.tex §5.2 line 1383. (Gated by iterative_penalty=true here;
    # solve_system runs the OUTER penalty loop.) A non-zero eps_phys would instead solve a genuinely
    # COMPRESSIBLE problem (εp in the residual AND εp_ex in the oracle g); only override for that experiment.
    eps_num = eps_mult * 1e-4 * ALPHA0 / (nu * (1.0 + RE + DA))
    config = build_config(kv, method; numerical_epsilon=eps_num, jfnk=jfnk, anderson=anderson,
                          jfnk_maxiter=jfnk_maxiter, jfnk_restart=jfnk_restart, jfnk_reltol=jfnk_reltol,
                          jfnk_precond_c1_mult=jfnk_precond_c1_mult,
                          iterative_penalty=iterative_penalty, osgs_skip_boot=osgs_skip_boot,
                          eps_tol_m_over=eps_tol_m_over, ftol_over=ftol_over, eps_tol_mass_over=eps_tol_mass_over,
                          ablation=ablation, recenter=recenter, alpha0=ALPHA0, r1=R1, r2=R2)
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
    # physical_epsilon = ε_phys (residual + Jacobian); ε_num is the Codina iterative penalty — it lives ONLY in the
    # Jacobian's pressure block (lagged to the iterate, it cancels in the residual) and vanishes at convergence.
    form = PNS.PaperGeneralFormulation(visc_op,
                                       PNS.ConstantSigmaLaw(sigma_c), proj, reg, nu, eps_phys;
                                       numerical_epsilon=eps_num)

    # dim=3 → z-extruded field; the oracle g reads ε_phys from form.physical_epsilon (0 ⇒ incompressible source).
    mms = PNS.PaperMMS(form, U_AMP, alpha_field; L=L, alpha_infty=ALPHAINF, dim=3)
    U_c, P_c = PNS.get_characteristic_scales(mms)
    u_ex = PNS.get_u_ex(mms); p_ex = PNS.get_p_ex(mms)

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
    # c₁/c₂ scaling happens AFTER h_array is built (element_aware needs the element geometry + the SAME
    # per-cell h the solver feeds τ). paper Remark (eq:conditions_on_num_param): coercivity needs
    # c1 > 2ξ·C_inv², and the OPTIMAL c1 is element-type dependent (c1_dimension_note.tex).

    # tet element size. [diagnostic h_conv]:
    #   "diameter"    = the LITERAL element diameter h_K = max‖xᵢ−xⱼ‖ (longest edge) — the mathematically
    #                   standard h_K the inverse inequality / interpolation theory (and the paper's C_inv, c₁)
    #                   are stated with. Correctly captures shape-anisotropic Kuhn tets (edges 1:√2:√3).
    #   "regular_tet" = (6√2·V)^{1/3} (regular-tet edge for volume V — the shipped default; a volume PROXY).
    #   "d_fact"      = (6·V)^{1/3} = (d!·V)^{1/d}, the dimension-consistent analog of the 2D harness's
    #                   √(2·Area) (→ h = grid spacing), i.e. the 3D volume formula WITHOUT the extra √2.
    # Tests whether the tet h-convention drives the P2-3D discrepancy (docs/mms/p2-3d.md §3).
    if h_conv == "diameter" || h_conv == "shortest_edge"
        cc = get_cell_coordinates(Ω)
        _redu = h_conv == "shortest_edge" ? minimum : maximum   # shortest edge (min ‖xᵢ−xⱼ‖) vs diameter (max)
        h_array = [_redu(sqrt((v[i]-v[j])⋅(v[i]-v[j])) for i in 1:length(v) for j in (i+1):length(v)) for v in cc]
    else
        _h_of_v = h_conv == "d_fact" ? (v -> (6.0*abs(v))^(1.0/3.0)) : (v -> (6.0*sqrt(2.0)*abs(v))^(1.0/3.0))
        h_array = collect(lazy_map(_h_of_v, get_cell_measure(Ω)))
    end
    h_cf = CellField(h_array, Ω)
    h_mean = sum(h_array) / length(h_array)   # ACHIEVED mesh size — the correct convergence abscissa
    alpha_cf = CellField(x -> PNS.alpha(alpha_field, x), Ω)

    # [element-aware-c1] scale c₁/c₂ now (h_array is available). "fixed": legacy scalar path (c₂ follows
    # c1_mult unless c2_mult given). "element_aware": c₁ ← c₁·mult_eff where mult_eff is the per-element
    # coercivity floor c₁*(K) reduced over the mesh at (c1_percentile, c1_safety) and floored at the paper
    # value; c₂ untouched (c₁-ONLY — only the viscous constant gates coercivity).
    c1_mult_eff = c1_mult
    if c1_mode == "element_aware"
        cc_cells = collect(get_cell_coordinates(Ω))
        c1_mult_eff = calibrated_c1_mult(cc_cells, h_array, kv;
                                         percentile=c1_percentile, safety=c1_safety, floor_at_paper=true)
        c_1 *= c1_mult_eff                       # c₁-ONLY; c₂ stays at the paper value
        @printf("    [c1 element-aware] kv=%d pctl=%.4g safety=%.3g -> mult_eff=%.3f (paper c₁=%.4g, used c₁=%.4g)\n",
                kv, c1_percentile, c1_safety, c1_mult_eff, 4.0*kv^4, c_1); flush(stdout)
    else
        c_1 *= c1_mult
        c_2 *= (isnan(c2_mult) ? c1_mult : c2_mult)
    end

    f_cf, g_cf = PNS.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, nothing)

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

    # [eps_pert homotopy] Port of the 2D run_test.jl Algorithm E (outer homotopy perturbation loop): try the
    # initial guess u0 = u_ex + eps_p·(‖u_ex‖/‖h_pert‖)·h_pert with eps_p = eps_pert_base/10^attempt down to
    # 0 (HARD→EASY), breaking at the first success. h_pert is a boundary-vanishing bubble × oscillatory field
    # (so u0 = u_ex on ∂Ω, respecting the Dirichlet BC) normalized so ‖perturbation‖ = eps_p·‖u_ex‖. eps_p=0
    # is the clean exact-guess. This mirrors how the 2D MMS sweep is run.
    bx0,bx1,by0,by1,bz0,bz1 = DOMAIN
    B_fn(x) = (x[1]-bx0)^2*(bx1-x[1])^2 * (x[2]-by0)^2*(by1-x[2])^2 * (x[3]-bz0)^2*(bz1-x[3])^2
    kp = pi / L
    h_raw_func(x) = B_fn(x) * VectorValue(sin(3*kp*x[1])*cos(2*kp*x[2]), -cos(3*kp*x[1])*sin(2*kp*x[2]), 0.0)
    u_ex_cf = CellField(u_ex, Ω)
    u_ex_L2 = sqrt(abs(sum(∫(u_ex_cf ⋅ u_ex_cf)dΩ)))
    h_pert_cf = CellField(h_raw_func, Ω)
    norm_h = sqrt(abs(sum(∫(h_pert_cf ⋅ h_pert_cf)dΩ)))
    norm_h > 0.0 || error("perturbation field norm must be > 0")

    # A far perturbed start can let the solver "converge" (the residual gate passes) into a SPURIOUS discrete
    # root — a different solution with O(1) MMS error, NOT the manufactured one (seen for P2 ASGS at eps_pert=1).
    # The exact-guess (eps_pert=0) start, by contrast, always lands in the TRUE root's basin, so it is the
    # reference (its error is the genuine O(h^{k+1}) discretization error). We solve the reference first, then
    # descend perturbed starts HARD→EASY and accept the largest whose converged field MATCHES the reference root
    # (relative-L² distance ≤ ROOT_MATCH_TOL). Errors are always reported from the reference; eps_used is the
    # largest perturbation that still reached it (the honest robustness metric).
    function _solve_from(eps_p)
        sc = eps_p * (u_ex_L2 / norm_h)
        u0_func = x -> u_ex(x) + sc * h_raw_func(x)
        x0 = interpolate_everywhere([u0_func, p_ex], X)
        d = Dict{String,Any}()
        s, _, fx, it, et = PNS.solve_system(setup, vmsform, iter_solvers, config, x0;
                                            diagnostics_cache=d, verifier=PNS.NoVerification())
        return (success=s, x=fx, iters=it, etime=et, diag=d)
    end

    @printf("    [eps_pert] reference solve (eps_pert=0, exact guess)\n"); flush(stdout)
    ref = _solve_from(0.0)
    success = ref.success; final_x0 = ref.x; iters = ref.iters; etime = ref.etime; diag = ref.diag
    eps_used = 0.0
    u_ref, _p_ref = ref.x
    u_ref_L2 = sqrt(abs(sum(∫(u_ref ⋅ u_ref)dΩ)))
    if ref.success
        for attempt in 0:max_n_pert
            eps_p = eps_pert_base / (10.0^attempt)   # HARD→EASY, perturbed starts only (eps_pert=0 is the reference)
            @printf("    [eps_pert attempt eps_pert=%.3g]\n", eps_p); flush(stdout)
            tr = _solve_from(eps_p)
            tr.success || continue
            u_try, _ = tr.x
            rel = sqrt(abs(sum(∫((u_try - u_ref) ⋅ (u_try - u_ref))dΩ))) / max(u_ref_L2, eps(Float64))
            if rel <= ROOT_MATCH_TOL
                eps_used = eps_p   # largest perturbation that reached the TRUE root
                break
            end
            @printf("      [rejected: spurious root, ‖u−u_ref‖/‖u_ref‖=%.3g > %.0e]\n", rel, ROOT_MATCH_TOL); flush(stdout)
        end
    end
    @printf("    [eps_pert] reference success=%s, robustness eps_used=%.3g\n", success, eps_used); flush(stdout)
    u_h, p_h = final_x0
    el2_u, el2_p, eh1_u, eh1_p = calc_errors3d(u_h, p_h, u_ex, p_ex, U_c, P_c, dΩ)
    # [OSGS-leak guard — pending-tasks §2c] A non-advancing OSGS "success" (the 0-iteration initial_ftol
    # short-circuit in nonlinear.jl) leaves final_x0 AT its entry iterate: the ASGS Stage-I boot root (default
    # path) or the eps_pert interpolant (boot-skip). That is NOT a converged OSGS root, so we mark the level
    # success=false. We KEEP the error value (do not NaN it): the entry state is itself diagnostic (e.g. OSGS
    # stuck at the interpolant — docs/mms/p2-3d.md §C), and `success` is the "special symbol" the analyzer/
    # plotter use to EXCLUDE it from rate fits and mark it distinctly, rather than hide it. The OSGS coupled
    # stage surfaces this path-agnostically (osgs_solver.jl; ASGS never sets the key → no-op for ASGS).
    if get(diag, "osgs_short_circuited_on_entry", false)
        @printf("    [OSGS-leak guard] OSGS reported success without advancing off its entry iterate — marking success=false (value kept; entry state, not a converged OSGS root).\n"); flush(stdout)
        success = false
    end
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
                                  tol_M=sol.eps_tol_momentum, tol_C=sol.eps_tol_mass, mesh_sequence=mesh_sequence,
                                  re=RE, da=DA, alpha0=ALPHA0)
    end
    return (success=success, ncells=num_cells(model), iters=iters, h_mean=h_mean,
            el2_u=el2_u, el2_p=el2_p, eh1_u=eh1_u, eh1_p=eh1_p, n_ns=n_ns, n_pic=n_pic,
            eps_used=eps_used,           # [eps_pert] largest perturbation from which this cell converged (robustness)
            c1_mult_eff=c1_mult_eff,     # [element-aware-c1] effective c₁ multiplier actually applied (provenance)
            numerical_epsilon=eps_num)   # the Codina iterative-penalty ε_num actually used (provenance)
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

# Full §5.2 sweep on the STRUCTURED Kuhn family (the "regular mesh"), run LIKE 2D: each cell uses solve_one's
# eps_pert HOMOTOPY (perturbed start, hard→easy) + the iterative penalty (default ON). Method-OUTER ordering:
# ALL ASGS first (P1 then P2), THEN all OSGS — so the fast, validated ASGS results land before the slow OSGS
# cells, per "run for ASGS and then for OSGS". ASGS uses the DEFAULT solver (coupled, ASGS boot ON — the honest
# baseline). OSGS uses the 3D recipe (boot-skip + JFNK): solve directly from the eps_pert guess (no ASGS-root
# detour) with the matrix-free JFNK inner solve that recovers ∂π/∂u — this also fast-fails doomed perturbations
# so the homotopy descent stays practical (the default coupled+boot OSGS grinds ~15min per failing attempt).
# Writes PER-kv to results/k<kv>/TET/structured/convergence3d_results.json (standard schema; mesh_sequence=
# "structured"), archiving any prior JSON to previous_results/convergence3d/ first (reproducible-results rule).
# Records the per-cell `eps_used` (largest perturbation it converged from) as the robustness map. Constant-aspect
# (1.2) Kuhn ladders, all LU-feasible: P1 (8,8,2)→(16,16,4)→(24,24,6)→(32,32,8); P2 (12,12,3)→(16,16,4)→(20,20,5).
function run_sweep_structured(; max_n_pert=3, c1_mult::Float64=1.0, recenter::Bool=true)
    ladders = Dict(1 => [(8,8,2),(16,16,4),(24,24,6),(32,32,8)], 2 => [(12,12,3),(16,16,4),(20,20,5)])
    # c1_mult>1 ⇒ ROBUST c₁×N in the RESIDUAL (c₁-only; c₂ stays at paper — only the viscous constant gates
    # coercivity). SINGLE canonical channel per test: always the `structured` leaf. The recipe (c1_mult,
    # recenter, solver block) is self-described INSIDE each JSON record, and the prior run is archived to
    # previous_results/convergence3d/ before overwrite — so provenance is preserved WITHOUT forking the
    # channel into variant leaves (single-channel-per-test rule). Default c1_mult=1.0 = paper c₁.
    seq = "structured"
    archdir = joinpath(@__DIR__, "previous_results", "convergence3d"); mkpath(archdir)
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    outdirs  = Dict(kv => joinpath(@__DIR__, "results", "k$(kv)", "TET", seq) for kv in (1, 2))
    outpaths = Dict(kv => joinpath(outdirs[kv], "convergence3d_results.json") for kv in (1, 2))
    for kv in (1, 2)
        mkpath(outdirs[kv])
        if isfile(outpaths[kv])   # archive the prior JSON before overwriting (provenance)
            cp(outpaths[kv], joinpath(archdir, "convergence3d_results_k$(kv)_$(seq)_$(stamp).json"); force=true)
        end
    end
    @printf("=== STRUCTURED SWEEP: c1_mult=%.3g (residual, c₁-only) -> leaf '%s' ===\n", c1_mult, seq); flush(stdout)
    results_by_kv = Dict(1 => Any[], 2 => Any[])   # accumulates each (method) block per kv across the outer loop
    for method in ("ASGS", "OSGS")
        osgs_recipe = method == "OSGS"
        for kv in (1, 2)
            @printf("\n=== STRUCTURED SWEEP method=%s kv=%d (%s) ===\n", method, kv, kv==1 ? "P1" : "P2"); flush(stdout)
            # [JFNK precond-c₁] OSGS-P2-3D: the frozen-π tangent is a WEAK preconditioner for the coupled ∂π/∂u
            # system — ρ(J_frozen⁻¹·∂π/∂u) ≈ 1178 at paper c₁, so inner GMRES stalls and the solver sits at the
            # exact-guess interpolant (success=false). A preconditioner-ONLY c₁×4 inflation drops ρ_prec to ≈3.8
            # (residual F stays paper-c₁, so the converged root is unchanged), restoring quadratic Newton and
            # eps_used=1 robustness. OSGS-P1 is robust at mult=1. See docs/mms/p2-3d.md §6.
            osgs_p2 = osgs_recipe && kv == 2
            # OSGS-P2 JFNK preconditioner c₁-inflation is ONLY needed when the residual is at paper c₁
            # (ρ_prec≈1178 ⇒ GMRES stalls). With c₁×4 ALREADY in the residual (c1_mult≥4) ρ_prec is O(1), so
            # NO extra preconditioner inflation (relative ×1). See docs/mms/p2-3d.md §6 + run_sweep_nested_red.
            pc_mult = osgs_p2 ? (c1_mult >= 4.0 ? nothing : 4.0) : nothing
            jfnk_mx = osgs_recipe ? (osgs_p2 ? 80 : 30) : nothing   # more GMRES headroom for P2
            hs=Float64[]; l2us=Float64[]; l2ps=Float64[]; h1us=Float64[]; h1ps=Float64[]; levels=Any[]
            eps_num_used = NaN   # the ε_num actually used (constant across levels here; captured for provenance)
            for (lvl, part) in enumerate(ladders[kv])
                t0 = time()
                model = structured_kuhn_model(part; domain=DOMAIN)
                r = solve_one(kv, method, model; visc="Deviatoric", linsolver="LU", max_n_pert=max_n_pert,
                              c1_mult=c1_mult, c2_mult=1.0,   # robust c₁×N in the residual, c₁-ONLY (c₂ at paper)
                              recenter=recenter,              # zero-mean the lagged pressure each penalty pass (honest gauge)
                              jfnk=osgs_recipe, osgs_skip_boot=osgs_recipe,
                              jfnk_maxiter=jfnk_mx, jfnk_restart=jfnk_mx, jfnk_precond_c1_mult=pc_mult,
                              trace_dir=outdirs[kv], run_name="sweep_structured", mesh_sequence=seq)
                eps_num_used = r.numerical_epsilon
                push!(hs, r.h_mean); push!(l2us, r.el2_u); push!(l2ps, r.el2_p); push!(h1us, r.eh1_u); push!(h1ps, r.eh1_p)
                push!(levels, Dict("level"=>lvl-1, "partition"=>collect(part), "h"=>r.h_mean, "ncells"=>r.ncells,
                                   "success"=>r.success, "iters"=>r.iters, "eps_used"=>r.eps_used,
                                   "l2u"=>r.el2_u, "l2p"=>r.el2_p, "h1u"=>r.eh1_u, "h1p"=>r.eh1_p))
                @printf("  L%d part=%s h=%.4g cells=%d success=%s eps_used=%.3g | L2u=%.4g H1u=%.4g L2p=%.4g  (%.0fs)\n",
                        lvl-1, string(part), r.h_mean, r.ncells, r.success, r.eps_used, r.el2_u, r.eh1_u, r.el2_p, time()-t0); flush(stdout)
                GC.gc()
            end
            # [provenance] self-describing solver recipe so an official result can be reconstructed to the exact
            # configuration that produced it (reproducible-results rule): ASGS uses the default coupled solve with
            # the ASGS Stage-I boot; OSGS uses the 3D recipe (boot-skip + matrix-free JFNK, recovering ∂π/∂u).
            solver_prov = Dict("recipe"=>(osgs_recipe ? (pc_mult !== nothing ? "boot_skip+JFNK+precond_c1x$(Int(round(pc_mult)))" : "boot_skip+JFNK") : "default_coupled+boot"),
                               "jfnk"=>osgs_recipe, "osgs_skip_asgs_boot"=>osgs_recipe,
                               "jfnk_maxiter"=>jfnk_mx, "jfnk_restart"=>jfnk_mx,
                               "jfnk_precond_c1_mult"=>something(pc_mult, 1.0),
                               "c1_scaling"=>"residual_c1_only", "c2_mult"=>1.0,
                               "iterative_penalty"=>true, "recenter_pressure"=>recenter, "numerical_epsilon"=>eps_num_used,
                               "eps_pert_base"=>1.0, "max_n_pert"=>max_n_pert, "linsolver"=>"LU",
                               "viscous_operator"=>"Deviatoric", "reaction"=>"Constant_Sigma")
            push!(results_by_kv[kv], Dict("kv"=>kv, "kp"=>kv, "method"=>method, "element_type"=>"TET",
                                "mesh"=>"structured_kuhn", "mesh_sequence"=>seq, "c1_mult"=>c1_mult,
                                "alpha_0"=>ALPHA0, "Re"=>RE, "Da"=>DA, "numerical_epsilon"=>eps_num_used,
                                "solver"=>solver_prov,
                                "hs"=>hs, "l2u"=>l2us, "l2p"=>l2ps, "h1u"=>h1us, "h1p"=>h1ps, "levels"=>levels))
            open(outpaths[kv], "w") do io
                JSON3.write(io, [Dict(k => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k,v) in d) for d in results_by_kv[kv]])
            end
            println("  [wrote incremental -> $(outpaths[kv])]"); flush(stdout)
        end
    end
    println("\nDONE. structured sweep (c1_mult=$c1_mult) -> results/k{1,2}/TET/$seq/convergence3d_results.json"); flush(stdout)
end

# Full paper §5.2 sweep on the IRREGULAR nested-red family — the paper's own 3D methodology (an UNSTRUCTURED
# Delaunay base mesh "to break the symmetry, so the velocity vectors have a nonzero z-component", refined by
# uniform red 1→8 subdivision so h halves EXACTLY down the sequence) — at the element-aware c₁ (c1_mult,
# default 4.0 = the adopted robust worst-case for high-C_inv tets; docs/mms/p2-3d.md §A, findings.md §3). This
# is the irregular-mesh sibling of run_sweep_structured: same schema, same OSGS-3D recipe (boot-skip + JFNK;
# the preconditioner inflation OSGS-P2 needs is DIFFERENT here — see the osgs_p2_precond_c1_mult note below),
# but over the refined unstructured family with P1→nlevels_p1+1
# and P2→nlevels_p2+1 meshes (paper: 4 for P1, 3 for P2). Auto linear backend (LU ≤ LU_DOF_LIMIT, ILU-GMRES
# above — the finest meshes at ~164K DOF). Writes results/k<kv>/TET/nested_red/convergence3d_results.json
# (standard plottable schema, self-describing with c1_mult + the solver recipe), archiving any prior JSON to
# previous_results/convergence3d/ first (reproducible-results rule). Plot: `plot_convergence3d.py nested_red`.
# NOTE on osgs_p2_precond_c1_mult (default 1.0): with c1_mult already inflating the RESIDUAL c₁ (default 4),
# the ∂π/∂u coupling (∝ τ₁ ∝ 1/c₁) is already shrunk, so JFNK converges QUADRATICALLY with the frozen-π
# tangent as-is — no ADDITIONAL preconditioner inflation is needed (de-risk 2026-07-11: OSGS-P2 L0 at c1×4,
# relative-×1 precond → 5 Newton iters to ‖f‖≈6e-12). Contrast run_sweep_structured, which keeps the residual
# at paper c₁ and therefore MUST inflate the preconditioner ×4 (ρ_prec 1249→3.8). Raise this only if a finer
# mesh's ILU preconditioner needs more margin.
function run_sweep_nested_red(; c1_mult::Float64=4.0, max_n_pert::Int=-1, base_lc::Float64=0.2,
                              nlevels_p1::Int=3, nlevels_p2::Int=2, osgs_p2_precond_c1_mult::Float64=4.0,
                              recenter::Bool=true)
    archdir = joinpath(@__DIR__, "previous_results", "convergence3d"); mkpath(archdir)
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    outdirs  = Dict(kv => joinpath(@__DIR__, "results", "k$(kv)", "TET", "nested_red") for kv in (1, 2))
    outpaths = Dict(kv => joinpath(outdirs[kv], "convergence3d_results.json") for kv in (1, 2))
    for kv in (1, 2)
        mkpath(outdirs[kv])
        if isfile(outpaths[kv])   # archive the prior nested_red JSON before overwriting (provenance)
            cp(outpaths[kv], joinpath(archdir, "convergence3d_results_k$(kv)_nested_red_$(stamp).json"); force=true)
        end
    end
    nlev_for = Dict(1 => nlevels_p1, 2 => nlevels_p2)
    maxlev = max(nlevels_p1, nlevels_p2)
    @printf("=== NESTED-RED (irregular) SWEEP: c1_mult=%.3g base_lc=%.3g | P1→%d meshes P2→%d meshes ===\n",
            c1_mult, base_lc, nlevels_p1 + 1, nlevels_p2 + 1); flush(stdout)
    println("[mesh] building irregular nested_red family to level $maxlev (reused across kv/method)..."); flush(stdout)
    fam = build_nested_family(maxlev; lc=base_lc, domain=DOMAIN)
    results_by_kv = Dict(1 => Any[], 2 => Any[])
    for method in ("ASGS", "OSGS")
        osgs_recipe = method == "OSGS"
        for kv in (1, 2)
            @printf("\n=== NESTED-RED SWEEP method=%s kv=%d (%s) ===\n", method, kv, kv==1 ? "P1" : "P2"); flush(stdout)
            # OSGS-P2-3D: the frozen-π tangent is a WEAK preconditioner for the coupled ∂π/∂u system, so the
            # matrix-free JFNK-GMRES needs a c₁-inflated preconditioner (residual F stays at the study c₁, so
            # the converged root is unchanged). OSGS-P1 is robust at relative mult=1. See docs/mms/p2-3d.md §C.
            osgs_p2 = osgs_recipe && kv == 2
            pc_mult = osgs_p2 ? osgs_p2_precond_c1_mult : nothing
            jfnk_mx = osgs_recipe ? (osgs_p2 ? 80 : 30) : nothing
            models = fam[1:(nlev_for[kv] + 1)]
            hs=Float64[]; l2us=Float64[]; l2ps=Float64[]; h1us=Float64[]; h1ps=Float64[]; levels=Any[]
            eps_num_used = NaN
            for (lvl, model) in enumerate(models)
                t0 = time()
                # [resilience] the finest ILU meshes can OOM/throw; catch so one bad cell doesn't kill the run.
                r = try
                    solve_one(kv, method, model; visc="Deviatoric", linsolver="auto", c1_mult=c1_mult,
                              c2_mult=1.0, recenter=recenter,   # c₁-only residual inflation + honest pressure gauge (re-centering)
                              max_n_pert=max_n_pert, jfnk=osgs_recipe, osgs_skip_boot=osgs_recipe,
                              jfnk_maxiter=jfnk_mx, jfnk_restart=jfnk_mx, jfnk_precond_c1_mult=pc_mult,
                              # trace_dir is the results ROOT; _write_trajectory_sidecar appends k<kv>/TET/<seq>/traces
                              # (so traces land at results/k<kv>/TET/nested_red/traces/, beside the JSON — NOT the
                              # doubly-nested path run_sweep_structured accidentally produces by passing the leaf).
                              trace_dir=joinpath(@__DIR__, "results"), run_name="sweep_nested_red", mesh_sequence="nested_red")
                catch err
                    @printf("  L%d THREW %s — recording failure, continuing\n", lvl-1, sprint(showerror, err)); flush(stdout)
                    (success=false, ncells=num_cells(model), iters=0, h_mean=mesh_hmean(model),
                     el2_u=NaN, el2_p=NaN, eh1_u=NaN, eh1_p=NaN, n_ns=0, n_pic=0, eps_used=0.0,
                     c1_mult_eff=NaN, numerical_epsilon=NaN)
                end
                isfinite(r.numerical_epsilon) && (eps_num_used = r.numerical_epsilon)
                push!(hs, r.h_mean); push!(l2us, r.el2_u); push!(l2ps, r.el2_p); push!(h1us, r.eh1_u); push!(h1ps, r.eh1_p)
                push!(levels, Dict("level"=>lvl-1, "h"=>r.h_mean, "ncells"=>r.ncells,
                                   "success"=>r.success, "iters"=>r.iters, "eps_used"=>r.eps_used,
                                   "l2u"=>r.el2_u, "l2p"=>r.el2_p, "h1u"=>r.eh1_u, "h1p"=>r.eh1_p))
                @printf("  L%d h=%.4g cells=%d success=%s | L2u=%.4g H1u=%.4g L2p=%.4g  (%.0fs)\n",
                        lvl-1, r.h_mean, r.ncells, r.success, r.el2_u, r.eh1_u, r.el2_p, time()-t0); flush(stdout)
                GC.gc()
            end
            # [provenance] self-describing recipe so the result reconstructs to the exact config that made it.
            _pc_label = (pc_mult !== nothing && pc_mult != 1.0) ? "+precond_c1x$(pc_mult)" : ""
            solver_prov = Dict("recipe"=>(osgs_recipe ? "boot_skip+JFNK" * _pc_label : "default_coupled+boot"),
                               "jfnk"=>osgs_recipe, "osgs_skip_asgs_boot"=>osgs_recipe,
                               "jfnk_maxiter"=>jfnk_mx, "jfnk_restart"=>jfnk_mx,
                               "jfnk_precond_c1_mult"=>something(pc_mult, 1.0),
                               "iterative_penalty"=>true, "recenter_pressure"=>recenter,
                               "c1_scaling"=>"residual_c1_only", "c2_mult"=>1.0, "numerical_epsilon"=>eps_num_used,
                               "max_n_pert"=>max_n_pert, "linsolver"=>"auto",
                               "viscous_operator"=>"Deviatoric", "reaction"=>"Constant_Sigma")
            push!(results_by_kv[kv], Dict("kv"=>kv, "kp"=>kv, "method"=>method, "element_type"=>"TET",
                                "mesh"=>"nested_red", "mesh_sequence"=>"nested_red", "c1_mult"=>c1_mult,
                                "base_lc"=>base_lc, "alpha_0"=>ALPHA0, "Re"=>RE, "Da"=>DA,
                                "numerical_epsilon"=>eps_num_used, "solver"=>solver_prov,
                                "hs"=>hs, "l2u"=>l2us, "l2p"=>l2ps, "h1u"=>h1us, "h1p"=>h1ps, "levels"=>levels))
            open(outpaths[kv], "w") do io
                JSON3.write(io, [Dict(k => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k,v) in d) for d in results_by_kv[kv]])
            end
            println("  [wrote incremental -> $(outpaths[kv])]"); flush(stdout)
        end
    end
    println("\nDONE. nested_red (irregular) sweep -> results/k{1,2}/TET/nested_red/convergence3d_results.json"); flush(stdout)
end

# =============================================================================================
# [element-aware-c1] C₁-STRATEGY STUDY on the IRREGULAR nested_red family.
# The overnight experiment for theory/numerical_constants/c1_dimension_note.tex: does clearing the
# irregular mesh's WORST-element coercivity floor (element_c1.jl) fix the k=2 rate depression that the
# Kuhn-calibrated empirical c1_mult=4 leaves? Builds the nested_red family ONCE (so every c₁ strategy
# is compared on IDENTICAL meshes — gmsh is not bit-reproducible across builds), then runs, per kv/
# method, a ladder of c₁ strategies:
#   k=2: c1fixed4 (baseline ×4) | c1aware_p90 | c1aware_p99 | c1aware_max   (element-aware, c₁-only)
#   k=1: c1fixed4 (baseline ×4) | c1aware (→ paper c₁, since c₁*(K)≡0 for P1 — the control)
# Each strategy writes its OWN self-describing leaf results/k<kv>/TET/nested_red_<label>/ (baseline
# nested_red/ untouched — provenance preserved), recording c1_mode/percentile/safety + the PER-LEVEL
# effective multiplier (element-aware mult grows with refinement as the tail degrades). Same OSGS-3D
# recipe as run_sweep_nested_red (boot-skip + JFNK; precond mult=1 — the inflated residual c₁ already
# shrinks ρ_prec). Plot each: `python plot_convergence3d.py nested_red_<label>`.
_c1study_strategies(kv) = kv == 2 ? [
        (label="c1fixed4",    mode="fixed",         mult=4.0, pctl=100.0, safety=1.0),
        (label="c1aware_p90", mode="element_aware", mult=1.0, pctl=90.0,  safety=1.0),
        (label="c1aware_p99", mode="element_aware", mult=1.0, pctl=99.0,  safety=1.0),
        (label="c1aware_max", mode="element_aware", mult=1.0, pctl=100.0, safety=1.0),
    ] : [
        (label="c1fixed4",    mode="fixed",         mult=4.0, pctl=100.0, safety=1.0),
        (label="c1aware",     mode="element_aware", mult=1.0, pctl=100.0, safety=1.0),   # →paper c₁ (P1: c₁*≡0)
    ]

function run_c1study_nested_red(; base_lc::Float64=0.2, nlevels_p1::Int=3, nlevels_p2::Int=2, max_n_pert::Int=-1)
    archdir = joinpath(@__DIR__, "previous_results", "convergence3d"); mkpath(archdir)
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    maxlev = max(nlevels_p1, nlevels_p2)
    nlev_for = Dict(1 => nlevels_p1, 2 => nlevels_p2)
    println("=== C1-STUDY (element-aware) on nested_red: building family to level $maxlev (reused across ALL strategies) ==="); flush(stdout)
    fam = build_nested_family(maxlev; lc=base_lc, domain=DOMAIN)
    # k=2 first (the decisive P2 science), then k=1 controls; ASGS before OSGS (cheaper) within each.
    for kv in (2, 1)
        strategies = _c1study_strategies(kv)
        models = fam[1:(nlev_for[kv] + 1)]
        for strat in strategies
            seq = "nested_red_$(strat.label)"
            outdir = joinpath(@__DIR__, "results", "k$(kv)", "TET", seq); mkpath(outdir)
            outpath = joinpath(outdir, "convergence3d_results.json")
            if isfile(outpath)
                cp(outpath, joinpath(archdir, "convergence3d_results_k$(kv)_$(seq)_$(stamp).json"); force=true)
            end
            @printf("\n########## kv=%d  strategy=%s (mode=%s pctl=%.4g)  ##########\n",
                    kv, strat.label, strat.mode, strat.pctl); flush(stdout)
            blocks = Any[]
            for method in ("ASGS", "OSGS")
                osgs_recipe = method == "OSGS"
                osgs_p2 = osgs_recipe && kv == 2
                jfnk_mx = osgs_recipe ? (osgs_p2 ? 80 : 30) : nothing
                @printf("\n--- kv=%d %s method=%s ---\n", kv, strat.label, method); flush(stdout)
                hs=Float64[]; l2us=Float64[]; l2ps=Float64[]; h1us=Float64[]; h1ps=Float64[]; levels=Any[]
                eps_num_used = NaN
                for (lvl, model) in enumerate(models)
                    t0 = time()
                    # [resilience] an unattended overnight run must survive a single bad solve (OOM on the
                    # finest mesh, ILU non-convergence throwing, …): catch, record a failure level, continue.
                    r = try
                        solve_one(kv, method, model; visc="Deviatoric", linsolver="auto",
                                  c1_mode=strat.mode, c1_mult=strat.mult, c1_percentile=strat.pctl, c1_safety=strat.safety,
                                  max_n_pert=max_n_pert, jfnk=osgs_recipe, osgs_skip_boot=osgs_recipe,
                                  jfnk_maxiter=jfnk_mx, jfnk_restart=jfnk_mx,
                                  trace_dir=joinpath(@__DIR__, "results"), run_name="c1study_$(strat.label)",
                                  mesh_sequence=seq)
                    catch err
                        @printf("  L%d THREW %s — recording failure, continuing\n", lvl-1, sprint(showerror, err)); flush(stdout)
                        (success=false, ncells=num_cells(model), iters=0, h_mean=mesh_hmean(model),
                         el2_u=NaN, el2_p=NaN, eh1_u=NaN, eh1_p=NaN, n_ns=0, n_pic=0, eps_used=0.0,
                         c1_mult_eff=NaN, numerical_epsilon=NaN)
                    end
                    eps_num_used = isfinite(r.numerical_epsilon) ? r.numerical_epsilon : eps_num_used
                    push!(hs, r.h_mean); push!(l2us, r.el2_u); push!(l2ps, r.el2_p); push!(h1us, r.eh1_u); push!(h1ps, r.eh1_p)
                    push!(levels, Dict("level"=>lvl-1, "h"=>r.h_mean, "ncells"=>r.ncells, "success"=>r.success,
                                       "iters"=>r.iters, "eps_used"=>r.eps_used, "c1_mult_eff"=>r.c1_mult_eff,
                                       "l2u"=>r.el2_u, "l2p"=>r.el2_p, "h1u"=>r.eh1_u, "h1p"=>r.eh1_p))
                    @printf("  L%d h=%.4g cells=%d success=%s c1×=%.3g | L2u=%.4g H1u=%.4g L2p=%.4g  (%.0fs)\n",
                            lvl-1, r.h_mean, r.ncells, r.success, r.c1_mult_eff, r.el2_u, r.eh1_u, r.el2_p, time()-t0); flush(stdout)
                    GC.gc()
                end
                solver_prov = Dict("recipe"=>(osgs_recipe ? "boot_skip+JFNK" : "default_coupled+boot"),
                                   "jfnk"=>osgs_recipe, "osgs_skip_asgs_boot"=>osgs_recipe,
                                   "jfnk_maxiter"=>jfnk_mx, "jfnk_precond_c1_mult"=>1.0,
                                   "iterative_penalty"=>true, "numerical_epsilon"=>eps_num_used,
                                   "max_n_pert"=>max_n_pert, "linsolver"=>"auto",
                                   "viscous_operator"=>"Deviatoric", "reaction"=>"Constant_Sigma")
                push!(blocks, Dict("kv"=>kv, "kp"=>kv, "method"=>method, "element_type"=>"TET",
                                   "mesh"=>"nested_red", "mesh_sequence"=>seq,
                                   "c1_mode"=>strat.mode, "c1_mult"=>strat.mult,
                                   "c1_percentile"=>strat.pctl, "c1_safety"=>strat.safety,
                                   "base_lc"=>base_lc, "alpha_0"=>ALPHA0, "Re"=>RE, "Da"=>DA,
                                   "numerical_epsilon"=>eps_num_used, "solver"=>solver_prov,
                                   "hs"=>hs, "l2u"=>l2us, "l2p"=>l2ps, "h1u"=>h1us, "h1p"=>h1ps, "levels"=>levels))
                open(outpath, "w") do io
                    JSON3.write(io, [Dict(k => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k,v) in d) for d in blocks])
                end
                println("  [wrote incremental -> $outpath]"); flush(stdout)
            end
        end
    end
    println("\nDONE. c1-study -> results/k{1,2}/TET/nested_red_<strategy>/convergence3d_results.json"); flush(stdout)
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

# ==============================================================================================
# [F6 config-driven] Official, committed, config-driven 3D-MMS convergence run. A self-describing JSON
# (data/smoke3d_p1.json) fixes the §5.2 PHYSICAL case (`study`) + the discretization / mesh study, and
# drives the SAME solve_one machinery — replacing the hand-edited-driver workflow for the official smoke
# path (audit F6 / pending-tasks 2a). Returns per-(kv,method) convergence data for the automated guard
# (test/extended/mms3d_config_smoke_extended_test.jl). The larger diagnostic studies keep the CLI sweep
# modes below.
# ==============================================================================================

# Build the physical §5.2 `study` NamedTuple from the parsed config (bounding_box is the 3D slab).
function _study_from_config(cfg)
    pp = cfg.physical_properties
    dm = cfg.domain
    bb = Float64.(collect(dm.bounding_box))                 # (x0,x1,y0,y1,z0,z1)
    length(bb) == 6 || error("config domain.bounding_box must be the 3D slab [x0,x1,y0,y1,z0,z1] (got $(length(bb)) entries)")
    return (re=Float64(pp.Re), da=Float64(pp.Da), alpha0=Float64(dm.alpha_0),
            alphainf=Float64(pp.alpha_infty), r1=Float64(dm.r_1), r2=Float64(dm.r_2),
            Lc=Float64(pp.L), u=Float64(pp.U), domain=Tuple(bb))
end

# Build the mesh sequence named by the config (structured Kuhn from explicit partitions — the uniform,
# refinement-invariant control the smoke uses — or the red-refined nested family), on the study slab.
function _models_from_config(cfg, domain)
    mesh = cfg.numerical_method.mesh
    seq = String(mesh.sequence)
    if seq == "structured"
        parts = [Tuple(Int.(collect(p))) for p in mesh.partitions]
        return [structured_kuhn_model(p; domain=domain) for p in parts], string(parts)
    elseif seq == "nested_red"
        nlev = Int(get(mesh, :levels, 2)); lc = Float64(get(mesh, :base_lc, 0.2))
        return build_nested_family(nlev; lc=lc, domain=domain), "nested_red(levels=$nlev, base_lc=$lc)"
    else
        error("config numerical_method.mesh.sequence must be \"structured\" or \"nested_red\" (got \"$seq\")")
    end
end

"""
    run_config(path) -> Vector of (kv, method, levels)

Run the §5.2 convergence study described by the JSON at `path` (see data/smoke3d_p1.json). Loops every
`kv × method` over the configured mesh sequence, driving `solve_one` with the config's physical `study`.
Prints per-level normalized errors + per-segment rates and returns, for each (kv, method), the vector of
per-level `(h, el2_u, el2_p, eh1_u, eh1_p, success, ncells)` — consumed by the official extended guard.
"""
function run_config(path::String)
    cfg = JSON3.read(read(path, String))
    study = _study_from_config(cfg)
    nm = cfg.numerical_method
    kvs = Int.(collect(nm.element_spaces.k_velocity))
    methods = String.(collect(nm.stabilization.methods))
    visc = String(get(nm.stabilization, :viscous_operator, "Deviatoric"))
    c1_mult = Float64(get(nm.stabilization, :c1_mult, 1.0))
    eps_mult = Float64(get(nm.stabilization, :eps_mult, 1.0))
    # Optional `solver` block = the OSGS-3D robustness recipe. jfnk (matrix-free full ∂π/∂u tangent) +
    # osgs_skip_boot let OSGS-P1-3D converge in a few inner Newton steps instead of the coupled Newton
    # capping out (the default-solver hang seen at the finest mesh); eps_pert_sweep=false runs the
    # reference exact-guess solve ONLY (a convergence smoke needs the reference error, not the
    # perturbation-robustness sweep). All three are no-ops for ASGS.
    solver_cfg = get(nm, :solver, nothing)
    _sget(k, d) = solver_cfg === nothing ? d : get(solver_cfg, k, d)
    jfnk = Bool(_sget(:jfnk, false))
    osgs_skip_boot = Bool(_sget(:osgs_skip_boot, false))
    eps_pert_sweep = Bool(_sget(:eps_pert_sweep, true))
    # max_n_pert < 0 ⇒ `for attempt in 0:max_n_pert` is an empty range ⇒ NO perturbed attempts ⇒ reference solve only.
    max_n_pert = eps_pert_sweep ? Int(_sget(:max_n_pert, 5)) : -1
    seq = String(nm.mesh.sequence)
    models, mesh_desc = _models_from_config(cfg, study.domain)
    tag = splitext(basename(path))[1]
    outroot = joinpath(@__DIR__, "results", "config_" * tag)

    println("=== 3D MMS CONFIG run: $(basename(path)) ==="); flush(stdout)
    println("    study: Re=$(study.re) Da=$(study.da) α0=$(study.alpha0) α∞=$(study.alphainf) " *
            "r1=$(study.r1) r2=$(study.r2) L=$(study.Lc) U=$(study.u) slab=$(study.domain)")
    println("    mesh:  $seq $mesh_desc;  kv=$kvs methods=$methods visc=$visc c1_mult=$c1_mult eps_mult=$eps_mult")
    println("    solver: jfnk=$jfnk osgs_skip_boot=$osgs_skip_boot eps_pert_sweep=$eps_pert_sweep (max_n_pert=$max_n_pert)"); flush(stdout)

    out = NamedTuple[]
    for kv in kvs, method in methods
        @printf("\n--- kv=%d (%s) method=%s ---\n", kv, kv == 1 ? "P1" : "P2", method); flush(stdout)
        levels = NamedTuple[]
        for (i, model) in enumerate(models)
            r = solve_one(kv, method, model; visc=visc, c1_mult=c1_mult, eps_mult=eps_mult,
                          study=study, mesh_sequence=seq, trace_dir=outroot, run_name="config_" * tag,
                          jfnk=jfnk, osgs_skip_boot=osgs_skip_boot, max_n_pert=max_n_pert)
            push!(levels, (h=r.h_mean, el2_u=r.el2_u, el2_p=r.el2_p, eh1_u=r.eh1_u, eh1_p=r.eh1_p,
                           success=r.success, ncells=r.ncells))
            @printf("  level=%d: cells=%d h=%.4g success=%s | L2u=%.4g H1u=%.4g L2p=%.4g\n",
                    i - 1, r.ncells, r.h_mean, r.success, r.el2_u, r.eh1_u, r.el2_p); flush(stdout)
        end
        println("  rates [P$kv opt: L2u $(kv+1), H1u $kv, L2p $kv]:")
        for i in 1:length(levels)-1
            a, b = levels[i], levels[i+1]
            @printf("    h %.3g->%.3g: L2u=%.2f H1u=%.2f L2p=%.2f\n",
                    a.h, b.h, slope(a.el2_u, b.el2_u, a.h, b.h), slope(a.eh1_u, b.eh1_u, a.h, b.h),
                    slope(a.el2_p, b.el2_p, a.h, b.h)); flush(stdout)
        end
        push!(out, (kv=kv, method=method, levels=levels))
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "config"
        # smoke3d.jl config <path.json>  — official config-driven §5.2 convergence run (audit F6).
        cfgpath = length(ARGS) >= 2 ? ARGS[2] : error("config mode needs a JSON path, e.g. data/smoke3d_p1.json")
        run_config(cfgpath)
    elseif length(ARGS) >= 1 && ARGS[1] == "sweep_structured"
        # smoke3d.jl sweep_structured [max_n_pert] [c1_mult]  — full §5.2 sweep on the STRUCTURED Kuhn family
        # with the eps_pert homotopy + iterative penalty + pressure re-centering. c1_mult>1 ⇒ ROBUST c₁×N in
        # the residual (c₁-only). SINGLE channel results/k*/TET/structured/ (recipe self-described in the JSON;
        # prior run archived to previous_results/). All LU-feasible, every cell certifies. Default 1.0 = paper c₁.
        mnp = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
        c1m = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
        run_sweep_structured(; max_n_pert=mnp, c1_mult=c1m)
    elseif length(ARGS) >= 1 && ARGS[1] == "sweep_nested_red"
        # smoke3d.jl sweep_nested_red [c1_mult] [max_n_pert] [nlevels_p1] [nlevels_p2] [osgs_p2_precond_c1_mult]
        #   — full §5.2 sweep on the IRREGULAR nested-red (unstructured-base + red-refined) family at the
        #   element-aware c₁ (default 4.0). P1→nlevels_p1+1 meshes, P2→nlevels_p2+1 (default 4/3, the paper
        #   counts). Writes results/k{1,2}/TET/nested_red/. Plot: `python plot_convergence3d.py nested_red`.
        c1m   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 4.0
        mnp   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : -1
        nlp1  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 3
        nlp2  = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2
        pcm   = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 4.0
        run_sweep_nested_red(; c1_mult=c1m, max_n_pert=mnp, nlevels_p1=nlp1, nlevels_p2=nlp2,
                             osgs_p2_precond_c1_mult=pcm)
    elseif length(ARGS) >= 1 && ARGS[1] == "c1study_nested_red"
        # smoke3d.jl c1study_nested_red [nlevels_p1] [nlevels_p2] [max_n_pert]
        #   — element-aware c₁ study on ONE nested_red family: baseline ×4 vs the element_c1 percentile
        #   ladder (k=2: p90/p99/max; k=1: →paper c₁). Writes results/k{1,2}/TET/nested_red_<strategy>/.
        nlp1 = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
        nlp2 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
        mnp  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : -1
        run_c1study_nested_red(; nlevels_p1=nlp1, nlevels_p2=nlp2, max_n_pert=mnp)
    elseif length(ARGS) >= 1 && ARGS[1] == "cells"
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
