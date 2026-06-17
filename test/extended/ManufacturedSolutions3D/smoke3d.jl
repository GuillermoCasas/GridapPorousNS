# test/extended/ManufacturedSolutions3D/smoke3d.jl
# Smoke / convergence check for the 3D MMS solve path (paper §5.2). Runs the full pipeline
# (gmsh tet mesh -> porosity -> ConstantSigma DeviatoricSymmetric formulation -> 3D FE spaces ->
# z-extruded MMS forcing oracle -> solve_system -> normalized errors) over a short mesh sequence
# and prints the convergence rate. Optimal velocity-L2 rate O(h^{kv+1}) validates the oracle. [must-test]
using Gridap
using Gridap.Algebra
using GridapGmsh
using PorousNSSolver
const PNS = PorousNSSolver
include("mesh3d.jl")
include("mms3d.jl")

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

function build_config(kv::Int, method::String)
    eps_tol_m = kv == 2 ? 1e-9 : 1e-6        # k=2 tightened gate (MEMORY lesson)
    ftol_v    = kv == 2 ? 1e-12 : 1e-10
    cfg = Dict(
        "physical_properties" => Dict("nu"=>1.0, "eps_val"=>1e-8,
                                      "reaction_model"=>"Constant_Sigma", "sigma_constant"=>1.0),
        "domain" => Dict("alpha_0"=>ALPHA0, "bounding_box"=>[0.0,1.0,0.0,1.0], "r_1"=>R1, "r_2"=>R2),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity"=>kv, "k_pressure"=>kv),
            "mesh" => Dict("element_type"=>"TRI", "partition"=>[1,1]),  # placeholders; gmsh model built directly
            "stabilization" => Dict("method"=>method),
            "solver" => Dict("eps_tol_momentum"=>eps_tol_m, "ftol"=>ftol_v),
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

function solve_one(kv::Int, method::String, model; visc::String="Deviatoric", eps_mult::Float64=1.0)
    config = build_config(kv, method)
    sol = config.numerical_method.solver

    nu = U_AMP * L / RE
    sigma_c = DA * ALPHAINF * nu / L^2
    eps_val = eps_mult * 1e-4 * ALPHA0 / (nu * (1.0 + RE + DA))   # paper ε = 1e-4·ε_ref, dimensional (×eps_mult to probe)
    alpha_field = PNS.SmoothRadialPorosity(ALPHA0, ALPHAINF, R1, R2)

    proj = sol.experimental_reaction_mode == "standard" ?
           PNS.ProjectResidualWithoutReactionWhenConstantSigma() : PNS.ProjectFullResidual()
    reg  = PNS.SmoothVelocityFloor(config.physical_properties.u_base_floor_ref, 0.0,
                                   config.physical_properties.epsilon_floor)
    visc_op = visc == "SymmetricGradient" ? PNS.SymmetricGradientViscosity() :
              visc == "Laplacian" ? PNS.LaplacianPseudoTractionViscosity() :
              PNS.DeviatoricSymmetricViscosity()
    form = PNS.PaperGeneralFormulation(visc_op,
                                       PNS.ConstantSigmaLaw(sigma_c), proj, reg, nu, eps_val)

    mms = Paper3DMMS(form, U_AMP, alpha_field, L, ALPHAINF)
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

    degree = PNS.get_quadrature_degree(PNS.PaperGeneralFormulation, kv, PNS.ConstantSigmaLaw(0.0))
    Ω = Triangulation(model); dΩ = Measure(Ω, degree + 4)
    c_1, c_2 = PNS.get_c1_c2(PNS.PaperGeneralFormulation, kv)

    # tet element size from cell VOLUME: regular-tet edge = (6√2·V)^{1/3}
    h_array = collect(lazy_map(v -> (6.0*sqrt(2.0)*abs(v))^(1.0/3.0), get_cell_measure(Ω)))
    h_cf = CellField(h_array, Ω)
    h_mean = sum(h_array) / length(h_array)   # ACHIEVED mesh size — the correct convergence abscissa
    alpha_cf = CellField(x -> PNS.alpha(alpha_field, x), Ω)

    f_cf, g_cf, _, _ = evaluate_exactness_diagnostics3d(mms, Ω, dΩ, c_1, c_2)

    h_scale = h_mean   # key tolerances off the ACHIEVED mesh size (nested family halves h exactly)
    spatial_err_est = h_scale^(kv + 1)
    dynamic_ftol = max(sol.ftol, min(sol.dynamic_ftol_ceiling, sol.dynamic_ftol_spatial_safety_factor * spatial_err_est))
    condition_scaling = (1.0/h_mean)^2 * max(1.0, RE)
    dnf = min(sol.stagnation_noise_floor, max(sol.condition_noise_floor_absolute_min,
                                              sol.condition_noise_floor_baseline * condition_scaling))
    dnf = max(dnf, dynamic_ftol * sol.condition_noise_floor_safety_factor)

    _iter = PNS.build_iter_solvers(sol, LUSolver();
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
    return (success=success, ncells=num_cells(model), iters=iters, h_mean=h_mean,
            el2_u=el2_u, el2_p=el2_p, eh1_u=eh1_u, eh1_p=eh1_p)
end

slope(e0,e1,h0,h1) = log(e0/e1)/log(h0/h1)

# Convergence run over the NESTED red-refined family (one base mesh, recursive 1->8 subdivision).
function run_convergence(kv, method, visc, eps_mult, geom, nlevels, base_lc)
    domain = geom == "cube" ? (0.0,1.0, 0.0,1.0, 0.0,1.0) : DOMAIN
    println("=== 3D MMS: kv=$kv method=$method visc=$visc eps_mult=$eps_mult geom=$geom, base_lc=$base_lc, $nlevels refinements ===")
    fam = build_nested_family(nlevels; lc=base_lc, domain=domain)
    res = NamedTuple[]
    for (lvl, model) in enumerate(fam)
        r = solve_one(kv, method, model; visc=visc, eps_mult=eps_mult)
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

if abspath(PROGRAM_FILE) == @__FILE__
    kv = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
    method = length(ARGS) >= 2 ? ARGS[2] : "ASGS"
    visc = length(ARGS) >= 3 ? ARGS[3] : "Deviatoric"
    eps_mult = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
    geom = length(ARGS) >= 5 ? ARGS[5] : "box"      # "box" (paper) or "cube" (clean control)
    nlevels = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
    base_lc = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.2
    run_convergence(kv, method, visc, eps_mult, geom, nlevels, base_lc)
end
