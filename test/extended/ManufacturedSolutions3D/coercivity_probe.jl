# test/extended/ManufacturedSolutions3D/coercivity_probe.jl
# =============================================================================================
# DIAGNOSTIC (NOT the official results path — writes to results/debug_results/).
#
# Directly MEASURES the discrete coercivity constant of the ASGS momentum (1,1)-block as a
# function of c₁, to settle by measurement (not inference) the leading verdict of
# docs/mms/p2-3d.md §A: is the paper constant c₁ = 4k⁴ below the coercivity threshold on
# structured Kuhn tetrahedra?
#
# Method (paper §5.2 setup, Re=Da=1, α₀=0.5, deviatoric subscale):
#   1. Interpolate the MMS field onto the (u,p) space → the linearization point x_h.
#   2. Assemble the ASGS tangent A = ∂F/∂x at x_h (π_h = 0). PicardMode = the frozen-coefficient
#      stabilized BILINEAR form B_S whose coercivity the theory analyzes (∂τ/∂u, ∂L*/∂u Newton
#      terms are NOT part of the coercivity form); ExactNewton reported as a cross-check.
#   3. Take the velocity (1,1) sub-block S = ½(A_uu + A_uuᵀ) (the momentum coercivity operator;
#      pressure coupling is handled by inf-sup, not coercivity).
#   4. G = Galerkin deviatoric-viscous + reaction energy Gram matrix (velocity only) — the natural
#      energy norm the coercivity constant is measured in.
#   5. λ_min = smallest generalized eigenvalue of (S, G). COERCIVE ⟺ λ_min > 0. The zero-crossing
#      over c₁ is the coercivity threshold; c₁ = 4k⁴ is c1_mult = 1.0.
#
# KEY DESIGN CHOICE — c₁-ONLY scaling. Every c1_mult hook in the repo scales BOTH c₁ and c₂
# (smoke3d.jl:235, run_test.jl:560), confounding the viscous-coercivity constant (c₁) with the
# τ₂ mass/pressure stabilization (c₂). Here we scale c₁ ALONE (c₂ frozen at the paper value), so
# this probe IS the referee-proof c₁-only experiment. τ₁⁻¹ ∝ c₁·ν/h², so LARGER c₁ ⇒ smaller τ₁
# ⇒ weaker (anti-coercive) viscous subscale ⇒ λ_min INCREASES with c1_mult.
#
# Mesh-independence: the anti-coercive subscale scales as τ₁‖𝓛_visc V‖² ~ (h²/c₁ν)(C_inv²/h²)‖·‖²
# = (C_inv²/c₁ν)‖·‖² — h cancels. So the λ_min zero-crossing must be ~mesh-independent; we run a
# small ladder to demonstrate that (a stronger claim than a single mesh).
# =============================================================================================
using Gridap
using LinearAlgebra
using Printf
using JSON3
using PorousNSSolver
const PNS = PorousNSSolver
include(joinpath(@__DIR__, "smoke3d.jl"))   # helpers + §5.2 consts + mesh3d.jl (ARGS block PROGRAM_FILE-guarded)

const TAU_REG = 1e-12   # config/base_config.json:17 tau_regularization_limit (production value)

"""
Assemble (S, G, n_u) for one (mesh partition, c1_mult, h_conv, lin_mode).
Returns the symmetric velocity block S, the Galerkin energy Gram G (both n_u×n_u dense-able
sparse), and diagnostic metadata. c₁ is scaled by `c1_mult`; c₂ is held at the paper value.
"""
function probe_matrices(partition; kv::Int=2, c1_mult::Float64=1.0, h_conv::String="shortest_edge",
                        lin_mode::Symbol=:picard, study=default_study3d())
    RE=study.re; DA=study.da; ALPHA0=study.alpha0; ALPHAINF=study.alphainf
    R1=study.r1; R2=study.r2; L=study.Lc; U_AMP=study.u; DOMAIN=study.domain
    nu = U_AMP*L/RE
    eps_num = 1e-4*ALPHA0/(nu*(1.0+RE+DA))
    config = build_config(kv, "ASGS"; numerical_epsilon=eps_num, alpha0=ALPHA0, r1=R1, r2=R2)
    sol = config.numerical_method.solver
    sigma_c = DA*ALPHAINF*nu/L^2
    alpha_field = PNS.SmoothRadialPorosity(ALPHA0, ALPHAINF, R1, R2)

    proj = sol.experimental_reaction_mode == "standard" ?
           PNS.ProjectResidualWithoutReactionWhenConstantSigma() : PNS.ProjectFullResidual()
    reg  = PNS.SmoothVelocityFloor(config.physical_properties.u_base_floor_ref, 0.0,
                                   config.physical_properties.epsilon_floor,
                                   config.physical_properties.velocity_magnitude_derivative_floor)
    visc_op = PNS.DeviatoricSymmetricViscosity()
    form = PNS.PaperGeneralFormulation(visc_op, PNS.ConstantSigmaLaw(sigma_c), proj, reg, nu, 0.0;
                                       numerical_epsilon=eps_num)
    mms = PNS.PaperMMS(form, U_AMP, alpha_field; L=L, alpha_infty=ALPHAINF, dim=3)
    u_ex = PNS.get_u_ex(mms); p_ex = PNS.get_p_ex(mms)

    model = structured_kuhn_model(partition; domain=DOMAIN)
    labels = get_face_labeling(model)
    refe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kv)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["boundary"])
    Q = TestFESpace(model, refe_p, conformity=:H1)
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)
    Ut = TrialFESpace(V, u_ex); Pt = TrialFESpace(Q, p_ex)
    X = MultiFieldFESpace([Ut, Pt]); Y = MultiFieldFESpace([V, Q])

    degree = PNS.get_quadrature_degree(PNS.PaperGeneralFormulation, kv, PNS.ConstantSigmaLaw(0.0))
    Ω = Triangulation(model); dΩ = Measure(Ω, degree + 4)
    c_1, c_2 = PNS.get_c1_c2(PNS.PaperGeneralFormulation, kv)
    c_1 *= c1_mult                     # <<< c₁-ONLY scaling (c₂ held at the paper value)

    if h_conv == "diameter" || h_conv == "shortest_edge"
        cc = get_cell_coordinates(Ω)
        _redu = h_conv == "shortest_edge" ? minimum : maximum
        h_array = [_redu(sqrt((v[i]-v[j])⋅(v[i]-v[j])) for i in 1:length(v) for j in (i+1):length(v)) for v in cc]
    else
        _h_of_v = h_conv == "d_fact" ? (v -> (6.0*abs(v))^(1.0/3.0)) : (v -> (6.0*sqrt(2.0)*abs(v))^(1.0/3.0))
        h_array = collect(lazy_map(_h_of_v, get_cell_measure(Ω)))
    end
    h_cf = CellField(h_array, Ω)
    alpha_cf = CellField(x -> PNS.alpha(alpha_field, x), Ω)
    f_cf, g_cf = PNS.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, nothing)
    setup = PNS.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    vmsform = PNS.VMSFormulation(form, c_1, c_2)
    phys_cfg = (; tau_regularization_limit = TAU_REG)

    xh = interpolate_everywhere([u_ex, p_ex], X)     # MMS interpolant = linearization point
    mode = lin_mode == :newton ? PNS.ExactNewtonMode() : PNS.PicardMode()
    A = assemble_matrix(
        (dx, y) -> PNS.build_stabilized_weak_form_jacobian(xh, dx, y, setup, vmsform, phys_cfg, false, mode;
                                                           pi_u=nothing, pi_p=nothing),
        X, Y)
    G = assemble_matrix(
        (u, v) -> ∫( PNS.weak_viscous_operator(visc_op, u, v, alpha_cf, nu) + sigma_c*(u ⋅ v) )dΩ,
        Ut, V)

    n_u = num_free_dofs(Ut)
    A_uu = A[1:n_u, 1:n_u]
    S = 0.5 * (A_uu + transpose(A_uu))
    meta = (kv=kv, partition=collect(partition), h_conv=h_conv, lin_mode=String(lin_mode),
            c1_mult=c1_mult, c1_paper=4.0*kv^4, c1_used=c_1, c2=c_2,
            ndof_u=n_u, ncells=num_cells(model), h_mean=sum(h_array)/length(h_array), nu=nu, sigma_c=sigma_c)
    return S, G, meta
end

"Smallest generalized eigenvalue of the symmetric-definite pair (S, G) via dense LAPACK."
function lambda_min(S, G)
    Sd = Symmetric(Matrix(S)); Gd = Symmetric(Matrix(G))
    try
        return minimum(eigvals(Sd, Gd)), true          # generalized symmetric-definite (sygvd)
    catch err
        @warn "generalized eigensolve failed; falling back to standard eig of S (G not numerically PD?)" err
        return minimum(eigvals(Sd)), false
    end
end

function run_probe(; kv::Int=2, partitions=[(4,4,1),(6,6,2),(8,8,2)],
                   c1_mults=[0.25,0.5,1.0,2.0,3.0,4.0], h_convs=["shortest_edge"],
                   lin_modes=[:picard, :newton], outtag::String="")
    rows = Vector{Any}()
    @printf("\n%-14s %-7s %-6s %-6s %8s %10s %12s  %s\n",
            "partition", "hconv", "mode", "c1×", "ndof_u", "c1_used", "λ_min", "coercive?")
    println(repeat("-", 92))
    for p in partitions, hc in h_convs, lm in lin_modes
        for m in c1_mults
            S, G, meta = probe_matrices(p; kv=kv, c1_mult=m, h_conv=hc, lin_mode=lm)
            lmin, ok = lambda_min(S, G)
            push!(rows, merge(Dict(pairs(meta)), Dict("lambda_min"=>lmin, "gen_eig_ok"=>ok,
                                                      "coercive"=>lmin > 0)))
            @printf("%-14s %-7s %-6s %-6.2g %8d %10.4g %12.5g  %s\n",
                    string(Tuple(meta.partition)), hc, String(lm), m, meta.ndof_u, meta.c1_used,
                    lmin, lmin > 0 ? "YES" : "no"); flush(stdout)
        end
        println()
    end
    outdir = joinpath(@__DIR__, "results", "debug_results"); mkpath(outdir)
    outpath = joinpath(outdir, "coercivity_probe$(outtag).json")
    open(outpath, "w") do io
        JSON3.pretty(io, Dict("purpose"=>"ASGS momentum (1,1)-block coercivity vs c₁ (c₁-only), Kuhn P$kv",
                              "kv"=>kv, "rows"=>rows))
    end
    @printf("\n[wrote] %s (%d rows)\n", outpath, length(rows))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
    # default: small decisive ladder (demonstrates mesh-independence of the threshold), c₁-only,
    # production :shortest_edge units, Picard (coercivity form) + Newton cross-check.
    run_probe(; kv=kv)
end
