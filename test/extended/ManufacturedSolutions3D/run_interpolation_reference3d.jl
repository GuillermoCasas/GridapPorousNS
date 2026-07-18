# ==============================================================================================
# INTERPOLATION REFERENCE for the 3D manufactured-solution tables (tab:3DL2, tab:3DH1).
#
# Pure nodal-interpolation error of the exact 3D manufactured fields on the SAME mesh sequences and with
# the SAME normalization the 3D solver rows use (calc_errors3d, smoke3d.jl) — no solver, no stabilization.
# Regular Kuhn family (P1: (8,8,2)->(32,32,8); P2: (12,12,3)->(24,24,6)) and irregular nested-red family
# (from the COMMITTED base mesh, so it is reproducible). High quadrature (degree 14) to rule out
# measurement pollution, as in interp_test.jl.
#
# The reference is Re/Da/method-independent and, for the pressure, alpha_0-independent (verified in 2D).
# Here alpha_0 = 0.5 as in the 3D tables. Run:
#   julia --project=../../.. run_interpolation_reference3d.jl
# ==============================================================================================
using Gridap, GridapGmsh, PorousNSSolver, JSON3
const PNS = PorousNSSolver
include("mesh3d.jl")

# Where the machine-readable interpolation reference lands, so make_3d_tables.py can transcribe the
# interp rows of tab:3DL2 / tab:3DH1 straight from data (no hand-copying). Overwrites in place; the
# family/kv are keyed inside, so the file is self-describing.
const INTERP_JSON = joinpath(@__DIR__, "results", "interp_reference3d.json")

# Match smoke3d.jl exactly.
const DOMAIN = (0.0,1.0, 0.0,1.0, 0.0,0.3)
const ALPHA0 = 0.5
const ALPHAINF = 1.0
const R1, R2 = 0.2, 0.4
const L = 1.0
const U_AMP = 1.0
const RE, DA = 1.0, 1.0
# Quadrature: match the solver rows exactly (smoke3d builds Measure at get_quadrature_degree+4 = 4*kv+4),
# so the reference is measured like-for-like with the table entries. This is also far cheaper than a flat
# high degree on the 200k-tet nested-red finest mesh. Quadrature adequacy at this degree was established
# in 2D (the reference converged to 4-5 s.f. and degree 8/12 vs 20 agreed to ~1e-7).
qdeg_for(kv) = 4*kv + 4

slope(e0,e1,h0,h1) = log(e0/e1)/log(h0/h1)

# calc_errors3d, replicated verbatim from smoke3d.jl:119-129 (the pressure error is mean-centred; L2 is
# divided by U_c/P_c * sqrt(|Omega|); H1 by U_c/P_c only).
function calc_errors3d(u_h, p_h, u_ex, p_ex, U_c, P_c, dΩ)
    e_u = u_ex - u_h; e_p = p_ex - p_h
    area = sum(∫(1.0)dΩ); sqrt_area = sqrt(abs(area))
    mean_e_p = sum(∫(e_p)dΩ) / area; e_p_c = e_p - mean_e_p
    el2_u = sqrt(abs(sum(∫(e_u ⋅ e_u)dΩ))) / (U_c * sqrt_area)
    el2_p = sqrt(abs(sum(∫(e_p_c * e_p_c)dΩ))) / (P_c * sqrt_area)
    eh1_u = sqrt(abs(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ))) / U_c
    eh1_p = sqrt(abs(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ))) / P_c
    return el2_u, el2_p, eh1_u, eh1_p
end

# The exact 3D fields + characteristic scales, exactly as smoke3d.jl builds them (a formulation is needed
# only to construct PaperMMS; the viscous/reaction choice does not affect the exact field or the scales).
function mms_setup(kv)
    nu = U_AMP * L / RE
    sigma_c = DA * ALPHAINF * nu / L^2
    alpha_field = PNS.SmoothRadialPorosity(ALPHA0, ALPHAINF, R1, R2)
    form = PNS.PaperGeneralFormulation(PNS.DeviatoricSymmetricViscosity(), PNS.ConstantSigmaLaw(sigma_c),
                                       PNS.ProjectFullResidual(), PNS.SmoothVelocityFloor(1e-4, 0.0, 1e-8, 1e-12),
                                       nu, 0.0)
    mms = PNS.PaperMMS(form, U_AMP, alpha_field; L=L, alpha_infty=ALPHAINF, dim=3)
    U_c, P_c = PNS.get_characteristic_scales(mms)
    return PNS.get_u_ex(mms), PNS.get_p_ex(mms), U_c, P_c
end

function reference_on(models, kv)
    u_ex, p_ex, U_c, P_c = mms_setup(kv)
    qdeg = qdeg_for(kv)
    res = []
    for model in models
        Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, kv), conformity=:H1)
        Qp = TestFESpace(model, ReferenceFE(lagrangian, Float64, kv), conformity=:H1)
        u_I = interpolate_everywhere(u_ex, Vu); p_I = interpolate_everywhere(p_ex, Qp)
        Ω = Triangulation(model); dΩ = Measure(Ω, qdeg)
        el2_u, el2_p, eh1_u, eh1_p = calc_errors3d(u_I, p_I, u_ex, p_ex, U_c, P_c, dΩ)
        h = mesh_hmean(model)
        push!(res, (h=h, cells=num_cells(model), el2_u=el2_u, el2_p=el2_p, eh1_u=eh1_u, eh1_p=eh1_p))
        @printf("    h=%.4g cells=%7d | L2u=%.4e H1u=%.4e | L2p=%.4e H1p=%.4e\n", h, num_cells(model), el2_u, eh1_u, el2_p, eh1_p); flush(stdout)
    end
    sl(f) = slope(f(res[end-1]), f(res[end]), res[end-1].h, res[end].h)
    @printf("    slope(two finest): L2u=%.2f H1u=%.2f L2p=%.2f H1p=%.2f  | FME(finest): L2u=%.3e H1u=%.3e L2p=%.3e H1p=%.3e\n",
            sl(x->x.el2_u), sl(x->x.eh1_u), sl(x->x.el2_p), sl(x->x.eh1_p),
            res[end].el2_u, res[end].eh1_u, res[end].el2_p, res[end].eh1_p); flush(stdout)
    # Machine-readable summary for the table generator: two-finest slopes + finest-mesh error (FME).
    summary = Dict(
        "kv" => kv, "alpha_0" => ALPHA0,
        "slope" => Dict("l2u"=>sl(x->x.el2_u), "h1u"=>sl(x->x.eh1_u), "l2p"=>sl(x->x.el2_p), "h1p"=>sl(x->x.eh1_p)),
        "fme"   => Dict("l2u"=>res[end].el2_u, "h1u"=>res[end].eh1_u, "l2p"=>res[end].el2_p, "h1p"=>res[end].eh1_p),
        "levels" => [Dict("h"=>r.h, "cells"=>r.cells, "l2u"=>r.el2_u, "h1u"=>r.eh1_u, "l2p"=>r.el2_p, "h1p"=>r.eh1_p) for r in res],
    )
    return res, summary
end

using Printf
function main()
    kuhn = Dict(1 => [(8,8,2),(16,16,4),(24,24,6),(32,32,8)], 2 => [(12,12,3),(16,16,4),(20,20,5),(24,24,6)])
    out = Dict{String,Any}("regular" => Dict{String,Any}(), "irregular" => Dict{String,Any}())
    # Regular Kuhn first (fast, deterministic) so those numbers land before the heavy nested-red family.
    for kv in (1, 2)
        println("\n=== REGULAR (Kuhn) P$kv interpolation reference (alpha_0=$ALPHA0) ==="); flush(stdout)
        models = [structured_kuhn_model(p; domain=DOMAIN) for p in kuhn[kv]]
        _, summary = reference_on(models, kv)
        out["regular"][string(kv)] = summary
    end
    for (kv, nlev) in ((1,3),(2,2))
        println("\n=== IRREGULAR (nested-red) P$kv interpolation reference (alpha_0=$ALPHA0) ==="); flush(stdout)
        fam = build_nested_family(nlev; lc=0.2, domain=DOMAIN, algorithm=1)
        _, summary = reference_on(fam, kv)
        out["irregular"][string(kv)] = summary
    end
    mkpath(dirname(INTERP_JSON))
    open(INTERP_JSON, "w") do io
        JSON3.pretty(io, out)
    end
    println("\nWrote interpolation reference -> $INTERP_JSON"); flush(stdout)
end

main()
