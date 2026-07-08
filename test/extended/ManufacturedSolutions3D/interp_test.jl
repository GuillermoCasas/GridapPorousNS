# Pure P1 interpolation error of u_ex — NO solver, NO oracle, NO stabilization, NO compressibility.
# Measures the best-approximation capability of the FE space on the gmsh tet meshes, with HIGH
# quadrature (degree 14) to rule out error-measurement quadrature pollution. Decisive separation of
# "mesh/field best-approximation" from "method". Optimal P1: L2 ~ h^2, H1 ~ h^1.
using Gridap, GridapGmsh, PorousNSSolver
const PNS = PorousNSSolver
include("mesh3d.jl")   # MMS oracle now lives in src/problems/mms_paper.jl (PNS.UExFunc{3}). [unified 2026-07-08]

slope(e0,e1,h0,h1) = log(e0/e1)/log(h0/h1)

function interp_errs(domain, lcs; kv=1, qdeg=14)
    alpha_field = PNS.SmoothRadialPorosity(0.5, 1.0, 0.2, 0.4)
    u_ex = PNS.UExFunc{3}(1.0, 0.5, alpha_field, 1.0)
    res = []
    for lc in lcs
        model = build_box_tet_model(lc; domain=domain, algorithm=1)
        refe = ReferenceFE(lagrangian, VectorValue{3,Float64}, kv)
        V = TestFESpace(model, refe, conformity=:H1)
        uh = interpolate_everywhere(u_ex, V)
        Ω = Triangulation(model); dΩ = Measure(Ω, qdeg)
        e = u_ex - uh
        l2 = sqrt(abs(sum(∫(e ⋅ e)dΩ)))
        h1 = sqrt(abs(sum(∫(∇(e) ⊙ ∇(e))dΩ)))
        ha = collect(lazy_map(v -> (6.0*sqrt(2.0)*abs(v))^(1/3), get_cell_measure(Ω)))
        hmean = sum(ha)/length(ha)
        push!(res, (lc=lc, h=hmean, cells=num_cells(model), l2=l2, h1=h1))
        println("  lc=$lc hmean=$(round(hmean,sigdigits=4)) cells=$(num_cells(model)) | interp L2u=$(round(l2,sigdigits=4)) H1u=$(round(h1,sigdigits=4))")
    end
    println("  --- interp rates vs measured h (opt: L2=$(kv+1), H1=$kv) ---")
    for i in 1:length(res)-1
        a,b=res[i],res[i+1]
        println("    h $(round(a.h,sigdigits=3))->$(round(b.h,sigdigits=3)): L2u=$(round(slope(a.l2,b.l2,a.h,b.h),digits=2)) H1u=$(round(slope(a.h1,b.h1,a.h,b.h),digits=2))")
    end
    return res
end

# Same interpolation error, but on a STRUCTURED simplexified Cartesian cube (uniform, high-quality
# tets) — isolates whether the gmsh unstructured mesh QUALITY is degrading the interpolation rate.
function interp_errs_structured(ns; kv=1, qdeg=14)
    alpha_field = PNS.SmoothRadialPorosity(0.5, 1.0, 0.2, 0.4)
    u_ex = PNS.UExFunc{3}(1.0, 0.5, alpha_field, 1.0)
    res = []
    for n in ns
        model = simplexify(CartesianDiscreteModel((0.0,1.0,0.0,1.0,0.0,1.0), (n,n,n)))
        refe = ReferenceFE(lagrangian, VectorValue{3,Float64}, kv)
        V = TestFESpace(model, refe, conformity=:H1)
        uh = interpolate_everywhere(u_ex, V)
        Ω = Triangulation(model); dΩ = Measure(Ω, qdeg)
        e = u_ex - uh
        l2 = sqrt(abs(sum(∫(e ⋅ e)dΩ))); h1 = sqrt(abs(sum(∫(∇(e) ⊙ ∇(e))dΩ)))
        ha = collect(lazy_map(v -> (6.0*sqrt(2.0)*abs(v))^(1/3), get_cell_measure(Ω)))
        hmean = sum(ha)/length(ha)
        push!(res, (h=hmean, cells=num_cells(model), l2=l2, h1=h1))
        println("  n=$n hmean=$(round(hmean,sigdigits=4)) cells=$(num_cells(model)) | interp L2u=$(round(l2,sigdigits=4)) H1u=$(round(h1,sigdigits=4))")
    end
    println("  --- structured interp rates vs measured h (opt: L2=$(kv+1), H1=$kv) ---")
    for i in 1:length(res)-1
        a,b=res[i],res[i+1]
        println("    L2u=$(round(slope(a.l2,b.l2,a.h,b.h),digits=2)) H1u=$(round(slope(a.h1,b.h1,a.h,b.h),digits=2))")
    end
end

# NESTED red-refined family from one base unstructured gmsh mesh (the production strategy):
# nested, exact h-halving, quality-stable. This is the decisive control — if pure interpolation
# on the nested family recovers optimal rates while the independent-gmsh family does not, the
# mesh STRATEGY (not the field) was the culprit.
function interp_errs_nested(domain, base_lc, nlevels; kv=1, qdeg=14)
    alpha_field = PNS.SmoothRadialPorosity(0.5, 1.0, 0.2, 0.4)
    u_ex = PNS.UExFunc{3}(1.0, 0.5, alpha_field, 1.0)
    fam = build_nested_family(nlevels; lc=base_lc, domain=domain)
    res = []
    for (lvl, model) in enumerate(fam)
        refe = ReferenceFE(lagrangian, VectorValue{3,Float64}, kv)
        V = TestFESpace(model, refe, conformity=:H1)
        uh = interpolate_everywhere(u_ex, V)
        Ω = Triangulation(model); dΩ = Measure(Ω, qdeg)
        e = u_ex - uh
        l2 = sqrt(abs(sum(∫(e ⋅ e)dΩ))); h1 = sqrt(abs(sum(∫(∇(e) ⊙ ∇(e))dΩ)))
        ha = collect(lazy_map(v -> (6.0*sqrt(2.0)*abs(v))^(1/3), get_cell_measure(Ω)))
        hmean = sum(ha)/length(ha)
        push!(res, (h=hmean, cells=num_cells(model), l2=l2, h1=h1))
        println("  level=$(lvl-1) hmean=$(round(hmean,sigdigits=4)) cells=$(num_cells(model)) | interp L2u=$(round(l2,sigdigits=4)) H1u=$(round(h1,sigdigits=4))")
    end
    println("  --- nested interp rates vs measured h (opt: L2=$(kv+1), H1=$kv) ---")
    for i in 1:length(res)-1
        a,b=res[i],res[i+1]
        println("    h $(round(a.h,sigdigits=3))->$(round(b.h,sigdigits=3)): L2u=$(round(slope(a.l2,b.l2,a.h,b.h),digits=2)) H1u=$(round(slope(a.h1,b.h1,a.h,b.h),digits=2))")
    end
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 1
    println("### PURE INTERPOLATION test, P$kv, qdeg=14 ###")
    println("--- (A) NESTED red-refined gmsh family, CUBE (0,1)^3  [production strategy] ---")
    interp_errs_nested((0.0,1.0,0.0,1.0,0.0,1.0), 0.2, 3; kv=kv)
    println("--- (B) STRUCTURED simplexified Cartesian, CUBE (0,1)^3  [known-good control] ---")
    interp_errs_structured([5,10,20]; kv=kv)
    println("--- (C) INDEPENDENT gmsh remeshes, CUBE (0,1)^3  [old strategy, expected suboptimal] ---")
    interp_errs((0.0,1.0,0.0,1.0,0.0,1.0), [0.2,0.1,0.05]; kv=kv)
end
