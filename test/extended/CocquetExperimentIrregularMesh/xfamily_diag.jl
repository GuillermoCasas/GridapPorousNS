# Decisive cross-mesh-FAMILY diagnostic: is the structured self-reference's small error a
# correlated-error-cancellation artifact?
#
# Build the Cocquet Galerkin P2/P1 solution at a coarse N on BOTH a structured and an unstructured
# mesh, and a reference at N_ref on BOTH, then measure the velocity-L2 error of each coarse solution
# against BOTH references (same family vs cross family). If same-family (structured-vs-structured) is
# much smaller than cross-family, the structured small error is a self-correlation artifact.
#
# Run: cd test/extended/CocquetExperimentIrregularMesh && julia --project=../../.. xfamily_diag.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "run_convergence.jl"))   # provides build_solver, execute_solver_galerkin, alpha_func
using PorousNSSolver, Gridap, JSON3

Ncoarse = 10
Nref    = 80
Re, c_in = 500.0, 0.5

cfg = JSON3.read(read(joinpath(@__DIR__, "data", "paper_comparison_irregular.json"), String), Dict{String,Any})
for k in ("Re","c_in","mesh_generator","comparison_runs","element_pairs","k_convergence_list","outlet_truncation_delta")
    haskey(cfg,k) && delete!(cfg,k)
end
cfg["numerical_method"]["element_spaces"]["k_velocity"] = 2
cfg["numerical_method"]["element_spaces"]["k_pressure"] = 1
cfg["numerical_method"]["stabilization"]["method"] = "ASGS"

function solve_galerkin(N, gen)
    mod, X, Y, dΩ, h, ah, ru, rp, c = build_solver(N, cfg, Re, c_in; porosity_order=1, mesh_generator=gen)
    xh, _, _ = execute_solver_galerkin(mod, X, Y, dΩ, h, ah, ru, rp, c)
    u, p = xh
    return (mod=mod, u=u, p=p, ru=ru, dΩ=dΩ)
end

ks = Gridap.CellData.KDTreeSearch(num_nearest_vertices=10)
function velL2(coarse, ref)
    iu = Gridap.FESpaces.Interpolable(ref.u; searchmethod=ks)
    Vf = TestFESpace(coarse.mod, coarse.ru, conformity=:H1)
    r = PorousNSSolver.compute_reference_errors(coarse.u, ref.u, iu, Vf, coarse.dΩ, ref.dΩ; search_method=ks)
    return r[3]  # l2_cons
end

println("Solving coarse N=$Ncoarse and reference N=$Nref on BOTH families (Galerkin P2/P1)...")
cS = solve_galerkin(Ncoarse, "STRUCTURED")
cU = solve_galerkin(Ncoarse, "UNSTRUCTURED")
rS = solve_galerkin(Nref,    "STRUCTURED")
rU = solve_galerkin(Nref,    "UNSTRUCTURED")

println("\n=== velocity-L2 error of coarse N=$Ncoarse vs reference N=$Nref ===")
println("structured coarse  vs  structured ref  (same family) : ", velL2(cS, rS))
println("structured coarse  vs  unstructured ref (cross)      : ", velL2(cS, rU))
println("unstructured coarse vs unstructured ref (same family): ", velL2(cU, rU))
println("unstructured coarse vs structured ref  (cross)       : ", velL2(cU, rS))
