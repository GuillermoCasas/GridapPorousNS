# Diagnostic: does the literal FreeFem `buildmesh(a(N)+b(N)+c(N)+d(N))` boundary recipe
# (N divisions on EACH of the four boundary parts, irrespective of length → anisotropic mesh,
# walls 2× coarser per edge than inlets) close the magnitude gap to the paper?
#
# Runs Cocquet Galerkin P2/P1 on three mesh recipes at the same Ns, computes vel-L² error of
# coarse vs ref:
#   1. STRUCTURED Cartesian-simplexified                                    (the "structured" benchmark)
#   2. UNSTRUCTURED uniform 1/N edges (walls 2N divisions)                  (our current irregular mesh)
#   3. UNSTRUCTURED literal FreeFem (N divisions on each border, anisotropic) (the paper's literal recipe)
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "run_convergence.jl"))
using PorousNSSolver, Gridap, JSON3

Re, c_in = 500.0, 0.5
Ncoarse, Nref = 10, 80

cfg = JSON3.read(read(joinpath(@__DIR__, "data", "paper_comparison_irregular.json"), String), Dict{String,Any})
for k in ("Re","c_in","mesh_generator","comparison_runs","element_pairs","k_convergence_list","outlet_truncation_delta")
    haskey(cfg,k) && delete!(cfg,k)
end
cfg["numerical_method"]["element_spaces"]["k_velocity"] = 2
cfg["numerical_method"]["element_spaces"]["k_pressure"] = 1
cfg["numerical_method"]["stabilization"]["method"] = "ASGS"

function solve_galerkin(N, gen, wd=:uniform, alg=5)
    m, X, Y, dΩ, h, ah, ru, rp, c = build_solver(N, cfg, Re, c_in;
        porosity_order=1, mesh_generator=gen, wall_divisions=wd, mesh_algorithm=alg)
    xh, _, _ = execute_solver_galerkin(m, X, Y, dΩ, h, ah, ru, rp, c)
    u, p = xh
    return (mod=m, u=u, p=p, ru=ru, dΩ=dΩ)
end

ks = Gridap.CellData.KDTreeSearch(num_nearest_vertices=10)
function velL2(coarse, ref)
    iu = Gridap.FESpaces.Interpolable(ref.u; searchmethod=ks)
    Vf = TestFESpace(coarse.mod, coarse.ru, conformity=:H1)
    r = PorousNSSolver.compute_reference_errors(coarse.u, ref.u, iu, Vf,
        coarse.dΩ, ref.dΩ; search_method=ks)
    return r[3]
end

println("Solving coarse N=$Ncoarse and ref N=$Nref on several mesh recipes (Galerkin P2/P1)...")
S_c = solve_galerkin(Ncoarse, "STRUCTURED")
S_r = solve_galerkin(Nref,    "STRUCTURED")
Uu_c = solve_galerkin(Ncoarse, "UNSTRUCTURED", :uniform, 5)   # Delaunay (current)
Uu_r = solve_galerkin(Nref,    "UNSTRUCTURED", :uniform, 5)
Uf_c = solve_galerkin(Ncoarse, "UNSTRUCTURED", :freefem, 5)   # literal FreeFem boundary recipe, gmsh Delaunay
Uf_r = solve_galerkin(Nref,    "UNSTRUCTURED", :freefem, 5)
B_c  = solve_galerkin(Ncoarse, "UNSTRUCTURED", :uniform, 7)   # BAMG (FreeFem's actual mesher), uniform boundary
B_r  = solve_galerkin(Nref,    "UNSTRUCTURED", :uniform, 7)
Bf_c = solve_galerkin(Ncoarse, "UNSTRUCTURED", :freefem, 7)   # BAMG + literal FreeFem recipe (= FreeFem buildmesh)
Bf_r = solve_galerkin(Nref,    "UNSTRUCTURED", :freefem, 7)

println("\n=== velocity-L2 error of coarse N=$Ncoarse vs reference N=$Nref ===")
println("structured (Cartesian)                        : ", velL2(S_c,  S_r))
println("Delaunay (alg 5), uniform 1/N walls 2N        : ", velL2(Uu_c, Uu_r))
println("Delaunay (alg 5), FreeFem literal N/side      : ", velL2(Uf_c, Uf_r))
println("BAMG     (alg 7), uniform 1/N walls 2N        : ", velL2(B_c,  B_r))
println("BAMG     (alg 7), FreeFem literal N/side  *** : ", velL2(Bf_c, Bf_r))

println("\n=== cell counts ===")
for (lbl,m) in (("struct",S_c),("Delaunay-uniform",Uu_c),("Delaunay-freefem",Uf_c),("BAMG-uniform",B_c),("BAMG-freefem",Bf_c))
    println("N=$Ncoarse $(lbl): cells = ", num_cells(m.mod))
end
for (lbl,m) in (("struct",S_r),("Delaunay-uniform",Uu_r),("Delaunay-freefem",Uf_r),("BAMG-uniform",B_r),("BAMG-freefem",Bf_r))
    println("N=$Nref $(lbl): cells = ", num_cells(m.mod))
end
