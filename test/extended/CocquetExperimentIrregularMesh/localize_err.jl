# Spatially localize the cross-mesh velocity-L² error for Cocquet Galerkin P2/P1 at the same
# (Re, c_in, N_coarse, N_ref) on TWO mesh families:
#   1. STRUCTURED Cartesian-simplexified
#   2. UNSTRUCTURED gmsh Delaunay (uniform 1/N)
#
# For each, integrate ‖e_u‖² (nested error on the coarse mesh, robust direction) over four masks:
#   A. outlet-corner disks: x≥1.9 ∧ (y<0.1 ∨ y>0.9)
#   B. inlet-corner disks:  x<0.1 ∧ (y<0.1 ∨ y>0.9)
#   C. wall strips minus corners
#   D. bulk channel
# Reports the fraction of total ‖e‖² in each region. Tells us where the 20× unstructured magnitude lives.
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

ks = Gridap.CellData.KDTreeSearch(num_nearest_vertices=10)
function solve_g(N, gen, wd=:uniform, alg=5)
    m, X, Y, dΩ, h, ah, ru, rp, c = build_solver(N, cfg, Re, c_in;
        porosity_order=1, mesh_generator=gen, wall_divisions=wd, mesh_algorithm=alg)
    xh, _, _ = execute_solver_galerkin(m, X, Y, dΩ, h, ah, ru, rp, c)
    u, p = xh
    return (mod=m, u=u, p=p, ru=ru, dΩ=dΩ, X=X, Y=Y)
end

# Mask helpers
in_outlet_corner(x) = (x[1] >= 1.9) && (x[2] < 0.1 || x[2] > 0.9)
in_inlet_corner(x)  = (x[1] <  0.1) && (x[2] < 0.1 || x[2] > 0.9)
in_wall_strip(x)    = (x[2] < 0.1 || x[2] > 0.9) && !in_outlet_corner(x) && !in_inlet_corner(x)
in_bulk(x)          = !( in_outlet_corner(x) || in_inlet_corner(x) || in_wall_strip(x) )

function localize(coarse, ref)
    # Robust direction: interpolate ref onto coarse free space, integrate on coarse Ω.
    iu = Gridap.FESpaces.Interpolable(ref.u; searchmethod=ks)
    Vf = TestFESpace(coarse.mod, coarse.ru, conformity=:H1)
    uref_proj = interpolate(iu, Vf)
    e = uref_proj - coarse.u
    e2(x) = e(x) ⋅ e(x)
    function masked(predicate)
        χ = CellField(x -> predicate(x) ? 1.0 : 0.0, Gridap.FESpaces.get_triangulation(coarse.u))
        return sum(∫(χ * (e ⊙ e)) * coarse.dΩ)
    end
    tot = masked(x->true)
    return (total=sqrt(tot),
            outlet_corner=masked(in_outlet_corner)/tot,
            inlet_corner =masked(in_inlet_corner )/tot,
            wall_strip   =masked(in_wall_strip   )/tot,
            bulk         =masked(in_bulk         )/tot)
end

println("Solving Galerkin P2/P1 on STRUCTURED + UNSTRUCTURED at (N=$Ncoarse, ref=$Nref)...")
S_c = solve_g(Ncoarse, "STRUCTURED")
S_r = solve_g(Nref,    "STRUCTURED")
U_c = solve_g(Ncoarse, "UNSTRUCTURED", :uniform, 5)
U_r = solve_g(Nref,    "UNSTRUCTURED", :uniform, 5)

ls = localize(S_c, S_r); lu = localize(U_c, U_r)
println("\n=== ‖e_u‖_L2 totals and fraction-of-‖e‖² per region ===")
println("                       outlet_corner   inlet_corner   wall_strip   bulk     ||e||")
for (lbl, l) in (("STRUCTURED ", ls), ("UNSTRUCTURED", lu))
    println("$(lbl) :  ",
            round(l.outlet_corner*100, sigdigits=3), "%        ",
            round(l.inlet_corner *100, sigdigits=3), "%         ",
            round(l.wall_strip   *100, sigdigits=3), "%       ",
            round(l.bulk         *100, sigdigits=3), "%     ",
            "  ", round(l.total, sigdigits=3))
end
