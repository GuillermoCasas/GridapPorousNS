# test/extended/ManufacturedSolutions/run_corner_osgs.jl
# ==============================================================================================
# [diagnostic-tool] OSGS counterpart of run_corner_article.jl for the stiff MMS fold corner
# (Re=1e6, α₀=0.05, k=1 TRI; article.tex tab:Linear2DL2/H1, OSGS columns). For each Da it runs the
# OSGS coupled solve (osgs_corner_lib.jl, = production solve_osgs_stage!) at the base mesh from the
# exact-solution guess (the fold has cleared by N≈512), then warm-starts the finer mesh from the
# interpolated base OSGS root, and reports the four normalized norms (vel/prs × L²/H¹) + the
# two-finest-mesh slopes. Production-like tolerances (stops at the dynamic noise floor where the
# error is already converged) — NOT a 1e-8 grind. Writes results/debug_results/corner_tri_k1_a005_osgs.json.
#
# RUN:  cd test/extended/ManufacturedSolutions && julia --project=../../.. run_corner_osgs.jl
# ==============================================================================================
include(joinpath(@__DIR__, "run_continuation.jl"))   # build_cell, calc_normalized_errors, Interpolable
include(joinpath(@__DIR__, "osgs_corner_lib.jl"))    # osgs_solve
using JSON3
using Gridap.CellData: Interpolable
using Gridap.FESpaces: get_free_dof_values

_rate(a, b, ha, hb) = log(a / b) / log(ha / hb)

# Recompute the four normalized error norms at a converged state (cell + free-dof vector).
function _norms_at(cell, dofs)
    uh, ph = FEFunction(cell.setup.X, dofs)
    l2u, l2p, h1u, h1p = calc_normalized_errors(uh, ph, cell.u_final, cell.p_final,
                                                cell.U_c, cell.P_c, cell.L, cell.setup.dΩ)
    return (l2u=l2u, l2p=l2p, h1u=h1u, h1p=h1p)
end

function osgs_corner(Re, Da, αt; kv=1, etype="TRI", base_N=512, fine_N=768, asgs_maxit=12, osgs_maxit=10)
    println("\n", "#"^86)
    println(@sprintf("# OSGS CORNER  Re=%.0e  Da=%.0e  α=%.3f  (%s, k=%d)  N=%d→%d", Re, Da, αt, etype, kv, base_N, fine_N))
    println("#"^86); flush(stdout)

    # --- base mesh: ASGS root (direct from exact) → OSGS warm-started from it ---
    println(@sprintf("  [base] ASGS direct N=%d ...", base_N)); flush(stdout)
    t0 = time()
    ab = solve_cell(Re, Da, αt, base_N, kv, etype, nothing; max_iters=asgs_maxit)
    na_b = _norms_at(ab.cell, ab.dofs)
    println(@sprintf("  [base] ASGS N=%d reached=%s ‖R‖=%.2e L2u=%.4e  (%.0fs) → OSGS warm-start...", base_N, ab.reached, ab.rnorm, na_b.l2u, time()-t0)); flush(stdout)
    t0b = time()
    ob = osgs_solve(ab.cell, ab.dofs; max_iters=osgs_maxit, verbose=true)
    println(@sprintf("  [base] OSGS N=%d reached=%s ‖R‖=%.2e iters=%d  L2u=%.4e L2p=%.4e H1u=%.4e H1p=%.4e  (%.0fs)",
        base_N, ob.reached, ob.rnorm, ob.iters, ob.l2u, ob.l2p, ob.h1u, ob.h1p, time()-t0b)); flush(stdout)

    # --- fine mesh: ASGS mesh-step up → OSGS warm-started from the fine ASGS root ---
    println(@sprintf("  [fine] ASGS mesh-step %d→%d ...", base_N, fine_N)); flush(stdout)
    t1 = time()
    af = mesh_step(ab.cell, ab.dofs, Re, Da, αt, fine_N, kv, etype; max_iters=asgs_maxit)
    if !af.reached
        println(@sprintf("  [fine] ASGS mesh-step N=%d NOT converged (‖R‖=%.2e) — base-only.", fine_N, af.rnorm)); flush(stdout)
        return Dict{String,Any}("Re"=>Re,"Da"=>Da,"alpha_0"=>αt,"kv"=>kv,"element_type"=>etype,"method"=>"OSGS","status"=>"base_only","Ns"=>[base_N])
    end
    na_f = _norms_at(af.cell, af.dofs)
    println(@sprintf("  [fine] ASGS N=%d reached=%s ‖R‖=%.2e L2u=%.4e  (%.0fs) → OSGS warm-start...", fine_N, af.reached, af.rnorm, na_f.l2u, time()-t1)); flush(stdout)
    t1b = time()
    of = osgs_solve(af.cell, af.dofs; max_iters=osgs_maxit, verbose=true)
    println(@sprintf("  [fine] OSGS N=%d reached=%s ‖R‖=%.2e iters=%d  L2u=%.4e L2p=%.4e H1u=%.4e H1p=%.4e  (%.0fs)",
        fine_N, of.reached, of.rnorm, of.iters, of.l2u, of.l2p, of.h1u, of.h1p, time()-t1b)); flush(stdout)

    hb = 1.0/base_N; hf = 1.0/fine_N
    out = Dict{String,Any}("Re"=>Re, "Da"=>Da, "alpha_0"=>αt, "kv"=>kv, "element_type"=>etype, "method"=>"OSGS",
        "status"=>"rescued", "Ns"=>[base_N, fine_N], "hs"=>[hb, hf],
        "residuals"=>[ob.rnorm, of.rnorm], "iters"=>[ob.iters, of.iters],
        "l2u"=>[ob.l2u, of.l2u], "l2p"=>[ob.l2p, of.l2p], "h1u"=>[ob.h1u, of.h1u], "h1p"=>[ob.h1p, of.h1p],
        "slope_l2u"=>_rate(ob.l2u, of.l2u, hb, hf), "slope_l2p"=>_rate(ob.l2p, of.l2p, hb, hf),
        "slope_h1u"=>_rate(ob.h1u, of.h1u, hb, hf), "slope_h1p"=>_rate(ob.h1p, of.h1p, hb, hf),
        "FME_l2u"=>of.l2u, "FME_l2p"=>of.l2p, "FME_h1u"=>of.h1u, "FME_h1p"=>of.h1p,
        # ASGS cross-check (recomputed here so OSGS vs ASGS sit in one record)
        "asgs_l2u"=>[na_b.l2u, na_f.l2u], "asgs_l2p"=>[na_b.l2p, na_f.l2p],
        "asgs_h1u"=>[na_b.h1u, na_f.h1u], "asgs_h1p"=>[na_b.h1p, na_f.h1p])
    println(@sprintf("\n   >>> OSGS  N=[%d,%d]  (two finest %d→%d)", base_N, fine_N, base_N, fine_N))
    println(@sprintf("       velocity  L2: slope=%.2f FME=%.3e | H1: slope=%.2f FME=%.3e",
        out["slope_l2u"], out["FME_l2u"], out["slope_h1u"], out["FME_h1u"]))
    println(@sprintf("       pressure  L2: slope=%.2f FME=%.3e | H1: slope=%.2f FME=%.3e",
        out["slope_l2p"], out["FME_l2p"], out["slope_h1p"], out["FME_h1p"]))
    println(@sprintf("       (ASGS xcheck FME: L2u=%.3e H1u=%.3e L2p=%.3e)", na_f.l2u, na_f.h1u, na_f.l2p)); flush(stdout)
    return out
end

# CLI:  run_corner_osgs.jl [etype kv base_N fine_N outname das]
#   etype  "TRI"(default)|"QUAD"; kv 1(default)|2; base_N/fine_N the two meshes; outname the
#   debug_results JSON; das comma list of Da. Defaults reproduce the TRI/k1 Da=1e6 single-cell run.
function main()
    Re = 1e6; αt = 0.05
    etype  = length(ARGS) >= 1 ? ARGS[1] : "TRI"
    kv     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    base_N = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 512
    fine_N = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 768
    outname = length(ARGS) >= 5 ? ARGS[5] : "corner_tri_k1_a005_osgs_da1e6.json"
    Das = length(ARGS) >= 6 ? parse.(Float64, split(ARGS[6], ",")) : [1e6]
    outpath = joinpath(@__DIR__, "results", "debug_results", outname)
    mkpath(dirname(outpath))
    println(@sprintf("[run_corner_osgs] etype=%s kv=%d N=%d→%d -> %s", etype, kv, base_N, fine_N, outname)); flush(stdout)
    results = Any[]
    for Da in Das
        r = osgs_corner(Re, Da, αt; kv=kv, etype=etype, base_N=base_N, fine_N=fine_N)
        push!(results, r)
        open(outpath, "w") do io
            JSON3.write(io, [Dict(k => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k,v) in d) for d in results])
        end
        println("   [wrote incremental OSGS results -> $outpath]"); flush(stdout)
    end
    println("\n", "="^86)
    println("DONE. All OSGS corner results -> $outpath"); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
