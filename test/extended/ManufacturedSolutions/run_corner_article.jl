# test/extended/ManufacturedSolutions/run_corner_article.jl
# ==============================================================================================
# [diagnostic-tool] Article-table reproduction for the stiff MMS fold corner (Re=1e6, α₀=0.05),
# k=1 P1/P1 — the three cells deferred from the standard sweep via `skip_cells` and handled by
# the documented α + mesh continuation (docs/mms/fold-recovery.md). For each Da ∈ {1e-6, 1, 1e6}
# it runs α-continuation to the target corner at a base mesh that clears the fold, interpolates
# the converged state up the fine mesh ladder, and reports ALL FOUR normalized error norms
# (velocity & pressure, L² & H¹) + their two-finest-mesh slopes — the exact quantities tabulated
# in article.tex tab:Linear2DL2 / tab:Linear2DH1. Reuses the validated continuation primitives
# (`alpha_continuation`, `mesh_step`, `build_cell`) unchanged; norms are recomputed from each
# converged (cell, free-dofs) so all four are captured without threading them through the solver.
#
# RUN:  cd test/extended/ManufacturedSolutions && julia --project=../../.. run_corner_article.jl
# OUT:  data/corner/corner_tri_k1_a005.json   (tracked; incremental — survives a partial run)
# ==============================================================================================

include(joinpath(@__DIR__, "run_continuation.jl"))   # brings alpha_continuation, mesh_step, build_cell
using JSON3

# Recompute the four normalized error norms at a converged state (cell + free-dof vector).
function norms_at(cell, dofs)
    uh, ph = FEFunction(cell.setup.X, dofs)
    l2u, l2p, h1u, h1p = calc_normalized_errors(uh, ph, cell.u_final, cell.p_final,
                                                cell.U_c, cell.P_c, cell.L, cell.setup.dΩ)
    return (l2u=l2u, l2p=l2p, h1u=h1u, h1p=h1p)
end

_rate(a, b, ha, hb) = log(a / b) / log(ha / hb)

# One corner cell. The fold has cleared by N≈512, so a DIRECT solve from the exact-solution guess
# reaches the true root (≈3 Newton iters) — no α-continuation needed. We solve directly at the
# first base mesh that converges, then mesh-continue (interpolate + ~2 iters) up the fine ladder.
# Returns a Dict with the per-mesh four-norm curve + two-finest-mesh slopes.
function run_corner(Re, Da, αt; kv=1, etype="TRI",
                    base_candidates=[512, 640, 768], fine_ladder=[768],
                    base_maxit=12, mesh_maxit=12)
    println("\n", "#"^86)
    println(@sprintf("# CORNER  Re=%.0e  Da=%.0e  α=%.3f  (%s, k=%d)", Re, Da, αt, etype, kv))
    println("#"^86); flush(stdout)

    base = nothing; base_N = 0
    for cand in base_candidates
        println(@sprintf("  [base] direct solve N=%d (Re=%.0e Da=%.0e α=%.3f)...", cand, Re, Da, αt)); flush(stdout)
        t0 = time()
        r = solve_cell(Re, Da, αt, cand, kv, etype, nothing; max_iters=base_maxit)
        println(@sprintf("  [base] N=%d  reached=%s  ‖R‖=%.2e  (%.0fs)", cand, r.reached, r.rnorm, time()-t0)); flush(stdout)
        if r.reached
            base = r; base_N = cand
            # any fine-ladder mesh at or below the chosen base is redundant
            fine_ladder = filter(N -> N > base_N, fine_ladder)
            break
        end
        println(@sprintf("  [base N=%d did not converge — escalating]", cand)); flush(stdout)
    end
    if base === nothing
        return Dict("Re"=>Re, "Da"=>Da, "alpha_0"=>αt, "kv"=>kv, "element_type"=>etype,
                    "status"=>"base_failed", "Ns"=>Int[])
    end

    Ns = [base_N]; hs = [base.h]
    nb = norms_at(base.cell, base.dofs)
    l2us=[nb.l2u]; l2ps=[nb.l2p]; h1us=[nb.h1u]; h1ps=[nb.h1p]; resids=[base.rnorm]
    iters=[base.niters]   # Newton iters of the direct exact-guess solve (Picard not used at the corner)
    prev = base
    for Nf in fine_ladder
        Nf > base_N || continue
        println(@sprintf("  [mesh] interpolate %d→%d + solve...", prev.cell.n, Nf)); flush(stdout)
        t0 = time()
        m = mesh_step(prev.cell, prev.dofs, Re, Da, αt, Nf, kv, etype; max_iters=mesh_maxit)
        if !m.reached
            println(@sprintf("   mesh-step N=%d NOT converged (‖R‖=%.2e, %.0fs); stopping ladder.", Nf, m.rnorm, time()-t0)); flush(stdout)
            break
        end
        nm = norms_at(m.cell, m.dofs)
        push!(Ns, Nf); push!(hs, m.h); push!(resids, m.rnorm); push!(iters, m.niters)
        push!(l2us, nm.l2u); push!(l2ps, nm.l2p); push!(h1us, nm.h1u); push!(h1ps, nm.h1p)
        println(@sprintf("   mesh-step N=%d: ‖R‖=%.2e  L2u=%.4e L2p=%.4e H1u=%.4e H1p=%.4e  (%.0fs)",
            Nf, m.rnorm, nm.l2u, nm.l2p, nm.h1u, nm.h1p, time()-t0)); flush(stdout)
        prev = m
    end

    out = Dict{String,Any}("Re"=>Re, "Da"=>Da, "alpha_0"=>αt, "kv"=>kv, "element_type"=>etype,
        "status"=> (length(Ns) >= 2 ? "rescued" : "base_only"),
        "Ns"=>Ns, "hs"=>hs, "residuals"=>resids, "iters"=>iters,
        "l2u"=>l2us, "l2p"=>l2ps, "h1u"=>h1us, "h1p"=>h1ps)
    if length(Ns) >= 2
        i, j = length(Ns)-1, length(Ns)   # two finest meshes
        out["slope_l2u"] = _rate(l2us[i], l2us[j], hs[i], hs[j])
        out["slope_l2p"] = _rate(l2ps[i], l2ps[j], hs[i], hs[j])
        out["slope_h1u"] = _rate(h1us[i], h1us[j], hs[i], hs[j])
        out["slope_h1p"] = _rate(h1ps[i], h1ps[j], hs[i], hs[j])
        out["FME_l2u"] = l2us[j]; out["FME_l2p"] = l2ps[j]
        out["FME_h1u"] = h1us[j]; out["FME_h1p"] = h1ps[j]
        println(@sprintf("\n   >>> [%s] N=%s  (two finest %d→%d)", out["status"], string(Ns), Ns[i], Ns[j]))
        println(@sprintf("       velocity  L2: slope=%.2f FME=%.3e | H1: slope=%.2f FME=%.3e",
            out["slope_l2u"], out["FME_l2u"], out["slope_h1u"], out["FME_h1u"]))
        println(@sprintf("       pressure  L2: slope=%.2f FME=%.3e | H1: slope=%.2f FME=%.3e",
            out["slope_l2p"], out["FME_l2p"], out["slope_h1p"], out["FME_h1p"]))
    end
    return out
end

# CLI:  run_corner_article.jl [etype kv base_candidates fine_ladder outname das]
#   etype           "TRI" (default) | "QUAD"
#   kv              1 (default) | 2
#   base_candidates comma list, e.g. "512,640,768" (default) — first that converges is the base
#   fine_ladder     comma list, e.g. "768" (default) — meshes to mesh-step up to (skipped if ≤ base)
#   outname         debug_results JSON filename (default corner_tri_k1_a005.json)
#   das             comma list of Da (default "1e-6,1,1e6")
function main()
    Re = 1e6; αt = 0.05
    etype = length(ARGS) >= 1 ? ARGS[1] : "TRI"
    kv    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    base_candidates = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [512, 640, 768]
    fine_ladder     = length(ARGS) >= 4 ? parse.(Int, split(ARGS[4], ",")) : [768]
    outname = length(ARGS) >= 5 ? ARGS[5] : "corner_tri_k1_a005.json"
    Das = length(ARGS) >= 6 ? parse.(Float64, split(ARGS[6], ",")) : [1e-6, 1.0, 1e6]
    outpath = joinpath(@__DIR__, "data", "corner", outname)   # TRACKED corner provenance (see data/corner/README.md)
    mkpath(dirname(outpath))
    println(@sprintf("[run_corner] etype=%s kv=%d base=%s fine=%s -> %s", etype, kv, string(base_candidates), string(fine_ladder), outname)); flush(stdout)
    results = Any[]
    for Da in Das
        r = run_corner(Re, Da, αt; kv=kv, etype=etype, base_candidates=base_candidates, fine_ladder=fine_ladder)
        push!(results, r)
        # incremental write so a partial run is not lost
        open(outpath, "w") do io
            JSON3.write(io, [Dict(k => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k,v) in d) for d in results])
        end
        println("   [wrote incremental results -> $outpath]"); flush(stdout)
    end
    println("\n", "="^86)
    println("DONE. All corner results -> $outpath"); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
