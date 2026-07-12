# test/extended/ManufacturedSolutions3D/c1_distribution_probe.jl
# =============================================================================================
# DIAGNOSTIC (writes to results/debug_results/). Measures the per-element elementwise coercivity
# floor c₁*_code(K) = 4 h_code(K)² λ_max(K) (theory/numerical_constants/c1_dimension_note.tex,
# transcribed in element_c1.jl) over the nested_red (irregular) and structured Kuhn tet families,
# for P2. Answers: does the IRREGULAR mesh's quality tail demand a LARGER c₁ than the empirical
# c1_mult=4 (calibrated to Kuhn) — i.e. is the k=2 nested_red sub-optimality a coercivity deficit?
# =============================================================================================
using Gridap, LinearAlgebra, Printf, JSON3
using PorousNSSolver
const PNS = PorousNSSolver
include(joinpath(@__DIR__, "smoke3d.jl"))       # mesh builders + §5.2 consts + element_c1.jl (guarded include)

# per-cell h in a given convention (mirrors solve_one lines 246-253 EXACTLY)
function _h_array(Ω, h_conv::String)
    if h_conv == "diameter" || h_conv == "shortest_edge"
        cc = get_cell_coordinates(Ω)
        _redu = h_conv == "shortest_edge" ? minimum : maximum
        return [_redu(sqrt((v[i]-v[j])⋅(v[i]-v[j])) for i in 1:length(v) for j in (i+1):length(v)) for v in cc]
    else
        _h_of_v = h_conv == "d_fact" ? (v -> (6.0*abs(v))^(1.0/3.0)) : (v -> (6.0*sqrt(2.0)*abs(v))^(1.0/3.0))
        return collect(lazy_map(_h_of_v, get_cell_measure(Ω)))
    end
end

function probe_family(name, models; k::Int=2, h_convs=["regular_tet","diameter","shortest_edge"])
    rows = Any[]
    println("\n################  $name  (P$k)  ################")
    for (lvl, model) in enumerate(models)
        Ω = Triangulation(model)
        cc = collect(get_cell_coordinates(Ω))
        ncell = length(cc)
        println("\n--- level $(lvl-1): $ncell tets ---")
        @printf("%-14s %10s %10s %10s %10s %10s | %10s %10s\n",
                "h_conv","h_mean","c1*_p50","c1*_p90","c1*_p99","c1*_max","mult_p99","mult_max")
        for hc in h_convs
            ha = _h_array(Ω, hc)
            hmean = sum(ha)/length(ha)
            _, percell, st = calibrate_c1_over_cells(cc, ha, k; percentile=100.0, safety=1.0)
            @printf("%-14s %10.4g %10.4g %10.4g %10.4g %10.4g | %10.3g %10.3g\n",
                    hc, hmean, st.c1_p50, st.c1_p90, st.c1_p99, st.c1_max,
                    st.c1_p99/st.c1_paper, st.c1_max/st.c1_paper)
            push!(rows, Dict("family"=>name, "level"=>lvl-1, "ncell"=>ncell, "nshapes"=>st.nshapes,
                             "h_conv"=>hc, "h_mean"=>hmean, "k"=>k, "c1_paper"=>st.c1_paper,
                             "c1_min"=>st.c1_min, "c1_p50"=>st.c1_p50, "c1_p90"=>st.c1_p90,
                             "c1_p99"=>st.c1_p99, "c1_max"=>st.c1_max,
                             "mult_p50"=>st.c1_p50/st.c1_paper, "mult_p90"=>st.c1_p90/st.c1_paper,
                             "mult_p99"=>st.c1_p99/st.c1_paper, "mult_max"=>st.c1_max/st.c1_paper,
                             "lam_min"=>st.lam_min, "lam_p50"=>st.lam_p50, "lam_max"=>st.lam_max))
        end
    end
    return rows
end

function main()
    kv = 2
    nested = build_nested_family(2; lc=0.2, domain=DOMAIN)             # L0,L1,L2 (the sweep's meshes)
    kuhn   = [structured_kuhn_model(p; domain=DOMAIN) for p in [(4,4,1),(8,8,2),(16,16,5)]]
    rows = Any[]
    append!(rows, probe_family("nested_red (irregular)", nested; k=kv))
    append!(rows, probe_family("structured Kuhn",        kuhn;   k=kv))
    outdir = joinpath(@__DIR__, "results", "debug_results"); mkpath(outdir)
    outpath = joinpath(outdir, "c1_distribution_probe.json")
    open(outpath, "w") do io; JSON3.pretty(io, Dict("purpose"=>"per-element c1*_code distribution, P2 nested_red vs Kuhn",
                                                     "rows"=>rows)); end
    @printf("\n[wrote] %s (%d rows)\n", outpath, length(rows))
    # headline: the empirical c1_mult=4 vs the theory-demanded worst/99th for nested_red regular_tet
    println("\n=== HEADLINE (nested_red, regular_tet convention — the sweep's actual units) ===")
    for r in rows
        r["family"]=="nested_red (irregular)" && r["h_conv"]=="regular_tet" || continue
        @printf("  L%d: mult_p90=%.2f mult_p99=%.2f mult_max=%.2f  (empirical baseline = 4.0)\n",
                r["level"], r["mult_p90"], r["mult_p99"], r["mult_max"])
    end
end

main()
