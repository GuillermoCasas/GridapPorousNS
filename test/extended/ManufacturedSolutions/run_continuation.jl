# test/extended/ManufacturedSolutions/run_continuation.jl
# ==============================================================================================
# Harness-level natural parameter-continuation driver for the stiff MMS corner.
#
# WHY: at the high-Re / low-porosity corner (e.g. Re=1e6, Da=1, α₀=0.05) the discrete VMS
# solution branch FOLDS — there is no root reachable from the exact-solution initial guess at
# coarse mesh (diagnosed in probe_stiff_diagnose.jl; see docs/mms/convergence-2d.md). But the
# branch is reachable by α-continuation from α=1 (smooth, easy), and the fold recedes with mesh
# refinement, so at fine enough N the branch reaches the target α and we recover the true FE
# solution. This driver automates that: warm-started α-continuation per mesh, then MMS
# error/rate reporting at the target corner.
#
# DESIGN: reuses the package's `solve_system` (the production Newton→Picard→retry cascade) and
# the diagnostic's `build_cell` (which mirrors run_test.jl's FE setup) via `include`. It makes
# ZERO changes to the paper-faithful core (src/solvers, src/formulations, src/stabilization,
# src/config, schema). Continuation knobs are read from a harness-level JSON (data/<cfg>.json),
# exactly as run_test.jl reads epsilon_pert / max_n_pert / mms_* — no magic numbers, no
# silent defaults beyond what is documented here.
#
# REGIMES (config "regime" field; defaults from key-presence for backward compatibility):
#   "alpha"       — per-mesh α-continuation: ramp α from `continuation.start` (default 1.0) to
#                   `target.alpha_0` at each mesh in `mesh.convergence_partitions`. Default when no
#                   "mesh_continuation" block is present.
#   "mesh_ladder" — α-continuation ONCE at `mesh_continuation.base_N`, then interpolate the converged
#                   state up `mesh_continuation.fine_Ns` (one solve per finer mesh). Yields the
#                   defensible L²/H¹-vs-h rate at the corner. Default when a "mesh_continuation" block
#                   is present.
#   EXTENSION POINT: a "da" regime (ramp Da at fixed α) is not wired here, but the underlying
#                   primitive already exists as `probe_continuation` in probe_stiff_diagnose.jl.
# BATCH:  `run_continuation.jl phase2 [flagged.json] [out.json] [defaults.json]` rescues every cell in
#         flagged_cells.json via mesh-continuation, routed by detected category.
#
# RUN:  cd test/extended/ManufacturedSolutions && julia --project=../../.. run_continuation.jl [config.json]
#       (the regime is auto-selected from the config; see REGIMES above.)
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using JSON3
using Printf
using LinearAlgebra
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.CellData: Interpolable   # cross-mesh interpolation (not exported by top-level Gridap)
using PorousNSSolver

# Reuse build_cell (mirrors run_test.jl FE setup) + calc_normalized_errors. The included file's
# `main()` is guarded by `abspath(PROGRAM_FILE) == @__FILE__`, so nothing runs on include.
include(joinpath(@__DIR__, "probe_stiff_diagnose.jl"))

# Solve one (Re, Da, α, N) cell to a TRUE root, warm-started from a free-DOF vector `x0_dofs`
# (or the exact-solution interpolant when `nothing`). A continuation driver requires each step to
# reach a genuine root, so we use the tight-tolerance, disabled-false-success solve from the
# diagnostic (`probe_a2_heavy_solve`, ftol=1e-8, noise_floor=1e-12) — NOT the production
# `solve_system`, whose loose dynamic noise floor can declare a false root mid-branch (the very
# defect this corner exposed). Exact-Newton first; on stall, fall back to Picard from the same
# warm start (cascade-style robustness) — both reuse the package operators unchanged.
function solve_cell(Re, Da, α, N, kv, etype, x0_dofs; max_iters=200)
    cell = build_cell("cont"; Re=Re, Da=Da, alpha_0=α, kv=kv, n=N, element_type=etype)
    r = probe_a2_heavy_solve(cell, :newton; verbose=false, max_iters=max_iters, x0_dofs=x0_dofs)
    if !get(r, "reached_root", false)
        rp = probe_a2_heavy_solve(cell, :picard; verbose=false, max_iters=max_iters, x0_dofs=x0_dofs)
        get(rp, "reached_root", false) && (r = rp)   # accept Picard only if IT reached the root
    end
    return (rnorm=get(r, "final_R", NaN), l2u=get(r, "l2_u", NaN), l2p=get(r, "l2_p", NaN),
            h1u=get(r, "h1_u", NaN), reached=get(r, "reached_root", false),
            niters=get(r, "iters", -1), dofs=get(r, "dofs", nothing), h=1.0 / N, cell=cell)
end

# Mesh-continuation step: take the converged (uh, ph) on a COARSE cell and use it (interpolated
# across meshes) as the initial guess for a single solve at the same physical point on a FINER
# mesh. Avoids re-running the whole α-continuation at every fine mesh — each finer rate point
# becomes ONE solve instead of ~30 continuation steps. Cross-mesh interpolation uses Gridap's
# `Interpolable` (fast point-location on the nested Cartesian meshes).
function mesh_step(coarse_cell, coarse_dofs, Re, Da, α, N_fine, kv, etype; max_iters=80)
    uh_c, ph_c = FEFunction(coarse_cell.setup.X, coarse_dofs)
    fine = build_cell("fine"; Re=Re, Da=Da, alpha_0=α, kv=kv, n=N_fine, element_type=etype)
    x0f = interpolate_everywhere([Interpolable(uh_c), Interpolable(ph_c)], fine.setup.X)
    r = probe_a2_heavy_solve(fine, :newton; verbose=false, max_iters=max_iters,
                             x0_dofs=get_free_dof_values(x0f))
    if !get(r, "reached_root", false)
        rp = probe_a2_heavy_solve(fine, :picard; verbose=false, max_iters=max_iters,
                                  x0_dofs=get_free_dof_values(x0f))
        get(rp, "reached_root", false) && (r = rp)
    end
    return (rnorm=get(r, "final_R", NaN), l2u=get(r, "l2_u", NaN), l2p=get(r, "l2_p", NaN),
            h1u=get(r, "h1_u", NaN), reached=get(r, "reached_root", false), niters=get(r, "iters", -1),
            dofs=get(r, "dofs", nothing), h=1.0 / N_fine, cell=fine)
end

# α-continuation toward `alpha_target` at fixed (Re, Da, N): geometric ramp from `alpha_start`
# with adaptive step-halving on failure (so a basin that shrinks faster than the step does not
# masquerade as a fold). Returns the converged target result, or `nothing` if the branch genuinely
# folds short of the target at this resolution.
function alpha_continuation(Re, Da, alpha_target, N, kv, etype;
                            alpha_start=1.0, nsteps=48, adaptive=true,
                            min_step_ratio=2e-3, max_iters=60)
    println(@sprintf("\n[α-cont] N=%d (%s, kv=%d): α %.4f → %.4f  (%d steps%s, ≤%d iters/step)",
        N, etype, kv, alpha_start, alpha_target, nsteps, adaptive ? ", adaptive" : "", max_iters))
    seed = solve_cell(Re, Da, alpha_start, N, kv, etype, nothing; max_iters=max_iters)
    println(@sprintf("  seed  α=%.4f  ‖R‖=%.2e  L2u=%.3e  iters=%d", alpha_start, seed.rnorm, seed.l2u, seed.niters))
    if !seed.reached
        println("  [!] seed is not a true root (‖R‖ above ftol) — aborting continuation.")
        return nothing
    end
    x_prev = seed.dofs
    last_good = seed
    logc = log(alpha_start); logt = log(alpha_target)
    step = (logt - logc) / nsteps
    base = abs(step); min_step = base * min_step_ratio
    while abs(logc - logt) > 1e-9
        trial = logc + step
        ((step > 0) == (trial > logt)) && (trial = logt)   # clamp exactly onto the target
        α = exp(trial)
        r = solve_cell(Re, Da, α, N, kv, etype, x_prev; max_iters=max_iters)
        if r.reached
            println(@sprintf("  step  α=%.4f  ‖R‖=%.2e  L2u=%.3e  H1u=%.3e  iters=%d", α, r.rnorm, r.l2u, r.h1u, r.niters))
            x_prev = r.dofs; logc = trial; last_good = r
            adaptive && (step = sign(step) * min(base, abs(step) * 2.0))   # grow back toward base
        elseif adaptive
            step /= 2.0
            if abs(step) < min_step
                println(@sprintf("  [!] FOLD near α=%.4f (‖R‖=%.2e): target α=%.4f unreachable at N=%d.",
                    α, r.rnorm, alpha_target, N))
                return nothing
            end
        else
            println(@sprintf("  [!] step failed at α=%.4f (‖R‖=%.2e), fixed steps — target unreachable.", α, r.rnorm))
            return nothing
        end
    end
    println(@sprintf("  [✓] reached target α=%.4f  ‖R‖=%.2e  L2u=%.3e  H1u=%.3e",
        alpha_target, last_good.rnorm, last_good.l2u, last_good.h1u))
    return last_good
end

# Rate study at the target corner via mesh-continuation: α-continuation ONCE at the (fine enough)
# base mesh to reach α_target, then interpolate-up to finer meshes with one solve each. Yields a
# defensible L2-vs-h sequence AT THE TARGET α (≥2 converged meshes ⇒ a real rate at the corner).
# Run base α-continuation at the first base candidate that reaches the target α (escalating to
# finer meshes if a coarser base still folds), then mesh-continue up the fine ladder. RETURNS the
# rate curve + status (used by both the single-cell driver and the Phase-2 batch).
function mesh_continuation_rate(Re, Da, αt, base_candidates, fine_ladder, kv, etype;
                                cont_nsteps=30, cont_minr=3e-2, cont_maxit=60, mesh_maxit=100,
                                direct_base=false, verbose=true)
    base = nothing; base_N = 0
    for cand in base_candidates
        if direct_base
            # suboptimal-rate cell: a true root ALREADY exists at this mesh (detection requires a
            # true root at both finest meshes for this category), so solve DIRECTLY from the
            # exact-solution guess — the α-ramp would only re-derive a known root (the "wasteful"
            # path). Escalate base_candidates only if a direct solve unexpectedly stalls.
            verbose && println(@sprintf(" [mesh-cont] DIRECT base solve N=%d (α=%.3f, root known to exist)", cand, αt))
            r0 = solve_cell(Re, Da, αt, cand, kv, etype, nothing; max_iters=cont_maxit)
            verbose && println(@sprintf("   base ‖R‖=%.2e L2u=%.3e H1u=%.3e iters=%d%s",
                r0.rnorm, r0.l2u, r0.h1u, r0.niters, r0.reached ? "" : "  [stall]"))
            r = r0.reached ? r0 : nothing
        else
            verbose && println(@sprintf(" [mesh-cont] base attempt N=%d (α=%.3f, Re=%.0e, Da=%.0e)", cand, αt, Re, Da))
            r = alpha_continuation(Re, Da, αt, cand, kv, etype;
                alpha_start=1.0, nsteps=cont_nsteps, adaptive=true, min_step_ratio=cont_minr, max_iters=cont_maxit)
        end
        if r !== nothing
            base = r; base_N = cand; break
        end
    end
    if base === nothing
        verbose && println(@sprintf("   [!] no base in %s reached α=%.3f.", string(base_candidates), αt))
        return Dict("status" => "base_failed", "base_N" => 0, "Ns" => Int[], "hs" => Float64[],
                    "l2us" => Float64[], "l2ps" => Float64[], "h1us" => Float64[], "residuals" => Float64[],
                    "slope_L2" => NaN, "slope_H1" => NaN, "finest_true_root" => nothing)
    end
    Ns = [base_N]; hs = [base.h]; l2us = [base.l2u]; l2ps = [base.l2p]; h1us = [base.h1u]; resids = [base.rnorm]
    prev = base
    for Nf in fine_ladder
        Nf > base_N || continue
        m = mesh_step(prev.cell, prev.dofs, Re, Da, αt, Nf, kv, etype; max_iters=mesh_maxit)
        if m.reached
            push!(Ns, Nf); push!(hs, m.h); push!(l2us, m.l2u); push!(l2ps, m.l2p); push!(h1us, m.h1u); push!(resids, m.rnorm)
            prev = m
            verbose && println(@sprintf("   mesh-step N=%d: ‖R‖=%.2e L2u=%.3e H1u=%.3e", Nf, m.rnorm, m.l2u, m.h1u))
        else
            verbose && println(@sprintf("   mesh-step N=%d NOT converged (‖R‖=%.2e); stopping ladder.", Nf, m.rnorm))
            break
        end
    end
    sL2 = length(Ns) >= 2 ? log(l2us[end-1]/l2us[end]) / log(hs[end-1]/hs[end]) : NaN
    sH1 = length(Ns) >= 2 ? log(h1us[end-1]/h1us[end]) / log(hs[end-1]/hs[end]) : NaN
    status = length(Ns) >= 2 ? "rescued" : "base_only"
    ftr = Dict("N" => Ns[end], "h" => hs[end], "l2u" => l2us[end], "l2p" => l2ps[end],
               "h1u" => h1us[end], "residual" => resids[end])
    if verbose
        println(@sprintf("   [%s] base N=%d; curve N=%s", status, base_N, string(Ns)))
        length(Ns) >= 2 && println(@sprintf("   slope L2u=%.2f (exp %d), H1u=%.2f (exp %d)", sL2, kv+1, sH1, kv))
    end
    return Dict("status" => status, "base_N" => base_N, "Ns" => Ns, "hs" => hs, "l2us" => l2us,
                "l2ps" => l2ps, "h1us" => h1us, "residuals" => resids, "slope_L2" => sL2,
                "slope_H1" => sH1, "finest_true_root" => ftr)
end

# Recursively replace non-finite floats (NaN/Inf) with `nothing` so the result is valid JSON
# (the JSON spec disallows NaN; base_failed/base_only cells carry NaN slopes).
_json_safe(x) = x
_json_safe(x::AbstractFloat) = isfinite(x) ? x : nothing
_json_safe(x::AbstractDict) = Dict(k => _json_safe(v) for (k, v) in x)
_json_safe(x::AbstractVector) = Any[_json_safe(v) for v in x]

# Phase-2 batch: read flagged_cells.json, dedup by physics cell (continuation is method-agnostic),
# rescue each via mesh-continuation, and write phase2_results.json (the merged report joins these
# back to BOTH ASGS & OSGS groups). Routing is by the detected `category` (hardest across methods):
#   suboptimal_rate (root exists) -> DIRECT base solve + mesh ladder (no wasteful α-ramp);
#   fold / total_failure at α<1    -> α-continuation escalating the base mesh past the fold;
#   α≥1 (pure NS, no fold)         -> coarse base, large budget, then mesh-continue.
function run_phase2_from_flagged(flagged_json::String, out_json::String; defaults_path=nothing)
    # Tunable ladders (override via a phase2_defaults.json).
    D = Dict{String, Any}(
        "fold_base_candidates" => [320, 512, 768, 1024], "fold_fine_ladder" => [512, 768, 1024, 1536], "fold_maxit" => 60,
        "slow_base_candidates" => [80], "slow_fine_ladder" => [160, 320], "slow_maxit" => 400,
        "subopt_base_candidates" => [320, 512], "subopt_fine_ladder" => [512, 768, 1024], "subopt_maxit" => 100,
        "mesh_maxit" => 150, "cont_nsteps" => 30, "cont_minr" => 3e-2)
    if defaults_path !== nothing && isfile(defaults_path)
        for (k, v) in JSON3.read(read(defaults_path, String), Dict{String, Any})
            D[String(k)] = v
        end
        println("[phase2] loaded ladder defaults from $defaults_path")
    end
    flagged = JSON3.read(read(flagged_json, String))
    # Dedup by physics cell, but track the categories seen across methods so we route by the
    # HARDEST one (a fold/total_failure for EITHER method means the cell needs the root-search path;
    # only an all-suboptimal_rate cell takes the cheap direct base solve).
    bykey = Dict{Tuple, Any}(); cats = Dict{Tuple, Set{String}}(); order = Tuple[]
    for fc in flagged
        key = (Float64(fc.Re), Float64(fc.Da), Float64(fc.alpha_0), Int(fc.k_velocity), String(fc.element_type))
        haskey(bykey, key) || (bykey[key] = fc; cats[key] = Set{String}(); push!(order, key))
        push!(cats[key], haskey(fc, :category) ? String(fc.category) : "")
    end
    cells = [bykey[k] for k in order]
    println(@sprintf("[phase2] %d flagged groups -> %d unique physics cells", length(flagged), length(cells)))
    results = Any[]
    for fc in cells
        Re = Float64(fc.Re); Da = Float64(fc.Da); αt = Float64(fc.alpha_0)
        kv = Int(fc.k_velocity); etype = String(fc.element_type)
        cset = cats[(Re, Da, αt, kv, etype)]
        # "suboptimal_rate" only (no fold/total_failure for any method) => root exists => direct base.
        subopt_only = ("suboptimal_rate" in cset) && !("fold" in cset) && !("total_failure" in cset)
        println("\n", "="^78)
        println(@sprintf("[phase2] cell Re=%.0e Da=%.0e α=%.3f k=%d %s  cats=%s", Re, Da, αt, kv, etype, string(collect(cset))))
        direct = false
        if subopt_only
            base_candidates = Int.(D["subopt_base_candidates"]); fine_ladder = Int.(D["subopt_fine_ladder"]); maxit = Int(D["subopt_maxit"]); direct = true
        elseif αt < 1.0       # fold / total_failure at a porous cell
            base_candidates = Int.(D["fold_base_candidates"]); fine_ladder = Int.(D["fold_fine_ladder"]); maxit = Int(D["fold_maxit"])
        else                  # slow (α≥1, pure NS, no fold)
            base_candidates = Int.(D["slow_base_candidates"]); fine_ladder = Int.(D["slow_fine_ladder"]); maxit = Int(D["slow_maxit"])
        end
        route = direct ? "suboptimal_direct" : (αt < 1.0 ? "fold" : "slow")
        r = mesh_continuation_rate(Re, Da, αt, base_candidates, fine_ladder, kv, etype;
            cont_nsteps=Int(D["cont_nsteps"]), cont_minr=Float64(D["cont_minr"]), cont_maxit=maxit,
            mesh_maxit=Int(D["mesh_maxit"]), direct_base=direct)
        push!(results, merge(Dict("Re" => Re, "Da" => Da, "alpha_0" => αt, "k_velocity" => kv,
                                  "element_type" => etype, "phase2_route" => route), r))
    end
    open(out_json, "w") do io
        JSON3.write(io, [_json_safe(r) for r in results])   # NaN/Inf -> null (JSON spec disallows NaN)
    end
    println(@sprintf("\n[phase2] wrote %d cell results -> %s", length(results), out_json))
    return results
end

function run_continuation(config_file="continuation_c24.json")
    cfg = JSON3.read(read(joinpath(@__DIR__, "data", config_file), String))
    Re = Float64(cfg["target"]["Re"]); Da = Float64(cfg["target"]["Da"]); αt = Float64(cfg["target"]["alpha_0"])

    # [regime] First-class, validated continuation regime. For backward compatibility an absent
    # "regime" defaults from the legacy key-presence convention (a "mesh_continuation" block ⇒
    # "mesh_ladder", otherwise ⇒ "alpha"), so every pre-existing continuation_*.json runs unchanged.
    default_regime = haskey(cfg, "mesh_continuation") ? "mesh_ladder" : "alpha"
    regime = String(get(cfg, "regime", default_regime))

    if regime == "mesh_ladder"
        # Mesh-continuation rate study: α-continuation ONCE at a base mesh, then interpolate-up.
        haskey(cfg, "mesh_continuation") || error("regime=\"mesh_ladder\" requires a \"mesh_continuation\" block (base_N, fine_Ns).")
        mc = cfg["mesh_continuation"]
        cc = get(cfg, "continuation", Dict())
        mesh_continuation_rate(Re, Da, αt, [Int(mc["base_N"])], Int.(collect(mc["fine_Ns"])),
            Int(get(cfg["element_spaces"], "k_velocity", 1)),
            String(get(cfg["mesh"], "element_type", "QUAD"));
            cont_nsteps=Int(get(cc, "nsteps", 30)), cont_minr=Float64(get(cc, "min_step_ratio", 3e-2)),
            cont_maxit=Int(get(cc, "max_iters_per_step", 60)), mesh_maxit=Int(get(mc, "mesh_max_iters", 100)))
        return
    elseif regime != "alpha"
        error("Unknown continuation regime \"$regime\". Supported: \"alpha\" (per-mesh α-ramp to the " *
              "target) and \"mesh_ladder\" (α-ramp at a base mesh + interpolate-up the fine ladder). " *
              "A \"da\" regime is a documented extension point — see the file header.")
    end

    # regime == "alpha": per-mesh α-continuation to the target corner.
    c = cfg["continuation"]
    αstart = Float64(get(c, "start", 1.0))
    nsteps = Int(get(c, "nsteps", 48))
    adaptive = Bool(get(c, "adaptive", true))
    minr = Float64(get(c, "min_step_ratio", 2e-3))
    maxit = Int(get(c, "max_iters_per_step", 60))
    Ns = Int.(collect(cfg["mesh"]["convergence_partitions"]))
    etype = String(get(cfg["mesh"], "element_type", "QUAD"))
    kv = Int(get(cfg["element_spaces"], "k_velocity", 1))

    println("="^78)
    println(@sprintf(" PARAMETER-CONTINUATION DRIVER  (axis=α)  target Re=%.0e Da=%.0e α=%.3f  N=%s",
        Re, Da, αt, string(Ns)))
    println("="^78)

    hs = Float64[]; l2us = Float64[]; h1us = Float64[]
    for N in Ns
        res = alpha_continuation(Re, Da, αt, N, kv, etype;
            alpha_start=αstart, nsteps=nsteps, adaptive=adaptive, min_step_ratio=minr, max_iters=maxit)
        if res === nothing
            println(@sprintf(" >> N=%d: target α=%.3f NOT reached (branch folds at this resolution).", N, αt))
        else
            push!(hs, res.h); push!(l2us, res.l2u); push!(h1us, res.h1u)
            println(@sprintf(" >> N=%d: CONVERGED at α=%.3f — ‖R‖=%.2e, L2u=%.3e, H1u=%.3e, iters=%d",
                N, αt, res.rnorm, res.l2u, res.h1u, res.niters))
        end
    end

    if length(hs) >= 2
        println("\n MMS convergence at the stiff corner (target α):")
        println(@sprintf("  %-10s %-13s %-13s", "h", "L2u", "H1u"))
        for i in eachindex(hs)
            println(@sprintf("  %-10.5f %-13.4e %-13.4e", hs[i], l2us[i], h1us[i]))
        end
        for i in 2:length(hs)
            ru = log(l2us[i-1] / l2us[i]) / log(hs[i-1] / hs[i])
            rh = log(h1us[i-1] / h1us[i]) / log(hs[i-1] / hs[i])
            println(@sprintf("  rate h=%.4f→%.4f:  L2u=%.2f (expect ~2),  H1u=%.2f (expect ~1)",
                hs[i-1], hs[i], ru, rh))
        end
    elseif length(hs) == 1
        println("\n Only one mesh reached the target — refine further for a rate estimate.")
    else
        println("\n No mesh reached the target α at the resolutions tried.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "phase2"
        flagged = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results", "flagged_cells.json")
        out = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "results", "phase2_results.json")
        defaults = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "data", "phase2_defaults.json")
        run_phase2_from_flagged(flagged, out; defaults_path=defaults)
    else
        run_continuation(length(ARGS) > 0 ? ARGS[1] : "continuation_c24.json")
    end
end
