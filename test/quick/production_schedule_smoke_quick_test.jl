# [must-test] P6 — production-path smoke test.
#
# Exercises the PRODUCTION entry point `run_simulation` (strict `load_frozen_config` → mesh → FE spaces →
# `solve_system` → export) with the iterator-scheduling features turned ON (stall guard + Newton↔Picard
# ping-pong), and asserts it completes with finite fields. This is the P6 anchor: the production path now
# (a) goes through the shared `build_iter_solvers`, (b) receives the stall guard from `SolverConfig`, and
# (c) reaches the ping-pong via `solve_system` reading `pingpong_*`. It also guards the P6 `cfg.phys` fix
# (the production path used to crash on `cfg.phys.f_x`). The default-OFF path is bit-identical to legacy
# (proven on the MMS harness via probe_k1); here we only require the FEATURES-ON run to converge + be finite.

using Test
using PorousNSSolver
using JSON3
using Gridap

const _CFG_DIR = joinpath(@__DIR__, "..", "..", "config")
const _OUT_DIR = joinpath(@__DIR__, "..", "extended", "ManufacturedSolutions", "results", "debug_results")

"Build a complete, self-contained config (base_config + the physical_epsilon the strict loader requires + a tiny
mesh + schedule features) and write it to a temp path. Returns the path."
function _write_production_config(; pingpong::Bool)
    cfg = JSON3.read(read(joinpath(_CFG_DIR, "base_config.json"), String), Dict{String,Any})
    # base_config intentionally omits physical_epsilon (so it can't be silently inherited; docs/known_issues.md);
    # the strict production loader requires it, so supply it explicitly for the smoke run.
    cfg["physical_properties"]["physical_epsilon"] = 1e-7
    cfg["numerical_method"]["mesh"]["partition"] = [4, 4]
    cfg["numerical_method"]["stabilization"]["method"] = "ASGS"
    sol = cfg["numerical_method"]["solver"]
    sol["pingpong_enabled"] = pingpong
    sol["pingpong_max_swaps"] = 2
    sol["pingpong_picard_gain_orders"] = 1.5
    sol["newton_stall_window"] = pingpong ? 2 : 0
    sol["newton_stall_min_rel_improvement"] = 1e-3
    cfg["output"]["directory"] = _OUT_DIR
    cfg["output"]["basename"] = "_p6_smoke_$(pingpong)"
    mkpath(_OUT_DIR)
    path = joinpath(_OUT_DIR, "_p6_smoke_$(pingpong).json")
    open(path, "w") do io; JSON3.write(io, cfg); end
    return path
end

@testset "production run_simulation smoke (schedule features) [P6]" begin
    for pingpong in (false, true)
        path = _write_production_config(; pingpong=pingpong)
        u_h, p_h, model, Ω, dΩ = PorousNSSolver.run_simulation(path)
        # Finite, real fields out of the production path.
        u_dofs = get_free_dof_values(u_h)
        p_dofs = get_free_dof_values(p_h)
        @test all(isfinite, u_dofs)
        @test all(isfinite, p_dofs)
        @test length(u_dofs) > 0
        rm(path; force=true)
    end
end
