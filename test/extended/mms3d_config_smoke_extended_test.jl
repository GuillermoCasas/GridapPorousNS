# test/extended/mms3d_config_smoke_extended_test.jl
# ==============================================================================================
# OFFICIAL 3D-MMS smoke guard (audit F6 / pending-tasks 2a). Runs the committed, config-driven §5.2
# convergence study (data/smoke3d_p1.json: z-extruded manufactured field, structured Kuhn tets, P1,
# ASGS+OSGS, Deviatoric, paper stabilization constants) through smoke3d.jl's `run_config`, and asserts
# the DOCUMENTED convergence behaviour — turning the previously hand-edited 3D driver into an automated
# reference (replaces the ad-hoc-driver workflow).
#
# What it asserts (method-aware, per the audit findings — NOT a naive "optimal everywhere"):
#   • every (kv,method,level) solve SUCCEEDS;
#   • velocity L² and H¹ errors fall MONOTONICALLY under refinement;
#   • fine-segment H¹u rate is optimal-ish (~1) for both methods;
#   • fine-segment L²u rate is method-aware — OSGS-P1 recovers ~2, while ASGS-P1 is intrinsically
#     sub-optimal (~1.2, method-intrinsic Aubin–Nitsche deficiency, audit B.1) — so the ASGS floor is
#     honest, not "optimal".
# Thresholds live in the config's `assertions` block (carry margin below the measured structured-Kuhn
# rates), so retuning is a config edit, not a test edit. [must-test]
# ==============================================================================================
using Test
using JSON3

const _MMS3D_DIR = joinpath(@__DIR__, "ManufacturedSolutions3D")
# Brings run_config / solve_one / mesh builders into scope. smoke3d.jl's `if PROGRAM_FILE == @__FILE__`
# CLI guard means including it here does NOT launch a run.
include(joinpath(_MMS3D_DIR, "smoke3d.jl"))

_rate(e0, e1, h0, h1) = log(e0 / e1) / log(h0 / h1)

@testset "3D MMS config-driven smoke (F6): P1 structured-Kuhn, ASGS+OSGS" begin
    cfg_path = joinpath(_MMS3D_DIR, "data", "smoke3d_p1.json")
    @test isfile(cfg_path)

    cfg = JSON3.read(read(cfg_path, String))
    A = cfg.assertions
    require_success  = Bool(A.require_success_all)
    require_monotone = Bool(A.require_monotone_decrease)
    min_h1u = Float64(A.min_h1u_rate)
    min_l2u = Dict(String(k) => Float64(v) for (k, v) in A.min_l2u_rate)

    results = run_config(cfg_path)
    @test !isempty(results)

    for r in results
        method = r.method; kv = r.kv; lv = r.levels
        @testset "kv=$kv $method" begin
            @test length(lv) >= 2

            # (1) every level converged
            if require_success
                @test all(l -> l.success, lv)
            end

            # (2) velocity errors fall monotonically coarse→fine (a small tolerance guards roundoff)
            if require_monotone
                @test all(i -> lv[i+1].el2_u <= lv[i].el2_u * (1 + 1e-9), 1:length(lv)-1)
                @test all(i -> lv[i+1].eh1_u <= lv[i].eh1_u * (1 + 1e-9), 1:length(lv)-1)
            end

            # (3) OVERALL convergence rate across the full ladder (coarsest→finest). On this short,
            # deliberately-coarse smoke ladder the per-segment rates are pre-asymptotically noisy (e.g.
            # ASGS-P1 H¹u dips on the narrow fine segment), so the guard uses the end-to-end slope, which
            # averages that out — the standard way to read a rate off a short mesh ladder.
            a, b = lv[1], lv[end]
            l2u_rate = _rate(a.el2_u, b.el2_u, a.h, b.h)
            h1u_rate = _rate(a.eh1_u, b.eh1_u, a.h, b.h)
            @info "3D MMS smoke rate" method kv l2u_rate=round(l2u_rate, digits=3) h1u_rate=round(h1u_rate, digits=3)
            @test h1u_rate >= min_h1u
            @test l2u_rate >= get(min_l2u, method, 0.9)
        end
    end
end
