# [must-test] P4 — Newton↔Picard ping-pong equivalence.
#
# Runs a stiff cell (high Re AND high Da, where Stage-I Newton stalls so the cascade actually engages)
# twice through the MMS harness — one-way cascade (pingpong_enabled=false) vs adaptive ping-pong
# (pingpong_enabled=true) — and asserts the schedule changes HOW the solve gets there, never WHERE:
# the converged errors must be identical to a tight tolerance, and ping-pong must take ≤ the one-way
# iteration count. This is the P4 regression anchor (the manual stiff A/B made permanent).

using Test
const _MMS_DIR = joinpath(@__DIR__, "ManufacturedSolutions")
include(joinpath(_MMS_DIR, "run_test.jl"))   # main-guarded; provides run_mms + JSON3 + HDF5 + helpers

const _PP_RE, _PP_DA, _PP_A0 = 1e6, 1e6, 0.5
const _PP_NS = [10, 20]
const _PP_RTOL = 1e-9   # "same converged solution" — well above the schedule's float noise

function _write_pp_config(pingpong::Bool, h5name::String)
    cfg = JSON3.read(read(joinpath(_MMS_DIR, "data", "phase1_quad_k1.json"), String), Dict{String,Any})
    cfg["physical_properties"]["Re"] = [_PP_RE]
    cfg["physical_properties"]["Da"] = [_PP_DA]
    cfg["domain"]["alpha_0"]          = [_PP_A0]
    cfg["numerical_method"]["mesh"]["convergence_partitions"] = _PP_NS
    cfg["numerical_method"]["mesh"]["element_type"] = ["QUAD"]
    cfg["numerical_method"]["stabilization"]["method"] = ["ASGS", "OSGS"]
    cfg["solver_stall_window"] = 2          # bail Newton fast so ping-pong hands off promptly
    cfg["solver_stall_min_rel_improvement"] = 0.001
    sol = cfg["numerical_method"]["solver"]
    sol["pingpong_enabled"] = pingpong
    sol["pingpong_max_swaps"] = 3
    sol["pingpong_picard_gain_orders"] = 1.5
    cfg["h5_filename"] = h5name
    cfg["skip_cells"]  = Any[]
    cfg["erase_past_results"] = true
    fname = "_pp_$(pingpong).json"
    open(joinpath(_MMS_DIR, "data", fname), "w") do io; JSON3.write(io, cfg); end
    return fname
end

"Read {method => (err_u_l2, err_p_l2, total_iters)} from an MMS results HDF5."
function _read_errs_iters(h5path::String)
    out = Dict{String,Any}()
    HDF5.h5open(h5path, "r") do f
        for k in keys(f)
            g = f[k]
            haskey(attributes(g), "method") || continue
            m = String(read(attributes(g)["method"]))
            out[m] = (eu = Float64.(vec(read(g["err_u_l2"]))),
                      ep = Float64.(vec(read(g["err_p_l2"]))),
                      it = Int(read(attributes(g)["total_iters"])))
        end
    end
    return out
end

@testset "ping-pong equivalence: one-way vs adaptive (Re=$_PP_RE, Da=$_PP_DA) [P4]" begin
    run_mms(_write_pp_config(false, "debug_results/_pp_off.h5"))
    run_mms(_write_pp_config(true,  "debug_results/_pp_on.h5"))
    off = _read_errs_iters(joinpath(_MMS_DIR, "results", "debug_results", "_pp_off.h5"))
    on  = _read_errs_iters(joinpath(_MMS_DIR, "results", "debug_results", "_pp_on.h5"))
    for m in ("ASGS", "OSGS")
        @test haskey(off, m) && haskey(on, m)
        # Same converged solution — the schedule changes HOW, not WHERE.
        @test maximum(abs.(on[m].eu .- off[m].eu) ./ max.(abs.(off[m].eu), eps())) <= _PP_RTOL
        @test maximum(abs.(on[m].ep .- off[m].ep) ./ max.(abs.(off[m].ep), eps())) <= _PP_RTOL
        # Ping-pong never costs MORE total nonlinear iterations than the one-way cascade.
        @test on[m].it <= off[m].it
    end
    for pp in (false, true)
        rm(joinpath(_MMS_DIR, "data", "_pp_$(pp).json"); force=true)
    end
end
