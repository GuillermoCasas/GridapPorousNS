# [must-test] JFNK ↔ frozen-π equivalence + the win, through the full MMS harness.
#
# Runs the SAME MMS cell twice through the production pipeline (solve_system → solve_osgs_stage!) —
# osgs_jfnk_enabled=false (the default frozen-π inexact-Newton coupled solve) vs true (matrix-free
# full-tangent JFNK) — and asserts JFNK changes HOW the OSGS solve converges, not WHERE:
#   • ASGS is byte-identical (the JFNK flag is a no-op for the ASGS path) — behavior preservation.
#   • OSGS converges to the SAME root (errors agree), and JFNK takes ≤ the frozen-π iteration count.
# The rigorous same-root proof against the exact full-Jacobian-Newton root (to 1e-6) and the C.1
# inner-honesty check live in test/quick/osgs_jfnk_quick_test.jl; this is the integration anchor.
# Gate / cost model: docs/solver/jfnk-phase0-preconditioner-gate.md.

using Test
const _MMS_DIR = joinpath(@__DIR__, "ManufacturedSolutions")
include(joinpath(_MMS_DIR, "run_test.jl"))   # main-guarded; provides run_mms + JSON3 + HDF5 + helpers

const _JF_RE, _JF_DA, _JF_A0 = 1.0, 1.0, 0.5    # a moderate cell where the frozen-π OSGS solve converges
const _JF_NS = [10, 20]
# ASGS is unaffected by the flag ⇒ byte-identical. OSGS uses a DIFFERENT (fuller) tangent, so the two
# paths can stop at slightly different iterates within the scale-free gate; 1e-4 confirms the same root
# (it would catch any genuine O(1) divergence) while tolerating that gate-level difference.
const _JF_RTOL_ASGS = 1e-12
const _JF_RTOL_OSGS = 1e-4

function _write_jfnk_config(jfnk::Bool, h5name::String)
    cfg = JSON3.read(read(joinpath(_MMS_DIR, "data", "phase1_quad_k1.json"), String), Dict{String,Any})
    cfg["physical_properties"]["Re"] = [_JF_RE]
    cfg["physical_properties"]["Da"] = [_JF_DA]
    cfg["domain"]["alpha_0"]          = [_JF_A0]
    cfg["numerical_method"]["mesh"]["convergence_partitions"] = _JF_NS
    cfg["numerical_method"]["mesh"]["element_type"] = ["QUAD"]
    cfg["numerical_method"]["stabilization"]["method"] = ["ASGS", "OSGS"]
    cfg["numerical_method"]["solver"]["osgs_jfnk_enabled"] = jfnk
    cfg["h5_filename"] = h5name
    cfg["skip_cells"]  = Any[]
    cfg["erase_past_results"] = true
    fname = "_jfnk_$(jfnk).json"
    open(joinpath(_MMS_DIR, "data", fname), "w") do io; JSON3.write(io, cfg); end
    return fname
end

"Read {method => (eu, ep, total_iters)} from an MMS results HDF5."
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

@testset "JFNK equivalence + win: frozen-π vs JFNK (Re=$_JF_RE, Da=$_JF_DA) [JFNK]" begin
    run_mms(_write_jfnk_config(false, "debug_results/_jfnk_off.h5"))
    run_mms(_write_jfnk_config(true,  "debug_results/_jfnk_on.h5"))
    off = _read_errs_iters(joinpath(_MMS_DIR, "results", "debug_results", "_jfnk_off.h5"))
    on  = _read_errs_iters(joinpath(_MMS_DIR, "results", "debug_results", "_jfnk_on.h5"))

    @test haskey(off, "ASGS") && haskey(on, "ASGS")
    @test haskey(off, "OSGS") && haskey(on, "OSGS")

    # Behavior preservation: the JFNK flag must NOT touch the ASGS path → byte-identical.
    @test maximum(abs.(on["ASGS"].eu .- off["ASGS"].eu) ./ max.(abs.(off["ASGS"].eu), eps())) <= _JF_RTOL_ASGS
    @test maximum(abs.(on["ASGS"].ep .- off["ASGS"].ep) ./ max.(abs.(off["ASGS"].ep), eps())) <= _JF_RTOL_ASGS
    @test on["ASGS"].it == off["ASGS"].it

    # Same OSGS root — JFNK changes HOW, not WHERE.
    @test maximum(abs.(on["OSGS"].eu .- off["OSGS"].eu) ./ max.(abs.(off["OSGS"].eu), eps())) <= _JF_RTOL_OSGS
    @test maximum(abs.(on["OSGS"].ep .- off["OSGS"].ep) ./ max.(abs.(off["OSGS"].ep), eps())) <= _JF_RTOL_OSGS
    # The win: JFNK never costs MORE total nonlinear iterations than the frozen-π coupled solve.
    @test on["OSGS"].it <= off["OSGS"].it

    for jf in (false, true)
        rm(joinpath(_MMS_DIR, "data", "_jfnk_$(jf).json"); force=true)
    end
end
