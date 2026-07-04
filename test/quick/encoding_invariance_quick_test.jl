# [must-test] Encoding / scale-covariance invariant (Quick version).
#
# A consistent non-dimensionalisation is a pure change of units: solving the SAME dimensionless
# cell (Re, Da, α₀) under two different (L, U) encodings must give the SAME *dimensionless* solution
# — hence identical dimensionless errors (`err_u_l2`, `err_u_h1`, `err_p_l2`), up to floating-point
# roundoff. The harness already reports dimensionless errors (`calculate_normalized_errors`), so two
# encodings of one cell should agree to ~roundoff. (Literal bit-identity is only achievable for
# power-of-two scalings; for the harness's √(Da)-style encodings we check a tight relative tolerance.)
#
# This guards scale-covariance. Both ASGS and OSGS now hold it (after the physical_epsilon + inner-gate
# covariance fixes, docs/lessons_learned.md 2026-06-02), so both are checked with a tight @test below.
#
# Cell chosen non-extreme but nontrivial: Re=10 (mild inertia), Da=2 (≠1 so the encodings give
# genuinely different L and U), α₀=0.5 (mid porosity). Small N for speed.

using Test

const _MMS_DIR = joinpath(@__DIR__, "..", "extended", "ManufacturedSolutions")
include(joinpath(_MMS_DIR, "run_test.jl"))   # main-guarded; provides run_mms + JSON3 + HDF5 + helpers

const _INV_RE, _INV_DA, _INV_A0 = 10.0, 2.0, 0.5
const _INV_NS = [4]
# Threshold near roundoff. Two covariance bugs were found and fixed (docs/lessons_learned.md 2026-06-02):
#   • ASGS: the DIMENSIONAL pressure penalty `physical_epsilon` was injected as a fixed 1e-8 for every encoding;
#     scaling it covariantly (physical_epsilon = ε̂·(U/L)/P_c in build_mms_formulation) → ASGS covariant to ~1e-10.
#   • OSGS: the inner-Newton stop was encoding-dependent — the per-field gate normalised by ‖x_k‖
#     (∝U, ∝P_c∝U² for pressure) left the iterates non-covariant (worst in pressure, ≈2e-2). Fix:
#     per-field gate normalised by the INITIAL residual ‖R₀_k‖ (nonlinear.jl) → OSGS covariant to
#     ~1e-10. BOTH methods must now hold to ≤_INV_RTOL — this guards both fixes.
const _INV_RTOL = 5e-8

"Write a one-cell config that differs from the others ONLY in `encoding_strategy`, return its filename.
`methods` defaults to an ASGS+OSGS run."
function _write_inv_config(enc::String, h5name::String;
                           methods::Vector{String}=["ASGS", "OSGS"], tag::String="")
    cfg = JSON3.read(read(joinpath(_MMS_DIR, "data", "phase1_quad_k1.json"), String), Dict{String,Any})
    cfg["mms_verification_enabled"] = false
    cfg["physical_properties"]["Re"] = [_INV_RE]
    cfg["physical_properties"]["Da"] = [_INV_DA]
    cfg["domain"]["alpha_0"]          = [_INV_A0]
    cfg["numerical_method"]["element_spaces"]["k_velocity"] = [1]
    cfg["numerical_method"]["element_spaces"]["k_pressure"] = [1]
    cfg["numerical_method"]["mesh"]["convergence_partitions"] = _INV_NS
    cfg["numerical_method"]["mesh"]["element_type"] = ["QUAD"]
    cfg["numerical_method"]["stabilization"]["method"] = methods
    cfg["encoding_strategy"] = enc
    cfg["h5_filename"] = h5name
    cfg["skip_cells"] = Any[]
    cfg["erase_past_results"] = true
    fname = "_inv_$(enc)$(tag).json"
    open(joinpath(_MMS_DIR, "data", fname), "w") do io
        JSON3.write(io, cfg)
    end
    return fname
end

"Read {method => err_u_l2 vector} from an MMS results HDF5."
function _read_method_errors(h5path::String, field::String)
    out = Dict{String,Vector{Float64}}()
    HDF5.h5open(h5path, "r") do f
        for k in keys(f)
            g = f[k]
            haskey(attributes(g), "method") || continue
            m = read(attributes(g)["method"])
            haskey(g, field) || continue
            out[String(m)] = Float64.(vec(read(g[field])))
        end
    end
    return out
end

@testset "encoding invariance (scale-covariance: Re=$_INV_RE, Da=$_INV_DA, α=$_INV_A0)" begin
    # Confirm the two encodings actually differ for this cell (else the test is vacuous).
    Lc, Uc = compute_L_and_U("centered", _INV_RE, _INV_DA, 1.0)
    Lm, Um = compute_L_and_U("minmax",   _INV_RE, _INV_DA, 1.0)
    @info "encodings" centered=(Lc, Uc) minmax=(Lm, Um)
    @test !(isapprox(Lc, Lm) && isapprox(Uc, Um))   # genuinely different scalings

    # Ad-hoc/debug scratch DBs go under results/debug_results/ to keep results/ clean (project convention).
    run_mms(_write_inv_config("centered", "debug_results/_inv_centered.h5"))
    run_mms(_write_inv_config("minmax",   "debug_results/_inv_minmax.h5"))

    cpath = joinpath(_MMS_DIR, "results", "debug_results", "_inv_centered.h5")
    mpath = joinpath(_MMS_DIR, "results", "debug_results", "_inv_minmax.h5")

    for field in ("err_u_l2", "err_u_h1", "err_p_l2")
        ec = _read_method_errors(cpath, field)
        em = _read_method_errors(mpath, field)
        for method in ("ASGS", "OSGS")
            @test haskey(ec, method) && haskey(em, method)
            reldiff = maximum(abs.(ec[method] .- em[method]) ./ max.(abs.(em[method]), eps()))
            @info "scale-covariance check" field method reldiff centered=ec[method] minmax=em[method]
            # Both ASGS and OSGS are now scale-covariant after the physical_epsilon + inner-gate/warmup fixes.
            @test reldiff <= _INV_RTOL
        end
    end

    # cleanup scratch configs (the results HDF5 / traces live under gitignored results/)
    for enc in ("centered", "minmax")
        rm(joinpath(_MMS_DIR, "data", "_inv_$(enc).json"); force=true)
    end
end
