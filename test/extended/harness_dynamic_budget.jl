# test/extended/harness_dynamic_budget.jl
# ==============================================================================================
# [harness-frame] Re/Da iteration-budget knobs for the MMS / Cocquet test harnesses.
#
# These were historically declared in the PRODUCTION `SolverConfig` (`src/config.jl`), but nothing
# under `src/` ever read them — only the test harnesses do. Re and Da are GLOBAL dimensionless numbers
# (they need characteristic scales U, L that only a benchmark fixes), so they are test-frame quantities;
# the production solver instead keys its adaptivity on the element-wise cell-Péclet |u|h_K/ν already
# embodied by τ_{1,NS}. They were therefore relocated here, out of production config
# (docs/formulation-audit-2026-06-24.md §A.1 / F1).
#
# This file is the SINGLE SOURCE OF TRUTH for the defaults (formerly `config/base_config.json`). The
# harnesses read an OPTIONAL top-level `mms_dynamic_budget` object from their raw test config dict; any
# unspecified knob falls back to the default below. The only knob ever overridden in practice is
# `newton_re_iterations` (set to 150 in the high-Re sweep configs); the rest are globally constant.
#
# Included (idempotently) by the harness entry points:
#   - test/extended/ManufacturedSolutions/run_test.jl          (reads via test_dict)
#   - test/extended/CocquetFormMMS/run_test.jl                 (reads via test_dict)
#   - test/extended/ManufacturedSolutions/probe_stiff_diagnose.jl                 (defaults)
#   - test/extended/ManufacturedSolutions/diagnostics/velocity_centering_probe.jl (defaults)
#   - test/extended/ManufacturedSolutions/diagnostics/jacobian_equilibration_osgs_probe.jl (defaults)
#   - test/extended/ManufacturedSolutions3D/smoke3d.jl         (defaults)
# ==============================================================================================

# Defaults — formerly the `config/base_config.json` solver-block values. Field names are the budget's
# own (no `dynamic_` prefix); the same names are the keys of the JSON `mms_dynamic_budget` block.
const MMS_DYNAMIC_BUDGET_DEFAULTS = (
    picard_re_threshold        = 1.0e4,   # Re above which Picard's iteration budget is widened
    picard_re_iterations       = 15,      # Picard iterations used past the Re threshold
    picard_da_threshold        = 1.0e4,   # Da above which Picard's iteration budget is widened
    picard_da_iterations       = 10,      # Picard iterations used past the Da threshold
    newton_re_threshold        = 1.0e4,   # Re above which Newton's iteration budget is widened
    newton_re_iterations       = 60,      # Newton iterations used past the Re threshold
    ftol_ceiling               = 1.0e-4,  # upper clamp on the mesh-scaled absolute ftol O(h^{kv+1})
    ftol_spatial_safety_factor = 1.0e-2,  # margin (0,1] applied to that O(h^{kv+1}) discretization-error ftol
)

# Read the harness-frame dynamic budget for one run.
#   `raw`  — the harness's raw test config dict (JSON3 → Dict{String,Any}); pass `nothing` for the
#            programmatic Dict-built probe cells, which inherit the defaults wholesale.
# Returns a NamedTuple with the eight fields above, coerced to the default types. Any key absent from the
# optional top-level `mms_dynamic_budget` block falls back to its default. FAILS LOUD if a legacy
# `dynamic_*` knob is still sitting under `numerical_method.solver` (a forgotten migration), so a stale
# config can never silently fall back to a default (e.g. newton_re_iterations 150 → 60).
function read_mms_dynamic_budget(raw=nothing)
    d = MMS_DYNAMIC_BUDGET_DEFAULTS
    raw === nothing && return d

    sol = get(get(raw, "numerical_method", Dict()), "solver", Dict())
    for k in ("dynamic_picard_re_threshold", "dynamic_picard_re_iterations",
              "dynamic_picard_da_threshold", "dynamic_picard_da_iterations",
              "dynamic_newton_re_threshold", "dynamic_newton_re_iterations",
              "dynamic_ftol_ceiling", "dynamic_ftol_spatial_safety_factor")
        if haskey(sol, k)
            error("Legacy harness-frame key '$k' found under numerical_method.solver. These knobs moved " *
                  "to a top-level `mms_dynamic_budget` block (test/extended/harness_dynamic_budget.jl); " *
                  "move it there so it is not silently dropped.")
        end
    end

    blk = get(raw, "mms_dynamic_budget", Dict())
    nt = (
        picard_re_threshold        = Float64(get(blk, "picard_re_threshold",        d.picard_re_threshold)),
        picard_re_iterations       = Int(    get(blk, "picard_re_iterations",       d.picard_re_iterations)),
        picard_da_threshold        = Float64(get(blk, "picard_da_threshold",        d.picard_da_threshold)),
        picard_da_iterations       = Int(    get(blk, "picard_da_iterations",       d.picard_da_iterations)),
        newton_re_threshold        = Float64(get(blk, "newton_re_threshold",        d.newton_re_threshold)),
        newton_re_iterations       = Int(    get(blk, "newton_re_iterations",       d.newton_re_iterations)),
        ftol_ceiling               = Float64(get(blk, "ftol_ceiling",               d.ftol_ceiling)),
        ftol_spatial_safety_factor = Float64(get(blk, "ftol_spatial_safety_factor", d.ftol_spatial_safety_factor)),
    )

    # Fail-loud guards (ported from the old SolverConfig `validate!` asserts; the ftol_ceiling >= ftol
    # cross-check is now inherent in the harness `max(ftol, min(ceiling, ...))` arithmetic).
    @assert nt.picard_re_threshold >= 1.0 "mms_dynamic_budget.picard_re_threshold must be >= 1"
    @assert nt.picard_re_iterations >= 1 "mms_dynamic_budget.picard_re_iterations must be >= 1"
    @assert nt.picard_da_threshold >= 1.0 "mms_dynamic_budget.picard_da_threshold must be >= 1"
    @assert nt.picard_da_iterations >= 1 "mms_dynamic_budget.picard_da_iterations must be >= 1"
    @assert nt.newton_re_threshold >= 1.0 "mms_dynamic_budget.newton_re_threshold must be >= 1"
    @assert nt.newton_re_iterations >= 1 "mms_dynamic_budget.newton_re_iterations must be >= 1"
    @assert nt.ftol_ceiling > 0.0 "mms_dynamic_budget.ftol_ceiling must be > 0"
    @assert 0.0 < nt.ftol_spatial_safety_factor <= 1.0 "mms_dynamic_budget.ftol_spatial_safety_factor must be in (0, 1]"
    return nt
end
