# ==============================================================================================
# Config-contract test for the OSGS "freeze_after_k" projection-coupling mode.
#
# Pins the validation contract:
#   osgs_projection_coupling = "freeze_after_k"  (k projection-updating warm-up steps, then freeze π)
#   osgs_freeze_after_k::Int  (k ≥ 1 ; number of projection-updating warm-up iterations before freezing)
#
# Guards the no-magic-numbers / no-silent-defaults doctrine (CLAUDE.md): the mode and its knob must be
# explicit, and out-of-contract values must fail loudly in validate!. This is the config anchor; the
# end-to-end behaviour (optimal rate, ~99% of the OSGS constant at k=2-3) is covered by the MMS A/B harness.
# ==============================================================================================

using Test
using PorousNSSolver
using JSON3

# Minimal override onto base_config.json. base_config intentionally omits `eps_val`, so any override that
# wants to load must supply it; we set the stabilization block to the new mode.
function _freeze_cfg(; coupling="freeze_after_k", k=2)
    override = Dict(
        "physical_properties" => Dict("eps_val" => 1e-6),
        "numerical_method" => Dict(
            "stabilization" => Dict(
                "osgs_projection_coupling" => coupling,
                "osgs_freeze_after_k" => k,
            ),
        ),
    )
    tmp = tempname() * ".json"
    open(tmp, "w") do io
        JSON3.write(io, override)
    end
    return tmp
end

@testset "freeze_after_k config contract" begin
    # Valid: mode + in-contract knob load and validate.
    cfg = PorousNSSolver.load_config_with_base_template(_freeze_cfg(k=3))
    @test cfg.numerical_method.stabilization.osgs_projection_coupling == "freeze_after_k"
    @test cfg.numerical_method.stabilization.osgs_freeze_after_k == 3

    # k must be ≥ 1 (k=0, i.e. pure ASGS, is expressed via method=ASGS, not via this knob).
    @test_throws Exception PorousNSSolver.load_config_with_base_template(_freeze_cfg(k=0))

    # Unknown coupling strings are rejected by the enum assertion.
    @test_throws Exception PorousNSSolver.load_config_with_base_template(_freeze_cfg(coupling="freeze_forever"))

    # The two legacy modes still validate (no regression in the enum).
    @test PorousNSSolver.load_config_with_base_template(_freeze_cfg(coupling="staggered")).numerical_method.stabilization.osgs_projection_coupling == "staggered"
    @test PorousNSSolver.load_config_with_base_template(_freeze_cfg(coupling="coupled")).numerical_method.stabilization.osgs_projection_coupling == "coupled"
end
