# ==============================================================================================
# Strict-loader contract test.
#
# `load_frozen_config` is the strict (no-inheritance, no-silent-default) entry point for
# loading a `PorousNSConfig` from a single self-contained JSON file. This test locks in
# its failure contract: a deliberately incomplete config must NOT silently fill in
# defaults from `base_config.json` — it must raise an error.
#
# This guards the no-silent-defaults doctrine (CLAUDE.md) and is the contract anchor for
# the planned Fix 7 migration that promotes `load_frozen_config` to the production loader.
# ==============================================================================================

using Test
using PorousNSSolver
using JSON3

@testset "load_frozen_config rejects incomplete configs [P-011 / Fix 7]" begin
    # Build a config dict that is intentionally missing required fields. The base
    # config has nu/eps_val in physical_properties; omitting eps_val must fail.
    incomplete = Dict(
        "physical_properties" => Dict(
            "nu" => 1.0,
            "f_x" => 0.0,
            "f_y" => 0.0,
            "reaction_model" => "Constant_Sigma",
            "sigma_constant" => 1.0,
            "sigma_linear" => 0.0,
            "sigma_nonlinear" => 0.0,
            "u_base_floor_ref" => 1e-4,
            "h_floor_weight" => 0.1,
            "epsilon_floor" => 1e-12,
            "tau_regularization_limit" => 1e-12
            # eps_val deliberately missing
        )
    )

    tmp = tempname() * ".json"
    open(tmp, "w") do io
        JSON3.write(io, incomplete)
    end

    @test_throws Exception PorousNSSolver.load_frozen_config(tmp)

    rm(tmp; force=true)
end

@testset "load_frozen_config rejects nonexistent path [P-011 / Fix 7]" begin
    @test_throws ErrorException PorousNSSolver.load_frozen_config("/nonexistent/config.json")
end
