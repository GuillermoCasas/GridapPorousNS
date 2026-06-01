# test/fast/test_fast_mms_exactness.jl
# ==============================================================================================
# Nature & Intent:
# Validates the physical correctness and exactness of the Manufactured Solutions setup. It enforces 
# the rules in `fast-verification.md` by directly testing the dimensional characteristic scaling 
# $P_c = (1 + Re + Da) U \nu / L$ from the theoretical formulation without invoking PDE solvers.
# This prevents ill-posed simulations by asserting that physical parameters loaded from JSON 
# precisely match the analytical closure properties.
#
# Mathematical Formulation Alignment:
# Highly consistent. Checks adherence to analytical characteristic values, guarding against 
# parameter drift or incorrect dimensional normalization during configuration loads.
#
# Associated Files / Functions:
# - `src/formulations/continuous_problem.jl` (`get_characteristic_scales`, `build_mms_formulation`)
# - `src/utils/porosity.jl` (`build_porosity_field`)
# - `test/extended/ManufacturedSolutions/run_test.jl` (Testing its configuration loading pipeline)
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Test
using PorousNSSolver
using Gridap
using JSON3

# Include the test framework builder script directly to ensure we strictly test its assembly pipeline
include("../extended/ManufacturedSolutions/run_test.jl")

@testset "MMS Physical Correctness (Da / Re) Check" begin
    # Create the baseline dummy configuration struct exactly as constructed inside run_test.jl
    config_path_dummy = joinpath(@__DIR__, "..", "extended", "ManufacturedSolutions", "data", "test_config.json")
    test_dict = JSON3.read(read(config_path_dummy, String), Dict{String, Any})
    
    # 1. Extract physical boundaries natively from our JSON pipeline instead of hardcoding
    Re_bound = Float64(last(get(get(test_dict, "physical_properties", Dict()), "Re", [1.0e6])))
    Da_bound = Float64(first(get(get(test_dict, "physical_properties", Dict()), "Da", [1.0e-6])))
    
    config_dict = Dict(
        "physical_properties" => Dict("nu" => 1.0, "eps_val" => 1e-8, "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
        # Set r_1, r_2 explicitly from test_config (0.2 / 0.4) instead of falling back to
        # base_config.json (which has r_2 = 0.5 — borderline under L-scaling: r_2_scaled =
        # L · 0.5 = L/2 fails the strict `r_2 < L/2` check in `check_mms_parameters`).
        "domain" => Dict(
            "alpha_0" => 0.1,
            "bounding_box" => test_dict["domain"]["bounding_box"],
            "r_1" => Float64(get(test_dict["domain"], "r_1", 0.2)),
            "r_2" => Float64(get(test_dict["domain"], "r_2", 0.4)),
        ),
        "numerical_method" => Dict(
            "element_spaces" => Dict("k_velocity" => 1, "k_pressure" => 1),
            "mesh" => Dict("element_type" => "TRI", "partition" => [10, 10]),
            "stabilization" => Dict("method" => "ASGS"),
            "solver" => get(get(test_dict, "numerical_method", Dict()), "solver", Dict())
        )
    )
    
    config = PorousNSSolver.load_config_from_dict(config_dict)
    
    # Mathematical constants tied precisely to the exact analytical MMS domain definition
    U_amp = 2.5
    L = 3.0
    alpha_infty = 0.85
    
    # 2. Invoke the exact state assembly closures used by run_test.jl with local dynamically tracking physical dimension variables.
    # `build_porosity_field` accepts an L positional argument (default 1.0) since the 2026-06-01
    # generalisation; passing L here couples the porosity radii r_1, r_2 to the L-scaled domain.
    alpha_field = build_porosity_field(config, 0.1, alpha_infty, L)
    form = build_mms_formulation(config, Da_bound, Re_bound, U_amp, L, alpha_infty)
    mms = PorousNSSolver.Paper2DMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)

    # 3. Retrieve Characteristic scale bindings
    U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)

    @test U_c ≈ U_amp

    # Internal logic backward validation
    nu_expected = U_amp * L / Re_bound

    # Check that dimensionless equivalence matches the exact math formula from Eq 6.2 dimensionlessly:
    # P_c = (1.0 + Re + Da) * U * nu / L
    P_theoretical = (1.0 + Re_bound + Da_bound) * U_c * nu_expected / L
    @test P_c ≈ P_theoretical

    # 4. Check that evaluate values correspond purely to these amplitudes.
    # The polynomial frequencies in u_ex and p_ex are now π/L instead of π (2026-06-01
    # generalisation), so we evaluate at L-scaled dimensionless coordinates to recover the
    # same trigonometric arguments as the L=1 reference (i.e. sin(π·0.5) = 1, etc.).
    u_ex_func = PorousNSSolver.get_u_ex(mms)
    p_ex_func = PorousNSSolver.get_p_ex(mms)

    # At dimensionless position (0.5, 0.5) — i.e. physical position (L/2, L/2) on the L-scaled
    # domain — `sin(π·(L/2)/L)·sin(π·(L/2)/L) = sin(π/2)² = 1`. The porosity bump's radial
    # transition lives between L·r_1 and L·r_2 (built above); at (L/2, L/2) the radius
    # √(L²/2) = L/√2 > L·r_2 in general so we expect α(x_center) ≈ α_∞.
    x_center = VectorValue(0.5 * L, 0.5 * L)
    u_val = u_ex_func(x_center)
    @test u_val[1] ≈ (0.1 / alpha_infty) * U_amp
    @test isapprox(u_val[2], 0.0, atol=1e-15)

    # For pressure: cos(π·x/L)·sin(π·y/L). At (L/4, L/4), cos(π/4)·sin(π/4) = 0.5.
    # So P_ex(L/4, L/4) should be EXACTLY 0.5 * P_c.
    x_quarter = VectorValue(0.25 * L, 0.25 * L)
    p_val = p_ex_func(x_quarter)
    @test p_val ≈ 0.5 * P_c
end
