# test/fast/test_fast_mms_exactness.jl
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
        "domain" => Dict("alpha_0" => 0.1, "bounding_box" => test_dict["domain"]["bounding_box"]),
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
    
    # 2. Invoke the exact state assembly closures used by run_test.jl with local dynamically tracking physical dimension variables 
    alpha_field = build_porosity_field(config, 0.1, alpha_infty)
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
    
    # 4. Check that evaluate values correspond purely to these amplitudes
    u_ex_func = PorousNSSolver.get_u_ex(mms)
    p_ex_func = PorousNSSolver.get_p_ex(mms)
    
    # Center of square is (0.5, 0.5). For u_ex: sin(pi/2)*sin(pi/2) = 1.
    # At (0.5, 0.5), radial porosity bump is fully evaluated at its algorithmic maximum of alpha_infty. 
    # Because alpha_0 is dynamic, we expect u(0.5, 0.5) to be [U_amp * alpha_0 / alpha_infty, 0]
    
    x_center = VectorValue(0.5, 0.5)
    u_val = u_ex_func(x_center)
    @test u_val[1] ≈ (0.1 / alpha_infty) * U_amp
    @test isapprox(u_val[2], 0.0, atol=1e-15)
    
    # For pressure: cos(pi*x)*sin(pi*y). At (0.25, 0.25), cos(pi/4)*sin(pi/4) = 0.5
    # So P_ex(0.25, 0.25) should be EXACTLY 0.5 * P_c
    x_quarter = VectorValue(0.25, 0.25)
    p_val = p_ex_func(x_quarter)
    @test p_val ≈ 0.5 * P_c
end
