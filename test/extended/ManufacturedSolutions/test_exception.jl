import Pkg
Pkg.activate("../../../")

using JSON3
using Gridap

include("../../../src/PorousNSSolver.jl")

# 1) Formulate 
config_path = "data/small_test_config.json"
test_dict = JSON3.read(read(config_path, String), Dict{String, Any})

config_dict = Dict(
    "physical_properties" => Dict(
        "nu" => 0.1, 
        "eps_val" => 1e-8, 
        "reaction_model" => "Constant_Sigma", 
        "sigma_constant" => 1.0,
        "u_base_floor_ref" => 1.0,
        "epsilon_floor" => 0.1,
        "tau_regularization_limit" => 1e-6
    ),
    "domain" => Dict(
        "alpha_0" => 0.4, 
        "bounding_box" => [0, 1, 0, 1],
        "r_1" => 0.1,
        "r_2" => 0.4
    ),
    "numerical_method" => Dict(
        "element_spaces" => Dict("k_velocity" => 2, "k_pressure" => 2),
        "mesh" => Dict("element_type" => "QUAD", "partition" => [4, 4]),
        "stabilization" => Dict("method" => "ASGS")
    )
)

config = PorousNSSolver.load_config_from_dict(config_dict)

Da, Re, U_amp, L, alpha_infty = 1.0, 10.0, 1.0, 1.0, 1.0
nu_calculated = 0.1
eps_calculated = 1e-8
rxn = PorousNSSolver.ConstantSigmaLaw(1.0)
proj = PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma()
reg = PorousNSSolver.SmoothVelocityFloor(1.0, 0.0, 0.1)

# WE FORCE DEVIATORIC HERE
form = PorousNSSolver.PaperGeneralFormulation(PorousNSSolver.DeviatoricSymmetricViscosity(), rxn, proj, reg, nu_calculated, eps_calculated, 0.0)

alpha_field = PorousNSSolver.SmoothRadialPorosity(0.4, 1.0, 0.1, 0.4)
mms = PorousNSSolver.Paper2DMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)

c_1, c_2 = 4.0*16, 2.0*4
tau_reg_lim = 1e-6

# Should trigger ERROR exactly when building the definitions
try
    model = CartesianDiscreteModel((0, 1, 0, 1), (2, 2))
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    h_cf = CellField(x -> 0.1, Ω)
    
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
    refe_p = ReferenceFE(lagrangian, Float64, 2)
    V = TestFESpace(model, refe_u, conformity=:H1)
    Q = TestFESpace(model, refe_p, conformity=:H1)
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([TrialFESpace(V), TrialFESpace(Q)])

    println("Attempting to evaluate MMS ...")
    f, g = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)
    
    # Evaluate at x=(0.5, 0.5)
    f(Point(0.5, 0.5))
    println("FAILED: Should have thrown error!")
catch e
    println("SUCCESS: Caught error flawlessly!")
    println(e)
end
