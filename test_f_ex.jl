using Pkg; Pkg.activate(".")
using Gridap
using PorousNSSolver

base_config = PorousNSSolver.load_config("tests/ManufacturedSolutions/data/test_config.json")
config = PorousNSSolver.PorousNSConfig(
    phys=base_config.phys,
    mesh=PorousNSSolver.MeshConfig(domain=[0.0, 2.0, 0.0, 2.0], partition=[10, 10]),
    output=PorousNSSolver.OutputConfig(directory="tests/ManufacturedSolutions/results", basename="mms_10")
)
include("tests/ManufacturedSolutions/run_test.jl")
