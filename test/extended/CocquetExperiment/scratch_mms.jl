using Pkg
Pkg.activate("../../..")
using PorousNSSolver

include("../ManufacturedSolutions/run_test.jl")

println("Running MMS Test Script (Bypassing HDF5)")
config_path = "../ManufacturedSolutions/data/small_test_config.json"
PorousNSSolver.run_mms_convergence(config_path, HDF5_FILE_PATH="dummy.h5")
