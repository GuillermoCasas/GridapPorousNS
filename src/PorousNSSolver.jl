module PorousNSSolver

using Gridap
using JSON
using WriteVTK
using HDF5
using Parameters

include("config.jl")
include("geometry_mesh.jl")
include("formulation.jl")
include("io.jl")
include("run_simulation.jl")

export run_simulation
export load_config
export PorousNSConfig

end # module
