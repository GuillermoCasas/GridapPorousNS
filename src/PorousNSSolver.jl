module PorousNSSolver

using Gridap
using JSON
using WriteVTK
using HDF5
using Parameters

# Core
include("config.jl")
include("geometry_mesh.jl")

# Models
include("models/porosity.jl")
include("models/regularization.jl")
include("models/reaction.jl")

# Formulations
include("formulations/viscous_operators.jl")
include("stabilization/projection.jl")
include("formulations/continuous_problem.jl")

# Stabilization
include("stabilization/tau.jl")

# Solvers
include("solvers/nonlinear.jl")
include("solvers/porous_solver.jl")

# Diagnostics and Problems
include("problems/mms_paper_2d.jl")
include("io.jl")
include("run_simulation.jl")

export run_simulation
export load_config
export PorousNSConfig

end # module
