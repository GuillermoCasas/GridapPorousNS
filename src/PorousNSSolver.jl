module PorousNSSolver

using Gridap
using JSON3
using StructTypes
using WriteVTK
using HDF5

# Core
include("config.jl")

# Models
include("models/porosity.jl")
include("models/regularization.jl")
include("models/reaction.jl")

# Formulations
include("formulations/gridap_extensions.jl")
include("formulations/viscous_operators.jl")
include("stabilization/projection.jl")
include("formulations/continuous_problem.jl")

# Stabilization
include("stabilization/tau.jl")

# Solvers
include("solvers/nonlinear.jl")
include("solvers/accelerators.jl")
include("solvers/porous_solver.jl")
include("metrics.jl")

# Diagnostics and Problems
include("problems/mms_paper_2d.jl")
include("io.jl")
include("run_simulation.jl")

export run_simulation
export load_config
export PorousNSConfig
export compute_reference_errors

end # module
