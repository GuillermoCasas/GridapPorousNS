module PorousNSSolver

using Gridap
using GridapGmsh
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
include("solvers/linear_solvers.jl")
include("solvers/nonlinear.jl")
include("solvers/accelerators.jl")   # available option (not yet wired): Anderson acceleration for the slow OSGS coupled iterations
include("solvers/asgs_solver.jl")    # shared solver core + ASGS Stage-I boot + orchestrator (defines FETopology/VMSFormulation + solve_system)
include("solvers/osgs_solver.jl")    # OSGS projections + solve_osgs_stage! — MUST follow asgs_solver.jl (its signature names those structs, resolved at parse time)
include("solvers/mms_verification.jl")   # MMS plateau verifier (Algorithm D); references the SolutionVerifier seam, so it MUST come after asgs_solver.jl
include("metrics.jl")

# Diagnostics and Problems
include("problems/mms_paper_2d.jl")
include("io.jl")
include("run_simulation.jl")

export run_simulation
export load_config
export PorousNSConfig
export compute_reference_errors
export compute_reference_errors_multimask
export compute_trial_projection_errors
export compute_mode_decomposition
export compute_corner_excluded_norm
export ILUGMRESSolver
export AndersonAccelerator   # available option for accelerating the slow OSGS coupled fixed-point (not yet wired into the solver)

end # module
