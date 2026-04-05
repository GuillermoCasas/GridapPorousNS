using Test
using PorousNSSolver
using Gridap
using Gridap.Algebra
using LinearAlgebra

include("test_fast_utils.jl")
include("test_fast_reaction_regularization.jl")
include("test_fast_tau.jl")
include("test_fast_projection.jl")
include("test_fast_viscous_operators.jl")
include("test_fast_formulation_smoke.jl")
include("test_fast_formulation_consistency.jl")
include("test_fast_nonlinear.jl")
