using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Test
using PorousNSSolver
using Gridap
using Gridap.Algebra
using LinearAlgebra

include("utils_fast_test.jl")
include("reaction_regularization_fast_test.jl")
include("tau_fast_test.jl")
include("projection_fast_test.jl")
include("viscous_operators_fast_test.jl")
include("formulation_smoke_fast_test.jl")
include("formulation_consistency_fast_test.jl")
include("nonlinear_fast_test.jl")
include("mms_exactness_fast_test.jl")
