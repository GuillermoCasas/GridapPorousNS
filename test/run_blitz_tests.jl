using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using PorousNSSolver
using Gridap
using Gridap.Algebra
using LinearAlgebra

include("test_utils.jl")

files = [
    "reaction_regularization_blitz_test.jl",
    "viscous_operators_blitz_test.jl",
    "tau_blitz_test.jl",
    "projection_blitz_test.jl",
    "nonlinear_blitz_test.jl",
    "mms_exactness_blitz_test.jl"
]

@testset "Blitz Tests (< 5s)" begin
    for file in files
        path = joinpath(@__DIR__, "blitz", file)
        t = @elapsed include(path)
        if t > 5.0
            @warn "Test file $file took $(round(t, digits=2))s (exceeds Blitz category limit of 5.0s). Consider moving it to the 'quick' category."
        else
            println("$(file) completed in $(round(t, digits=2))s")
        end
    end
end
