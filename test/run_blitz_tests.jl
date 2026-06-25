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

    "formulation_consistency_blitz_test.jl",
    "picard_jacobian_equivalence_blitz_test.jl",
    "tau_blitz_test.jl",
    "porosity_hessian_blitz_test.jl",
    "projection_blitz_test.jl",
    "nonlinear_blitz_test.jl",
    "mms_exactness_blitz_test.jl",
    "mms_unit_length_blitz_test.jl",
    "config_strict_loader_blitz_test.jl",
    "config_validation_blitz_test.jl",
    "cascade_policy_symmetry_blitz_test.jl",
    "stall_guard_blitz_test.jl",
    "pingpong_schedule_blitz_test.jl",
    "cocquet_modified_corner_topology_blitz_test.jl"
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
