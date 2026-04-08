using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using PorousNSSolver
using Gridap
using Gridap.Algebra
using LinearAlgebra

include("test_utils.jl")

files = [
    "formulation_consistency_quick_test.jl",
    "formulation_smoke_quick_test.jl"
]

@testset "Quick Tests (6s - 2m)" begin
    for file in files
        path = joinpath(@__DIR__, "quick", file)
        t = @elapsed include(path)
        if t < 5.0
            @warn "Test file $file took $(round(t, digits=2))s (faster than 5.0s). Consider moving it to the 'blitz' category."
        elseif t > 130.0
            @warn "Test file $file took $(round(t, digits=2))s (exceeds Quick category limit of 120.0s, accounting for compilation margins). Consider moving it to the 'extended' category."
        else
            println("$(file) completed in $(round(t, digits=2))s")
        end
    end
end
