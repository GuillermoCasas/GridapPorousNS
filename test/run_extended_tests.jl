using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test

# Include test entry points for extended suites
files = [
    "utilities_extended_test.jl",
]

@testset "Extended Tests (> 2m)" begin
    for file in files
        path = joinpath(@__DIR__, "extended", file)
        t = @elapsed include(path)
        if t < 120.0
            @warn "Test file $file took $(round(t, digits=2))s (faster than 120.0s). Consider moving it to the 'quick' category."
        else
            println("$(file) completed in $(round(t, digits=2))s")
        end
    end
end
