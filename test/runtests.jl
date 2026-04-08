using Test

@testset "PorousNSSolver Full Test Suite" begin
    println("\n==================================")
    println("Running Blitz Tests (< 5s)")
    println("==================================")
    include("run_blitz_tests.jl")

    println("\n==================================")
    println("Running Quick Tests (6s - 2m)")
    println("==================================")
    include("run_quick_tests.jl")

    println("\n==================================")
    println("Running Extended Tests (> 2m)")
    println("==================================")
    include("run_extended_tests.jl")
end
