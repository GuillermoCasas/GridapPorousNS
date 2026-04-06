using Test

@testset "PorousNSSolver" begin
    include("fast/runtests_fast.jl")
    include("long/runtests.jl")
end
