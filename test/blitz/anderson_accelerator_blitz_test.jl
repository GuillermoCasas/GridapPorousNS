# ==============================================================================================
# Nature & Intent:
# Unit test for the Anderson accelerator (src/solvers/accelerators.jl), previously dead + untested
# (NONL-03). Exercises update! on a known contractive affine fixed point G(x) = A x + b: it must reach
# the exact fixed point, degrade to a relaxed-Picard step on the bootstrap iteration, and not be slower
# than plain Picard. This guards the accelerator now that it is wired into the OSGS stage
# (osgs_solver.jl, opt-in via osgs_anderson_enabled).
# ==============================================================================================

using Test
using PorousNSSolver
using LinearAlgebra

@testset "fast: Anderson accelerator (NONL-03)" begin
    # Contractive affine map: G(x) = A x + b with spectral radius(A) < 1, fixed point x* = (I-A)^{-1} b.
    A = [0.5 0.1; -0.2 0.4]
    b = [1.0, -2.0]
    G(x) = A * x + b
    x_star = (I - A) \ b

    @testset "converges to the fixed point" begin
        acc = PorousNSSolver.AndersonAccelerator(5, 1.0, 1e6)   # depth 5, β = 1, loose safety
        x = [0.0, 0.0]
        res = Inf
        for k in 1:50
            x = PorousNSSolver.update!(acc, x, G(x))
            res = norm(G(x) - x, Inf)
            res < 1e-12 && break
        end
        @test res < 1e-10
        @test isapprox(x, x_star; atol = 1e-9)
    end

    @testset "bootstrap step is a relaxed-Picard step" begin
        acc = PorousNSSolver.AndersonAccelerator(5, 0.7, 1e6)
        x0 = [0.3, 0.5]
        g0 = G(x0)
        x1 = PorousNSSolver.update!(acc, x0, g0)   # no history yet ⇒ x + β·(g - x)
        @test x1 ≈ x0 .+ 0.7 .* (g0 .- x0)
    end

    @testset "not slower than plain Picard" begin
        picard_iters = let x = [0.0, 0.0], n = 0
            for k in 1:500
                x = G(x); n += 1
                norm(G(x) - x, Inf) < 1e-10 && break
            end
            n
        end
        anderson_iters = let acc = PorousNSSolver.AndersonAccelerator(5, 1.0, 1e6), x = [0.0, 0.0], n = 0
            for k in 1:500
                x = PorousNSSolver.update!(acc, x, G(x)); n += 1
                norm(G(x) - x, Inf) < 1e-10 && break
            end
            n
        end
        @test anderson_iters <= picard_iters
    end
end
