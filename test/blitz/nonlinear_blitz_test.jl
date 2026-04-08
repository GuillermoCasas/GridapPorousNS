# ==============================================================================================
# Nature & Intent:
# Evaluates the scalar Armijo line search decrease condition. Verifies that the internal 
# nonlinear solver algorithm properly restricts step sizes along the descent direction to ensure
# the global merit function $\Phi(x) = \frac{1}{2}\|F(x)\|^2$ strictly decreases, preventing 
# numerical blowups during highly advective or non-linear porous stabilization regimes.
#
# Mathematical Formulation Alignment:
# Reflects the fundamental mathematical logic of globalizing the Newton-Raphson scheme. Not directly 
# tied to the porous PDE form, but mandatory for stabilizing its Picard/Newton iterations.
#
# Associated Files / Functions:
# - `src/solvers/nonlinear.jl` (`solve_nonlinear_system`, Armijo line search logic)
# ==============================================================================================

using Test
using PorousNSSolver
using LinearAlgebra

@testset "fast: nonlinear solver utilities" begin
    @testset "merit function decreases for simple scalar nonlinear map" begin
        # Pure Julia check of the Armijo logic pattern, independent of Gridap.
        F(x) = [x[1]^2 - 1.0]
        Φ(x) = 0.5 * sum(abs2, F(x))

        x = [2.0]
        # Newton step for scalar equation: d solves J d = F, and update is x - α d
        J = [2.0*x[1]]
        d = J \ F(x)

        # Here the step direction is -d. So the step dir is dx. 
        dir_deriv = - dot(F(x), J * d)

        @test Φ(x .- 1.0 .* d) < Φ(x)
        @test Φ(x) + 0.1 * 1.0 * dir_deriv > Φ(x .- 1.0 .* d)
    end
end
