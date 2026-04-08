using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

@testset "fast: reaction and regularization" begin
    @testset "ConstantSigmaLaw value and derivative" begin
        kin, med, du = make_local_states()
        law = PorousNSSolver.ConstantSigmaLaw(3.5)
        @test PorousNSSolver.sigma(law, kin, med, kin.mag_u) == 3.5
        @test PorousNSSolver.dsigma_du(law, kin, med, kin.mag_u, du) == 0.0 * (kin.u ⋅ du)
    end

    @testset "ForchheimerErgunLaw value" begin
        kin, med, du = make_local_states()
        law = PorousNSSolver.ForchheimerErgunLaw(2.0, 4.0)

        α = med.alpha
        expected = 2.0*((1-α)/α)^2 + 4.0*((1-α)/α)*kin.mag_u
        @test isapprox(PorousNSSolver.sigma(law, kin, med, kin.mag_u), expected; rtol=1e-12, atol=1e-12)
    end

    @testset "ForchheimerErgunLaw directional derivative" begin
        u0 = [0.2, -0.1]
        du0 = [0.3, 0.4]
        grad_u = TensorValue(0.1, 0.2, -0.3, 0.4)
        med = PorousNSSolver.MediumState(0.7, VectorValue(0.05,-0.02), 0.25)
        law = PorousNSSolver.ForchheimerErgunLaw(2.0, 4.0)

        f(uvec) = begin
            u = VectorValue(uvec...)
            mag_u = sqrt(u⋅u + 1e-8)
            kin = PorousNSSolver.KinematicState(u, grad_u, mag_u)
            PorousNSSolver.sigma(law, kin, med, mag_u)
        end

        mag_u = sqrt(sum(abs2, u0) + 1e-8)
        kin = PorousNSSolver.KinematicState(VectorValue(u0...), grad_u, mag_u)
        dσ = PorousNSSolver.dsigma_du(law, kin, med, mag_u, VectorValue(du0...))
        dσ_fd = directional_fd(f, u0, du0)
        @test isapprox(dσ, dσ_fd; rtol=1e-5, atol=1e-7)
    end

    @testset "NoRegularization and SmoothVelocityFloor" begin
        u = VectorValue(0.0, 0.0)
        ν = 1e-2
        h = 0.1
        c1 = 4.0
        c2 = 2.0

        @test PorousNSSolver.effective_speed(PorousNSSolver.NoRegularization(), VectorValue(3.0,4.0), ν, h, c1, c2) ≈ 5.0

        reg = PorousNSSolver.SmoothVelocityFloor(1e-3, 0.5, 1e-8)
        val = PorousNSSolver.effective_speed(reg, u, ν, h, c1, c2)
        @test isfinite(val)
        @test val > 0.0
    end
end
