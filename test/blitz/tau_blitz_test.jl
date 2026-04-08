# ==============================================================================================
# Nature & Intent:
# Verifies the mathematical integrity of the Variational Multiscale (VMS) stabilization parameters 
# ($\tau_1, \tau_2$) and their exact linearizations via finite difference comparisons. Protects against
# sign errors or missing chain-rule cross terms in stabilization exactness limits.
#
# Mathematical Formulation Alignment:
# Enforces the core mandate of `ExactNewtonMode`. Proves that continuous state abstractions 
# (KinematicState, MediumState) evaluate accurately and their chain-ruled derivatives match numerical
# approximations without simplifying components (such as dropping convective velocity variation effects).
#
# Associated Files / Functions:
# - `src/formulations/tau.jl` (`compute_tau_1`, `compute_tau_2`, `compute_dtau_1_du`, `compute_dtau_2_du`)
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

@testset "fast: tau and dtau" begin
    @testset "tau positivity" begin
        kin, med, du = make_local_states()
        law = PorousNSSolver.ConstantSigmaLaw(2.0)

        τ1 = PorousNSSolver.compute_tau_1(kin, med, 1e-2, 4.0, 2.0, 1e-8, law)
        τ2 = PorousNSSolver.compute_tau_2(kin, med, 1e-2, 4.0, 2.0, 1e-8)

        @test isfinite(τ1) && τ1 > 0
        @test isfinite(τ2) && τ2 > 0
    end

    @testset "dtau_1 matches finite differences: constant sigma" begin
        u0 = [0.2, -0.1]
        du0 = [0.3, 0.4]
        grad_u = TensorValue(0.1, 0.2, -0.3, 0.4)
        med = PorousNSSolver.MediumState(0.7, VectorValue(0.05,-0.02), 0.25)
        law = PorousNSSolver.ConstantSigmaLaw(2.0)

        f(uvec) = begin
            u = VectorValue(uvec...)
            mag_u = sqrt(u⋅u + 1e-8)
            kin = PorousNSSolver.KinematicState(u, grad_u, mag_u)
            PorousNSSolver.compute_tau_1(kin, med, 1e-2, 4.0, 2.0, 1e-8, law)
        end

        mag_u = sqrt(sum(abs2, u0) + 1e-8)
        kin = PorousNSSolver.KinematicState(VectorValue(u0...), grad_u, mag_u)
        dτ = PorousNSSolver.compute_dtau_1_du(kin, med, VectorValue(du0...), 1e-2, 4.0, 2.0, 1e-8, false, law)
        dτ_fd = directional_fd(f, u0, du0)

        @test isapprox(dτ, dτ_fd; rtol=1e-5, atol=1e-7)
    end

    @testset "dtau_1 matches finite differences: Forchheimer" begin
        u0 = [0.2, -0.1]
        du0 = [0.3, 0.4]
        grad_u = TensorValue(0.1, 0.2, -0.3, 0.4)
        med = PorousNSSolver.MediumState(0.7, VectorValue(0.05,-0.02), 0.25)
        law = PorousNSSolver.ForchheimerErgunLaw(2.0, 4.0)

        f(uvec) = begin
            u = VectorValue(uvec...)
            mag_u = sqrt(u⋅u + 1e-8)
            kin = PorousNSSolver.KinematicState(u, grad_u, mag_u)
            PorousNSSolver.compute_tau_1(kin, med, 1e-2, 4.0, 2.0, 1e-8, law)
        end

        mag_u = sqrt(sum(abs2, u0) + 1e-8)
        kin = PorousNSSolver.KinematicState(VectorValue(u0...), grad_u, mag_u)
        dτ = PorousNSSolver.compute_dtau_1_du(kin, med, VectorValue(du0...), 1e-2, 4.0, 2.0, 1e-8, false, law)
        dτ_fd = directional_fd(f, u0, du0)

        @test isapprox(dτ, dτ_fd; rtol=1e-5, atol=1e-7)
    end

    @testset "dtau_2 matches finite differences" begin
        u0 = [0.2, -0.1]
        du0 = [0.3, 0.4]
        grad_u = TensorValue(0.1, 0.2, -0.3, 0.4)
        med = PorousNSSolver.MediumState(0.7, VectorValue(0.05,-0.02), 0.25)

        f(uvec) = begin
            u = VectorValue(uvec...)
            mag_u = sqrt(u⋅u + 1e-8)
            kin = PorousNSSolver.KinematicState(u, grad_u, mag_u)
            PorousNSSolver.compute_tau_2(kin, med, 1e-2, 4.0, 2.0, 1e-8)
        end

        mag_u = sqrt(sum(abs2, u0) + 1e-8)
        kin = PorousNSSolver.KinematicState(VectorValue(u0...), grad_u, mag_u)
        dτ = PorousNSSolver.compute_dtau_2_du(kin, med, VectorValue(du0...), 1e-2, 4.0, 2.0, 1e-8, false)
        dτ_fd = directional_fd(f, u0, du0)

        @test isapprox(dτ, dτ_fd; rtol=1e-5, atol=1e-7)
    end

    @testset "freeze_cusp zeroes derivatives" begin
        kin, med, du = make_local_states()
        law = PorousNSSolver.ConstantSigmaLaw(1.0)

        @test PorousNSSolver.compute_dtau_1_du(kin, med, du, 1e-2, 4.0, 2.0, 1e-8, true, law) == 0.0 * (kin.u ⋅ du)
        @test PorousNSSolver.compute_dtau_2_du(kin, med, du, 1e-2, 4.0, 2.0, 1e-8, true) == 0.0 * (kin.u ⋅ du)
    end
end
