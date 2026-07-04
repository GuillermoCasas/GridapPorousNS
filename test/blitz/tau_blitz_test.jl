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
        dτ = PorousNSSolver.compute_dtau_1_du(kin, med, VectorValue(du0...), 1e-2, 4.0, 2.0, 1e-8, false, law, PorousNSSolver.VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR)
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
        dτ = PorousNSSolver.compute_dtau_1_du(kin, med, VectorValue(du0...), 1e-2, 4.0, 2.0, 1e-8, false, law, PorousNSSolver.VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR)
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
        dτ = PorousNSSolver.compute_dtau_2_du(kin, med, VectorValue(du0...), 1e-2, 4.0, 2.0, 1e-8, false, PorousNSSolver.VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR)
        dτ_fd = directional_fd(f, u0, du0)

        @test isapprox(dτ, dτ_fd; rtol=1e-5, atol=1e-7)
    end

    @testset "freeze_cusp zeroes derivatives" begin
        kin, med, du = make_local_states()
        law = PorousNSSolver.ConstantSigmaLaw(1.0)

        @test PorousNSSolver.compute_dtau_1_du(kin, med, du, 1e-2, 4.0, 2.0, 1e-8, true, law, PorousNSSolver.VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR) == 0.0 * (kin.u ⋅ du)
        @test PorousNSSolver.compute_dtau_2_du(kin, med, du, 1e-2, 4.0, 2.0, 1e-8, true, PorousNSSolver.VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR) == 0.0 * (kin.u ⋅ du)
    end

    @testset "Tau1/Tau2 simplified paper form is intentional [P-001, P-008]" begin
        # Locks in the simplifications that drop:
        #   - the `ε·h²` term from τ₂ (paper eq:Tau2 → eq:Tau2Final, article.tex L755/L778)
        #   - the porosity-gradient `(h/|k_0|)|∇α|` contribution to C_α in τ₁
        #     (paper eq:Tau1 → eq:Tau1Final, article.tex L754/L777)
        # These are intentional per article.tex L762 and L764–768; the assumptions
        # are documented in `docs/solver/paper-code-divergences.md`. The two assertions
        # below are the regression anchor — a future audit that re-raises P-001 or
        # P-008 should fail this test rather than mis-identifying a bug.
        u = VectorValue(0.2, -0.1)
        grad_u = TensorValue(0.1, 0.2, -0.3, 0.4)
        mag_u = sqrt(u ⋅ u + 1e-8)
        kin = PorousNSSolver.KinematicState(u, grad_u, mag_u)
        law = PorousNSSolver.ConstantSigmaLaw(2.0)
        h = 0.25
        alpha = 0.7

        # P-001: compute_tau_2 must NOT take physical_epsilon. Varying the surrounding ε value
        # (passed at call sites unrelated to τ₂) cannot affect τ₂ because it's not a
        # parameter of the function — this is a structural lock.
        med = PorousNSSolver.MediumState(alpha, VectorValue(0.05, -0.02), h)
        τ2 = PorousNSSolver.compute_tau_2(kin, med, 1e-2, 4.0, 2.0, 1e-8)
        @test isfinite(τ2) && τ2 > 0
        # The full eq:Tau2 would yield h²/(c₁·α·τ_NS + ε·h² + reg); the simplified
        # eq:Tau2Final yields h²/(c₁·α·τ_NS + reg). Numerically reconstruct the
        # simplified denominator and confirm code matches.
        c1, c2 = 4.0, 2.0
        tau_reg = 1e-8
        nu = 1e-2
        tau_ns_inv = c1 * nu / h^2 + c2 * mag_u / h + tau_reg
        tau_ns = 1.0 / tau_ns_inv
        expected_tau2_final = h^2 / (c1 * alpha * tau_ns + tau_reg)
        @test isapprox(τ2, expected_tau2_final; rtol=1e-12)

        # P-008: compute_tau_1 must not depend on med.grad_alpha when the local α is
        # held fixed. Vary grad_alpha by an order of magnitude and confirm τ₁ is bit-
        # identical. (The full eq:Tau1 would add (h/|k_0|)|∇α| inside C_α; the
        # simplified eq:Tau1Final drops it.)
        med_small_grad = PorousNSSolver.MediumState(alpha, VectorValue(1e-6, 1e-6), h)
        med_large_grad = PorousNSSolver.MediumState(alpha, VectorValue(10.0, 10.0), h)
        τ1_small = PorousNSSolver.compute_tau_1(kin, med_small_grad, nu, c1, c2, tau_reg, law)
        τ1_large = PorousNSSolver.compute_tau_1(kin, med_large_grad, nu, c1, c2, tau_reg, law)
        @test τ1_small == τ1_large
    end
end
