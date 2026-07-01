# ==============================================================================================
# Nature & Intent:
# Pins the scale-free outer-iteration convergence criterion (src/solvers/convergence_criterion.jl,
# spec docs/solver/nonlinear-convergence-criterion-prompt.md). Checks the load-bearing mathematical
# properties. STRONG-FORM DIAGNOSTIC (`mass_criterion`, no longer the gate): the ε_C^strong ≤ √d ceiling
# and its tightness for pure dilatation; → 0 for divergence-free flow; scale-freeness (u ↦ c·u invariant);
# the constant-α reduction ε_C^strong = ‖∇·u‖/‖∇u‖. PHILOSOPHY-A GATE MEASURES (both → 0 at the discrete
# solution): the momentum term-envelope breakdown and its u=0 limit; the SYMMETRIC mass term-envelope
# (`mass_force_envelope`); ε_M = ‖r_M‖/D_M, ε_C = ‖r_C‖/D_C and the converged verdict; and the
# degenerate-state (underflow floor) guard. Lives in the Quick tier (a few FE assembly passes push it
# just past the 5 s Blitz bound).
#
# Associated Files / Functions:
# - src/solvers/convergence_criterion.jl (`evaluate_convergence`, `mass_criterion` [diagnostic],
#   `momentum_force_envelope`, `mass_force_envelope`)
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

const _PNS = PorousNSSolver

@testset "quick: scale-free convergence criterion" begin
    model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (4, 4))
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1))
    U = TrialFESpace(V)
    Q = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1))
    P = TrialFESpace(Q)

    αc   = interpolate(x -> 0.7, P)                       # constant porosity ⇒ ∇α = 0
    σc   = interpolate(x -> 1.5, P)                       # constant reaction coefficient field
    fc   = interpolate(x -> VectorValue(1.0, 0.5), U)     # nonzero body force
    visc = _PNS.DeviatoricSymmetricViscosity()
    ν    = 0.3
    d    = 2
    p0   = interpolate(x -> 0.0, P)
    gz   = interpolate(x -> 0.0, P)                       # zero mass source g (unforced ⇒ ε_C = pure-div ratio)

    @testset "diagnostic ε_C^strong ≤ √d, bound achieved by pure dilatation u=(x,y)" begin
        uh = interpolate(x -> VectorValue(x[1], x[2]), U)         # ∇u = I, ∇·u = 2
        eC, num, den, _ = _PNS.mass_criterion(uh, p0, αc, 0.0, gz, d, dΩ)
        @test eC ≤ sqrt(d) * (1 + 1e-6)
        @test isapprox(eC, sqrt(2.0); rtol = 1e-6)               # 2/√2 = √2, α cancels
    end

    @testset "diagnostic ε_C^strong ≈ 0 for divergence-free shear" begin
        uh = interpolate(x -> VectorValue(x[2], 0.0), U)         # ∇·u = 0
        eC, _, _, _ = _PNS.mass_criterion(uh, p0, αc, 0.0, gz, d, dΩ)
        @test eC < 1e-8
    end

    @testset "diagnostic ε_C^strong is scale-free (u ↦ 10u invariant)" begin
        g(x)  = VectorValue(x[1]*x[2], -0.5*x[2]^2 + 0.3*x[1])
        e1, _, _, _  = _PNS.mass_criterion(interpolate(g, U), p0, αc, 0.0, gz, d, dΩ)
        e10, _, _, _ = _PNS.mass_criterion(interpolate(x -> 10.0*g(x), U), p0, αc, 0.0, gz, d, dΩ)
        @test isapprox(e1, e10; rtol = 1e-8)
    end

    @testset "diagnostic ε_C^strong == ‖∇·u‖/‖∇u‖ (constant-α reduction)" begin
        uh = interpolate(x -> VectorValue(x[1]^2, x[1]*x[2]), U)
        eC, _, _, _ = _PNS.mass_criterion(uh, p0, αc, 0.0, gz, d, dΩ)
        divu  = sqrt(sum(∫((∇ ⋅ uh) * (∇ ⋅ uh))dΩ))
        gradu = sqrt(sum(∫(∇(uh) ⊙ ∇(uh))dΩ))
        @test isapprox(eC, divu / gradu; rtol = 1e-6)
    end

    @testset "momentum envelope: terms ≥ 0, sum to D_M, and u=0 ⇒ only body force" begin
        uh = interpolate(x -> VectorValue(x[1], x[2]), U)
        ph = interpolate(x -> x[1], P)
        D_M, t = _PNS.momentum_force_envelope(uh, ph, αc, ν, visc, σc, fc, V, dΩ)
        @test D_M > 0
        @test all(≥(0.0), values(t))
        @test isapprox(D_M, t.convection + t.viscous + t.pressure_grad + t.resistance + t.body_force; rtol = 1e-10)

        u0 = interpolate(x -> VectorValue(0.0, 0.0), U)
        D0, t0 = _PNS.momentum_force_envelope(u0, p0, αc, ν, visc, σc, fc, V, dΩ)
        @test t0.convection < 1e-12 && t0.viscous < 1e-12 && t0.pressure_grad < 1e-12 && t0.resistance < 1e-12
        @test t0.body_force > 0
        @test isapprox(D0, t0.body_force; rtol = 1e-10)
    end

    @testset "mass envelope (Philosophy A): terms ≥ 0, sum to D_C, u=0,ε=0,g=0 ⇒ D_C = 0" begin
        uh = interpolate(x -> VectorValue(x[1], x[2]), U)         # ∇·u = 2 ⇒ nonzero divergence term
        ph = interpolate(x -> x[1], P)
        D_C, t = _PNS.mass_force_envelope(uh, ph, αc, 0.5, gz, Q, dΩ)   # ε=0.5 ⇒ nonzero penalty term
        @test D_C > 0
        @test all(≥(0.0), values(t))
        @test isapprox(D_C, t.divergence + t.penalty + t.source; rtol = 1e-10)
        @test t.divergence > 0 && t.penalty > 0
        @test t.source < 1e-12                                   # g = 0

        u0 = interpolate(x -> VectorValue(0.0, 0.0), U)
        D0, t0 = _PNS.mass_force_envelope(u0, p0, αc, 0.0, gz, Q, dΩ) # u=0, ε=0, g=0 ⇒ all terms 0
        @test t0.divergence < 1e-12 && t0.penalty < 1e-12 && t0.source < 1e-12
        @test D0 < 1e-12
    end

    @testset "evaluate_convergence: ε_M=‖r_M‖/D_M, ε_C=‖r_C‖/D_C (symmetric) and verdict" begin
        uh = interpolate(x -> VectorValue(x[1], x[2]), U)
        ph = interpolate(x -> x[1], P)
        D_M, _ = _PNS.momentum_force_envelope(uh, ph, αc, ν, visc, σc, fc, V, dΩ)
        D_C, _ = _PNS.mass_force_envelope(uh, ph, αc, 0.0, gz, Q, dΩ)
        r_M = 0.5 * D_M
        r_C = 0.5 * D_C                                          # ⇒ ε_M = ε_C = 0.5 (both gates symmetric)
        m = _PNS.evaluate_convergence(r_M, r_C, uh, ph, αc, ν, visc, σc, fc, 0.0, gz, V, Q, dΩ, d; tol = 0.6)
        @test isapprox(m.eps_M, 0.5; rtol = 1e-8)
        @test isapprox(m.eps_C, 0.5; rtol = 1e-8)
        @test m.degenerate == false
        @test m.converged == (m.eps_M ≤ 0.6 && m.eps_C ≤ 0.6)
        @test isfinite(m.eps_C_strong)                          # diagnostic still computed & reported
        m_tight = _PNS.evaluate_convergence(r_M, r_C, uh, ph, αc, ν, visc, σc, fc, 0.0, gz, V, Q, dΩ, d; tol = 0.4)
        @test m_tight.converged == false                         # ε_M = ε_C = 0.5 > 0.4
    end

    @testset "degenerate state (underflow floor) is flagged and finite" begin
        u0 = interpolate(x -> VectorValue(0.0, 0.0), U)
        f0 = interpolate(x -> VectorValue(0.0, 0.0), U)
        σ0 = interpolate(x -> 0.0, P)
        m = _PNS.evaluate_convergence(0.0, 0.0, u0, p0, αc, ν, visc, σ0, f0, 0.0, gz, V, Q, dΩ, d; tol = 1e-6)
        @test m.degenerate == true
        @test isfinite(m.eps_M) && isfinite(m.eps_C)
    end
end
