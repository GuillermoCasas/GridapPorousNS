# ==============================================================================================
# Nature & Intent:
# Validates that the L-rescaling of the manufactured solution polynomial (introduced 2026-06-01)
# is bit-identical to the legacy hardcoded `sin(πx)` form when L=1, and that at L=10 the
# polynomial / gradient / Laplacian transform correctly under the chain rule (each derivative
# picks up a factor 1/L, the value at coordinates (L·x̂) reproduces the L=1 value at x̂).
#
# Together with `mms_exactness_blitz_test.jl`, this ensures that:
#   - the L=1 behaviour of `UExFunc`, `grad_u_ex`, `lap_u_ex`, `grad_div_u_ex`, `get_p_ex` is
#     unchanged at machine precision (the substitution `pi → pi/L` collapses to `pi`);
#   - the L≠1 behaviour matches the analytical similarity transformation
#     (u_ex(L·x̂; L) ≡ u_ex(x̂; 1)) within machine precision.
#
# Associated Files / Functions:
# - `src/problems/mms_paper_2d.jl` (`UExFunc`, `grad_u_ex`, `lap_u_ex`, `grad_div_u_ex`, `get_p_ex`)
# - `theory/centered_encoding/centered_encoding.tex` Section 6 (closed-form derivation)
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Test
using PorousNSSolver
using Gridap
using Gridap.Fields: ∇, Δ

# A uniform-α porosity field with α(x) ≡ α_const everywhere — keeps the test focused on the
# polynomial / derivative algebra of u_ex without coupling to the radial bump's non-trivial
# derivatives ∇α, ∇²α. Construct using SmoothRadialPorosity with r_1 = r_2 (degenerate bump
# of zero width sitting at the origin): outside r > r_2 it returns α_∞ exactly. Pick eval
# points outside this disk so α is constant and its derivatives vanish.
#
# Actually simpler: pick alpha_0 = alpha_infty so the bump is uniform regardless of geometry.
function _build_uniform_alpha_field(alpha_const::Float64, L::Float64)
    return PorousNSSolver.SmoothRadialPorosity(alpha_const, alpha_const, 0.05*L, 0.10*L)
end

# Pre-rescaling reference closed form (the OLD code at L=1). All evaluations done at
# dimensionless position x̂. Returns u_ex value, gradient tensor, Laplacian vector.
#
# UExFunc evaluates as `U · α₀ · (1/α(x)) · S(x)`. We construct the test field with
# α_0 = α_∞ = alpha_const so α(x) ≡ alpha_const everywhere — therefore the factor
# `α₀ · (1/α) ≡ 1` cancels out. The references below reflect that simplification
# (they ARE the closed form U · S evaluated, NOT U · (1/α) · S which would be off
# by a factor of α_const).
function _ref_uex_L1(U::Float64, alpha_const::Float64, x̂::VectorValue{2,Float64})
    S = VectorValue(sin(pi*x̂[1])*sin(pi*x̂[2]), cos(pi*x̂[1])*cos(pi*x̂[2]))
    return U * S
end

function _ref_grad_uex_L1(U::Float64, alpha_const::Float64, x̂::VectorValue{2,Float64})
    # ∇α = 0 for uniform-α field, and α₀/α = 1, so ∇u_ex = U · ∇S.
    grad_S = TensorValue(
        pi*cos(pi*x̂[1])*sin(pi*x̂[2]), pi*sin(pi*x̂[1])*cos(pi*x̂[2]),
        -pi*sin(pi*x̂[1])*cos(pi*x̂[2]), -pi*cos(pi*x̂[1])*sin(pi*x̂[2])
    )
    return U * grad_S
end

function _ref_lap_uex_L1(U::Float64, alpha_const::Float64, x̂::VectorValue{2,Float64})
    # ΔS = -2π²·S for the chosen pair; uniform α gives Δu_ex = U · ΔS.
    S = VectorValue(sin(pi*x̂[1])*sin(pi*x̂[2]), cos(pi*x̂[1])*cos(pi*x̂[2]))
    return -2.0 * pi^2 * U * S
end

@testset "fast: MMS polynomial L=1 reproduces legacy hardcoded form" begin
    U = 2.5
    alpha_const = 0.8

    # Build a minimal PorousNSSolver formulation so we can construct Paper2DMMS.
    # We don't actually need the dimensional ν, σ to be physically meaningful — we just need
    # something Paper2DMMS will accept. Use ConstantSigmaLaw with σ=1, viscous op irrelevant.
    rxn = PorousNSSolver.ConstantSigmaLaw(1.0)
    visc = PorousNSSolver.DeviatoricSymmetricViscosity()
    proj = PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma()
    reg = PorousNSSolver.SmoothVelocityFloor(0.0, 0.0, 1e-8, PorousNSSolver.VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR)
    form = PorousNSSolver.PaperGeneralFormulation(visc, rxn, proj, reg, 1.0, 1e-8)

    alpha_field_L1 = _build_uniform_alpha_field(alpha_const, 1.0)
    mms_L1 = PorousNSSolver.Paper2DMMS(form, U, alpha_field_L1; L=1.0, alpha_infty=alpha_const)
    u_func_L1 = PorousNSSolver.get_u_ex(mms_L1)

    # Sample points safely outside the (collapsed) porosity bump (r > 0.10) so the field is
    # uniformly α_const there.
    sample_points = [
        VectorValue(0.5,  0.5),
        VectorValue(0.25, 0.75),
        VectorValue(0.3,  0.4),
        VectorValue(0.45, 0.2),
    ]

    for x̂ in sample_points
        u_val = u_func_L1(x̂)
        u_ref = _ref_uex_L1(U, alpha_const, x̂)
        @test isapprox(u_val[1], u_ref[1]; atol=1e-15, rtol=1e-12)
        @test isapprox(u_val[2], u_ref[2]; atol=1e-15, rtol=1e-12)

        g_val = PorousNSSolver.grad_u_ex(u_func_L1, x̂)
        g_ref = _ref_grad_uex_L1(U, alpha_const, x̂)
        @test isapprox(g_val[1,1], g_ref[1,1]; atol=1e-14, rtol=1e-12)
        @test isapprox(g_val[1,2], g_ref[1,2]; atol=1e-14, rtol=1e-12)
        @test isapprox(g_val[2,1], g_ref[2,1]; atol=1e-14, rtol=1e-12)
        @test isapprox(g_val[2,2], g_ref[2,2]; atol=1e-14, rtol=1e-12)

        lap_val = PorousNSSolver.lap_u_ex(u_func_L1, x̂)
        lap_ref = _ref_lap_uex_L1(U, alpha_const, x̂)
        @test isapprox(lap_val[1], lap_ref[1]; atol=1e-13, rtol=1e-12)
        @test isapprox(lap_val[2], lap_ref[2]; atol=1e-13, rtol=1e-12)
    end
end

@testset "fast: MMS similarity transformation u_ex(L·x̂; L) ≡ u_ex(x̂; 1)" begin
    # Mathematical statement: the manufactured solution should be invariant under the
    # similarity transformation that defines the L-scaling. Specifically:
    #   u_ex(x; L) = U · α₀ · (1/α) · (sin(πx₁/L)·sin(πx₂/L), cos(πx₁/L)·cos(πx₂/L))
    # so at x = L·x̂ (physical position on the L-scaled domain):
    #   u_ex(L·x̂; L) = U · α₀ · (1/α(L·x̂)) · (sin(π·x̂₁)·sin(π·x̂₂), …)
    # Provided α is the same at L·x̂ as at x̂ on the L=1 domain (uniform α here), this MUST
    # equal u_ex(x̂; 1). Derivatives transform with 1/L:
    #   ∇u_ex(L·x̂; L) = (1/L) · ∇u_ex(x̂; 1).
    U = 1.7
    alpha_const = 0.6
    L_test = 10.0

    rxn = PorousNSSolver.ConstantSigmaLaw(1.0)
    visc = PorousNSSolver.DeviatoricSymmetricViscosity()
    proj = PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma()
    reg = PorousNSSolver.SmoothVelocityFloor(0.0, 0.0, 1e-8, PorousNSSolver.VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR)
    form = PorousNSSolver.PaperGeneralFormulation(visc, rxn, proj, reg, 1.0, 1e-8)

    # L=1 reference field and MMS
    alpha_field_L1 = _build_uniform_alpha_field(alpha_const, 1.0)
    mms_L1 = PorousNSSolver.Paper2DMMS(form, U, alpha_field_L1; L=1.0, alpha_infty=alpha_const)
    u_func_L1 = PorousNSSolver.get_u_ex(mms_L1)

    # L=10 field (porosity bump radii scaled with L) and MMS
    alpha_field_Lx = _build_uniform_alpha_field(alpha_const, L_test)
    mms_Lx = PorousNSSolver.Paper2DMMS(form, U, alpha_field_Lx; L=L_test, alpha_infty=alpha_const)
    u_func_Lx = PorousNSSolver.get_u_ex(mms_Lx)

    sample_points_unit = [
        VectorValue(0.5,  0.5),
        VectorValue(0.25, 0.75),
        VectorValue(0.3,  0.4),
    ]

    for x̂ in sample_points_unit
        x_phys = VectorValue(L_test * x̂[1], L_test * x̂[2])

        # Similarity at the value level: u_ex(L·x̂; L) ≡ u_ex(x̂; 1).
        u_val_Lx = u_func_Lx(x_phys)
        u_val_L1 = u_func_L1(x̂)
        @test isapprox(u_val_Lx[1], u_val_L1[1]; atol=1e-13, rtol=1e-11)
        @test isapprox(u_val_Lx[2], u_val_L1[2]; atol=1e-13, rtol=1e-11)

        # Similarity at the gradient level: ∇u_ex(L·x̂; L) ≡ (1/L) · ∇u_ex(x̂; 1).
        g_val_Lx = PorousNSSolver.grad_u_ex(u_func_Lx, x_phys)
        g_val_L1 = PorousNSSolver.grad_u_ex(u_func_L1, x̂)
        @test isapprox(g_val_Lx[1,1] * L_test, g_val_L1[1,1]; atol=1e-12, rtol=1e-10)
        @test isapprox(g_val_Lx[2,2] * L_test, g_val_L1[2,2]; atol=1e-12, rtol=1e-10)

        # Similarity at the Laplacian level: Δu_ex(L·x̂; L) ≡ (1/L²) · Δu_ex(x̂; 1).
        lap_val_Lx = PorousNSSolver.lap_u_ex(u_func_Lx, x_phys)
        lap_val_L1 = PorousNSSolver.lap_u_ex(u_func_L1, x̂)
        @test isapprox(lap_val_Lx[1] * L_test^2, lap_val_L1[1]; atol=1e-11, rtol=1e-9)
        @test isapprox(lap_val_Lx[2] * L_test^2, lap_val_L1[2]; atol=1e-11, rtol=1e-9)
    end
end
