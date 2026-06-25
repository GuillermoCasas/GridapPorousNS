# src/models/reaction.jl
#
# Reaction (viscous-resistance) laws for the porous Navier-Stokes momentum
# equation. The reaction term σ(α, u)·u models the drag the porous medium exerts
# on the fluid; σ is the inverse permeability tensor (here scalar), symmetric and
# positive semidefinite, and enters the strong momentum balance
# (paper eq:StrongMomentumEquation) and the stabilization parameter τ₁
# (src/stabilization/tau.jl).
#
# A concrete `AbstractReactionLaw` supplies two callbacks consumed by both the
# formulation residual/Jacobian and τ:
#   sigma(...)     -> the scalar σ at a quadrature point.
#   dsigma_du(...) -> its directional derivative wrt u (the Exact-Newton term).
using Gridap
using Gridap.Algebra
using LinearAlgebra

# Interface every reaction closure implements. Subtypes pick a functional form
# for σ(α, u); the formulation and τ call `sigma`/`dsigma_du` without knowing
# which concrete law is in play.
abstract type AbstractReactionLaw end

"""
    min_quadrature_degree(law::AbstractReactionLaw, k_v::Int) -> Int

Minimum total quadrature degree the reaction law requires for residual /
Jacobian assembly on a velocity space of polynomial order `k_v`. The
formulation takes the maximum of this and its own base rule
(`compute_consistent_quadrature_degree`) when choosing the Gauss order.

Default 0: a law whose integrand is polynomial in `u` adds no requirement.
Laws with a non-polynomial integrand (e.g. Forchheimer's `|u|·u` factor)
override this to request extra integration points so the quadrature error
stays below the leading FE consistency error.
"""
min_quadrature_degree(::AbstractReactionLaw, ::Int) = 0

"""
    is_sigma_constant(law::AbstractReactionLaw) -> Bool

True iff σ does NOT depend on u (constant in velocity). Gates the OSGS §4.4 reaction trim
(`ProjectResidualWithoutReactionWhenConstantSigma`), whose correctness rests on `(1−Π)(σu)=0` — valid
only when σ is constant in u. Conservative default `false`, so any future nonlinear law is rejected by
the trim guard until it explicitly opts in (rather than silently corrupting the OSGS stabilization).
"""
is_sigma_constant(::AbstractReactionLaw) = false

# Bundle of the local solution quantities a reaction law / τ may need at one
# quadrature point. Type-parameterized so it works whether the entries are plain
# numbers (pointwise evaluation) or Gridap CellField operands (lazy assembly).
#   u      — velocity vector.
#   grad_u — velocity gradient ∇u (a second-order tensor).
#   mag_u  — speed |u|, the magnitude that drives Forchheimer's nonlinear drag.
struct KinematicState{T1,T2,T3}
    u::T1
    grad_u::T2
    mag_u::T3
end

# Bundle of the local porous-medium descriptors at one quadrature point.
#   alpha      — fluid volume fraction α ∈ (0,1]; α→0 is fully solid.
#   grad_alpha — its spatial gradient ∇α (inhomogeneous media).
#   h          — local element size, used by τ (not by the reaction laws here).
struct MediumState{T1,T2,T3}
    alpha::T1
    grad_alpha::T2
    h::T3
end

# Constant scalar resistance σ ≡ sigma_val, independent of α and u. Recovers the
# linear Darcy / Brinkman drag and the `Constant_Sigma` reaction model; its L²
# projection onto the FE space is exactly zero, which is why OSGS can drop it
# from the orthogonal projection (paper §4.4).
struct ConstantSigmaLaw{T} <: AbstractReactionLaw
    sigma_val::T
end

function sigma(law::ConstantSigmaLaw, kin::KinematicState, med::MediumState, mag_u)
    return law.sigma_val
end

# σ does not depend on u, so its derivative vanishes. Multiplying the zero by
# (u ⋅ du) preserves the operand type so the Exact-Newton Jacobian assembles
# uniformly across reaction laws.
function dsigma_du(law::ConstantSigmaLaw, kin::KinematicState, med::MediumState, mag_u, du, deriv_floor)
    return 0.0 * (kin.u ⋅ du)
end

# σ ≡ sigma_val is constant in u, so the §4.4 reaction-trim projection is exact for this law.
is_sigma_constant(::ConstantSigmaLaw) = true

# Forchheimer-Ergun drag: σ(α, u) = a(α) + b(α)|u| (paper eq:DBFResistanceTerm),
# with the porosity dependence given the Ergun closure
#   a(α) = sigma_linear   · ((1-α)/α)²   (linear/viscous term),
#   b(α) = sigma_nonlinear · (1-α)/α      (inertial term, scaled by |u|).
# `sigma_linear`/`sigma_nonlinear` are the base resistance coefficients (drag
# scale at the reference packing); the (1-α)/α factors blow the drag up as the
# medium tightens (α→0) and send it to zero in pure fluid (α→1).
struct ForchheimerErgunLaw{T} <: AbstractReactionLaw
    sigma_linear::T
    sigma_nonlinear::T
end

# The Forchheimer residual integrand σ(u)·u = (a + b|u|)·u carries a
# non-polynomial `|u|` factor (only C⁰ at u=0), so no finite Gauss rule is
# exact. Requesting degree 4·k_v + ⌊k_v/2⌋ keeps the Forchheimer quadrature
# error below the leading FE consistency error O(h^(k_v+1)). The extra ⌊k_v/2⌋
# is a no-op at k_v=1 and adds only a few points per cell at k_v≥2.
min_quadrature_degree(::ForchheimerErgunLaw, k_v::Int) = 4 * k_v + (k_v ÷ 2)

# Evaluate σ = a(α) + b(α)|u|. `a_term`/`b_term` are the Ergun coefficients
# a(α), b(α) above; the speed enters linearly through `mag_u`.
function sigma(law::ForchheimerErgunLaw, kin::KinematicState, med::MediumState, mag_u)
    α = med.alpha
    a_term = law.sigma_linear * ((1.0 - α) / α)^2
    b_term = law.sigma_nonlinear * (1.0 - α) / α
    return a_term + b_term * mag_u
end

# Directional derivative of σ wrt u (the Exact-Newton ∂σ/∂u contribution): only
# b(α)|u| depends on u, and d|u| = (u·du)/|u|. `deriv_floor` (ε_d, the config-driven
# velocity_magnitude_derivative_floor) adds a tiny floor to the denominator so the kink in |u| at u=0
# cannot divide by zero. See theory/velocity_floor_regularization/.
function dsigma_du(law::ForchheimerErgunLaw, kin::KinematicState, med::MediumState, mag_u, du, deriv_floor)
    α = med.alpha
    u = kin.u
    b_term = law.sigma_nonlinear * (1.0 - α) / α
    mag_u_reg = mag_u + deriv_floor
    return b_term * (u ⋅ du) / mag_u_reg
end
