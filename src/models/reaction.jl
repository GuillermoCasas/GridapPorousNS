# src/models/reaction.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

abstract type AbstractReactionLaw end

"""
    min_quadrature_degree(law::AbstractReactionLaw, k_v::Int) -> Int

Minimum total quadrature degree the reaction law requires for residual /
Jacobian assembly on a velocity space of polynomial order `k_v`. Default
0: the law has no requirement beyond the formulation's base quadrature
rule (`compute_consistent_quadrature_degree`). Laws whose integrand is
non-polynomial in `u` (e.g. Forchheimer's `|u|·u` factor) should
override.
"""
min_quadrature_degree(::AbstractReactionLaw, ::Int) = 0

struct KinematicState{T1,T2,T3}
    u::T1
    grad_u::T2
    mag_u::T3
end

struct MediumState{T1,T2,T3}
    alpha::T1
    grad_alpha::T2
    h::T3
end

struct ConstantSigmaLaw{T} <: AbstractReactionLaw
    sigma_val::T
end

function sigma(law::ConstantSigmaLaw, kin::KinematicState, med::MediumState, mag_u)
    return law.sigma_val
end

function dsigma_du(law::ConstantSigmaLaw, kin::KinematicState, med::MediumState, mag_u, du)
    return 0.0 * (kin.u ⋅ du)
end

struct ForchheimerErgunLaw{T} <: AbstractReactionLaw
    sigma_linear::T
    sigma_nonlinear::T
end

# Forchheimer-Ergun residual contains σ(u)·u = (a + b|u|)·u, which carries a
# non-polynomial `|u|` factor (only C^0 at u=0). The exact integral requires
# infinite quadrature; bumping the base degree by ⌊k_v/2⌋ keeps the
# Forchheimer quadrature error below the leading FE consistency error
# O(h^(k_v+1)). For k_v=1 this is a no-op; for k_v≥2 it adds a few Gauss
# points per cell at negligible cost.
min_quadrature_degree(::ForchheimerErgunLaw, k_v::Int) = 4 * k_v + (k_v ÷ 2)

function sigma(law::ForchheimerErgunLaw, kin::KinematicState, med::MediumState, mag_u)
    α = med.alpha
    a_term = law.sigma_linear * ((1.0 - α) / α)^2
    b_term = law.sigma_nonlinear * (1.0 - α) / α
    return a_term + b_term * mag_u
end

function dsigma_du(law::ForchheimerErgunLaw, kin::KinematicState, med::MediumState, mag_u, du)
    α = med.alpha
    u = kin.u
    b_term = law.sigma_nonlinear * (1.0 - α) / α
    mag_u_reg = mag_u + 1e-12
    return b_term * (u ⋅ du) / mag_u_reg
end
