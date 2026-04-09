# src/models/reaction.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

abstract type AbstractReactionLaw end

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
