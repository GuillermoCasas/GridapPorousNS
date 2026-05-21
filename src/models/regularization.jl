# src/models/regularization.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

abstract type AbstractVelocityRegularization end

struct NoRegularization <: AbstractVelocityRegularization end

function effective_speed(::NoRegularization, u, ν, h, c_1, c_2)
    return sqrt(u ⋅ u)
end

struct SmoothVelocityFloor{T} <: AbstractVelocityRegularization
    u_base_floor_ref::T
    h_floor_weight::T
    epsilon_floor::T
end

function effective_speed(policy::SmoothVelocityFloor, u, ν, h, c_1, c_2)
    u_floor = (policy.u_base_floor_ref + policy.h_floor_weight * (c_1 * ν) / (c_2 * h + policy.epsilon_floor))
    return sqrt(u ⋅ u + u_floor^2)
end

# `reaction_speed` is the velocity magnitude used inside the PHYSICAL reaction law
# σ(α,u) = a(α) + b(α)|u| (Forchheimer-Ergun). Unlike `effective_speed` (used for the τ
# stabilization parameters), it must NOT carry the mesh-dependent diffusive floor
# `h_floor_weight·c₁ν/(c₂h)`: that term scales like ν/h and grows under refinement, so
# injecting it into the drag makes every mesh solve a different effective drag law and
# destroys h-convergence (see docs/cocquet_convergence_analysis.md, Phase 6). Only a small
# constant floor `u_base_floor_ref` is kept, purely to keep |u| differentiable at u = 0
# (stagnation); it is mesh-independent so it does not break consistency. The ν, h, c₁, c₂
# arguments are accepted for signature parity with `effective_speed` but intentionally unused.
function reaction_speed(::NoRegularization, u, ν, h, c_1, c_2)
    return sqrt(u ⋅ u)
end

function reaction_speed(policy::SmoothVelocityFloor, u, ν, h, c_1, c_2)
    return sqrt(u ⋅ u + policy.u_base_floor_ref^2)
end
