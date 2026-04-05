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
