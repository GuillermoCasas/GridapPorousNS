# src/models/regularization.jl
#
# Velocity-magnitude regularization policies.
#
# Several pieces of the formulation need a speed |u| but break down when the raw
# magnitude `sqrt(u⋅u)` vanishes or is too small: it is non-differentiable at u = 0
# (so Newton derivatives blow up), and the convective τ stabilization parameters
# (eq:Tau1 / eq:TauNavierStokes) want a strictly positive effective velocity even in
# stagnant regions. A regularization policy supplies a "floored" speed that stays
# smooth and bounded away from zero.
#
# Two distinct floored speeds are exposed, because the two consumers have different
# correctness requirements:
#   - `effective_speed`  → feeds the τ stabilization parameters; may carry a
#                          mesh-dependent diffusive floor (a numerical knob).
#   - `reaction_speed`   → feeds the PHYSICAL drag law σ(α,u); must stay
#                          mesh-independent so it does not corrupt h-convergence.
using Gridap
using Gridap.Algebra
using LinearAlgebra

# Denominator guard for the `d|u|/du = u/|u|` factor in the Exact-Newton reaction/τ derivatives
# (`dsigma_du`, `compute_dtau_1_du`, `compute_dtau_2_du`): |u| can be exactly 0 at a perfect Dirichlet
# stagnation point, where that factor is singular. A tiny additive floor keeps it finite. This is the
# single source of truth for the value previously copy-pasted as an inline `1e-12` across `reaction.jl`
# and `tau.jl`; it equals the default `PhysicalProperties.epsilon_floor`, and could be threaded from
# config if a tunable is ever needed. It enters ONLY the Jacobian (these derivatives are zeroed in Picard
# and never appear in the residual), so it cannot move a converged solution.
const VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR = 1e-12

# Common interface for the floor policies below. A policy is a small struct holding
# the floor parameters; dispatch on its type selects how `effective_speed` /
# `reaction_speed` regularize the raw velocity magnitude.
abstract type AbstractVelocityRegularization end

# Identity policy: no floor at all. Both speeds reduce to the raw magnitude `sqrt(u⋅u)`.
# Use when the flow never stagnates and τ does not need a positive lower bound.
struct NoRegularization <: AbstractVelocityRegularization end

function effective_speed(::NoRegularization, u, ν, h, c_1, c_2)
    return sqrt(u ⋅ u)
end

# Smooth-floor policy for the τ stabilization speed. Fields:
#   u_base_floor_ref : mesh-independent constant floor [velocity units]; the minimum
#                      speed kept everywhere so |u| is differentiable at stagnation.
#   h_floor_weight   : dimensionless weight on the mesh-dependent diffusive floor
#                      term below (set 0 to disable it).
#   epsilon_floor    : tiny denominator guard so the diffusive floor stays finite as
#                      h → 0.
# T is the shared scalar element type of the three parameters.
struct SmoothVelocityFloor{T} <: AbstractVelocityRegularization
    u_base_floor_ref::T
    h_floor_weight::T
    epsilon_floor::T
end

# Floored speed used to evaluate the convective stabilization parameters τ.
# The floor blends a constant part with a diffusive part `c₁ν/(c₂h)` (the same
# c₁, c₂ that define τ for equal-order interpolation): on coarse / diffusion-dominated
# cells this raises the effective speed so τ does not over-stabilize. The floor is
# combined with the true magnitude as a smooth sum-of-squares, `sqrt(u⋅u + u_floor²)`,
# which is C∞ in u (good for the Newton Jacobian) and never below `u_floor`.
function effective_speed(policy::SmoothVelocityFloor, u, ν, h, c_1, c_2)
    u_floor = (policy.u_base_floor_ref + policy.h_floor_weight * (c_1 * ν) / (c_2 * h + policy.epsilon_floor))
    return sqrt(u ⋅ u + u_floor^2)
end

# `reaction_speed` is the velocity magnitude |u| that enters the PHYSICAL reaction
# law σ(α,u) = a(α) + b(α)|u| (Forchheimer-Ergun, eq:DBFResistanceTerm). It floors
# the speed only with the mesh-independent constant `u_base_floor_ref`, purely to keep
# |u| differentiable at the stagnation point u = 0.
#
# [known-fragility] This speed must NOT include the mesh-dependent diffusive floor
# `h_floor_weight·c₁ν/(c₂h)` that `effective_speed` uses. That term scales like ν/h and
# grows under refinement, so feeding it into the drag would give every mesh a different
# effective drag law and destroy h-convergence (see docs/cocquet/convergence-analysis.md,
# Phase 6). The ν, h, c₁, c₂ arguments are accepted only for signature parity with
# `effective_speed` and are deliberately unused here.
function reaction_speed(::NoRegularization, u, ν, h, c_1, c_2)
    return sqrt(u ⋅ u)
end

function reaction_speed(policy::SmoothVelocityFloor, u, ν, h, c_1, c_2)
    return sqrt(u ⋅ u + policy.u_base_floor_ref^2)
end
