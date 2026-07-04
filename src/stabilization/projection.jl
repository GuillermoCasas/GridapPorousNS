# src/stabilization/projection.jl
#
# OSGS projection policy — controls which terms of the strong residual enter the orthogonal sub-grid
# scale (the L²-projection π_h that OSGS subtracts; see eq:OSGSProblem, Section 4.4). [paper-faithful]
#
# In OSGS the sub-scale is taken in X_{h0}^⊥, so the projected residual that drives the stabilization
# is (1 - Π)R(U_h) = R(U_h) - π_h. Any residual term that is *linear in the FE unknown with constant
# coefficient* already lives in the FE space, so its L²-projection equals itself and (1 - Π) annihilates
# it exactly — keeping it on both sides is wasted work that can also inject projection noise. A policy
# names which such terms to drop. ASGS uses identity projection (π_h = 0) and is unaffected by the choice.
#
# Three function families dispatch on the policy (each has a momentum `_u` and a mass `_p` variant):
#   apply_projectable_residual_*  — the term fed INTO the L² projection (computed in osgs_solver.jl);
#                                   policies pre-subtract the trimmed term here so it is never projected.
#   apply_projection_*            — the stabilization residual R - π_h actually used (OSGS only).
#   apply_jacobian_projection_*   — the Newton linearization of apply_projection_*.
# All are called from continuous_problem.jl (residual/Jacobian assembly) and osgs_solver.jl (projection).
abstract type AbstractProjectionPolicy end

# Project the entire strong residual — the canonical OSGS choice, valid for any reaction law. π_h is
# the full L²-projection of R(U_h); nothing is trimmed.
struct ProjectFullResidual <: AbstractProjectionPolicy end

# Section 4.4 trim: when σ is constant (ConstantSigmaLaw), the reaction term σ·u is linear in u with
# constant coefficient, so its L²-projection onto the FE space is exact — (1 - Π)(σ·u) = 0. This policy
# omits σ·u from both the projection and the stabilization residual, and likewise drops the linear
# pressure-penalty term ε·p from the mass residual. [paper-faithful] Only valid for constant σ;
# sanitize_projection_policy below rejects pairing it with a nonlinear (Forchheimer) law.
struct ProjectResidualWithoutReactionWhenConstantSigma <: AbstractProjectionPolicy end

# Trims only the ε·p pressure-penalty term from the mass residual (same FE-exact-projection argument as
# above), while projecting the full momentum residual including reaction. Useful when σ is nonlinear
# (reaction must stay) but the constant ε·p term can still be dropped.
struct ProjectResidualWithoutPressurePenalty <: AbstractProjectionPolicy end

# --- ProjectFullResidual: nothing trimmed; project all of R(U_h). -----------------------------------

# Term fed into the momentum L² projection: the whole strong residual R_u.
function apply_projectable_residual_u(::ProjectFullResidual, R_u, σ, u)
    return R_u
end

# Term fed into the mass L² projection: the whole strong residual R_p.
function apply_projectable_residual_p(::ProjectFullResidual, R_p, physical_epsilon, p)
    return R_p
end

# Momentum stabilization residual: subtract the projection π_h (=pi_u) in OSGS, pass through in ASGS.
function apply_projection_u(::ProjectFullResidual, R_u, σ, u, pi_u, is_osgs)
    if is_osgs
        return R_u - pi_u
    end
    return R_u
end

# Newton linearization of apply_projection_u: the ∂π_h/∂u term is deliberately dropped — π_h is held
# frozen in the Jacobian (a local frozen-π, Picard-type coupling; the coupled OSGS solve re-projects
# π_h in the residual at each Newton evaluation), so the Jacobian of the projected residual is just R_du.
function apply_jacobian_projection_u(::ProjectFullResidual, R_du, σ, dsigma_val, u, du, is_osgs)
    return R_du
end

# --- ProjectResidualWithoutReactionWhenConstantSigma: drop the FE-exact σ·u and ε·p terms. -----------

# Remove σ·u before projecting: it is FE-exact for constant σ, so projecting it would add no information.
function apply_projectable_residual_u(::ProjectResidualWithoutReactionWhenConstantSigma, R_u, σ, u)
    return R_u - (σ * u)
end

# Remove the linear pressure-penalty ε·p before projecting (likewise FE-exact for constant ε).
function apply_projectable_residual_p(::ProjectResidualWithoutReactionWhenConstantSigma, R_p, physical_epsilon, p)
    return R_p - (physical_epsilon * p)
end

# Momentum stabilization residual: trim σ·u and subtract π_h (OSGS only); ASGS passes R_u through.
function apply_projection_u(::ProjectResidualWithoutReactionWhenConstantSigma, R_u, σ, u, pi_u, is_osgs)
    if is_osgs
        return R_u - (σ * u) - pi_u
    end
    return R_u
end

# Newton linearization of the trimmed momentum residual: the directional derivative of -σ·u is
# -(σ·du + dσ·u). With constant σ the caller passes dsigma_val = 0, so this reduces to -σ·du.
function apply_jacobian_projection_u(::ProjectResidualWithoutReactionWhenConstantSigma, R_du, σ, dsigma_val, u, du, is_osgs)
    if is_osgs
        return R_du - (σ * du + dsigma_val * u)
    end
    return R_du
end

# --- Mass-residual counterparts (pi_p is the L²-projection of R_p). ---------------------------------

# ProjectFullResidual: subtract π_h in OSGS, pass through in ASGS; π_h frozen, so the Jacobian is R_dp.
function apply_projection_p(::ProjectFullResidual, R_p, physical_epsilon, p, pi_p, is_osgs)
    if is_osgs
        return R_p - pi_p
    end
    return R_p
end

function apply_jacobian_projection_p(::ProjectFullResidual, R_dp, physical_epsilon, dp, is_osgs)
    return R_dp
end

# WithoutReactionWhenConstantSigma: also trim the FE-exact ε·p term from the mass residual / Jacobian.
function apply_projection_p(::ProjectResidualWithoutReactionWhenConstantSigma, R_p, physical_epsilon, p, pi_p, is_osgs)
    if is_osgs
        return R_p - (physical_epsilon * p) - pi_p
    end
    return R_p
end

function apply_jacobian_projection_p(::ProjectResidualWithoutReactionWhenConstantSigma, R_dp, physical_epsilon, dp, is_osgs)
    if is_osgs
        return R_dp - (physical_epsilon * dp)
    end
    return R_dp
end

# --- ProjectResidualWithoutPressurePenalty: compose the two halves of the behavior above. -----------
# Momentum side behaves like ProjectFullResidual (reaction stays — σ may be nonlinear); mass side
# behaves like WithoutReactionWhenConstantSigma (drop the FE-exact ε·p term). Implemented by delegating
# each method to the policy that already encodes the wanted half.
function apply_projectable_residual_u(::ProjectResidualWithoutPressurePenalty, R_u, σ, u)
    return apply_projectable_residual_u(ProjectFullResidual(), R_u, σ, u)
end
function apply_projectable_residual_p(::ProjectResidualWithoutPressurePenalty, R_p, physical_epsilon, p)
    return apply_projectable_residual_p(ProjectResidualWithoutReactionWhenConstantSigma(), R_p, physical_epsilon, p)
end

function apply_projection_u(::ProjectResidualWithoutPressurePenalty, R_u, σ, u, pi_u, is_osgs)
    return apply_projection_u(ProjectFullResidual(), R_u, σ, u, pi_u, is_osgs)
end
function apply_jacobian_projection_u(::ProjectResidualWithoutPressurePenalty, R_du, σ, dsigma_val, u, du, is_osgs)
    return apply_jacobian_projection_u(ProjectFullResidual(), R_du, σ, dsigma_val, u, du, is_osgs)
end
function apply_projection_p(::ProjectResidualWithoutPressurePenalty, R_p, physical_epsilon, p, pi_p, is_osgs)
    return apply_projection_p(ProjectResidualWithoutReactionWhenConstantSigma(), R_p, physical_epsilon, p, pi_p, is_osgs)
end
function apply_jacobian_projection_p(::ProjectResidualWithoutPressurePenalty, R_dp, physical_epsilon, dp, is_osgs)
    return apply_jacobian_projection_p(ProjectResidualWithoutReactionWhenConstantSigma(), R_dp, physical_epsilon, dp, is_osgs)
end

# --- Policy/reaction-law compatibility check (run once at formulation construction). -----------------
# Guards the Section-4.4 invariant: trimming σ·u from the projection is only correct when σ is constant.
# Called from PaperGeneralFormulation's constructor (continuous_problem.jl). With autocorrect=true a
# mismatched policy is downgraded to ProjectFullResidual; otherwise it is a hard configuration error.

# Default: any policy is accepted with any reaction law (the generic case imposes no constraint).
function sanitize_projection_policy(policy, reaction_law; autocorrect=false)
    return policy
end

# The σ·u-trimming policy is INVALID unless σ is constant in u: (1 - Π)(σ(u)·u) ≠ 0 when σ depends on u,
# so dropping that term silently corrupts the OSGS stabilization. [known-fragility] Gated on the
# `is_sigma_constant` TRAIT (reaction.jl), not a concrete law type, so any future nonlinear reaction law
# is rejected by default (conservative `is_sigma_constant = false`) instead of slipping through.
function sanitize_projection_policy(policy::ProjectResidualWithoutReactionWhenConstantSigma, reaction_law; autocorrect=false)
    is_sigma_constant(reaction_law) && return policy
    if autocorrect
        @info "ProjectResidualWithoutReactionWhenConstantSigma is invalid with a velocity-dependent reaction law. Auto-correcting to ProjectFullResidual."
        return ProjectFullResidual()
    else
        error("ProjectResidualWithoutReactionWhenConstantSigma requires a constant-σ reaction law (is_sigma_constant(law) == true); got $(typeof(reaction_law)).")
    end
end
