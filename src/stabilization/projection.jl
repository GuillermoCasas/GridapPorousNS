# src/stabilization/projection.jl
abstract type AbstractProjectionPolicy end
struct ProjectFullResidual <: AbstractProjectionPolicy end
struct ProjectResidualWithoutReactionWhenConstantSigma <: AbstractProjectionPolicy end
struct ProjectResidualWithoutMassPenalty <: AbstractProjectionPolicy end

function apply_projectable_residual_u(::ProjectFullResidual, R_u, σ, u)
    return R_u
end

function apply_projectable_residual_p(::ProjectFullResidual, R_p, eps_val, p)
    return R_p
end

function apply_projection_u(::ProjectFullResidual, R_u, σ, u, pi_u, is_osgs)
    if is_osgs
        return R_u - pi_u
    end
    return R_u
end

function apply_jacobian_projection_u(::ProjectFullResidual, R_du, σ, dsigma_val, u, du, is_osgs)
    return R_du
end

function apply_projectable_residual_u(::ProjectResidualWithoutReactionWhenConstantSigma, R_u, σ, u)
    return R_u - (σ * u)
end

function apply_projectable_residual_p(::ProjectResidualWithoutReactionWhenConstantSigma, R_p, eps_val, p)
    return R_p - (eps_val * p)
end

function apply_projection_u(::ProjectResidualWithoutReactionWhenConstantSigma, R_u, σ, u, pi_u, is_osgs)
    if is_osgs
        return R_u - (σ * u) - pi_u
    end
    return R_u
end

function apply_jacobian_projection_u(::ProjectResidualWithoutReactionWhenConstantSigma, R_du, σ, dsigma_val, u, du, is_osgs)
    if is_osgs
        return R_du - (σ * du + dsigma_val * u)
    end
    return R_du
end

# Similarly for mass
function apply_projection_p(::ProjectFullResidual, R_p, eps_val, p, pi_p, is_osgs)
    if is_osgs
        return R_p - pi_p
    end
    return R_p
end

function apply_jacobian_projection_p(::ProjectFullResidual, R_dp, eps_val, dp, is_osgs)
    return R_dp
end

function apply_projection_p(::ProjectResidualWithoutReactionWhenConstantSigma, R_p, eps_val, p, pi_p, is_osgs)
    if is_osgs
        return R_p - (eps_val * p) - pi_p
    end
    return R_p
end

function apply_jacobian_projection_p(::ProjectResidualWithoutReactionWhenConstantSigma, R_dp, eps_val, dp, is_osgs)
    if is_osgs
        return R_dp - (eps_val * dp)
    end
    return R_dp
end

# Fallback definitions
function apply_projectable_residual_u(::ProjectResidualWithoutMassPenalty, R_u, σ, u)
    return apply_projectable_residual_u(ProjectFullResidual(), R_u, σ, u)
end
function apply_projectable_residual_p(::ProjectResidualWithoutMassPenalty, R_p, eps_val, p)
    return apply_projectable_residual_p(ProjectResidualWithoutReactionWhenConstantSigma(), R_p, eps_val, p)
end

function apply_projection_u(::ProjectResidualWithoutMassPenalty, R_u, σ, u, pi_u, is_osgs)
    return apply_projection_u(ProjectFullResidual(), R_u, σ, u, pi_u, is_osgs)
end
function apply_jacobian_projection_u(::ProjectResidualWithoutMassPenalty, R_du, σ, dsigma_val, u, du, is_osgs)
    return apply_jacobian_projection_u(ProjectFullResidual(), R_du, σ, dsigma_val, u, du, is_osgs)
end
function apply_projection_p(::ProjectResidualWithoutMassPenalty, R_p, eps_val, p, pi_p, is_osgs)
    return apply_projection_p(ProjectResidualWithoutReactionWhenConstantSigma(), R_p, eps_val, p, pi_p, is_osgs)
end
function apply_jacobian_projection_p(::ProjectResidualWithoutMassPenalty, R_dp, eps_val, dp, is_osgs)
    return apply_jacobian_projection_p(ProjectResidualWithoutReactionWhenConstantSigma(), R_dp, eps_val, dp, is_osgs)
end

# Sanitize policies against reaction laws
function sanitize_projection_policy(policy, reaction_law; autocorrect=false)
    return policy
end

function sanitize_projection_policy(policy::ProjectResidualWithoutReactionWhenConstantSigma, reaction_law::ForchheimerErgunLaw; autocorrect=false)
    if autocorrect
        @info "ProjectResidualWithoutReactionWhenConstantSigma is invalid with nonlinear reaction laws. Auto-correcting to ProjectFullResidual."
        return ProjectFullResidual()
    else
        error("ProjectResidualWithoutReactionWhenConstantSigma cannot be paired with a nonlinear reaction law (e.g. ForchheimerErgunLaw). Expected a ConstantSigmaLaw.")
    end
end
