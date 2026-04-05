# src/formulations/continuous_problem.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

abstract type AbstractFormulation end
abstract type AbstractLinearizationMode end

struct ExactNewtonMode <: AbstractLinearizationMode end
struct PicardMode <: AbstractLinearizationMode end

# Typed Callable Operators to prevent anonymous closure duplication
struct MagOp{R<:AbstractVelocityRegularization} <: Function
    reg::R
    ν::Float64
    c_1::Float64
    c_2::Float64
end
(op::MagOp)(u_v, h_v) = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)

struct SigOp{R<:AbstractReactionLaw} <: Function
    law::R
end
(op::SigOp)(u_v, grad_v, a_v, grad_a_v, h_v, mag) = sigma(op.law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag)

struct DSigOp{R<:AbstractReactionLaw} <: Function
    law::R
end
(op::DSigOp)(u_v, grad_v, a_v, grad_a_v, h_v, mag, du_v) = dsigma_du(op.law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag, du_v)

struct Tau1Op{R<:AbstractReactionLaw} <: Function
    law::R
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
end
(op::Tau1Op)(u_v, grad_v, a_v, grad_a_v, h_v, mag) = compute_tau_1(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.law)

struct Tau2Op <: Function
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
end
(op::Tau2Op)(u_v, grad_v, a_v, grad_a_v, h_v, mag) = compute_tau_2(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), op.ν, op.c_1, op.c_2, op.tau_reg_lim)

struct DTau1Op{R<:AbstractReactionLaw} <: Function
    law::R
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
    freeze_cusp::Bool
end
(op::DTau1Op)(u_v, grad_v, a_v, grad_a_v, h_v, mag, du_v) = compute_dtau_1_du(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), du_v, op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.freeze_cusp, op.law)

struct DTau2Op <: Function
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
    freeze_cusp::Bool
end
(op::DTau2Op)(u_v, grad_v, a_v, grad_a_v, h_v, mag, du_v) = compute_dtau_2_du(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), du_v, op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.freeze_cusp)

struct PaperGeneralFormulation{V<:AbstractViscousOperator, R<:AbstractReactionLaw, P<:AbstractProjectionPolicy, Reg<:AbstractVelocityRegularization} <: AbstractFormulation
    viscous_operator::V
    reaction_law::R
    projection_policy::P
    regularization::Reg
    ν::Float64
    eps_val::Float64
    
    function PaperGeneralFormulation(v::V, r::R, p_in::P, reg::Reg, ν::Float64, eps_val::Float64; autocorrect_policy=false) where {V, R, P, Reg}
        valid_policy = sanitize_projection_policy(p_in, r; autocorrect=autocorrect_policy)
        new{V, R, typeof(valid_policy), Reg}(v, r, valid_policy, reg, ν, eps_val)
    end
end

struct PseudoTractionFormulation{R<:AbstractReactionLaw, P<:AbstractProjectionPolicy, Reg<:AbstractVelocityRegularization} <: AbstractFormulation
    viscous_operator::LaplacianPseudoTractionViscosity
    reaction_law::R
    projection_policy::P
    regularization::Reg
    ν::Float64
    eps_val::Float64
    
    function PseudoTractionFormulation(r::R, p_in::P, reg::Reg, ν::Float64, eps_val::Float64; autocorrect_policy=false) where {R, P, Reg}
        valid_policy = sanitize_projection_policy(p_in, r; autocorrect=autocorrect_policy)
        new{R, typeof(valid_policy), Reg}(LaplacianPseudoTractionViscosity(), r, valid_policy, reg, ν, eps_val)
    end
end

function get_reaction(form::AbstractFormulation)
    return form.reaction_law
end

function get_regularization(form::AbstractFormulation)
    return form.regularization
end

function get_projection_policy(form::AbstractFormulation)
    return form.projection_policy
end

# Reusable Residual evaluations
function eval_strong_residual_u(form::AbstractFormulation, u, p, h, α, f_custom, c_1, c_2)
    ν = form.ν
    conv_u = α * (∇(u)' ⋅ u)
    div_visc_u = strong_viscous_operator(form.viscous_operator, u, α, ν)
    grad_u_dummy = ∇(u)
    
    mag_u = Operation(MagOp(form.regularization, ν, c_1, c_2))(u, h)
    σ = Operation(SigOp(form.reaction_law))(u, grad_u_dummy, α, ∇(α), h, mag_u)
    
    return conv_u + α * ∇(p) + σ * u - div_visc_u - f_custom
end

function eval_strong_residual_p(form::AbstractFormulation, u, p, α, g_custom)
    eps_val = form.eps_val
    div_alpha_u = α * (∇⋅u) + u ⋅ ∇(α)
    return eps_val * p + div_alpha_u - g_custom
end

function build_stabilized_weak_form_residual(X, Y, form::AbstractFormulation, dΩ, h, f_custom, alpha_custom, g_custom, pi_u, pi_p, c_1, c_2, tau_reg_lim; mult_mom=1.0, mult_mass=1.0)
    u, p = X; v, q = Y
    α = alpha_custom
    f = f_custom
    g_mass = g_custom
    ν = form.ν
    eps_val = form.eps_val

    mag_u = Operation(MagOp(form.regularization, ν, c_1, c_2))(u, h)
    σ = Operation(SigOp(form.reaction_law))(u, ∇(u), α, ∇(α), h, mag_u)
    τ_1 = Operation(Tau1Op(form.reaction_law, ν, c_1, c_2, tau_reg_lim))(u, ∇(u), α, ∇(α), h, mag_u)
    τ_2 = Operation(Tau2Op(ν, c_1, c_2, tau_reg_lim))(u, ∇(u), α, ∇(α), h, mag_u)

    # Base weak operators
    conv_term = v ⋅ (α * (∇(u)' ⋅ u))
    visc_term = weak_viscous_operator(form.viscous_operator, u, v, α, ν)
    pres_term = - p * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term  = v ⋅ ( σ * u )
    
    div_alpha_u = α * (∇⋅u) + u ⋅ ∇(α)
    mass_term = q * (eps_val * p + div_alpha_u)
    src_term  = v ⋅ f + q * g_mass

    # Strong residuals for stabilization
    R_u = eval_strong_residual_u(form, u, p, h, α, f, c_1, c_2)
    R_p = eval_strong_residual_p(form, u, p, α, g_mass)
    
    # Adjoints 
    L_u_star_v = strong_adjoint_momentum(form, u, v, q, α) - (σ * v)
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    # Apply projection policies (using dimension-aware zeros)
    proj_pi_u = pi_u === nothing ? (0.0 * u) : pi_u
    proj_pi_p = pi_p === nothing ? 0.0 : pi_p
    
    stab_R_u = apply_projection_u(form.projection_policy, R_u, σ, u, proj_pi_u, pi_u !== nothing)
    stab_R_p = apply_projection_p(form.projection_policy, R_p, eps_val, p, proj_pi_p, pi_p !== nothing)

    stab_mom = mult_mom * (L_u_star_v ⋅ (τ_1 * stab_R_u))
    stab_mass = mult_mass * (L_q_star * (τ_2 * stab_R_p))

    return ∫( conv_term + visc_term + pres_term + res_term + mass_term - src_term + stab_mom + stab_mass )dΩ
end

function build_stabilized_weak_form_jacobian(X, dX, Y, form::AbstractFormulation, dΩ, h, f_custom, alpha_custom, g_custom, pi_u, pi_p, c_1, c_2, tau_reg_lim, freeze_cusp, lin_mode::AbstractLinearizationMode=ExactNewtonMode(); mult_mom=1.0, mult_mass=1.0)
    u, p = X; du, dp = dX; v, q = Y
    α = alpha_custom
    f = f_custom
    g_mass = g_custom
    ν = form.ν
    eps_val = form.eps_val

    mag_u = Operation(MagOp(form.regularization, ν, c_1, c_2))(u, h)
    σ = Operation(SigOp(form.reaction_law))(u, ∇(u), α, ∇(α), h, mag_u)
    dsigma_du_val = Operation(DSigOp(form.reaction_law))(u, ∇(u), α, ∇(α), h, mag_u, du)
    
    if isa(lin_mode, PicardMode)
        dsigma_du_val = 0.0 * (u ⋅ du)
    end
    
    τ_1 = Operation(Tau1Op(form.reaction_law, ν, c_1, c_2, tau_reg_lim))(u, ∇(u), α, ∇(α), h, mag_u)
    τ_2 = Operation(Tau2Op(ν, c_1, c_2, tau_reg_lim))(u, ∇(u), α, ∇(α), h, mag_u)
    
    dtau_1_op = Operation(DTau1Op(form.reaction_law, ν, c_1, c_2, tau_reg_lim, freeze_cusp))
    dtau_2_op = Operation(DTau2Op(ν, c_1, c_2, tau_reg_lim, freeze_cusp))
    
    dtau_1_du = isa(lin_mode, PicardMode) ? (0.0 * (u ⋅ du)) : dtau_1_op(u, ∇(u), α, ∇(α), h, mag_u, du)
    dtau_2_du = isa(lin_mode, PicardMode) ? (0.0 * (u ⋅ du)) : dtau_2_op(u, ∇(u), α, ∇(α), h, mag_u, du)

    conv_du = α * (∇(du)' ⋅ u)
    if isa(lin_mode, ExactNewtonMode)
        conv_du = conv_du + α * (∇(u)' ⋅ du)
    end
    
    visc_term_jac = weak_viscous_jacobian(form.viscous_operator, du, v, α, ν)
    pres_term_jac = - dp * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term_jac  = v ⋅ ( σ * du + (dsigma_du_val * u) )
    
    div_alpha_du = α * (∇⋅du) + du ⋅ ∇(α)
    mass_term_jac = q * (eps_val * dp + div_alpha_du)
    
    conv_term_jac = v ⋅ conv_du

    # Derivatives for strong operators
    div_visc_du = strong_viscous_operator(form.viscous_operator, du, α, ν)
    R_du = conv_du + α * ∇(dp) + σ * du + (dsigma_du_val * u) - div_visc_du
    R_dp = eps_val * dp + div_alpha_du

    # Recompute base residual since stabilizing jacobian needs R(u_old)
    R_u_old = eval_strong_residual_u(form, u, p, h, α, f, c_1, c_2)
    R_p_old = eval_strong_residual_p(form, u, p, α, g_mass)
    
    L_u_star_v = strong_adjoint_momentum(form, u, v, q, α) - (σ * v)
    dL_du_star_v = α * (∇(v)' ⋅ du) - (dsigma_du_val * v)
    if isa(lin_mode, PicardMode)
        dL_du_star_v = 0.0 * (∇(v)' ⋅ du)
    end
        
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    proj_pi_u = pi_u === nothing ? (0.0 * u) : pi_u
    proj_pi_p = pi_p === nothing ? 0.0 : pi_p
    is_osgs = pi_u !== nothing
    
    stab_R_du = apply_jacobian_projection_u(form.projection_policy, R_du, σ, dsigma_du_val, u, du, is_osgs)
    stab_R_old_u = apply_projection_u(form.projection_policy, R_u_old, σ, u, proj_pi_u, is_osgs)
    
    stab_R_dp = apply_jacobian_projection_p(form.projection_policy, R_dp, eps_val, dp, is_osgs)
    stab_R_old_p = apply_projection_p(form.projection_policy, R_p_old, eps_val, p, proj_pi_p, is_osgs)

    stab_mom_jac = mult_mom * (L_u_star_v ⋅ (τ_1 * stab_R_du + dtau_1_du * stab_R_old_u) + dL_du_star_v ⋅ (τ_1 * stab_R_old_u))
    stab_mass_jac = mult_mass * (L_q_star * (τ_2 * stab_R_dp + dtau_2_du * stab_R_old_p))

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end

function strong_adjoint_momentum(form::AbstractFormulation, u, v, q, α)
    ν = form.ν
    conv_adj = α * (∇(v)' ⋅ u)
    pres_adj = α * ∇(q)
    visc_adj = adjoint_viscous_operator(form.viscous_operator, v, α, ν)
    return conv_adj + pres_adj + visc_adj
end
