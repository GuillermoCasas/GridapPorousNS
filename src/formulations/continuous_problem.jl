# src/formulations/continuous_problem.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

abstract type AbstractFormulation end
abstract type AbstractLinearizationMode end

struct ExactNewtonMode <: AbstractLinearizationMode end
struct PicardMode <: AbstractLinearizationMode end

# Typed Callable Operators to prevent anonymous closure duplication

struct SigOp{R<:AbstractReactionLaw, Reg<:AbstractVelocityRegularization} <: Function
    law::R
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
end
function (op::SigOp)(u_v, grad_v, a_v, grad_a_v, h_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return sigma(op.law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag)
end

struct DSigOp{R<:AbstractReactionLaw, Reg<:AbstractVelocityRegularization} <: Function
    law::R
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
end
function (op::DSigOp)(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return dsigma_du(op.law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag, du_v)
end

struct Tau1Op{R<:AbstractReactionLaw, Reg<:AbstractVelocityRegularization} <: Function
    law::R
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
end
function (op::Tau1Op)(u_v, grad_v, a_v, grad_a_v, h_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return compute_tau_1(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.law)
end

struct Tau2Op{Reg<:AbstractVelocityRegularization} <: Function
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
end
function (op::Tau2Op)(u_v, grad_v, a_v, grad_a_v, h_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return compute_tau_2(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), op.ν, op.c_1, op.c_2, op.tau_reg_lim)
end

struct DTau1Op{R<:AbstractReactionLaw, Reg<:AbstractVelocityRegularization} <: Function
    law::R
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
    freeze_cusp::Bool
end
function (op::DTau1Op)(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return compute_dtau_1_du(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), du_v, op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.freeze_cusp, op.law)
end

struct DTau2Op{Reg<:AbstractVelocityRegularization} <: Function
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
    freeze_cusp::Bool
end
function (op::DTau2Op)(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return compute_dtau_2_du(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), du_v, op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.freeze_cusp)
end

struct PaperGeneralFormulation{V<:AbstractViscousOperator, R<:AbstractReactionLaw, P<:AbstractProjectionPolicy, Reg<:AbstractVelocityRegularization} <: AbstractFormulation
    viscous_operator::V
    reaction_law::R
    projection_policy::P
    regularization::Reg
    ν::Float64
    eps_val::Float64
    
    function PaperGeneralFormulation(v::V, r::R, p_in::P, reg::Reg, ν::Float64, eps_val::Float64, eps_floor::Float64=1e-8; autocorrect_policy=false) where {V, R, P, Reg}
        valid_policy = sanitize_projection_policy(p_in, r; autocorrect=autocorrect_policy)
        safe_eps = max(eps_val, eps_floor)
        new{V, R, typeof(valid_policy), Reg}(v, r, valid_policy, reg, ν, safe_eps)
    end
end

struct Legacy90d5749Mode{R<:AbstractReactionLaw, P<:AbstractProjectionPolicy, Reg<:AbstractVelocityRegularization} <: AbstractFormulation
    viscous_operator::LaplacianPseudoTractionViscosity
    reaction_law::R
    projection_policy::P
    regularization::Reg
    ν::Float64
    eps_val::Float64
    
    function Legacy90d5749Mode(r::R, p_in::P, reg::Reg, ν::Float64, eps_val::Float64, eps_floor::Float64=1e-8; autocorrect_policy=false) where {R, P, Reg}
        valid_policy = sanitize_projection_policy(p_in, r; autocorrect=autocorrect_policy)
        safe_eps = max(eps_val, eps_floor)
        new{R, typeof(valid_policy), Reg}(LaplacianPseudoTractionViscosity(), r, valid_policy, reg, ν, safe_eps)
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

# ==============================================================================
# Discretization Consistent Parameters
# ==============================================================================

function compute_consistent_quadrature_degree(k_velocity::Int)
    return 4 * k_velocity
end

function compute_stabilization_constants(k_velocity::Int)
    c_1 = 4.0 * k_velocity^4
    c_2 = 2.0 * k_velocity^2
    return c_1, c_2
end

get_quadrature_degree(::Type{PaperGeneralFormulation}, k_velocity::Int) = compute_consistent_quadrature_degree(k_velocity)
get_quadrature_degree(::Type{Legacy90d5749Mode}, k_velocity::Int) = compute_consistent_quadrature_degree(k_velocity)

get_c1_c2(::Type{PaperGeneralFormulation}, k_velocity::Int) = compute_stabilization_constants(k_velocity)
get_c1_c2(::Type{Legacy90d5749Mode}, k_velocity::Int) = compute_stabilization_constants(k_velocity)

# Reusable Operators for Type Stability
_get_dsigma_du_val(::ExactNewtonMode, law, u, α, h, du, reg, ν, c_1, c_2) = Operation(DSigOp(law, reg, ν, c_1, c_2))(u, ∇(u), α, ∇(α), h, du)
_get_dsigma_du_val(::PicardMode, law, u, α, h, du, reg, ν, c_1, c_2) = 0.0 * (u ⋅ du)

_get_dtau_1_du(::ExactNewtonMode, op, u, α, h, du) = op(u, ∇(u), α, ∇(α), h, du)
_get_dtau_1_du(::PicardMode, op, u, α, h, du) = 0.0 * (u ⋅ du)

_get_dtau_2_du(::ExactNewtonMode, op, u, α, h, du) = op(u, ∇(u), α, ∇(α), h, du)
_get_dtau_2_du(::PicardMode, op, u, α, h, du) = 0.0 * (u ⋅ du)

_get_conv_du(::ExactNewtonMode, α, u, du) = α * (∇(du)' ⋅ u) + α * (∇(u)' ⋅ du)
_get_conv_du(::PicardMode, α, u, du) = α * (∇(du)' ⋅ u)

_get_proj_pi_u(pi_u, u) = pi_u
_get_proj_pi_u(::Nothing, u) = 0.0 * u

_get_proj_pi_p(pi_p) = pi_p
_get_proj_pi_p(::Nothing) = 0.0

# Reusable Residual evaluations
function eval_strong_residual_u(form::AbstractFormulation, u, p, h, α, f_custom, c_1, c_2)
    ν = form.ν
    conv_u = α * (∇(u)' ⋅ u)
    div_visc_u = strong_viscous_operator(form.viscous_operator, u, α, ν)
    grad_u_dummy = ∇(u)
    
    function _sigma_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return sigma(form.reaction_law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag)
    end
    
    σ = Operation(_sigma_closure)(u, grad_u_dummy, α, ∇(α), h)
    
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

    grad_u_dummy = ∇(u)
    
    function _sigma_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return sigma(form.reaction_law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag)
    end
    function _tau1_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return compute_tau_1(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), ν, c_1, c_2, tau_reg_lim, form.reaction_law)
    end
    function _tau2_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return compute_tau_2(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), ν, c_1, c_2, tau_reg_lim)
    end
    
    σ = Operation(_sigma_closure)(u, grad_u_dummy, α, ∇(α), h)
    τ_1 = Operation(_tau1_closure)(u, grad_u_dummy, α, ∇(α), h)
    τ_2 = Operation(_tau2_closure)(u, grad_u_dummy, α, ∇(α), h)

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
    proj_pi_u = _get_proj_pi_u(pi_u, u)
    proj_pi_p = _get_proj_pi_p(pi_p)
    
    stab_R_u = apply_projection_u(form.projection_policy, R_u, σ, u, proj_pi_u, pi_u !== nothing)
    stab_R_p = apply_projection_p(form.projection_policy, R_p, eps_val, p, proj_pi_p, pi_p !== nothing)

    stab_mom = mult_mom * (L_u_star_v ⋅ (τ_1 * stab_R_u))
    stab_mass = mult_mass * (L_q_star * (τ_2 * stab_R_p))

    return ∫( conv_term + visc_term + pres_term + res_term + mass_term - src_term + stab_mom + stab_mass )dΩ
end

function build_picard_jacobian(X, dX, Y, form::AbstractFormulation, dΩ, h, f_custom, alpha_custom, g_custom, pi_u, pi_p, c_1, c_2, tau_reg_lim; mult_mom=1.0, mult_mass=1.0)
    u, p = X; du, dp = dX; v, q = Y
    α = alpha_custom
    f = f_custom
    g_mass = g_custom
    ν = form.ν
    eps_val = form.eps_val

    grad_u_dummy = ∇(u)
    
    # Pure native evaluations via lightweight localized closures to avoid deep AST stacking
    function _sigma_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return sigma(form.reaction_law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag)
    end
    function _tau1_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return compute_tau_1(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), ν, c_1, c_2, tau_reg_lim, form.reaction_law)
    end
    function _tau2_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return compute_tau_2(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), ν, c_1, c_2, tau_reg_lim)
    end
    
    σ = Operation(_sigma_closure)(u, grad_u_dummy, α, ∇(α), h)
    τ_1 = Operation(_tau1_closure)(u, grad_u_dummy, α, ∇(α), h)
    τ_2 = Operation(_tau2_closure)(u, grad_u_dummy, α, ∇(α), h)
    
    # Picard linearizes convective acceleration as purely u ⋅ ∇(du)
    conv_du = α * (∇(du)' ⋅ u)
    
    visc_term_jac = weak_viscous_jacobian(form.viscous_operator, du, v, α, ν)
    pres_term_jac = - dp * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term_jac  = v ⋅ ( σ * du )
    
    div_alpha_du = α * (∇⋅du) + du ⋅ ∇(α)
    mass_term_jac = q * (eps_val * dp + div_alpha_du)
    
    conv_term_jac = v ⋅ conv_du

    div_visc_du = strong_viscous_operator(form.viscous_operator, du, α, ν)
    R_du = conv_du + α * ∇(dp) + σ * du - div_visc_du
    R_dp = eps_val * dp + div_alpha_du

    L_u_star_v = strong_adjoint_momentum(form, u, v, q, α) - (σ * v)
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    proj_pi_u = _get_proj_pi_u(pi_u, u)
    proj_pi_p = _get_proj_pi_p(pi_p)
    is_osgs = pi_u !== nothing
    
    stab_R_du = apply_jacobian_projection_u(form.projection_policy, R_du, σ, 0.0, u, du, is_osgs)
    stab_R_dp = apply_jacobian_projection_p(form.projection_policy, R_dp, eps_val, dp, is_osgs)

    # In Picard, dtau = 0 and dL_du_star = 0 completely eliminating all catastrophic cross-terms AST
    stab_mom_jac = mult_mom * (L_u_star_v ⋅ (τ_1 * stab_R_du))
    stab_mass_jac = mult_mass * (L_q_star * (τ_2 * stab_R_dp))

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end

function build_stabilized_weak_form_jacobian(X, dX, Y, form::AbstractFormulation, dΩ, h, f_custom, alpha_custom, g_custom, pi_u, pi_p, c_1, c_2, tau_reg_lim, freeze_cusp, lin_mode::AbstractLinearizationMode=ExactNewtonMode(); mult_mom=1.0, mult_mass=1.0)
    u, p = X; du, dp = dX; v, q = Y
    α = alpha_custom
    f = f_custom
    g_mass = g_custom
    ν = form.ν
    eps_val = form.eps_val

    grad_u_dummy = ∇(u)
    
    function _sigma_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return sigma(form.reaction_law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag)
    end
    function _dsigma_closure(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return dsigma_du(form.reaction_law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag, du_v)
    end
    function _tau1_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return compute_tau_1(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), ν, c_1, c_2, tau_reg_lim, form.reaction_law)
    end
    function _tau2_closure(u_v, grad_v, a_v, grad_a_v, h_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return compute_tau_2(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), ν, c_1, c_2, tau_reg_lim)
    end
    function _dtau1_closure(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return compute_dtau_1_du(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), du_v, ν, c_1, c_2, tau_reg_lim, freeze_cusp, form.reaction_law)
    end
    function _dtau2_closure(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
        mag = effective_speed(form.regularization, u_v, ν, h_v, c_1, c_2)
        return compute_dtau_2_du(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), du_v, ν, c_1, c_2, tau_reg_lim, freeze_cusp)
    end

    σ = Operation(_sigma_closure)(u, grad_u_dummy, α, ∇(α), h)
    
    if lin_mode isa PicardMode
        dsigma_du_val = 0.0 * (u ⋅ du)
        dtau_1_du = 0.0 * (u ⋅ du)
        dtau_2_du = 0.0 * (u ⋅ du)
    else
        dsigma_du_val = Operation(_dsigma_closure)(u, grad_u_dummy, α, ∇(α), h, du)
        dtau_1_du = Operation(_dtau1_closure)(u, grad_u_dummy, α, ∇(α), h, du)
        dtau_2_du = Operation(_dtau2_closure)(u, grad_u_dummy, α, ∇(α), h, du)
    end
    
    τ_1 = Operation(_tau1_closure)(u, grad_u_dummy, α, ∇(α), h)
    τ_2 = Operation(_tau2_closure)(u, grad_u_dummy, α, ∇(α), h)

    conv_du = _get_conv_du(lin_mode, α, u, du)
    dL_du_star_v = _get_dL_du_star_v(lin_mode, form, α, v, du, dsigma_du_val)
    
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
    dL_du_star_v = _get_dL_du_star_v(lin_mode, form, α, v, du, dsigma_du_val)
        
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    proj_pi_u = _get_proj_pi_u(pi_u, u)
    proj_pi_p = _get_proj_pi_p(pi_p)
    is_osgs = pi_u !== nothing
    
    stab_R_du = apply_jacobian_projection_u(form.projection_policy, R_du, σ, dsigma_du_val, u, du, is_osgs)
    stab_R_old_u = apply_projection_u(form.projection_policy, R_u_old, σ, u, proj_pi_u, is_osgs)
    
    stab_R_dp = apply_jacobian_projection_p(form.projection_policy, R_dp, eps_val, dp, is_osgs)
    stab_R_old_p = apply_projection_p(form.projection_policy, R_p_old, eps_val, p, proj_pi_p, is_osgs)

    stab_mom_jac = mult_mom * (L_u_star_v ⋅ (τ_1 * stab_R_du + dtau_1_du * stab_R_old_u) + dL_du_star_v ⋅ (τ_1 * stab_R_old_u))
    stab_mass_jac = mult_mass * (L_q_star * (τ_2 * stab_R_dp + dtau_2_du * stab_R_old_p))

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end

function strong_adjoint_momentum(form::PaperGeneralFormulation, u, v, q, α)
    ν = form.ν
    conv_adj = α * (∇(v)' ⋅ u)
    pres_adj = α * ∇(q)
    visc_adj = adjoint_viscous_operator(form.viscous_operator, v, α, ν)
    return conv_adj + pres_adj + visc_adj
end

function _get_dL_du_star_v(::ExactNewtonMode, form::PaperGeneralFormulation, α, v, du, dsigma_du_val)
    return α * (∇(v)' ⋅ du) - (dsigma_du_val * v)
end
function _get_dL_du_star_v(::PicardMode, form::PaperGeneralFormulation, α, v, du, dsigma_du_val)
    return 0.0 * (∇(v)' ⋅ du)
end

function strong_adjoint_momentum(form::Legacy90d5749Mode, u, v, q, α)
    ν = form.ν
    conv_adj = α * (∇(v)' ⋅ u)
    pres_adj = α * ∇(q)
    return conv_adj + pres_adj
end

function _get_dL_du_star_v(::ExactNewtonMode, form::Legacy90d5749Mode, α, v, du, dsigma_du_val)
    return α * (∇(v)' ⋅ du) - (dsigma_du_val * v)
end
function _get_dL_du_star_v(::PicardMode, form::Legacy90d5749Mode, α, v, du, dsigma_du_val)
    return 0.0 * (∇(v)' ⋅ du)
end
