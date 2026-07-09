# src/formulations/continuous_problem.jl
#
# Assembles the stabilized VMS weak form (residual + Jacobian) for the porous
# Navier-Stokes / Darcy-Brinkman-Forchheimer system. This is the literal
# transcription of the OSGS problem `eq:OSGSProblem` (paper Eqs. 4.10a-d), with
# ASGS recovered when no projection is supplied (`π_h = 0`). Everything here is
# expressed as Gridap `CellField` algebra so the resulting `∫(...)dΩ` integrand
# can be assembled and differentiated by Gridap's symbolic machinery.
using Gridap
using Gridap.Algebra
using LinearAlgebra

abstract type AbstractFormulation end

# Selects which terms enter the Jacobian linearization.
abstract type AbstractLinearizationMode end

# Full Newton: includes the derivative-of-coefficient terms ∂σ/∂u, ∂τ/∂u,
# and the full convective derivative. Quadratic convergence near the root.
struct ExactNewtonMode <: AbstractLinearizationMode end
# Frozen-coefficient linearization: ∂σ/∂u, ∂τ/∂u and the reactive part of the
# convective derivative are dropped (treated as constant). Linear-only, more
# robust far from the root. [paper-faithful] These are distinct mathematical
# contracts, not a performance toggle — never relabel one as the other.
struct PicardMode <: AbstractLinearizationMode end

# ------------------------------------------------------------------------------
# Typed callable operators.
#
# Each struct below wraps a coefficient law (σ, τ₁, τ₂ and their u-derivatives)
# together with the scalar parameters it needs (ν, c₁, c₂, regularization),
# and is invoked pointwise by Gridap's `Operation(...)`. Using concrete callable
# structs instead of anonymous closures keeps the generated integrand AST
# type-stable and prevents Gridap from duplicating the closure body across the
# weak form. [debugging-lore]
# ------------------------------------------------------------------------------

# σ(α, u): the Darcy-Brinkman-Forchheimer reaction coefficient — the
# (symmetric positive-semidefinite) inverse-permeability tensor, here a scalar
# (`eq:DBFResistanceTerm`) — evaluated pointwise.
struct SigOp{R<:AbstractReactionLaw, Reg<:AbstractVelocityRegularization} <: Function
    law::R
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
end
@inline function (op::SigOp)(u_v, grad_v, a_v, grad_a_v, h_v)
    # `mag` = the velocity magnitude |u| that feeds the Forchheimer term σ = a + b|u|.
    # [known-fragility] Use `reaction_speed` (constant velocity floor only), NOT
    # `effective_speed` (which adds the mesh-dependent diffusive floor belonging to τ).
    # The diffusive floor here injects an O(ν/h) drag that destroys h-convergence for
    # varying-porosity Forchheimer flows.
    mag = reaction_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return sigma(op.law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag)
end

# ∂σ/∂u · du: directional derivative of the reaction tensor, the ExactNewton
# linearization of the Forchheimer drag.
struct DSigOp{R<:AbstractReactionLaw, Reg<:AbstractVelocityRegularization} <: Function
    law::R
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
end
@inline function (op::DSigOp)(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
    # [known-fragility] Must differentiate σ at the SAME speed `reaction_speed` used in
    # `SigOp` (not `effective_speed`); otherwise the Jacobian no longer matches the
    # residual and ExactNewton loses its quadratic convergence.
    mag = reaction_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return dsigma_du(op.law, KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), mag, du_v, velocity_magnitude_derivative_floor(op.reg))
end

# τ₁ (`eq:Tau1`): the momentum stabilization parameter scaling the subgrid
# velocity. Here `mag` uses `effective_speed` — the velocity floor that INCLUDES
# the mesh-dependent diffusive term — because τ must reflect the local diffusive
# scale (this is the floor SigOp must avoid). `tau_reg_lim` is the lower bound
# that regularizes τ where the local speed vanishes.
struct Tau1Op{R<:AbstractReactionLaw, Reg<:AbstractVelocityRegularization} <: Function
    law::R
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
end
@inline function (op::Tau1Op)(u_v, grad_v, a_v, grad_a_v, h_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return compute_tau_1(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.law)
end

# τ₂ (`eq:Tau2`): the mass (pressure) stabilization parameter, paired with the
# strong continuity residual. Like τ₁ it uses the diffusive `effective_speed`.
struct Tau2Op{Reg<:AbstractVelocityRegularization} <: Function
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
end
@inline function (op::Tau2Op)(u_v, grad_v, a_v, grad_a_v, h_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return compute_tau_2(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), op.ν, op.c_1, op.c_2, op.tau_reg_lim)
end

# ∂τ₁/∂u · du: directional derivative of τ₁, an ExactNewton-only contribution.
# `freeze_cusp` zeroes the derivative across the non-smooth |u|→0 cusp of the
# parameter, trading exactness for Jacobian smoothness where τ is non-differentiable.
struct DTau1Op{R<:AbstractReactionLaw, Reg<:AbstractVelocityRegularization} <: Function
    law::R
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
    freeze_cusp::Bool
end
@inline function (op::DTau1Op)(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return compute_dtau_1_du(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), du_v, op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.freeze_cusp, op.law, velocity_magnitude_derivative_floor(op.reg))
end

# ∂τ₂/∂u · du: directional derivative of τ₂, the ExactNewton counterpart for the
# mass stabilization parameter (same `freeze_cusp` convention as DTau1Op).
struct DTau2Op{Reg<:AbstractVelocityRegularization} <: Function
    reg::Reg
    ν::Float64
    c_1::Float64
    c_2::Float64
    tau_reg_lim::Float64
    freeze_cusp::Bool
end
@inline function (op::DTau2Op)(u_v, grad_v, a_v, grad_a_v, h_v, du_v)
    mag = effective_speed(op.reg, u_v, op.ν, h_v, op.c_1, op.c_2)
    return compute_dtau_2_du(KinematicState(u_v, grad_v, mag), MediumState(a_v, grad_a_v, h_v), du_v, op.ν, op.c_1, op.c_2, op.tau_reg_lim, op.freeze_cusp, velocity_magnitude_derivative_floor(op.reg))
end

# The canonical formulation. Bundles the four pluggable physics/numerics policies
# that together define the weak form:
#   `viscous_operator`   — how the viscous stress is written (canonical:
#                          DeviatoricSymmetricViscosity, ∇·(2μ∇ˢu)).
#   `reaction_law`       — the σ(α,u) law (ForchheimerErgunLaw | ConstantSigmaLaw).
#   `projection_policy`  — ASGS (identity) vs OSGS (orthogonal projection) handling
#                          of the strong residual in the stabilization term.
#   `regularization`     — velocity floor for the |u| appearing in σ and τ.
# `ν` is the kinematic viscosity; `physical_epsilon ≥ 0` is the pressure-stabilization /
# grad-div coefficient in the mass equation (physical_epsilon·p). The constructor rejects
# negative `physical_epsilon` and runs `sanitize_projection_policy` so the projection
# policy is consistent with the chosen reaction law.
struct PaperGeneralFormulation{V<:AbstractViscousOperator, R<:AbstractReactionLaw, P<:AbstractProjectionPolicy, Reg<:AbstractVelocityRegularization} <: AbstractFormulation
    viscous_operator::V
    reaction_law::R
    projection_policy::P
    regularization::Reg
    ν::Float64
    physical_epsilon::Float64             # ε_phys: physical compressibility, in BOTH residual and Jacobian (mass LHS)
    numerical_epsilon::Float64   # ε_num: Codina iterative penalty. Lagging ε_num·p to the iterate makes it
                                 # CANCEL in the residual and survive only as ε_num·dp in the JACOBIAN — a
                                 # pressure-block regularization that vanishes at convergence (no consistency
                                 # error, no outer loop). 0 ⇒ off (Jacobian == residual ε).

    function PaperGeneralFormulation(v::V, r::R, p_in::P, reg::Reg, ν::Float64, physical_epsilon::Float64;
                                     numerical_epsilon::Float64=0.0, autocorrect_policy=false) where {V, R, P, Reg}
        physical_epsilon >= 0.0 || throw(ArgumentError("physical_epsilon must be nonnegative; got $physical_epsilon"))
        numerical_epsilon >= 0.0 || throw(ArgumentError("numerical_epsilon must be nonnegative; got $numerical_epsilon"))
        valid_policy = sanitize_projection_policy(p_in, r; autocorrect=autocorrect_policy)
        new{V, R, typeof(valid_policy), Reg}(v, r, valid_policy, reg, ν, physical_epsilon, numerical_epsilon)
    end
end

# Accessors for the pluggable policies a formulation carries, used by the solver
# and projection code without depending on the concrete formulation type.
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
# Discretization-consistent parameters
#
# Quadrature degree and the stabilization constants c₁, c₂ are derived from the
# velocity interpolation order k_velocity so they scale correctly with the FE
# space; nothing here is hand-tuned.
# ==============================================================================

# Base quadrature degree: 4·k_v exactly integrates the highest-degree products
# that appear in the (nonlinear) stabilized integrand for order-k_v velocity.
function compute_consistent_quadrature_degree(k_velocity::Int)
    return 4 * k_velocity
end

# The stabilization constants for equal-order interpolation: c₁ = 4·k⁴, c₂ = 2·k²
# (paper Remark after `eq:conditions_on_num_param`). They enter τ₁ and τ₂.
function compute_stabilization_constants(k_velocity::Int)
    c_1 = 4.0 * k_velocity^4
    c_2 = 2.0 * k_velocity^2
    return c_1, c_2
end

# Reaction-law-aware quadrature degree (primary API, §3.5). Each reaction law
# declares a `min_quadrature_degree` (default 0) to cover any non-polynomial
# integrand it introduces; the actual degree is the `max` of that and the base
# rule. Prefer this 3-arg form.
get_quadrature_degree(::Type{<:PaperGeneralFormulation}, k_velocity::Int, rxn_law::AbstractReactionLaw) =
    max(compute_consistent_quadrature_degree(k_velocity), min_quadrature_degree(rxn_law, k_velocity))

# Reaction-law-agnostic fallback. Returns the base rule only and so misses any
# reaction-specific bump (e.g. Forchheimer's ⌊k_v/2⌋). Use only where the caller
# genuinely cannot supply a reaction law at the quadrature decision point.
get_quadrature_degree(::Type{<:PaperGeneralFormulation}, k_velocity::Int) = compute_consistent_quadrature_degree(k_velocity)

# Exposes c₁, c₂ keyed on the formulation type (the equal-order rule).
get_c1_c2(::Type{<:PaperGeneralFormulation}, k_velocity::Int) = compute_stabilization_constants(k_velocity)

# ------------------------------------------------------------------------------
# Mode-dispatched coefficient derivatives.
#
# Each helper returns the ExactNewton derivative term, or a dimension-correct
# zero in PicardMode. The Picard zeros (`0.0 * (u ⋅ du)`, `0.0 * (∇(du)' ⋅ u)`)
# are written as products of live fields so the result is a CellField of the
# right shape — a bare `0.0` would not assemble into the Jacobian integrand.
# Freezing these terms is exactly what makes Picard a frozen-coefficient method.
# ------------------------------------------------------------------------------
_get_dsigma_du_val(::ExactNewtonMode, law, u, α, h, du, reg, ν, c_1, c_2) = Operation(DSigOp(law, reg, ν, c_1, c_2))(u, ∇(u), α, ∇(α), h, du)
_get_dsigma_du_val(::PicardMode, law, u, α, h, du, reg, ν, c_1, c_2) = 0.0 * (u ⋅ du)

_get_dtau_1_du(::ExactNewtonMode, law, u, α, h, du, reg, ν, c_1, c_2, tau_reg_lim, freeze_cusp) = Operation(DTau1Op(law, reg, ν, c_1, c_2, tau_reg_lim, freeze_cusp))(u, ∇(u), α, ∇(α), h, du)
_get_dtau_1_du(::PicardMode, law, u, α, h, du, reg, ν, c_1, c_2, tau_reg_lim, freeze_cusp) = 0.0 * (u ⋅ du)

_get_dtau_2_du(::ExactNewtonMode, u, α, h, du, reg, ν, c_1, c_2, tau_reg_lim, freeze_cusp) = Operation(DTau2Op(reg, ν, c_1, c_2, tau_reg_lim, freeze_cusp))(u, ∇(u), α, ∇(α), h, du)
_get_dtau_2_du(::PicardMode, u, α, h, du, reg, ν, c_1, c_2, tau_reg_lim, freeze_cusp) = 0.0 * (u ⋅ du)

# Linearized convective acceleration α·(u·∇)u. ExactNewton keeps both terms from
# the product rule; Picard keeps only the transport-of-increment term α·(u·∇)du
# and freezes the reactive term α·(du·∇)u.
_get_conv_du(::ExactNewtonMode, α, u, du) = α * (∇(du)' ⋅ u) + α * (∇(u)' ⋅ du)
_get_conv_du(::PicardMode, α, u, du) = α * (∇(du)' ⋅ u)

# Select the supplied OSGS projection (π_h) or, for ASGS, a dimension-correct
# zero (identity projection ⇒ π_h = 0, the full strong residual is stabilized).
_get_proj_pi_u(pi_u, u) = pi_u
_get_proj_pi_u(::Nothing, u) = 0.0 * u

_get_proj_pi_p(pi_p) = pi_p
_get_proj_pi_p(::Nothing) = 0.0

# [FORM-05] Build the per-cell VMS stabilization coefficient CellFields (σ, τ₁, τ₂) and unpack the
# setup/formulation scalars that the residual and BOTH Jacobian modes share verbatim. Centralizing it here
# is the single source of truth for HOW σ/τ are constructed from the typed operators, so the three builders
# can never drift in their coefficient definitions. `u` is the velocity field (unpacked from X by the
# caller). Returns a NamedTuple the callers destructure with `(; …) = _build_stabilization_coefficients(…)`.
function _build_stabilization_coefficients(u, setup, formulation, phys_cfg)
    form = formulation.form
    ν = form.ν
    physical_epsilon = form.physical_epsilon
    c_1 = formulation.c_1
    c_2 = formulation.c_2
    tau_reg_lim = phys_cfg.tau_regularization_limit
    α = setup.alpha_cf
    h = setup.h_cf
    grad_u_dummy = ∇(u)

    # Pointwise coefficients σ, τ₁, τ₂ as CellFields built from the typed operators.
    sig_op = SigOp(form.reaction_law, form.regularization, ν, c_1, c_2)
    tau1_op = Tau1Op(form.reaction_law, form.regularization, ν, c_1, c_2, tau_reg_lim)
    tau2_op = Tau2Op(form.regularization, ν, c_1, c_2, tau_reg_lim)
    σ = Operation(sig_op)(u, grad_u_dummy, α, ∇(α), h)
    τ_1 = Operation(tau1_op)(u, grad_u_dummy, α, ∇(α), h)
    τ_2 = Operation(tau2_op)(u, grad_u_dummy, α, ∇(α), h)

    return (; form, ν, physical_epsilon, c_1, c_2, tau_reg_lim,
              α, f=setup.f_cf, g_mass=setup.g_cf, h, dΩ=setup.dΩ,
              σ, τ_1, τ_2)
end

# Strong (pointwise) momentum residual R_u = α(u·∇)u + α∇p + σu − ∇·(2μ∇ˢu) − f,
# i.e. the left side of `eq:StrongMomentumEquation` minus the forcing. This is the
# quantity the stabilization weights against the momentum adjoint. `σ` and the
# viscous divergence may be passed in to reuse already-built CellFields (and to
# stop Gridap re-expanding their AST); otherwise they are constructed here.
function eval_strong_residual_u(form::AbstractFormulation, u, p, h, α, f_custom, c_1, c_2; σ=nothing, div_visc_u=nothing)
    ν = form.ν
    conv_u = α * (∇(u)' ⋅ u)

    if div_visc_u === nothing
        div_visc_u = strong_viscous_operator(form.viscous_operator, u, α, ν)
    end

    if σ === nothing
        grad_u_dummy = ∇(u)
        sig_op = SigOp(form.reaction_law, form.regularization, ν, c_1, c_2)
        σ = Operation(sig_op)(u, grad_u_dummy, α, ∇(α), h)
    end

    return conv_u + α * ∇(p) + σ * u - div_visc_u - f_custom
end

# Strong (pointwise) mass residual R_p = physical_epsilon·p + ∇·(αu) − g, the left side of
# `eq:StrongMassEquation` (with the physical_epsilon·p pressure-stabilization term) minus
# the mass source. ∇·(αu) is expanded as α(∇·u) + u·∇α for variable porosity α.
function eval_strong_residual_p(form::AbstractFormulation, u, p, α, g_custom)
    physical_epsilon = form.physical_epsilon
    div_alpha_u = α * (∇⋅u) + u ⋅ ∇(α)
    return physical_epsilon * p + div_alpha_u - g_custom
end

# Builds the full stabilized weak residual integrand for the (u,p) system —
# the nonlinear form Gridap drives to zero. It is the sum of the standard Galerkin
# terms and the VMS stabilization terms of `eq:OSGSProblem`.
#
# Arguments: `X = (u, p)` trial fields, `Y = (v, q)` test fields; `setup` carries
# the precomputed CellFields (porosity α, momentum source f, mass source g, mesh
# size h, measure dΩ); `formulation` wraps the formulation `form` plus the cached
# c₁, c₂; `phys_cfg` supplies the τ regularization limit. `pi_u`/`pi_p` are the
# OSGS projections π_h — passing them switches the stabilization from ASGS (their
# default `nothing`) to OSGS. `mult_mom`/`mult_mass` scale the momentum/mass
# stabilization blocks (used for homotopy dilution).
function build_stabilized_weak_form_residual(X, Y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0, p_prev=nothing)
    u, p = X; v, q = Y
    # [FORM-05] σ/τ₁/τ₂ + the setup/formulation scalars come from the shared coefficient helper.
    (; form, ν, physical_epsilon, c_1, c_2, α, f, g_mass, h, dΩ, σ, τ_1, τ_2) =
        _build_stabilization_coefficients(u, setup, formulation, phys_cfg)

    # Standard Galerkin terms tested against (v, q): convection, viscous stress,
    # pressure (integrated by parts, hence the −p ∇·(αv) form), Forchheimer
    # reaction σu, the mass/continuity equation, and the source forcing.
    conv_term = v ⋅ (α * (∇(u)' ⋅ u))
    visc_term = weak_viscous_operator(form.viscous_operator, u, v, α, ν)
    pres_term = - p * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term  = v ⋅ ( σ * u )

    div_alpha_u = α * (∇⋅u) + u ⋅ ∇(α)
    # ε_num (Codina ITERATIVE PENALTY, article.tex §5.2 line ~1383): the mass equation carries ε_num·pⁿ on
    # the LHS and ε_num·pⁿ⁻¹ (the PREVIOUS nonlinear iterate's pressure, `p_prev`) lagged onto the RHS, so the
    # net residual contribution is ε_num·(pⁿ − pⁿ⁻¹). It is NONZERO during the iterations (pinning the
    # constant-pressure null mode → well-posedness, required for the 3D case where ε=0 is ill-posed) and
    # vanishes at convergence (pⁿ = pⁿ⁻¹), so the manufactured/converged solution is NOT altered. Its exact
    # linearization ε_num·dp is already in the Jacobian (build_stabilized_weak_form_jacobian, mass_term_jac),
    # so residual+Jacobian are consistent. p_prev === nothing ⇒ no iterative penalty (byte-identical legacy).
    iter_penalty = p_prev === nothing ? (0.0 * p) : (form.numerical_epsilon * (p - p_prev))
    mass_term = q * (physical_epsilon * p + div_alpha_u + iter_penalty)
    src_term  = v ⋅ f + q * g_mass

    # Strong residuals R_u, R_p that the stabilization weights — these measure how
    # far the discrete solution misses the strong PDE pointwise.
    R_u = eval_strong_residual_u(form, u, p, h, α, f, c_1, c_2; σ=σ)
    R_p = eval_strong_residual_p(form, u, p, α, g_mass)

    # Adjoint operators L*(v,q) applied to the test functions. The convective
    # adjoint enters strong_adjoint_momentum with a +α(∇v')·u sign and the σv is
    # SUBTRACTED here, matching the B_S definition under `eq:OSGSProblem`.
    # [known-fragility] Flipping either sign produces anti-SUPG / anti-diffusion
    # and destroys coercivity at parameter extremes.
    L_u_star_v = strong_adjoint_momentum(form, u, v, q, α) - (σ * v)
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - physical_epsilon * q

    # Resolve the projections: ASGS ⇒ dimension-correct zeros, OSGS ⇒ π_h.
    proj_pi_u = _get_proj_pi_u(pi_u, u)
    proj_pi_p = _get_proj_pi_p(pi_p)

    # OSGS stabilizes the ORTHOGONAL residual R − π_h(R); ASGS stabilizes R itself.
    # The final boolean tells the policy which branch it is in.
    stab_R_u = apply_projection_u(form.projection_policy, R_u, σ, u, proj_pi_u, pi_u !== nothing)
    stab_R_p = apply_projection_p(form.projection_policy, R_p, physical_epsilon, p, proj_pi_p, pi_p !== nothing)

    # The VMS stabilization terms: adjoint-of-test ⋅ (τ · projected strong residual).
    stab_mom = mult_mom * (L_u_star_v ⋅ (τ_1 * stab_R_u))
    stab_mass = mult_mass * (L_q_star * (τ_2 * stab_R_p))

    return ∫( conv_term + visc_term + pres_term + res_term + mass_term - src_term + stab_mom + stab_mass )dΩ
end

# The Picard (frozen-coefficient) Jacobian integrand: the bilinear form in the increment `dX = (du, dp)`
# tested against `Y = (v, q)`, with σ, τ₁, τ₂ and the convective adjoint frozen at the current iterate
# (u, p) — a smaller, more robust (linear-only) tangent than ExactNewton.
#
# [FORM-01] This is exactly the general builder in `PicardMode()`: there the ∂σ/∂u, ∂τ/∂u and dL*/∂u
# cross-terms collapse to structural zeros (×0.0), so the assembled matrices are byte-for-byte identical
# (guarded by test/blitz/picard_jacobian_equivalence_blitz_test.jl). It is kept as a named entry point for
# the OSGS Picard fallback / Stage-I boot. `freeze_cusp=false` is irrelevant: PicardMode zeroes the dτ
# terms regardless of it. (docs/formulation-audit-2026-06-24.md §D, FORM-01.)
function build_picard_jacobian(X, dX, Y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    return build_stabilized_weak_form_jacobian(X, dX, Y, setup, formulation, phys_cfg, false, PicardMode();
                                               pi_u=pi_u, pi_p=pi_p, mult_mom=mult_mom, mult_mass=mult_mass)
end

# Builds the general stabilized Jacobian integrand. With `lin_mode = ExactNewtonMode()`
# (the default) it is the full consistent tangent of the residual, including the
# ∂σ/∂u, ∂τ/∂u and dL*/∂u derivative terms; with PicardMode those terms collapse
# to zero. `freeze_cusp` is forwarded to the dτ operators to zero their derivative
# across the |u|→0 cusp. The mode dispatch is centralized in the `_get_*` helpers
# so this body reads identically for both contracts. Argument roles match the
# residual/Picard builders; `dX = (du, dp)` is the increment.
function build_stabilized_weak_form_jacobian(X, dX, Y, setup, formulation, phys_cfg, freeze_cusp, lin_mode::AbstractLinearizationMode=ExactNewtonMode(); pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    u, p = X; du, dp = dX; v, q = Y
    # [FORM-05] σ/τ₁/τ₂ + the setup/formulation scalars come from the shared coefficient helper.
    (; form, ν, physical_epsilon, c_1, c_2, tau_reg_lim, α, f, g_mass, h, dΩ, σ, τ_1, τ_2) =
        _build_stabilization_coefficients(u, setup, formulation, phys_cfg)

    # Mode-dependent coefficient derivatives: nonzero CellFields in ExactNewton,
    # dimension-correct zeros in Picard.
    dsigma_du_val = _get_dsigma_du_val(lin_mode, form.reaction_law, u, α, h, du, form.regularization, ν, c_1, c_2)
    dtau_1_du = _get_dtau_1_du(lin_mode, form.reaction_law, u, α, h, du, form.regularization, ν, c_1, c_2, tau_reg_lim, freeze_cusp)
    dtau_2_du = _get_dtau_2_du(lin_mode, u, α, h, du, form.regularization, ν, c_1, c_2, tau_reg_lim, freeze_cusp)

    conv_du = _get_conv_du(lin_mode, α, u, du)
    dL_du_star_v = _get_dL_du_star_v(lin_mode, form, α, v, du, dsigma_du_val)

    # Galerkin tangent terms. The reaction tangent carries both the frozen σ·du and
    # the ExactNewton (dσ/∂u)·u contribution.
    visc_term_jac = weak_viscous_jacobian(form.viscous_operator, du, v, α, ν)
    pres_term_jac = - dp * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term_jac  = v ⋅ ( σ * du + (dsigma_du_val * u) )

    div_alpha_du = α * (∇⋅du) + du ⋅ ∇(α)
    # ε_num (Codina iterative penalty): lagging ε_num·p to the iterate cancels it in the residual and
    # leaves ε_num·dp ONLY here, a pressure-block regularization that vanishes at convergence. The VMS
    # subscale (R_dp, L_q_star) stays at ε_phys so it remains residual-consistent (no consistency error).
    mass_term_jac = q * ((physical_epsilon + form.numerical_epsilon) * dp + div_alpha_du)

    conv_term_jac = v ⋅ conv_du

    # Strong residual of the increment, including the dσ/∂u term for ExactNewton.
    div_visc_du = strong_viscous_operator(form.viscous_operator, du, α, ν)
    R_du = conv_du + α * ∇(dp) + σ * du + (dsigma_du_val * u) - div_visc_du
    R_dp = physical_epsilon * dp + div_alpha_du

    # The dτ stabilization tangent multiplies the residual at the CURRENT iterate,
    # so R(u_old) is needed too. `σ` is passed explicitly into the strong residual
    # so Gridap reuses the already-built CellField instead of re-expanding its AST.
    R_u_old = eval_strong_residual_u(form, u, p, h, α, f, c_1, c_2; σ=σ)
    R_p_old = eval_strong_residual_p(form, u, p, α, g_mass)

    # Adjoint of the test functions. Same sign discipline as the residual builder.
    # (dL_du_star_v — the derivative-of-adjoint term, zero in Picard — was already built above right
    # after dsigma_du_val; it is reused at the stabilization tangent below, no need to recompute it.)
    L_u_star_v = strong_adjoint_momentum(form, u, v, q, α) - (σ * v)

    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - physical_epsilon * q

    proj_pi_u = _get_proj_pi_u(pi_u, u)
    proj_pi_p = _get_proj_pi_p(pi_p)
    is_osgs = pi_u !== nothing

    # Project both the increment residual (with its dσ/∂u term) and the old residual
    # through the OSGS/ASGS policy.
    stab_R_du = apply_jacobian_projection_u(form.projection_policy, R_du, σ, dsigma_du_val, u, du, is_osgs)
    stab_R_old_u = apply_projection_u(form.projection_policy, R_u_old, σ, u, proj_pi_u, is_osgs)

    stab_R_dp = apply_jacobian_projection_p(form.projection_policy, R_dp, physical_epsilon, dp, is_osgs)
    stab_R_old_p = apply_projection_p(form.projection_policy, R_p_old, physical_epsilon, p, proj_pi_p, is_osgs)

    # Full stabilization tangent (product rule over τ·R and L*·R):
    #   L* ⋅ (τ·R_du  +  dτ·R_old)   +   dL* ⋅ (τ·R_old).
    # In Picard the dτ and dL* pieces vanish, recovering the build_picard_jacobian form.
    stab_mom_jac = mult_mom * (L_u_star_v ⋅ (τ_1 * stab_R_du + dtau_1_du * stab_R_old_u) + dL_du_star_v ⋅ (τ_1 * stab_R_old_u))
    stab_mass_jac = mult_mass * (L_q_star * (τ_2 * stab_R_dp + dtau_2_du * stab_R_old_p))

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end

# The momentum adjoint operator L*_mom applied to the test function, minus the
# reaction part (which the callers subtract separately as σ·v). It sums:
#   convective adjoint  +α(∇v')·u  — [known-fragility] the POSITIVE sign is required
#       because the stabilization bilinear form B_S subtracts the adjoint; the
#       (1/α)∇·(αa)v term is intentionally omitted to preserve the A²−B² symmetry
#       in the stability estimate (paper §5);
#   pressure adjoint    α∇q;
#   viscous adjoint     from the chosen viscous operator.
function strong_adjoint_momentum(form::PaperGeneralFormulation, u, v, q, α)
    ν = form.ν
    conv_adj = α * (∇(v)' ⋅ u)
    pres_adj = α * ∇(q)
    visc_adj = adjoint_viscous_operator(form.viscous_operator, v, α, ν)
    # [diagnostic] env-gated multiplier on the viscous adjoint L*_visc(v) in the stabilization pairing
    # τ₁(R_u, L*(v)). Read ONCE per assembly (not per quad point). Default "1.0" ⇒ byte-identical. Set
    # VISC_ADJ_MULT=0 to DROP the viscous adjoint, =-1 to FLIP its sign — tests whether the viscous
    # 2nd-derivative subscale coupling is the P2-3D destabilizer (docs/mms/p2-3d.md §A).
    _vam = parse(Float64, get(ENV, "VISC_ADJ_MULT", "1.0"))
    return conv_adj + pres_adj + _vam * visc_adj
end

# dL*/∂u · du: derivative of the momentum adjoint w.r.t. the linearization point.
# ExactNewton: convective part α(∇v')·du minus the reaction-derivative part
# (dσ/∂u)·v. Picard: zero (the adjoint is frozen).
function _get_dL_du_star_v(::ExactNewtonMode, form::PaperGeneralFormulation, α, v, du, dsigma_du_val)
    return α * (∇(v)' ⋅ du) - (dsigma_du_val * v)
end
function _get_dL_du_star_v(::PicardMode, form::PaperGeneralFormulation, α, v, du, dsigma_du_val)
    return 0.0 * (∇(v)' ⋅ du)
end

