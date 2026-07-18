# src/stabilization/tau.jl
#
# Element-local VMS stabilization parameters τ₁ (momentum) and τ₂ (continuity),
# plus their Exact-Newton derivatives wrt the velocity u. τ₁/τ₂ scale the subgrid
# (fine-scale) contribution that makes the equal-order velocity/pressure pair
# inf-sup stable; they are evaluated pointwise at each quadrature node and feed
# the stabilized weak form (ASGS and OSGS alike).
#
# Definitions follow paper eq:Tau1, eq:Tau2 and eq:TauNavierStokes. The shared
# building block is the Navier-Stokes parameter
#     τ_{1,NS} = ( c₁ ν/h² + c₂ |u|/h )⁻¹           (eq:TauNavierStokes)
# which balances viscous diffusion (c₁ ν/h²) against convective transport
# (c₂ |u|/h). Then
#     τ₁ = ( α/τ_{1,NS} + σ )⁻¹                      (eq:Tau1, with σ = ρ_{Λ⁻¹}(σ))
#     τ₂ = h² / ( c₁ α τ_{1,NS} )                    (eq:Tau2)
# so the porous α and the reaction (drag) σ both sharpen the momentum scale τ₁.
#
# Conventions used throughout:
#   kin        — KinematicState at the point: kin.u (velocity), kin.mag_u (|u|).
#   med        — MediumState at the point: med.h (element size), med.alpha (α).
#   ν          — kinematic viscosity.
#   c_1, c_2   — equal-order interpolation constants (c₁=4k⁴, c₂=2k² via get_c1_c2).
#   tau_reg_lim — tau_regularization_limit: a small floor added to each inverse
#                 denominator so τ stays finite as |u|→0 and σ→0; without it the
#                 reciprocals diverge at wall stagnation.
#   rxn_law    — AbstractReactionLaw supplying σ via sigma / dsigma_du.
using Gridap
using Gridap.Algebra

# [TAU-05] τ_{1,NS}⁻¹ = c₁ν/h² + c₂|u|/h + reg-floor — the bracketed inverse denominator of
# eq:TauNavierStokes (A_NS). Single source of truth shared by compute_tau_1 / compute_tau_2 and their
# u-derivatives, so the four cannot drift in how they form the NS scale.
#
# [no-hard-coded-parameters] There is deliberately NO knob here scaling the viscous eigenvalue c₁ν/h².
# A `TAU_VISC_MULT` environment variable formerly did so (removed 2026-07-17). It was a diagnostic for the
# deviatoric spectral-radius question — ρ(∇·(2ανεᵈ)) is (4/3)αν|k|² in 3D (longitudinal mode) vs αν|k|² for
# the Laplacian, and EXACTLY αν|k|² in 2D — whose verdict was that the 3D remedy is an element-aware c₁, not
# a 4/3 viscous correction (docs/mms/p2-3d.md §A; lessons_learned.md 2026-07-06). It was inert by default
# (1.0), but it violated the rule three ways: not in the schema, not threaded, and — being read from the
# ambient environment — absent from every result's provenance, so a stored result could not reconstruct the
# τ that produced it. Any c₁ rescaling must instead come through `stabilization_c1_multiplier`, which is
# config-borne and recorded (see src/config.jl and config/porous_ns.schema.json).
@inline _tau_ns_inv(mag_u, h, ν, c_1, c_2, tau_reg_lim) =
    (c_1 * ν / (h * h)) + (c_2 * mag_u / h) + tau_reg_lim

# τ₁ — the momentum-equation stabilization parameter (eq:Tau1). Builds τ_{1,NS}
# (eq:TauNavierStokes), folds in the porous fraction α and the reaction σ, and
# returns the reciprocal of the combined inverse scale. Larger drag σ or finer
# meshes shrink τ₁.
function compute_tau_1(kin, med, ν, c_1, c_2, tau_reg_lim, rxn_law::AbstractReactionLaw)
    mag_u = kin.mag_u
    τ_1_NS_val = 1.0 / _tau_ns_inv(mag_u, med.h, ν, c_1, c_2, tau_reg_lim)
    sig_val = sigma(rxn_law, kin, med, mag_u)
    return 1.0 / ( (med.alpha / τ_1_NS_val) + sig_val + tau_reg_lim )
end

# τ₂ — the continuity (mass-equation) stabilization parameter (eq:Tau2),
# τ₂ = h² / (c₁ α τ_{1,NS}). It controls the pressure-stabilizing grad-div /
# subgrid-pressure term; unlike τ₁ it depends on σ only through τ_{1,NS}.
function compute_tau_2(kin, med, ν, c_1, c_2, tau_reg_lim)
    mag_u = kin.mag_u
    τ_1_NS_val = 1.0 / _tau_ns_inv(mag_u, med.h, ν, c_1, c_2, tau_reg_lim)
    return (med.h * med.h) / ( (c_1 * med.alpha * τ_1_NS_val) + tau_reg_lim )
end

# ∂τ₁/∂u in the direction du — the Exact-Newton derivative of τ₁ wrt the velocity.
# τ₁ depends on u twice: through the convective term c₂|u|/h inside τ_{1,NS}, and
# through σ(α,u) when the reaction law is velocity-dependent (Forchheimer). The
# chain rule gives dτ₁ = -τ₁² · d(τ₁⁻¹), and d(τ₁⁻¹) = α·d(A_NS) + dσ, where A_NS
# is τ_{1,NS}⁻¹. This term is included only by ExactNewtonMode; Picard freezes it.
#
# [known-fragility] When freeze_cusp is set, return an exact zero (typed via
# kin.u⋅du_val so Gridap keeps the right field type) — this is the Picard-style
# "freeze ∂τ/∂u" contract, not a perf shortcut. [debugging-lore] mag_u_reg adds the
# config-driven `deriv_floor` (ε_d = velocity_magnitude_derivative_floor) before dividing by |u|, because
# d|u|/du = u/|u| is singular at stagnation (perfect Dirichlet walls give |u|=0) and would otherwise yield
# NaN. See theory/velocity_floor_regularization/.
function compute_dtau_1_du(kin, med, du_val, ν, c_1, c_2, tau_reg_lim, freeze_cusp, rxn_law::AbstractReactionLaw, deriv_floor)
    if freeze_cusp
         return 0.0 * (kin.u ⋅ du_val)
    end
    u_val = kin.u
    h_val = med.h
    alpha_val = med.alpha
    mag_u = kin.mag_u

    # A_NS = τ_{1,NS}⁻¹, the bracketed inverse of eq:TauNavierStokes (with reg floor).
    A_NS_val = _tau_ns_inv(mag_u, h_val, ν, c_1, c_2, tau_reg_lim)
    τ_1_NS_val = 1.0 / A_NS_val

    sig_val = sigma(rxn_law, kin, med, mag_u)
    τ_1_val = 1.0 / ( (alpha_val * A_NS_val) + sig_val + tau_reg_lim )

    mag_u_reg = mag_u + deriv_floor
    # dA_NS = (c₂/h)·d|u|, with d|u| = (u·du)/|u| (floored denominator; deriv_floor = ε_d).
    dA_NS_du = (c_2 / h_val) / mag_u_reg * (u_val ⋅ du_val)
    dsig_du = dsigma_du(rxn_law, kin, med, mag_u, du_val, deriv_floor)
    
    # d(τ₁⁻¹) = α·dA_NS + dσ; then dτ₁ = -τ₁²·d(τ₁⁻¹).
    dTauInv_du = alpha_val * dA_NS_du + dsig_du
    
    return - (τ_1_val * τ_1_val) * dTauInv_du
end

# ∂τ₂/∂u in the direction du — the Exact-Newton derivative of τ₂ wrt the velocity.
# τ₂ = h²/(c₁ α τ_{1,NS}) depends on u only through τ_{1,NS} = A_NS⁻¹, so the chain
# of derivatives runs τ₂ → A_NS → |u| → u: differentiate w.r.t. A_NS, then A_NS
# w.r.t. |u| (= c₂/h), then |u| w.r.t. u (= u/|u|). scalar_deriv collects the first
# two stages; the final (u·du)/|u| direction is applied at the return.
#
# [known-fragility] freeze_cusp returns a typed zero (Picard contract). [debugging-lore]
# mag_u_reg floors |u| by the config-driven `deriv_floor` (ε_d) to keep the d|u|/du = u/|u| factor finite
# at stagnation. denom is τ₂'s inverse-scale denominator carrying the reg floor.
function compute_dtau_2_du(kin, med, du_val, ν, c_1, c_2, tau_reg_lim, freeze_cusp, deriv_floor)
    if freeze_cusp
        return 0.0 * (kin.u ⋅ du_val)
    end
    h_val = med.h
    alpha_val = med.alpha
    mag_u = kin.mag_u
    u_val = kin.u

    A_NS_val = _tau_ns_inv(mag_u, h_val, ν, c_1, c_2, tau_reg_lim)
    dA_NS_dmag = c_2 / h_val
    denom = c_1 * alpha_val / A_NS_val + tau_reg_lim

    mag_u_reg = mag_u + deriv_floor
    scalar_deriv = (h_val * h_val) / (denom * denom) * (c_1 * alpha_val) / (A_NS_val * A_NS_val) * dA_NS_dmag / mag_u_reg
    return scalar_deriv * (u_val ⋅ du_val)
end
