# src/models/porosity.jl
#=
Spatially-varying porosity field α(x) for the Darcy-Brinkman-Forchheimer / porous
Navier-Stokes solver. α ∈ (0, 1] is the local volume fraction available to the fluid; it
enters the momentum equation through the convective term and the reaction tensor
σ(α, u) = a(α) + b(α)|u|. A varying α is what makes the manufactured-solution (MMS) forcing
nontrivial, so the field must supply not just α but its exact spatial derivatives ∇α, the
Laplacian ∇²α (scalar trace), and the full Hessian ∇²α (tensor) — all evaluated in closed
form so the MMS source has no finite-difference error.
=#
using Gridap

# Common supertype for every porosity model. Subtyping `Function` lets a field be wrapped
# directly as a Gridap `CellField` (it is callable: `field(x) -> α`).
abstract type AbstractPorosityField <: Function end

# Radially-symmetric porosity with a smooth logistic transition between two constant plateaus.
# It is α_0 inside the core radius r_1, α_∞ outside the outer radius r_2, and interpolates
# smoothly (C∞ in the open annulus) in between — giving a differentiable field whose exact
# derivatives can be written analytically for the MMS forcing.
#   alpha_0     — porosity at and inside the core (r ≤ r_1), the obstructed/packed region.
#   alpha_infty — far-field porosity (r ≥ r_2), typically 1.0 (free fluid).
#   r_1, r_2    — inner/outer radii of the transition annulus (r_1 < r_2), in domain length units.
struct SmoothRadialPorosity{T} <: AbstractPorosityField
    alpha_0::T
    alpha_infty::T
    r_1::T
    r_2::T
end

# Convenience constructor defaulting the far-field porosity to 1.0 (free fluid outside r_2).
function SmoothRadialPorosity(alpha_0::T, r_1::T, r_2::T) where T
    return SmoothRadialPorosity(alpha_0, one(T), r_1, r_2)
end

# Make it natively callable for CellField encapsulation: evaluating the field at a point
# returns the scalar porosity α(x).
(field::SmoothRadialPorosity)(x) = alpha(field, x)

# Hook the field into Gridap's gradient machinery so `∇(field)` yields the exact ∇α(x)
# rather than an automatic-differentiation approximation.
import Gridap.Fields: ∇
∇(field::SmoothRadialPorosity) = x -> grad_alpha(field, x)

# Evaluate the scalar porosity α at point x. The radial profile is a logistic blend between
# the two plateaus: a normalized coordinate η ∈ (0,1) measures progress across the annulus
# (in terms of r², which keeps the algebra rational), and γ(η) maps η onto the logistic's
# argument so that the transition is symmetric and saturates smoothly at both ends. Outside
# the annulus the result is the constant plateau; the γ saturation guards (±100) clamp to a
# plateau before exp(γ) overflows, since the logistic is numerically indistinguishable from
# its limit there.
function alpha(field::SmoothRadialPorosity, x)
    alpha_0 = field.alpha_0
    alpha_infty = field.alpha_infty
    r1 = field.r_1
    r2 = field.r_2

    r_sq = x[1]^2 + x[2]^2
    r = sqrt(r_sq)

    if r <= r1
        return alpha_0
    elseif r >= r2
        return alpha_infty
    end

    eta = (r_sq - r1^2) / (r2^2 - r1^2)
    gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))

    if gamma_val > 100.0
        return alpha_infty
    elseif gamma_val < -100.0
        return alpha_0
    end

    return alpha_infty - (alpha_infty - alpha_0) / (1.0 + exp(gamma_val))
end

# Single-pass evaluator that returns the porosity together with all of its spatial
# derivatives in one call: (α, ∇α, ∇²α_scalar, ∇²α_tensor). It is the analytic kernel the
# public `grad_alpha`/`lap_alpha`/`hess_alpha` accessors share, so the chain rule is written
# once and stays consistent across all consumers (the MMS forcing relies on all four being
# mutually exact).
#
# Because α depends on x only through the radius r, every Cartesian derivative is built by
# composing scalar one-dimensional chain-rule pieces:
#   α(γ(η(r)))  ⇒  dα/dr = (dα/dγ)(dγ/dη)(dη/dr),  d²α/dr² via the second-derivative chain rule,
# then promoted to ∇α and the Laplacian/Hessian using the standard radial-field identities.
# Inside the core / outside the annulus α is constant, so all derivatives vanish; the same
# ±100 saturation guard as `alpha` returns flat derivatives where the logistic has saturated.
function _analyze_alpha(field::SmoothRadialPorosity, x)
    alpha_0 = field.alpha_0
    alpha_infty = field.alpha_infty
    r1 = field.r_1
    r2 = field.r_2

    r_sq = x[1]^2 + x[2]^2
    r = sqrt(r_sq)

    if r <= r1 || r >= r2
        return alpha(field, x), VectorValue(0.0, 0.0), 0.0, TensorValue(0.0, 0.0, 0.0, 0.0)
    end

    eta = (r_sq - r1^2) / (r2^2 - r1^2)
    gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))

    if gamma_val > 100.0 || gamma_val < -100.0
        return alpha(field, x), VectorValue(0.0, 0.0), 0.0, TensorValue(0.0, 0.0, 0.0, 0.0)
    end

    exp_g = exp(gamma_val)
    alpha_val = alpha_infty - (alpha_infty - alpha_0) / (1.0 + exp_g)

    # η and γ as functions of r, and their first/second r-derivatives (η is rational in r²,
    # γ is rational in η).
    deta_dr = 2.0 * r / (r2^2 - r1^2)
    d2eta_dr2 = 2.0 / (r2^2 - r1^2)

    dgamma_deta = (2.0*eta^2 - 2.0*eta + 1.0) / (eta^2 * (1.0 - eta)^2)
    d2gamma_deta2 = (4.0*eta - 2.0) / (eta^2 * (1.0 - eta)^2) - 2.0 * (2.0*eta^2 - 2.0*eta + 1.0)*(1.0 - 2.0*eta) / (eta^3 * (1.0 - eta)^3)

    # Logistic derivatives w.r.t. its argument γ.
    da_dg = (alpha_infty - alpha_0) * exp_g / (1.0 + exp_g)^2
    d2a_dg2 = (alpha_infty - alpha_0) * exp_g * (1.0 - exp_g) / (1.0 + exp_g)^3

    # Compose into the radial derivatives dα/dr and d²α/dr².
    da_dr = da_dg * dgamma_deta * deta_dr
    d2a_dr2 = d2a_dg2 * (dgamma_deta * deta_dr)^2 + da_dg * (d2gamma_deta2 * deta_dr^2 + dgamma_deta * d2eta_dr2)

    # Radial-field identities: ∇²α = f'' + f'/r (2D Laplacian of a radial f), and ∇α = f'(r) x̂.
    lap_alpha_val = d2a_dr2 + (1.0/r) * da_dr
    grad_alpha_val = VectorValue(da_dr * (x[1]/r), da_dr * (x[2]/r))

    # Exact Hessian of the radial field α(r): for f(r), ∂_i∂_j f =
    #   f''(r) x_i x_j / r² + f'(r) (δ_ij / r − x_i x_j / r³).
    # It is symmetric, and its trace equals f'' + f'/r = lap_alpha_val above — these two must
    # stay consistent because the MMS forcing uses the Laplacian and the Hessian together.
    x1 = x[1]; x2 = x[2]
    r3 = r_sq * r
    H11 = d2a_dr2 * (x1*x1) / r_sq + da_dr * (1.0/r - (x1*x1)/r3)
    H22 = d2a_dr2 * (x2*x2) / r_sq + da_dr * (1.0/r - (x2*x2)/r3)
    H12 = d2a_dr2 * (x1*x2) / r_sq - da_dr * (x1*x2)/r3
    hess_alpha_val = TensorValue(H11, H12, H12, H22)
    return alpha_val, grad_alpha_val, lap_alpha_val, hess_alpha_val
end

# Exact porosity gradient ∇α (2-vector). Drives Gridap's `∇(field)` and the MMS source terms.
function grad_alpha(field::SmoothRadialPorosity, x)
    return _analyze_alpha(field, x)[2]
end

# Exact scalar Laplacian ∇²α = ∂²α/∂x² + ∂²α/∂y².
function lap_alpha(field::SmoothRadialPorosity, x)
    return _analyze_alpha(field, x)[3]
end

# Exact Hessian ∇²α (symmetric 2×2 TensorValue). Used by the manufactured-solution
# forcing to evaluate ∇(∇·u_ex) analytically rather than by finite differences.
function hess_alpha(field::SmoothRadialPorosity, x)
    return _analyze_alpha(field, x)[4]
end
