# src/formulations/viscous_operators.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

abstract type AbstractViscousOperator end

struct DeviatoricSymmetricViscosity <: AbstractViscousOperator end
struct SymmetricGradientViscosity <: AbstractViscousOperator end
struct LaplacianPseudoTractionViscosity <: AbstractViscousOperator end

# =======================
# Laplacian/Pseudo-Traction
# =======================
function strong_viscous_operator(::LaplacianPseudoTractionViscosity, u, α, ν)
    return ν * (∇(u)' ⋅ ∇(α)) + α * ν * Δ(u)
end

function weak_viscous_operator(::LaplacianPseudoTractionViscosity, u, v, α, ν)
    return α * ν * ( ∇(u) ⊙ ∇(v) )
end

function weak_viscous_jacobian(::LaplacianPseudoTractionViscosity, du, v, α, ν)
    return α * ν * ( ∇(du) ⊙ ∇(v) )
end

function adjoint_viscous_operator(::LaplacianPseudoTractionViscosity, v, α, ν)
    return ν * (∇(v)' ⋅ ∇(α)) + α * ν * Δ(v)
end

# =======================
# Symmetric Gradient Viscosity
# =======================
function strong_viscous_operator(::SymmetricGradientViscosity, u, α, ν)
    # Note: ∇⋅(ε(u)) is mathematically equivalent to 0.5 * Δ(u) + 0.5 * ∇(∇⋅u)
    # Using the explicit form avoids Gridap tensor divergence limitations if any
    div_eps_u = 0.5 * Δ(u) + 0.5 * ∇(∇⋅u)
    return 2.0 * ν * (ε(u) ⋅ ∇(α)) + 2.0 * α * ν * div_eps_u
end

function weak_viscous_operator(::SymmetricGradientViscosity, u, v, α, ν)
    return 2.0 * α * ν * ( ε(u) ⊙ ε(v) )
end

function weak_viscous_jacobian(::SymmetricGradientViscosity, du, v, α, ν)
    return 2.0 * α * ν * ( ε(du) ⊙ ε(v) )
end

function adjoint_viscous_operator(::SymmetricGradientViscosity, v, α, ν)
    div_eps_v = 0.5 * Δ(v) + 0.5 * ∇(∇⋅v)
    return 2.0 * ν * (ε(v) ⋅ ∇(α)) + 2.0 * α * ν * div_eps_v
end

# =======================
# Deviatoric Symmetric Viscosity
# =======================
_dyn_dev_tensor(e) = e - (1.0 / size(e, 1)) * tr(e) * one(typeof(e))
_dyn_div_dev_tensor(lap_u, grad_div_u) = 0.5 * lap_u + 0.5 * grad_div_u - (1.0 / length(lap_u)) * grad_div_u

function strong_viscous_operator(::DeviatoricSymmetricViscosity, u, α, ν)
    # deviatoric(ε(u)) = ε(u) - 1/d * (∇⋅u) * I
    # ∇⋅deviatoric(ε(u)) = ∇⋅ε(u) - 1/d * ∇(∇⋅u)
    D_u = Operation(_dyn_dev_tensor)(ε(u))
    div_D_u = Operation(_dyn_div_dev_tensor)(Δ(u), ∇(∇⋅u))
    return 2.0 * ν * (D_u ⋅ ∇(α)) + 2.0 * α * ν * div_D_u
end

function weak_viscous_operator(::DeviatoricSymmetricViscosity, u, v, α, ν)
    D_u = Operation(_dyn_dev_tensor)(ε(u))
    return 2.0 * α * ν * ( D_u ⊙ ε(v) )
end

function weak_viscous_jacobian(::DeviatoricSymmetricViscosity, du, v, α, ν)
    D_du = Operation(_dyn_dev_tensor)(ε(du))
    return 2.0 * α * ν * ( D_du ⊙ ε(v) )
end

function adjoint_viscous_operator(::DeviatoricSymmetricViscosity, v, α, ν)
    D_v = Operation(_dyn_dev_tensor)(ε(v))
    div_D_v = Operation(_dyn_div_dev_tensor)(Δ(v), ∇(∇⋅v))
    return 2.0 * ν * (D_v ⋅ ∇(α)) + 2.0 * α * ν * div_D_v
end
