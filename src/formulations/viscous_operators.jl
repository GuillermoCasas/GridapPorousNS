# src/formulations/viscous_operators.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

"""
    AbstractViscousOperator

Abstract mathematical type representing the viscous stress tensor divergence operator 
``-2\\nabla \\cdot ( \\alpha \\nu \\ViscProj \\nabla \\boldsymbol{u})`` 
from Eq. (2.1, 199) in `article.tex`. The precise form depends on the chosen 
projection operator ``\\ViscProj`` (e.g., Deviatoric-Symmetric).
"""
abstract type AbstractViscousOperator end

"""
    DeviatoricSymmetricViscosity

[paper-faithful] Baseline canonical formulation corresponding to ``\\ViscProj = \\DPi\\SPi``.
In this formulation (used in `PaperGeneralFormulation`), the viscous term uses the 
deviatoric part of the symmetric strain rate tensor. 
See `article.tex` Eq. 206.
"""
struct DeviatoricSymmetricViscosity <: AbstractViscousOperator end

"""
    SymmetricGradientViscosity

[code-actual] Corresponds to ``\\ViscProj = \\SPi``, dropping the deviatoric projection.
This formulation is mathematically related to certain DBF equations; see `article.tex` Eq. 210.
Note that the actual implemented strong operator drops the ``\\nabla(\\nabla \\cdot u)`` term 
to appease Gridap compilation limits, making its strong form algebraically approximate for 
compressible or inexactly divergence-free fields.
"""
struct SymmetricGradientViscosity <: AbstractViscousOperator end

"""
    LaplacianPseudoTractionViscosity

[legacy] Unphysical simplification where the viscous term is algebraically reduced to 
``-\\nabla \\cdot (\\alpha \\nu \\nabla \\boldsymbol{u})``. Kept purely for 
historical regression testing and stabilization baseline comparisons. Never to be used 
as the authoritative formulation.
"""
struct LaplacianPseudoTractionViscosity <: AbstractViscousOperator end

# =======================
# Laplacian/Pseudo-Traction
# =======================
# [legacy] Standard scalar Laplacian applied component-wise.
# Mathematical contract: returns `+ ∇⋅(α*ν*∇u)`. Note the implicit application 
# of the product rule: `ν*(∇u' ⋅ ∇α) + α*ν*Δu`.
function strong_viscous_operator(::LaplacianPseudoTractionViscosity, u, α, ν)
    return ν * (∇(u)' ⋅ ∇(α)) + α * ν * Δ(u)
end

# [legacy] Standard inner product weak form `(α*ν*∇u, ∇v)`.
# The function returns the integrand without the integration weights.
function weak_viscous_operator(::LaplacianPseudoTractionViscosity, u, v, α, ν)
    return α * ν * ( ∇(u) ⊙ ∇(v) )
end

# Linear exact Fréchet derivative of the weak operator with respect to `u`.
# [must-test] Any changes to `weak_viscous_operator` require an exact match here, verified via the porousns-fast-verification skill.
function weak_viscous_jacobian(::LaplacianPseudoTractionViscosity, du, v, α, ν)
    return α * ν * ( ∇(du) ⊙ ∇(v) )
end

# Formal adjoint of the viscous operator, needed for ASGS/OSGS stabilization weighting.
# The formal continuous adjoint of `-∇⋅(α*ν*∇u)` is symmetric, hence it matches the strong operator.
function adjoint_viscous_operator(::LaplacianPseudoTractionViscosity, v, α, ν)
    return ν * (∇(v)' ⋅ ∇(α)) + α * ν * Δ(v)
end

# =======================
# Symmetric Gradient Viscosity
# =======================
# [code-actual] Computes the positive evaluation `2*∇⋅(α*ν*ε(u))` for the strong residual.
function strong_viscous_operator(::SymmetricGradientViscosity, u, α, ν)
    # [debugging-lore] Note: ∇⋅(ε(u)) is mathematically equivalent to 0.5 * Δ(u) + 0.5 * ∇(∇⋅u).
    # Using the explicit divergence form avoids Gridap tensor divergence limitations.
    # Furthermore, `∇(∇⋅u)` fails on Gridap Lagrangian continuous elements due to a lack 
    # of full Hessian support. Thus, we explicitly drop it here. 
    # This implies the strong operator evaluated in the ASGS subgrid-scale residual is mathematically approximate.
    div_eps_u = 0.5 * Δ(u) # + 0.5 * ∇(∇⋅u)
    return 2.0 * ν * (ε(u) ⋅ ∇(α)) + 2.0 * α * ν * div_eps_u
end

function weak_viscous_operator(::SymmetricGradientViscosity, u, v, α, ν)
    return 2.0 * α * ν * ( ε(u) ⊙ ε(v) )
end

# [must-test] Exact Fréchet derivative of the symmetric gradient weak form.
function weak_viscous_jacobian(::SymmetricGradientViscosity, du, v, α, ν)
    return 2.0 * α * ν * ( ε(du) ⊙ ε(v) )
end

# [code-actual] Formal continuous adjoint is `-2*∇⋅(α*ν*ε(v))`, returned with positive sign.
function adjoint_viscous_operator(::SymmetricGradientViscosity, v, α, ν)
    # [debugging-lore] `∇(∇⋅v)` fails on Gridap Lagrangian elements due to lack of full Hessian support.
    # In 2D incompressible or analytically divergence-free limits, we can mathematically drop it.
    # Otherwise, this omission represents a loss of exact formal mathematical symmetry in the VMS stabilization.
    div_eps_v = 0.5 * Δ(v) # + 0.5 * ∇(∇⋅v)
    return 2.0 * ν * (ε(v) ⋅ ∇(α)) + 2.0 * α * ν * div_eps_v
end

# =======================
# Deviatoric Symmetric Viscosity
# =======================

# [paper-faithful] Type-stable Callable Operators for AST compiler health.
struct EvalDevSymOp <: Function end
@inline function (::EvalDevSymOp)(∇u::TensorValue{D, D, T}) where {D, T}
    eps_u = 0.5 * (∇u + transpose(∇u))
    return eps_u - (1.0 / D) * tr(∇u) * one(∇u)
end

struct EvalDivDevSymOp <: Function end
@inline function (::EvalDivDevSymOp)(Δu::VectorValue{2, T1}, ∇∇u::ThirdOrderTensorValue{2,2,2,T2}) where {T1, T2}
    grad_div_u = VectorValue(∇∇u[1,1,1] + ∇∇u[1,2,2], ∇∇u[2,1,1] + ∇∇u[2,2,2])
    return 0.5 * Δu + 0.0 * grad_div_u
end
@inline function (::EvalDivDevSymOp)(Δu::VectorValue{3, T1}, ∇∇u::ThirdOrderTensorValue{3,3,3,T2}) where {T1, T2}
    grad_div_u = VectorValue(
        ∇∇u[1,1,1] + ∇∇u[1,2,2] + ∇∇u[1,3,3], 
        ∇∇u[2,1,1] + ∇∇u[2,2,2] + ∇∇u[2,3,3],
        ∇∇u[3,1,1] + ∇∇u[3,2,2] + ∇∇u[3,3,3]
    )
    return 0.5 * Δu + (0.5 - 1.0/3.0) * grad_div_u
end

# [paper-faithful] Computes `2*∇⋅(α*ν*D(u))` where `D(u) = ε(u) - (1/d)*(∇⋅u)*I`.
function strong_viscous_operator(::DeviatoricSymmetricViscosity, u, α, ν)
    D_u = Operation(EvalDevSymOp())(∇(u))
    div_D_u = Operation(EvalDivDevSymOp())(Δ(u), ∇∇(u))
    
    return 2.0 * ν * (D_u ⋅ ∇(α)) + 2.0 * α * ν * div_D_u
end

function weak_viscous_operator(::DeviatoricSymmetricViscosity, u, v, α, ν)
    D_u = Operation(EvalDevSymOp())(∇(u))
    return 2.0 * α * ν * ( D_u ⊙ ε(v) )
end

# Exact Fréchet derivative.
# [must-test] This must match the analytical Fréchet derivative of `weak_viscous_operator` exactly for `ExactNewtonMode`.
function weak_viscous_jacobian(::DeviatoricSymmetricViscosity, du, v, α, ν)
    D_du = Operation(EvalDevSymOp())(∇(du))
    return 2.0 * α * ν * ( D_du ⊙ ε(v) )
end

function adjoint_viscous_operator(::DeviatoricSymmetricViscosity, v, α, ν)
    D_v = Operation(EvalDevSymOp())(∇(v))
    div_D_v = Operation(EvalDivDevSymOp())(Δ(v), ∇∇(v))
    
    # [paper-faithful] This implies the VMS continuous formal adjoint is evaluated EXACTLY
    # across any dimension without losing geometric rigor or AST parity.
    return 2.0 * ν * (D_v ⋅ ∇(α)) + 2.0 * α * ν * div_D_v
end
