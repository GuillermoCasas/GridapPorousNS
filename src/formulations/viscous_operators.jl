# src/formulations/viscous_operators.jl
#
# The viscous stress term of the porous Navier-Stokes / Darcy-Brinkman-Forchheimer
# momentum equation, in four flavours. Each concrete `AbstractViscousOperator` supplies a
# consistent quartet of integrands used elsewhere in the formulation:
#   - `strong_viscous_operator`  : the strong-form divergence (drives the stabilization residual),
#   - `weak_viscous_operator`    : the Galerkin bilinear integrand (the test-function pairing),
#   - `weak_viscous_jacobian`    : its exact Fréchet derivative w.r.t. u (for Exact Newton),
#   - `adjoint_viscous_operator` : the formal continuous adjoint L*v (the stabilization weighting).
# The strong/adjoint forms feed the VMS subgrid-scale terms; the weak form/Jacobian feed the
# assembled monolithic (u, p) system. `α` is the porosity field, `ν` the kinematic viscosity.
using Gridap
using Gridap.Algebra
using Gridap.TensorValues   # ThirdOrderTensorValue (the velocity-Hessian type used by the grad-div ops)
using LinearAlgebra

"""
    AbstractViscousOperator

The viscous stress-tensor divergence operator
``-2\\nabla \\cdot ( \\alpha \\nu \\ViscProj \\nabla \\boldsymbol{u})``
— the viscous stress term of the strong momentum equation (`eq:StrongMomentumEquation` in
`article.tex`). Concrete subtypes differ only in the projection
``\\ViscProj`` applied to the velocity gradient before taking the divergence (deviatoric-symmetric,
plain symmetric, or none), and dispatch selects the matching strong/weak/Jacobian/adjoint quartet.
"""
abstract type AbstractViscousOperator end

"""
    DeviatoricSymmetricViscosity

[paper-faithful] The canonical operator used by `PaperGeneralFormulation`:
``\\ViscProj = \\DPi\\SPi``, i.e. the deviatoric (trace-free) part of the symmetric strain-rate
tensor ``\\SPi\\nabla\\boldsymbol{u} = \\varepsilon(\\boldsymbol{u})``. This is the physically
correct viscous stress for incompressible flow. See `article.tex` Eq. 206.
"""
struct DeviatoricSymmetricViscosity <: AbstractViscousOperator end

"""
    SymmetricGradientViscosity

[code-actual] Uses ``\\ViscProj = \\SPi`` — the full symmetric gradient ``\\varepsilon(\\boldsymbol{u})``
without removing the trace (deviatoric) part; see `article.tex` Eq. 210. Equivalent to the
deviatoric operator only for exactly divergence-free fields.

[known-fragility] The strong form keeps the full ``0.5\\,\\Delta u + 0.5\\,\\nabla(\\nabla\\cdot u)``
expansion of ``\\nabla\\cdot\\varepsilon(u)``; for fields that are not exactly divergence-free the
strong residual it produces therefore differs from the deviatoric variant by the trace term.
"""
struct SymmetricGradientViscosity <: AbstractViscousOperator end

"""
    LaplacianPseudoTractionViscosity

[legacy] Component-wise vector Laplacian, ``-\\nabla \\cdot (\\alpha \\nu \\nabla \\boldsymbol{u})``,
i.e. no symmetric/deviatoric projection at all. Physically incorrect (it implies a pseudo-traction
boundary condition rather than the true viscous traction); kept only as a regression and
stabilization-baseline comparison. Do not use it as the authoritative formulation.
"""
struct LaplacianPseudoTractionViscosity <: AbstractViscousOperator end

# =======================
# Laplacian/Pseudo-Traction
# =======================
# [legacy] Strong form of the component-wise Laplacian: the divergence `∇⋅(α*ν*∇u)`.
# The product rule splits it into a porosity-gradient term `ν*(∇u' ⋅ ∇α)` plus the bulk
# diffusion `α*ν*Δu`; the first term vanishes for constant porosity α.
function strong_viscous_operator(::LaplacianPseudoTractionViscosity, u, α, ν)
    return ν * (∇(u)' ⋅ ∇(α)) + α * ν * Δ(u)
end

# [legacy] Galerkin weak integrand `(α*ν*∇u, ∇v)` (the gradient–gradient pairing), returned
# without integration measure — the caller integrates it over Ω.
function weak_viscous_operator(::LaplacianPseudoTractionViscosity, u, v, α, ν)
    return α * ν * ( ∇(u) ⊙ ∇(v) )
end

# Exact Fréchet derivative of the weak integrand w.r.t. u. Because that integrand is linear in u,
# the Jacobian is structurally identical with `du` swapped in for `u`.
# [must-test] Any change to `weak_viscous_operator` must be mirrored here exactly (Exact Newton).
function weak_viscous_jacobian(::LaplacianPseudoTractionViscosity, du, v, α, ν)
    return α * ν * ( ∇(du) ⊙ ∇(v) )
end

# Formal continuous adjoint L*v, the weighting applied to the test function in the ASGS/OSGS
# stabilization term. The plain-Laplacian operator `-∇⋅(α*ν*∇·)` is self-adjoint, so L*v has
# the same shape as the strong operator evaluated on v.
function adjoint_viscous_operator(::LaplacianPseudoTractionViscosity, v, α, ν)
    return ν * (∇(v)' ⋅ ∇(α)) + α * ν * Δ(v)
end

# ∇(∇·u) extracted from the velocity Hessian H (H[i,j,k] = ∂_i ∂_j u_k, so component i = Σ_j H[i,j,j]).
# [VISC-01] Single shared contraction for the symmetric AND deviatoric divergence operators (2D & 3D),
# replacing four byte-identical inline copies (formerly the dead ContractGradDivOp in gridap_extensions.jl).
@inline _grad_div(H::ThirdOrderTensorValue{2,2,2,T}) where T =
    VectorValue(H[1,1,1] + H[1,2,2], H[2,1,1] + H[2,2,2])
@inline _grad_div(H::ThirdOrderTensorValue{3,3,3,T}) where T =
    VectorValue(H[1,1,1] + H[1,2,2] + H[1,3,3], H[2,1,1] + H[2,2,2] + H[2,3,3], H[3,1,1] + H[3,2,2] + H[3,3,3])

# =======================
# Symmetric Gradient Viscosity
# =======================
# [code-actual] Pointwise callable that maps the velocity Hessian to ∇·ε(u). For the symmetric
# gradient, ∇·ε(u) = 0.5·Δu + 0.5·∇(∇·u); the grad-div coefficient is 0.5 in EVERY dimension (no
# trace removal), so one D-generic method covers 2D and 3D. Used by both the strong and adjoint operators.
struct EvalStrongViscSymOp <: Function end
@inline function (::EvalStrongViscSymOp)(Δu::VectorValue{D, T1}, ∇∇u::ThirdOrderTensorValue{D,D,D,T2}) where {D, T1, T2}
    grad_div_u = _grad_div(∇∇u)
    return 0.5 * Δu + 0.5 * grad_div_u
end

function strong_viscous_operator(::SymmetricGradientViscosity, u, α, ν)
    # Strong form 2*∇⋅(α*ν*ε(u)): a porosity-gradient term plus the bulk divergence 2*α*ν*∇·ε(u).
    # [debugging-lore] Gridap's AST cannot expand ∇(∇·u) directly as an Operation, but it does
    # evaluate exact element Hessians (∇∇). We therefore feed the Laplacian Δ(u) and Hessian ∇∇(u)
    # into the type-stable EvalStrongViscSymOp callable, which assembles ∇·ε(u) analytically.
    div_eps_u = Operation(EvalStrongViscSymOp())(Δ(u), ∇∇(u))
    return 2.0 * ν * (ε(u) ⋅ ∇(α)) + 2.0 * α * ν * div_eps_u
end

# Galerkin weak integrand 2*α*ν*(ε(u) : ε(v)): the symmetric-strain double contraction.
function weak_viscous_operator(::SymmetricGradientViscosity, u, v, α, ν)
    return 2.0 * α * ν * ( ε(u) ⊙ ε(v) )
end

# [must-test] Exact Fréchet derivative of the weak integrand; linear in u, so `du` replaces `u`.
function weak_viscous_jacobian(::SymmetricGradientViscosity, du, v, α, ν)
    return 2.0 * α * ν * ( ε(du) ⊙ ε(v) )
end

# [code-actual] Formal continuous adjoint L*v = -2*∇⋅(α*ν*ε(v)), returned with positive sign
# (the stabilization bilinear form subtracts the adjoint, so the sign here must be positive).
function adjoint_viscous_operator(::SymmetricGradientViscosity, v, α, ν)
    # Per article.tex:479. Since this operator is self-adjoint, L*v reuses the strong-operator
    # shape on v: ∇·ε(v) = 0.5·Δv + 0.5·∇(∇·v), assembled by the same EvalStrongViscSymOp callable.
    div_eps_v = Operation(EvalStrongViscSymOp())(Δ(v), ∇∇(v))
    return 2.0 * ν * (ε(v) ⋅ ∇(α)) + 2.0 * α * ν * div_eps_v
end

# =======================
# Deviatoric Symmetric Viscosity
# =======================

# [paper-faithful] Two type-stable pointwise callables (kept as structs so Gridap's AST stays
# type-stable and compiles cleanly). EvalDevSymOp builds the deviatoric strain tensor from ∇u;
# EvalDivDevSymOp builds its divergence from the velocity Hessian.

# Maps ∇u -> deviatoric symmetric strain ε^d(u) = ε(u) − (1/D)·tr(∇u)·I, where ε(u) is the
# symmetric part and the subtracted trace term removes the volumetric (dilatation) component.
struct EvalDevSymOp <: Function end
@inline function (::EvalDevSymOp)(∇u::TensorValue{D, D, T}) where {D, T}
    eps_u = 0.5 * (∇u + transpose(∇u))
    return eps_u - (1.0 / D) * tr(∇u) * one(∇u)
end

# Maps (Δu, ∇∇u) -> ∇·ε^d(u). The deviatoric divergence expands to 0.5·Δu + (0.5 − 1/D)·∇(∇·u): the
# trace removal makes the grad-div coefficient DIMENSION-DEPENDENT (0 in 2D, +1/6 in 3D), unlike the
# plain symmetric case where it is uniformly 0.5.
# [VISC-01] The coefficient is computed as `0.5 − 1/D` from the Hessian's type parameter D — a
# compile-time constant, so it constant-folds — rather than hand-transcribed per dimension (`0.0` for 2D,
# `0.5 − 1/3` for 3D, byte-identical to those). One D-generic method ⇒ a new dimension cannot silently
# drift the coefficient; `_grad_div` still gates the actually-supported D ∈ {2, 3}.
struct EvalDivDevSymOp <: Function end
@inline function (::EvalDivDevSymOp)(Δu::VectorValue{D, T1}, ∇∇u::ThirdOrderTensorValue{D,D,D,T2}) where {D, T1, T2}
    grad_div_u = _grad_div(∇∇u)
    return 0.5 * Δu + (0.5 - 1.0/D) * grad_div_u
end

# [paper-faithful] Strong form 2*∇⋅(α*ν*\DPi\SPi\nabla u), with the deviatoric symmetric strain
# \DPi\SPi\nabla u = \SPi\nabla u − (1/d)(\nabla·u)I. Splits (product rule) into a porosity-gradient
# term plus the bulk divergence 2*α*ν*∇·ε^d(u); the two callables supply ε^d(u) and ∇·ε^d(u).
function strong_viscous_operator(::DeviatoricSymmetricViscosity, u, α, ν)
    ViscProj_u = Operation(EvalDevSymOp())(∇(u))
    div_ViscProj_u = Operation(EvalDivDevSymOp())(Δ(u), ∇∇(u))
    return 2.0 * ν * (ViscProj_u ⋅ ∇(α)) + 2.0 * α * ν * div_ViscProj_u
end

# Galerkin weak integrand 2*α*ν*(ε^d(u) : ε(v)): deviatoric strain of the trial field contracted
# against the symmetric strain of the test field.
function weak_viscous_operator(::DeviatoricSymmetricViscosity, u, v, α, ν)
    ViscProj_u = Operation(EvalDevSymOp())(∇(u))
    return 2.0 * α * ν * ( ViscProj_u ⊙ ε(v) )
end

# [must-test] Exact Fréchet derivative of the weak integrand (must match it exactly for
# `ExactNewtonMode`); linear in u, so `du` replaces `u`.
function weak_viscous_jacobian(::DeviatoricSymmetricViscosity, du, v, α, ν)
    ViscProj_du = Operation(EvalDevSymOp())(∇(du))
    return 2.0 * α * ν * ( ViscProj_du ⊙ ε(v) )
end

# Formal continuous adjoint L*v, the stabilization weighting. This operator is self-adjoint, so
# L*v reuses the strong-operator shape on v.
function adjoint_viscous_operator(::DeviatoricSymmetricViscosity, v, α, ν)
    ViscProj_v = Operation(EvalDevSymOp())(∇(v))
    # Per article.tex:479 `L*V = -∂_i(K_{ji}^T ∂_j V)`. The deviatoric divergence is
    # ∇·ε^d(v) = 0.5·Δv + (0.5 − 1/d)·∇(∇·v) (grad-div coefficient 0 in 2D, +1/6 in 3D),
    # supplied by the same EvalDivDevSymOp callable. For kv = 1 the Hessian ∇∇(v) is zero, so the
    # grad-div contribution vanishes in any dimension.
    div_ViscProj_v = Operation(EvalDivDevSymOp())(Δ(v), ∇∇(v))
    return 2.0 * ν * (ViscProj_v ⋅ ∇(α)) + 2.0 * α * ν * div_ViscProj_v
end
