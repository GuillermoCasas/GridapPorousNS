# src/formulations/gridap_extensions.jl
#
# [unused — kept for possible future use] `ContractGradDivOp` / `grad_div_op` compute ∇(∇·u) by
# contracting the element Hessian. As of 2026-06 they have NO callers: viscous_operators.jl
# re-implements the same contraction inline (EvalDivDevSymOp / EvalStrongViscSymOp). Retained as a
# single-purpose, dimension-generic building block a future refactor of those operators could call
# (docs/formulation-audit-2026-06-24.md §D / VISC-02). The include is harmless — it only defines names.
using Gridap
using Gridap.TensorValues

"""
    ContractGradDivOp

Callable operator that contracts a vector field's 3rd-order Hessian tensor `H` into the
vector `∇(∇⋅u)` (the gradient of the divergence). For a vector field `u`, `H[i, j, k]`
holds the second derivative `∂_i ∂_j u_k`, so the wanted vector has components
`v_i = ∑_j ∂_i ∂_j u_j = ∑_j H[i, j, j]`. Defined for 2D and 3D Hessians.

This is the building block used to obtain `∇(∇⋅u)` analytically: Gridap can evaluate exact
element Hessians (`∇∇`) on Lagrangian cells, so contracting that Hessian sidesteps Gridap's
lack of a native chain-rule expansion for `∇(∇⋅u)` over composed operations.
"""
struct ContractGradDivOp <: Function end
@inline function (::ContractGradDivOp)(H::ThirdOrderTensorValue{2,2,2,T}) where T
    return VectorValue(H[1,1,1] + H[1,2,2], H[2,1,1] + H[2,2,2])
end
@inline function (::ContractGradDivOp)(H::ThirdOrderTensorValue{3,3,3,T}) where T
    return VectorValue(H[1,1,1] + H[1,2,2] + H[1,3,3], H[2,1,1] + H[2,2,2] + H[2,3,3], H[3,1,1] + H[3,2,2] + H[3,3,3])
end

"""
    grad_div_op(u)

Gridap-level operator that computes the exact analytical `∇(∇⋅u)` for a vector FE field `u`
by wrapping `ContractGradDivOp` as a `CellField` `Operation` over the element Hessian `∇∇(u)`.
The result is itself a `CellField`, so it composes inside weak/strong-form expressions (it
feeds the dilatancy term `∇(∇⋅u)` that appears in the symmetric-gradient viscous strong
operator; compare `EvalStrongViscSymOp` in `viscous_operators.jl`).

The interpolation order must be at least quadratic (\$P^2\$): on lower-order Lagrangian cells
the element Hessian is identically zero, so this would silently evaluate to the zero vector.
"""
grad_div_op(u) = Operation(ContractGradDivOp())(∇∇(u))
