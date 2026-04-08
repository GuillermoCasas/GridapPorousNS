# src/formulations/gridap_extensions.jl
using Gridap
using Gridap.TensorValues

"""
    _contract_grad_div(H)

Internal function to contract the 3rd-order Hessian tensor `H` to compute `∇(∇⋅u)`.
For a vector field `u`, `H` is a 3D tensor where `H[i, j, k]` represents `∂_i ∂_j u_k`.
The gradient of the divergence is a vector with components `v_i = ∑_j ∂_i ∂_j u_j = ∑_j H[i, j, j]`.
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

Custom Gridap algebraic operator that computes the exact analytical `∇(∇⋅u)` 
by contracting the true element-wise Hessian `∇∇(u)`.
Note: Your finite element space must use at least quadratic elements (\$P^2\$) 
for this to evaluate to non-zero values on standard Lagrangian cells.
"""
grad_div_op(u) = Operation(ContractGradDivOp())(∇∇(u))
