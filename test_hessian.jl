using Gridap
using Gridap.TensorValues

domain = (0, 1, 0, 1)
partition = (2, 2)
model = CartesianDiscreteModel(domain, partition)

u(x) = VectorValue(x[1]^2 * x[2], x[1] * x[2]^2)
# u_1 = x^2 y
# u_2 = x y^2

# ∂x u_1 = 2xy, ∂y u_1 = x^2
# ∂x u_2 = y^2, ∂y u_2 = 2xy

# ∂x∂x u_1 = 2y, ∂y∂y u_1 = 0, ∂x∂y u_1 = 2x
# ∂x∂x u_2 = 0, ∂y∂y u_2 = 2x, ∂x∂y u_2 = 2y

# ∇(∇⋅u) = ∇(2xy + 2xy) = ∇(4xy) = (4y, 4x)

f(x) = (∇∇(u))(x)
pt = VectorValue(1.0, 1.0)
H = f(pt)

println("Hessian structure at (1,1):")
for i in 1:2, j in 1:2, k in 1:2
    println("H[$i,$j,$k] = ", H[i,j,k])
end
