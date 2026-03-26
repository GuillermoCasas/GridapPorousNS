using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Gridap

u(x) = VectorValue(sin(x[1])*x[2], x[1]*cos(x[2]))
x = Point(1.0, 2.0)

# Automatic differentiation
grad_u_ad = ∇(u)(x)
println("AD Gradient: ", grad_u_ad)

println("∇(u) ⋅ u : ", grad_u_ad ⋅ u(x))
println("∇(u)' ⋅ u : ", transpose(grad_u_ad) ⋅ u(x))

# evaluate hand-calculated convective term (u * ∇) u
# (u1 d/dx1 + u2 d/dx2) u1 = u1 * u1_x1 + u2 * u1_x2
# (u1 d/dx1 + u2 d/dx2) u2 = u1 * u2_x1 + u2 * u2_x2
u1_val = sin(x[1])*x[2]
u2_val = x[1]*cos(x[2])
conv1 = u1_val * cos(x[1])*x[2] + u2_val * sin(x[1])
conv2 = u1_val * cos(x[2]) + u2_val * (-x[1]*sin(x[2]))
println("Hand (u ⋅ ∇) u : ", VectorValue(conv1, conv2))

