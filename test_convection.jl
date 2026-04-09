using Gridap
using Gridap.TensorValues

domain = (0, 1, 0, 1)
partition = (2, 2)
model = CartesianDiscreteModel(domain, partition)
refe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
V = TestFESpace(model, refe, conformity=:H1)
U = TrialFESpace(V)

u_h = interpolate_everywhere(x -> VectorValue(x[1]^2, 2.0*x[1]), U)

# Evaluate at a test point
pt = Point(0.25, 0.25)
u_val = u_h(pt)
grad_val = ∇(u_h)(pt)

println("u_val = ", u_val)
println("grad_val = ", grad_val)

val1 = grad_val' ⋅ u_val
println("grad_val' ⋅ u_val = ", val1)

val2 = grad_val ⋅ u_val
println("grad_val ⋅ u_val = ", val2)

# analytical (u \cdot \nabla) u
# u_1 = x^2, u_2 = 2x
# (u \partial_x + v \partial_y) u_1 = (x^2)(2x) + (2x)(0) = 2x^3
# at x=0.25 => 2 * (1/64) = 0.03125
# (u \partial_x + v \partial_y) u_2 = (x^2)(2) + (2x)(0) = 2x^2
# at x=0.25 => 2 * (1/16) = 0.125
# analytical 1/2 \nabla |u|^2
# |u|^2 = x^4 + 4x^2
# \partial_x (1/2 |u|^2) = 1/2 (4x^3 + 8x) = 2x^3 + 4x
# at x=0.25 => 2(1/64) + 4(1/4) = 0.03125 + 1.0 = 1.03125
# \partial_y (1/2 |u|^2) = 0

