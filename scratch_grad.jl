using Gridap
using Gridap.TensorValues

u_f(x) = VectorValue(x[1]^2 * x[2], x[1] * x[2]^2)
v_f(x) = VectorValue(1.0, 2.0)

domain = (0, 1, 0, 1)
partition = (2, 2)
model = CartesianDiscreteModel(domain, partition)
refe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
V = TestFESpace(model, refe, conformity=:H1)
u_h = interpolate_everywhere(u_f, V)
v_h = interpolate_everywhere(v_f, V)

pt = Point(2.0, 3.0)

# Exact values:
# u = (12, 18)
# v = (1, 2)
# du/dx = [2xy, y^2] = [12, 9]
# du/dy = [x^2, 2xy] = [4, 12]
# T = [du1/dx du1/dy; du2/dx du2/dy] = [12, 4; 9, 12]
# (v . nabla) u = 1 * [12, 9] + 2 * [4, 12] = [20, 33]

grad_u = ∇(u_h)(pt)
println("Gridap gradient tensor:")
println(grad_u)

val1 = (grad_u * v_h(pt))
println("∇u * v = ", val1)

val2 = (transpose(grad_u) * v_h(pt))
println("∇u' * v = ", val2)

# Using Gridap's dot product:
val3 = ∇(u_h)(pt) ⋅ v_h(pt)
println("∇u ⋅ v = ", val3)

val4 = transpose(∇(u_h)(pt)) ⋅ v_h(pt)
println("∇u' ⋅ v = ", val4)
