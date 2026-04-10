using Gridap
domain = (0, 1, 0, 1)
model = CartesianDiscreteModel(domain, (2, 2))
Ω = Triangulation(model)
u_func(x) = VectorValue(x[2], -x[1])
u = CellField(x -> u_func(x), Ω)
x_eval = Point(0.5, 0.5)

grad_u = ∇(u)(x_eval)
u_val = u(x_eval)

println("grad_u' ⋅ u : ", grad_u' ⋅ u_val)
println("u ⋅ grad_u  : ", u_val ⋅ grad_u)
