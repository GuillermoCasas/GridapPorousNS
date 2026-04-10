using Gridap
domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain, partition)
u(x) = VectorValue(x[1]^2, x[1]*x[2])
x = Point(2.0, 3.0)
println( ∇(u)(x) )

grad_u = ∇(u)(x)
println("u . ∇(u) = ", u(x) ⋅ grad_u)
println("∇(u)' . u = ", transpose(grad_u) ⋅ u(x))
