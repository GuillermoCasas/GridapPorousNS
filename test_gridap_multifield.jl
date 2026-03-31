using Gridap
domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain, partition)
refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
refe_p = ReferenceFE(lagrangian, Float64, 1)
V = TestFESpace(model, refe_u)
Q = TestFESpace(model, refe_p)
Y = MultiFieldFESpace([V, Q])
f_u(x) = VectorValue(1.0, 1.0)
f_p(x) = 2.0
y_h = interpolate_everywhere([f_u, f_p], Y)
u_h, p_h = y_h
println(typeof(u_h))
