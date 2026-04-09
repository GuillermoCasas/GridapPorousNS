using Gridap

domain = (0, 1, 0, 1)
partition = (4, 4)
model = CartesianDiscreteModel(domain, partition)
Ω = Triangulation(model)

f(x) = VectorValue(sin(x[1]), cos(x[2]))
V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2), conformity=:H1)
f_h = interpolate_everywhere(f, V)

dΩ = Measure(Ω, 2)
∫( Δ(f_h) ) * dΩ
println("Laplacian evaluation succeeded!")
