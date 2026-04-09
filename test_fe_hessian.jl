using Gridap
using Gridap.TensorValues

domain = (0, 1, 0, 1)
partition = (2, 2)
model = CartesianDiscreteModel(domain, partition)

refe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
V = TestFESpace(model, refe, conformity=:H1)
U = TrialFESpace(V)

u_h = interpolate_everywhere(x -> VectorValue(x[1]^2 * x[2], x[1] * x[2]^2), U)
pt = Point(0.25, 0.25)

grad_pt = ∇(u_h)(pt)
println("Gradient tensor at (0.25, 0.25):")
for i in 1:2, j in 1:2
    println("grad[$i,$j] = ", grad_pt[i,j])
end

H_pt = ∇∇(u_h)(pt)
println("Hessian tensor at (0.25, 0.25):")
for i in 1:2, j in 1:2, k in 1:2
    println("H[$i,$j,$k] = ", H_pt[i,j,k])
end
