using Gridap
domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7,8])
V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1, labels=labels, dirichlet_tags=["dirichlet"])
U = TrialFESpace(V, [x->0.0])

f(x) = 10.0
uh = interpolate(f, U)

println("interpolate(f, U) values: ", get_free_dof_values(uh))
# Evaluate the FEFunction at domain bounds to see if it's 10 or 0
println("Evaluating uh at interior (0.5,0.5): ", uh(VectorValue(0.5,0.5)))
println("Evaluating uh at boundary (0.0,0.0): ", uh(VectorValue(0.0,0.0)))

vh = interpolate(f, V)
println("Evaluating vh at boundary (0.0,0.0): ", vh(VectorValue(0.0,0.0)))
