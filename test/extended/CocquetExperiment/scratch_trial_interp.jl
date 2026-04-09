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
println("interpolate(f, U) dirichlet values: ", get_dirichlet_dof_values(uh))

# What if we interpolate(f, V)?
vh = interpolate(f, V)
println("interpolate(f, V) values: ", get_free_dof_values(vh))
println("interpolate(f, V) dirichlet values: ", get_dirichlet_dof_values(vh))
