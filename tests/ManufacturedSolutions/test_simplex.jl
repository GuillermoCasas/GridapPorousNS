using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Gridap

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain, partition)
model_tri = simplexify(model)
println("Simplexify succeeded on CartesianDiscreteModel")

labels = get_face_labeling(model_tri)
add_tag_from_tags!(labels, "inlet", [5])
println("Labels successfully tagged on simplex model")
