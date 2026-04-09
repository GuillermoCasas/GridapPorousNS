using Gridap

domain = (0, 2, 0, 1)
partition = (4, 2)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)

println("Tags available: ", get_tags(labels))
println("Tag 7 name: ", get_name(labels, 7))
println("Tag 8 name: ", get_name(labels, 8))
for i in 1:8
    println("Tag ", i, " name: ", get_name(labels, i))
end
