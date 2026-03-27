# tests/CocquetExperiment/debug_bc.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PorousNSSolver
using Gridap

config_path = joinpath(@__DIR__, "data", "test_config.json")
config = PorousNSSolver.load_config(config_path)

model = PorousNSSolver.create_mesh(config)

refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, config.discretization.k_velocity)
labels = get_face_labeling(model)

for face_tag in 5:8
    try
        bΩ = BoundaryTriangulation(model, tags=[face_tag])
        coords = get_cell_coordinates(bΩ)[1]
        println("Face $face_tag coordinates: ", coords)
    catch
    end
end
