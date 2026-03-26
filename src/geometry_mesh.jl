# src/geometry_mesh.jl

function create_mesh(config::PorousNSConfig)
    domain = Tuple(config.mesh.domain)
    partition = Tuple(config.mesh.partition)
    
    model = CartesianDiscreteModel(domain, partition)
    
    # Automatically tag boundaries for a 2D box
    # Default Gridap face labels for a 2D Cartesian box:
    # 1,2,3,4: Corners
    # 5: Left (x=xmin)
    # 6: Right (x=xmax)
    # 7: Bottom (y=ymin)
    # 8: Top (y=ymax)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [5])
    add_tag_from_tags!(labels, "outlet", [6])
    
    # Add top and bottom walls, plus all corners to "walls" tag
    add_tag_from_tags!(labels, "walls", [1, 2, 3, 4, 7, 8])
    
    return model
end
