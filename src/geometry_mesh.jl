# src/geometry_mesh.jl

function create_mesh(config::PorousNSConfig)
    domain = Tuple(config.mesh.domain)
    partition = Tuple(config.mesh.partition)
    
    if config.mesh.element_type == "TRI"
        model = CartesianDiscreteModel(domain, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain, partition)
    end
    
    # Default Gridap face labels for a 2D Cartesian box:
    # 1..4: Corners (Vertices)
    # 5: Bottom (y=ymin), 6: Top (y=ymax)
    # 7: Left (x=xmin) -> Inlet, 8: Right (x=xmax) -> Outlet
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [7])
    add_tag_from_tags!(labels, "outlet", [8])
    
    # Add top and bottom walls, plus all geometric corners to "walls" tag natively.
    # Grouping corners explicitly is critical to structurally isolate the singular points from the free Neumann traction evaluations natively tracking boundaries.
    add_tag_from_tags!(labels, "walls", [1, 2, 3, 4, 5, 6])
    
    return model
end
