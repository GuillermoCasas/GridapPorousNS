# test/blitz/cocquet_modified_corner_topology_blitz_test.jl
# ==============================================================================================
# Nature & Intent:
# Sanity-checks the outlet-corner DOF release performed by
# test/extended/CocquetExperimentModifiedCorner/run_convergence.jl. The corner-untagging
# experiment depends on two latent Gridap conventions: (i) vertex entity IDs 2 and 4 in a 2D
# CartesianDiscreteModel correspond to the (x_max, y_min) and (x_max, y_max) corner nodes;
# (ii) omitting an entity ID from the list passed to `add_tag_from_tags!` releases the matching
# nodes from any Dirichlet constraint built on that tag. If a future Gridap release reorders
# the corner-entity numbering or changes the tag-driven constraint logic, the convergence
# sweep would silently solve a different problem; this blitz catches that drift in < 5 s.
#
# Mathematical / topological assertions:
#   1. In a CartesianDiscreteModel over `[0,2]×[0,1]`, the vertices carrying entity IDs 2 and 4
#      sit exactly at (2, 0) and (2, 1) — the two outlet corners we intend to release.
#   2. Building a TestFESpace with `dirichlet_tags=["inlet","walls"]` over the modified-corner
#      mesh yields exactly four more free DOFs than the default mesh (two vector components ×
#      two released corner vertices). This holds for both P1 and P2 velocity spaces (edge
#      midpoints of the wall edges are unchanged).
#
# Associated files:
# - test/extended/CocquetExperimentModifiedCorner/run_convergence.jl (consumes the convention)
# - src/run_simulation.jl:104-121 (`_build_default_mesh`, the unmodified reference)
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Test
using PorousNSSolver
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs

const DOMAIN_BBOX = [0.0, 2.0, 0.0, 1.0]
const PARTITION = (4, 2)

# Standalone replica of the modified-corner mesh builder. Kept verbatim in lock-step with the
# definition in test/extended/CocquetExperimentModifiedCorner/run_convergence.jl so the blitz
# can run without needing to include the entire convergence driver (avoids `using Pkg` /
# JSON3 / HDF5 startup costs and keeps the test under the blitz time budget).
function _build_modified_corner_mesh(domain_tuple::NTuple{4,Float64}, partition::Tuple, element_type::String)
    if element_type == "TRI"
        model = CartesianDiscreteModel(domain_tuple, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain_tuple, partition)
    end
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet",  [7])
    add_tag_from_tags!(labels, "outlet", [8])
    add_tag_from_tags!(labels, "walls",  [1, 3, 5, 6])   # entities 2 and 4 INTENTIONALLY omitted.
    return model
end

# Identical to PorousNSSolver._build_default_mesh body (src/run_simulation.jl:104-121) but invoked
# directly here so the test does not depend on the symbol staying non-private.
function _build_default_mesh_local(domain_tuple::NTuple{4,Float64}, partition::Tuple, element_type::String)
    if element_type == "TRI"
        model = CartesianDiscreteModel(domain_tuple, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain_tuple, partition)
    end
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet",  [7])
    add_tag_from_tags!(labels, "outlet", [8])
    add_tag_from_tags!(labels, "walls",  [1, 2, 3, 4, 5, 6])
    return model
end

domain_tuple = (DOMAIN_BBOX[1], DOMAIN_BBOX[2], DOMAIN_BBOX[3], DOMAIN_BBOX[4])

@testset "CocquetExperimentModifiedCorner — outlet-corner topology" begin

    @testset "Gridap 2D Cartesian vertex-entity convention" begin
        # Build a default-tagged model and recover the vertex-to-entity map from the labeling.
        for element_type in ("QUAD", "TRI")
            model = _build_default_mesh_local(domain_tuple, PARTITION, element_type)
            labels = get_face_labeling(model)
            vertex_entity = get_face_entity(labels, 0)        # one entity per 0D face (vertex), flat
            # `get_node_coordinates` returns a 2D array for the QUAD CartesianDiscreteModel and a
            # flat Vector for the simplexified TRI variant — flatten once so the linear index
            # returned by `findfirst` lines up with `vertex_entity` in both cases.
            node_coords = vec(get_node_coordinates(model))

            # Locate the (2.0, 0.0) and (2.0, 1.0) outlet corner vertices and read their entity tags.
            br_idx = findfirst(p -> isapprox(p[1], 2.0; atol=1e-12) && isapprox(p[2], 0.0; atol=1e-12), node_coords)
            tr_idx = findfirst(p -> isapprox(p[1], 2.0; atol=1e-12) && isapprox(p[2], 1.0; atol=1e-12), node_coords)
            @test br_idx !== nothing
            @test tr_idx !== nothing
            @test vertex_entity[br_idx] == 2    # bottom-right outlet corner: must be entity 2
            @test vertex_entity[tr_idx] == 4    # top-right    outlet corner: must be entity 4

            # And confirm the inlet-side corners are entities 1 (bottom-left) and 3 (top-left) —
            # both deliberately KEPT in the walls list so the test guards against a swap with 2/4.
            bl_idx = findfirst(p -> isapprox(p[1], 0.0; atol=1e-12) && isapprox(p[2], 0.0; atol=1e-12), node_coords)
            tl_idx = findfirst(p -> isapprox(p[1], 0.0; atol=1e-12) && isapprox(p[2], 1.0; atol=1e-12), node_coords)
            @test vertex_entity[bl_idx] == 1
            @test vertex_entity[tl_idx] == 3
        end
    end

    @testset "Free-DOF count rises by exactly 4 for P1 and P2 (QUAD + TRI)" begin
        # The inflow profile vanishes at y=0,1 so it matches the no-slip walls at the inlet corners
        # — same closure used by run_convergence.jl. Any concrete profile suffices for DOF counting;
        # only the dirichlet_tags topology determines `num_free_dofs`.
        u_in = x -> VectorValue(0.5 * x[2] * (1.0 - x[2]), 0.0)
        u_wall = x -> VectorValue(0.0, 0.0)
        dirichlet_tags = ["inlet", "walls"]
        dirichlet_masks = [(true, true), (true, true)]
        dirichlet_values = [u_in, u_wall]

        for element_type in ("QUAD", "TRI"), kv in (1, 2)
            model_default  = _build_default_mesh_local(domain_tuple, PARTITION, element_type)
            model_modified = _build_modified_corner_mesh(domain_tuple, PARTITION, element_type)

            elem_cfg = PorousNSSolver.ElementSpacesConfig(k_velocity=kv, k_pressure=1)
            X_def, _, _, _ = PorousNSSolver.build_fe_spaces(model_default,  elem_cfg, dirichlet_tags, dirichlet_masks, dirichlet_values)
            X_mod, _, _, _ = PorousNSSolver.build_fe_spaces(model_modified, elem_cfg, dirichlet_tags, dirichlet_masks, dirichlet_values)

            U_def, _ = X_def
            U_mod, _ = X_mod
            delta = num_free_dofs(U_mod) - num_free_dofs(U_def)
            @test delta == 4   # 2 outlet corner vertices × 2 velocity components
        end
    end

end
