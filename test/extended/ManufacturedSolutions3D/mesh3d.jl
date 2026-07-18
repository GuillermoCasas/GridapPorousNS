# test/extended/ManufacturedSolutions3D/mesh3d.jl
# ==============================================================================================
# Unstructured tetrahedral mesh generation for the paper's 3D manufactured-solution case
# (§5.2, "Manufactured solutions: 3D cases"). The paper uses a (0,1)x(0,1)x(0,0.3) parallelepiped
# meshed with UNSTRUCTURED tetrahedra "to break the symmetry, so the velocity vectors have a
# nonzero z-component due to discretization errors", with the sequence obtained by successively
# DOUBLING the average element size (4 meshes for P1, 3 for P2).
#
# We reproduce that methodology with gmsh (GridapGmsh is already a project dependency): an
# OpenCASCADE box meshed by the Delaunay algorithm at a target characteristic length `lc`, which
# we halve down the sequence. The whole boundary is one physical group "boundary" (the 3D MMS
# imposes Dirichlet u_ex on the entire boundary); the volume is the physical group "domain".
# ==============================================================================================
using Gridap
using GridapGmsh
using Gridap.Adaptivity
using Gridap.Geometry: get_node_coordinates, get_cell_node_ids, get_grid
using HDF5
using Printf   # @sprintf for the committed base-mesh filename (nested_red_base_path)

"""
    build_box_tet_model(lc; domain=(0.0,1.0, 0.0,1.0, 0.0,0.3), algorithm=5, save_msh="")

Generate an UNSTRUCTURED tetrahedral mesh of the box `domain=(x0,x1,y0,y1,z0,z1)` at target
characteristic element length `lc`, returned as a Gridap `GmshDiscreteModel`. Boundary faces are
tagged "boundary" (all 6 faces) and the volume "domain". `algorithm` is the gmsh 3D mesher
(1=Delaunay, 4=Frontal, 10=HXT); 1 (Delaunay) gives genuinely unstructured, symmetry-breaking tets.
"""
function build_box_tet_model(lc::Float64; domain=(0.0,1.0, 0.0,1.0, 0.0,0.3),
                             algorithm::Int=1, save_msh::String="",
                             optimize_passes::Int=0, optimize_threshold::Float64=0.5)
    x0, x1, y0, y1, z0, z1 = domain
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

    gmsh = GridapGmsh.gmsh
    dir = ""
    mshfile = ""
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
        # Uniform unstructured size: clamp both bounds to lc so the Delaunay mesher fills the
        # interior at ~lc. Halving lc down the sequence doubles the resolution per the paper.
        # [rate-fix] Force gmsh to honor lc as the actual size (not a loose average): disable the
        # point/curvature/boundary size heuristics so the achieved mean h tracks lc and the mesh
        # family halves cleanly (otherwise hmean shrinks slower than lc and deflates the rate).
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        # [quality] Post-optimize tet quality (remove slivers/high-aspect tets) so interpolation —
        # especially the gradient/H1 error — is not degraded. The base quality propagates UNCHANGED
        # through red refinement, so a clean base lifts the whole nested family toward optimal rates.
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", optimize_threshold)
        gmsh.model.add("box3d")

        # This gmsh build has no OpenCASCADE, so the box is assembled with the built-in `geo`
        # kernel: 8 corner points, 12 edges, 6 plane faces, 1 surface loop -> 1 volume.
        geo = gmsh.model.geo
        # bottom face z=z0 (p1..p4), top face z=z1 (p5..p8)
        p1 = geo.addPoint(x0, y0, z0, lc); p2 = geo.addPoint(x1, y0, z0, lc)
        p3 = geo.addPoint(x1, y1, z0, lc); p4 = geo.addPoint(x0, y1, z0, lc)
        p5 = geo.addPoint(x0, y0, z1, lc); p6 = geo.addPoint(x1, y0, z1, lc)
        p7 = geo.addPoint(x1, y1, z1, lc); p8 = geo.addPoint(x0, y1, z1, lc)
        l1 = geo.addLine(p1, p2); l2 = geo.addLine(p2, p3); l3 = geo.addLine(p3, p4); l4 = geo.addLine(p4, p1)
        l5 = geo.addLine(p5, p6); l6 = geo.addLine(p6, p7); l7 = geo.addLine(p7, p8); l8 = geo.addLine(p8, p5)
        l9 = geo.addLine(p1, p5); l10 = geo.addLine(p2, p6); l11 = geo.addLine(p3, p7); l12 = geo.addLine(p4, p8)
        # 6 faces (each a closed, consistently-signed curve loop)
        f_bot = geo.addPlaneSurface([geo.addCurveLoop([l1, l2, l3, l4])])            # z=z0
        f_top = geo.addPlaneSurface([geo.addCurveLoop([l5, l6, l7, l8])])            # z=z1
        f_fro = geo.addPlaneSurface([geo.addCurveLoop([l1, l10, -l5, -l9])])         # y=y0
        f_rig = geo.addPlaneSurface([geo.addCurveLoop([l2, l11, -l6, -l10])])        # x=x1
        f_bck = geo.addPlaneSurface([geo.addCurveLoop([l3, l12, -l7, -l11])])        # y=y1
        f_lef = geo.addPlaneSurface([geo.addCurveLoop([l4, l9, -l8, -l12])])         # x=x0
        faces = [f_bot, f_top, f_fro, f_rig, f_bck, f_lef]
        sl = geo.addSurfaceLoop(faces)
        vol = geo.addVolume([sl])
        geo.synchronize()

        # Physical-group NAMES become Gridap face-label tag names. The whole boundary (all 6 faces)
        # is one Dirichlet group; the volume is the domain.
        g_bnd = gmsh.model.addPhysicalGroup(2, faces); gmsh.model.setPhysicalName(2, g_bnd, "boundary")
        g_dom = gmsh.model.addPhysicalGroup(3, [vol]); gmsh.model.setPhysicalName(3, g_dom, "domain")

        gmsh.model.mesh.generate(3)
        # [quality] Extra UNSTRUCTURED-quality smoothing (Netgen tet-quality + Relocate3D vertex
        # smoothing), to lift the worst-tet quality without changing the mesh's unstructured character.
        # optimize_passes=0 ⇒ legacy behaviour. (Frontal `algorithm=4` + these passes is the recipe that
        # recovers optimal 3D-P1 convergence — see docs.)
        for _ in 1:optimize_passes
            try; gmsh.model.mesh.optimize("Netgen");     catch e; @warn "gmsh optimize(Netgen) failed (skipped)"     exception=e; end
            try; gmsh.model.mesh.optimize("Relocate3D"); catch e; @warn "gmsh optimize(Relocate3D) failed (skipped)" exception=e; end
        end

        dir = mktempdir()
        mshfile = joinpath(dir, "box3d.msh")
        gmsh.write(mshfile)
        if !isempty(save_msh)
            mkpath(dirname(save_msh)); gmsh.write(save_msh)
        end
    finally
        gmsh.finalize()
    end

    model = GmshDiscreteModel(mshfile)
    rm(dir; recursive=true, force=true)
    return model
end

# ==============================================================================================
# NESTED REFINEMENT FAMILY (a convergence-study mesh strategy)
# Build ONE coarse, NON-UNIFORM unstructured tet mesh, then refine RECURSIVELY by uniform red
# (1->8) tetrahedral subdivision (Gridap.Adaptivity.refine, "red_green"). Each level halves h
# EXACTLY and PRESERVES the "boundary"/"domain" FaceLabeling tags (refine_face_labeling inherits the
# parent-face entity).
#
# [known-fragility] CAUTION: red refinement does NOT preserve the worst-case element quality on tets.
# Bey/Freudenthal red refinement bounds the number of congruence classes ASYMPTOTICALLY, but the
# octahedral sub-tets DEGRADE the minimum quality for the first couple of levels before it settles —
# MEASURED (2026-06-24, Frontal+optimize base lc=0.137): qmin 0.466 -> 0.247 -> 0.151 (L0->L1->L2).
# So this family is NOT quality-preserving in the regime the convergence study actually samples; the
# min-quality TAIL falls under refinement and adds rate noise. (Earlier comments here claimed the
# opposite — that was wrong; see docs/mms/p2-3d.md §"mesh-quality mechanics".)
# Independent gmsh remeshes are the other extreme: each level is a fresh, differently-flawed realization
# whose error level is set by its own quality, so per-segment "rates" are realization SCATTER, not
# h-convergence. The only refinement-INVARIANT-quality option is a STRUCTURED Kuhn tet family
# (simplexify of a Cartesian grid, `structured_kuhn_model`) — used as the clean control.
# ==============================================================================================

"The single initial NON-UNIFORM unstructured tet mesh (coarse root of the nested family)."
build_base_mesh(; lc::Float64=0.2, domain=(0.0,1.0, 0.0,1.0, 0.0,0.3), algorithm::Int=1, save_msh::String="") =
    build_box_tet_model(lc; domain=domain, algorithm=algorithm, save_msh=save_msh)

# ----------------------------------------------------------------------------------------------
# [reproducible-results] Committed base mesh for the irregular (nested-red) family.
#
# The ENTIRE irregular family is one gmsh base mesh plus DETERMINISTIC red refinement, so the base mesh
# alone fixes every level. Until 2026-07-17 that base was generated into an `mktempdir()` and deleted
# (`build_box_tet_model` writes it, then `rm(dir; recursive=true)`), `save_msh` existed but no caller
# passed it, and the gmsh version is not pinned — so the published irregular-family numbers could not be
# guaranteed to correspond to any regenerable mesh. gmsh's unstructured tet generator is not contractually
# stable across versions: a different gmsh can return a different (equally valid) mesh, and on THIS family
# the element-quality tail is exactly what the results are sensitive to (docs/mms/p2-3d.md).
#
# `nested_red_base_path` is therefore committed alongside the harness and PREFERRED over regeneration:
# loading it makes the family reproducible independently of the local gmsh. If it is absent, we generate
# and persist it, and say so loudly — a freshly generated base is a NEW mesh family, not a reproduction.
nested_red_base_path(; lc::Float64=0.2, algorithm::Int=1) =
    joinpath(@__DIR__, "meshes", @sprintf("nested_red_base_lc%.3f_alg%d.msh", lc, algorithm))

"""
    load_or_build_base_mesh(; lc, domain, algorithm, path=nested_red_base_path(...))

Return the nested family's base mesh, loading the committed `.msh` when present and otherwise
generating it with gmsh and persisting it to `path`. Prefer this over `build_base_mesh` for anything
whose numbers are published: it is what makes the irregular family regenerable (see the note above).
"""
function load_or_build_base_mesh(; lc::Float64=0.2, domain=(0.0,1.0, 0.0,1.0, 0.0,0.3),
                                 algorithm::Int=1, path::String=nested_red_base_path(; lc=lc, algorithm=algorithm))
    if isfile(path)
        @info "[mesh3d] loading COMMITTED nested-red base mesh (reproducible, gmsh-version-independent)" path
        return GmshDiscreteModel(path)
    end
    @warn """
    [mesh3d] no committed nested-red base mesh at $(path) — generating a FRESH one with the local gmsh and
    persisting it there. The result is a NEW mesh family: gmsh's unstructured tet generator is not stable
    across versions, and this family's results are sensitive to its element-quality tail. Commit the file
    so the numbers are reproducible, and do not compare fresh-base results against previously published
    ones as if they were the same family.""" lc algorithm
    model = build_base_mesh(; lc=lc, domain=domain, algorithm=algorithm, save_msh=path)
    return model
end

"Uniform red (1->8) refinement applied `n` times; returns a plain UnstructuredDiscreteModel with tags preserved."
function refine_n_times(model, n::Int)
    m = model
    for _ in 1:n
        m = Gridap.Adaptivity.get_model(refine(m; refinement_method="red_green"))
    end
    return m
end

"""
Nested family [base, refine¹, …, refineⁿ]; built incrementally (each level refines the previous).

The base mesh comes from `load_or_build_base_mesh`, i.e. the COMMITTED `.msh` when it exists, so the whole
family is reproducible without depending on the local gmsh version (red refinement is deterministic, so
the base fixes every level). See the note above `nested_red_base_path`.
"""
function build_nested_family(nlevels::Int; lc::Float64=0.2, domain=(0.0,1.0, 0.0,1.0, 0.0,0.3), algorithm::Int=1)
    base = load_or_build_base_mesh(; lc=lc, domain=domain, algorithm=algorithm)
    fam = Vector{Any}(undef, nlevels + 1)
    fam[1] = base
    for k in 1:nlevels
        fam[k+1] = Gridap.Adaptivity.get_model(refine(fam[k]; refinement_method="red_green"))
    end
    return fam
end

"Mean element size (regular-tet edge from cell volume), the convergence abscissa."
function mesh_hmean(model)
    Ω = Triangulation(model)
    ha = collect(lazy_map(v -> (6.0*sqrt(2.0)*abs(v))^(1/3), get_cell_measure(Ω)))
    return sum(ha) / length(ha)
end

# ----------------------------------------------------------------------------------------------
# HDF5 mesh export for Python plotting. Layout (Julia column-major -> h5py sees the TRANSPOSE):
#   node_coords : Julia (3, N)  -> h5py (N, 3)   vertex coordinates
#   tets        : Julia (4, K)  -> h5py (K, 4)   tet connectivity, 1-BASED node indices (Python: -1)
# The Python script derives the boundary surface (faces in exactly one tet) for drawing.
# ----------------------------------------------------------------------------------------------
function export_mesh_h5(model, path::String)
    grid = get_grid(model)
    nodes = get_node_coordinates(grid); N = length(nodes)
    coords = Array{Float64}(undef, 3, N)
    @inbounds for n in 1:N, d in 1:3
        coords[d, n] = nodes[n][d]
    end
    cids = get_cell_node_ids(grid); K = length(cids); npc = length(cids[1])
    tets = Array{Int}(undef, npc, K)
    @inbounds for k in 1:K
        c = cids[k]
        for j in 1:npc; tets[j, k] = c[j]; end
    end
    mkpath(dirname(path))
    h5open(path, "w") do f
        f["node_coords"] = coords
        f["tets"] = tets
        attributes(f)["nnodes"] = N
        attributes(f)["ncells"] = K
        attributes(f)["hmean"] = mesh_hmean(model)
    end
    return path
end

# ==============================================================================================
# MESH SEQUENCES for the convergence studies. Each named sequence's meshes are exported to
# results/meshes/<sequence>/mesh_level{k}.h5 (geometry only) and rendered as a mosaic by
# plot_mesh3d.py. "meshes" is the LIBRARY of mesh sequences; the convergence RESULTS for each
# sequence live separately under results/k<kv>/TET/<sequence>/.
# ==============================================================================================

"Structured Kuhn tet mesh: simplexify of a Cartesian grid — UNIFORM, refinement-invariant quality."
structured_kuhn_model(partition; domain=(0.0,1.0, 0.0,1.0, 0.0,0.3)) =
    simplexify(CartesianDiscreteModel(domain, partition))

"Export a named mesh SEQUENCE's models to results/meshes/<seq>/mesh_level{k}.h5 (geometry only)."
function export_mesh_sequence(seq::String, models; outroot=joinpath(@__DIR__, "results", "meshes"))
    dir = joinpath(outroot, seq); mkpath(dir)
    for (k, m) in enumerate(models)
        export_mesh_h5(m, joinpath(dir, "mesh_level$(k-1).h5"))
    end
    println("[mesh3d] sequence \"$seq\": exported $(length(models)) level(s) -> $dir"); flush(stdout)
    return dir
end

"""
    build_sequence(seq; domain) -> Vector of models

Build the meshes of a named convergence-study mesh sequence (coarse→mid levels, sized for legible,
fast-rendering mosaics — the mosaic shows the sequence's CHARACTER and refinement, not the finest
production meshes). Known sequences: "structured" (Kuhn), "nested_red" (red-refined gmsh family),
"frontal" (independent gmsh Frontal-Delaunay remeshes).
"""
function build_sequence(seq::String; domain=(0.0,1.0, 0.0,1.0, 0.0,0.3))
    if seq == "structured"
        return [structured_kuhn_model(p; domain=domain) for p in [(8,8,2), (12,12,3), (16,16,4), (24,24,6)]]
    elseif seq == "nested_red"
        return build_nested_family(2; lc=0.2, domain=domain)                  # 3 levels (L0–L2)
    elseif seq == "frontal"
        return [build_box_tet_model(lc; domain=domain, algorithm=4, optimize_passes=5)
                for lc in [0.137, 0.098, 0.070]]
    else
        error("unknown mesh sequence \"$seq\" (known: structured, nested_red, frontal)")
    end
end

# ------------------------------------------------------------------------------------------------
# Standalone validation + export when run directly: julia --project=../../.. mesh3d.jl [nlevels] [lc]
#   - proves tag survival + clean ~8x cell / ~2x h-halving across the nested family
#   - exports results/mesh/mesh_level{k}.h5 for plotting with plot_mesh3d.py
# ------------------------------------------------------------------------------------------------
function _run_mesh_validation(nlevels::Int, lc0::Float64, outdir::String)
    println("=== nested tet family: base lc=$lc0, $nlevels refinements ===")
    fam = build_nested_family(nlevels; lc=lc0)
    prev_cells = 0; prev_h = 0.0
    for (i, model) in enumerate(fam)
        lvl = i - 1
        labels = get_face_labeling(model)
        Γ = BoundaryTriangulation(model, tags="boundary")
        nc = num_cells(model); hm = mesh_hmean(model)
        cr = prev_cells == 0 ? NaN : nc/prev_cells
        hr = prev_h == 0.0 ? NaN : prev_h/hm
        println("  level=$lvl: cells=$nc (×$(round(cr,digits=2))) hmean=$(round(hm,sigdigits=4)) (h↓×$(round(hr,digits=2))) bfacets=$(num_cells(Γ))")
        @assert "boundary" in labels.tag_to_name "level $lvl: missing 'boundary' tag"
        @assert "domain" in labels.tag_to_name "level $lvl: missing 'domain' tag"
        @assert num_cells(Γ) > 0 "level $lvl: boundary empty"
        export_mesh_h5(model, joinpath(outdir, "mesh_level$(lvl).h5"))
        prev_cells = nc; prev_h = hm
    end
    println("[mesh3d] OK — nested family built, tags survive, exported to $outdir/mesh_level*.h5")
end

if abspath(PROGRAM_FILE) == @__FILE__
    # julia --project=../../.. mesh3d.jl [seq ...]   -> build + export each sequence's meshes to
    #   results/meshes/<seq>/mesh_level*.h5  (default: all standard sequences). Then make mosaics with
    #   `python plot_mesh3d.py` (every sequence) or `python plot_mesh3d.py <seq>`.
    seqs = isempty(ARGS) ? ["structured", "nested_red", "frontal"] : ARGS
    for seq in seqs
        export_mesh_sequence(seq, build_sequence(seq))
    end
    println("[mesh3d] done — meshes under results/meshes/<seq>/. Plot mosaics: python plot_mesh3d.py"); flush(stdout)
end
