# src/run_simulation.jl

"""
    run_simulation.jl

# Role
This file is the high-level orchestration layer and primary entry point for the Porous Navier-Stokes solver. 
Its purpose is to convert structured, human-readable JSON configurations seamlessly into fully solved physical simulations.

# Pipeline Overview
1. **Configuration**: Parses hierarchical JSON parameters into strongly-typed physical and numerical schemas (`load_frozen_config`, the strict single-file loader that fails on any missing field).
2. **Topology & Spaces**: Generates the geometric mesh and constructs the finite element Trial and Test function spaces, natively applying required boundary conditions (`build_fe_spaces`).
3. **Formulation Binding**: Dynamically constructs the mathematical `AbstractFormulation` (encapsulating exact viscous stress operators, drag laws, and projection policies) based on the inputs (`build_formulation`).
4. **Assembly**: Computes fundamental geometric scales (like element characteristic sizes `h`) and safely translates the theoretical continuous operators into discrete, evaluatable `FEOperator` closures.
5. **Execution**: Hands off the assembled, highly non-linear matrix system to the robust solver (`SafeNewtonSolver`) to converge to a discrete equilibrium, ultimately exporting the velocity and pressure fields.
"""

using LineSearches

"""
    build_formulation(phys::PhysicalProperties, num_method::NumericalMethodConfig)

Constructs the continuous mathematical formulation object defining the physical model logic.
This object encapsulates the exact viscous operators, reaction/porosity fluid models 
(Darcy or Forchheimer-Ergun), and SubGrid scale projection strategies needed for the VMS approach.
"""
function build_formulation(phys::PhysicalProperties, num_method::NumericalMethodConfig)
    # 1. Define the macroscopic drag reaction law imposed by the porous matrix.
    rxn_mode = phys.reaction_model
    if rxn_mode == "Constant_Sigma"
        reaction_law = ConstantSigmaLaw(phys.sigma_constant)
    else
        reaction_law = ForchheimerErgunLaw(phys.sigma_linear, phys.sigma_nonlinear)
    end

    # 2. Velocity regularization strictly averts singular Jacobian matrices and non-differentiable 
    # states when local velocity magnitudes approach exact zero in non-linear or Forchheimer flows.
    reg = SmoothVelocityFloor(phys.u_base_floor_ref, phys.h_floor_weight, phys.epsilon_floor)

    # 3. Determine the algebraic SubGrid projection policy. Normally projects the full 
    # convective residual; optionally trims reactions in legacy comparison modes.
    proj = ProjectFullResidual()
    if num_method.solver.experimental_reaction_mode == "standard" && rxn_mode == "Constant_Sigma"
        proj = ProjectResidualWithoutReactionWhenConstantSigma()
    end

    eps_val = phys.eps_val
    nu = phys.nu

    visc_type = num_method.viscous_operator_type
    if visc_type == "DeviatoricSymmetric"
        visc_op = DeviatoricSymmetricViscosity()
    elseif visc_type == "SymmetricGradient"
        visc_op = SymmetricGradientViscosity()
    else
        visc_op = LaplacianPseudoTractionViscosity()
    end

    # 4. Assemble the chosen viscous operator, reaction law, projection policy and regularization
    #    into the PaperGeneralFormulation (the VMS weak form transcribed from article.tex).
    form = PaperGeneralFormulation(visc_op, reaction_law, proj, reg, nu, eps_val)
    return form
end

"""
    build_fe_spaces(model, elem::ElementSpacesConfig, dirichlet_tags, dirichlet_masks, dirichlet_values)

Assembles the continuous Galerkin Test and Trial spaces. Generates a MultiField structure 
for the tightly-coupled velocity-pressure system. LBB stability guides the respective `kv` and `kp` 
interpolation orders.
"""
function build_fe_spaces(model, elem::ElementSpacesConfig, dirichlet_tags, dirichlet_masks, dirichlet_values)
    # Standard interpolation degree limits
    kv = elem.k_velocity
    kp = elem.k_pressure
    
    # Abstract definitions of the exact finite elements (Lagrangian polynomials).
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    
    # Test spaces V (Velocity) and Q (Pressure). H1-conformity enforces the integrability of PDE gradients.
    V = TestFESpace(model, refe_u, conformity=:H1, labels=get_face_labeling(model), 
                    dirichlet_tags=dirichlet_tags, dirichlet_masks=dirichlet_masks)
    Q = TestFESpace(model, refe_p, conformity=:H1)
    
    # Trial spaces built inheriting the strong Dirichlet boundary constraint values.
    U = TrialFESpace(V, dirichlet_values)
    P = TrialFESpace(Q)
    
    # Bundle into strongly-coupled Monolithic MultiField space enabling joint matrix inversion 
    # during Exact Newton or Picard sweeps.
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    return X, Y, kv, kp
end

"""
    run_simulation(config_path::String; ...)

The foundational entry execution method. Evaluates JSON configurations to enforce parameter strictness. 
Generates mesh mappings, extracts element geometries, determines VMS mathematical constants dynamically, 
and drives the SafeNewton robust solver.
"""
function _build_default_mesh(domain_cfg::DomainConfig, mesh_cfg::MeshConfig)
    domain = Tuple(domain_cfg.bounding_box)
    partition = Tuple(mesh_cfg.partition)
    
    if mesh_cfg.element_type == "TRI"
        model = CartesianDiscreteModel(domain, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain, partition)
    end
    
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [7])
    add_tag_from_tags!(labels, "outlet", [8])
    add_tag_from_tags!(labels, "walls", [1, 2, 3, 4, 5, 6])

    return model
end

function freefem_to_gmsh(freefem_msh::String, gmsh_msh::String)
    nv, nt, nbe = 0, 0, 0
    verts = Vector{NTuple{3,Float64}}()   # (x, y, label) — label unused in gmsh nodes
    tris  = Vector{NTuple{4,Int}}()       # (v1, v2, v3, label) — FreeFem subdomain label
    edges = Vector{NTuple{3,Int}}()       # (e1, e2, label)
    open(freefem_msh, "r") do f
        h = split(strip(readline(f)))
        nv = parse(Int, h[1]); nt = parse(Int, h[2]); nbe = parse(Int, h[3])
        for _ in 1:nv
            p = split(strip(readline(f)))
            push!(verts, (parse(Float64, p[1]), parse(Float64, p[2]), parse(Float64, p[3])))
        end
        for _ in 1:nt
            p = split(strip(readline(f)))
            push!(tris, (parse(Int, p[1]), parse(Int, p[2]), parse(Int, p[3]), parse(Int, p[4])))
        end
        for _ in 1:nbe
            p = split(strip(readline(f)))
            push!(edges, (parse(Int, p[1]), parse(Int, p[2]), parse(Int, p[3])))
        end
    end

    # Physical-group ids (arbitrary distinct positive integers per group, named below)
    pg_inlet, pg_outlet, pg_walls, pg_domain = 1, 2, 3, 4

    mkpath(dirname(gmsh_msh))
    open(gmsh_msh, "w") do f
        println(f, "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat")
        # Physical groups (named) — GridapGmsh maps these names directly to face-label tags.
        println(f, "\$PhysicalNames\n4")
        println(f, "1 $(pg_inlet) \"inlet\"")
        println(f, "1 $(pg_outlet) \"outlet\"")
        println(f, "1 $(pg_walls) \"walls\"")
        println(f, "2 $(pg_domain) \"domain\"")
        println(f, "\$EndPhysicalNames")
        # Nodes (1-based ids; gmsh wants 3D coords)
        println(f, "\$Nodes\n$nv")
        for i in 1:nv
            x, y, _ = verts[i]
            println(f, "$i $x $y 0.0")
        end
        println(f, "\$EndNodes")
        # Elements: boundary edges first (type 1 = 2-node line), then triangles (type 2).
        ne = nbe + nt
        println(f, "\$Elements\n$ne")
        elem_id = 0
        for (e1, e2, lab) in edges
            elem_id += 1
            pg = lab == 1 ? pg_inlet :
                 lab == 2 ? pg_outlet :
                 lab == 3 ? pg_walls :
                 lab == 4 ? pg_walls : pg_walls
            # `elem_id elem_type num_tags phys_grp elem_grp v1 v2`
            println(f, "$elem_id 1 2 $pg $pg $e1 $e2")
        end
        for (v1, v2, v3, _lab) in tris
            elem_id += 1
            println(f, "$elem_id 2 2 $pg_domain $pg_domain $v1 $v2 $v3")
        end
        println(f, "\$EndElements")
    end
    return gmsh_msh
end

function load_freefem_mesh(freefem_msh::String)
    dir = mktempdir()
    try
        gmsh_path = joinpath(dir, splitext(basename(freefem_msh))[1] * ".gmsh.msh")
        freefem_to_gmsh(freefem_msh, gmsh_path)
        return GmshDiscreteModel(gmsh_path)
    finally
        rm(dir; recursive=true, force=true)
    end
end

"""
    build_unstructured_model(N::Int; domain=(0.0, 2.0, 0.0, 1.0),
                              wall_divisions::Symbol=:uniform, save_msh::String="",
                              algorithm::Int=5)

Build an UNSTRUCTURED (Delaunay) triangular mesh of the rectangle `domain` via the
GridapGmsh-bundled gmsh API, returning a Gridap `DiscreteModel`. This mirrors the
FreeFem `buildmesh` meshes used by Cocquet et al. (2020), as opposed to the
structured Cartesian-simplexified mesh produced by `_build_default_mesh`.

**Boundary divisions are PRESCRIBED, interior is Delaunay** — exactly how FreeFem's
`buildmesh(a(N)+b(N)+…)` works (it fixes the vertex count on each boundary part `a,b,…`
and fills the interior with an unstructured triangulation). The gmsh equivalent is a
*transfinite curve* on each boundary edge with NO transfinite surface, so the boundary
node count is exact while the interior stays irregular. We prescribe a uniform edge
length `≈ 1/N`: the unit-height inlet/outlet get `N` segments and the length-2 walls get
`2N` segments, giving triangle diameter `≈ √2/N` (the paper's reported `h = √2/N`) and
matching the structured mesh's resolution per `N`. (FreeFem's literal "N per part" with
length-2 walls would instead give `2/N` wall edges; the uniform-`1/N` choice keeps the
resolution comparable to both the paper's stated `h` and the structured benchmark.)

Boundary curves are placed in named physical groups `"inlet"` (x = x_min),
`"walls"` (y = y_min ∪ y = y_max) and `"outlet"` (x = x_max), plus a 2D group
`"domain"`; `GmshDiscreteModel` preserves those names as Gridap face-label tags, so
`build_fe_spaces(model, …, ["inlet","walls"], …)` resolves them exactly as for the
structured model (the outlet is a natural/traction-free boundary — no Dirichlet tag).

The mesh generator is pinned (`Mesh.Algorithm=5`, single-threaded) for reproducibility
given the gmsh version locked in the Manifest. `N` is the nominal resolution the
convergence study sweeps against.
"""
function build_unstructured_model(N::Int; domain=(0.0, 2.0, 0.0, 1.0),
                                   wall_divisions::Symbol=:uniform, save_msh::String="",
                                   algorithm::Int=5)
    x0, x1, y0, y1 = domain
    # Prescribed boundary divisions. setTransfiniteCurve takes the number of NODES = segments + 1.
    # Walls span length |x1-x0|, inlet/outlet span |y1-y0|.
    #   :uniform  — uniform edge length 1/N everywhere (walls get |x1-x0|·N divisions, inlets get
    #               |y1-y0|·N). Triangle diameter ≈ √2/N (matches the paper's stated h).
    #   :freefem  — LITERAL Cocquet recipe: N divisions on each of the four boundary parts,
    #               irrespective of length, so walls (length 2) end up 2× coarser per edge than
    #               inlets (length 1). This is what `buildmesh(a(N)+b(N)+c(N)+d(N))` does verbatim.
    ny = max(round(Int, abs(y1 - y0) * N), 1)
    nx = wall_divisions === :freefem ? max(round(Int, abs(y1 - y0) * N), 1) :
                                       max(round(Int, abs(x1 - x0) * N), 1)
    lc = 1.0 / N                                  # fallback size field for the interior

    gmsh = GridapGmsh.gmsh
    dir = ""
    mshfile = ""
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        # 5 = Delaunay (gmsh default in our setup), 7 = BAMG (the mesher FreeFem uses for buildmesh),
        # 6 = Frontal-Delaunay, 1 = MeshAdapt. Algorithm 7 reproduces FreeFem's mesh generator exactly.
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.model.add("cocquet_$(N)")

        p1 = gmsh.model.geo.addPoint(x0, y0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(x1, y0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(x1, y1, 0.0, lc)
        p4 = gmsh.model.geo.addPoint(x0, y1, 0.0, lc)

        lbot = gmsh.model.geo.addLine(p1, p2)   # y = y_min  (wall)
        lout = gmsh.model.geo.addLine(p2, p3)   # x = x_max  (outlet, natural)
        ltop = gmsh.model.geo.addLine(p3, p4)   # y = y_max  (wall)
        lin  = gmsh.model.geo.addLine(p4, p1)   # x = x_min  (inlet)

        cl   = gmsh.model.geo.addCurveLoop([lbot, lout, ltop, lin])
        surf = gmsh.model.geo.addPlaneSurface([cl])

        # Prescribe boundary node counts (FreeFem buildmesh behaviour). NO setTransfiniteSurface ⇒ the
        # interior is meshed by the unstructured Delaunay algorithm constrained to these boundary nodes.
        gmsh.model.geo.mesh.setTransfiniteCurve(lbot, nx + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(ltop, nx + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(lin,  ny + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(lout, ny + 1)
        gmsh.model.geo.synchronize()

        # Physical-group NAMES become Gridap face-label tag names.
        g_in  = gmsh.model.addPhysicalGroup(1, [lin]);        gmsh.model.setPhysicalName(1, g_in,  "inlet")
        g_wl  = gmsh.model.addPhysicalGroup(1, [lbot, ltop]); gmsh.model.setPhysicalName(1, g_wl,  "walls")
        g_out = gmsh.model.addPhysicalGroup(1, [lout]);       gmsh.model.setPhysicalName(1, g_out, "outlet")
        g_dom = gmsh.model.addPhysicalGroup(2, [surf]);       gmsh.model.setPhysicalName(2, g_dom, "domain")

        gmsh.model.mesh.generate(2)

        dir = mktempdir()
        mshfile = joinpath(dir, "cocquet_$(N).msh")
        gmsh.write(mshfile)
        # Optional persistent copy (e.g. into the test's data/ folder) — kept verbatim alongside
        # the temp copy so the file can be inspected (gmsh GUI, Paraview) or imported elsewhere.
        if !isempty(save_msh)
            mkpath(dirname(save_msh))
            gmsh.write(save_msh)
        end
    finally
        # Close our generation session BEFORE the file loader opens its own gmsh session.
        gmsh.finalize()
    end

    model = GmshDiscreteModel(mshfile)
    rm(dir; recursive=true, force=true)
    return model
end

function run_simulation(config_path::String; 
                        dirichlet_tags=["walls"], 
                        dirichlet_masks=[(true,true)],
                        dirichlet_values=[VectorValue(0.0,0.0)])
    
    # Strict load: every field must be present in the JSON; no silent defaults.
    cfg = load_frozen_config(config_path)
    model = _build_default_mesh(cfg.domain, cfg.numerical_method.mesh)
    
    X, Y, kv, kp = build_fe_spaces(model, cfg.numerical_method.element_spaces, dirichlet_tags, dirichlet_masks, dirichlet_values)
    
    # Build the formulation first: the quadrature degree below depends on its type and reaction law.
    form = build_formulation(cfg.physical_properties, cfg.numerical_method)

    # Quadrature degree from the formulation type, velocity order kv, and the reaction law's minimum.
    degree = get_quadrature_degree(typeof(form), kv, form.reaction_law)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    # Element characteristic size h: √(2·area) for TRI, √area for QUAD (feeds the τ inverse estimates).
    is_tri_val = (cfg.numerical_method.mesh.element_type == "TRI")
    cell_measures = get_cell_measure(Ω)
    h_array = lazy_map(v -> is_tri_val ? sqrt(2.0 * abs(v)) : sqrt(abs(v)), cell_measures)
    h = CellField(h_array, Ω)
    
    # Project analytical configurations smoothly into continuous CellFields
    alpha_0_val = cfg.domain.alpha_0
    alpha_fn(x) = alpha_0_val
    alpha_cf = CellField(alpha_fn, Ω)
    
    f_x_val = cfg.physical_properties.f_x
    f_y_val = cfg.physical_properties.f_y
    f_fn(x) = VectorValue(f_x_val, f_y_val)
    f_cf = CellField(f_fn, Ω)
    
    # Establish specific stabilization weights for momentum convective and viscous tau evaluations
    c_1, c_2 = get_c1_c2(typeof(form), kv)
    
    # Unconstrained (no-Dirichlet) test spaces V_free/Q_free — required for the OSGS L² projection of
    # the strong residual; projecting on the Dirichlet-constrained space breaks O(h^{k+1}) convergence.
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)
    
    setup = FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h, f_cf, alpha_cf, CellField(0.0, Ω))
    formulation = VMSFormulation(form, c_1, c_2)
    
    sol_cfg = cfg.numerical_method.solver
    p_ls = LUSolver()

    # [P0 shared builder + P6 parity] Production builds the (picard, newton) pair through the same builder
    # as the MMS harness. It keeps the bare scalar-ftol path (distinct picard_handoff_ftol / ftol; no
    # per-field relative tolerances; static iteration budgets — option (b): production has no manufactured
    # solution or characteristic Re/Da to drive the dynamic budgets, so those stay harness-only) and passes
    # `noise_floor_success_max_ftol_multiple = Inf` to reproduce the prior positional-ctor default byte-for-byte.
    # P6 wires the no-progress STALL GUARD from the config (`newton_stall_*`, default 0 ⇒ off ⇒ bit-identical);
    # the Newton↔Picard PING-PONG needs no plumbing here — `solve_system` reads `pingpong_*` from `sol_cfg`
    # and builds the gain-targeted Picard itself, so a production user opts in purely via config.
    solvers = build_iter_solvers(sol_cfg, p_ls;
        newton_max_iters = sol_cfg.newton_iterations,
        picard_max_iters = sol_cfg.picard_iterations,
        newton_ftol = sol_cfg.ftol,
        picard_ftol = sol_cfg.picard_handoff_ftol,
        noise_floor_success_max_ftol_multiple = Inf,
        stall_window = sol_cfg.newton_stall_window,
        stall_min_rel_improvement = sol_cfg.newton_stall_min_rel_improvement)

    iter_solvers = StageSolvers(solvers.picard, solvers.newton)
    
    x0 = interpolate_everywhere([VectorValue(0.0,0.0), 0.0], X)
    
    println("\n[*] Solving stabilized VMS system...")
    success, mms_plateau_success, final_x0, iters, eval_time = solve_system(setup, formulation, iter_solvers, cfg, x0)

    if !success
        @warn "Nonlinear solver did not converge; check the configuration and mesh resolution."
    end
    
    x_h = final_x0
    
    u_h, p_h = x_h
    
    export_results(cfg, model, u_h, p_h)
    
    return u_h, p_h, model, Ω, dΩ
end
