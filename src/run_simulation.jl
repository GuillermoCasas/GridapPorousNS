# src/run_simulation.jl

"""
    run_simulation.jl

# Role
The high-level orchestration layer and primary entry point of the solver: it turns a single
JSON configuration into a solved velocity/pressure field. `run_simulation` runs the pipeline
below end to end; the other functions here are the per-stage building blocks it calls.

# Pipeline Overview
1. **Configuration**: `load_frozen_config` parses the JSON into strongly-typed physical and
   numerical config structs, failing loudly on any missing field (no silent defaults).
2. **Topology & Spaces**: `_build_default_mesh` builds the discrete mesh; `build_fe_spaces`
   constructs the Taylor-Hood-style velocity/pressure Trial and Test spaces and applies the
   Dirichlet boundary conditions.
3. **Formulation Binding**: `build_formulation` assembles the continuous VMS weak form from a
   viscous stress operator, a porous-drag reaction law, an OSGS projection policy, and a
   velocity-floor regularization.
4. **Assembly**: computes geometric scales (the element characteristic size `h` and the
   stabilization constants `c₁`, `c₂`) and wires everything into the `FETopology` /
   `VMSFormulation` bundles consumed by the solver.
5. **Execution**: hands the nonlinear discrete system to `solve_system` (the safeguarded
   Newton/Picard solver), then exports the converged velocity and pressure.
"""

using LineSearches

"""
    build_formulation(phys::PhysicalProperties, num_method::NumericalMethodConfig)

Assembles the `PaperGeneralFormulation` — the continuous VMS weak form transcribed from
`theory/paper/article.tex`. It bundles four interchangeable pieces selected from the config:
the viscous stress operator, the porous-drag reaction law `σ` (constant or Forchheimer-Ergun),
the OSGS subgrid projection policy, and the velocity-floor regularization, together with the
viscosity `nu` and porosity-coupling parameter `physical_epsilon`.
"""
function build_formulation(phys::PhysicalProperties, num_method::NumericalMethodConfig)
    # 1. Reaction law σ: the macroscopic drag the porous matrix exerts on the fluid.
    #    Constant σ, or the Forchheimer-Ergun form σ = a(α) + b(α)|u| (linear + |u|-nonlinear).
    rxn_mode = phys.reaction_model
    if rxn_mode == "Constant_Sigma"
        reaction_law = ConstantSigmaLaw(phys.sigma_constant)
    else
        reaction_law = ForchheimerErgunLaw(phys.sigma_linear, phys.sigma_nonlinear)
    end

    # 2. Velocity-floor regularization. The Forchheimer term |u| is non-differentiable at u = 0,
    #    which would make the Jacobian singular there; this smooths |u| near zero so Newton stays
    #    well-posed. The floor scale combines a reference magnitude, an h-weight, and an ε offset.
    reg = SmoothVelocityFloor(phys.u_base_floor_ref, phys.h_floor_weight, phys.epsilon_floor, phys.velocity_magnitude_derivative_floor)

    # 3. Choose the OSGS subgrid projection policy. The default projects the full strong
    #    residual. For constant σ in "standard" mode the reaction term is dropped from the
    #    projection because its L² projection onto the FE space is exactly zero (paper §4.4).
    proj = ProjectFullResidual()
    if num_method.solver.experimental_reaction_mode == "standard" && rxn_mode == "Constant_Sigma"
        proj = ProjectResidualWithoutReactionWhenConstantSigma()
    end

    # physical_epsilon is the PHYSICAL compressibility ε_phys — it enters BOTH the residual and the Jacobian
    # (the mass-equation LHS and the manufactured source). The numerical penalty ε_num is carried
    # separately into the formulation; lagged to the iterate it survives ONLY in the Jacobian's
    # pressure block (Codina iterative penalty), so it must NOT be folded into physical_epsilon here.
    physical_epsilon = phys.physical_epsilon
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
    form = PaperGeneralFormulation(visc_op, reaction_law, proj, reg, nu, physical_epsilon;
                                   numerical_epsilon=phys.numerical_epsilon)
    return form
end

"""
    build_fe_spaces(model, elem::ElementSpacesConfig, dirichlet_tags, dirichlet_masks, dirichlet_values)

Builds the continuous-Galerkin velocity/pressure finite element spaces and returns the
monolithic Trial space `X = U×P`, monolithic Test space `Y = V×Q`, and the interpolation
orders `(kv, kp)`. Equal-order interpolation (kv == kp) is the primary configuration — the VMS
stabilization supplies the LBB inf-sup stability the pair would otherwise lack; unequal Taylor-Hood
pairs (kv > kp) are also representable.
"""
function build_fe_spaces(model, elem::ElementSpacesConfig, dirichlet_tags, dirichlet_masks, dirichlet_values)
    # Polynomial interpolation orders: kv for velocity, kp for pressure.
    kv = elem.k_velocity
    kp = elem.k_pressure

    # Reference elements: vector-valued Lagrangian for velocity, scalar Lagrangian for pressure.
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)

    # Test spaces: V (velocity) carries the Dirichlet structure (tags/masks mark constrained DOFs);
    # Q (pressure). H1-conformity makes the discrete gradients integrable, as the weak form needs.
    V = TestFESpace(model, refe_u, conformity=:H1, labels=get_face_labeling(model),
                    dirichlet_tags=dirichlet_tags, dirichlet_masks=dirichlet_masks)
    Q = TestFESpace(model, refe_p, conformity=:H1)

    # Trial spaces: U lifts the prescribed Dirichlet velocity values into V; pressure is unconstrained.
    U = TrialFESpace(V, dirichlet_values)
    P = TrialFESpace(Q)

    # Monolithic MultiField spaces so (u, p) are solved as one coupled system in each
    # Exact-Newton or Picard linear solve.
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])

    return X, Y, kv, kp
end

"""
    _build_default_mesh(domain_cfg::DomainConfig, mesh_cfg::MeshConfig)

Builds the canonical structured mesh for a run: a `CartesianDiscreteModel` over the
`bounding_box`, partitioned by `mesh_cfg.partition`. For `element_type == "TRI"` the
Cartesian cells are `simplexify`d into triangles; otherwise QUAD cells are kept.

Boundary face groups follow the conventional tag IDs used throughout the solver:
inlet = 7, outlet = 8, walls = 1..6. Downstream `build_fe_spaces` references these tag
names ("inlet", "outlet", "walls") to impose Dirichlet velocity conditions.
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

"""
    freefem_to_gmsh(freefem_msh::String, gmsh_msh::String)

Transcribe a FreeFem++ `.msh` (its header is `nv nt nbe` = #vertices, #triangles, #boundary
edges, then the vertex/triangle/edge records) into a gmsh 2.2 ASCII `.msh` that `GmshDiscreteModel`
can load. FreeFem boundary-edge labels 1/2/3/4 are mapped to named physical groups inlet/outlet/walls
(labels 4+ fold into walls), and the triangles into the 2D "domain" group; those names become Gridap
face-label tags. Used to import externally generated meshes (e.g. the Cocquet reference meshes).
"""
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

"""
    load_freefem_mesh(freefem_msh::String)

Load a FreeFem++ `.msh` as a Gridap `DiscreteModel`: convert it to a temporary gmsh file via
`freefem_to_gmsh`, hand that to `GmshDiscreteModel`, and clean up the temp directory afterwards.
"""
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
        # Optional persistent copy (e.g. into the test's data/ folder) so the mesh can be
        # inspected in the gmsh GUI / Paraview or imported elsewhere; the temp copy above is
        # what this run loads.
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

"""
    run_simulation(config_path; dirichlet_tags, dirichlet_masks, dirichlet_values)

Top-level entry point: solve one porous Navier-Stokes simulation described by the JSON at
`config_path` and return `(u_h, p_h, model, Ω, dΩ)` (the converged velocity/pressure fields plus
the mesh and integration measure). The keyword arguments set the velocity Dirichlet boundary:
which tag names are constrained, which components of each are fixed (the masks), and the imposed
values; the default is no-slip walls. Runs the full pipeline — load config, build mesh and FE
spaces, build the VMS formulation, assemble the stabilized system, and drive `solve_system`,
warning if the nonlinear solve fails to converge before exporting results.
"""
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
    
    # Element characteristic size h feeding the τ inverse estimates, via the configured convention
    # (StabilizationConfig.element_size): "shortest_edge" (Codina min edge, default) ≡ the legacy
    # √(2·A)/√A grid spacing on the structured mesh; "volume" reproduces the legacy formula; "diameter"
    # / "average_edge" rescale τ. See src/geometry.jl.
    h = element_size_field(Ω, model, element_size_convention(cfg.numerical_method.stabilization.element_size))
    
    # Lift the scalar config inputs into CellFields over Ω so the formulation can evaluate them
    # pointwise at quadrature: α is the (here uniform) porosity field; f is the constant body force.
    alpha_0_val = cfg.domain.alpha_0
    alpha_fn(x) = alpha_0_val
    alpha_cf = CellField(alpha_fn, Ω)

    f_x_val = cfg.physical_properties.f_x
    f_y_val = cfg.physical_properties.f_y
    f_fn(x) = VectorValue(f_x_val, f_y_val)
    f_cf = CellField(f_fn, Ω)

    # Stabilization constants c₁, c₂ entering τ₁/τ₂ (eq:Tau1/eq:Tau2): for equal-order interpolation
    # get_c1_c2 returns c₁ = 4k⁴, c₂ = 2k² (paper Remark after eq:conditions_on_num_param).
    #
    # The equal-order rule is then rescaled by the config-borne multipliers. This is the ONLY place c₁/c₂
    # are adjusted: `get_c1_c2` states the k-scaling, the config states the element-family calibration.
    # eq:conditions_on_num_param (c₁ > 2ξ·C̄_inv²) is element-dependent, and the paper adopts c₁ = 4k⁴ in
    # 2D but c₁ = 16k⁴ for the 3D tetrahedra (§7.2) — i.e. c1_multiplier 1.0 vs 4.0 — because C̄_inv is
    # markedly larger there. Threading it here (rather than scaling inside the τ kernel) keeps the
    # multiplier out of the quadrature-point hot path and puts it in the frozen config, so every result
    # records the c₁ that produced it.
    c_1_base, c_2_base = get_c1_c2(typeof(form), kv)
    c_1 = cfg.numerical_method.stabilization.c1_multiplier * c_1_base
    c_2 = cfg.numerical_method.stabilization.c2_multiplier * c_2_base
    
    # Unconstrained (no-Dirichlet) test spaces V_free/Q_free — required for the OSGS L² projection of
    # the strong residual; projecting on the Dirichlet-constrained space breaks O(h^{k+1}) convergence.
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)
    
    setup = FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h, f_cf, alpha_cf, CellField(0.0, Ω))
    formulation = VMSFormulation(form, c_1, c_2)
    
    sol_cfg = cfg.numerical_method.solver
    p_ls = instantiate_linear_solver(sol_cfg.linear_solver)   # config-selected backend: LU (default) | ILU_GMRES

    # Build the (picard, newton) solver pair via the shared `build_iter_solvers`, the same builder
    # the MMS harness uses. Production runs use the plain scalar-ftol path: a Newton residual
    # tolerance (`ftol`) and a separate, looser Picard hand-off tolerance (`picard_ftol`),
    # with fixed iteration budgets. Dynamic Re/Da-driven budgets and per-field relative tolerances
    # stay harness-only — production has no manufactured solution or characteristic Re/Da to key them
    # to. `noise_floor_success_max_ftol_multiple = Inf` disables noise-floor success acceptance.
    # The no-progress STALL GUARD comes straight from the config (`newton_stall_*`; defaults of 0
    # turn it off). The Newton↔Picard ping-pong is not wired here — `solve_system` reads `pingpong_*`
    # from `sol_cfg` and constructs the gain-targeted Picard itself, so it is opted into via config.
    solvers = build_iter_solvers(sol_cfg, p_ls;
        newton_max_iters = sol_cfg.newton_iterations,
        picard_max_iters = sol_cfg.picard_iterations,
        newton_ftol = sol_cfg.ftol,
        picard_ftol = sol_cfg.picard_ftol,
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
