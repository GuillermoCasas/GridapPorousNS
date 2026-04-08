# src/run_simulation.jl

"""
    run_simulation.jl

# Role
This file is the high-level orchestration layer and primary entry point for the Porous Navier-Stokes solver. 
Its purpose is to convert structured, human-readable JSON configurations seamlessly into fully solved physical simulations.

# Pipeline Overview
1. **Configuration**: Parses hierarchical JSON parameters into strongly-typed physical and numerical schemas (`load_config_with_defaults`).
2. **Topology & Spaces**: Generates the geometric mesh and constructs the finite element Trial and Test function spaces, natively applying required boundary conditions (`build_fe_spaces`).
3. **Formulation Binding**: Dynamically constructs the mathematical `AbstractFormulation` (encapsulating exact viscous stress operators, drag laws, and projection policies) based on the inputs (`build_formulation`).
4. **Assembly**: Computes fundamental geometric scales (like element characteristic sizes `h`) and safely translates the theoretical continuous operators into discrete, evaluatable `FEOperator` closures.
5. **Execution**: Hands off the assembled, highly non-linear matrix system to the robust solver (`SafeNewtonSolver`) to converge to a discrete equilibrium, ultimately exporting the velocity and pressure fields.
"""

using LineSearches

"""
    build_formulation(phys::PhysicalProperties, solver_cfg::SolverConfig)

Constructs the continuous mathematical formulation object defining the physical model logic.
This object encapsulates the exact viscous operators, reaction/porosity fluid models 
(Darcy or Forchheimer-Ergun), and SubGrid scale projection strategies needed for the VMS approach.
"""
function build_formulation(phys::PhysicalProperties, solver_cfg::SolverConfig)
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
    if solver_cfg.experimental_reaction_mode == "standard" && rxn_mode == "Constant_Sigma"
        proj = ProjectResidualWithoutReactionWhenConstantSigma()
    end

    eps_val = phys.eps_val
    eps_floor = phys.eps_floor
    nu = phys.nu
    
    # 4. Bind the canonical PaperGeneralFormulation embodying the authoritative rigorous mathematical baseline.
    form = PaperGeneralFormulation(DeviatoricSymmetricViscosity(), reaction_law, proj, reg, nu, eps_val, eps_floor)
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

function run_simulation(config_path::String; 
                        dirichlet_tags=["walls"], 
                        dirichlet_masks=[(true,true)],
                        dirichlet_values=[VectorValue(0.0,0.0)])
    
    # Enforce pure mathematical parameter hierarchy (Zero internal hardcoding allowed).
    cfg = load_config_with_defaults(config_path)
    model = _build_default_mesh(cfg.domain, cfg.numerical_method.mesh)
    
    X, Y, kv, kp = build_fe_spaces(model, cfg.numerical_method.element_spaces, dirichlet_tags, dirichlet_masks, dirichlet_values)
    
    # Bugfix preservation: Formulate before probing underlying type signatures 
    form = build_formulation(cfg.phys, cfg.numerical_method.solver)
    
    # Quadrature logic strictly bound to mathematical formulation identity and velocity polynomial degree
    degree = get_quadrature_degree(typeof(form), kv)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    # Evaluate characteristic mapping scale `h`. The underlying scaling impacts inverse estimate constants implicitly.
    is_tri_val = (cfg.numerical_method.mesh.element_type == "TRI")
    cell_measures = get_cell_measure(Ω)
    h_array = lazy_map(v -> is_tri_val ? sqrt(2.0 * abs(v)) : sqrt(abs(v)), cell_measures)
    h = CellField(h_array, Ω)
    
    # Project analytical configurations smoothly into continuous CellFields
    alpha_0_val = cfg.domain.alpha_0
    alpha_fn(x) = alpha_0_val
    alpha_cf = CellField(alpha_fn, Ω)
    
    f_x_val = cfg.phys.f_x
    f_y_val = cfg.phys.f_y
    f_fn(x) = VectorValue(f_x_val, f_y_val)
    f_cf = CellField(f_fn, Ω)
    
    # Establish specific stabilization weights for momentum convective and viscous tau evaluations
    c_1, c_2 = get_c1_c2(typeof(form), kv)
    
    # CRITICAL: Prevent Julia implicit struct boxing inside inner iterative closures.
    # Extracted here locally to guarantee completely flat and explicitly type-stable Gridap AST structures.
    tau_reg_lim = cfg.phys.tau_regularization_limit
    freeze_cusp = cfg.numerical_method.solver.freeze_jacobian_cusp
    
    # --- NON-LINEAR FE OPERATOR CLOSURES ---
    res(x, y) = build_stabilized_weak_form_residual(x, y, form, dΩ, h, f_cf, alpha_cf, 0.0, nothing, nothing, c_1, c_2, tau_reg_lim)
    # Bugfix preservation: Exact Newtonian routing prevents exact cross-term dropping which guarantees ideal Fréchet gradients
    jac(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, form, dΩ, h, f_cf, alpha_cf, 0.0, nothing, nothing, c_1, c_2, tau_reg_lim, freeze_cusp, ExactNewtonMode())
    
    op = FEOperator(res, jac, X, Y)
    
    # Implement nonlinear iteration mechanism with explicit exact LU decomposition 
    nls = SafeNewtonSolver(LUSolver(), cfg.numerical_method.solver)
    solver = FESolver(nls)
    
    println("Solving non-linear system...")
    x_h = solve(solver, op)
    
    u_h, p_h = x_h
    
    export_results(cfg, model, u_h, p_h)
    
    return u_h, p_h, model, Ω, dΩ
end
