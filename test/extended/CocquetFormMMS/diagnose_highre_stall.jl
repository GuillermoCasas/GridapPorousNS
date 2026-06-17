# [debugging-lore] Diagnostic for the VMS high-Re stall (Re=1e5, α=0.1) on the CocquetFormMMS cell.
# Reuses run_test.jl's helpers (build_mms_formulation, build_porosity_field, _build_local_mesh) and the
# exact per-cell setup, then answers two questions:
#   (1) MECHANISM: residual + scale-free ε_M/ε_C AT THE EXACT SOLUTION, Re=1 (works) vs Re=1e5 (stalls).
#       If ε_M is already ≫ tol at u_exact and won't reduce, it's a residual/gate problem, not a basin one.
#   (2) CONTINUATION (user hypothesis): does a Re-ramp with warm-starting reach Re=1e5 where a cold solve fails?
# Output: stdout only. Run:  julia --project=../../.. diagnose_highre_stall.jl

include("run_test.jl")   # defines helpers + imports; the __FILE__ guard keeps run_mms from auto-running

using Printf

const ALPHA0 = 0.1
const NMESH  = 10
const KV = 1
const KP = 1
const METHOD = "ASGS"

# Build EVERYTHING for one (Re) cell — mirrors run_test.jl lines ~435-620 exactly.
function build_cell(test_dict, Re)
    pp_in = get(test_dict, "physical_properties", Dict())
    nm_dict = get(test_dict, "numerical_method", Dict())
    config_dict = Dict(
        "physical_properties" => Dict(
            "nu" => 1.0, "eps_val" => 1e-8,
            "reaction_model" => get(pp_in, "reaction_model", "Constant_Sigma"),
            "sigma_constant" => get(pp_in, "sigma_constant", 1.0),
            "sigma_linear" => get(pp_in, "sigma_linear", 0.0),
            "sigma_nonlinear" => get(pp_in, "sigma_nonlinear", 0.0),
        ),
        "domain" => Dict(
            "alpha_0" => 0.4,
            "bounding_box" => test_dict["domain"]["bounding_box"],
            "r_1" => test_dict["domain"]["r_1"],
            "r_2" => test_dict["domain"]["r_2"],
        ),
        "numerical_method" => Dict(
            "viscous_operator_type" => get(nm_dict, "viscous_operator_type", "DeviatoricSymmetric"),
            "element_spaces" => Dict("k_velocity" => KV, "k_pressure" => KP),
            "mesh" => Dict("element_type" => "TRI", "partition" => [NMESH, NMESH]),
            "stabilization" => Dict("method" => METHOD),
            "solver" => get(nm_dict, "solver", Dict()),
        ),
    )
    config = PorousNSSolver.load_config_from_dict(config_dict)
    model = _build_local_mesh(config.domain, config.numerical_method.mesh)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])

    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, KV)
    refe_p = ReferenceFE(lagrangian, Float64, KP)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
    Q = TestFESpace(model, refe_p, conformity=:H1)
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)

    quad_rxn_law = config.physical_properties.reaction_model == "Constant_Sigma" ?
        PorousNSSolver.ConstantSigmaLaw(0.0) : PorousNSSolver.ForchheimerErgunLaw(0.0, 0.0)
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, KV, quad_rxn_law)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree + 4)
    h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
    h_cf = CellField(collect(h_array), Ω)

    Y = MultiFieldFESpace([V, Q])
    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, KV)
    tau_reg_lim = config.physical_properties.tau_regularization_limit

    alpha_infty = 1.0
    alpha_field = build_porosity_field(config, ALPHA0, alpha_infty)
    alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)

    U_amp = 1.0; L = 1.0; Da = 1.0
    form = build_mms_formulation(config, Da, Re, U_amp, L, alpha_infty)
    mms = PorousNSSolver.Paper2DMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)
    U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)
    u_final = PorousNSSolver.get_u_ex(mms)
    p_final = PorousNSSolver.get_p_ex(mms)
    U = TrialFESpace(V, u_final); P = TrialFESpace(Q, p_final)
    X = MultiFieldFESpace([U, P])
    f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)

    s = config.numerical_method.solver
    # dynamic ftol exactly as the harness computes it for this (n, kv)
    spatial_err_est = (1.0/NMESH)^(KV+1)
    dynamic_ftol = max(s.ftol, min(s.dynamic_ftol_ceiling, s.dynamic_ftol_spatial_safety_factor * spatial_err_est))
    condition_scaling = Float64(NMESH)^2 * max(1.0, Float64(Re))
    dyn_nf = min(s.stagnation_noise_floor, max(s.condition_noise_floor_absolute_min, s.condition_noise_floor_baseline * condition_scaling))
    dyn_nf = max(dyn_nf, dynamic_ftol * s.condition_noise_floor_safety_factor)
    local_picard_it = Re >= s.dynamic_picard_re_threshold ? max(s.picard_iterations, s.dynamic_picard_re_iterations) : s.picard_iterations
    local_newton_it = Re >= s.dynamic_newton_re_threshold ? max(s.newton_iterations, s.dynamic_newton_re_iterations) : s.newton_iterations

    nls_picard = PorousNSSolver.SafeNewtonSolver(LUSolver(), local_picard_it, s.max_increases, s.xtol, dynamic_ftol, s.linesearch_alpha_min, s.armijo_c1, s.divergence_merit_factor, dyn_nf, s.max_linesearch_iterations, s.linesearch_contraction_factor; mode=:picard)
    nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), local_newton_it, s.max_increases, s.xtol, dynamic_ftol, s.linesearch_alpha_min, s.armijo_c1, s.divergence_merit_factor, dyn_nf, s.max_linesearch_iterations, s.linesearch_contraction_factor)

    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
    iter_solvers = PorousNSSolver.StageSolvers(FESolver(nls_picard), FESolver(nls_newton))
    x0_exact = interpolate_everywhere([u_final, p_final], X)

    nU = num_free_dofs(U); nP = num_free_dofs(P)
    return (; config, setup, formulation, iter_solvers, X, Y, U, P, nU, nP, x0_exact, u_final, p_final, U_c, P_c, dΩ, dynamic_ftol)
end

# Residual + scale-free convergence measure AT a given state x0.
function probe_state(cell, x0; tag="")
    setup, formulation, config = cell.setup, cell.formulation, cell.config
    phys = config.physical_properties
    res_fn(x, y) = PorousNSSolver.build_stabilized_weak_form_residual(x, y, setup, formulation, phys; pi_u=nothing, pi_p=nothing)
    b = assemble_vector(y -> res_fn(x0, y), cell.Y)
    bv = view(b, 1:cell.nU); bp = view(b, (cell.nU+1):(cell.nU+cell.nP))
    conv = PorousNSSolver.build_convergence_probe(setup, formulation, config.numerical_method.solver.eps_tol_momentum, config.numerical_method.solver.eps_tol_mass)
    cm = conv(get_free_dof_values(x0), b, [1:cell.nU, (cell.nU+1):(cell.nU+cell.nP)])
    @printf("    [%s] ‖R‖∞=%.3e  ‖R_vel‖₂=%.3e  ‖R_pre‖₂=%.3e | ε_M=%.3e (tol %.0e)  ε_C=%.3e | D_M=%.3e  r_M=%.3e | degenerate=%s\n",
            tag, norm(b, Inf), norm(bv), norm(bp), cm.eps_M, config.numerical_method.solver.eps_tol_momentum, cm.eps_C, cm.D_M, cm.r_M, cm.degenerate)
    @printf("        D_M terms: conv=%.2e visc=%.2e ∇p=%.2e react=%.2e f=%.2e\n",
            cm.terms.convection, cm.terms.viscous, cm.terms.pressure_grad, cm.terms.resistance, cm.terms.body_force)
    return cm
end

println("="^90)
println("PART 1 — MECHANISM: residual + scale-free measures AT THE EXACT SOLUTION")
println("="^90)
test_dict = JSON3.read(read(joinpath(@__DIR__, "data", "cocquet_form_mms_vms.json"), String), Dict)
for Re in (1.0, 1.0e5)
    println("\n--- Re=$Re, α=$ALPHA0, N=$NMESH, k=$KV, $METHOD ---")
    cell = build_cell(test_dict, Re)
    probe_state(cell, cell.x0_exact; tag="@u_exact")
end

println("\n" * "="^90)
println("PART 2 — CONTINUATION: cold Re=1e5 vs Re-ramp warm-started to 1e5")
println("="^90)

# (a) Cold solve directly at Re=1e5 from the exact solution (reproduce the stall).
println("\n[COLD] solve_system directly at Re=1e5 from u_exact:")
cold = build_cell(test_dict, 1.0e5)
csucc, _, cxf, cit, ct = PorousNSSolver.solve_system(cold.setup, cold.formulation, cold.iter_solvers, cold.config, cold.x0_exact)
@printf("  COLD Re=1e5: success=%s iters=%s time=%.1fs\n", csucc, cit, ct)

# (b) Re-continuation: ramp geometrically, warm-start each level from the previous solution.
ramp = [1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0, 3162.0, 1.0e4, 3.16e4, 1.0e5]
println("\n[RAMP] Re-continuation, warm-started:")
x0 = nothing
for (i, Re) in enumerate(ramp)
    cell = build_cell(test_dict, Re)
    start = x0 === nothing ? cell.x0_exact : FEFunction(cell.X, copy(x0))
    succ, _, xf, it, tt = PorousNSSolver.solve_system(cell.setup, cell.formulation, cell.iter_solvers, cell.config, start)
    if succ
        global x0 = copy(get_free_dof_values(xf))
        eu = sqrt(abs(sum(∫((cell.u_final - xf[1]) ⋅ (cell.u_final - xf[1]))cell.dΩ))) / cell.U_c
        @printf("  level %2d Re=%9.1f: success=%s iters=%s time=%.1fs  err_u_L2(norm)=%.3e\n", i, Re, succ, it, tt, eu)
    else
        @printf("  level %2d Re=%9.1f: success=%s iters=%s time=%.1fs  <== RAMP BROKE HERE\n", i, Re, succ, it, tt)
        break
    end
end
println("\nDONE.")
