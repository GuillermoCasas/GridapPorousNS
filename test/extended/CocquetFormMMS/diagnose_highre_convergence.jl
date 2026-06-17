# [debugging-lore] DECISIVE: is the VMS solution actually OPTIMAL at Re=1e5, with only the algebraic
# gate unreachable? Convergence study at Re=1e5, α=0.1, ASGS, k=1 across N=[10,20,40], reaching 1e5 by
# Re-continuation through production solve_system with a RELAXED gate (eps_tol_momentum override), then
# reporting the normalized err_u L2 + observed rate + the achievable ε_M floor at each mesh.
#   rate≈2 (O(h²) for P1)  => VMS works at Re=1e5; production NaN is purely the over-tight gate (gate fix).
#   rate<2 / err_u large    => VMS genuinely degrades at high Re (deeper stabilization issue).
# Run:  julia --project=../../.. diagnose_highre_convergence.jl

include("run_test.jl")
using Printf, LinearAlgebra

const ALPHA0=0.1; const KV=1; const KP=1; const METHOD="ASGS"

function build_cell(test_dict, Re, N; eps_tol_M=nothing)
    pp_in = get(test_dict,"physical_properties",Dict()); nm_dict = get(test_dict,"numerical_method",Dict())
    solver_dict = copy(get(nm_dict,"solver",Dict()))
    eps_tol_M === nothing || (solver_dict["eps_tol_momentum"] = eps_tol_M)
    config_dict = Dict(
        "physical_properties"=>Dict("nu"=>1.0,"eps_val"=>1e-8,"reaction_model"=>get(pp_in,"reaction_model","Constant_Sigma"),
            "sigma_constant"=>get(pp_in,"sigma_constant",1.0),"sigma_linear"=>get(pp_in,"sigma_linear",0.0),"sigma_nonlinear"=>get(pp_in,"sigma_nonlinear",0.0)),
        "domain"=>Dict("alpha_0"=>0.4,"bounding_box"=>test_dict["domain"]["bounding_box"],"r_1"=>test_dict["domain"]["r_1"],"r_2"=>test_dict["domain"]["r_2"]),
        "numerical_method"=>Dict("viscous_operator_type"=>get(nm_dict,"viscous_operator_type","DeviatoricSymmetric"),
            "element_spaces"=>Dict("k_velocity"=>KV,"k_pressure"=>KP),"mesh"=>Dict("element_type"=>"TRI","partition"=>[N,N]),
            "stabilization"=>Dict("method"=>METHOD),"solver"=>solver_dict))
    config = PorousNSSolver.load_config_from_dict(config_dict)
    model = _build_local_mesh(config.domain, config.numerical_method.mesh)
    labels = get_face_labeling(model); add_tag_from_tags!(labels,"all_boundaries",[1,2,3,4,5,6,7,8])
    refe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},KV); refe_p = ReferenceFE(lagrangian,Float64,KP)
    V = TestFESpace(model,refe_u,conformity=:H1,labels=labels,dirichlet_tags=["all_boundaries"]); Q = TestFESpace(model,refe_p,conformity=:H1)
    V_free = TestFESpace(model,refe_u,conformity=:H1); Q_free = TestFESpace(model,refe_p,conformity=:H1)
    quad_rxn_law = config.physical_properties.reaction_model=="Constant_Sigma" ? PorousNSSolver.ConstantSigmaLaw(0.0) : PorousNSSolver.ForchheimerErgunLaw(0.0,0.0)
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation,KV,quad_rxn_law)
    Ω = Triangulation(model); dΩ = Measure(Ω,degree+4)
    h_cf = CellField(collect(lazy_map(v->sqrt(2.0*abs(v)),get_cell_measure(Ω))),Ω)
    Y = MultiFieldFESpace([V,Q]); c_1,c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation,KV)
    tau_reg_lim = config.physical_properties.tau_regularization_limit
    alpha_field = build_porosity_field(config,ALPHA0,1.0); alpha_cf = CellField(x->PorousNSSolver.alpha(alpha_field,x),Ω)
    form = build_mms_formulation(config,1.0,Re,1.0,1.0,1.0)
    mms = PorousNSSolver.Paper2DMMS(form,1.0,alpha_field;L=1.0,alpha_infty=1.0)
    U_c,P_c = PorousNSSolver.get_characteristic_scales(mms); u_final=PorousNSSolver.get_u_ex(mms); p_final=PorousNSSolver.get_p_ex(mms)
    U=TrialFESpace(V,u_final); P=TrialFESpace(Q,p_final); X=MultiFieldFESpace([U,P])
    f_cf,g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms,model,Ω,dΩ,h_cf,X,Y,c_1,c_2,tau_reg_lim)
    s = config.numerical_method.solver
    spatial = (1.0/N)^(KV+1); dyn_ftol = max(s.ftol, min(s.dynamic_ftol_ceiling, s.dynamic_ftol_spatial_safety_factor*spatial))
    cs = Float64(N)^2*max(1.0,Float64(Re)); dnf = min(s.stagnation_noise_floor, max(s.condition_noise_floor_absolute_min, s.condition_noise_floor_baseline*cs)); dnf = max(dnf, dyn_ftol*s.condition_noise_floor_safety_factor)
    lpi = Re>=s.dynamic_picard_re_threshold ? max(s.picard_iterations,s.dynamic_picard_re_iterations) : s.picard_iterations
    lni = Re>=s.dynamic_newton_re_threshold ? max(s.newton_iterations,s.dynamic_newton_re_iterations) : s.newton_iterations
    nlp = PorousNSSolver.SafeNewtonSolver(LUSolver(),lpi,s.max_increases,s.xtol,dyn_ftol,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor; mode=:picard)
    nln = PorousNSSolver.SafeNewtonSolver(LUSolver(),lni,s.max_increases,s.xtol,dyn_ftol,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor)
    setup = PorousNSSolver.FETopology(X,Y,model,Ω,dΩ,V_free,Q_free,h_cf,f_cf,alpha_cf,g_cf)
    formulation = PorousNSSolver.VMSFormulation(form,c_1,c_2)
    iter_solvers = PorousNSSolver.StageSolvers(FESolver(nlp),FESolver(nln))
    x0_exact = interpolate_everywhere([u_final,p_final],X)
    nU=num_free_dofs(U); nP=num_free_dofs(P)
    return (; config,setup,formulation,iter_solvers,X,Y,nU,nP,x0_exact,u_final,U_c,dΩ)
end

erru(cell, xh) = sqrt(abs(sum(∫((cell.u_final - xh[1])⋅(cell.u_final - xh[1]))cell.dΩ)))/cell.U_c

# Reach Re_target at mesh N by continuation through solve_system with a RELAXED gate; return (final FEFunction, err_u, eps_M).
function continue_to(test_dict, Re_target, N; gate)
    ramp = [1.0,10.0,100.0,1000.0,1.0e4,3.0e4,Re_target]
    x0vals = nothing
    local cell
    for Re in ramp
        cell = build_cell(test_dict, Re, N; eps_tol_M=gate)
        start = x0vals === nothing ? cell.x0_exact : FEFunction(cell.X, copy(x0vals))
        succ,_,xf,it,tt = PorousNSSolver.solve_system(cell.setup, cell.formulation, cell.iter_solvers, cell.config, start)
        x0vals = copy(get_free_dof_values(xf))
    end
    xh = FEFunction(cell.X, x0vals)
    conv = PorousNSSolver.build_convergence_probe(cell.setup, cell.formulation, gate, cell.config.numerical_method.solver.eps_tol_mass)
    res(x,y) = PorousNSSolver.build_stabilized_weak_form_residual(x,y,cell.setup,cell.formulation,cell.config.physical_properties; pi_u=nothing,pi_p=nothing)
    b = assemble_vector(y->res(xh,y), cell.Y)
    cm = conv(x0vals, b, [1:cell.nU,(cell.nU+1):(cell.nU+cell.nP)])
    return xh, erru(cell,xh), cm.eps_M
end

test_dict = JSON3.read(read(joinpath(@__DIR__,"data","cocquet_form_mms_vms.json"),String), Dict)
println("="^80)
println("CONVERGENCE STUDY at Re=1e5, α=0.1, ASGS, k=1 (P1/P1), continuation + RELAXED gate 1e-4")
println("  N        h         err_u(L2,norm)   rate     ε_M_floor")
prev_e = NaN; prev_h = NaN
for N in (10, 20, 40, 80)
    xh, e, em = continue_to(test_dict, 1.0e5, N; gate=1e-4)
    h = 1.0/N
    rate = isnan(prev_e) ? NaN : log(prev_e/e)/log(prev_h/h)
    @printf("  %3d   %.5f    %.6e   %s   %.2e\n", N, h, e, isnan(rate) ? "  -- " : @sprintf("%.3f",rate), em)
    global prev_e = e; global prev_h = h
end
println("\n(rate≈2 ⟹ VMS is OPTIMAL at Re=1e5; the production NaN is purely the unreachable 1e-9 gate.)")
println("DONE.")
