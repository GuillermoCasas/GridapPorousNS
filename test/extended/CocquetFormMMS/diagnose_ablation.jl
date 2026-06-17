# [debugging-lore] Isolate the residual high-Re convergence failure AFTER the reaction-scaling fix.
# 2×2 ablation at the SAME well-conditioned encoding (Da≈40 ⇒ L≈6.32, U≈126, ν=σ≈8e-3): vary only
#   viscous operator  ∈ {SymmetricGradient (CocquetFormMMS), DeviatoricSymmetric (regular harness)}
#   reaction model    ∈ {Forchheimer_Ergun (nonlinear σ), Constant_Sigma (matched Da=40)}
# COLD solve_system from u_exact, tight gate. Whichever combos converge vs NaN tells us the culprit.
# Run:  julia --project=../../.. diagnose_ablation.jl

include("run_test.jl")
using Printf, LinearAlgebra

const ALPHA0=0.1; const NMESH=10; const KV=1; const KP=1; const RE=1.0e5

function build_mesh_L(bbox, partition, L)
    m = CartesianDiscreteModel(Tuple(L .* bbox), Tuple(partition); isperiodic=(false,false), map=identity)
    m = simplexify(m); labels = get_face_labeling(m); add_tag_from_tags!(labels,"all_boundaries",[1,2,3,4,5,6,7,8]); return m
end

function build_cell(test_dict, visc_type, reaction_model, Da, U_amp, L)
    pp_in=get(test_dict,"physical_properties",Dict())
    config_dict=Dict(
        "physical_properties"=>Dict("nu"=>1.0,"eps_val"=>1e-8,"reaction_model"=>reaction_model,
            "sigma_constant"=>get(pp_in,"sigma_constant",1.0),
            "sigma_linear"=>get(pp_in,"sigma_linear",0.3),"sigma_nonlinear"=>get(pp_in,"sigma_nonlinear",1.75)),
        "domain"=>Dict("alpha_0"=>0.4,"bounding_box"=>test_dict["domain"]["bounding_box"],"r_1"=>test_dict["domain"]["r_1"],"r_2"=>test_dict["domain"]["r_2"]),
        "numerical_method"=>Dict("viscous_operator_type"=>visc_type,
            "element_spaces"=>Dict("k_velocity"=>KV,"k_pressure"=>KP),"mesh"=>Dict("element_type"=>"TRI","partition"=>[NMESH,NMESH]),
            "stabilization"=>Dict("method"=>"ASGS"),"solver"=>get(get(test_dict,"numerical_method",Dict()),"solver",Dict())))
    config=PorousNSSolver.load_config_from_dict(config_dict)
    model=build_mesh_L(config.domain.bounding_box, config.numerical_method.mesh.partition, L)
    labels=get_face_labeling(model)
    refe_u=ReferenceFE(lagrangian,VectorValue{2,Float64},KV); refe_p=ReferenceFE(lagrangian,Float64,KP)
    V=TestFESpace(model,refe_u,conformity=:H1,labels=labels,dirichlet_tags=["all_boundaries"]); Q=TestFESpace(model,refe_p,conformity=:H1)
    V_free=TestFESpace(model,refe_u,conformity=:H1); Q_free=TestFESpace(model,refe_p,conformity=:H1)
    qrl=reaction_model=="Constant_Sigma" ? PorousNSSolver.ConstantSigmaLaw(0.0) : PorousNSSolver.ForchheimerErgunLaw(0.0,0.0)
    degree=PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation,KV,qrl)
    Ω=Triangulation(model); dΩ=Measure(Ω,degree+4)
    h_cf=CellField(collect(lazy_map(v->sqrt(2.0*abs(v)),get_cell_measure(Ω))),Ω)
    Y=MultiFieldFESpace([V,Q]); c_1,c_2=PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation,KV)
    trl=config.physical_properties.tau_regularization_limit
    af=PorousNSSolver.SmoothRadialPorosity(ALPHA0,1.0,L*config.domain.r_1,L*config.domain.r_2); acf=CellField(x->PorousNSSolver.alpha(af,x),Ω)
    form=build_mms_formulation(config,Da,RE,U_amp,L,1.0)
    mms=PorousNSSolver.Paper2DMMS(form,U_amp,af;L=L,alpha_infty=1.0)
    U_c,P_c=PorousNSSolver.get_characteristic_scales(mms); uf=PorousNSSolver.get_u_ex(mms); pf=PorousNSSolver.get_p_ex(mms)
    U=TrialFESpace(V,uf); P=TrialFESpace(Q,pf); X=MultiFieldFESpace([U,P])
    fcf,gcf=PorousNSSolver.evaluate_exactness_diagnostics(mms,model,Ω,dΩ,h_cf,X,Y,c_1,c_2,trl)
    s=config.numerical_method.solver
    spatial=(1.0/NMESH)^(KV+1); dft=max(s.ftol,min(s.dynamic_ftol_ceiling,s.dynamic_ftol_spatial_safety_factor*spatial))
    cs=Float64(NMESH)^2*max(1.0,RE); dnf=min(s.stagnation_noise_floor,max(s.condition_noise_floor_absolute_min,s.condition_noise_floor_baseline*cs)); dnf=max(dnf,dft*s.condition_noise_floor_safety_factor)
    lpi=RE>=s.dynamic_picard_re_threshold ? max(s.picard_iterations,s.dynamic_picard_re_iterations) : s.picard_iterations
    lni=RE>=s.dynamic_newton_re_threshold ? max(s.newton_iterations,s.dynamic_newton_re_iterations) : s.newton_iterations
    nlp=PorousNSSolver.SafeNewtonSolver(LUSolver(),lpi,s.max_increases,s.xtol,dft,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor; mode=:picard)
    nln=PorousNSSolver.SafeNewtonSolver(LUSolver(),lni,s.max_increases,s.xtol,dft,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor)
    setup=PorousNSSolver.FETopology(X,Y,model,Ω,dΩ,V_free,Q_free,h_cf,fcf,acf,gcf)
    formulation=PorousNSSolver.VMSFormulation(form,c_1,c_2)
    its=PorousNSSolver.StageSolvers(FESolver(nlp),FESolver(nln))
    return (; config,setup,formulation,iter_solvers=its,X,Y,nU=num_free_dofs(U),nP=num_free_dofs(P),x0=interpolate_everywhere([uf,pf],X),u_final=uf,U_c,dΩ,nu=form.ν)
end

function trial(test_dict, visc, reaction, Da, U, L)
    cell=build_cell(test_dict, visc, reaction, Da, U, L)
    res(x,y)=PorousNSSolver.build_stabilized_weak_form_residual(x,y,cell.setup,cell.formulation,cell.config.physical_properties; pi_u=nothing,pi_p=nothing)
    succ,_,xf,it,_=PorousNSSolver.solve_system(cell.setup,cell.formulation,cell.iter_solvers,cell.config,cell.x0)
    b=assemble_vector(y->res(xf,y),cell.Y)
    conv=PorousNSSolver.build_convergence_probe(cell.setup,cell.formulation,cell.config.numerical_method.solver.eps_tol_momentum,cell.config.numerical_method.solver.eps_tol_mass)
    cm=conv(get_free_dof_values(xf),b,[1:cell.nU,(cell.nU+1):(cell.nU+cell.nP)])
    erru=sqrt(abs(sum(∫((cell.u_final-xf[1])⋅(cell.u_final-xf[1]))cell.dΩ)))/cell.U_c
    @printf("  visc=%-18s reaction=%-16s | ν=%.2e | COLD success=%-5s iters=%-4s ε_M=%.2e err_u=%.3e\n",
            visc, reaction, cell.nu, string(succ), string(it), cm.eps_M, erru)
end

test_dict=JSON3.read(read(joinpath(@__DIR__,"data","cocquet_form_mms_vms.json"),String),Dict)
Da=40.05; L=sqrt(Da); U=(RE^2/Da)^0.25
println("="^110)
@printf("ABLATION @ Re=%.0e α=%.2f N=%d ASGS  (encoding: Da=%.1f ⇒ L=%.2f U=%.1f, COLD solve from u_exact, gate 1e-9)\n", RE, ALPHA0, NMESH, Da, L, U)
println("="^110)
trial(test_dict, "SymmetricGradient",   "Forchheimer_Ergun", Da, U, L)   # current CocquetFormMMS
trial(test_dict, "DeviatoricSymmetric", "Forchheimer_Ergun", Da, U, L)   # swap viscous op only
trial(test_dict, "SymmetricGradient",   "Constant_Sigma",    Da, U, L)   # swap reaction only
trial(test_dict, "DeviatoricSymmetric", "Constant_Sigma",    Da, U, L)   # regular-harness-like
println("\nKEY: compare which (viscous, reaction) combos converge. If only Forchheimer fails ⇒ nonlinear reaction.")
println("     If only SymmetricGradient fails ⇒ viscous operator. If all fail ⇒ deeper (porosity bump / regime).")
println("DONE.")
