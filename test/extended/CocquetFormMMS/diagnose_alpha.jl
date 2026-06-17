# [debugging-lore] Ablation ruled out viscous-op + reaction model (all 4 identical ε_M=0.029). Remaining
# suspect: the porosity (low α / strong bump) — the paper SKIPS the high-Re×low-α corner (centered_encoding
# .tex). Sweep α at fixed Re=1e5, SymmetricGradient + Forchheimer, encoding ON, COLD solve from u_exact.
# Each α has its own Da=σ_lin·((1-α)/α)²+σ_nl·((1-α)/α) ⇒ its own (L,U). Whether convergence degrades as
# α→0 isolates the porosity gradient as the cause.  Run:  julia --project=../../.. diagnose_alpha.jl

include("run_test.jl")
using Printf, LinearAlgebra
const NMESH=10; const KV=1; const KP=1; const RE=1.0e5

function build_mesh_L(bbox, partition, L)
    m=CartesianDiscreteModel(Tuple(L .* bbox),Tuple(partition);isperiodic=(false,false),map=identity)
    m=simplexify(m); lb=get_face_labeling(m); add_tag_from_tags!(lb,"all_boundaries",[1,2,3,4,5,6,7,8]); return m
end

function build_cell(test_dict, alpha_0, U_amp, L)
    pp_in=get(test_dict,"physical_properties",Dict())
    config_dict=Dict(
        "physical_properties"=>Dict("nu"=>1.0,"eps_val"=>1e-8,"reaction_model"=>"Forchheimer_Ergun",
            "sigma_linear"=>get(pp_in,"sigma_linear",0.3),"sigma_nonlinear"=>get(pp_in,"sigma_nonlinear",1.75)),
        "domain"=>Dict("alpha_0"=>0.4,"bounding_box"=>test_dict["domain"]["bounding_box"],"r_1"=>test_dict["domain"]["r_1"],"r_2"=>test_dict["domain"]["r_2"]),
        "numerical_method"=>Dict("viscous_operator_type"=>"SymmetricGradient",
            "element_spaces"=>Dict("k_velocity"=>KV,"k_pressure"=>KP),"mesh"=>Dict("element_type"=>"TRI","partition"=>[NMESH,NMESH]),
            "stabilization"=>Dict("method"=>"ASGS"),"solver"=>get(get(test_dict,"numerical_method",Dict()),"solver",Dict())))
    config=PorousNSSolver.load_config_from_dict(config_dict)
    model=build_mesh_L(config.domain.bounding_box,config.numerical_method.mesh.partition,L); labels=get_face_labeling(model)
    refe_u=ReferenceFE(lagrangian,VectorValue{2,Float64},KV); refe_p=ReferenceFE(lagrangian,Float64,KP)
    V=TestFESpace(model,refe_u,conformity=:H1,labels=labels,dirichlet_tags=["all_boundaries"]); Q=TestFESpace(model,refe_p,conformity=:H1)
    V_free=TestFESpace(model,refe_u,conformity=:H1); Q_free=TestFESpace(model,refe_p,conformity=:H1)
    degree=PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation,KV,PorousNSSolver.ForchheimerErgunLaw(0.0,0.0))
    Ω=Triangulation(model); dΩ=Measure(Ω,degree+4); h_cf=CellField(collect(lazy_map(v->sqrt(2.0*abs(v)),get_cell_measure(Ω))),Ω)
    Y=MultiFieldFESpace([V,Q]); c_1,c_2=PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation,KV); trl=config.physical_properties.tau_regularization_limit
    af=PorousNSSolver.SmoothRadialPorosity(alpha_0,1.0,L*config.domain.r_1,L*config.domain.r_2); acf=CellField(x->PorousNSSolver.alpha(af,x),Ω)
    form=build_mms_formulation(config,1.0,RE,U_amp,L,1.0)
    mms=PorousNSSolver.Paper2DMMS(form,U_amp,af;L=L,alpha_infty=1.0); U_c,P_c=PorousNSSolver.get_characteristic_scales(mms)
    uf=PorousNSSolver.get_u_ex(mms); pf=PorousNSSolver.get_p_ex(mms); U=TrialFESpace(V,uf); P=TrialFESpace(Q,pf); X=MultiFieldFESpace([U,P])
    fcf,gcf=PorousNSSolver.evaluate_exactness_diagnostics(mms,model,Ω,dΩ,h_cf,X,Y,c_1,c_2,trl)
    s=config.numerical_method.solver; spatial=(1.0/NMESH)^(KV+1); dft=max(s.ftol,min(s.dynamic_ftol_ceiling,s.dynamic_ftol_spatial_safety_factor*spatial))
    cs=Float64(NMESH)^2*max(1.0,RE); dnf=max(min(s.stagnation_noise_floor,max(s.condition_noise_floor_absolute_min,s.condition_noise_floor_baseline*cs)),dft*s.condition_noise_floor_safety_factor)
    lpi=RE>=s.dynamic_picard_re_threshold ? max(s.picard_iterations,s.dynamic_picard_re_iterations) : s.picard_iterations
    lni=RE>=s.dynamic_newton_re_threshold ? max(s.newton_iterations,s.dynamic_newton_re_iterations) : s.newton_iterations
    nlp=PorousNSSolver.SafeNewtonSolver(LUSolver(),lpi,s.max_increases,s.xtol,dft,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor;mode=:picard)
    nln=PorousNSSolver.SafeNewtonSolver(LUSolver(),lni,s.max_increases,s.xtol,dft,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor)
    setup=PorousNSSolver.FETopology(X,Y,model,Ω,dΩ,V_free,Q_free,h_cf,fcf,acf,gcf); formulation=PorousNSSolver.VMSFormulation(form,c_1,c_2)
    return (; config,setup,formulation,iter_solvers=PorousNSSolver.StageSolvers(FESolver(nlp),FESolver(nln)),X,Y,nU=num_free_dofs(U),nP=num_free_dofs(P),x0=interpolate_everywhere([uf,pf],X),u_final=uf,U_c,dΩ,nu=form.ν)
end

function trial(test_dict, alpha_0)
    sl=Float64(get(get(test_dict,"physical_properties",Dict()),"sigma_linear",0.3)); sn=Float64(get(get(test_dict,"physical_properties",Dict()),"sigma_nonlinear",1.75))
    r=(1-alpha_0)/alpha_0; Da=sl*r^2+sn*r; L=sqrt(max(Da,1e-12)); U=(RE^2/max(Da,1e-12))^0.25
    cell=build_cell(test_dict,alpha_0,U,L)
    res(x,y)=PorousNSSolver.build_stabilized_weak_form_residual(x,y,cell.setup,cell.formulation,cell.config.physical_properties;pi_u=nothing,pi_p=nothing)
    succ,_,xf,it,_=PorousNSSolver.solve_system(cell.setup,cell.formulation,cell.iter_solvers,cell.config,cell.x0)
    b=assemble_vector(y->res(xf,y),cell.Y); conv=PorousNSSolver.build_convergence_probe(cell.setup,cell.formulation,1e-9,cell.config.numerical_method.solver.eps_tol_mass)
    cm=conv(get_free_dof_values(xf),b,[1:cell.nU,(cell.nU+1):(cell.nU+cell.nP)])
    area=sum(∫(1.0)cell.dΩ); erru=sqrt(abs(sum(∫((cell.u_final-xf[1])⋅(cell.u_final-xf[1]))cell.dΩ)))/(cell.U_c*sqrt(abs(area)))
    @printf("  α=%.2f | Da=%.2f L=%.2f U=%.1f ν=%.2e | COLD success=%-5s iters=%-4s ε_M=%.2e err_u(inv)=%.3e\n",
            alpha_0, Da, L, U, cell.nu, string(succ), string(it), cm.eps_M, erru)
end

test_dict=JSON3.read(read(joinpath(@__DIR__,"data","cocquet_form_mms_vms.json"),String),Dict)
println("="^110); @printf("α-SWEEP @ Re=%.0e N=%d SymmetricGradient+Forchheimer, encoding ON, COLD from u_exact, gate 1e-9\n",RE,NMESH); println("="^110)
for a in (1.0, 0.5, 0.3, 0.1); trial(test_dict, a); end
println("\nKEY: if convergence degrades monotonically as α→0 ⇒ the porosity gradient (low-α) is the hard factor — the")
println("     same high-Re×low-α corner the paper skips. err_u(inv) small at the stall ⇒ solution still usable (fold).")
println("DONE.")
