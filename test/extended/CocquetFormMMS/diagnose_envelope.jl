# [debugging-lore] Decide the fix fork: is the high-Re eps_M floor an ENVELOPE inconsistency (numerator
# carries tau-stab term, denominator D_M does not) that adding ||stab_mom|| to D_M would cure, OR a
# genuine residual/solver floor that envelope-repair cannot fix? At an ACCURATE high-Re iterate (reached
# by continuation + relaxed gate), measure: r_M (full vel residual), ||gal|| (mult_mom=0), ||stab_mom|| =
# ||full-gal||, D_M (current Galerkin envelope), and compare eps_M_current = r_M/D_M vs
# eps_M_envelope = r_M/(D_M+||stab_mom||). If eps_M_envelope << eps_M_current and -> ~0, the envelope fix
# (paper-faithful) works. If ||stab_mom|| << D_M so eps_M is ~unchanged, envelope repair does NOT help and
# the floor is in r_M itself (gate-relax or harness-accept needed). Also report err_u (solution quality).
# Run:  julia --project=../../.. diagnose_envelope.jl

include("run_test.jl")
using Printf, LinearAlgebra

const ALPHA0=0.1; const KV=1; const KP=1; const METHOD="ASGS"

function build_cell(test_dict, Re, N; eps_tol_M=nothing)
    pp_in = get(test_dict,"physical_properties",Dict()); nm_dict = get(test_dict,"numerical_method",Dict())
    sd = copy(get(nm_dict,"solver",Dict())); eps_tol_M===nothing||(sd["eps_tol_momentum"]=eps_tol_M)
    config_dict = Dict(
        "physical_properties"=>Dict("nu"=>1.0,"eps_val"=>1e-8,"reaction_model"=>get(pp_in,"reaction_model","Constant_Sigma"),
            "sigma_constant"=>get(pp_in,"sigma_constant",1.0),"sigma_linear"=>get(pp_in,"sigma_linear",0.0),"sigma_nonlinear"=>get(pp_in,"sigma_nonlinear",0.0)),
        "domain"=>Dict("alpha_0"=>0.4,"bounding_box"=>test_dict["domain"]["bounding_box"],"r_1"=>test_dict["domain"]["r_1"],"r_2"=>test_dict["domain"]["r_2"]),
        "numerical_method"=>Dict("viscous_operator_type"=>get(nm_dict,"viscous_operator_type","DeviatoricSymmetric"),
            "element_spaces"=>Dict("k_velocity"=>KV,"k_pressure"=>KP),"mesh"=>Dict("element_type"=>"TRI","partition"=>[N,N]),
            "stabilization"=>Dict("method"=>METHOD),"solver"=>sd))
    config = PorousNSSolver.load_config_from_dict(config_dict)
    model = _build_local_mesh(config.domain, config.numerical_method.mesh)
    labels=get_face_labeling(model); add_tag_from_tags!(labels,"all_boundaries",[1,2,3,4,5,6,7,8])
    refe_u=ReferenceFE(lagrangian,VectorValue{2,Float64},KV); refe_p=ReferenceFE(lagrangian,Float64,KP)
    V=TestFESpace(model,refe_u,conformity=:H1,labels=labels,dirichlet_tags=["all_boundaries"]); Q=TestFESpace(model,refe_p,conformity=:H1)
    V_free=TestFESpace(model,refe_u,conformity=:H1); Q_free=TestFESpace(model,refe_p,conformity=:H1)
    qrl = config.physical_properties.reaction_model=="Constant_Sigma" ? PorousNSSolver.ConstantSigmaLaw(0.0) : PorousNSSolver.ForchheimerErgunLaw(0.0,0.0)
    degree=PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation,KV,qrl)
    Ω=Triangulation(model); dΩ=Measure(Ω,degree+4)
    h_cf=CellField(collect(lazy_map(v->sqrt(2.0*abs(v)),get_cell_measure(Ω))),Ω)
    Y=MultiFieldFESpace([V,Q]); c_1,c_2=PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation,KV)
    trl=config.physical_properties.tau_regularization_limit
    af=build_porosity_field(config,ALPHA0,1.0); acf=CellField(x->PorousNSSolver.alpha(af,x),Ω)
    form=build_mms_formulation(config,1.0,Re,1.0,1.0,1.0)
    mms=PorousNSSolver.Paper2DMMS(form,1.0,af;L=1.0,alpha_infty=1.0)
    U_c,P_c=PorousNSSolver.get_characteristic_scales(mms); uf=PorousNSSolver.get_u_ex(mms); pf=PorousNSSolver.get_p_ex(mms)
    U=TrialFESpace(V,uf); P=TrialFESpace(Q,pf); X=MultiFieldFESpace([U,P])
    fcf,gcf=PorousNSSolver.evaluate_exactness_diagnostics(mms,model,Ω,dΩ,h_cf,X,Y,c_1,c_2,trl)
    s=config.numerical_method.solver
    spatial=(1.0/N)^(KV+1); dft=max(s.ftol,min(s.dynamic_ftol_ceiling,s.dynamic_ftol_spatial_safety_factor*spatial))
    cs=Float64(N)^2*max(1.0,Float64(Re)); dnf=min(s.stagnation_noise_floor,max(s.condition_noise_floor_absolute_min,s.condition_noise_floor_baseline*cs)); dnf=max(dnf,dft*s.condition_noise_floor_safety_factor)
    lpi=Re>=s.dynamic_picard_re_threshold ? max(s.picard_iterations,s.dynamic_picard_re_iterations) : s.picard_iterations
    lni=Re>=s.dynamic_newton_re_threshold ? max(s.newton_iterations,s.dynamic_newton_re_iterations) : s.newton_iterations
    nlp=PorousNSSolver.SafeNewtonSolver(LUSolver(),lpi,s.max_increases,s.xtol,dft,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor; mode=:picard)
    nln=PorousNSSolver.SafeNewtonSolver(LUSolver(),lni,s.max_increases,s.xtol,dft,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor)
    setup=PorousNSSolver.FETopology(X,Y,model,Ω,dΩ,V_free,Q_free,h_cf,fcf,acf,gcf)
    formulation=PorousNSSolver.VMSFormulation(form,c_1,c_2)
    its=PorousNSSolver.StageSolvers(FESolver(nlp),FESolver(nln))
    x0=interpolate_everywhere([uf,pf],X)
    return (; config,setup,formulation,iter_solvers=its,X,Y,nU=num_free_dofs(U),nP=num_free_dofs(P),x0,u_final=uf,U_c,dΩ)
end

function continue_to(test_dict, Re, N; gate)
    ramp=[1.0,10.0,100.0,1000.0,1.0e4,3.0e4,Re]; xv=nothing; local cell
    for r in ramp
        cell=build_cell(test_dict,r,N; eps_tol_M=gate)
        st = xv===nothing ? cell.x0 : FEFunction(cell.X,copy(xv))
        _,_,xf,_,_=PorousNSSolver.solve_system(cell.setup,cell.formulation,cell.iter_solvers,cell.config,st)
        xv=copy(get_free_dof_values(xf))
    end
    return cell, xv
end

function probe(test_dict, Re, N)
    cell, xv = continue_to(test_dict, Re, N; gate=1e-4)
    xh = FEFunction(cell.X, xv); s=cell.setup; fm=cell.formulation; phys=cell.config.physical_properties
    full(x,y)=PorousNSSolver.build_stabilized_weak_form_residual(x,y,s,fm,phys; pi_u=nothing,pi_p=nothing,mult_mom=1.0,mult_mass=1.0)
    gal(x,y) =PorousNSSolver.build_stabilized_weak_form_residual(x,y,s,fm,phys; pi_u=nothing,pi_p=nothing,mult_mom=0.0,mult_mass=0.0)
    bf=assemble_vector(y->full(xh,y),cell.Y); bg=assemble_vector(y->gal(xh,y),cell.Y)
    vb=1:cell.nU
    rM=norm(view(bf,vb)); galn=norm(view(bg,vb)); stabn=norm(view(bf,vb).-view(bg,vb))
    conv=PorousNSSolver.build_convergence_probe(s,fm,1e-9,cell.config.numerical_method.solver.eps_tol_mass)
    cm=conv(xv,bf,[1:cell.nU,(cell.nU+1):(cell.nU+cell.nP)])
    DM=cm.D_M
    erru=sqrt(abs(sum(∫((cell.u_final-xh[1])⋅(cell.u_final-xh[1]))cell.dΩ)))/cell.U_c
    @printf("Re=%.0e N=%-3d | r_M=%.3e ||gal||=%.3e ||stab_mom||=%.3e | D_M=%.3e\n", Re, N, rM, galn, stabn, DM)
    @printf("            eps_M(current=r_M/D_M)=%.3e   eps_M(envelope=r_M/(D_M+||stab||))=%.3e   ratio=%.2f\n", rM/DM, rM/(DM+stabn), (rM/DM)/(rM/(DM+stabn)))
    @printf("            ||stab_mom||/r_M=%.3f (=>fraction of residual that is the stab term)   err_u=%.3e\n", stabn/rM, erru)
end

test_dict=JSON3.read(read(joinpath(@__DIR__,"data","cocquet_form_mms_vms.json"),String),Dict)
println("="^90)
println("ENVELOPE TEST: does adding ||stab_mom|| to D_M restore eps_M -> 0 at an accurate high-Re iterate?")
println("="^90)
for N in (10,20,40); probe(test_dict, 1.0e5, N); end
println("\nKEY: if eps_M(envelope) << eps_M(current) AND ||stab_mom||/r_M ~ 1  => envelope fix works (paper-faithful).")
println("     if eps_M(envelope) ~ eps_M(current) (||stab_mom|| << D_M)     => envelope repair insufficient; r_M itself floors.")
println("DONE.")
