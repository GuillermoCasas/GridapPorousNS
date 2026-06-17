# [debugging-lore] Does the α=0.1×Re=1e5 "fold" cell give a USABLE O(h²) solution across meshes? Sweep
# N∈{10,20,40,80} at the well-conditioned encoding (Da≈40), reaction-scaling ON. Use a RELAXED gate
# (eps_tol_momentum=0.1, above the ~0.029 floor) so solve_system ACCEPTS the achievable iterate and returns
# the real solution (not the rolled-back seed). Report encoding-invariant err_u + rate. rate≈2 ⇒ usable
# fold (record it); rate<2/erratic ⇒ genuinely unresolved (skip it).  Run: julia --project=../../.. diagnose_alpha_meshes.jl

include("run_test.jl")
using Printf, LinearAlgebra
const ALPHA0=0.1; const KV=1; const KP=1; const RE=1.0e5; const GATE=0.1

build_mesh_L(bbox,part,L) = (m=simplexify(CartesianDiscreteModel(Tuple(L .* bbox),Tuple(part);isperiodic=(false,false),map=identity)); add_tag_from_tags!(get_face_labeling(m),"all_boundaries",[1,2,3,4,5,6,7,8]); m)

function build_cell(test_dict, N, U_amp, L)
    pp_in=get(test_dict,"physical_properties",Dict()); sd=copy(get(get(test_dict,"numerical_method",Dict()),"solver",Dict())); sd["eps_tol_momentum"]=GATE
    config_dict=Dict(
        "physical_properties"=>Dict("nu"=>1.0,"eps_val"=>1e-8,"reaction_model"=>"Forchheimer_Ergun",
            "sigma_linear"=>get(pp_in,"sigma_linear",0.3),"sigma_nonlinear"=>get(pp_in,"sigma_nonlinear",1.75)),
        "domain"=>Dict("alpha_0"=>0.4,"bounding_box"=>test_dict["domain"]["bounding_box"],"r_1"=>test_dict["domain"]["r_1"],"r_2"=>test_dict["domain"]["r_2"]),
        "numerical_method"=>Dict("viscous_operator_type"=>"SymmetricGradient",
            "element_spaces"=>Dict("k_velocity"=>KV,"k_pressure"=>KP),"mesh"=>Dict("element_type"=>"TRI","partition"=>[N,N]),
            "stabilization"=>Dict("method"=>"ASGS"),"solver"=>sd))
    config=PorousNSSolver.load_config_from_dict(config_dict)
    model=build_mesh_L(config.domain.bounding_box,config.numerical_method.mesh.partition,L); labels=get_face_labeling(model)
    refe_u=ReferenceFE(lagrangian,VectorValue{2,Float64},KV); refe_p=ReferenceFE(lagrangian,Float64,KP)
    V=TestFESpace(model,refe_u,conformity=:H1,labels=labels,dirichlet_tags=["all_boundaries"]); Q=TestFESpace(model,refe_p,conformity=:H1)
    V_free=TestFESpace(model,refe_u,conformity=:H1); Q_free=TestFESpace(model,refe_p,conformity=:H1)
    degree=PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation,KV,PorousNSSolver.ForchheimerErgunLaw(0.0,0.0))
    Ω=Triangulation(model); dΩ=Measure(Ω,degree+4); h_cf=CellField(collect(lazy_map(v->sqrt(2.0*abs(v)),get_cell_measure(Ω))),Ω)
    Y=MultiFieldFESpace([V,Q]); c_1,c_2=PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation,KV); trl=config.physical_properties.tau_regularization_limit
    af=PorousNSSolver.SmoothRadialPorosity(ALPHA0,1.0,L*config.domain.r_1,L*config.domain.r_2); acf=CellField(x->PorousNSSolver.alpha(af,x),Ω)
    form=build_mms_formulation(config,1.0,RE,U_amp,L,1.0); mms=PorousNSSolver.Paper2DMMS(form,U_amp,af;L=L,alpha_infty=1.0)
    U_c,P_c=PorousNSSolver.get_characteristic_scales(mms); uf=PorousNSSolver.get_u_ex(mms); pf=PorousNSSolver.get_p_ex(mms)
    U=TrialFESpace(V,uf); P=TrialFESpace(Q,pf); X=MultiFieldFESpace([U,P]); fcf,gcf=PorousNSSolver.evaluate_exactness_diagnostics(mms,model,Ω,dΩ,h_cf,X,Y,c_1,c_2,trl)
    s=config.numerical_method.solver; spatial=(1.0/N)^(KV+1); dft=max(s.ftol,min(s.dynamic_ftol_ceiling,s.dynamic_ftol_spatial_safety_factor*spatial))
    cs=Float64(N)^2*max(1.0,RE); dnf=max(min(s.stagnation_noise_floor,max(s.condition_noise_floor_absolute_min,s.condition_noise_floor_baseline*cs)),dft*s.condition_noise_floor_safety_factor)
    lni=RE>=s.dynamic_newton_re_threshold ? max(s.newton_iterations,s.dynamic_newton_re_iterations) : s.newton_iterations
    lpi=RE>=s.dynamic_picard_re_threshold ? max(s.picard_iterations,s.dynamic_picard_re_iterations) : s.picard_iterations
    nlp=PorousNSSolver.SafeNewtonSolver(LUSolver(),lpi,s.max_increases,s.xtol,dft,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor;mode=:picard)
    nln=PorousNSSolver.SafeNewtonSolver(LUSolver(),lni,s.max_increases,s.xtol,dft,s.linesearch_alpha_min,s.armijo_c1,s.divergence_merit_factor,dnf,s.max_linesearch_iterations,s.linesearch_contraction_factor)
    setup=PorousNSSolver.FETopology(X,Y,model,Ω,dΩ,V_free,Q_free,h_cf,fcf,acf,gcf); formulation=PorousNSSolver.VMSFormulation(form,c_1,c_2)
    return (; config,setup,formulation,iter_solvers=PorousNSSolver.StageSolvers(FESolver(nlp),FESolver(nln)),X,Y,nU=num_free_dofs(U),nP=num_free_dofs(P),x0=interpolate_everywhere([uf,pf],X),u_final=uf,p_final=pf,U_c,P_c,dΩ)
end

# Manual Exact-Newton + Armijo that TRACKS the best-residual iterate (so a fold returns the real best
# solution, not the rolled-back seed) and reports whether the tight 1e-9 gate is reached.
function newton_best(cell; maxit=80, ls_min=1e-8, c1=1e-4)
    s=cell.setup; fm=cell.formulation; phys=cell.config.physical_properties; fc=cell.config.numerical_method.solver.freeze_jacobian_cusp
    res(x,y)=PorousNSSolver.build_stabilized_weak_form_residual(x,y,s,fm,phys;pi_u=nothing,pi_p=nothing)
    jacN(x,dx,y)=PorousNSSolver.build_stabilized_weak_form_jacobian(x,dx,y,s,fm,phys,fc,PorousNSSolver.ExactNewtonMode();pi_u=nothing,pi_p=nothing)
    op=FEOperator(res,jacN,cell.X,cell.Y); conv=PorousNSSolver.build_convergence_probe(s,fm,1e-9,cell.config.numerical_method.solver.eps_tol_mass)
    fb=[1:cell.nU,(cell.nU+1):(cell.nU+cell.nP)]
    x=copy(get_free_dof_values(cell.x0)); best_x=copy(x); best_nb=Inf; best_eM=Inf; reached=false
    for it in 1:maxit
        xh=FEFunction(cell.X,x); b=residual(op,xh); nb=norm(b); cm=conv(x,b,fb)
        if nb<best_nb; best_nb=nb; best_x=copy(x); best_eM=cm.eps_M; end
        if !cm.degenerate && cm.converged; reached=true; best_x=copy(x); best_eM=cm.eps_M; break; end
        A=jacobian(op,xh); d=-(A\b); Φ0=0.5*nb^2; gd=dot(b,A*d); α=1.0; acc=false
        while α>=ls_min
            if 0.5*norm(residual(op,FEFunction(cell.X,x.+α.*d)))^2 <= Φ0+c1*α*gd; x=x.+α.*d; acc=true; break; end
            α*=0.5
        end
        acc || break
    end
    return FEFunction(cell.X,best_x), reached, best_eM
end

test_dict=JSON3.read(read(joinpath(@__DIR__,"data","cocquet_form_mms_vms.json"),String),Dict)
sl=Float64(get(get(test_dict,"physical_properties",Dict()),"sigma_linear",0.3)); sn=Float64(get(get(test_dict,"physical_properties",Dict()),"sigma_nonlinear",1.75))
r=(1-ALPHA0)/ALPHA0; Da=sl*r^2+sn*r; L=sqrt(Da); U=(RE^2/Da)^0.25
println("="^100); @printf("α=%.2f Re=%.0e MESH SWEEP (Da=%.1f L=%.2f U=%.1f; encoding+reaction-scaling; best-iterate Newton, gate 1e-9)\n",ALPHA0,RE,Da,L,U); println("="^100)
println("  N     h        err_u(L2,inv)   rate    err_p(L2,inv)   ε_M_floor   reached_gate?")
prev_e=NaN; prev_h=NaN
for N in (10,20,40,80)
    cell=build_cell(test_dict,N,U,L)
    xh, reached, eM = newton_best(cell)
    eu,ep,_,_=calculate_normalized_errors(xh[1],xh[2],cell.u_final,cell.p_final,cell.U_c,cell.P_c,L,cell.dΩ)
    h=1.0/N; rate=isnan(prev_e) ? NaN : log(prev_e/eu)/log(prev_h/h)
    @printf("  %-4d  %.5f  %.6e  %s  %.6e  %.2e   %s\n", N, h, eu, isnan(rate) ? "  --" : @sprintf("%.3f",rate), ep, eM, reached ? "YES ✅" : "no (fold)")
    global prev_e=eu; global prev_h=h
end
println("\nIf ε_M_floor DROPS toward 1e-9 as N grows (gate reached at fine N) ⇒ RECOVERS like the regular harness.")
println("If err_u rate→2 even while folding ⇒ usable solution (record). If both stuck ⇒ genuinely unresolved corner.")
println("DONE.")
