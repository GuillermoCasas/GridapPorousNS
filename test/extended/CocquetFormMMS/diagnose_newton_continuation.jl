# [debugging-lore] PROOF-OF-FIX: pure Exact-Newton + Re-continuation reaches Re=1e5 where the
# production cascade (which leans on Picard at high Re, producing NON-descent directions) stalls.
# Manual Exact-Newton with backtracking Armijo, warm-started across a geometric Re ramp.
# Run:  julia --project=../../.. diagnose_newton_continuation.jl

include("run_test.jl")
using Printf, LinearAlgebra

const ALPHA0=0.1; const NMESH=10; const KV=1; const KP=1; const METHOD="ASGS"

function build_cell(test_dict, Re)
    pp_in = get(test_dict,"physical_properties",Dict()); nm_dict = get(test_dict,"numerical_method",Dict())
    config_dict = Dict(
        "physical_properties"=>Dict("nu"=>1.0,"eps_val"=>1e-8,"reaction_model"=>get(pp_in,"reaction_model","Constant_Sigma"),
            "sigma_constant"=>get(pp_in,"sigma_constant",1.0),"sigma_linear"=>get(pp_in,"sigma_linear",0.0),"sigma_nonlinear"=>get(pp_in,"sigma_nonlinear",0.0)),
        "domain"=>Dict("alpha_0"=>0.4,"bounding_box"=>test_dict["domain"]["bounding_box"],"r_1"=>test_dict["domain"]["r_1"],"r_2"=>test_dict["domain"]["r_2"]),
        "numerical_method"=>Dict("viscous_operator_type"=>get(nm_dict,"viscous_operator_type","DeviatoricSymmetric"),
            "element_spaces"=>Dict("k_velocity"=>KV,"k_pressure"=>KP),"mesh"=>Dict("element_type"=>"TRI","partition"=>[NMESH,NMESH]),
            "stabilization"=>Dict("method"=>METHOD),"solver"=>get(nm_dict,"solver",Dict())))
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
    setup = PorousNSSolver.FETopology(X,Y,model,Ω,dΩ,V_free,Q_free,h_cf,f_cf,alpha_cf,g_cf)
    formulation = PorousNSSolver.VMSFormulation(form,c_1,c_2)
    x0_exact = interpolate_everywhere([u_final,p_final],X)
    nU=num_free_dofs(U); nP=num_free_dofs(P)
    return (; config,setup,formulation,X,Y,nU,nP,x0_exact,u_final,U_c,dΩ)
end

# Pure Exact-Newton with backtracking Armijo on Φ=½‖R‖². Converges on scale-free ε_M ≤ tol_M.
function newton_solve(cell, x0; maxit=40, ls_min=1e-6, c1=1e-4)
    s = cell.setup; fm = cell.formulation; phys = cell.config.physical_properties; fc = cell.config.numerical_method.solver.freeze_jacobian_cusp
    res(x,y) = PorousNSSolver.build_stabilized_weak_form_residual(x,y,s,fm,phys; pi_u=nothing,pi_p=nothing)
    jacN(x,dx,y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x,dx,y,s,fm,phys,fc,PorousNSSolver.ExactNewtonMode(); pi_u=nothing,pi_p=nothing)
    op = FEOperator(res,jacN,cell.X,cell.Y)
    conv = PorousNSSolver.build_convergence_probe(s, fm, cell.config.numerical_method.solver.eps_tol_momentum, cell.config.numerical_method.solver.eps_tol_mass)
    fb = [1:cell.nU, (cell.nU+1):(cell.nU+cell.nP)]
    x = copy(x0)
    for it in 1:maxit
        xh = FEFunction(cell.X,x); b = residual(op,xh); nb = norm(b)
        cm = conv(x, b, fb)
        if !cm.degenerate && cm.converged
            return (true, x, it, cm.eps_M, nb)
        end
        A = jacobian(op,xh); d = -(A \ b)
        Φ0 = 0.5*nb^2; gd = dot(b, A*d)   # = -‖b‖²  (true directional deriv of Φ)
        α = 1.0; accepted = false
        while α >= ls_min
            bt = residual(op, FEFunction(cell.X, x .+ α.*d))
            if 0.5*norm(bt)^2 <= Φ0 + c1*α*gd
                x = x .+ α.*d; accepted = true; break
            end
            α *= 0.5
        end
        accepted || return (false, x, it, conv(x,residual(op,FEFunction(cell.X,x)),fb).eps_M, nb)
    end
    bb = residual(op,FEFunction(cell.X,x))
    return (false, x, maxit, conv(x,bb,fb).eps_M, norm(bb))
end

test_dict = JSON3.read(read(joinpath(@__DIR__,"data","cocquet_form_mms_vms.json"),String), Dict)
ramp = [1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0,500.0,1000.0,2000.0,5000.0,1.0e4,2.0e4,5.0e4,1.0e5]

println("="^80)
println("A) PURE EXACT-NEWTON, COLD at Re=1e5 (no continuation):")
cell = build_cell(test_dict, 1.0e5)
s,xv,it,em,nb = newton_solve(cell, copy(get_free_dof_values(cell.x0_exact)))
@printf("   cold Newton Re=1e5: success=%s iters=%d ε_M=%.3e ‖R‖=%.3e\n", s, it, em, nb)

println("\nB) PURE EXACT-NEWTON + Re-CONTINUATION (warm-started):")
x0 = nothing
allok = true
for Re in ramp
    cell = build_cell(test_dict, Re)
    start = x0 === nothing ? copy(get_free_dof_values(cell.x0_exact)) : x0
    s,xv,it,em,nb = newton_solve(cell, start)
    erru = sqrt(abs(sum(∫((cell.u_final - FEFunction(cell.X,xv)[1])⋅(cell.u_final - FEFunction(cell.X,xv)[1]))cell.dΩ)))/cell.U_c
    @printf("   Re=%9.1f: success=%s iters=%2d ε_M=%.2e ‖R‖=%.2e err_u=%.3e\n", Re, s, it, em, nb, erru)
    if s; global x0 = copy(xv); else; global allok=false; println("   <== NEWTON+CONTINUATION BROKE HERE"); break; end
end
println(allok ? "\nRESULT: Exact-Newton + continuation REACHED Re=1e5. Fix direction confirmed." : "\nRESULT: broke before 1e5 — continuation alone insufficient even with Newton.")
println("DONE.")
