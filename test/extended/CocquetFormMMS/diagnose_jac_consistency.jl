# [debugging-lore] DECISIVE TEST for the VMS high-Re stall: is the assembled Jacobian actually dR/dx?
# At high Re the line search depletes even next to the solution -> bad search direction. Either (a) the
# Exact-Newton J is inconsistent with R (velocity-floor / τ-reg makes R non-smooth, or a term is wrong),
# or (b) the merit landscape is non-smooth. This script, for Re ∈ {1, 1e4, 1e5}, α=0.1, N=10, ASGS:
#   - assembles R and BOTH Jacobians (ExactNewton, Picard) at u_exact AND at a stalled iterate;
#   - FINITE-DIFFERENCE checks J·du vs (R(x+εdu)-R(x-εdu))/2ε  (the consistency test);
#   - reports the Newton descent dot product (A'R)·(-A\R) (should be ≈ -‖R‖²);
#   - reports ε_M, ε_C, MMS error, and cond(J).
# Run:  julia --project=../../.. diagnose_jac_consistency.jl

include("run_test.jl")
using Printf, LinearAlgebra

const ALPHA0=0.1; const NMESH=10; const KV=1; const KP=1; const METHOD="ASGS"

function build_cell(test_dict, Re)
    pp_in = get(test_dict, "physical_properties", Dict()); nm_dict = get(test_dict, "numerical_method", Dict())
    config_dict = Dict(
        "physical_properties" => Dict("nu"=>1.0,"eps_val"=>1e-8,
            "reaction_model"=>get(pp_in,"reaction_model","Constant_Sigma"),"sigma_constant"=>get(pp_in,"sigma_constant",1.0),
            "sigma_linear"=>get(pp_in,"sigma_linear",0.0),"sigma_nonlinear"=>get(pp_in,"sigma_nonlinear",0.0)),
        "domain" => Dict("alpha_0"=>0.4,"bounding_box"=>test_dict["domain"]["bounding_box"],"r_1"=>test_dict["domain"]["r_1"],"r_2"=>test_dict["domain"]["r_2"]),
        "numerical_method" => Dict("viscous_operator_type"=>get(nm_dict,"viscous_operator_type","DeviatoricSymmetric"),
            "element_spaces"=>Dict("k_velocity"=>KV,"k_pressure"=>KP),"mesh"=>Dict("element_type"=>"TRI","partition"=>[NMESH,NMESH]),
            "stabilization"=>Dict("method"=>METHOD),"solver"=>get(nm_dict,"solver",Dict())))
    config = PorousNSSolver.load_config_from_dict(config_dict)
    model = _build_local_mesh(config.domain, config.numerical_method.mesh)
    labels = get_face_labeling(model); add_tag_from_tags!(labels,"all_boundaries",[1,2,3,4,5,6,7,8])
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, KV); refe_p = ReferenceFE(lagrangian, Float64, KP)
    V = TestFESpace(model,refe_u,conformity=:H1,labels=labels,dirichlet_tags=["all_boundaries"]); Q = TestFESpace(model,refe_p,conformity=:H1)
    V_free = TestFESpace(model,refe_u,conformity=:H1); Q_free = TestFESpace(model,refe_p,conformity=:H1)
    quad_rxn_law = config.physical_properties.reaction_model=="Constant_Sigma" ? PorousNSSolver.ConstantSigmaLaw(0.0) : PorousNSSolver.ForchheimerErgunLaw(0.0,0.0)
    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, KV, quad_rxn_law)
    Ω = Triangulation(model); dΩ = Measure(Ω, degree+4)
    h_cf = CellField(collect(lazy_map(v->sqrt(2.0*abs(v)), get_cell_measure(Ω))), Ω)
    Y = MultiFieldFESpace([V,Q]); c_1,c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, KV)
    tau_reg_lim = config.physical_properties.tau_regularization_limit
    alpha_field = build_porosity_field(config, ALPHA0, 1.0); alpha_cf = CellField(x->PorousNSSolver.alpha(alpha_field,x), Ω)
    form = build_mms_formulation(config, 1.0, Re, 1.0, 1.0, 1.0)
    mms = PorousNSSolver.Paper2DMMS(form, 1.0, alpha_field; L=1.0, alpha_infty=1.0)
    U_c,P_c = PorousNSSolver.get_characteristic_scales(mms); u_final = PorousNSSolver.get_u_ex(mms); p_final = PorousNSSolver.get_p_ex(mms)
    U = TrialFESpace(V,u_final); P = TrialFESpace(Q,p_final); X = MultiFieldFESpace([U,P])
    f_cf,g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)
    setup = PorousNSSolver.FETopology(X,Y,model,Ω,dΩ,V_free,Q_free,h_cf,f_cf,alpha_cf,g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
    x0_exact = interpolate_everywhere([u_final,p_final], X)
    nU = num_free_dofs(U); nP = num_free_dofs(P)
    return (; config, setup, formulation, X, Y, nU, nP, x0_exact, u_final, p_final, U_c, dΩ)
end

# residual / jacobian closures (ASGS: pi_u=pi_p=nothing)
function ops(cell)
    s = cell.setup; fm = cell.formulation; phys = cell.config.physical_properties; fc = cell.config.numerical_method.solver.freeze_jacobian_cusp
    res(x,y) = PorousNSSolver.build_stabilized_weak_form_residual(x,y,s,fm,phys; pi_u=nothing,pi_p=nothing)
    jacN(x,dx,y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x,dx,y,s,fm,phys,fc,PorousNSSolver.ExactNewtonMode(); pi_u=nothing,pi_p=nothing)
    jacP(x,dx,y) = PorousNSSolver.build_picard_jacobian(x,dx,y,s,fm,phys; pi_u=nothing,pi_p=nothing,mult_mom=1.0,mult_mass=1.0)
    return FEOperator(res,jacN,cell.X,cell.Y), FEOperator(res,jacP,cell.X,cell.Y)
end

# Manual Picard to reach a stalled iterate (frozen-coeff, fixed point); returns free-dof vec after k steps.
function picard_iterate(cell, opP, k)
    x = copy(get_free_dof_values(cell.x0_exact))
    for _ in 1:k
        xh = FEFunction(cell.X, x); b = residual(opP, xh); A = jacobian(opP, xh)
        x = x .- (A \ b)
    end
    return x
end

function analyze(cell, label, xvec)
    opN, opP = ops(cell)
    xh = FEFunction(cell.X, xvec)
    b  = residual(opN, xh)
    A  = jacobian(opN, xh)      # Exact-Newton J
    Ap = jacobian(opP, xh)      # Picard J
    # conv measure
    conv = PorousNSSolver.build_convergence_probe(cell.setup, cell.formulation, cell.config.numerical_method.solver.eps_tol_momentum, cell.config.numerical_method.solver.eps_tol_mass)
    cm = conv(xvec, b, [1:cell.nU, (cell.nU+1):(cell.nU+cell.nP)])
    erru = sqrt(abs(sum(∫((cell.u_final - xh[1])⋅(cell.u_final - xh[1]))cell.dΩ)))/cell.U_c
    # FD consistency of Exact-Newton J along the Newton direction d = -A\b
    d = -(A \ b); d ./= (norm(d)+eps())
    relerrs = Float64[]
    for ε in (1e-5, 1e-6, 1e-7)
        rp = residual(opN, FEFunction(cell.X, xvec .+ ε.*d)); rm = residual(opN, FEFunction(cell.X, xvec .- ε.*d))
        Jd_fd = (rp .- rm)./(2ε); Jd_an = A*d
        push!(relerrs, norm(Jd_an .- Jd_fd)/(norm(Jd_fd)+eps()))
    end
    # descent dot products (should be ≈ -‖b‖²): exact-newton dir and picard dir
    dN = -(A \ b); descN = dot(transpose(A)*b, dN)
    dP = -(Ap \ b); descP = dot(transpose(A)*b, dP)   # is the PICARD step a descent dir for the TRUE merit?
    cN = cond(Matrix(A)); cP = cond(Matrix(Ap))
    @printf("  [%s] ‖R‖₂=%.3e ε_M=%.3e ε_C=%.3e err_u=%.3e\n", label, norm(b), cm.eps_M, cm.eps_C, erru)
    @printf("      FD-consistency relerr(J_exact·d vs dR) @ε=1e-5,-6,-7: %.2e %.2e %.2e   (≈0 ⟺ J=dR/dx)\n", relerrs...)
    @printf("      descent: Newton (A'b)·dN=%.3e (want -‖b‖²=%.3e) | Picard (A'b)·dP=%.3e (neg⟺descent)\n", descN, -dot(b,b), descP)
    @printf("      cond(J_exact)=%.3e  cond(J_picard)=%.3e\n", cN, cP)
end

test_dict = JSON3.read(read(joinpath(@__DIR__,"data","cocquet_form_mms_vms.json"),String), Dict)
for Re in (1.0, 1.0e4, 1.0e5)
    println("\n", "="^80, "\nRe=$Re, α=$ALPHA0, N=$NMESH, k=$KV, ASGS");
    cell = build_cell(test_dict, Re)
    analyze(cell, "@u_exact", copy(get_free_dof_values(cell.x0_exact)))
    _, opP = ops(cell)
    xstall = picard_iterate(cell, opP, 12)
    analyze(cell, "@picard12", xstall)
end
println("\nDONE.")
