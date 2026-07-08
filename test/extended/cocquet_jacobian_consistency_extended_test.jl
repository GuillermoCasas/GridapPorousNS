# ==============================================================================================
# Nature & Intent:
# Exact-Newton Jacobian consistency for the COCQUET formulation — SymmetricGradient viscous operator +
# velocity-dependent Forchheimer-Ergun reaction σ(α,u)=a(α)+b(α)|u| — assembled in ExactNewtonMode and
# checked against a centered finite-difference of the residual. In ASGS the orthogonal projection is
# π≡0, so there is NO dropped ∂π/∂u term: the Exact-Newton Jacobian MUST equal ∂R/∂U exactly (up to FD
# precision) if the code is correct. This pins that the velocity-dependent reaction is differentiated
# correctly everywhere it enters — the reaction term σ(u)u AND the stabilization scale τ₁(σ(u)) via
# ∂σ/∂u — for both k=1 and k=2.
#
# Coverage gap this closes (found 2026-07-07 while diagnosing the CocquetFormMMS α=0.1×Re=1e5 fold):
#   - picard_jacobian_equivalence_blitz_test.jl uses Forchheimer but only checks PICARD-mode equivalence
#     (two frozen builders agree); it never differentiates σ(u).
#   - osgs_frozen_pi_jacobian_quick_test.jl checks Exact-Newton J vs FD, but with ConstantSigma (no
#     ∂σ/∂u) — the velocity-dependent reaction's exact tangent was unguarded.
# The exact combination that governs the Cocquet corner (SymmetricGradient + Forchheimer, ExactNewton)
# was therefore untested. A defect here would break Newton's quadratic convergence; verified here it is
# exact ⇒ the CocquetFormMMS low-α "fold" is a genuine nonlinear property, not a linearization bug.
#
# Associated Files / Functions:
# - src/formulations/continuous_problem.jl (build_stabilized_weak_form_{residual,jacobian}, the ∂σ/∂u
#   product-rule terms via _get_dsigma_du_val / _get_dtau_1_du in ExactNewtonMode)
# - src/models/reaction.jl (ForchheimerErgunLaw sigma / dsigma_du)
# - docs/cocquet/cocquet-form-mms-status.md §4.1
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
using Gridap.Algebra
using LinearAlgebra
const _PNS = PorousNSSolver

# Cocquet ASGS problem (π≡0): equal-order kv on TRI, viscous operator `visc`, Forchheimer(a,b), α=0.1.
function _cocquet_asgs_problem(; n, kv, nu, a, b, visc)
    model = simplexify(CartesianDiscreteModel((0.0,1.0,0.0,1.0), (n,n)))   # TRI, like the harness
    Ω = Triangulation(model); dΩ = Measure(Ω, 4*kv + 2)
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kv)
    u_bc(x) = VectorValue(0.3*sin(pi*x[1])*cos(pi*x[2]), -0.3*cos(pi*x[1])*sin(pi*x[2]))
    V = TestFESpace(model, refe_u, conformity=:H1, dirichlet_tags="boundary")
    Q = TestFESpace(model, refe_p, conformity=:H1)
    U = TrialFESpace(V, u_bc); P = TrialFESpace(Q)
    X = MultiFieldFESpace([U,P]); Y = MultiFieldFESpace([V,Q])
    V_free = TestFESpace(model, refe_u, conformity=:H1); Q_free = TestFESpace(model, refe_p, conformity=:H1)
    U_proj = TrialFESpace(V_free); P_proj = TrialFESpace(Q_free)

    form = _PNS.PaperGeneralFormulation(visc, _PNS.ForchheimerErgunLaw(a, b),
              _PNS.ProjectFullResidual(), _PNS.SmoothVelocityFloor(1e-3, 0.0, 1e-10, 1e-12), nu, 0.0)
    c_1, c_2 = _PNS.get_c1_c2(_PNS.PaperGeneralFormulation, kv)
    alpha_cf = CellField(0.1, Ω); h_cf = CellField(1.0/n, Ω)
    f_cf = CellField(VectorValue(0.2, -0.1), Ω); g_cf = CellField(0.0, Ω)
    setup = _PNS.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    formulation = _PNS.VMSFormulation(form, c_1, c_2)
    phys_cfg = (tau_regularization_limit = 1e-8,)

    zu = FEFunction(U_proj, zeros(num_free_dofs(U_proj)))   # ASGS ⇒ π ≡ 0
    zp = FEFunction(P_proj, zeros(num_free_dofs(P_proj)))
    res_vec = (Ufe) -> assemble_vector(y -> _PNS.build_stabilized_weak_form_residual(Ufe, y, setup, formulation, phys_cfg; pi_u=zu, pi_p=zp), Y)
    jac_mat = (Ufe) -> assemble_matrix((dx, y) -> _PNS.build_stabilized_weak_form_jacobian(Ufe, dx, y, setup, formulation, phys_cfg, false, _PNS.ExactNewtonMode(); pi_u=zu, pi_p=zp), X, Y)
    return (; X, res_vec, jac_mat, ndof=num_free_dofs(X))
end

# Deterministic non-trivial iterate (no RNG — quick tests must be reproducible), |u|>0 so ∂σ/∂u is live.
_det_iterate(nd) = [0.05*sin(0.7*i) + 0.03*cos(0.31*i) for i in 1:nd]

# Relative error between the assembled Exact-Newton Jacobian and a centered-FD Jacobian of the residual.
function _jac_fd_relerr(pb; ϵ=1e-6)
    u0 = _det_iterate(pb.ndof); Ufe = FEFunction(pb.X, u0)
    J = Array(pb.jac_mat(Ufe))
    nd = pb.ndof; Jfd = zeros(nd, nd)
    for j in 1:nd
        up = copy(u0); up[j] += ϵ; um = copy(u0); um[j] -= ϵ
        Jfd[:, j] = (pb.res_vec(FEFunction(pb.X, up)) .- pb.res_vec(FEFunction(pb.X, um))) ./ (2ϵ)
    end
    return norm(J .- Jfd) / norm(Jfd)
end

@testset "extended: Exact-Newton Jacobian consistency — Cocquet (SymmetricGradient + Forchheimer)" begin
    # FD-limited rel err is ~1e-8 when the Jacobian is exact; a real ∂σ/∂u (or dτ/dL*) bug gives O(1e-2)+.
    # Threshold 1e-6 leaves ~2 orders of margin while still catching any genuine linearization defect.
    TOL = 1e-6
    SG = _PNS.SymmetricGradientViscosity()      # the Cocquet viscous operator
    DS = _PNS.DeviatoricSymmetricViscosity()     # canonical, cross-check

    @testset "SymmetricGradient + Forchheimer, ExactNewton, ASGS" begin
        for kv in (1, 2)
            e_mod = _jac_fd_relerr(_cocquet_asgs_problem(n=4, kv=kv, nu=1e-2, a=10.0, b=5.0,  visc=SG))
            e_cor = _jac_fd_relerr(_cocquet_asgs_problem(n=4, kv=kv, nu=1e-3, a=30.0, b=17.0, visc=SG))  # corner-like
            @test e_mod < TOL
            @test e_cor < TOL
        end
    end

    @testset "DeviatoricSymmetric + Forchheimer, ExactNewton (canonical cross-check)" begin
        for kv in (1, 2)
            e = _jac_fd_relerr(_cocquet_asgs_problem(n=4, kv=kv, nu=1e-3, a=30.0, b=17.0, visc=DS))
            @test e < TOL
        end
    end
end
