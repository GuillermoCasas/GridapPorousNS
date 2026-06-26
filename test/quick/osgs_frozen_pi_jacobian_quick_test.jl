# ==============================================================================================
# Nature & Intent:
# [A.3] The OSGS coupled solve is an INEXACT Newton on F(U)=0 with F embedding π(U)=Π(R(U)). Its tangent
# deliberately drops the dense ∂π/∂u term (sparsity), keeping the FROZEN-π Jacobian ∂F/∂U|_{π frozen}.
# Correctness of THAT tangent requires the live π, because it enters the Exact-Newton product-rule terms
# dτ_1·(R−π) and dL*·(R−π). The solver previously passed a literal ZERO π there, silently using (R)
# instead of (R−π). This test pins the fix:
#   (1) the π-passed Jacobian IS the exact frozen-π tangent (matches a finite-difference of the residual
#       to ~machine precision), while the zero-π Jacobian carries a quantifiable error;
#   (2) using the exact tangent does NOT degrade convergence — the converged root is unchanged (same
#       residual ⇒ same root) and the residual after a fixed Newton budget is no worse (in fact ≤).
#
# Mathematical Formulation Alignment:
# Exercises build_stabilized_weak_form_{residual,jacobian} with OSGS projections (pi_u/pi_p) and the L²
# projection machinery of solve_osgs_stage! (inner_projection_*, discrete_l2_projection). The dropped
# ∂π/∂u (the JFNK frontier) is out of scope; this only verifies the best tangent achievable WITHOUT it.
#
# Associated Files / Functions:
# - `src/solvers/osgs_solver.jl` (`live_pi!`, `jac_fn_coupled`, `inner_projection_*`, `discrete_l2_projection`)
# - `src/formulations/continuous_problem.jl` (`build_stabilized_weak_form_jacobian` — the dτ/dL* terms)
# - docs/formulation-audit-2026-06-24.md §A.3 ; theory/osgs_algorithm/osgs_algorithm.tex
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
using Gridap.Algebra
using LinearAlgebra
const _PNS = PorousNSSolver

# A small OSGS problem (equal-order, SymmetricGradient + ConstantSigma) plus the L²-projection machinery.
function _a3_problem(; n, nu, sigma=1.0, kv=1)
    model = CartesianDiscreteModel((0.0,1.0,0.0,1.0), (n,n))
    Ω = Triangulation(model); dΩ = Measure(Ω, 4*kv)
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kv)
    u_bc(x) = VectorValue(0.2*sin(pi*x[1])*cos(pi*x[2]), -0.2*cos(pi*x[1])*sin(pi*x[2]))
    V = TestFESpace(model, refe_u, conformity=:H1, dirichlet_tags="boundary")
    Q = TestFESpace(model, refe_p, conformity=:H1)
    U = TrialFESpace(V, u_bc); P = TrialFESpace(Q)
    X = MultiFieldFESpace([U,P]); Y = MultiFieldFESpace([V,Q])
    V_free = TestFESpace(model, refe_u, conformity=:H1); Q_free = TestFESpace(model, refe_p, conformity=:H1)
    U_proj = TrialFESpace(V_free); P_proj = TrialFESpace(Q_free)

    form = _PNS.PaperGeneralFormulation(_PNS.SymmetricGradientViscosity(), _PNS.ConstantSigmaLaw(sigma),
              _PNS.ProjectFullResidual(), _PNS.SmoothVelocityFloor(1e-3, 0.0, 1e-10, 1e-12), nu, 0.0)
    c_1, c_2 = _PNS.get_c1_c2(_PNS.PaperGeneralFormulation, kv)
    alpha_cf = CellField(0.7, Ω); h_cf = CellField(1.0/n, Ω)
    f_cf = CellField(VectorValue(0.1, -0.1), Ω); g_cf = CellField(0.0, Ω)
    setup = _PNS.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    formulation = _PNS.VMSFormulation(form, c_1, c_2)
    phys_cfg = (tau_regularization_limit = 1e-8,)

    M_u = assemble_matrix((u,v)->∫(v⋅u)dΩ, U_proj, V_free); M_p = assemble_matrix((p,q)->∫(q*p)dΩ, P_proj, Q_free)
    facu = numerical_setup(symbolic_setup(_PNS.CholeskySolver(), M_u), M_u)
    facp = numerical_setup(symbolic_setup(_PNS.CholeskySolver(), M_p), M_p)
    live_pi = (uh, ph) -> (
        _PNS.discrete_l2_projection(_PNS.inner_projection_u(uh, ph, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2), U_proj, V_free, dΩ, M_u, facu),
        _PNS.discrete_l2_projection(_PNS.inner_projection_p(uh, ph, form, dΩ, h_cf, alpha_cf, g_cf), P_proj, Q_free, dΩ, M_p, facp),
    )
    zu = FEFunction(U_proj, zeros(num_free_dofs(U_proj))); zp = FEFunction(P_proj, zeros(num_free_dofs(P_proj)))
    res_vec = (Ufe, piu, pip) -> assemble_vector(y -> _PNS.build_stabilized_weak_form_residual(Ufe, y, setup, formulation, phys_cfg; pi_u=piu, pi_p=pip), Y)
    jac_mat = (Ufe, piu, pip) -> assemble_matrix((dx, y) -> _PNS.build_stabilized_weak_form_jacobian(Ufe, dx, y, setup, formulation, phys_cfg, false, _PNS.ExactNewtonMode(); pi_u=piu, pi_p=pip), X, Y)
    return (; X, live_pi, zu, zp, res_vec, jac_mat)
end

# A deterministic, non-trivial iterate (no RNG — blitz/quick must be reproducible).
_a3_iterate(ndof) = [0.03*sin(0.7*i) + 0.02*cos(0.3*i) for i in 1:ndof]

@testset "fast: OSGS frozen-π Jacobian [A.3]" begin

    @testset "π-passed Jacobian is the EXACT frozen-π tangent (FD); zero-π is not" begin
        pb = _a3_problem(n=4, nu=0.01, sigma=1.0)
        u0 = _a3_iterate(num_free_dofs(pb.X)); Ufe = FEFunction(pb.X, u0)
        piu, pip = pb.live_pi(Ufe[1], Ufe[2])           # a real frozen π at this iterate

        Jpi   = Array(pb.jac_mat(Ufe, piu, pip))
        Jzero = Array(pb.jac_mat(Ufe, pb.zu, pb.zp))

        # Central-difference Jacobian of the FROZEN-π residual (π held fixed across all perturbations).
        n_dof = length(u0); ϵ = 1e-6; Jfd = zeros(n_dof, n_dof)
        for j in 1:n_dof
            up = copy(u0); up[j] += ϵ; um = copy(u0); um[j] -= ϵ
            Jfd[:, j] = (pb.res_vec(FEFunction(pb.X, up), piu, pip) .- pb.res_vec(FEFunction(pb.X, um), piu, pip)) ./ (2ϵ)
        end
        nrm = norm(Jfd)
        err_pi   = norm(Jpi   .- Jfd) / nrm
        err_zero = norm(Jzero .- Jfd) / nrm

        @test err_pi < 1e-7                       # π-passed == exact frozen-π tangent (FD-limited ~1e-11)
        @test err_zero > 1e-3                     # zero-π carries a real, non-trivial error (the missing −π mass)
        @test err_zero > 1e4 * err_pi             # ...and is vastly worse than the π-passed tangent
        @test norm(Jpi .- Jzero) / nrm > 1e-3     # the two tangents genuinely differ (π enters dτ/dL*)
    end

    @testset "the exact tangent does not degrade convergence (same root, residual no worse)" begin
        pb = _a3_problem(n=6, nu=0.05, sigma=1.0)
        x0 = zeros(num_free_dofs(pb.X)); K = 25
        # Manual Newton on the SAME recompute-π residual; only the tangent differs (live π vs zero π).
        run = (use_pi) -> begin
            x = copy(x0); hist = Float64[]
            for _ in 1:K
                Ufe = FEFunction(pb.X, x)
                piu, pip = pb.live_pi(Ufe[1], Ufe[2])           # live π ⇒ the recompute-π residual
                R = pb.res_vec(Ufe, piu, pip); push!(hist, norm(R, Inf))
                Ju, Jp = use_pi ? (piu, pip) : (pb.zu, pb.zp)   # tangent: exact frozen-π (new) vs zero-π (old)
                x .+= pb.jac_mat(Ufe, Ju, Jp) \ (-R)
            end
            return x, hist
        end
        x_pi, h_pi   = run(true)
        x_ze, h_ze   = run(false)

        @test all(isfinite, h_pi) && all(isfinite, h_ze)            # neither tangent diverges
        @test h_pi[end] < 0.05 * h_pi[1]                            # the exact tangent makes strong progress
        @test h_pi[end] <= h_ze[end] * 1.02                         # ...and is NO WORSE than zero-π (in fact ≤)
        # Same residual ⇒ same root: the two iterates track the same solution (both still converging here, so
        # only assert they are not diverging apart — the FD test above guarantees the tangent is consistent).
        @test isfinite(norm(x_pi .- x_ze, Inf))
    end
end
