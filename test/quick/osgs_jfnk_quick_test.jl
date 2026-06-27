# ==============================================================================================
# Nature & Intent:
# [JFNK Phase-1] On-the-real-OSGS-operator checks for the Jacobian-Free Newton–Krylov inner solve (gate:
# docs/solver/jfnk-phase0-preconditioner-gate.md; theory §sec:jfnk). Three claims:
#   (1) The matrix-free action of `JFNKLinearSolver` reproduces the TRUE full coupled tangent (a
#       finite-difference of the re-projecting residual), i.e. it captures the dense ∂π/∂u coupling the
#       frozen-π tangent drops — so the mat-vec matches the FD-assembled full Jacobian and genuinely
#       DIFFERS from the frozen-π tangent.
#   (2) Driven through the PRODUCTION `JFNKLinearSolver.solve!` (matrix-free GMRES preconditioned by the
#       factored frozen-π Jacobian) inside a Newton loop, it converges quadratically to the OSGS root —
#       the same root as a reference full-Jacobian Newton (same residual ⇒ same root).
#   (3) [C.1] honesty: a deliberately starved inner GMRES (maxiter=1, tol=1e-14) raises
#       GMRESNotConvergedError rather than returning a step it did not compute to tolerance.
#
# Behavior-preservation when osgs_jfnk_enabled=false is covered by the unchanged default path
# (osgs_orthogonality_quick_test.jl runs OSGS with JFNK off); the wired-in solve_system→_osgs_jfnk_solve!
# path is exercised by test/extended (an MMS A/B), kept out of the quick tier for runtime.
#
# Associated Files / Functions:
# - `src/solvers/linear_solvers.jl` (`JFNKLinearSolver`, `JFNKMatVec`, `JFNKPrecond`, GMRESNotConvergedError)
# - `src/solvers/osgs_solver.jl` (`_osgs_jfnk_solve!`, `live_pi!`, projection helpers)
# ==============================================================================================
using Test
using PorousNSSolver
using Gridap
using Gridap.Algebra
using LinearAlgebra
const _PNS = PorousNSSolver

# Self-contained equal-order OSGS problem with a NON-zero ε_phys (the Phase-0 conditioning fix), plus the
# L²-projection machinery. Mirrors _a3_problem from osgs_frozen_pi_jacobian_quick_test.jl but threads eps_val.
function _jfnk_problem(; n, nu, sigma=1.0, eps_val=1e-6, amp=0.2, kv=1)
    model = CartesianDiscreteModel((0.0,1.0,0.0,1.0), (n,n))
    Ω = Triangulation(model); dΩ = Measure(Ω, 4*kv)
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv); refe_p = ReferenceFE(lagrangian, Float64, kv)
    u_bc(x) = VectorValue(amp*sin(pi*x[1])*cos(pi*x[2]), -amp*cos(pi*x[1])*sin(pi*x[2]))
    V = TestFESpace(model, refe_u, conformity=:H1, dirichlet_tags="boundary"); Q = TestFESpace(model, refe_p, conformity=:H1)
    U = TrialFESpace(V, u_bc); P = TrialFESpace(Q); X = MultiFieldFESpace([U,P]); Y = MultiFieldFESpace([V,Q])
    V_free = TestFESpace(model, refe_u, conformity=:H1); Q_free = TestFESpace(model, refe_p, conformity=:H1)
    U_proj = TrialFESpace(V_free); P_proj = TrialFESpace(Q_free)
    form = _PNS.PaperGeneralFormulation(_PNS.SymmetricGradientViscosity(), _PNS.ConstantSigmaLaw(sigma),
              _PNS.ProjectFullResidual(), _PNS.SmoothVelocityFloor(1e-3, 0.0, 1e-10, 1e-12), nu, eps_val)
    c_1, c_2 = _PNS.get_c1_c2(_PNS.PaperGeneralFormulation, kv)
    alpha_cf = CellField(0.7, Ω); h_cf = CellField(1.0/n, Ω); f_cf = CellField(VectorValue(0.1,-0.1), Ω); g_cf = CellField(0.0, Ω)
    setup = _PNS.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    formulation = _PNS.VMSFormulation(form, c_1, c_2); phys_cfg = (tau_regularization_limit = 1e-8,)
    M_u = assemble_matrix((u,v)->∫(v⋅u)dΩ, U_proj, V_free); M_p = assemble_matrix((p,q)->∫(q*p)dΩ, P_proj, Q_free)
    facu = numerical_setup(symbolic_setup(_PNS.CholeskySolver(), M_u), M_u); facp = numerical_setup(symbolic_setup(_PNS.CholeskySolver(), M_p), M_p)
    live_pi = (uh, ph) -> (
        _PNS.discrete_l2_projection(_PNS.inner_projection_u(uh, ph, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2), U_proj, V_free, dΩ, M_u, facu),
        _PNS.discrete_l2_projection(_PNS.inner_projection_p(uh, ph, form, dΩ, h_cf, alpha_cf, g_cf), P_proj, Q_free, dΩ, M_p, facp))
    res_vec = (Ufe, piu, pip) -> assemble_vector(y -> _PNS.build_stabilized_weak_form_residual(Ufe, y, setup, formulation, phys_cfg; pi_u=piu, pi_p=pip), Y)
    jac_mat = (Ufe, piu, pip) -> assemble_matrix((dx, y) -> _PNS.build_stabilized_weak_form_jacobian(Ufe, dx, y, setup, formulation, phys_cfg, false, _PNS.ExactNewtonMode(); pi_u=piu, pi_p=pip), X, Y)
    # F (recompute-π) and the frozen-π tangent, as functions of the free-DOF vector
    Fvec = (vec) -> (Ufe = FEFunction(X, vec); (pu, pp) = live_pi(Ufe[1], Ufe[2]); res_vec(Ufe, pu, pp))
    Jfrozen = (vec) -> (Ufe = FEFunction(X, vec); (pu, pp) = live_pi(Ufe[1], Ufe[2]); jac_mat(Ufe, pu, pp))
    return (; X, Y, ndof=num_free_dofs(X), Fvec, Jfrozen)
end

# Central-difference full Jacobian of the recompute-π residual (the true tangent, incl. ∂π/∂u).
function _full_fd_jacobian(pb, x; ϵ=1e-6)
    n = length(x); J = zeros(n, n)
    for j in 1:n
        xp = copy(x); xp[j] += ϵ; xm = copy(x); xm[j] -= ϵ
        J[:, j] = (pb.Fvec(xp) .- pb.Fvec(xm)) ./ (2ϵ)
    end
    return J
end

@testset "quick: OSGS JFNK on the real coupled operator" begin

    pb = _jfnk_problem(n=4, nu=0.05, sigma=1.0, eps_val=1e-6)
    # a deterministic, developed iterate (no RNG)
    x0 = [0.02*sin(0.6*i) + 0.01*cos(0.25*i) for i in 1:pb.ndof]

    @testset "matrix-free action == full FD tangent (captures ∂π/∂u), ≠ frozen-π tangent" begin
        F0 = pb.Fvec(x0)
        mv = _PNS.JFNKMatVec(pb.Fvec, x0, F0, 1e-7)
        Jfull = _full_fd_jacobian(pb, x0)
        Jfz   = Array(pb.Jfrozen(x0))
        drop_seen = false
        for s in 1:5
            v = [sin(0.5*i + 0.3*s) for i in 1:pb.ndof]; v ./= norm(v)
            y = similar(v); mul!(y, mv, v)
            # (1) matches the TRUE full tangent
            @test norm(y .- Jfull * v) / norm(Jfull * v) < 1e-4
            # (2) genuinely differs from the frozen-π tangent ⇒ it carries the dropped ∂π/∂u
            if norm(y .- Jfz * v) / norm(Jfull * v) > 1e-2
                drop_seen = true
            end
        end
        @test drop_seen   # the dropped coupling is non-trivial on at least one probe direction
    end

    @testset "JFNKLinearSolver.solve! drives Newton to the OSGS root (production code path)" begin
        # Reference: full-Jacobian Newton (ideal JFNK) to define the root.
        xref_root = copy(zeros(pb.ndof))
        for _ in 1:12
            F = pb.Fvec(xref_root); norm(F, Inf) < 1e-11 && break
            xref_root .-= _full_fd_jacobian(pb, xref_root) \ F
        end
        @test norm(pb.Fvec(xref_root), Inf) < 1e-9   # reference converged

        # Production path: the actual JFNKLinearSolver.solve! (matrix-free GMRES + frozen-π preconditioner),
        # threaded exactly as _osgs_jfnk_solve! does (xref set before each solve; A = frozen-π tangent).
        xr = Ref{Vector{Float64}}(copy(zeros(pb.ndof)))
        ls = _PNS.JFNKLinearSolver(LUSolver(), pb.Fvec, xr, 1e-2, 200, 200, 1e-8)
        x = zeros(pb.ndof)
        A0 = pb.Jfrozen(x); ss = symbolic_setup(ls, A0)
        hist = Float64[]
        for _ in 1:12
            F = pb.Fvec(x); push!(hist, norm(F, Inf)); norm(F, Inf) < 1e-11 && break
            xr[] = copy(x)                       # the FD base point (as the jac wrapper records it)
            ns = numerical_setup(ss, pb.Jfrozen(x))   # factor the frozen-π preconditioner at x
            dx = similar(F); solve!(dx, ns, F)        # matrix-free full-tangent GMRES
            x .-= dx                                  # Newton step x ← x − dx
        end
        @test hist[end] < 1e-9                         # converged
        @test length(hist) <= 6                        # in a handful of steps (≪ frozen-π's 60+)
        @test norm(x .- xref_root, Inf) < 1e-6         # SAME root as the full-Jacobian Newton
    end

    @testset "[C.1] starved inner GMRES raises GMRESNotConvergedError (no silent inexact step)" begin
        xr = Ref{Vector{Float64}}(copy(x0))
        # tol far tighter than 1 GMRES iter can reach ⇒ must NOT silently return a half-baked step
        ls = _PNS.JFNKLinearSolver(LUSolver(), pb.Fvec, xr, 1e-14, 1, 1, 1e-8)
        A = pb.Jfrozen(x0); ns = numerical_setup(symbolic_setup(ls, A), A)
        b = pb.Fvec(x0); dx = similar(b)
        @test_throws _PNS.GMRESNotConvergedError solve!(dx, ns, b)
    end
end
