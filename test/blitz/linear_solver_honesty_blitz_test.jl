# ==============================================================================================
# Nature & Intent:
# [C.1] Guards the ILU_GMRES "honesty contract": the iterative linear solver must NEVER hand back a
# step it did not actually compute to tolerance. A GMRES that does not reach rel_tol within maxiter, and
# an ILU factorization failure when the unpreconditioned fallback is disabled, must surface as TYPED
# exceptions (GMRESNotConvergedError / ILUFactorizationFailure) that the nonlinear cascade catches and
# rolls back on — so a non-converged inner solve can never be silently recorded as a converged iterate.
#
# Mathematical Formulation Alignment:
# Backend choice (LU vs ILU_GMRES) does not change the converged solution; this test only pins the
# FAILURE semantics + config plumbing of the iterative path, not the formulation.
#
# Associated Files / Functions:
# - `src/solvers/linear_solvers.jl` (`ILUGMRESSolver`, `_build_ilu_preconditioner`, `solve!`,
#   `GMRESNotConvergedError`, `ILUFactorizationFailure`, `instantiate_linear_solver`)
# - docs/formulation-audit-2026-06-24.md §C.1
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
using Gridap.Algebra
using LinearAlgebra
using SparseArrays

# A deterministic, moderately-coupled SPD-ish sparse system (no randomness — blitz must be reproducible).
function _coupled_sparse_system(n::Int)
    rows = Int[]; cols = Int[]; vals = Float64[]
    push_e!(i, j, v) = (push!(rows, i); push!(cols, j); push!(vals, v))
    for i in 1:n
        push_e!(i, i, 4.0)                                  # diagonal
        i < n      && push_e!(i, i + 1, 1.0)               # near off-diagonals
        i > 1      && push_e!(i, i - 1, 1.0)
        i <= n - 5 && push_e!(i, i + 5, 0.7)               # far couplings (slow a weak preconditioner)
        i > 5      && push_e!(i, i - 5, 0.7)
    end
    A = sparse(rows, cols, vals, n, n)
    b = collect(1.0:n)
    return A, b
end

@testset "fast: ILU_GMRES honesty contract [C.1]" begin

    @testset "constructor requires the fallback policy (no silent default)" begin
        # Missing `allow_unpreconditioned_fallback` ⇒ a loud UndefKeywordError, never a backfilled default.
        @test_throws UndefKeywordError ILUGMRESSolver(m=10, drop_tolerance=1e-4, rel_tol=1e-8, maxiter=50)
        s = ILUGMRESSolver(m=10, drop_tolerance=1e-4, rel_tol=1e-8, maxiter=50,
                           allow_unpreconditioned_fallback=true)
        @test s.allow_unpreconditioned_fallback == true
    end

    @testset "instantiate_linear_solver threads the fallback policy through" begin
        lsc = LinearSolverConfig(method="ILU_GMRES", ilu_drop_tolerance=1e-4, gmres_restart=20,
                                 gmres_rel_tol=1e-10, gmres_maxiter=100,
                                 allow_unpreconditioned_fallback=true)
        solver = instantiate_linear_solver(lsc)
        @test solver isa ILUGMRESSolver
        @test solver.allow_unpreconditioned_fallback == true
        # LU ignores the field but the config still carries it (required even for LU).
        lsc_lu = LinearSolverConfig(method="LU", ilu_drop_tolerance=1e-4, gmres_restart=20,
                                    gmres_rel_tol=1e-10, gmres_maxiter=100,
                                    allow_unpreconditioned_fallback=false)
        @test instantiate_linear_solver(lsc_lu) isa LUSolver
    end

    @testset "a CONVERGED solve returns the step (no exception)" begin
        A, b = _coupled_sparse_system(40)
        solver = ILUGMRESSolver(m=40, drop_tolerance=1e-6, rel_tol=1e-8, maxiter=300,
                                allow_unpreconditioned_fallback=false)
        ns = numerical_setup(symbolic_setup(solver, A), A)
        x = zeros(length(b))
        solve!(x, ns, b)
        @test norm(A * x - b) / norm(b) < 1e-6   # the returned step really solves the system
    end

    @testset "a NON-CONVERGED GMRES throws (never returned as exact)" begin
        A, b = _coupled_sparse_system(40)
        # drop_tolerance huge ⇒ ILU keeps ~only the diagonal (weak); maxiter=1, m=1 ⇒ one Krylov step,
        # rel_tol=1e-12 ⇒ unreachable in one step on a 40-dim coupled system. Guaranteed non-convergence.
        solver = ILUGMRESSolver(m=1, drop_tolerance=100.0, rel_tol=1e-12, maxiter=1,
                                allow_unpreconditioned_fallback=false)
        ns = numerical_setup(symbolic_setup(solver, A), A)
        x = zeros(length(b))
        err = nothing
        try
            solve!(x, ns, b)
        catch e
            err = e
        end
        @test err isa GMRESNotConvergedError
        @test err.iters <= err.maxiter
        @test err.rel_tol == 1e-12
        @test err.preconditioned == true        # an ILU factor (not the identity fallback) was in use
    end

    @testset "ILU-failure fallback is config-gated (default = fail loud)" begin
        # Force the factorization-failure branch deterministically by handing `ilu` an unsupported (dense)
        # matrix — `_build_ilu_preconditioner` catches ANY factorization exception, so the POLICY branch is
        # exercised identically to a genuine ILU breakdown.
        dense = [4.0 1.0; 1.0 4.0]
        s_no_fb = ILUGMRESSolver(m=10, drop_tolerance=1e-4, rel_tol=1e-8, maxiter=50,
                                 allow_unpreconditioned_fallback=false)
        @test_throws ILUFactorizationFailure PorousNSSolver._build_ilu_preconditioner(s_no_fb, dense)

        s_fb = ILUGMRESSolver(m=10, drop_tolerance=1e-4, rel_tol=1e-8, maxiter=50,
                              allow_unpreconditioned_fallback=true)
        pc, is_identity = PorousNSSolver._build_ilu_preconditioner(s_fb, dense)
        @test is_identity == true
        @test pc === I                            # the unpreconditioned (identity) fallback
    end

    @testset "the failure messages are unambiguous (no false 'crash'/'converged' reading)" begin
        m_nc = sprint(showerror, GMRESNotConvergedError(3.2e-4, 1e-11, 300, 300, true))
        @test occursin("did NOT converge", m_nc)
        @test occursin("REJECTED", m_nc)
        @test occursin("ILU-preconditioned", m_nc)
        @test occursin("UNPRECONDITIONED",
                       sprint(showerror, GMRESNotConvergedError(1.0, 1e-11, 1, 1, false)))

        m_fail = sprint(showerror, ILUFactorizationFailure(120, 1e-4, ErrorException("zero pivot")))
        @test occursin("factorization FAILED", m_fail)
        @test occursin("allow_unpreconditioned_fallback=false", m_fail)
    end
end
