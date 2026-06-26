# src/solvers/linear_solvers.jl

using Gridap
using Gridap.Algebra
using IterativeSolvers
using IncompleteLU
using LinearAlgebra
using SparseArrays

# =============================================================================
# CholeskySolver — direct SPD factorization for L² mass matrices.
#
# The L² mass matrices `M_u` and `M_p` that drive the OSGS sub-grid projections
# (osgs_solver.jl) are symmetric positive-definite, so they admit a Cholesky
# factorization `M = LLᵀ`. Exploiting SPD structure (rather than a general LU
# with pivoting) does half the work, stores a smaller factor, and keeps the
# operation structurally symmetric, avoiding machine-epsilon drift from an
# asymmetric pivot permutation.
#
# This solver plugs into Gridap's `LinearSolver` interface and dispatches on the
# matrix storage:
# - `cholesky(::Symmetric{Float64,SparseMatrixCSC})` → CHOLMOD (sparse SPD), or
# - `cholesky(::Symmetric{Float64,Matrix})`          → LAPACK POTRF (dense SPD).
# Both factor types solve `M x = b` by forward/back substitution against `L`.
# =============================================================================

struct CholeskySolver <: LinearSolver end

struct CholeskySymbolicSetup <: SymbolicSetup end

mutable struct CholeskyNumericalSetup{F} <: NumericalSetup
    factors::F
end

Gridap.Algebra.symbolic_setup(::CholeskySolver, mat::AbstractMatrix) = CholeskySymbolicSetup()

Gridap.Algebra.numerical_setup(::CholeskySymbolicSetup, mat::AbstractMatrix) =
    CholeskyNumericalSetup(cholesky(Symmetric(mat)))

function Gridap.Algebra.solve!(x::AbstractVector, ns::CholeskyNumericalSetup, b::AbstractVector)
    # Solve `M x = b` from the stored Cholesky factor. [debugging-lore]
    # CHOLMOD's `Factor` does not implement the 2-arg `ldiv!(A, b)` that
    # `LinearAlgebra.ldiv!(x, A, b)` would internally rely on, so we dispatch
    # through `\`, which CHOLMOD implements directly. This allocates one
    # intermediate vector per backsolve; acceptable because the dominant cost
    # of the OSGS coupled solve is the inner Newton solve, not the projection.
    x .= ns.factors \ b
    x
end

# =============================================================================
# ILUGMRESSolver — restarted GMRES preconditioned by an incomplete LU.
#
# An iterative `LinearSolver` for the (generally non-symmetric) Newton/Picard
# Jacobians of the monolithic (u, p) system. It builds a left preconditioner by
# incomplete LU factorization (`IncompleteLU.ilu`, drop tolerance `τ`) and feeds
# it to restarted GMRES (`IterativeSolvers.gmres!`). Fields:
#   m              — Krylov subspace size before restart (GMRES `restart`).
#   drop_tolerance — ILU drop tolerance `τ`: smaller ⇒ denser, stronger
#                    preconditioner; larger ⇒ cheaper, weaker.
#   rel_tol        — relative residual tolerance for GMRES convergence.
#   maxiter        — cap on GMRES iterations.
#   allow_unpreconditioned_fallback — policy for an ILU FACTORIZATION failure:
#                    `false` (the safe default) ⇒ raise `ILUFactorizationFailure`
#                    so the nonlinear cascade rolls back / falls to Picard;
#                    `true`  ⇒ run GMRES UNPRECONDITIONED (identity `Pl`) after a
#                    loud warning (the (u,p) saddle point rarely converges that way,
#                    so the step is usually rejected downstream regardless).
#
# [C.1] Honesty contract (docs/formulation-audit-2026-06-24.md §C.1): this solver
# NEVER returns a step it did not actually compute to tolerance. Two failure modes
# are surfaced as typed exceptions rather than swallowed:
#   * GMRES that does not reach `rel_tol` within `maxiter`  → `GMRESNotConvergedError`;
#   * ILU factorization failure with the fallback disabled  → `ILUFactorizationFailure`.
# `eval_linear_system_resolution!` (nonlinear.jl) catches both, marks the linear solve
# failed, and the cascade rolls back — so a non-converged inner solve can never be
# silently recorded as a converged iterate (the failure-mode the audit's B-section flags).
# =============================================================================

# Raised when the ILU preconditioner cannot be built AND `allow_unpreconditioned_fallback` is false.
# Carries the matrix size, the drop tolerance, and the underlying `ilu(...)` exception for the trace.
struct ILUFactorizationFailure <: Exception
    n::Int
    drop_tolerance::Float64
    cause::Any
end
function Base.showerror(io::IO, e::ILUFactorizationFailure)
    print(io, "ILU_GMRES: incomplete-LU factorization FAILED on a ", e.n, "×", e.n,
              " Jacobian (drop tolerance τ=", e.drop_tolerance,
              ") and allow_unpreconditioned_fallback=false, so the unpreconditioned fallback is ",
              "disabled by config. The linear solve is reported as FAILED (not solved). Underlying cause: ",
              e.cause)
end

# Raised when restarted GMRES does not reach `rel_tol` within `maxiter`: the returned step is
# inaccurate, so it must be rejected rather than handed back as if exact. `rel_resnorm` is the achieved
# ‖r_k‖/‖r_0‖, `preconditioned` says whether an ILU factor (true) or the identity fallback (false) was used.
struct GMRESNotConvergedError <: Exception
    rel_resnorm::Float64
    rel_tol::Float64
    iters::Int
    maxiter::Int
    preconditioned::Bool
end
function Base.showerror(io::IO, e::GMRESNotConvergedError)
    pc = e.preconditioned ? "ILU-preconditioned" : "UNPRECONDITIONED (identity fallback)"
    print(io, "ILU_GMRES: ", pc, " GMRES did NOT converge — achieved relative residual ",
              e.rel_resnorm, " > rel_tol=", e.rel_tol, " after ", e.iters, "/", e.maxiter,
              " iterations. The step is unreliable and is being REJECTED (not accepted as a solved iterate).")
end

struct ILUGMRESSolver <: LinearSolver
    m::Int
    drop_tolerance::Float64
    rel_tol::Float64
    maxiter::Int
    allow_unpreconditioned_fallback::Bool
end

# Keyword constructor with NO defaults (the kwargs are required): every control must be supplied
# explicitly from `LinearSolverConfig` via `instantiate_linear_solver` — no magic-number fallbacks
# (repo hard rule). `ILUGMRESSolver()` with no args is therefore a (loud) MethodError, by design.
function ILUGMRESSolver(; m::Int, drop_tolerance::Float64, rel_tol::Float64, maxiter::Int,
                          allow_unpreconditioned_fallback::Bool)
    return ILUGMRESSolver(m, drop_tolerance, rel_tol, maxiter, allow_unpreconditioned_fallback)
end

struct ILUGMRESSymbolicSetup <: SymbolicSetup
    solver::ILUGMRESSolver
end

function Gridap.Algebra.symbolic_setup(solver::ILUGMRESSolver, mat::AbstractMatrix)
    return ILUGMRESSymbolicSetup(solver)
end

mutable struct ILUGMRESNumericalSetup{T} <: NumericalSetup
    solver::ILUGMRESSolver
    mat::T
    ilu_cache
    preconditioner_is_identity::Bool   # true ⇒ ILU failed and the config allowed the identity fallback
end

# Build the ILU left preconditioner for `mat`, honoring the fallback policy. Returns
# `(preconditioner, is_identity)`. On factorization failure: raise `ILUFactorizationFailure`
# (the default, so the cascade rolls back) UNLESS `allow_unpreconditioned_fallback`, in which case
# substitute the identity and warn LOUDLY into the trace — unpreconditioned GMRES rarely converges on
# the (u,p) saddle point, so a `true` policy just gives it one honest attempt before the convergence
# check rejects it.
function _build_ilu_preconditioner(solver::ILUGMRESSolver, mat::AbstractMatrix)
    try
        return ilu(mat, τ=solver.drop_tolerance), false
    catch e
        if solver.allow_unpreconditioned_fallback
            println("  [LINEAR SOLVER][ILU_GMRES][WARNING] ILU factorization FAILED; falling back to ",
                    "UNPRECONDITIONED GMRES (allow_unpreconditioned_fallback=true). Convergence on the ",
                    "(u,p) saddle-point Jacobian is unlikely; the step will be REJECTED if GMRES does not ",
                    "reach rel_tol=", solver.rel_tol, ". Underlying cause: ", e)
            return I, true
        end
        throw(ILUFactorizationFailure(size(mat, 1), solver.drop_tolerance, e))
    end
end

function Gridap.Algebra.numerical_setup(ss::ILUGMRESSymbolicSetup, mat::AbstractMatrix)
    pc, is_identity = _build_ilu_preconditioner(ss.solver, mat)
    return ILUGMRESNumericalSetup(ss.solver, mat, pc, is_identity)
end

# In-place refresh: reuse this setup for a new Jacobian (next Newton/Picard iterate) by swapping in
# `mat` and recomputing the ILU preconditioner under the same fallback policy.
function Gridap.Algebra.numerical_setup!(ns::ILUGMRESNumericalSetup, mat::AbstractMatrix)
    ns.mat = mat
    pc, is_identity = _build_ilu_preconditioner(ns.solver, mat)
    ns.ilu_cache = pc
    ns.preconditioner_is_identity = is_identity
    return ns
end

# Last recorded relative residual ‖r_k‖/‖r_0‖ from an IterativeSolvers `ConvergenceHistory`
# (NaN if the history is absent/degenerate). Used only to annotate the non-convergence trace.
function _final_rel_resnorm(history)
    data = get(history.data, :resnorm, Float64[])
    (isempty(data) || iszero(first(data))) && return NaN
    return last(data) / first(data)
end

function Gridap.Algebra.solve!(x::AbstractVector, ns::ILUGMRESNumericalSetup, b::AbstractVector)
    # Solve `mat x = b` with restarted GMRES, using the ILU factor as a left preconditioner (`Pl`).
    # `log=true` returns the convergence history so we can VERIFY the iterative solve actually reached
    # `rel_tol` within `maxiter`. [C.1] A non-converged GMRES yields an inaccurate step; returning it as
    # if exact is precisely the silent-failure bug this guards against — so on non-convergence we throw
    # `GMRESNotConvergedError`, which `eval_linear_system_resolution!` turns into a rejected step.
    _, history = gmres!(x, ns.mat, b; reltol=ns.solver.rel_tol, maxiter=ns.solver.maxiter,
                        restart=ns.solver.m, Pl=ns.ilu_cache, log=true)
    if !history.isconverged
        throw(GMRESNotConvergedError(_final_rel_resnorm(history), ns.solver.rel_tol,
                                     history.iters, ns.solver.maxiter, !ns.preconditioner_is_identity))
    end
    return x
end


# =============================================================================
# instantiate_linear_solver — the single seam mapping a `LinearSolverConfig` to a concrete Gridap
# `LinearSolver` for the monolithic (u, p) Jacobian solves (production `run_simulation.jl` and the
# MMS harness). "LU" is the exact sparse direct solver (the previously-hardcoded backend); "ILU_GMRES"
# is the low-memory iterative path for large 3D systems whose LU fill-in would exhaust RAM. The backend
# choice does not change the converged solution. `validate!` already enforces the method enum; we guard
# again here so the factory is total on its own.
# =============================================================================
function instantiate_linear_solver(lsc::LinearSolverConfig)::LinearSolver
    if lsc.method == "LU"
        return LUSolver()
    elseif lsc.method == "ILU_GMRES"
        return ILUGMRESSolver(m=lsc.gmres_restart, drop_tolerance=lsc.ilu_drop_tolerance,
                              rel_tol=lsc.gmres_rel_tol, maxiter=lsc.gmres_maxiter,
                              allow_unpreconditioned_fallback=lsc.allow_unpreconditioned_fallback)
    else
        error("Unknown linear_solver.method \"$(lsc.method)\" (expected \"LU\" or \"ILU_GMRES\")")
    end
end


