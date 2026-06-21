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
# =============================================================================

struct ILUGMRESSolver <: LinearSolver
    m::Int
    drop_tolerance::Float64
    rel_tol::Float64
    maxiter::Int
end

function ILUGMRESSolver(; m::Int=30, drop_tolerance::Float64=1e-4, rel_tol::Float64=1e-11, maxiter::Int=300)
    return ILUGMRESSolver(m, drop_tolerance, rel_tol, maxiter)
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
end

function Gridap.Algebra.numerical_setup(ss::ILUGMRESSymbolicSetup, mat::AbstractMatrix)
    # Build the ILU preconditioner for this matrix. If factorization fails
    # (e.g. a near-singular Jacobian), fall back to the identity `I` so GMRES
    # still runs unpreconditioned rather than aborting the whole solve.
    try
        ilu_cache = ilu(mat, τ=ss.solver.drop_tolerance)
        return ILUGMRESNumericalSetup(ss.solver, mat, ilu_cache)
    catch e
        println("[WARNING] ILU Factorization failed to construct optimally dynamically: ", e)
        return ILUGMRESNumericalSetup(ss.solver, mat, I) # default to Identity if failed
    end
end

# In-place refresh: reuse this setup for a new Jacobian (next Newton/Picard
# iterate) by swapping in `mat` and recomputing the ILU preconditioner, again
# falling back to the identity `I` if factorization fails.
function Gridap.Algebra.numerical_setup!(ns::ILUGMRESNumericalSetup, mat::AbstractMatrix)
    ns.mat = mat
    try
        ns.ilu_cache = ilu(mat, τ=ns.solver.drop_tolerance)
    catch e
        println("[WARNING] ILU Factorization update failed: ", e)
        ns.ilu_cache = I
    end
end



function Gridap.Algebra.solve!(x::AbstractVector, ns::ILUGMRESNumericalSetup, b::AbstractVector)
    # Solve `mat x = b` with restarted GMRES, using the ILU factor as a left
    # preconditioner (`Pl`). GMRES restarts every `m` Krylov vectors and runs up
    # to `maxiter` iterations or until the relative residual drops below `rel_tol`.
    gmres!(x, ns.mat, b; reltol=ns.solver.rel_tol, maxiter=ns.solver.maxiter, restart=ns.solver.m, Pl=ns.ilu_cache, log=false)
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
                              rel_tol=lsc.gmres_rel_tol, maxiter=lsc.gmres_maxiter)
    else
        error("Unknown linear_solver.method \"$(lsc.method)\" (expected \"LU\" or \"ILU_GMRES\")")
    end
end


