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
# Mathematically the L² mass matrices `M_u` and `M_p` used by the OSGS sub-grid
# projections (osgs_solver.jl) are symmetric positive-definite. The previous
# `LUSolver` path runs UMFPack with partial pivoting, which (a) does extra work
# vs Cholesky on SPD, (b) carries a larger factor and unnecessary permutation
# tree, and (c) is structurally non-symmetric — the asymmetric permutation can
# produce machine-epsilon-level drift on otherwise-symmetric operations.
#
# This solver dispatches to:
# - `cholesky(::Symmetric{Float64,SparseMatrixCSC})` → CHOLMOD (sparse SPD), or
# - `cholesky(::Symmetric{Float64,Matrix})` → LAPACK POTRF (dense SPD).
# Both expose `ldiv!(x, fac, b)` for backsubstitution.
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
    # CHOLMOD's `Factor` does not implement the 2-arg `ldiv!(A, b)` that
    # `LinearAlgebra.ldiv!(x, A, b)` would internally rely on, so dispatch
    # through `\` which CHOLMOD implements directly. This allocates one
    # intermediate vector per backsolve; acceptable because the dominant cost
    # of the OSGS coupled solve is the inner Newton solve, not the projection.
    x .= ns.factors \ b
    x
end

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
    # Perform Incomplete LU factorization natively
    try
        ilu_cache = ilu(mat, τ=ss.solver.drop_tolerance)
        return ILUGMRESNumericalSetup(ss.solver, mat, ilu_cache)
    catch e
        println("[WARNING] ILU Factorization failed to construct optimally dynamically: ", e)
        return ILUGMRESNumericalSetup(ss.solver, mat, I) # default to Identity if failed
    end
end

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
    # Execute native GMRES mapped securely through left ILU preconditioner
    # Use maxiter * m for total total iterations potentially
    gmres!(x, ns.mat, b; reltol=ns.solver.rel_tol, maxiter=ns.solver.maxiter, restart=ns.solver.m, Pl=ns.ilu_cache, log=false)
end


