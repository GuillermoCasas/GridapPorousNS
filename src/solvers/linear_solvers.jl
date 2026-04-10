# src/solvers/linear_solvers.jl

using Gridap
using Gridap.Algebra
using IterativeSolvers
using IncompleteLU

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


