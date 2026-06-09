# src/solvers/accelerators.jl
#
# Anderson acceleration (a.k.a. Anderson mixing) for fixed-point iterations x_{k+1} = G(x_k).
# This is a convergence accelerator: instead of taking the raw next iterate, it extrapolates from a
# short history of past states and fixed-point residuals by solving a small least-squares problem,
# turning linearly-convergent fixed-point loops into faster (super-linear in practice) ones.
#
# [available-option] The accelerator is self-contained here and exported as `AndersonAccelerator`.
# It is intended for the OSGS coupled solve, whose inexact-Newton iteration (the dense ∂π/∂U term is
# dropped) converges only *linearly*. If wired into the solver, the controlling config knob (depth /
# relaxation / safety) must be added AT THE POINT OF CONSUMPTION so it is never dead config.
using LinearAlgebra

"""
    AndersonAccelerator

State object for Anderson acceleration. It carries the rolling history of past iterate states `X` and
their fixed-point residuals `F` (residual = G(x) - x), which `update!` uses to build the extrapolation.

The optional mass matrix `M_mat` makes the internal least-squares problem L²-weighted rather than a
plain Euclidean one. This matters for a coupled (velocity, pressure) DOF vector: an `L²` mass-weighted
norm measures the residual in the physically meaningful function-space norm instead of letting raw DOF
magnitudes (which differ in scale across fields/domains) dominate the fit.
"""
mutable struct AndersonAccelerator
    m::Int                       # Maximum history depth: at most m past differences are mixed in
    relaxation_factor::Float64   # Damping/mixing factor β: how much of the residual to add to the mixed state
    safety_factor::Float64       # Powell-style restart threshold; a too-large extrapolation is rejected and the history cleared
    iter::Int                    # Accumulated iteration count since the last restart
    history_X::Vector{Vector{Float64}}   # Stored past states x_k (newest pushed last)
    history_F::Vector{Vector{Float64}}   # Stored past residuals f_k = G(x_k) - x_k, aligned with history_X
    M_mat::Union{Nothing, AbstractMatrix}   # Optional L²-mass matrix; `nothing` falls back to Euclidean least-squares

    function AndersonAccelerator(m::Int, relaxation_factor::Float64, safety_factor::Float64, M_mat::Union{Nothing, AbstractMatrix}=nothing)
        new(m, relaxation_factor, safety_factor, 0, Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), M_mat)
    end
end

"""
    update!(acc::AndersonAccelerator, x_k::Vector{Float64}, g_k::Vector{Float64})

Advance one Anderson-accelerated step of the fixed-point iteration x_{k+1} = G(x_k).

Inputs: `x_k` is the current state and `g_k = G(x_k)` is the result of applying the fixed-point map once.
The method records the residual f_k = g_k - x_k, then forms `m_active` columns of consecutive
differences ΔF (in residuals) and ΔX (in states). It finds coefficients `gamma` that best approximate the
current residual in the span of those ΔF columns (the least-squares step), and combines the matching
ΔX directions to produce the extrapolated next state. With only one history entry it degrades
gracefully to a relaxed-Picard step x_k + β f_k.

Returns the extrapolated state `x_next` (or the relaxed-Picard fallback after a safety restart).
"""
function update!(acc::AndersonAccelerator, x_k::Vector{Float64}, g_k::Vector{Float64})
    # Fixed-point residual of the current iterate; it is zero exactly at the fixed point G(x*) = x*.
    f_k = g_k .- x_k

    # Append the current (state, residual) pair to the rolling history.
    push!(acc.history_X, copy(x_k))
    push!(acc.history_F, copy(f_k))

    # Cap the history length at m+1 entries (m differences) by evicting the oldest.
    if length(acc.history_X) > acc.m + 1
        popfirst!(acc.history_X)
        popfirst!(acc.history_F)
    end

    acc.iter += 1

    # Bootstrap: without a previous entry there are no differences to mix, so take a relaxed-Picard step.
    if length(acc.history_X) < 2
        return x_k .+ acc.relaxation_factor .* f_k
    end

    n_history = length(acc.history_X)
    m_active = n_history - 1     # number of usable consecutive differences
    dof_size = length(f_k)

    # Difference matrices: column i is current minus the i-th historical residual/state.
    DeltaF = zeros(Float64, dof_size, m_active)
    DeltaX = zeros(Float64, dof_size, m_active)

    for i in 1:m_active
        DeltaF[:, i] .= f_k .- acc.history_F[i]
        DeltaX[:, i] .= x_k .- acc.history_X[i]
    end

    # Solve the small (m_active × m_active) least-squares system for the mixing coefficients `gamma`,
    # i.e. find gamma minimizing the residual reproduced by ΔF·gamma against f_k. The Tikhonov term
    # λ·I, scaled to the matrix trace, regularizes the often ill-conditioned normal equations.
    if acc.M_mat !== nothing
        # L²-weighted variant: argmin || M^(1/2) (DeltaF * gamma - f_k) ||_2^2
        # Normal equations: (DeltaF^T * M * DeltaF) * gamma = DeltaF^T * M * f_k
        A_ls = DeltaF' * (acc.M_mat * DeltaF)
        b_ls = DeltaF' * (acc.M_mat * f_k)
        λ = eps(Float64) * (tr(A_ls) / size(A_ls, 1))
        gamma = (A_ls + λ * I) \ b_ls
    else
        # Euclidean variant: argmin || DeltaF * gamma - f_k ||_2^2
        A_ls = DeltaF' * DeltaF
        b_ls = DeltaF' * f_k
        λ = eps(Float64) * (tr(A_ls) / size(A_ls, 1))
        gamma = (A_ls + λ * I) \ b_ls
    end

    # Build the mixed state/residual by subtracting the gamma-weighted history directions, then add the
    # relaxed residual: x_next = (x_k - ΔX·gamma) + β (f_k - ΔF·gamma).
    x_mixed = copy(x_k)
    f_mixed = copy(f_k)
    for i in 1:m_active
        x_mixed .-= gamma[i] .* DeltaX[:, i]
        f_mixed .-= gamma[i] .* DeltaF[:, i]
    end

    x_next = x_mixed .+ acc.relaxation_factor .* f_mixed

    # Powell-style safeguard: if the extrapolation moved much farther from the plain map output g_k than
    # the safety_factor allows, distrust it — discard the step, reset the history, and fall back to a
    # relaxed-Picard update. This prevents a bad least-squares solve from throwing the iteration off.
    if norm(x_next .- g_k, Inf) > acc.safety_factor * norm(f_k, Inf)
        empty!(acc.history_X)
        empty!(acc.history_F)
        acc.iter = 0
        return x_k .+ acc.relaxation_factor .* f_k
    end

    return x_next
end
