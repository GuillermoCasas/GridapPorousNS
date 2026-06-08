# src/solvers/accelerators.jl
#
# [available-option — not yet wired into the solver] Anderson acceleration (mixing) for fixed-point
# iterations. Kept self-contained in this file as a future lever to speed up the OSGS coupled solve,
# whose inexact-Newton (dropped dense ∂π/∂U) converges only *linearly* and can cost 30–104 inner
# iterations per mesh. When wired in, the config surface (an OSGS-acceleration knob) should be added
# AT THE POINT OF CONSUMPTION so it is never dead config. Exported as `AndersonAccelerator`.
using LinearAlgebra

"""
    AndersonAccelerator

Maintains the history of states `X` and fixed-point residuals `F` to compute 
Anderson Mixing (Anderson Acceleration) extrapolations. Supports optional block-weighted 
mass matrices for physically accurate least-squares mappings across separate physical domains.
"""
mutable struct AndersonAccelerator
    m::Int                       # Maximum depth of history
    relaxation_factor::Float64   # Damping factor
    safety_factor::Float64       # Powell-style restart threshold (rejects step + clears history when tripped)
    iter::Int                    # Current iteration count
    history_X::Vector{Vector{Float64}}
    history_F::Vector{Vector{Float64}}
    M_mat::Union{Nothing, AbstractMatrix}                   # Optional mass matrix for L2 weighted least-squares

    function AndersonAccelerator(m::Int, relaxation_factor::Float64, safety_factor::Float64, M_mat::Union{Nothing, AbstractMatrix}=nothing)
        new(m, relaxation_factor, safety_factor, 0, Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), M_mat)
    end
end

"""
    update!(acc::AndersonAccelerator, x_k::Vector{Float64}, g_k::Vector{Float64})

Applies Anderson Acceleration to the fixed point iteration x_{k+1} = G(x_k).
Here `g_k` is the latest evaluation G(x_k), and `x_k` is the previous state.
Returns the extrapolated state `x_next`.
"""
function update!(acc::AndersonAccelerator, x_k::Vector{Float64}, g_k::Vector{Float64})
    # The fixed point residual is f_k = g_k - x_k
    f_k = g_k .- x_k
    
    # Store in history
    push!(acc.history_X, copy(x_k))
    push!(acc.history_F, copy(f_k))
    
    # Maintain maximum depth
    if length(acc.history_X) > acc.m + 1
        popfirst!(acc.history_X)
        popfirst!(acc.history_F)
    end
    
    acc.iter += 1
    
    # If not enough history, perform standard relaxed Picard
    if length(acc.history_X) < 2
        return x_k .+ acc.relaxation_factor .* f_k
    end
    
    n_history = length(acc.history_X)
    m_active = n_history - 1
    dof_size = length(f_k)
    
    DeltaF = zeros(Float64, dof_size, m_active)
    DeltaX = zeros(Float64, dof_size, m_active)
    
    for i in 1:m_active
        DeltaF[:, i] .= f_k .- acc.history_F[i]
        DeltaX[:, i] .= x_k .- acc.history_X[i]
    end
    
    if acc.M_mat !== nothing
        # Weighted least squares using mass matrix: argmin || M^(1/2) (DeltaF * gamma - f_k) ||_2^2
        # Normal equations: (DeltaF^T * M * DeltaF) * gamma = DeltaF^T * M * f_k
        A_ls = DeltaF' * (acc.M_mat * DeltaF)
        b_ls = DeltaF' * (acc.M_mat * f_k)
        λ = eps(Float64) * (tr(A_ls) / size(A_ls, 1))
        gamma = (A_ls + λ * I) \ b_ls
    else
        A_ls = DeltaF' * DeltaF
        b_ls = DeltaF' * f_k
        λ = eps(Float64) * (tr(A_ls) / size(A_ls, 1))
        gamma = (A_ls + λ * I) \ b_ls
    end
    
    x_mixed = copy(x_k)
    f_mixed = copy(f_k)
    for i in 1:m_active
        x_mixed .-= gamma[i] .* DeltaX[:, i]
        f_mixed .-= gamma[i] .* DeltaF[:, i]
    end
    
    x_next = x_mixed .+ acc.relaxation_factor .* f_mixed
    
    if norm(x_next .- g_k, Inf) > acc.safety_factor * norm(f_k, Inf)
        empty!(acc.history_X)
        empty!(acc.history_F)
        acc.iter = 0
        return x_k .+ acc.relaxation_factor .* f_k
    end
    
    return x_next
end
