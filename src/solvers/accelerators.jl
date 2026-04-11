# src/solvers/accelerators.jl

using LinearAlgebra

"""
    AndersonAccelerator

Maintains the history of states `X` and fixed-point residuals `F` to compute 
Anderson Mixing (Anderson Acceleration) extrapolations.
"""
mutable struct AndersonAccelerator
    m::Int                       # Maximum depth of history
    relaxation_factor::Float64   # Damping factor
    iter::Int                    # Current iteration count
    history_X::Vector{Vector{Float64}}
    history_F::Vector{Vector{Float64}}
    
    function AndersonAccelerator(m::Int, relaxation_factor::Float64=1.0)
        new(m, relaxation_factor, 0, Vector{Vector{Float64}}(), Vector{Vector{Float64}}())
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
    
    # If not enough history, just perform standard relaxed Picard
    if length(acc.history_X) < 2
        return x_k .+ acc.relaxation_factor .* f_k
    end
    
    n_history = length(acc.history_X)
    
    # Construct Delta F matrix and Delta X matrix
    # Columns are f_k - f_i
    m_active = n_history - 1
    dof_size = length(f_k)
    
    DeltaF = zeros(Float64, dof_size, m_active)
    DeltaX = zeros(Float64, dof_size, m_active)
    
    for i in 1:m_active
        # history_F[i] is an older residual, f_k is the newest
        DeltaF[:, i] .= f_k .- acc.history_F[i]
        DeltaX[:, i] .= x_k .- acc.history_X[i]
    end
    
    # Solve least squares problem: DeltaF * gamma = f_k
    # gamma = argmin || DeltaF * gamma - f_k ||_2
    # We use standard QR via the backslash operator.
    gamma = DeltaF \ f_k
    
    # The mixed state and mixed residual:
    # x_mixed = x_k - sum(gamma_i * DeltaX_i)
    # f_mixed = f_k - sum(gamma_i * DeltaF_i)
    
    x_mixed = copy(x_k)
    f_mixed = copy(f_k)
    for i in 1:m_active
        x_mixed .-= gamma[i] .* DeltaX[:, i]
        f_mixed .-= gamma[i] .* DeltaF[:, i]
    end
    
    # Final accelerated update
    x_next = x_mixed .+ acc.relaxation_factor .* f_mixed
    
    # -------------------------------------------------------------------------
    # SAFETY GUARD: Prevent Anderson Extrapolation Explosion
    # If the least-squares mapping of historical noise massively shoots the
    # state out of proportion relative to standard Picard, reject the matrix 
    # inversion and conservatively map directly to the simple fixed-point jump.
    # -------------------------------------------------------------------------
    if norm(x_next .- g_k, Inf) > 10.0 * norm(f_k, Inf)
        return x_k .+ acc.relaxation_factor .* f_k
    end
    
    return x_next
end
