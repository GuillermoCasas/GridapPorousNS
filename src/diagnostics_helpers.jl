# src/diagnostics_helpers.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

function weak_form_g_sigma(u, v, config::PorousNSSolver.PorousNSConfig, dΩ, h, alpha_custom=nothing)
    model = PorousNSSolver.build_reaction_model(config)
    return _weak_form_g_sigma_impl(u, v, config, model, dΩ, h, alpha_custom)
end

function _weak_form_g_sigma_impl(u, v, config::PorousNSSolver.PorousNSConfig, model::M, dΩ, h, alpha_custom) where {M<:PorousNSSolver.AbstractReactionModel}
    α = isnothing(alpha_custom) ? config.domain.alpha_0 : alpha_custom
    σ = Operation((u_v, h_v, a_v) -> model(PorousNSSolver.ThermodynamicState(u_v, h_v, a_v)))(u, h, α)
    return ∫( v ⋅ (σ * u) )dΩ
end

function weak_form_s_sigma(X, v, config::PorousNSSolver.PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing)
    model = PorousNSSolver.build_reaction_model(config)
    return _weak_form_s_sigma_impl(X, v, config, model, dΩ, h, f_custom, alpha_custom)
end

function _weak_form_s_sigma_impl(X, v, config::PorousNSSolver.PorousNSConfig, model::M, dΩ, h, f_custom, alpha_custom) where {M<:PorousNSSolver.AbstractReactionModel}
    u, p = X
    α = isnothing(alpha_custom) ? config.domain.alpha_0 : alpha_custom
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    
    let ν = config.phys.nu,
        k_deg = config.numerical_method.element_spaces.k_velocity,
        c_1 = 4.0 * config.numerical_method.element_spaces.k_velocity^4,
        c_2 = 2.0 * config.numerical_method.element_spaces.k_velocity^2,
        reaction_trait = PorousNSSolver.get_reaction_mode_trait(config.numerical_method.solver.experimental_reaction_mode),
        tau_reg_lim = config.phys.tau_regularization_limit
        
        σ = Operation((u_v, h_v, a_v) -> model(PorousNSSolver.ThermodynamicState(u_v, h_v, a_v)))(u, h, α)
        τ_1 = Operation((u_v, h_v, a_v) -> PorousNSSolver.compute_tau_1(PorousNSSolver.ThermodynamicState(u_v, h_v, a_v), ν, c_1, c_2, tau_reg_lim, model, reaction_trait))(u, h, α)
        alpha_conv = α
        
        conv_u = ∇(u)' ⋅ u
        div_visc_u = ν * (∇(u)' ⋅ ∇(α))
        if k_deg > 1
            div_visc_u = div_visc_u + α * ν * Δ(u)
        end
        R_u = alpha_conv * conv_u + alpha_conv * ∇(p) + σ * u - div_visc_u - f
        
        # Reaction adjoint contribution within ASGS stabilization
        return ∫( (- σ * v) ⋅ (τ_1 * R_u) )dΩ
    end
end

function evaluate_residual_components(u_h, p_h, config::PorousNSSolver.PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing, g_custom=nothing)
    model = PorousNSSolver.build_reaction_model(config)
    return _evaluate_residual_components_impl(u_h, p_h, config, model, dΩ, h, f_custom, alpha_custom, g_custom)
end

function _evaluate_residual_components_impl(u_h, p_h, config::PorousNSSolver.PorousNSConfig, model::M, dΩ, h, f_custom, alpha_custom, g_custom) where {M<:PorousNSSolver.AbstractReactionModel}
    α = isnothing(alpha_custom) ? config.domain.alpha_0 : alpha_custom
    ν = config.phys.nu
    k_deg = config.numerical_method.element_spaces.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    eps_val = config.phys.eps_val
    g_mass = isnothing(g_custom) ? 0.0 : g_custom
    let ν = config.phys.nu,
        k_deg = config.numerical_method.element_spaces.k_velocity,
        c_1 = 4.0 * k_deg^4,
        c_2 = 2.0 * k_deg^2
        
        σ = Operation((u_v, h_v, a_v) -> model(PorousNSSolver.ThermodynamicState(u_v, h_v, a_v)))(u_h, h, α)
        alpha_conv = α
    
    conv_u = ∇(u_h)' ⋅ u_h
    term_conv = alpha_conv * conv_u
    term_pres = alpha_conv * ∇(p_h)
    term_reac = σ * u_h
    
    div_visc_u = ν * (∇(u_h)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_u = div_visc_u + α * ν * Δ(u_h)
    end
    term_diff = - div_visc_u
    term_forc = - f
    
    R_u = term_conv + term_pres + term_reac + term_diff + term_forc
    
    term_eps_p = eps_val * p_h
    div_alpha_u = α * (∇⋅u_h) + u_h ⋅ ∇(α)
    term_div = div_alpha_u
    term_mass_forc = - g_mass
    
    R_p = term_eps_p + term_div + term_mass_forc
    
    norm_sq(v) = v ⋅ v
    
    n_conv = sqrt(abs(sum(∫(norm_sq(term_conv))dΩ)))
    n_pres = sqrt(abs(sum(∫(norm_sq(term_pres))dΩ)))
    n_reac = sqrt(abs(sum(∫(norm_sq(term_reac))dΩ)))
    n_diff = sqrt(abs(sum(∫(norm_sq(term_diff))dΩ)))
    n_forc = sqrt(abs(sum(∫(norm_sq(term_forc))dΩ)))
    n_Ru   = sqrt(abs(sum(∫(norm_sq(R_u))dΩ)))
    
    n_epsp = sqrt(abs(sum(∫(term_eps_p * term_eps_p)dΩ)))
    n_div  = sqrt(abs(sum(∫(term_div * term_div)dΩ)))
    n_mfor = sqrt(abs(sum(∫(term_mass_forc * term_mass_forc)dΩ)))
    n_Rp   = sqrt(abs(sum(∫(R_p * R_p)dΩ)))
    
    return Dict(
        "mom_conv" => n_conv, "mom_pres" => n_pres, "mom_reac" => n_reac,
        "mom_diff" => n_diff, "mom_forc" => n_forc, "mom_total" => n_Ru,
        "mass_eps" => n_epsp, "mass_div" => n_div, "mass_forc" => n_mfor, "mass_total" => n_Rp
    )
    end
end

function test_jacobian_fd(op, U_space, x::AbstractVector, d::AbstractVector)
    x_state = FEFunction(U_space, x)
    b0 = allocate_residual(op, x_state)
    residual!(b0, op, x_state)
    
    A = allocate_jacobian(op, x_state)
    jacobian!(A, op, x_state)
    
    J_d = A * d
    
    eps_vals = [1e-4, 1e-6, 1e-8]
    x_pert = similar(x)
    b_pert = similar(b0)
    
    results = Dict{Float64, Dict{String, Float64}}()
    for ep in eps_vals
        x_pert .= x .+ ep .* d
        x_pert_state = FEFunction(U_space, x_pert)
        residual!(b_pert, op, x_pert_state)
        
        FD = (b_pert .- b0) ./ ep
        diff = FD .- J_d
        
        n_Jd = norm(J_d)
        n_FD = norm(FD)
        n_diff = norm(diff)
        rel_diff = n_diff / (n_FD + 1e-16)
        
        results[ep] = Dict(
            "norm_Jd" => n_Jd,
            "norm_FD" => n_FD,
            "norm_diff" => n_diff,
            "rel_diff" => rel_diff
        )
    end
    return results
end

function test_merit_descent(b::AbstractVector, A, dx::AbstractVector)
    # phi(x) = 0.5 * ||R(x)||^2
    # phi'(0) = d/dalpha [0.5 * ||R(x + alpha*d)||] = R(x)^T * J(x) * d
    # For Newton step: d = dx where A * dx = b  (or A * dx = -b depending on convention)
    # The Gridap convention is Jacobian * dx = -Residual, BUT in safesolver.jl: solve!(dx, ns, b) implies A * dx = b,
    # and then x .= x_old .- alpha .* dx.
    # Wait, if A * dx = b, then J * dx = R.
    # The step direction is d_true = -dx.
    # Therefore phi'(0) = b^T * A * (-dx) = - b^T * (A * dx) = - b^T * b < 0.
    # Let's verify this analytically.
    Adx = A * dx
    phi_prime = dot(b, -Adx)  # direction is -dx
    return phi_prime
end

function estimate_jacobian_condition(A)
    n = size(A, 1)
    if n > 5000
        # For large matrices, an exact condition number via SVD is too expensive.
        # We compute sparse estimates or simply the L1/Inf norms.
        norm_inf = opnorm(A, Inf)
        return Dict("size" => n, "norm_inf" => norm_inf, "cond_estimate" => -1.0)
    else
        # Small matrix, compute SVD
        # Matrix must be dense
        try
            Adense = Matrix(A)
            S = svdvals(Adense)
            cnd = S[1] / S[end]
            return Dict("size" => n, "norm_A" => S[1], "norm_inv_A" => 1.0/S[end], "cond" => cnd, "min_sv" => S[end], "max_sv" => S[1])
        catch
            return Dict("size" => n, "norm_inf" => opnorm(A, Inf), "cond_estimate" => -1.0)
        end
    end
end
