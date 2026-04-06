# src/formulation.jl

function a_resistance(alpha, Re, Da)
    return (1.0 / Da) * (150.0 / Re) * ((1.0 - alpha) / alpha)^2
end

function b_resistance(alpha)
    return 1.75 * (1.0 - alpha) / alpha
end

function strong_residual_u(u, p, config::PorousNSConfig, f_custom=nothing, alpha_custom=nothing)
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom

    U_ref = 1.0 # Characteristic velocity scale (maintains dimensional consistency)
    eps_v_sq = (1e-6 * U_ref)^2 
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_term + b_term * mag_u
    end
    σ = Operation(compute_sigma)(u, α)
    alpha_conv = Operation(a -> a)(α)

    conv_u = ∇(u)' ⋅ u
    div_visc_u = ν * (∇(u)' ⋅ ∇(α))
    if config.discretization.k_velocity > 1
        div_visc_u = div_visc_u + α * ν * Δ(u)
    end
    return alpha_conv * conv_u + alpha_conv * ∇(p) + σ * u - div_visc_u - f
end

function strong_residual_p(u, p, config::PorousNSConfig, alpha_custom=nothing, g_custom=nothing)
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_val = max(config.phys.physical_epsilon + config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)), 1e-8)
    g_mass = isnothing(g_custom) ? 0.0 : g_custom
    div_alpha_u = α * (∇⋅u) + u ⋅ ∇(α)
    return eps_val * p + div_alpha_u - g_mass
end


# ================================
# ASGS Formulations
# ================================
function weak_form_residual(X, Y, config::PorousNSConfig, dΩ, h, f_custom, alpha_custom, g_custom, pi_u::Nothing, pi_p::Nothing)
    u, p = X; v, q = Y
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_val = max(config.phys.physical_epsilon + config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)), 1e-8)
    k_deg = config.discretization.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    c_1 = 4.0 * k_deg^4; c_2 = 2.0 * k_deg^2

    U_ref = 1.0 # Characteristic velocity scale (maintains dimensional consistency)
    eps_v_sq = (1e-6 * U_ref)^2 
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_term + b_term * mag_u
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        a_t = a_resistance(alpha_val, Re, Da)
        b_t = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_t * mag_u + eps_v_sq )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + eps_v_sq )
    end

    σ = Operation(compute_sigma)(u, α); τ_1 = Operation(compute_tau_1)(u, h, α); τ_2 = Operation(compute_tau_2)(u, h, α)
    g_mass = isnothing(g_custom) ? 0.0 : g_custom
    alpha_conv = Operation(a -> a)(α); alpha_nu = Operation(a -> a * ν)(α)
    
    conv_u = ∇(u)' ⋅ u
    div_visc_u = ν * (∇(u)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_u = div_visc_u + α * ν * Δ(u)
    end
    R_u = alpha_conv * conv_u + alpha_conv * ∇(p) + σ * u - div_visc_u - f
    
    L_u_star_v = alpha_conv * (∇(v)' ⋅ u) + alpha_conv * ∇(q) - σ * v
    
    div_alpha_u = α * (∇⋅u) + u ⋅ ∇(α)
    R_p = eps_val * p + div_alpha_u - g_mass
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    conv_term = v ⋅ ( alpha_conv * conv_u )
    visc_term = alpha_nu * ( ∇(u) ⊙ ∇(v) ) 
    pres_term = - p * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term  = v ⋅ ( σ * u )
    mass_term = q * (eps_val * p + div_alpha_u)
    src_term  = v ⋅ f + q * g_mass

    stab_mom = L_u_star_v ⋅ (τ_1 * R_u)
    stab_mass = L_q_star * (τ_2 * R_p)

    return ∫( conv_term + visc_term + pres_term + res_term + mass_term - src_term + stab_mom + stab_mass )dΩ
end

function weak_form_jacobian(X, dX, Y, config::PorousNSConfig, dΩ, h, f_custom, alpha_custom, g_custom, pi_u::Nothing, pi_p::Nothing)
    u, p = X; du, dp = dX; v, q = Y
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_val = max(config.phys.physical_epsilon + config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)), 1e-8)
    k_deg = config.discretization.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    c_1 = 4.0 * k_deg^4; c_2 = 2.0 * k_deg^2

    U_ref = 1.0 
    eps_v_sq = (1e-6 * U_ref)^2 
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_term + b_term * mag_u
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        a_t = a_resistance(alpha_val, Re, Da)
        b_t = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_t * mag_u + eps_v_sq )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + eps_v_sq )
    end
    function compute_dsigma_du(u_val, du_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        b_term = b_resistance(alpha_val)
        return b_term * (u_val ⋅ du_val) / mag_u
    end
    function compute_dtau_1_du(u_val, du_val, h_val, alpha_val)
        if config.solver.freeze_jacobian_cusp
            return 0.0 * (u_val ⋅ du_val)
        end
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        df_dmag = (alpha_val * c_2 / h_val) + b_resistance(alpha_val)
        a_t = a_resistance(alpha_val, Re, Da)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        τ_1_val = 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_resistance(alpha_val) * mag_u + eps_v_sq )
        scalar_deriv = - (τ_1_val * τ_1_val) * df_dmag / mag_u
        return scalar_deriv * (u_val ⋅ du_val)
    end
    function compute_dtau_2_du(u_val, du_val, h_val, alpha_val)
        if config.solver.freeze_jacobian_cusp
            return 0.0 * (u_val ⋅ du_val)
        end
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        A_NS_val = (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq
        dA_NS_dmag = c_2 / h_val
        denom = c_1 * alpha_val / A_NS_val + eps_v_sq
        scalar_deriv = (h_val * h_val) / (denom * denom) * (c_1 * alpha_val) / (A_NS_val * A_NS_val) * dA_NS_dmag / mag_u
        return scalar_deriv * (u_val ⋅ du_val)
    end

    σ = Operation(compute_sigma)(u, α); τ_1 = Operation(compute_tau_1)(u, h, α); τ_2 = Operation(compute_tau_2)(u, h, α)
    dsigma_du = Operation(compute_dsigma_du)(u, du, α)
    dtau_1_du = Operation(compute_dtau_1_du)(u, du, h, α)
    dtau_2_du = Operation(compute_dtau_2_du)(u, du, h, α)
    alpha_conv_jac = Operation(a -> a)(α); alpha_nu_jac = Operation(a -> a * ν)(α)
    g_mass = isnothing(g_custom) ? 0.0 : g_custom

    conv_du = ∇(du)' ⋅ u + ∇(u)' ⋅ du
    div_visc_du = ν * (∇(du)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_du = div_visc_du + α * ν * Δ(du)
    end
    R_du = alpha_conv_jac * conv_du + alpha_conv_jac * ∇(dp) + σ * du + (dsigma_du * u) - div_visc_du
    
    L_u_star_v = alpha_conv_jac * (∇(v)' ⋅ u) + alpha_conv_jac * ∇(q) - σ * v
    dL_du_star_v = alpha_conv_jac * (∇(v)' ⋅ du) - (dsigma_du * v)
    
    div_alpha_du = α * (∇⋅du) + du ⋅ ∇(α)
    R_dp = eps_val * dp + div_alpha_du

    conv_term_jac = v ⋅ ( alpha_conv_jac * conv_du )
    visc_term_jac = alpha_nu_jac * ( ∇(du) ⊙ ∇(v) ) 
    pres_term_jac = - dp * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term_jac  = v ⋅ ( σ * du + (dsigma_du * u) )
    mass_term_jac = q * R_dp

    div_visc_u_old = ν * (∇(u)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_u_old = div_visc_u_old + α * ν * Δ(u)
    end
    R_u_old = alpha_conv_jac * (∇(u)' ⋅ u) + alpha_conv_jac * ∇(p) + σ * u - div_visc_u_old - f
    
    div_alpha_u_old = α * (∇⋅u) + u ⋅ ∇(α)
    R_p_old = eps_val * p + div_alpha_u_old - g_mass
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    stab_mom_jac = L_u_star_v ⋅ (τ_1 * R_du + dtau_1_du * R_u_old) + dL_du_star_v ⋅ (τ_1 * R_u_old)
    stab_mass_jac = L_q_star * (τ_2 * R_dp + dtau_2_du * R_p_old)

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end

function weak_form_jacobian_picard(X, dX, Y, config::PorousNSConfig, dΩ, h, f_custom, alpha_custom, g_custom, pi_u::Nothing, pi_p::Nothing)
    u, p = X; du, dp = dX; v, q = Y
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_val = max(config.phys.physical_epsilon + config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)), 1e-8)
    k_deg = config.discretization.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    c_1 = 4.0 * k_deg^4; c_2 = 2.0 * k_deg^2

    U_ref = 1.0 # Characteristic velocity scale (maintains dimensional consistency)
    eps_v_sq = (1e-6 * U_ref)^2 
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_term + b_term * mag_u
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        a_t = a_resistance(alpha_val, Re, Da)
        b_t = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_t * mag_u + eps_v_sq )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + eps_v_sq )
    end

    σ = Operation(compute_sigma)(u, α); τ_1 = Operation(compute_tau_1)(u, h, α); τ_2 = Operation(compute_tau_2)(u, h, α)
    alpha_conv_jac = Operation(a -> a)(α); alpha_nu_jac = Operation(a -> a * ν)(α)
    g_mass = isnothing(g_custom) ? 0.0 : g_custom

    conv_du = ∇(du)' ⋅ u 
    div_visc_du = ν * (∇(du)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_du = div_visc_du + α * ν * Δ(du)
    end
    R_du = alpha_conv_jac * conv_du + alpha_conv_jac * ∇(dp) + σ * du - div_visc_du
    
    L_u_star_v = alpha_conv_jac * (∇(v)' ⋅ u) + alpha_conv_jac * ∇(q) - σ * v
    dL_du_star_v = 0.0 * (∇(v)' ⋅ du)
    
    div_alpha_du = α * (∇⋅du) + du ⋅ ∇(α)
    R_dp = eps_val * dp + div_alpha_du

    conv_term_jac = v ⋅ ( alpha_conv_jac * conv_du )
    visc_term_jac = alpha_nu_jac * ( ∇(du) ⊙ ∇(v) ) 
    pres_term_jac = - dp * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term_jac  = v ⋅ ( σ * du )
    mass_term_jac = q * (eps_val * dp + div_alpha_du)

    div_visc_u_old = ν * (∇(u)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_u_old = div_visc_u_old + α * ν * Δ(u)
    end
    R_u_old = alpha_conv_jac * (∇(u)' ⋅ u) + alpha_conv_jac * ∇(p) + σ * u - div_visc_u_old - f
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    stab_mom_jac = L_u_star_v ⋅ (τ_1 * R_du) + dL_du_star_v ⋅ (τ_1 * R_u_old)
    stab_mass_jac = L_q_star * (τ_2 * R_dp)

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end

# ================================
# OSGS Formulations (with orthogonal residuals pi_u, pi_p)
# ================================
function weak_form_residual(X, Y, config::PorousNSConfig, dΩ, h, f_custom, alpha_custom, g_custom, pi_u, pi_p)
    u, p = X; v, q = Y
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_val = max(config.phys.physical_epsilon + config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)), 1e-8)
    k_deg = config.discretization.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    c_1 = 4.0 * k_deg^4; c_2 = 2.0 * k_deg^2

    U_ref = 1.0 # Characteristic velocity scale (maintains dimensional consistency)
    eps_v_sq = (1e-6 * U_ref)^2 
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_term + b_term * mag_u
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        a_t = a_resistance(alpha_val, Re, Da)
        b_t = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_t * mag_u + eps_v_sq )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + eps_v_sq )
    end

    σ = Operation(compute_sigma)(u, α); τ_1 = Operation(compute_tau_1)(u, h, α); τ_2 = Operation(compute_tau_2)(u, h, α)
    g_mass = isnothing(g_custom) ? 0.0 : g_custom
    alpha_conv = Operation(a -> a)(α); alpha_nu = Operation(a -> a * ν)(α)
    
    conv_u = ∇(u)' ⋅ u
    div_visc_u = ν * (∇(u)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_u = div_visc_u + α * ν * Δ(u)
    end
    R_u = alpha_conv * conv_u + alpha_conv * ∇(p) + σ * u - div_visc_u - f
    
    L_u_star_v = alpha_conv * (∇(v)' ⋅ u) + alpha_conv * ∇(q) - σ * v
    
    div_alpha_u = α * (∇⋅u) + u ⋅ ∇(α)
    R_p = eps_val * p + div_alpha_u - g_mass
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    conv_term = v ⋅ ( alpha_conv * conv_u )
    visc_term = alpha_nu * ( ∇(u) ⊙ ∇(v) ) 
    pres_term = - p * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term  = v ⋅ ( σ * u )
    mass_term = q * (eps_val * p + div_alpha_u)
    src_term  = v ⋅ f + q * g_mass

    stab_mom = L_u_star_v ⋅ (τ_1 * (R_u - (σ * u) - pi_u))
    stab_mass = L_q_star * (τ_2 * (R_p - (eps_val * p) - pi_p))

    return ∫( conv_term + visc_term + pres_term + res_term + mass_term - src_term + stab_mom + stab_mass )dΩ
end

function weak_form_jacobian(X, dX, Y, config::PorousNSConfig, dΩ, h, f_custom, alpha_custom, g_custom, pi_u, pi_p)
    u, p = X; du, dp = dX; v, q = Y
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_val = max(config.phys.physical_epsilon + config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)), 1e-8)
    k_deg = config.discretization.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    c_1 = 4.0 * k_deg^4; c_2 = 2.0 * k_deg^2

    U_ref = 1.0 
    eps_v_sq = (1e-6 * U_ref)^2 
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_term + b_term * mag_u
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        a_t = a_resistance(alpha_val, Re, Da)
        b_t = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_t * mag_u + eps_v_sq )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + eps_v_sq )
    end
    function compute_dsigma_du(u_val, du_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        b_term = b_resistance(alpha_val)
        return b_term * (u_val ⋅ du_val) / mag_u
    end
    function compute_dtau_1_du(u_val, du_val, h_val, alpha_val)
        if config.solver.freeze_jacobian_cusp
            return 0.0 * (u_val ⋅ du_val)
        end
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        df_dmag = (alpha_val * c_2 / h_val) + b_resistance(alpha_val)
        a_t = a_resistance(alpha_val, Re, Da)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        τ_1_val = 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_resistance(alpha_val) * mag_u + eps_v_sq )
        scalar_deriv = - (τ_1_val * τ_1_val) * df_dmag / mag_u
        return scalar_deriv * (u_val ⋅ du_val)
    end
    function compute_dtau_2_du(u_val, du_val, h_val, alpha_val)
        if config.solver.freeze_jacobian_cusp
            return 0.0 * (u_val ⋅ du_val)
        end
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        A_NS_val = (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq
        dA_NS_dmag = c_2 / h_val
        denom = c_1 * alpha_val / A_NS_val + eps_v_sq
        scalar_deriv = (h_val * h_val) / (denom * denom) * (c_1 * alpha_val) / (A_NS_val * A_NS_val) * dA_NS_dmag / mag_u
        return scalar_deriv * (u_val ⋅ du_val)
    end

    σ = Operation(compute_sigma)(u, α); τ_1 = Operation(compute_tau_1)(u, h, α); τ_2 = Operation(compute_tau_2)(u, h, α)
    dsigma_du = Operation(compute_dsigma_du)(u, du, α)
    dtau_1_du = Operation(compute_dtau_1_du)(u, du, h, α)
    dtau_2_du = Operation(compute_dtau_2_du)(u, du, h, α)
    alpha_conv_jac = Operation(a -> a)(α); alpha_nu_jac = Operation(a -> a * ν)(α)
    g_mass = isnothing(g_custom) ? 0.0 : g_custom

    conv_du = ∇(du)' ⋅ u + ∇(u)' ⋅ du
    div_visc_du = ν * (∇(du)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_du = div_visc_du + α * ν * Δ(du)
    end
    R_du = alpha_conv_jac * conv_du + alpha_conv_jac * ∇(dp) + σ * du + (dsigma_du * u) - div_visc_du
    
    L_u_star_v = alpha_conv_jac * (∇(v)' ⋅ u) + alpha_conv_jac * ∇(q) - σ * v
    dL_du_star_v = alpha_conv_jac * (∇(v)' ⋅ du) - (dsigma_du * v)
    
    div_alpha_du = α * (∇⋅du) + du ⋅ ∇(α)
    R_dp = eps_val * dp + div_alpha_du

    conv_term_jac = v ⋅ ( alpha_conv_jac * conv_du )
    visc_term_jac = alpha_nu_jac * ( ∇(du) ⊙ ∇(v) ) 
    pres_term_jac = - dp * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term_jac  = v ⋅ ( σ * du + (dsigma_du * u) )
    mass_term_jac = q * R_dp

    div_visc_u_old = ν * (∇(u)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_u_old = div_visc_u_old + α * ν * Δ(u)
    end
    R_u_old = alpha_conv_jac * (∇(u)' ⋅ u) + alpha_conv_jac * ∇(p) + σ * u - div_visc_u_old - f
    
    div_alpha_u_old = α * (∇⋅u) + u ⋅ ∇(α)
    R_p_old = eps_val * p + div_alpha_u_old - g_mass
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    stab_mom_jac = L_u_star_v ⋅ (τ_1 * (R_du - (σ * du + (dsigma_du * u))) + dtau_1_du * (R_u_old - (σ * u) - pi_u)) + dL_du_star_v ⋅ (τ_1 * (R_u_old - (σ * u) - pi_u))
    stab_mass_jac = L_q_star * (τ_2 * (R_dp - (eps_val * dp)) + dtau_2_du * (R_p_old - (eps_val * p) - pi_p))

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end

function weak_form_jacobian_picard(X, dX, Y, config::PorousNSConfig, dΩ, h, f_custom, alpha_custom, g_custom, pi_u, pi_p)
    u, p = X; du, dp = dX; v, q = Y
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re; ν = 1.0 / Re; Da = config.phys.Da
    eps_val = max(config.phys.physical_epsilon + config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)), 1e-8)
    k_deg = config.discretization.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    c_1 = 4.0 * k_deg^4; c_2 = 2.0 * k_deg^2

    U_ref = 1.0 # Characteristic velocity scale (maintains dimensional consistency)
    eps_v_sq = (1e-6 * U_ref)^2 
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_term + b_term * mag_u
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        a_t = a_resistance(alpha_val, Re, Da)
        b_t = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_t * mag_u + eps_v_sq )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + eps_v_sq )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + eps_v_sq )
    end

    σ = Operation(compute_sigma)(u, α); τ_1 = Operation(compute_tau_1)(u, h, α); τ_2 = Operation(compute_tau_2)(u, h, α)
    alpha_conv_jac = Operation(a -> a)(α); alpha_nu_jac = Operation(a -> a * ν)(α)
    g_mass = isnothing(g_custom) ? 0.0 : g_custom

    conv_du = ∇(du)' ⋅ u 
    div_visc_du = ν * (∇(du)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_du = div_visc_du + α * ν * Δ(du)
    end
    R_du = alpha_conv_jac * conv_du + alpha_conv_jac * ∇(dp) + σ * du - div_visc_du
    
    L_u_star_v = alpha_conv_jac * (∇(v)' ⋅ u) + alpha_conv_jac * ∇(q) - σ * v
    dL_du_star_v = 0.0 * (∇(v)' ⋅ du)
    
    div_alpha_du = α * (∇⋅du) + du ⋅ ∇(α)
    R_dp = eps_val * dp + div_alpha_du

    conv_term_jac = v ⋅ ( alpha_conv_jac * conv_du )
    visc_term_jac = alpha_nu_jac * ( ∇(du) ⊙ ∇(v) ) 
    pres_term_jac = - dp * ( α * (∇⋅v) + ∇(α) ⋅ v )
    res_term_jac  = v ⋅ ( σ * du )
    mass_term_jac = q * (eps_val * dp + div_alpha_du)

    div_visc_u_old = ν * (∇(u)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_u_old = div_visc_u_old + α * ν * Δ(u)
    end
    R_u_old = alpha_conv_jac * (∇(u)' ⋅ u) + alpha_conv_jac * ∇(p) + σ * u - div_visc_u_old - f
    L_q_star = α * (∇⋅v) + v ⋅ ∇(α) - eps_val * q

    stab_mom_jac = L_u_star_v ⋅ (τ_1 * (R_du - (σ * du))) + dL_du_star_v ⋅ (τ_1 * (R_u_old - (σ * u) - pi_u))
    stab_mass_jac = L_q_star * (τ_2 * (R_dp - (eps_val * dp)))

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end

# Default fallback wrapper
function weak_form_residual(X, Y, config::PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing, g_custom=nothing)
    return weak_form_residual(X, Y, config, dΩ, h, f_custom, alpha_custom, g_custom, nothing, nothing)
end

function weak_form_jacobian(X, dX, Y, config::PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing, g_custom=nothing)
    return weak_form_jacobian(X, dX, Y, config, dΩ, h, f_custom, alpha_custom, g_custom, nothing, nothing)
end

function weak_form_jacobian_picard(X, dX, Y, config::PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing, g_custom=nothing)
    return weak_form_jacobian_picard(X, dX, Y, config, dΩ, h, f_custom, alpha_custom, g_custom, nothing, nothing)
end
