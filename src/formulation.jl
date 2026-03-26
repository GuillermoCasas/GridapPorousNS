# src/formulation.jl

# Definition of the resistance term components and sigma
function a_resistance(alpha, Re, Da)
    return (1.0 / Da) * (150.0 / Re) * ((1.0 - alpha) / alpha)^2
end

function b_resistance(alpha)
    return 1.75 * (1.0 - alpha) / alpha
end

function weak_form_residual(X, Y, config::PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing)
    u, p = X
    v, q = Y
    
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re
    ν = 1.0 / Re
    Da = config.phys.Da
    eps_val = config.phys.epsilon
    k_deg = config.discretization.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom

    c_1 = 4.0 * k_deg^4
    c_2 = 2.0 * k_deg^2
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        return a_term + b_term * sqrt(u_val ⋅ u_val + 1e-12)
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + 1e-12)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + 1e-12 )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_term + b_term * mag_u + 1e-12 )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + 1e-12)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + 1e-12 )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + (eps_val * h_val * h_val) + 1e-12 )
    end

    σ = Operation(compute_sigma)(u, α)
    τ_1 = Operation(compute_tau_1)(u, h, α)
    τ_2 = Operation(compute_tau_2)(u, h, α)

    alpha_conv = Operation(a -> a)(α)
    alpha_nu2 = Operation(a -> 2.0 * a * ν)(α)
    alpha_eps = Operation(a -> eps_val)(α)
    
    conv_u = ∇(u)' ⋅ u
    div_visc_u = α * ν * Δ(u) + ν * (∇(u) + transpose(∇(u))) ⋅ ∇(α)
    R_u = alpha_conv * conv_u + alpha_conv * ∇(p) + σ * u - div_visc_u - f
    
    conv_v = ∇(v)' ⋅ u
    L_u_star_v = alpha_conv * conv_v + alpha_conv * ∇(q)
    
    R_p = eps_val * p + alpha_conv * (∇⋅u)

    conv_term = v ⋅ ( alpha_conv * conv_u )
    visc_term = alpha_nu2 * ( ε(u) ⊙ ε(v) ) 
    pres_term = v ⋅ ( alpha_conv * ∇(p) )
    res_term  = v ⋅ ( σ * u )
    mass_term = q * (eps_val * p + alpha_conv * (∇⋅u))
    src_term  = v ⋅ f

    stab_mom = L_u_star_v ⋅ (τ_1 * R_u)
    stab_mass = (α * (∇⋅v)) * (τ_2 * R_p)

    return ∫( conv_term + visc_term + pres_term + res_term + mass_term - src_term + stab_mom + stab_mass )dΩ
end

function weak_form_jacobian(X, dX, Y, config::PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing)
    u, p = X
    du, dp = dX
    v, q = Y
    
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    Re = config.phys.Re
    ν = 1.0 / Re
    Da = config.phys.Da
    eps_val = config.phys.epsilon
    k_deg = config.discretization.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom

    c_1 = 4.0 * k_deg^4
    c_2 = 2.0 * k_deg^2
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        return a_term + b_term * sqrt(u_val ⋅ u_val + 1e-12)
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + 1e-12)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + 1e-12 )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_term + b_term * mag_u + 1e-12 )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + 1e-12)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + 1e-12 )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + (eps_val * h_val * h_val) + 1e-12 )
    end

    σ = Operation(compute_sigma)(u, α)
    τ_1 = Operation(compute_tau_1)(u, h, α)
    τ_2 = Operation(compute_tau_2)(u, h, α)

    alpha_conv = Operation(a -> a)(α)
    alpha_nu2 = Operation(a -> 2.0 * a * ν)(α)
    
    conv_du = ∇(u)' ⋅ du + ∇(du)' ⋅ u
    div_visc_du = α * ν * Δ(du) + ν * (∇(du) + transpose(∇(du))) ⋅ ∇(α)
    R_du = alpha_conv * conv_du + alpha_conv * ∇(dp) + σ * du - div_visc_du
    
    conv_v = ∇(v)' ⋅ u
    L_u_star_v = alpha_conv * conv_v + alpha_conv * ∇(q)
    
    dL_du_star_v = alpha_conv * (∇(v)' ⋅ du)
    
    R_dp = eps_val * dp + alpha_conv * (∇⋅du)

    conv_term_jac = v ⋅ ( alpha_conv * conv_du )
    visc_term_jac = alpha_nu2 * ( ε(du) ⊙ ε(v) ) 
    pres_term_jac = v ⋅ ( alpha_conv * ∇(dp) )
    res_term_jac  = v ⋅ ( σ * du )
    mass_term_jac = q * (eps_val * dp + alpha_conv * (∇⋅du))

    div_visc_u_old = α * ν * Δ(u) + ν * (∇(u) + transpose(∇(u))) ⋅ ∇(α)
    R_u_old = alpha_conv * (∇(u)' ⋅ u) + alpha_conv * ∇(p) + σ * u - div_visc_u_old - f
    stab_mom_jac = L_u_star_v ⋅ (τ_1 * R_du) + dL_du_star_v ⋅ (τ_1 * R_u_old)
    stab_mass_jac = (alpha_conv * (∇⋅v)) * (τ_2 * R_dp)

    return ∫( conv_term_jac + visc_term_jac + pres_term_jac + res_term_jac + mass_term_jac + stab_mom_jac + stab_mass_jac )dΩ
end
