# src/osgs.jl

"""
    project_residuals(u, p, config, dΩ, h, f_custom, alpha_custom, g_custom, V_pi, Q_pi, U_pi, P_pi)

Computes the L2 orthogonal projection of the strong residuals R_u and R_p onto the unconstrained
finite element spaces V_pi and Q_pi respectively, dynamically stripping zero-order reactions 
(σu and εp) to avert lagging Damköhler sub-grid errors.
"""
function project_residuals(u, p, config::PorousNSConfig, dΩ, h, f_custom, alpha_custom, g_custom, V_pi, Q_pi, U_pi, P_pi)
    
    # Evaluate native strong residuals at the previous state
    R_u_full = strong_residual_u(u, p, config, f_custom, alpha_custom)
    R_p_full = strong_residual_p(u, p, config, alpha_custom, g_custom)
    
    # Dynamically extract pure reacting forces
    Re = config.phys.Re; Da = config.phys.Da
    α = isnothing(alpha_custom) ? config.porosity.alpha_0 : alpha_custom
    ν = 1.0 / Re
    eps_val = max(config.phys.physical_epsilon + config.phys.numerical_epsilon_coefficient * config.porosity.alpha_0 / (ν * (1.0 + Re + Da)), 1e-8)

    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + 1e-12)
        return a_term + b_term * mag_u
    end
    σ = Operation(compute_sigma)(u, α)
    
    R_u_val = R_u_full - (σ * u)
    R_p_val = R_p_full - (eps_val * p)
    
    # Formulate explicit L2 projection variational systems
    a_u(x, y) = ∫(y ⋅ x)dΩ
    l_u(y) = ∫(y ⋅ R_u_val)dΩ
    
    a_p(x, y) = ∫(y * x)dΩ
    l_p(y) = ∫(y * R_p_val)dΩ
    
    # Assemble and solve momentum residual projection
    op_u = AffineFEOperator(a_u, l_u, U_pi, V_pi)
    ls_u = LUSolver()
    solver_u = FESolver(ls_u)
    pi_u_h = solve(solver_u, op_u)
    
    # Assemble and solve continuity residual projection 
    op_p = AffineFEOperator(a_p, l_p, P_pi, Q_pi)
    ls_p = LUSolver()
    solver_p = FESolver(ls_p)
    pi_p_h = solve(solver_p, op_p)
    
    return pi_u_h, pi_p_h
end
