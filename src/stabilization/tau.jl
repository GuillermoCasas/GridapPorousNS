# src/stabilization/tau.jl
using Gridap
using Gridap.Algebra

function compute_tau_1(kin, med, ν, c_1, c_2, tau_reg_lim, rxn_law::AbstractReactionLaw)
    mag_u = kin.mag_u
    τ_1_NS_val = 1.0 / ( (c_1 * ν / (med.h * med.h)) + (c_2 * mag_u / med.h) + tau_reg_lim )
    sig_val = sigma(rxn_law, kin, med, mag_u)
    return 1.0 / ( (med.alpha / τ_1_NS_val) + sig_val + tau_reg_lim )
end

function compute_tau_2(kin, med, ν, c_1, c_2, tau_reg_lim)
    mag_u = kin.mag_u
    τ_1_NS_val = 1.0 / ( (c_1 * ν / (med.h * med.h)) + (c_2 * mag_u / med.h) + tau_reg_lim )
    return (med.h * med.h) / ( (c_1 * med.alpha * τ_1_NS_val) + tau_reg_lim )
end

function compute_dtau_1_du(kin, med, du_val, ν, c_1, c_2, tau_reg_lim, freeze_cusp, rxn_law::AbstractReactionLaw)
    if freeze_cusp
         return 0.0 * (kin.u ⋅ du_val)
    end
    u_val = kin.u
    h_val = med.h
    alpha_val = med.alpha
    mag_u = kin.mag_u
    
    A_NS_val = (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + tau_reg_lim
    τ_1_NS_val = 1.0 / A_NS_val
    
    sig_val = sigma(rxn_law, kin, med, mag_u)
    τ_1_val = 1.0 / ( (alpha_val * A_NS_val) + sig_val + tau_reg_lim )
    
    dA_NS_du = (c_2 / h_val) / mag_u * (u_val ⋅ du_val)
    dsig_du = dsigma_du(rxn_law, kin, med, mag_u, du_val)
    
    dTauInv_du = alpha_val * dA_NS_du + dsig_du
    
    return - (τ_1_val * τ_1_val) * dTauInv_du
end

function compute_dtau_2_du(kin, med, du_val, ν, c_1, c_2, tau_reg_lim, freeze_cusp)
    if freeze_cusp
        return 0.0 * (kin.u ⋅ du_val)
    end
    h_val = med.h
    alpha_val = med.alpha
    mag_u = kin.mag_u
    u_val = kin.u
    
    A_NS_val = (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + tau_reg_lim
    dA_NS_dmag = c_2 / h_val
    denom = c_1 * alpha_val / A_NS_val + tau_reg_lim
    
    scalar_deriv = (h_val * h_val) / (denom * denom) * (c_1 * alpha_val) / (A_NS_val * A_NS_val) * dA_NS_dmag / mag_u
    return scalar_deriv * (u_val ⋅ du_val)
end
