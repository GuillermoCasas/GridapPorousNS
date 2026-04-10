# src/models/porosity.jl
using Gridap

abstract type AbstractPorosityField <: Function end

struct SmoothRadialPorosity{T} <: AbstractPorosityField
    alpha_0::T
    alpha_infty::T
    r_1::T
    r_2::T
end

function SmoothRadialPorosity(alpha_0::T, r_1::T, r_2::T) where T
    return SmoothRadialPorosity(alpha_0, one(T), r_1, r_2)
end

# Make it natively callable for CellField encapsulation
(field::SmoothRadialPorosity)(x) = alpha(field, x)

import Gridap.Fields: ∇
∇(field::SmoothRadialPorosity) = x -> grad_alpha(field, x)

function alpha(field::SmoothRadialPorosity, x)
    alpha_0 = field.alpha_0
    alpha_infty = field.alpha_infty
    r1 = field.r_1
    r2 = field.r_2
    
    r_sq = x[1]^2 + x[2]^2
    r = sqrt(r_sq)
    
    if r <= r1
        return alpha_0
    elseif r >= r2
        return alpha_infty
    end
    
    eta = (r_sq - r1^2) / (r2^2 - r1^2)
    gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
    
    if gamma_val > 100.0
        return alpha_infty
    elseif gamma_val < -100.0
        return alpha_0
    end
    
    return alpha_infty - (alpha_infty - alpha_0) / (1.0 + exp(gamma_val))
end

function _analyze_alpha(field::SmoothRadialPorosity, x)
    alpha_0 = field.alpha_0
    alpha_infty = field.alpha_infty
    r1 = field.r_1
    r2 = field.r_2
    
    r_sq = x[1]^2 + x[2]^2
    r = sqrt(r_sq)
    
    if r <= r1 || r >= r2
        return alpha(field, x), VectorValue(0.0, 0.0), 0.0
    end
    
    eta = (r_sq - r1^2) / (r2^2 - r1^2)
    gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
    
    if gamma_val > 100.0 || gamma_val < -100.0
        return alpha(field, x), VectorValue(0.0, 0.0), 0.0
    end
    
    exp_g = exp(gamma_val)
    alpha_val = alpha_infty - (alpha_infty - alpha_0) / (1.0 + exp_g)
    
    deta_dr = 2.0 * r / (r2^2 - r1^2)
    d2eta_dr2 = 2.0 / (r2^2 - r1^2)
    
    dgamma_deta = (2.0*eta^2 - 2.0*eta + 1.0) / (eta^2 * (1.0 - eta)^2)
    d2gamma_deta2 = (4.0*eta - 2.0) / (eta^2 * (1.0 - eta)^2) - 2.0 * (2.0*eta^2 - 2.0*eta + 1.0)*(1.0 - 2.0*eta) / (eta^3 * (1.0 - eta)^3)
    
    da_dg = (alpha_infty - alpha_0) * exp_g / (1.0 + exp_g)^2
    d2a_dg2 = (alpha_infty - alpha_0) * exp_g * (1.0 - exp_g) / (1.0 + exp_g)^3
    
    da_dr = da_dg * dgamma_deta * deta_dr
    d2a_dr2 = d2a_dg2 * (dgamma_deta * deta_dr)^2 + da_dg * (d2gamma_deta2 * deta_dr^2 + dgamma_deta * d2eta_dr2)
    
    lap_alpha_val = d2a_dr2 + (1.0/r) * da_dr
    grad_alpha_val = VectorValue(da_dr * (x[1]/r), da_dr * (x[2]/r))
    return alpha_val, grad_alpha_val, lap_alpha_val
end

function grad_alpha(field::SmoothRadialPorosity, x)
    return _analyze_alpha(field, x)[2]
end

function lap_alpha(field::SmoothRadialPorosity, x)
    return _analyze_alpha(field, x)[3]
end
