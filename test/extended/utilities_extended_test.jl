# ==============================================================================================
# Nature & Intent:
# Advanced continuous derivative checks. Verifies complex mathematical logic, specifically that 
# the analytical gradient definitions of radial smooth bump porosity match finite differences. Also
# rigorously tests the Gridap AD rules mapping tensor contractions: $(u \cdot \nabla)u \equiv \nabla u^T \cdot u$.
#
# Mathematical Formulation Alignment:
# Ensures that the explicit continuous gradients (used when `freeze_jacobian_cusp=false`) are
# structurally correct and that mathematical notation translation to Julia operators is sound.
#
# Associated Files / Functions:
# - `src/utils/porosity.jl`
# - `test/extended/ManufacturedSolutions/run_test.jl`
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Test
using LinearAlgebra
using Gridap
using PorousNSSolver

function alpha_exact(x, alpha_0, r1, r2)
    dx = x[1] - 0.0
    dy = x[2] - 0.0
    r = sqrt(dx^2 + dy^2)
    if r <= r1
        return alpha_0
    elseif r >= r2
        return 1.0
    else
        eta = (r^2 - r1^2) / (r2^2 - r1^2)
        gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
        return 1.0 - (1.0 - alpha_0) / (1.0 + exp(gamma_val))
    end
end

function grad_alpha_exact(x, alpha_0, r1, r2)
    dx = x[1] - 0.0
    dy = x[2] - 0.0
    r = sqrt(dx^2 + dy^2)
    if r <= r1 || r >= r2
        return [0.0, 0.0]
    else
        eta = (r^2 - r1^2) / (r2^2 - r1^2)
        gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
        
        deta_dr = 2.0 * r / (r2^2 - r1^2)
        dgamma_deta = (2.0*eta^2 - 2.0*eta + 1.0) / (eta^2 * (1.0 - eta)^2)
        
        exp_g = exp(gamma_val)
        dalpha_dgamma = (1.0 - alpha_0) * exp_g / (1.0 + exp_g)^2
        
        dalpha_dr = dalpha_dgamma * dgamma_deta * deta_dr
        
        return [dalpha_dr * (dx / r), dalpha_dr * (dy / r)]
    end
end

@testset "PorousNSSolver Tests" begin

    @testset "Analytical Porosity Gradient Evaluation" begin
        # Run finite difference check against analytical gradient
        x_test = [0.2, 0.25]
        eps = 1e-7
        a0 = alpha_exact(x_test, 0.5, 0.2, 0.5)
        ax = alpha_exact(x_test .+ [eps, 0.0], 0.5, 0.2, 0.5)
        ay = alpha_exact(x_test .+ [0.0, eps], 0.5, 0.2, 0.5)

        grad_fd = [(ax - a0)/eps, (ay - a0)/eps]
        grad_an = grad_alpha_exact(x_test, 0.5, 0.2, 0.5)

        # Analytical gradient should extremely closely match the finite differences (Float64 eps scale limits)
        @test isapprox(grad_fd[1], grad_an[1], atol=1e-5)
        @test isapprox(grad_fd[2], grad_an[2], atol=1e-5)
    end

    @testset "Gridap AD Tensor Convection Evaluation" begin
        u_test(x) = VectorValue(sin(x[1])*x[2], x[1]*cos(x[2]))
        x_eval = Point(1.0, 2.0)
        
        # Gridap AD functionally processes `(u ⋅ ∇)u` as `∇(u)' ⋅ u`
        grad_u_ad = ∇(u_test)(x_eval)
        conv_ad = transpose(grad_u_ad) ⋅ u_test(x_eval)
        
        u1_val = sin(x_eval[1])*x_eval[2]
        u2_val = x_eval[1]*cos(x_eval[2])
        conv1 = u1_val * cos(x_eval[1])*x_eval[2] + u2_val * sin(x_eval[1])
        conv2 = u1_val * cos(x_eval[2]) + u2_val * (-x_eval[1]*sin(x_eval[2]))
        conv_hand = VectorValue(conv1, conv2)
        
        @test isapprox(conv_ad[1], conv_hand[1], atol=1e-12)
        @test isapprox(conv_ad[2], conv_hand[2], atol=1e-12)
    end

end
