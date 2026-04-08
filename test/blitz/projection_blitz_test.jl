# ==============================================================================================
# Nature & Intent:
# Verifies the behavior of varying SubGrid Scale (SGS) fine-scale projection modes.
# Validates proper calculation of fine-scale velocity and pressure based on whether the 
# projection operates on the full residual or omits specific matrices (like omitting 
# Reaction when ConstantSigma is violated, to avoid uninvertible bounds).
#
# Mathematical Formulation Alignment:
# Maps strictly to the VMS logic specifying how the unresolved subscales $\mathbf{v}'$ and $p'$ 
# are algebraically approximated. Confirms the correct mapping of `ProjectFullResidual` vs fallback modes.
#
# Associated Files / Functions:
# - `src/formulations/projection.jl` (`apply_projection_u`, `apply_projection_p`, `sanitize_projection_policy`)
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

@testset "fast: projection policies" begin
    @testset "ProjectFullResidual" begin
        R_u = VectorValue(1.0, -2.0)
        σ = 3.0
        u = VectorValue(0.5, 0.25)
        pi_u = VectorValue(0.1, 0.2)

        @test PorousNSSolver.apply_projection_u(PorousNSSolver.ProjectFullResidual(), R_u, σ, u, pi_u, false) == R_u
        @test PorousNSSolver.apply_projection_u(PorousNSSolver.ProjectFullResidual(), R_u, σ, u, pi_u, true) == R_u - pi_u

        R_p = 2.0
        p = 0.3
        pi_p = 0.4
        @test PorousNSSolver.apply_projection_p(PorousNSSolver.ProjectFullResidual(), R_p, 0.1, p, pi_p, false) == R_p
        @test PorousNSSolver.apply_projection_p(PorousNSSolver.ProjectFullResidual(), R_p, 0.1, p, pi_p, true) == R_p - pi_p
    end

    @testset "ProjectResidualWithoutReactionWhenConstantSigma algebra" begin
        R_u = VectorValue(1.0, -2.0)
        σ = 3.0
        u = VectorValue(0.5, 0.25)
        pi_u = VectorValue(0.1, 0.2)

        @test PorousNSSolver.apply_projection_u(PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma(), R_u, σ, u, pi_u, true) ==
              R_u - (σ*u) - pi_u

        R_p = 2.0
        p = 0.3
        pi_p = 0.4
        eps_val = 0.1
        @test PorousNSSolver.apply_projection_p(PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma(), R_p, eps_val, p, pi_p, true) ==
              R_p - eps_val*p - pi_p
    end

    @testset "ProjectResidualWithoutMassPenalty fallback behavior" begin
        R_p = 2.0
        p = 0.3
        pi_p = 0.4
        eps_val = 0.1

        @test PorousNSSolver.apply_projection_p(PorousNSSolver.ProjectResidualWithoutMassPenalty(), R_p, eps_val, p, pi_p, true) ==
              PorousNSSolver.apply_projection_p(PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma(), R_p, eps_val, p, pi_p, true)
    end
    
    @testset "Sanitize invalid bounds smoothly gracefully correctly" begin
        policy = PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma()
        law1 = PorousNSSolver.ConstantSigmaLaw(1.0)
        law2 = PorousNSSolver.ForchheimerErgunLaw(1.0, 1.0)
        
        @test PorousNSSolver.sanitize_projection_policy(policy, law1) === policy
        
        @test_throws ErrorException PorousNSSolver.sanitize_projection_policy(policy, law2; autocorrect=false)
        @test PorousNSSolver.sanitize_projection_policy(policy, law2; autocorrect=true) === PorousNSSolver.ProjectFullResidual()
    end
end
