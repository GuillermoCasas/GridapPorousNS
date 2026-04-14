# ==============================================================================================
# Nature & Intent:
# A critical Gridap AST (Abstract Syntax Tree) compiler hygiene check. Verifies that the nested
# algebraic complexity within `PaperGeneralFormulation` and `Legacy90d5749Mode` does not overflow 
# the Julia compiler or trigger extreme >10 minute symbolic auto-differentiation tree build times.
# Evaluates matrix block assembly without actually solving the resultant linear system.
#
# Mathematical Formulation Alignment:
# Ensures that mathematical abstractions (such as chaining `KinematicState` -> `MediumState` -> 
# `Reaction` -> `Projection` -> `tau`) remain practically computable. Protects against "refactoring" 
# closures into infinitely deep trees that break `Gridap.jl` JIT routines.
#
# Associated Files / Functions:
# - `src/formulations/continuous_problem.jl` (`build_stabilized_weak_form_residual`, `build_stabilized_weak_form_jacobian`)
# - `src/formulations/formulation.jl`
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

include("../test_utils.jl")

@testset "fast: formulation smoke assembly" begin
    model = tiny_model_2d(n=(1,1))
    X, Y, V, Q = tiny_spaces_2d(model; kv=1, kp=1)
    Ω, dΩ, h = tiny_measure(model; degree=6)

    αf = CellField(alpha_lin, Ω)
    ff = CellField(f_zero, Ω)
    gf = CellField(g_zero, Ω)

    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, nothing, nothing, h, ff, αf, gf)

    form_paper = PorousNSSolver.PaperGeneralFormulation(
        PorousNSSolver.SymmetricGradientViscosity(),
        PorousNSSolver.ConstantSigmaLaw(1.0),
        PorousNSSolver.ProjectFullResidual(),
        PorousNSSolver.SmoothVelocityFloor(1e-3, 0.5, 1e-8),
        1e-2,
        1e-6
    )

    form_pseudo = PorousNSSolver.Legacy90d5749Mode(
        PorousNSSolver.ConstantSigmaLaw(1.0),
        PorousNSSolver.ProjectFullResidual(),
        PorousNSSolver.SmoothVelocityFloor(1e-3, 0.5, 1e-8),
        1e-2,
        1e-6
    )

    formulation_paper = PorousNSSolver.VMSFormulation(form_paper, 4.0, 2.0)
    formulation_pseudo = PorousNSSolver.VMSFormulation(form_pseudo, 4.0, 2.0)
    
    # Fake Phys Config
    phys_cfg = PorousNSSolver.PhysicalProperties(
        nu=1e-2, f_x=0.0, f_y=0.0, eps_val=1e-6, eps_floor=1e-8,
        reaction_model="Constant", sigma_constant=1.0, sigma_linear=0.0,
        sigma_nonlinear=0.0, u_base_floor_ref=1e-2, h_floor_weight=1.0,
        epsilon_floor=1e-12, tau_regularization_limit=1e-8
    )

    x = interpolate_everywhere([u_poly, p_poly], X)

    res_paper(y) = PorousNSSolver.build_stabilized_weak_form_residual(
        x, y, setup, formulation_paper, phys_cfg; pi_u=nothing, pi_p=nothing
    )
    jac_paper(x0, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(
        x0, dx, y, setup, formulation_paper, phys_cfg, false, PorousNSSolver.ExactNewtonMode(); pi_u=nothing, pi_p=nothing
    )

    res_pseudo(y) = PorousNSSolver.build_stabilized_weak_form_residual(
        x, y, setup, formulation_pseudo, phys_cfg; pi_u=nothing, pi_p=nothing
    )
    jac_pseudo(x0, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(
        x0, dx, y, setup, formulation_pseudo, phys_cfg, false, PorousNSSolver.ExactNewtonMode(); pi_u=nothing, pi_p=nothing
    )

    # Smoke Test: actual sparse structure assembly
    @test_nowarn assemble_vector(res_pseudo, Y)
    @test_nowarn assemble_matrix((dx,y)->jac_pseudo(x, dx, y), X, Y)
end
