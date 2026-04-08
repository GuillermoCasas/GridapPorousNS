using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

@testset "fast: formulation smoke assembly" begin
    model = tiny_model_2d(n=(1,1))
    X, Y, V, Q = tiny_spaces_2d(model; kv=1, kp=1)
    Ω, dΩ, h = tiny_measure(model; degree=6)

    αf = CellField(alpha_lin, Ω)
    ff = CellField(f_zero, Ω)
    gf = CellField(g_zero, Ω)

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

    x = interpolate_everywhere([u_poly, p_poly], X)

    res_paper(y) = PorousNSSolver.build_stabilized_weak_form_residual(
        x, y, form_paper, dΩ, h, ff, αf, gf, nothing, nothing, 4.0, 2.0, 1e-8
    )
    jac_paper(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(
        x, dx, y, form_paper, dΩ, h, ff, αf, gf, nothing, nothing, 4.0, 2.0, 1e-8, false, PorousNSSolver.ExactNewtonMode()
    )

    res_pseudo(y) = PorousNSSolver.build_stabilized_weak_form_residual(
        x, y, form_pseudo, dΩ, h, ff, αf, gf, nothing, nothing, 4.0, 2.0, 1e-8
    )
    jac_pseudo(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(
        x, dx, y, form_pseudo, dΩ, h, ff, αf, gf, nothing, nothing, 4.0, 2.0, 1e-8, false, PorousNSSolver.ExactNewtonMode()
    )

    # Smoke Test: actual sparse structure assembly
    @test_nowarn assemble_vector(res_pseudo, Y)
    @test_nowarn assemble_matrix((dx,y)->jac_pseudo(x, dx, y), X, Y)
end
