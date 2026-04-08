using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

@testset "fast: formulation consistency" begin
    @testset "strong mass residual is exact for matched g" begin
        model = tiny_model_2d()
        Ω, dΩ, h = tiny_measure(model; degree=6)
        αf = CellField(alpha_lin, Ω)
        form = PorousNSSolver.Legacy90d5749Mode(
            PorousNSSolver.ConstantSigmaLaw(1.0),
            PorousNSSolver.ProjectFullResidual(),
            PorousNSSolver.SmoothVelocityFloor(1e-3, 0.5, 1e-8),
            1e-2,
            1e-3
        )

        gmatch(x) = form.eps_val * p_poly(x) + alpha_lin(x)*(∇⋅u_poly)(x) + u_poly(x)⋅(∇(alpha_lin))(x)
        gf = CellField(gmatch, Ω)
        
        uf = CellField(u_poly, Ω)
        pf = CellField(p_poly, Ω)

        Rp = PorousNSSolver.eval_strong_residual_p(form, uf, pf, αf, gf)
        val = sum(∫(Rp * Rp)dΩ)

        @test isapprox(val, 0.0; atol=1e-10)
    end
end
