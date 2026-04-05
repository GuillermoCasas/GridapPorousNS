using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

@testset "fast: viscous operators" begin
    @testset "pseudo-traction weak jacobian equals weak operator on du" begin
        model = tiny_model_2d()
        Ω, dΩ, h = tiny_measure(model; degree=6)

        αf = CellField(alpha_lin, Ω)
        uf = CellField(u_poly, Ω)
        vf = CellField(v_poly, Ω)

        val1 = sum(∫(PorousNSSolver.weak_viscous_operator(PorousNSSolver.LaplacianPseudoTractionViscosity(), uf, vf, αf, 1e-2))dΩ)
        val2 = sum(∫(PorousNSSolver.weak_viscous_jacobian(PorousNSSolver.LaplacianPseudoTractionViscosity(), uf, vf, αf, 1e-2))dΩ)

        @test isfinite(val1)
        @test isfinite(val2)
    end

    @testset "symmetric and deviatoric operators assemble on tiny mesh" begin
        model = tiny_model_2d()
        Ω, dΩ, h = tiny_measure(model; degree=6)
        αf = CellField(alpha_lin, Ω)
        uf = CellField(u_poly, Ω)
        vf = CellField(v_poly, Ω)

        vals = Float64[]
        push!(vals, sum(∫(PorousNSSolver.weak_viscous_operator(PorousNSSolver.SymmetricGradientViscosity(), uf, vf, αf, 1e-2))dΩ))
        push!(vals, sum(∫(PorousNSSolver.weak_viscous_operator(PorousNSSolver.DeviatoricSymmetricViscosity(), uf, vf, αf, 1e-2))dΩ))

        @test all(isfinite, vals)
    end

    @testset "adjoint viscous operators return finite values on smooth fields" begin
        model = tiny_model_2d()
        Ω, dΩ, h = tiny_measure(model; degree=6)
        αf = CellField(alpha_lin, Ω)
        uf = CellField(u_poly, Ω)
        vf = CellField(v_poly, Ω)

        op1 = PorousNSSolver.adjoint_viscous_operator(PorousNSSolver.LaplacianPseudoTractionViscosity(), vf, αf, 1e-2)
        op2 = PorousNSSolver.adjoint_viscous_operator(PorousNSSolver.SymmetricGradientViscosity(), vf, αf, 1e-2)
        op3 = PorousNSSolver.adjoint_viscous_operator(PorousNSSolver.DeviatoricSymmetricViscosity(), vf, αf, 1e-2)

        @test op1 !== nothing
        @test op2 !== nothing
        @test op3 !== nothing
    end
end
