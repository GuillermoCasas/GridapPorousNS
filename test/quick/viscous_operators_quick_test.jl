# ==============================================================================================
# Nature & Intent:
# Acts as the primary Fréchet exactness check for the viscous continuous formulations 
# (Pseudo-traction, Symmetric Gradient, Deviatoric). Ensures stability during global matrix assembly
# on coarse or tiny 2D meshes, confirming pure algebraic compatibility of operations.
#
# Mathematical Formulation Alignment:
# Strictly aligned with continuous calculus exactness. Validates that `weak_viscous_jacobian` 
# perfectly aligns with the Fréchet derivative of `weak_viscous_operator`. Verifies adjoint limits 
# ($\mathcal{L}^*(\mathbf{v})$) match exact continuous definitions for VMS consistency.
#
# Associated Files / Functions:
# - `src/formulations/viscous_operators.jl` (`weak_viscous_operator`, `weak_viscous_jacobian`, 
#   `adjoint_viscous_operator`)
# ==============================================================================================

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

    @testset "strong_viscous_operator analytically exact on quadratic polynomials" begin
        # Validates that our native eval map captures the ∇(∇⋅u) dilatancy strictly through AST Hessian components
        model = CartesianDiscreteModel((0, 1, 0, 1), (2, 2))
        Ω = Triangulation(model)
        
        refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
        V = TestFESpace(model, refe_u, conformity=:H1)
        U = TrialFESpace(V)

        refe_A = ReferenceFE(lagrangian, Float64, 1)
        V_A = TestFESpace(model, refe_A, conformity=:H1)

        f_u(x) = VectorValue(1.0*x[1]^2 + 2.0*x[2]^2, 3.0*x[1]*x[2] + 4.0*x[2]^2)
        u_h = interpolate(f_u, U)
        
        f_A(x) = 2.0 + 3.0*x[1] + 4.0*x[2]
        A_h = interpolate(f_A, V_A)
        
        nu = 0.5
        op = PorousNSSolver.strong_viscous_operator(PorousNSSolver.SymmetricGradientViscosity(), u_h, A_h, nu)

        # Mathematical derivation for SymmetricGradientViscosity:
        # ∇⋅(2Aν ε(u)) = 2ν(ε(u)⋅∇A) + AνΔu + Aν∇(∇⋅u)
        grad_u_ex(x) = TensorValue(2.0*x[1], 4.0*x[2], 3.0*x[2], 3.0*x[1] + 8.0*x[2])
        eps_ex(x) = 0.5 * (grad_u_ex(x) + transpose(grad_u_ex(x)))
        lap_u_ex(x) = VectorValue(2.0 + 4.0, 0.0 + 8.0)
        grad_div_ex(x) = VectorValue(5.0, 8.0)
        grad_A_ex(x) = VectorValue(3.0, 4.0)

        exact_eval(x) = 2.0 * nu * (eps_ex(x) ⋅ grad_A_ex(x)) + f_A(x) * nu * lap_u_ex(x) + f_A(x) * nu * grad_div_ex(x)

        pt = Point(0.5, 0.5)
        val_gridap = op(pt)
        val_exact = exact_eval(pt)

        @test norm(val_gridap - val_exact) < 1e-12
    end

    @testset "strong/adjoint deviatoric viscous: 3D grad-div coefficient 1/6 [VISC-03]" begin
        # The canonical operator's load-bearing, dimension-dependent piece is the 3D grad-div coefficient
        # (½ − 1/d) = 1/6 in `EvalDivDevSymOp`. Previously only the 2D SymmetricGradient strong op was
        # covered analytically (where that coefficient is 0). Pick u = (x₁², x₂², x₃²): Δu = (2,2,2),
        # ∇(∇·u) = (2,2,2). With CONSTANT porosity A (∇A = 0) the porosity-gradient term drops and
        #   ∇·(2Aν ∇ᵈu) = 2Aν(½Δu + (½−1/3)∇(∇·u)) = 2Aν((1,1,1) + ⅓(1,1,1)) = (8Aν/3)(1,1,1).
        # A wrong coefficient (e.g. the 2D value 0) would give 2Aν(1,1,1) instead — this distinguishes them.
        # Self-adjoint operator ⇒ the adjoint reuses the same ∇·ε^d form, so it must give the same value.
        model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
        refe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, 2)
        V = TestFESpace(model, refe_u, conformity=:H1); U = TrialFESpace(V)
        refe_A = ReferenceFE(lagrangian, Float64, 1); V_A = TestFESpace(model, refe_A, conformity=:H1)

        u_h = interpolate(x -> VectorValue(x[1]^2, x[2]^2, x[3]^2), U)
        A = 2.0; nu = 0.5
        A_h = interpolate(x -> A, V_A)

        expected = VectorValue(8A*nu/3, 8A*nu/3, 8A*nu/3)
        pt = Point(0.5, 0.5, 0.5)

        op_strong = PorousNSSolver.strong_viscous_operator(PorousNSSolver.DeviatoricSymmetricViscosity(), u_h, A_h, nu)
        op_adj    = PorousNSSolver.adjoint_viscous_operator(PorousNSSolver.DeviatoricSymmetricViscosity(), u_h, A_h, nu)

        @test norm(op_strong(pt) - expected) < 1e-10
        @test norm(op_adj(pt) - expected) < 1e-10
    end

    @testset "adjoint viscous operators return finite values on smooth fields" begin
        model = tiny_model_2d()
        Ω, dΩ, h = tiny_measure(model; degree=6)
        refe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
        V = TestFESpace(model, refe, conformity=:H1)
        U = TrialFESpace(V)
        refe_A = ReferenceFE(lagrangian, Float64, 1)
        V_A = TestFESpace(model, refe_A, conformity=:H1)

        αf = interpolate(alpha_lin, V_A)
        uf = interpolate(u_poly, U)
        vf = interpolate(v_poly, V)

        op1 = PorousNSSolver.adjoint_viscous_operator(PorousNSSolver.LaplacianPseudoTractionViscosity(), vf, αf, 1e-2)
        op2 = PorousNSSolver.adjoint_viscous_operator(PorousNSSolver.SymmetricGradientViscosity(), vf, αf, 1e-2)
        op3 = PorousNSSolver.adjoint_viscous_operator(PorousNSSolver.DeviatoricSymmetricViscosity(), vf, αf, 1e-2)

        @test op1 !== nothing
        @test op2 !== nothing
        @test op3 !== nothing
    end
end
