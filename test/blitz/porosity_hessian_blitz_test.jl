# test/blitz/porosity_hessian_blitz_test.jl
# ==============================================================================================
# Nature & Intent:
# Verifies the analytical second-derivative extensions used by the manufactured-solution forcing:
#   1. `hess_alpha` — the exact Hessian ∇²α of the smooth radial porosity field.
#   2. `grad_div_u_ex` — the exact ∇(∇·u_ex), which previously was evaluated by finite differences.
# Both are checked against high-accuracy central finite differences of the (already trusted)
# first-derivative routines `grad_alpha` and `grad_u_ex`, plus internal consistency
# (trace(∇²α) == Δα, symmetry of ∇²α, and vanishing outside the porosity transition annulus).
#
# Associated Files / Functions:
# - `src/models/porosity.jl` (`grad_alpha`, `lap_alpha`, `hess_alpha`)
# - `src/problems/mms_paper.jl` (`grad_u_ex`, `grad_div_u_ex`)
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

@testset "Analytic ∇²α and ∇(∇·u) match finite differences" begin
    # Radial porosity with a genuine transition annulus r ∈ (0.2, 0.4); alpha_0 < 1 so the
    # field (hence ∇α, ∇²α) is non-trivial.
    field = PorousNSSolver.SmoothRadialPorosity(0.3, 1.0, 0.2, 0.4)
    h = 1.0e-5
    rtol = 1.0e-4

    e1 = VectorValue(1.0, 0.0)
    e2 = VectorValue(0.0, 1.0)

    # Points strictly inside the transition annulus (radii ≈ 0.25–0.32), across quadrants.
    annulus_pts = [VectorValue(0.25, 0.0), VectorValue(0.2, 0.2),
                   VectorValue(0.18, 0.24), VectorValue(-0.15, 0.25)]

    for x in annulus_pts
        gA(p) = PorousNSSolver.grad_alpha(field, p)

        # Central-difference Hessian from the analytic gradient.
        Axx_fd = (gA(x + h*e1)[1] - gA(x - h*e1)[1]) / (2h)
        Ayy_fd = (gA(x + h*e2)[2] - gA(x - h*e2)[2]) / (2h)
        Axy_fd = 0.5 * ((gA(x + h*e2)[1] - gA(x - h*e2)[1]) / (2h) +
                        (gA(x + h*e1)[2] - gA(x - h*e1)[2]) / (2h))

        H = PorousNSSolver.hess_alpha(field, x)
        @test isapprox(H[1,1], Axx_fd; rtol=rtol, atol=1e-8)
        @test isapprox(H[2,2], Ayy_fd; rtol=rtol, atol=1e-8)
        @test isapprox(H[1,2], Axy_fd; rtol=rtol, atol=1e-8)

        # Symmetry, and trace consistency with the independently-derived Laplacian.
        @test H[1,2] == H[2,1]
        @test isapprox(H[1,1] + H[2,2], PorousNSSolver.lap_alpha(field, x); rtol=1e-12, atol=1e-12)
    end

    # Outside the annulus the field is constant ⇒ ∇²α ≡ 0 (and ∇α ≡ 0).
    for x in (VectorValue(0.05, 0.05), VectorValue(0.45, 0.0))
        H = PorousNSSolver.hess_alpha(field, x)
        @test all(isapprox.(Tuple(H), 0.0; atol=1e-14))
    end

    # ---- Exact ∇(∇·u_ex) vs. central FD of tr(∇u_ex) -------------------------------------
    uex = PorousNSSolver.UExFunc(1.0, field.alpha_0, field)   # U = 1, α₀ = 0.3
    divu(p) = tr(PorousNSSolver.grad_u_ex(uex, p))

    for x in annulus_pts
        gdx_fd = (divu(x + h*e1) - divu(x - h*e1)) / (2h)
        gdy_fd = (divu(x + h*e2) - divu(x - h*e2)) / (2h)
        gd = PorousNSSolver.grad_div_u_ex(uex, x)
        @test isapprox(gd[1], gdx_fd; rtol=rtol, atol=1e-7)
        @test isapprox(gd[2], gdy_fd; rtol=rtol, atol=1e-7)
    end
end
