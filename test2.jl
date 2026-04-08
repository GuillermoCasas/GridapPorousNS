using PorousNSSolver
using Gridap
using LinearAlgebra

model = tiny_model_2d(n=(1,1))
X, Y, V, Q = tiny_spaces_2d(model; kv=1, kp=1)
Ω, dΩ, h = tiny_measure(model; degree=6)

αf = CellField(alpha_lin, Ω)
ff = CellField(f_zero, Ω)
gf = CellField(g_zero, Ω)

form_pseudo = PorousNSSolver.Legacy90d5749Mode(
    PorousNSSolver.ConstantSigmaLaw(1.0),
    PorousNSSolver.ProjectFullResidual(),
    PorousNSSolver.SmoothVelocityFloor(1e-3, 0.5, 1e-8),
    1e-2,
    1e-6
)

x = interpolate_everywhere([u_poly, p_poly], X)

jac_newton(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(
    x, dx, y, form_pseudo, dΩ, h, ff, αf, gf, nothing, nothing, 4.0, 2.0, 1e-8, false, PorousNSSolver.ExactNewtonMode()
)

jac_picard(x, dx, y) = PorousNSSolver.build_picard_jacobian(
    x, dx, y, form_pseudo, dΩ, h, ff, αf, gf, nothing, nothing, 4.0, 2.0, 1e-8
)

println("Timing Picard:")
@time assemble_matrix((dx,y)->jac_picard(x, dx, y), X, Y)
println("Timing Newton:")
@time assemble_matrix((dx,y)->jac_newton(x, dx, y), X, Y)
