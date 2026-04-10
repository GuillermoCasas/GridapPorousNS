using Gridap
include("../../../src/models/porosity.jl")
include("../../../src/problems/mms_paper_2d.jl")

using .Main.PorousNSSolver: Paper2DMMS, SmoothRadialPorosity, get_u_ex, get_p_ex, UExFunc, grad_u_ex, lap_u_ex

model = CartesianDiscreteModel((0, 1, 0, 1), (10, 10))
Ω = Triangulation(model)

α_f = SmoothRadialPorosity(0.5, 1.0, 0.2, 0.4)
x_test = VectorValue(0.5, 0.5)

u_f = UExFunc(1.0, 0.5, α_f)
grad_u = grad_u_ex(u_f, x_test)
lap_u = lap_u_ex(u_f, x_test)

A = α_f(x_test)
grad_A = Main.PorousNSSolver.grad_alpha(α_f, x_test)

eps_u = 0.5 * (grad_u + transpose(grad_u))
try
    eps_dot_grad_A = eps_u ⋅ grad_A
    println("Dot product successful: ", eps_dot_grad_A)
catch e
    println("Dot product failed: ", e)
end
