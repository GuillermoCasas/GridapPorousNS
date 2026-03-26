using Pkg; Pkg.activate(".")
using Gridap

alpha_field(x) = x[1]^2 + x[2]^2
u_ex(x) = (1.0 / alpha_field(x)) * VectorValue(sin(pi*x[1])*sin(pi*x[2]), cos(pi*x[1])*cos(pi*x[2]))

x_test = Point(1.0, 1.0)
println("alpha: ", alpha_field(x_test))
println("grad_u: ", ∇(u_ex)(x_test))
println("lap_u: ", Δ(u_ex)(x_test))
