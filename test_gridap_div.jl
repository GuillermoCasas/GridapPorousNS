using Pkg; Pkg.activate(".")
using Gridap

alpha_field(x) = x[1]^2 + x[2]^2
u_ex(x) = (1.0 / alpha_field(x)) * VectorValue(sin(pi*x[1]), cos(pi*x[2]))
stress(x) = alpha_field(x) * 1.0 * (∇(u_ex)(x) + transpose(∇(u_ex)(x)))

x_test = Point(1.0, 1.0)
try
    println("div_stress: ", (∇⋅stress)(x_test))
catch e
    println("ERROR: ", e)
end
