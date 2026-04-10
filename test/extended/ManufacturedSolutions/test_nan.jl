using Gridap
include("../../../src/models/porosity.jl")
import .Main: SmoothRadialPorosity, grad_alpha, lap_alpha

alpha_field = SmoothRadialPorosity(0.5, 1.0, 0.2, 0.4)
x_vals = range(0.0, stop=1.0, length=1000)

for x_v in x_vals
    for y_v in x_vals
        val = VectorValue(x_v, y_v)
        a, g, l = Main._analyze_alpha(alpha_field, val)
        if isnan(a) || isnan(g[1]) || isnan(g[2]) || isnan(l)
            println("NaN DETECTED at x = $val!")
            exit(1)
        end
    end
end
println("Dense sweep finished.")
