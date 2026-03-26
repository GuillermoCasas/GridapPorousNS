using Pkg; Pkg.activate(".")
import Gridap.ForwardDiff

function alpha_field(x)
    return x[1]^2 + x[2]^2
end

println("Gridap ForwardDiff works: ", ForwardDiff.gradient(alpha_field, [1.0, 2.0]))
