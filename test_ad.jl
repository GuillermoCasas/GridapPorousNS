using Pkg
Pkg.activate(".")
using ForwardDiff

function alpha_field(x)
    alpha_0 = 0.5
    r1 = 0.2
    r2 = 0.5
    r = sqrt((x[1]-1.0)^2 + (x[2]-1.0)^2)
    if r <= r1
        return alpha_0
    elseif r >= r2
        return 1.0
    else
        eta = (r^2 - r1^2) / (r2^2 - r1^2)
        gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
        return 1.0 - (1.0 - alpha_0) / (1.0 + exp(gamma_val))
    end
end

println("r = 0.1: ", ForwardDiff.gradient(alpha_field, [1.1, 1.0]))
println("r = 0.2: ", ForwardDiff.gradient(alpha_field, [1.2, 1.0]))
println("r = 0.2001: ", ForwardDiff.gradient(alpha_field, [1.2001, 1.0]))
println("r = 0.35: ", ForwardDiff.gradient(alpha_field, [1.35, 1.0]))
println("r = 0.5: ", ForwardDiff.gradient(alpha_field, [1.5, 1.0]))
println("r = 0.5001: ", ForwardDiff.gradient(alpha_field, [1.5001, 1.0]))
println("r = 0.6: ", ForwardDiff.gradient(alpha_field, [1.6, 1.0]))
