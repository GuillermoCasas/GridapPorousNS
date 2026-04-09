using Printf

# Imagine a 1D domain [0, 1] with an exponential boundary layer at x=1
delta = 0.01
f_exact(x) = exp(-(1.0 - x) / delta)

# We define a P2 element over interval [a, b]. It interpolates at a, (a+b)/2, b
function get_p2_interpolant(x, a, b, f_a, f_mid, f_b)
    h = b - a
    xi = (x - a) / h  # maps to [0, 1]
    # P2 basis functions on [0,1] with nodes 0, 0.5, 1
    N1 = 2.0 * (xi - 0.5) * (xi - 1.0)
    N2 = -4.0 * xi * (xi - 1.0)
    N3 = 2.0 * xi * (xi - 0.5)
    return f_a * N1 + f_mid * N2 + f_b * N3
end

function compute_l2_error(N)
    h = 1.0 / N
    err2 = 0.0
    # Integral over each element using 10-point midpoint or simple trapz approximation heavily refined
    for i in 1:N
        a = (i-1)*h
        b = i*h
        f_a = f_exact(a)
        f_mid = f_exact(a + h/2)
        f_b = f_exact(b)
        
        # Integrate (f_exact - P2)^2 over [a,b]
        for step in 1:100
            x = a + (step - 0.5) * (h / 100)
            diff = f_exact(x) - get_p2_interpolant(x, a, b, f_a, f_mid, f_b)
            err2 += diff^2 * (h / 100)
        end
    end
    return sqrt(err2)
end

Ns = [10, 50, 100, 200, 400]
errs = [compute_l2_error(n) for n in Ns]
println("N values: ", Ns)
println("L2 errors: ", errs)
for i in 1:length(Ns)-1
    rate = log(errs[i]/errs[i+1]) / log(Ns[i+1]/Ns[i])
    @printf("Rate N=%d to %d: %.3f\n", Ns[i], Ns[i+1], rate)
end
