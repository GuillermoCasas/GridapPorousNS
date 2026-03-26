# tests/ManufacturedSolutions/run_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PorousNSSolver
using Gridap
using JSON
using DelimitedFiles

function alpha_field(x, config::PorousNSSolver.PorousNSConfig)
    alpha_0 = config.porosity.alpha_0
    r1 = config.porosity.r_1
    r2 = config.porosity.r_2
    dx = x[1] - 1.0
    dy = x[2] - 1.0
    r_sq = dx^2 + dy^2
    r = sqrt(r_sq)
    if r <= r1
        return alpha_0
    elseif r >= r2
        return 1.0
    else
        eta = (r_sq - r1^2) / (r2^2 - r1^2)
        gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
        if gamma_val > 100.0
            return 1.0
        elseif gamma_val < -100.0
            return alpha_0
        end
        return 1.0 - (1.0 - alpha_0) / (1.0 + exp(gamma_val))
    end
end

# Global closures strictly preventing recreation within quadrature loops
u_ex(x, config) = VectorValue(sin(pi*x[1])*sin(pi*x[2]), cos(pi*x[1])*cos(pi*x[2]))
p_ex(x) = cos(pi*x[1])*sin(pi*x[2])

function analyze_alpha(x, config)
    alpha_0 = config.porosity.alpha_0
    r1 = config.porosity.r_1
    r2 = config.porosity.r_2
    dx = x[1] - 1.0
    dy = x[2] - 1.0
    r_sq = dx^2 + dy^2
    r = sqrt(r_sq)
    
    if r <= r1 || r >= r2
        return alpha_field(x, config), VectorValue(0.0, 0.0), 0.0
    end
    
    eta = (r_sq - r1^2) / (r2^2 - r1^2)
    gamma_val = (2.0*eta - 1.0) / (eta * (1.0 - eta))
    
    # Float64 double-precision asymptote cutoff to prevent NaN from Inf/Inf
    if gamma_val > 100.0 || gamma_val < -100.0
        return alpha_field(x, config), VectorValue(0.0, 0.0), 0.0
    end
    
    exp_g = exp(gamma_val)
    alpha_val = 1.0 - (1.0 - alpha_0) / (1.0 + exp_g)
    
    deta_dr = 2.0 * r / (r2^2 - r1^2)
    d2eta_dr2 = 2.0 / (r2^2 - r1^2)
    
    dgamma_deta = (2.0*eta^2 - 2.0*eta + 1.0) / (eta^2 * (1.0 - eta)^2)
    d2gamma_deta2 = (4.0*eta - 2.0) / (eta^2 * (1.0 - eta)^2) - 2.0 * (2.0*eta^2 - 2.0*eta + 1.0)*(1.0 - 2.0*eta) / (eta^3 * (1.0 - eta)^3)
    
    da_dg = (1.0 - alpha_0) * exp_g / (1.0 + exp_g)^2
    d2a_dg2 = (1.0 - alpha_0) * exp_g * (1.0 - exp_g) / (1.0 + exp_g)^3
    
    da_dr = da_dg * dgamma_deta * deta_dr
    d2a_dr2 = d2a_dg2 * (dgamma_deta * deta_dr)^2 + da_dg * (d2gamma_deta2 * deta_dr^2 + dgamma_deta * d2eta_dr2)
    
    lap_alpha = d2a_dr2 + (1.0/r) * da_dr
    grad_alpha = VectorValue(da_dr * (dx/r), da_dr * (dy/r))
    return alpha_val, grad_alpha, lap_alpha
end

function grad_u_ex(x, config)
    return TensorValue(
        pi * cos(pi*x[1])*sin(pi*x[2]), pi * sin(pi*x[1])*cos(pi*x[2]),
        -pi * sin(pi*x[1])*cos(pi*x[2]), -pi * cos(pi*x[1])*sin(pi*x[2])
    )
end

function lap_u_ex(x, config)
    return -2.0 * pi^2 * u_ex(x, config)
end

function grad_p_ex(x)
    return VectorValue(-pi * sin(pi*x[1])*sin(pi*x[2]), pi * cos(pi*x[1])*cos(pi*x[2]))
end

# Analytical Forcing term
function f_ex(x, config::PorousNSSolver.PorousNSConfig)
    α_val, grad_alpha_val, lap_alpha_val = analyze_alpha(x, config)
    Re = config.phys.Re
    ν = 1.0 / Re
    Da = config.phys.Da
    
    u_ex_val = u_ex(x, config)
    grad_u_val = grad_u_ex(x, config)
    lap_u_val = lap_u_ex(x, config)
    grad_p_val = grad_p_ex(x)
    
    conv_u_val = grad_u_val ⋅ u_ex_val
    div_stress = α_val * ν * lap_u_val + ν * (grad_u_val + transpose(grad_u_val)) ⋅ grad_alpha_val
    
    a_term = PorousNSSolver.a_resistance(α_val, Re, Da)
    b_term = PorousNSSolver.b_resistance(α_val)
    σ_val = a_term + b_term * norm(u_ex_val)
    
    return α_val * conv_u_val - div_stress + α_val * grad_p_val + σ_val * u_ex_val
end

function run_mms()
    config_path = joinpath(@__DIR__, "data", "test_config.json")
    base_config = PorousNSSolver.load_config(config_path)
    
    partitions = base_config.mesh.convergence_partitions
    hs = Float64[]
    err_u_L2 = Float64[]
    err_p_L2 = Float64[]
    err_u_H1 = Float64[]
    err_p_H1 = Float64[]
    
    for n in partitions
        println("Running MMS for partition $n x $n")
        
        config = PorousNSSolver.PorousNSConfig(
            phys=base_config.phys,
            porosity=base_config.porosity,
            discretization=base_config.discretization,
            mesh=PorousNSSolver.MeshConfig(domain=base_config.mesh.domain, partition=[n, n]),
            output=PorousNSSolver.OutputConfig(directory=joinpath(@__DIR__, "results"), basename="mms_$n")
        )
        
        model = PorousNSSolver.create_mesh(config)
        
        # For MMS, all boundaries are Dirichlet for velocity
        labels = get_face_labeling(model)
        add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])
        
        refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, config.discretization.k_velocity)
        refe_p = ReferenceFE(lagrangian, Float64, config.discretization.k_pressure)
        
        V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
        Q = TestFESpace(model, refe_p, conformity=:H1)
        
        u_final(x) = u_ex(x, config)
        U = TrialFESpace(V, u_final)  
        # Fix pressure at one point to remove nullspace, actually we can just use zero mean pressure or fix it on one node, 
        # but Gridap allows fixing one node by modifying Q.
        # Alternatively, evaluate L2 error using the mean subtracted.
        # Let's fix pressure at the bottom left corner (tag 1).
        Q_fixed = TestFESpace(model, refe_p, conformity=:H1, labels=labels, dirichlet_tags=[1])
        P = TrialFESpace(Q_fixed, p_ex)
        
        Y = MultiFieldFESpace([V, Q_fixed])
        X = MultiFieldFESpace([U, P])
        
        degree = 2 * config.discretization.k_velocity
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree)
        
        h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
        h = CellField(h_array, Ω)
        
        f(x) = f_ex(x, config)
        alpha_fn(x) = alpha_field(x, config)
        
        println("Nodally interpolating alpha and f...")
        alpha_h = interpolate_everywhere(alpha_fn, Q)
        f_h = interpolate_everywhere(f, V)
        
        println("Defining formulation closures...")
        res(x, y) = PorousNSSolver.weak_form_residual(x, y, config, dΩ, h, f_h, alpha_h)
        jac(x, dx, y) = PorousNSSolver.weak_form_jacobian(x, dx, y, config, dΩ, h, f_h, alpha_h)
        
        println("Creating FEOperator...")
        op = FEOperator(res, jac, X, Y)
        
        nls = NLSolver(show_trace=true, method=:newton, iterations=10)
        solver = FESolver(nls)
        
        println("Solving non-linear system...")
        x_h = solve(solver, op)
        u_h, p_h = x_h
        
        # Compute errors
        e_u = u_final - u_h
        e_p = p_ex - p_h
        
        PorousNSSolver.export_results(config, model, u_h, p_h,
            "u_ex" => u_final,
            "p_ex" => p_ex,
            "e_u" => e_u,
            "e_p" => e_p,
            "alpha" => alpha_fn)
        
        el2_u = sqrt(sum(∫(e_u ⋅ e_u)dΩ))
        el2_p = sqrt(sum(∫(e_p * e_p)dΩ))
        
        eh1_semi_u = sqrt(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ))
        eh1_semi_p = sqrt(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ))
        eh1_u = sqrt(el2_u^2 + eh1_semi_u^2)
        eh1_p = sqrt(el2_p^2 + eh1_semi_p^2)
        
        push!(hs, 1.0 / n)
        push!(err_u_L2, el2_u)
        push!(err_p_L2, el2_p)
        push!(err_u_H1, eh1_u)
        push!(err_p_H1, eh1_p)
        println("L2 Error u: $el2_u, p: $el2_p")
        println("H1 Error u: $eh1_u, p: $eh1_p")
    end
    
    # Data to file
    open(joinpath(@__DIR__, "results", "errors.csv"), "w") do io
        writedlm(io, ["h" "err_u_l2" "err_p_l2" "err_u_h1" "err_p_h1"])
        writedlm(io, [hs err_u_L2 err_p_L2 err_u_H1 err_p_H1])
    end
    
    # Calculate rates
    compute_slope(x, y) = sum((x .- sum(x)/length(x)) .* (y .- sum(y)/length(y))) / sum((x .- sum(x)/length(x)).^2)
    log_h = log.(hs)
    rate_u_l2 = compute_slope(log_h, log.(err_u_L2))
    rate_u_h1 = compute_slope(log_h, log.(err_u_H1))
    rate_p_l2 = compute_slope(log_h, log.(err_p_L2))
    rate_p_h1 = compute_slope(log_h, log.(err_p_H1))
    
    # Validate against theory
    k = base_config.discretization.k_velocity
    theo_u_l2 = float(k + 1)
    theo_u_h1 = float(k)
    theo_p_l2 = float(k)
    theo_p_h1 = float(max(k - 1, 0)) # Not explicitly requested to validate, but kept for completeness
    
    summary_file = joinpath(@__DIR__, "results", "summary.txt")
    open(summary_file, "w") do io
        println(io, "=== Convergence Summary ===")
        println(io, "Variable & Norm | Empirical Rate | Theoretical Rate")
        println(io, "----------------|----------------|-----------------")
        println(io, "Velocity L2     | $(round(rate_u_l2, digits=2))           | $(theo_u_l2)")
        println(io, "Velocity H1     | $(round(rate_u_h1, digits=2))           | $(theo_u_h1)")
        println(io, "Pressure L2     | $(round(rate_p_l2, digits=2))           | $(theo_p_l2)")
        println(io, "Pressure H1     | $(round(rate_p_h1, digits=2))           | $(theo_p_h1)")
    end
    
    println("MMS setup complete. Rates saved to summary.txt")
end

run_mms()
