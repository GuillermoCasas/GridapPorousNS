# test/quick/test_interpolation_projection.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Gridap
using LinearAlgebra
using Printf

# Dynamically link the solver environment to access metrics module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using PorousNSSolver

function run_interpolation_diagnostic()
    println("--- Gridap Topological Projection Diagnostic ---")
    
    # 1. Define a smooth analytical reference function mimicking complex resolved physics
    u_exact(x) = VectorValue(sin(π * x[1]) * cos(π * x[2]), -cos(π * x[1]) * sin(π * x[2]))
    p_exact(x) = sin(2π * x[1]) * sin(2π * x[2])
    
    # 2. Define the FINE reference grid.
    N_ref = 200
    domain = (0.0, 2.0, 0.0, 1.0)
    
    println("Building Reference Mesh N=$N_ref...")
    mod_ref = CartesianDiscreteModel(domain, (2*N_ref, N_ref))
    Ω_ref = Triangulation(mod_ref)
    
    for k in [1, 2]
        println("\n==========================================================================================")
        println("   TESTING EQUAL-ORDER P$(k) / P$(k)")
        println("==========================================================================================")
        
        refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, k)
        refe_p = ReferenceFE(lagrangian, Float64, k)
        
        V_ref = TestFESpace(mod_ref, refe_u, conformity=:H1)
        Q_ref = TestFESpace(mod_ref, refe_p, conformity=:H1)
        
        # Nodally interpolate exact analytical parameters onto the highest resolution FESpace
        u_ref = interpolate_everywhere(u_exact, V_ref)
        p_ref = interpolate_everywhere(p_exact, Q_ref)
        
        # Wrap in native topological projector
        iu_ref = Gridap.FESpaces.Interpolable(u_ref)
        ip_ref = Gridap.FESpaces.Interpolable(p_ref)

        N_list = [25, 50, 100]
        
        println("Running convergence sequence mapping [25, 50, 100] -> $N_ref...")
        println("--------------------------------------------------------------------------------------------------------------------------")
        println("  N  |  Nested Projection (u_ref_proj - u_h) | True Interpolation (u_h - u_exact) | Consistent L2-Norm (u_ref - u_h) ")
        println("--------------------------------------------------------------------------------------------------------------------------")
        
        errors_l2_u_exact = Float64[]
        errors_l2_u_nested = Float64[]
        errors_l2_u_consistent = Float64[]
        
        dΩ_ref = Measure(Ω_ref, 4*k)
        
        for N in N_list
            mod_h = CartesianDiscreteModel(domain, (2*N, N))
            Ω_h = Triangulation(mod_h)
            dΩ_h = Measure(Ω_h, 4*k) # exact quadrature scaling (2k * 2)
            
            V_h = TestFESpace(mod_h, refe_u, conformity=:H1)
            Q_h = TestFESpace(mod_h, refe_p, conformity=:H1)
            
            # Build the exact counterpart "calculated numerical solution" using nodal interpolation
            u_h = interpolate_everywhere(u_exact, V_h)
            p_h = interpolate_everywhere(p_exact, Q_h)
            
            # ---------------------------------------------------------------------
            # Exact metric encapsulation
            # ---------------------------------------------------------------------
            res_u = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h, dΩ_h, dΩ_ref)
            
            # Since the native test focuses on velocity mapping validation, we primarily print L2(u)
            l2_nested = res_u[1]
            l2_consistent = res_u[3]
            
            # 2. True Analytical Interpolation 
            # (Validates Gridap evaluates purely against true exact analytical target flawlessly)
            eu_exact = u_h - u_exact
            l2_exact = sqrt(sum(∫( eu_exact ⋅ eu_exact ) * dΩ_h))
            
            push!(errors_l2_u_nested, l2_nested)
            push!(errors_l2_u_exact, l2_exact)
            push!(errors_l2_u_consistent, l2_consistent)
            
            @printf("%4d | %35.6e | %34.6e | %34.6e\n", N, l2_nested, l2_exact, l2_consistent)
        end
        
        println("--------------------------------------------------------------------------------------------------------------------------")
        
        # Prove mathematical scaling explicitly via Log-Log limits
        println("Slope Extrapolation (Testing geometric scaling limits):")
        target_k = k + 1.0 # O(h^{k+1}) for L2 error
        for i in 1:(length(N_list)-1)
            h1 = 1.0 / N_list[i]
            h2 = 1.0 / N_list[i+1]
            
            e1 = errors_l2_u_consistent[i]
            e2 = errors_l2_u_consistent[i+1]
            
            slope = (log(e2) - log(e1)) / (log(h2) - log(h1))
            @printf("  -> N(%3d -> %3d): %.3f (Target: %.1f for O(h^%d))\n", N_list[i], N_list[i+1], slope, target_k, k+1)
        end
    end
end

run_interpolation_diagnostic()
