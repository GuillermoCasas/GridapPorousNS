using Pkg
Pkg.activate("../../..")
using Gridap
using PorousNSSolver

domain = (0, 2, 0, 1)

function test_interpolation(k)
    println("--- Testing order k = $k ---")
    u_exact = x -> VectorValue(sin(pi*x[1])*cos(pi*x[2]), cos(pi*x[1])*sin(pi*x[2]))
    
    N_ref = 100
    model_ref = CartesianDiscreteModel(domain, (2*N_ref, N_ref))
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, k)
    V_ref = TestFESpace(model_ref, refe_u, conformity=:H1)
    
    u_ref = interpolate(u_exact, V_ref)
    iu_ref = Gridap.FESpaces.Interpolable(u_ref)
    
    dΩ_ref = Measure(Triangulation(model_ref), 2*k+2)
    
    N_list = [10, 20, 40]
    for N in N_list
        model_h = CartesianDiscreteModel(domain, (2*N, N))
        V_h = TestFESpace(model_h, refe_u, conformity=:H1)
        
        u_h = interpolate(iu_ref, V_h)
        
        dΩ_h = Measure(Triangulation(model_h), 2*k+2)
        res = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h, dΩ_h, dΩ_ref)
        l2_cons = res[3]
        h1_cons = res[4]
        
        println("N=$N  L2: $l2_cons  H1: $h1_cons")
    end
end

test_interpolation(1)
test_interpolation(2)
