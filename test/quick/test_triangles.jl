using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Gridap
# Dynamically link the solver environment to access metrics module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using PorousNSSolver

u_exact(x) = VectorValue(sin(π * x[1]) * cos(π * x[2]), -cos(π * x[1]) * sin(π * x[2]))
p_exact(x) = sin(2π * x[1]) * sin(2π * x[2])

domain = (0.0, 1.0, 0.0, 1.0)
partition_ref = (200, 200)
model_ref = simplexify(CartesianDiscreteModel(domain, partition_ref))

refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
refe_p = ReferenceFE(lagrangian, Float64, 2)

V_ref = TestFESpace(model_ref, refe_u, conformity=:H1)
Q_ref = TestFESpace(model_ref, refe_p, conformity=:H1)

u_ref = interpolate_everywhere(u_exact, V_ref)
p_ref = interpolate_everywhere(p_exact, Q_ref)

iu_ref = Gridap.FESpaces.Interpolable(u_ref)
ip_ref = Gridap.FESpaces.Interpolable(p_ref)

Ω_ref = Triangulation(model_ref)
dΩ_ref = Measure(Ω_ref, 4)

for N in [25, 50, 100]
    partition_h = (N, N)
    model_h = simplexify(CartesianDiscreteModel(domain, partition_h))
    
    V_h = TestFESpace(model_h, refe_u, conformity=:H1)
    
    u_h = interpolate_everywhere(u_exact, V_h)
    
    res_u = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h, Measure(Triangulation(model_h),4), dΩ_ref)
    l2_cons = res_u[3]
    println("N=$N  L2_cons = $l2_cons")
end
