using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Gridap
# Dynamically link the solver environment to access metrics module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using PorousNSSolver
u_exact(x) = VectorValue(sin(π * x[1]) * cos(π * x[2]), -cos(π * x[1]) * sin(π * x[2]))

N_ref = 200
partition_ref = (N_ref, N_ref)
domain = (0.0, 1.0, 0.0, 1.0)
model_ref = CartesianDiscreteModel(domain, partition_ref)
refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
V_ref = TestFESpace(model_ref, refe_u, conformity=:H1)
u_ref = interpolate_everywhere(u_exact, V_ref)
iu_ref = Gridap.FESpaces.Interpolable(u_ref)

Ω_ref = Triangulation(model_ref)
dΩ_ref = Measure(Ω_ref, 4)

for N in [25, 50, 100]
    partition_h = (N, N)
    model_h = CartesianDiscreteModel(domain, partition_h)
    V_h = TestFESpace(model_h, refe_u, conformity=:H1)
    
    # Method A: interpolate from analytical (optimal convergence)
    u_h_A = interpolate_everywhere(u_exact, V_h)
    
    # Method B: interpolate from the fine grid function (like run_convergence bypass)
    u_h_B = interpolate(iu_ref, V_h)
    
    l2_A = sqrt(sum(∫( (u_ref - Gridap.FESpaces.Interpolable(u_h_A)) ⊙ (u_ref - Gridap.FESpaces.Interpolable(u_h_A)) ) * dΩ_ref))
    l2_B = sqrt(sum(∫( (u_ref - Gridap.FESpaces.Interpolable(u_h_B)) ⊙ (u_ref - Gridap.FESpaces.Interpolable(u_h_B)) ) * dΩ_ref))
    
    println("N=$N  L2_A (from analytical) = $l2_A")
    println("N=$N  L2_B (from u_ref)      = $l2_B")
end
