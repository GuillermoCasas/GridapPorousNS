using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Gridap

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain, partition)
refe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
V = TestFESpace(model, refe, conformity=:H1)
U = TrialFESpace(V)

Ω = Triangulation(model)
dΩ = Measure(Ω, 2)

u_h = interpolate_everywhere(VectorValue(1.0, 2.0), U)

# Test if Laplacian works on P1
lap_u = sum(∫( Δ(u_h) ⋅ Δ(u_h) )dΩ)
println("Laplacian of P1: ", lap_u)

