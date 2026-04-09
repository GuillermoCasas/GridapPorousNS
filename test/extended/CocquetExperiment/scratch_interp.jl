using Gridap
domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain, partition)
V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
uh = interpolate(x->x[1]*x[2], V)
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)
# Is Interpolable(∇(uh)) valid?
try
    ih = Gridap.FESpaces.Interpolable(∇(uh))
    println("Interpolable(∇(uh)) works")
catch e
    println("Interpolable(∇(uh)) failed: ", typeof(e))
end
