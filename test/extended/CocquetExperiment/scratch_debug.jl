using Pkg
Pkg.activate("../../..")
using Gridap
using PorousNSSolver

domain = (0, 2, 0, 1)

println("Running Debug Test...")
k = 2
N_ref = 20
model_ref = CartesianDiscreteModel(domain, (2*N_ref, N_ref))
refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, k)
V_ref = TestFESpace(model_ref, refe_u, conformity=:H1)

u_exact = x -> VectorValue(sin(pi*x[1])*cos(pi*x[2]), cos(pi*x[1])*sin(pi*x[2]))
u_ref = interpolate(u_exact, V_ref)
iu_ref = Gridap.FESpaces.Interpolable(u_ref)
dΩ_ref = Measure(Triangulation(model_ref), 2*k+2)

N = 10
model_h = CartesianDiscreteModel(domain, (2*N, N))
V_h_free = TestFESpace(model_h, refe_u, conformity=:H1)

# Method A: straight interpolation like scratch_convergence.jl
u_h1 = interpolate(iu_ref, V_h_free)
dΩ_h = Measure(Triangulation(model_h), 2*k+2)
res1 = PorousNSSolver.compute_reference_errors(u_h1, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref)
println("Method A L2: ", res1[3], " H1: ", res1[4])

# Method B: with dirichlet labels but no boundaries enforced ?
labels_h = get_face_labeling(model_h)
add_tag_from_tags!(labels_h, "inlet", [7])
add_tag_from_tags!(labels_h, "walls", [1,2,3,4,5,6])
V_h_labeled = TestFESpace(model_h, refe_u, conformity=:H1, labels=labels_h, dirichlet_tags=["inlet", "walls"])
U_h_labeled = TrialFESpace(V_h_labeled, [u_exact, x->VectorValue(0.0,0.0)])
u_h2 = interpolate(iu_ref, U_h_labeled)
res2 = PorousNSSolver.compute_reference_errors(u_h2, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref)
println("Method B L2: ", res2[3], " H1: ", res2[4])

