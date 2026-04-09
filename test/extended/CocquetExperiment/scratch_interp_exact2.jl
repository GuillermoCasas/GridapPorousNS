using Pkg
Pkg.activate("../../..")
using Gridap
using PorousNSSolver

domain = (0, 2, 0, 1)

println("Running Exact Test With Singularity...")
k = 2
N_ref = 100
model_ref = CartesianDiscreteModel(domain, (2*N_ref, N_ref))
refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, k)

labels_ref = get_face_labeling(model_ref)
add_tag_from_tags!(labels_ref, "inlet", [7])
add_tag_from_tags!(labels_ref, "walls", [1,2,3,4,5,6])

# Let u_exact have a singularity at x=2, y=0 : like r^0.5
u_exact = x -> VectorValue(sqrt((x[1]-2.0)^2 + x[2]^2), 0.0)

V_ref = TestFESpace(model_ref, refe_u, conformity=:H1, labels=labels_ref, dirichlet_tags=["inlet", "walls"])
# The labels don't impose exact properly but we can just interpolate into a free space to trace the mathematical projection properties!
U_ref = TrialFESpace(V_ref, [u_exact, u_exact])

u_ref = interpolate(u_exact, U_ref)
iu_ref = Gridap.FESpaces.Interpolable(u_ref)

dΩ_ref = Measure(Triangulation(model_ref), 2*k+2)

N_list = [10, 20, 40]
for N in N_list
    model_h = CartesianDiscreteModel(domain, (2*N, N))

    V_h_free = TestFESpace(model_h, refe_u, conformity=:H1)
    u_h = interpolate(iu_ref, V_h_free)
    
    dΩ_h = Measure(Triangulation(model_h), 2*k+2)
    
    res = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref)
    l2_cons = res[3]
    println("N=$N  L2: $l2_cons")
end
