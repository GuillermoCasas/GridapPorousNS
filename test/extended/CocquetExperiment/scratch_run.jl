using Pkg
Pkg.activate("../../..")
using Gridap
using PorousNSSolver

domain = (0, 2, 0, 1)

k = 2
N_ref = 100
model_ref = CartesianDiscreteModel(domain, (2*N_ref, N_ref))
refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, k)

labels_ref = get_face_labeling(model_ref)
add_tag_from_tags!(labels_ref, "inlet", [7])
add_tag_from_tags!(labels_ref, "walls", [1,2,3,4,5,6])

u_in = x -> VectorValue(0.5 * x[2] * (1.0 - x[2]), 0.0)
u_wall = x -> VectorValue(0.0, 0.0)

# Simulate u_ref having smooth interior but exact boundary
u_exact = x -> VectorValue(0.5 * x[2] * (1.0 - x[2]) * sin(pi*x[1]/2), 0.0)

V_ref = TestFESpace(model_ref, refe_u, conformity=:H1, labels=labels_ref, dirichlet_tags=["inlet", "walls"])
U_ref = TrialFESpace(V_ref, [u_in, u_wall])

# interpolate into U_ref sets the boundary to u_in EXACTLY
u_ref = interpolate(u_exact, U_ref)
iu_ref = Gridap.FESpaces.Interpolable(u_ref)

dΩ_ref = Measure(Triangulation(model_ref), 2*k+2)

N_list = [25, 50]
for N in N_list
    model_h = CartesianDiscreteModel(domain, (2*N, N))
    labels_h = get_face_labeling(model_h)
    add_tag_from_tags!(labels_h, "inlet", [7])
    add_tag_from_tags!(labels_h, "walls", [1,2,3,4,5,6])

    V_h = TestFESpace(model_h, refe_u, conformity=:H1, labels=labels_h, dirichlet_tags=["inlet", "walls"])
    U_h = TrialFESpace(V_h, [u_in, u_wall])
    
    # Method 1: interpolate into U_h
    u_h1 = interpolate(iu_ref, U_h)
    
    # Method 2: interpolate into V_h_free
    V_h_free = TestFESpace(model_h, refe_u, conformity=:H1)
    u_h2 = interpolate(iu_ref, V_h_free)
    
    dΩ_h = Measure(Triangulation(model_h), 2*k+2)
    
    res1 = PorousNSSolver.compute_reference_errors(u_h1, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref)
    res2 = PorousNSSolver.compute_reference_errors(u_h2, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref)
    
    println("N=$N U_h   L2: ", res1[3], " (nested: ", res1[1], ")")
    println("N=$N V_free L2: ", res2[3], " (nested: ", res2[1], ")")
end
