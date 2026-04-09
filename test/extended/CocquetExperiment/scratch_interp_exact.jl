using Pkg
Pkg.activate("../../..")
using Gridap
using PorousNSSolver

domain = (0, 2, 0, 1)

println("Running Exact Test...")
k = 2
N_ref = 200
model_ref = CartesianDiscreteModel(domain, (2*N_ref, N_ref))
refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, k)

labels_ref = get_face_labeling(model_ref)
add_tag_from_tags!(labels_ref, "inlet", [7])
add_tag_from_tags!(labels_ref, "walls", [1,2,3,4,5,6])

# Let u_exact be something not perfectly polynomial!
u_exact = x -> VectorValue(sin(pi*x[1])*cos(pi*x[2]), cos(pi*x[1])*sin(pi*x[2]))

V_ref = TestFESpace(model_ref, refe_u, conformity=:H1, labels=labels_ref, dirichlet_tags=["inlet", "walls"])
U_ref = TrialFESpace(V_ref, [u_exact, x->VectorValue(0.0,0.0)])
# U_ref_free = TestFESpace(model_ref, refe_u, conformity=:H1)

# Mimic the true solver: u_ref is the FEM interpolation
u_ref = interpolate(u_exact, U_ref)
iu_ref = Gridap.FESpaces.Interpolable(u_ref)

dΩ_ref = Measure(Triangulation(model_ref), 2*k+2)

N_list = [25, 50, 100]
for N in N_list
    model_h = CartesianDiscreteModel(domain, (2*N, N))
    labels_h = get_face_labeling(model_h)
    add_tag_from_tags!(labels_h, "inlet", [7])
    add_tag_from_tags!(labels_h, "walls", [1,2,3,4,5,6])

    V_h = TestFESpace(model_h, refe_u, conformity=:H1, labels=labels_h, dirichlet_tags=["inlet", "walls"])
    # Mimic the solver boundary conditions Exactly!
    U_h = TrialFESpace(V_h, [u_exact, x->VectorValue(0.0,0.0)])
    
    # Try interpolating to U_h (as in script)
    u_h = interpolate(iu_ref, U_h)
    
    # Consistent metric setup
    V_h_free = TestFESpace(model_h, refe_u, conformity=:H1)
    dΩ_h = Measure(Triangulation(model_h), 2*k+2)
    
    res = PorousNSSolver.compute_reference_errors(u_h, u_ref, iu_ref, V_h_free, dΩ_h, dΩ_ref)
    l2_cons = res[3]
    h1_cons = res[4]
    
    println("N=$N  L2: $l2_cons  H1: $h1_cons")
end
