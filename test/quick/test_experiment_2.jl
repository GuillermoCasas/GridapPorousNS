using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Gridap

domain = (0.0, 2.0, 0.0, 1.0)
N_ref = 200
model_ref = CartesianDiscreteModel(domain, (2*N_ref, N_ref))
refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)

u_in(x) = VectorValue(1.0 * x[2] * (1.0 - x[2]), 0.0)
u_wall(x) = VectorValue(0.0, 0.0)

labels_ref = get_face_labeling(model_ref)
V_ref = TestFESpace(model_ref, refe_u, conformity=:H1, labels=labels_ref, dirichlet_tags=["inlet", "walls"])
U_ref = TrialFESpace(V_ref, [u_in, u_wall])

# Create a highly chaotic "fake" u_ref that is fully P2 across the grid but satisfies BCs.
# Let's just solve the Poisson problem to get a real valid P2 u_ref.
degree = 4
Ω_ref = Triangulation(model_ref)
dΩ_ref = Measure(Ω_ref, degree)
a(u,v) = ∫( ∇(v) ⊙ ∇(u) )*dΩ_ref
l(v) = ∫( v ⊙ VectorValue(1.0, 1.0) )*dΩ_ref
op = AffineFEOperator(a, l, U_ref, V_ref)
u_ref = solve(op)
iu_ref = Gridap.FESpaces.Interpolable(u_ref)

for N in [25, 50, 100]
    model_h = CartesianDiscreteModel(domain, (2*N, N))
    labels_h = get_face_labeling(model_h)
    V_h = TestFESpace(model_h, refe_u, conformity=:H1, labels=labels_h, dirichlet_tags=["inlet", "walls"])
    U_h = TrialFESpace(V_h, [u_in, u_wall])
    
    # Try interpolating:
    u_h = interpolate(iu_ref, U_h)
    u_h_V = interpolate(iu_ref, V_h)  # without forced BC test
    
    Ω_ref_h = Triangulation(model_ref) # Always evaluate error on ref grid measure
    dΩ_ref_h = Measure(Ω_ref_h, degree)
    
    e = u_ref - Gridap.FESpaces.Interpolable(u_h)
    l2 = sqrt(sum(∫( e ⊙ e )*dΩ_ref))
    println("N=$N, L2 error (Trial) = $l2")
end
