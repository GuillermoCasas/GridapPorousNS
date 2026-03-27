using Gridap

domain = (0, 1, 0, 1)
model_h = CartesianDiscreteModel(domain, (10, 10))
model_ref = CartesianDiscreteModel(domain, (20, 20))

refe_u = ReferenceFE(lagrangian, Float64, 1)

V_h = TestFESpace(model_h, refe_u)
U_h = TrialFESpace(V_h)
u_h = interpolate_everywhere(x -> x[1], U_h)

V_ref = TestFESpace(model_ref, refe_u)
U_ref = TrialFESpace(V_ref)
u_ref = interpolate_everywhere(x -> x[1] + 1.0, U_ref)

degree = 2
Ω_h = Triangulation(model_h)
dΩ_h = Measure(Ω_h, degree)

println("Attempting cross-mesh analytical mapping...")
# Wrap the reference CellField natively into a geographical analytical mapping
u_ref_analytic(x) = u_ref(x)

# Evaluate against coarse spaces
u_ref_interp = interpolate_everywhere(u_ref_analytic, U_h)

eh = u_h - u_ref_interp
l2_err = sqrt(sum(∫(eh * eh) * dΩ_h))
println("Error: $l2_err")
