using Gridap

domain = (0, 1, 0, 1)
model_h = CartesianDiscreteModel(domain, (40, 40))
model_ref = CartesianDiscreteModel(domain, (200, 200))

refe_u = ReferenceFE(lagrangian, Float64, 1)

V_h = TestFESpace(model_h, refe_u)
U_h = TrialFESpace(V_h)
u_h = interpolate_everywhere(x -> x[1], U_h)

V_ref = TestFESpace(model_ref, refe_u)
U_ref = TrialFESpace(V_ref)
u_ref = interpolate_everywhere(x -> x[1] + 1.0, U_ref)

degree = 2
dΩ_ref = Measure(Triangulation(model_ref), degree)

println("Executing optimized Gridap Batched Evaluation...")
@time begin
    pts_ref = get_cell_points(dΩ_ref)
    dv_h = u_h(pts_ref)
    dv_ref = u_ref(pts_ref)
    
    e = dv_h - dv_ref
    w = Gridap.CellData.get_cell_weights(dΩ_ref)
    
    # Map element-wise squared inner products across the quadrature vectors
    cell_integrals = map(e, w) do ei, wi
        sum(ei .* ei .* wi)
    end
    
    l2_err = sqrt(sum(cell_integrals))
    println("Error: $l2_err")
end
