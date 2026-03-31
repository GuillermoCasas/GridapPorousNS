using Gridap
domain = (0,1,0,1)
model1 = CartesianDiscreteModel(domain, (2,2))
model2 = CartesianDiscreteModel(domain, (4,4))
refe = ReferenceFE(lagrangian, Float64, 1)
V1 = TestFESpace(model1, refe)
V2 = TestFESpace(model2, refe)
u1 = interpolate_everywhere(x->x[1], V1)

Ω2 = Triangulation(model2)
dΩ2 = Measure(Ω2, 2)

# evaluate u1 on quad points of dΩ2
# The correct way to evaluate a CellField on different mesh quadrature points:
pts = get_cell_points(dΩ2)
try
    vals = u1(pts)
    println("Native evaluation works: ", typeof(vals))
catch e
    println("Native evaluation failed: ", e)
end

# another way: create a CellField by mapping
# Gridap provides a way to evaluate functions
# We can try wrapper: u1_wrapper(x) = u1(x)
