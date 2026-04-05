using Test
using Gridap
using LinearAlgebra
using PorousNSSolver

# ---------- tiny geometry ----------

function tiny_model_2d(; n=(1,1), domain=(0.0,1.0,0.0,1.0))
    xmin, xmax, ymin, ymax = domain
    return CartesianDiscreteModel((xmin, xmax, ymin, ymax), n)
end

function tiny_spaces_2d(model; kv=1, kp=1, dirichlet_tags=String[])
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
    refe_p = ReferenceFE(lagrangian, Float64, kp)

    labels = get_face_labeling(model)
    V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=dirichlet_tags)
    Q = TestFESpace(model, refe_p, conformity=:H1)

    U = TrialFESpace(V)
    P = TrialFESpace(Q)
    X = MultiFieldFESpace([U,P])
    Y = MultiFieldFESpace([V,Q])

    return X, Y, V, Q
end

function tiny_measure(model; degree=4)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    cell_measures = get_cell_measure(Ω)
    h_array = lazy_map(v -> sqrt(abs(v)), cell_measures)
    h = CellField(h_array, Ω)
    return Ω, dΩ, h
end

# ---------- local finite differences ----------

function directional_fd(f, x, dx; ϵ=1e-7)
    return (f(x .+ ϵ .* dx) - f(x .- ϵ .* dx)) / (2ϵ)
end

# ---------- tiny local states ----------

function make_local_states(; 
    u=VectorValue(0.2, -0.1),
    du=VectorValue(0.3, 0.4),
    grad_u=TensorValue(0.1, 0.2, -0.3, 0.4),
    alpha=0.7,
    grad_alpha=VectorValue(0.05, -0.02),
    h=0.25,
    mag_u=sqrt(u⋅u + 1e-8)
)
    kin = PorousNSSolver.KinematicState(u, grad_u, mag_u)
    med = PorousNSSolver.MediumState(alpha, grad_alpha, h)
    return kin, med, du
end

# ---------- analytic fields on x = (x,y) ----------

u_poly(x) = VectorValue(x[1]^2 + x[2], x[1] - x[2]^2)
v_poly(x) = VectorValue(x[1] + 2x[2], -x[1] + x[2])
p_poly(x) = x[1]^2 - x[2]
q_poly(x) = x[1] + x[2]^2
alpha_lin(x) = 0.8 + 0.1*x[1] - 0.05*x[2]
g_zero(x) = 0.0
f_zero(x) = VectorValue(0.0, 0.0)
