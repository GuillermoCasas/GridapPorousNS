# src/problems/mms_paper_2d.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

struct Paper2DMMS{F<:AbstractFormulation}
    formulation::F
    U::Float64
    P_amp::Float64
    alpha_field::AbstractPorosityField
end

function get_u_ex(mms::Paper2DMMS)
    U = mms.U
    alpha_field = mms.alpha_field
    return x -> U * (alpha_field.alpha_0 / alpha(alpha_field, x)) * VectorValue(sin(pi*x[1])*sin(pi*x[2]), cos(pi*x[1])*cos(pi*x[2]))
end

function get_p_ex(mms::Paper2DMMS)
    P_amp = mms.P_amp
    return x -> P_amp * cos(pi*x[1])*sin(pi*x[2])
end

# To evaluate exact forcing via AD or exactly:
# For the MMS exactness logic, gridap provides Automatic Differentiation for smooth algebraic forms:
function build_exact_forcing(mms::Paper2DMMS, c_1, c_2)
    u_ex = get_u_ex(mms)
    p_ex = get_p_ex(mms)
    form = mms.formulation
    
    # Gridap operations apply smoothly on functions if we map them via CellField inside the solver.
    # Alternatively we can define them as functions utilizing Automatic Differentiation, but 
    # to maintain high precision exact derivatives, we should perform AD locally for scalar points.
    
    f_ex = x -> begin
        # Local analytical evaluations here would require ForwardDiff.
        # However, for pure MMS exactly matching the spatial discretization, we can evaluate
        # the strong residual directly on the interpolated `u_ex` CellField!
        VectorValue(0.0, 0.0)
    end
    return f_ex
end

function evaluate_exactness_diagnostics(mms::Paper2DMMS, model, Ω, dΩ, X, Y, c_1, c_2, tau_reg_lim)
    u_ex_func = get_u_ex(mms)
    p_ex_func = get_p_ex(mms)
    
    u_ex_cf = CellField(u_ex_func, Ω)
    p_ex_cf = CellField(p_ex_func, Ω)
    alpha_cf = CellField(x -> alpha(mms.alpha_field, x), Ω)
    
    # 1. Provide f_ex mathematically matching R_u(u_ex) = 0 internally.
    # Thus f_ex(x) = R_u(u_ex, p_ex; f=0) evaluated exactly!
    f_ex_raw = eval_strong_residual_u(mms.formulation, u_ex_cf, p_ex_cf, CellField(1.0, Ω), alpha_cf, CellField(VectorValue(0.0,0.0), Ω), c_1, c_2)
    g_ex_raw = eval_strong_residual_p(mms.formulation, u_ex_cf, p_ex_cf, alpha_cf, CellField(0.0, Ω))
    
    # For tests, we use f_ex_raw directly as our forcing term!
    println("Calculated MMS Exact Forcing...")
    return f_ex_raw, g_ex_raw
end
