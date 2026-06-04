# src/problems/mms_paper_2d.jl
using Gridap
using Gridap.Algebra
using LinearAlgebra

struct Paper2DMMS{F<:AbstractFormulation}
    formulation::F
    U::Float64
    alpha_field::AbstractPorosityField
    L::Float64
    alpha_infty::Float64
end

function check_mms_parameters(mms::Paper2DMMS)
    if mms.L <= 0.0
        throw(ArgumentError("Characteristic length L must be strictly positive. Passed: $(mms.L)"))
    end
    if mms.U <= 0.0
        throw(ArgumentError("Characteristic velocity U must be strictly positive. Passed: $(mms.U)"))
    end
    if mms.formulation.ν <= 0.0
        throw(ArgumentError("Kinematic viscosity nu must be strictly positive. Passed: $(mms.formulation.ν)"))
    end
    
    if mms.alpha_infty <= 0.0 || mms.alpha_infty > 1.0
        throw(ArgumentError("Upper porosity bound alpha_infty must be strictly in (0, 1]. Passed: $(mms.alpha_infty)"))
    end
    
    if hasproperty(mms.alpha_field, :alpha_0)
        alpha_0 = mms.alpha_field.alpha_0
        if alpha_0 <= 0.0 || alpha_0 > mms.alpha_infty
            throw(ArgumentError("Base porosity alpha_0 must be strictly within (0, alpha_infty]. Passed: $alpha_0"))
        end
    end
    
    if hasproperty(mms.alpha_field, :r_1) && hasproperty(mms.alpha_field, :r_2)
        r_1 = mms.alpha_field.r_1
        r_2 = mms.alpha_field.r_2
        if !(0.0 < r_1 < r_2 < mms.L / 2)
            throw(ArgumentError("Radial bump configuration invalid. Requirement: 0 < r_1 < r_2 < L/2. Passed: r_1=$r_1, r_2=$r_2, L=$(mms.L)"))
        end
    end
end

function Paper2DMMS(form::AbstractFormulation, U::Float64, alpha_field::AbstractPorosityField; L=1.0, alpha_infty=1.0)
    mms = Paper2DMMS(form, U, alpha_field, L, alpha_infty)
    check_mms_parameters(mms)
    return mms
end

import Gridap.Fields: ∇, Δ

struct UExFunc <: Function
    U::Float64
    alpha_0::Float64
    alpha_field::AbstractPorosityField
    L::Float64              # characteristic length: polynomial frequency is π/L
end

# 3-arg constructor defaulting the domain length L=1.0, for callers that do not rescale the domain.
# New code should pass L explicitly; the manufactured frequency is k = π/L.
UExFunc(U::Float64, alpha_0::Float64, alpha_field::AbstractPorosityField) =
    UExFunc(U, alpha_0, alpha_field, 1.0)
function (f::UExFunc)(x)
    A = f.alpha_field(x)
    k = pi / f.L            # spatial frequency under L-rescaling
    S = VectorValue(sin(k*x[1])*sin(k*x[2]), cos(k*x[1])*cos(k*x[2]))
    return f.U * f.alpha_0 * (1.0 / A) * S
end

function grad_u_ex(f::UExFunc, x)
    A = f.alpha_field(x)
    grad_A = PorousNSSolver.grad_alpha(f.alpha_field, x)
    k = pi / f.L
    S = VectorValue(sin(k*x[1])*sin(k*x[2]), cos(k*x[1])*cos(k*x[2]))

    # ∇S_ij = ∂_i S_j — each spatial derivative picks up factor k=π/L by the chain rule.
    grad_S = TensorValue(
        k*cos(k*x[1])*sin(k*x[2]), k*sin(k*x[1])*cos(k*x[2]),
        -k*sin(k*x[1])*cos(k*x[2]), -k*cos(k*x[1])*sin(k*x[2])
    )

    outer_A_S = outer(grad_A, S)
    return f.U * f.alpha_0 * ( (1.0 / A) * grad_S - (1.0 / A^2) * outer_A_S )
end
∇(f::UExFunc) = x -> grad_u_ex(f, x)

function lap_u_ex(f::UExFunc, x)
    A = f.alpha_field(x)
    grad_A = PorousNSSolver.grad_alpha(f.alpha_field, x)
    lap_A = PorousNSSolver.lap_alpha(f.alpha_field, x)

    k = pi / f.L
    S = VectorValue(sin(k*x[1])*sin(k*x[2]), cos(k*x[1])*cos(k*x[2]))
    grad_S = TensorValue(
        k*cos(k*x[1])*sin(k*x[2]), k*sin(k*x[1])*cos(k*x[2]),
        -k*sin(k*x[1])*cos(k*x[2]), -k*cos(k*x[1])*sin(k*x[2])
    )
    # ΔS picks up k² from two derivative orders.
    lap_S = VectorValue(-2.0*k^2 * sin(k*x[1])*sin(k*x[2]), -2.0*k^2 * cos(k*x[1])*cos(k*x[2]))

    U_a0 = f.U * f.alpha_0
    P = U_a0 / A
    grad_P = - (U_a0 / A^2) * grad_A
    lap_P = - (U_a0 / A^2) * lap_A + 2.0 * (U_a0 / A^3) * (grad_A ⋅ grad_A)

    # grad_P ⋅ grad_S applies directional derivative of S along grad_P
    term2 = 2.0 * (grad_P ⋅ grad_S)

    return P * lap_S + term2 + lap_P * S
end
Δ(f::UExFunc) = x -> lap_u_ex(f, x)

# Exact ∇(∇·u_ex), needed by the symmetric-gradient viscous strong operator. The base shape
# S = (sin(kx₁) sin(kx₂), cos(kx₁) cos(kx₂)) with k=π/L is divergence-free (∇·S ≡ 0), so
# for u = U α₀ α⁻¹ S we have ∇·u = -U α₀ (S·∇α)/α², and ∇(∇·u) follows by the
# quotient/product rule using the exact porosity gradient ∇α and Hessian ∇²α. Each spatial
# derivative of S picks up factor k=π/L (chain rule on x ↦ x/L).
function grad_div_u_ex(f::UExFunc, x)
    c = f.U * f.alpha_0
    A = f.alpha_field(x)
    gA = PorousNSSolver.grad_alpha(f.alpha_field, x)
    H  = PorousNSSolver.hess_alpha(f.alpha_field, x)
    Ax = gA[1]; Ay = gA[2]
    Axx = H[1,1]; Axy = H[1,2]; Ayy = H[2,2]

    k = pi / f.L
    s1 = sin(k*x[1]); s2 = sin(k*x[2])
    c1 = cos(k*x[1]); c2 = cos(k*x[2])
    S1 = s1*s2;  S2 = c1*c2
    S1x =  k*c1*s2;  S1y =  k*s1*c2
    S2x = -k*s1*c2;  S2y = -k*c1*s2

    φ  = S1*Ax + S2*Ay                       # = S·∇α
    φx = S1x*Ax + S1*Axx + S2x*Ay + S2*Axy   # = ∂₁(S·∇α)
    φy = S1y*Ax + S1*Axy + S2y*Ay + S2*Ayy   # = ∂₂(S·∇α)

    gx = -c*φx/A^2 + 2.0*c*φ*Ax/A^3
    gy = -c*φy/A^2 + 2.0*c*φ*Ay/A^3
    return VectorValue(gx, gy)
end

function get_u_ex(mms::Paper2DMMS)
    return UExFunc(mms.U, mms.alpha_field.alpha_0, mms.alpha_field, mms.L)
end

function get_characteristic_scales(mms::Paper2DMMS)
    nu = mms.formulation.ν
    
    # Gridap native parameter extracts
    sigma_0 = 0.0
    if hasproperty(mms.formulation.reaction_law, :sigma_val)
        sigma_0 = mms.formulation.reaction_law.sigma_val
    end
    
    L = mms.L
    alpha_infty = mms.alpha_infty
    U = mms.U
    
    Re = U * L / nu
    Da = sigma_0 * L^2 / (alpha_infty * nu)
    
    P = (1.0 + Re + Da) * U * nu / L
    return U, P
end

function get_p_ex(mms::Paper2DMMS)
    U_amp, P_amp = get_characteristic_scales(mms)
    L = mms.L
    k = pi / L
    return x -> P_amp * cos(k*x[1])*sin(k*x[2])
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

function evaluate_exactness_diagnostics(mms::Paper2DMMS, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)
    u_ex_func = get_u_ex(mms)
    p_ex_func = get_p_ex(mms)
    
    # By mapping the forcing evaluation perfectly to an exact point-wise oracle closure,
    # we mathematically resolve all continuous physics inside the continuum manifold,
    # bypassing any CellField AutoDiff, OperationField transposition constraints, and interpolation noise.
    
    alpha_field = mms.alpha_field
    nu = mms.formulation.ν
    P_c = get_characteristic_scales(mms)[2]
    
    u_f = get_u_ex(mms)
    L_local = mms.L
    k_local = pi / L_local
    p_f(x) = P_c * cos(k_local*x[1])*sin(k_local*x[2])

    # We must properly evaluate the reaction operator exactly.
    reaction_law = mms.formulation.reaction_law

    # The reaction term is evaluated through the SAME `reaction_speed` routine the assembly uses
    # (src/models/regularization.jl) so the manufactured forcing is exact for the nonlinear
    # Forchheimer law as well as the constant-σ law. `reaction_speed` carries only a constant
    # velocity floor (no mesh-dependent ν/h term), so it is mesh-independent by construction —
    # this pointwise oracle (which has no element `h`) is therefore exact regardless of
    # `h_floor_weight`. The `h` argument below is accepted for signature parity and unused.
    reg_oracle = mms.formulation.regularization
    h_unused = 1.0

    f_ex_oracle = function(x)
        u = u_f(x)
        grad_u = grad_u_ex(u_f, x)
        lap_u = lap_u_ex(u_f, x)
        
        grad_p = VectorValue(P_c * -k_local*sin(k_local*x[1])*sin(k_local*x[2]), P_c * k_local*cos(k_local*x[1])*cos(k_local*x[2]))
        
        A = alpha_field(x)
        grad_A = PorousNSSolver.grad_alpha(alpha_field, x)
        
        # In Gridap, Vector ⋅ Tensor corresponds to directional derivative v_i T_{ij}
        # So u ⋅ grad_u evaluates identical to (u ⋅ ∇)u. 
        conv = A * (u ⋅ grad_u)
        
        # Reaction term σ(α,u)·u. This is the one place the two formulations differ in the
        # reaction: ConstantSigmaLaw returns its constant, while the Forchheimer-Ergun law
        # σ = a(α) + b(α)|u| is porosity-dependent and nonlinear in u. Both are evaluated through
        # the same `sigma`/`effective_speed` routines used by the assembly, so the forcing is exact.
        if reaction_law isa PorousNSSolver.ConstantSigmaLaw
            sig_val = reaction_law.sigma_val
        elseif reaction_law isa PorousNSSolver.ForchheimerErgunLaw
            mag = PorousNSSolver.reaction_speed(reg_oracle, u, nu, h_unused, c_1, c_2)
            sig_val = PorousNSSolver.sigma(
                reaction_law,
                PorousNSSolver.KinematicState(u, grad_u, mag),
                PorousNSSolver.MediumState(A, grad_A, h_unused),
                mag,
            )
        else
            error("MMS Oracle missing native analytical derivation for reaction law $(typeof(reaction_law)).")
        end
        rxn = sig_val * u
        
        pres = A * grad_p
        
        # STRICT VISCOUS EVALUATOR DISPATCH
        # Ensure that the mathematical manufactured solution identically aligns with the 
        # actual theoretical derivation of the executed operator.
        viscous_op = mms.formulation.viscous_operator
        if viscous_op isa PorousNSSolver.SymmetricGradientViscosity
            # Weak form: 2*nu*(α * \SPi\nabla \boldsymbol{u} ⊙ \SPi\nabla \boldsymbol{v})
            # Integrated mathematical equality for the Strong form evaluated in Gridap RHS:
            # ∇⋅(2Aν \SPi\nabla \boldsymbol{u}) = 2ν(\SPi\nabla \boldsymbol{u} ⋅ \nabla A) + AνΔu + Aν∇(∇⋅u)
            SPi_u = 0.5 * (grad_u + transpose(grad_u))
            SPi_u_dot_grad_A = SPi_u ⋅ grad_A

            # ∇(∇⋅u): evaluated in closed form from the exact porosity gradient and Hessian
            # (grad_div_u_ex), replacing the previous finite-difference approximation.
            grad_div_u = grad_div_u_ex(u_f, x)
            visc = 2.0 * nu * SPi_u_dot_grad_A + nu * A * lap_u + nu * A * grad_div_u
        elseif viscous_op isa PorousNSSolver.DeviatoricSymmetricViscosity
            # Weak form: 2*nu*(α * \ViscProj \nabla \boldsymbol{u} ⊙ \SPi\nabla \boldsymbol{v})
            # Mathematical exact strong mapping: ∇⋅(2Aν \ViscProj \nabla \boldsymbol{u}) = 2ν(\ViscProj \nabla \boldsymbol{u} ⋅ \nabla A) + 2Aν∇⋅(\ViscProj \nabla \boldsymbol{u})
            # In 2D exactly: ∇⋅(\ViscProj \nabla \boldsymbol{u}) = 0.5Δu, neutralizing the ∇(∇⋅u) dilatancy requirement mathematically!
            SPi_u = 0.5 * (grad_u + transpose(grad_u))
            div_u_val = tr(grad_u)
            
            # Evaluate \ViscProj \nabla \boldsymbol{u} ⋅ ∇A = \SPi\nabla \boldsymbol{u} ⋅∇A - 0.5*(∇⋅u)*∇A
            SPi_u_dot_grad_A = SPi_u ⋅ grad_A
            ViscProj_u_dot_grad_A = SPi_u_dot_grad_A - 0.5 * div_u_val * grad_A
            
            visc = 2.0 * nu * ViscProj_u_dot_grad_A + nu * A * lap_u
        elseif viscous_op isa PorousNSSolver.LaplacianPseudoTractionViscosity
            # Strong form: ∇⋅(α ν ∇u) = ν*(∇u ⋅ ∇A) + α*ν*Δu
            visc = nu * (grad_u ⋅ grad_A) + A * nu * lap_u
        else
            error("MMS Oracle missing native analytical derivation for $(typeof(viscous_op)).")
        end
        
        return conv + pres + rxn - visc
    end
    
    f_ex_raw = CellField(f_ex_oracle, Ω)
    
    # For pressure exactness (continuity constraint), exact strong equation is ∇⋅(A*u) = A*∇⋅u + ∇A⋅u
    # We evaluate it analytically to enforce machine exact continuity balances.
    g_ex_oracle = function(x)
        u = u_f(x)
        grad_u = grad_u_ex(u_f, x)
        A = alpha_field(x)
        grad_A = PorousNSSolver.grad_alpha(alpha_field, x)
        
        div_u_val = tr(grad_u)
        return mms.formulation.eps_val * p_f(x) + A * div_u_val + (grad_A ⋅ u)
    end
    
    g_ex_raw = CellField(g_ex_oracle, Ω)
    
    # Interpolations purely for error metric generation! Not for forcing!
    u_ex_cf = interpolate_everywhere(x -> u_f(x), X.spaces[1])
    p_ex_cf = interpolate_everywhere(p_f, X.spaces[2])
    
    # For tests, we use f_ex_raw directly as our forcing term!
    println("Calculated MMS Exact Forcing...")
    return f_ex_raw, g_ex_raw
end
