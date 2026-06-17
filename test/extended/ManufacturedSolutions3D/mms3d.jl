# test/extended/ManufacturedSolutions3D/mms3d.jl
# ==============================================================================================
# 3D manufactured-solution oracle for the paper's §5.2 case. The exact field is the Z-WISE
# EXTRUSION of the 2D manufactured field (article.tex §5.2): u_z = 0 and nothing depends on z.
#   base shape  S = (sin(kx₁)sin(kx₂), cos(kx₁)cos(kx₂), 0),  k = π/L
#   velocity    u_ex = U·α₀·α(x)⁻¹·S       (α radial in (x₁,x₂), z-invariant)
#   pressure    p_ex = P·cos(kx₁)sin(kx₂)
#
# Because the field is z-extruded, every analytic derivative is the corresponding 2D quantity
# ZERO-PADDED in the z slots (∂/∂z ≡ 0, third component ≡ 0). The momentum forcing is therefore
# (f_x(x,y), f_y(x,y), 0) — IDENTICAL to the 2D forcing for the SymmetricGradient viscous operator
# (whose strong divergence identity is dimension-agnostic), and differs only in the viscous term
# for the Deviatoric operator (3D uses ∇ᵈu = ∇ˢu − (1/3)(∇·u)I and ∇·(∇ᵈu) = ½Δu + (1/6)∇(∇·u),
# vs the 2D ½Δu with no grad-div term). This file derives f and g exactly for the SAME formulation
# the solver assembles. [must-test]
#
# Kept self-contained in the test folder (not src/) so the validated 2D core is untouched;
# PorousNSSolver internals are reached via the `PNS` alias.
# ==============================================================================================
using Gridap
using LinearAlgebra
import Gridap.Fields: ∇, Δ
const PNS = PorousNSSolver

# Pad the 2D porosity gradient/Hessian into 3D (z-slots zero — α is z-invariant).
_grad_alpha3d(field, x) = (g = PNS.grad_alpha(field, x); VectorValue(g[1], g[2], 0.0))
function _hess_alpha3d(field, x)
    H = PNS.hess_alpha(field, x)   # symmetric 2×2
    # column-major TensorValue{3,3}: (11,21,31, 12,22,32, 13,23,33)
    return TensorValue(H[1,1], H[1,2], 0.0,  H[1,2], H[2,2], 0.0,  0.0, 0.0, 0.0)
end

# ---- exact velocity (z-extruded) ----
struct UExFunc3D <: Function
    U::Float64
    alpha_0::Float64
    alpha_field::PNS.AbstractPorosityField
    L::Float64
end
function (f::UExFunc3D)(x)
    A = f.alpha_field(x)
    k = pi / f.L
    S = VectorValue(sin(k*x[1])*sin(k*x[2]), cos(k*x[1])*cos(k*x[2]), 0.0)
    return f.U * f.alpha_0 * (1.0 / A) * S
end

# base shape S and its 3D gradient (column-major T[i,j] = ∂_i S_j), z-padded.
@inline function _S3d(k, x)
    VectorValue(sin(k*x[1])*sin(k*x[2]), cos(k*x[1])*cos(k*x[2]), 0.0)
end
@inline function _gradS3d(k, x)
    s1=sin(k*x[1]); s2=sin(k*x[2]); c1=cos(k*x[1]); c2=cos(k*x[2])
    # ∂1S1, ∂2S1, ∂3S1,  ∂1S2, ∂2S2, ∂3S2,  ∂1S3, ∂2S3, ∂3S3
    TensorValue( k*c1*s2,  k*s1*c2, 0.0,
                -k*s1*c2, -k*c1*s2, 0.0,
                 0.0,      0.0,     0.0)
end
@inline _lapS3d(k, x) = VectorValue(-2.0*k^2*sin(k*x[1])*sin(k*x[2]),
                                    -2.0*k^2*cos(k*x[1])*cos(k*x[2]), 0.0)

function grad_u_ex3d(f::UExFunc3D, x)
    A = f.alpha_field(x); gA = _grad_alpha3d(f.alpha_field, x)
    k = pi / f.L
    S = _S3d(k, x); gS = _gradS3d(k, x)
    return f.U * f.alpha_0 * ((1.0/A)*gS - (1.0/A^2)*outer(gA, S))
end
∇(f::UExFunc3D) = x -> grad_u_ex3d(f, x)

function lap_u_ex3d(f::UExFunc3D, x)
    A = f.alpha_field(x); gA = _grad_alpha3d(f.alpha_field, x); lA = PNS.lap_alpha(f.alpha_field, x)
    k = pi / f.L
    S = _S3d(k, x); gS = _gradS3d(k, x); lS = _lapS3d(k, x)
    Ua0 = f.U * f.alpha_0
    P    = Ua0 / A
    gP   = -(Ua0 / A^2) * gA
    lP   = -(Ua0 / A^2) * lA + 2.0*(Ua0 / A^3) * (gA ⋅ gA)
    return P*lS + 2.0*(gP ⋅ gS) + lP*S
end
Δ(f::UExFunc3D) = x -> lap_u_ex3d(f, x)

# ∇(∇·u): for the z-extruded field ∇·u = 2D divergence, so ∇(∇·u) = (2D grad_div_x, _y, 0).
function grad_div_u_ex3d(f::UExFunc3D, x)
    c = f.U * f.alpha_0
    A = f.alpha_field(x)
    gA = PNS.grad_alpha(f.alpha_field, x)   # 2D
    H  = PNS.hess_alpha(f.alpha_field, x)   # 2D
    Ax=gA[1]; Ay=gA[2]; Axx=H[1,1]; Axy=H[1,2]; Ayy=H[2,2]
    k = pi / f.L
    s1=sin(k*x[1]); s2=sin(k*x[2]); c1=cos(k*x[1]); c2=cos(k*x[2])
    S1=s1*s2; S2=c1*c2
    S1x= k*c1*s2; S1y= k*s1*c2
    S2x=-k*s1*c2; S2y=-k*c1*s2
    φ  = S1*Ax + S2*Ay
    φx = S1x*Ax + S1*Axx + S2x*Ay + S2*Axy
    φy = S1y*Ax + S1*Axy + S2y*Ay + S2*Ayy
    gx = -c*φx/A^2 + 2.0*c*φ*Ax/A^3
    gy = -c*φy/A^2 + 2.0*c*φ*Ay/A^3
    return VectorValue(gx, gy, 0.0)
end

# ---- bundle ----
struct Paper3DMMS{F<:PNS.AbstractFormulation}
    formulation::F
    U::Float64
    alpha_field::PNS.AbstractPorosityField
    L::Float64
    alpha_infty::Float64
end

get_u_ex3d(mms::Paper3DMMS) = UExFunc3D(mms.U, mms.alpha_field.alpha_0, mms.alpha_field, mms.L)

# Pressure scale P = (1 + Re + Da)·U·ν/L  (Da from a constant reaction coeff, 0 for Forchheimer).
function characteristic_scales3d(mms::Paper3DMMS)
    nu = mms.formulation.ν
    sigma_0 = hasproperty(mms.formulation.reaction_law, :sigma_val) ? mms.formulation.reaction_law.sigma_val : 0.0
    Re = mms.U * mms.L / nu
    Da = sigma_0 * mms.L^2 / (mms.alpha_infty * nu)
    P  = (1.0 + Re + Da) * mms.U * nu / mms.L
    return mms.U, P
end
function get_p_ex3d(mms::Paper3DMMS)
    _, P = characteristic_scales3d(mms); k = pi / mms.L
    return x -> P * cos(k*x[1]) * sin(k*x[2])
end

"""
    evaluate_exactness_diagnostics3d(mms, Ω, dΩ, c_1, c_2) -> (f_ex, g_ex, u_ex_func, p_ex_func)

Pointwise-exact 3D manufactured momentum forcing `f_ex` and mass source `g_ex` as Gridap
CellFields, plus the exact field closures (for Dirichlet data + error metrics). Mirrors the 2D
oracle; the viscous branch is dimension-correct for the z-extruded field. [must-test]
"""
function evaluate_exactness_diagnostics3d(mms::Paper3DMMS, Ω, dΩ, c_1, c_2)
    u_f = get_u_ex3d(mms)
    _, P_c = characteristic_scales3d(mms)
    k = pi / mms.L
    p_f(x) = P_c * cos(k*x[1]) * sin(k*x[2])
    αf = mms.alpha_field
    nu = mms.formulation.ν
    rxn_law = mms.formulation.reaction_law
    reg = mms.formulation.regularization
    visc_op = mms.formulation.viscous_operator
    h_unused = 1.0

    f_oracle = function(x)
        u  = u_f(x)
        gu = grad_u_ex3d(u_f, x)
        lu = lap_u_ex3d(u_f, x)
        gp = VectorValue(P_c * -k*sin(k*x[1])*sin(k*x[2]),
                         P_c *  k*cos(k*x[1])*cos(k*x[2]), 0.0)
        A  = αf(x)
        gA = _grad_alpha3d(αf, x)

        conv = A * (u ⋅ gu)                       # α (u·∇)u
        # reaction σ(α,u)·u via the SAME routines the assembly uses (exact for Forchheimer too)
        if rxn_law isa PNS.ConstantSigmaLaw
            sig = rxn_law.sigma_val
        elseif rxn_law isa PNS.ForchheimerErgunLaw
            mag = PNS.reaction_speed(reg, u, nu, h_unused, c_1, c_2)
            sig = PNS.sigma(rxn_law, PNS.KinematicState(u, gu, mag), PNS.MediumState(A, gA, h_unused), mag)
        else
            error("3D MMS oracle: no reaction derivation for $(typeof(rxn_law))")
        end
        rxn  = sig * u
        pres = A * gp

        if visc_op isa PNS.SymmetricGradientViscosity
            # ∇·(2Aν∇ˢu) = 2ν(∇ˢu·∇A) + Aν Δu + Aν ∇(∇·u)  — dimension-agnostic identity.
            SPi = 0.5 * (gu + transpose(gu))
            visc = 2.0*nu*(SPi ⋅ gA) + nu*A*lu + nu*A*grad_div_u_ex3d(u_f, x)
        elseif visc_op isa PNS.DeviatoricSymmetricViscosity
            # 3D deviatoric: ∇ᵈu = ∇ˢu − (1/3)(∇·u)I ; ∇·(∇ᵈu) = ½Δu + (1/6)∇(∇·u).
            #   ∇·(2Aν∇ᵈu) = 2ν(∇ˢu·∇A − (1/3)(∇·u)∇A) + Aν Δu + (Aν/3) ∇(∇·u).
            SPi = 0.5 * (gu + transpose(gu)); divu = tr(gu)
            visc = 2.0*nu*((SPi ⋅ gA) - (1.0/3.0)*divu*gA) + nu*A*lu + (nu*A/3.0)*grad_div_u_ex3d(u_f, x)
        elseif visc_op isa PNS.LaplacianPseudoTractionViscosity
            visc = nu*(gu ⋅ gA) + nu*A*lu
        else
            error("3D MMS oracle: no viscous derivation for $(typeof(visc_op))")
        end
        return conv + pres + rxn - visc
    end

    g_oracle = function(x)
        u  = u_f(x); gu = grad_u_ex3d(u_f, x)
        A  = αf(x);  gA = _grad_alpha3d(αf, x)
        return mms.formulation.eps_val * p_f(x) + A * tr(gu) + (gA ⋅ u)
    end

    return CellField(f_oracle, Ω), CellField(g_oracle, Ω), u_f, p_f
end
