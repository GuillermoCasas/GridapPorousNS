#=
src/problems/mms_paper.jl

Method of Manufactured Solutions (MMS) for the porous Navier-Stokes problem, in 2D AND 3D.

The MMS picks a smooth, analytically known velocity/pressure pair (u_ex, p_ex), plugs it
into the strong PDE, and reads off the body force f and mass source g that make that pair
the exact solution. Driving the solver with (f, g) and the Dirichlet data of u_ex turns the
discretization error into the only error left, so a convergence sweep over h must hit the
theoretical O(h^{k+1}) rate. This file builds the exact fields, their derivatives, and the
forcing oracle used by the MMS test harnesses (2D: test/extended/ManufacturedSolutions and
CocquetFormMMS; 3D: test/extended/ManufacturedSolutions3D).

Manufactured fields (frequency k = π/L):
  base shape   S = (sin(kx₁) sin(kx₂), cos(kx₁) cos(kx₂), [0])   — divergence-free, ∇·S ≡ 0
  velocity     u_ex = U·α₀·α(x)⁻¹·S   — the α⁻¹ factor makes A·u physically meaningful in
               porous media (porosity-weighted flux) while keeping u smooth and bounded
  pressure     p_ex = P·cos(kx₁) sin(kx₂)

Dimension (2D vs 3D). The 3D case (article.tex §5.2) is the Z-WISE EXTRUSION of the 2D field:
u_z ≡ 0 and nothing depends on z. So α is radial in (x₁,x₂) and z-invariant (∂_z α ≡ 0), and
EVERY 3D analytic quantity is the corresponding 2D quantity with its z-slots zero-padded. The
implementation is therefore a single dimension-generic code path, parameterized by the
dimension `D ∈ {2, 3}` carried on `PaperMMS{D}` / `UExFunc{D}` / `PExFunc{D}` and dispatched
through the `_shape_*`/`_grad_alpha_d` helpers below. The ONLY term whose analytic form
genuinely differs between 2D and 3D is the deviatoric viscous strong operator (its trace factor
is 1/D and its dilatational ∇(∇·u) contribution is Aν(1 − 2/D)·∇(∇·u), which VANISHES in 2D and
is (Aν/3)∇(∇·u) in 3D) — handled by the `Val(D)`-dispatched `_deviatoric_visc` below. All other
terms (convection, reaction, pressure, mass, and the SymmetricGradient / Laplacian viscous
operators) share one dimension-agnostic expression. [must-test]
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

import Gridap.Fields: ∇, Δ

# ---------------------------------------------------------------------------------------------
# Dimension-generic shape primitives (Val(D)-dispatched).
#
# The manufactured base shape S and its derivatives are the SAME divergence-free 2D expressions
# in every dimension, with the z-slots zero-padded in 3D (S_z ≡ 0, ∂_z ≡ 0). Likewise the
# porosity field is z-invariant, so ∇α is padded with a 0 z-component while ∇²α (scalar
# Laplacian) and the in-plane Hessian need no padding. These helpers build the correctly-shaped
# `VectorValue`/`TensorValue` for D ∈ {2, 3}; D is a compile-time type parameter, so the
# dispatch is resolved statically (no runtime dimension branching).
# ---------------------------------------------------------------------------------------------

@inline _mms_vec(::Val{2}, a, b) = VectorValue(a, b)
@inline _mms_vec(::Val{3}, a, b) = VectorValue(a, b, 0.0)

# base shape S = (sin(kx₁)sin(kx₂), cos(kx₁)cos(kx₂), [0]); ∇·S ≡ 0.
@inline _shape_S(dv::Val, k, x) =
    _mms_vec(dv, sin(k*x[1])*sin(k*x[2]), cos(k*x[1])*cos(k*x[2]))

# ∇S with T[i,j] = ∂_i S_j — each spatial derivative picks up factor k = π/L (chain rule).
# TensorValue components are column-major: 2D (∂₁S₁,∂₂S₁,∂₁S₂,∂₂S₂); 3D adds z rows/cols = 0.
@inline function _shape_gradS(::Val{2}, k, x)
    s1=sin(k*x[1]); s2=sin(k*x[2]); c1=cos(k*x[1]); c2=cos(k*x[2])
    TensorValue(k*c1*s2, k*s1*c2, -k*s1*c2, -k*c1*s2)
end
@inline function _shape_gradS(::Val{3}, k, x)
    s1=sin(k*x[1]); s2=sin(k*x[2]); c1=cos(k*x[1]); c2=cos(k*x[2])
    # (∂₁S₁,∂₂S₁,∂₃S₁, ∂₁S₂,∂₂S₂,∂₃S₂, ∂₁S₃,∂₂S₃,∂₃S₃)
    TensorValue(k*c1*s2, k*s1*c2, 0.0,  -k*s1*c2, -k*c1*s2, 0.0,  0.0, 0.0, 0.0)
end

# ΔS picks up k² from two derivative orders.
@inline _shape_lapS(dv::Val, k, x) =
    _mms_vec(dv, -2.0*k^2*sin(k*x[1])*sin(k*x[2]), -2.0*k^2*cos(k*x[1])*cos(k*x[2]))

# Porosity gradient promoted to D components (α z-invariant ⇒ ∂_z α = 0). In 2D this is exactly
# `grad_alpha`; in 3D it appends a zero z-component. `lap_alpha` (scalar, the 3D Laplacian equals
# the 2D one because ∂²_z α = 0) and `hess_alpha` (used only for the in-plane grad-div) need no
# promotion, so they are called raw below.
@inline _grad_alpha_d(::Val{2}, field, x) = PorousNSSolver.grad_alpha(field, x)
@inline function _grad_alpha_d(::Val{3}, field, x)
    g = PorousNSSolver.grad_alpha(field, x)
    VectorValue(g[1], g[2], 0.0)
end

"""
    PaperMMS{D, F<:AbstractFormulation}

Bundles everything needed to manufacture an exact solution for one concrete formulation, in
spatial dimension `D ∈ {2, 3}`. The forcing must be derived against the *same* viscous operator
and reaction law the solver will assemble, so the formulation is stored here and dispatched on
when building the oracle.

Fields:
- `formulation`  — the `AbstractFormulation` whose strong operators define f and g (carries
                   ν, the viscous operator, the reaction law, the regularization, and the
                   PHYSICAL compressibility ε_phys used in the mass source g).
- `U`            — characteristic velocity amplitude (sets the magnitude of u_ex).
- `alpha_field`  — the porosity field α(x); also supplies ∇α, ∇²α used in the exact derivatives.
- `L`            — characteristic domain length; the manufactured spatial frequency is k = π/L.
- `alpha_infty`  — upper porosity bound (used to non-dimensionalize Da in the scale estimate).
"""
struct PaperMMS{D, F<:AbstractFormulation}
    formulation::F
    U::Float64
    alpha_field::AbstractPorosityField
    L::Float64
    alpha_infty::Float64
    # Manufactured-field family: :extruded (default, the z-invariant 2D field zero-padded in z)
    # or :genuine3d (u = U α₀ curl(A)/α, all components depending on x,y,z with u_z ≠ 0; 3D only).
    field_variant::Symbol
end

# Spatial dimension carried by the type parameter.
_dim(::PaperMMS{D}) where {D} = D

"""
    check_mms_parameters(mms::PaperMMS)

Validate that the manufactured-solution parameters are physically admissible before any
exact field is built. Catches the silent failure modes that would otherwise surface only as
NaNs or wrong convergence rates: non-positive L/U/ν, a porosity bound outside (0, 1], a base
porosity α₀ outside (0, α_∞], and — for radial-bump porosity fields — radii that do not
satisfy 0 < r₁ < r₂ < L/2 (the bump must fit inside the domain). Throws `ArgumentError` on
the first violation.
"""
function check_mms_parameters(mms::PaperMMS)
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

"""
    PaperMMS(form, U, alpha_field; L=1.0, alpha_infty=1.0, dim=2)

Convenience constructor that builds the struct and immediately runs `check_mms_parameters`,
so an inadmissible configuration fails at construction rather than mid-sweep. `dim` selects the
spatial dimension (2 or 3); the 3D field is the z-extrusion of the 2D one (see file header).
"""
function PaperMMS(form::AbstractFormulation, U::Float64, alpha_field::AbstractPorosityField;
                  L=1.0, alpha_infty=1.0, dim::Integer=2, field_variant::Symbol=:extruded)
    (dim == 2 || dim == 3) || throw(ArgumentError("PaperMMS dim must be 2 or 3. Passed: $dim"))
    (field_variant == :extruded || field_variant == :genuine3d) ||
        throw(ArgumentError("PaperMMS field_variant must be :extruded or :genuine3d. Passed: $field_variant"))
    (field_variant == :extruded || dim == 3) ||
        throw(ArgumentError("field_variant=:genuine3d requires dim=3."))
    mms = PaperMMS{Int(dim), typeof(form)}(form, U, alpha_field, L, alpha_infty, field_variant)
    check_mms_parameters(mms)
    return mms
end

"""
    UExFunc{D} <: Function

Callable exact velocity field u_ex(x) = U·α₀·α(x)⁻¹·S(x), where the base shape
S = (sin(kx₁) sin(kx₂), cos(kx₁) cos(kx₂), [0]) carries the manufactured frequency k = π/L in
spatial dimension `D`. Subtyping `Function` lets Gridap treat it as an ordinary field and lets
us attach exact analytic ∇ and Δ overloads (below) instead of differentiating numerically.

Fields: `U` velocity amplitude, `alpha_0` base porosity, `alpha_field` the porosity field
α(x) (and its derivatives), `L` characteristic length fixing the frequency k = π/L.
"""
struct UExFunc{D} <: Function
    U::Float64
    alpha_0::Float64
    alpha_field::AbstractPorosityField
    L::Float64              # characteristic length: spatial frequency is k = π/L
end

# Default-dimension (2D) convenience constructors: the 4-arg form fixes L, the 3-arg form is for
# callers on the unit domain (L = 1.0, so k = π). 3D instances come from `get_u_ex` / UExFunc{3}.
UExFunc(U::Float64, alpha_0::Float64, alpha_field::AbstractPorosityField, L::Float64) =
    UExFunc{2}(U, alpha_0, alpha_field, L)
UExFunc(U::Float64, alpha_0::Float64, alpha_field::AbstractPorosityField) =
    UExFunc{2}(U, alpha_0, alpha_field, 1.0)

function (f::UExFunc{D})(x) where {D}
    A = f.alpha_field(x)
    k = pi / f.L            # spatial frequency k = π/L
    S = _shape_S(Val(D), k, x)
    return f.U * f.alpha_0 * (1.0 / A) * S
end

# Exact velocity gradient ∇u_ex = U·α₀·∇(α⁻¹ S). By the quotient rule on α⁻¹ S this is
# α⁻¹∇S − α⁻²(∇α ⊗ S), using the analytic porosity gradient ∇α. The overload below registers
# this as Gridap's ∇ for UExFunc so the strong residual sees machine-exact derivatives.
function grad_u_ex(f::UExFunc{D}, x) where {D}
    A = f.alpha_field(x)
    grad_A = _grad_alpha_d(Val(D), f.alpha_field, x)
    k = pi / f.L
    S = _shape_S(Val(D), k, x)
    grad_S = _shape_gradS(Val(D), k, x)

    outer_A_S = outer(grad_A, S)
    return f.U * f.alpha_0 * ( (1.0 / A) * grad_S - (1.0 / A^2) * outer_A_S )
end
∇(f::UExFunc) = x -> grad_u_ex(f, x)

# Exact vector Laplacian Δu_ex, needed by the viscous strong operator. Writing u = P(x)·S(x)
# with the scalar prefactor P = U·α₀·α⁻¹, the product rule gives
#   Δ(P S) = P ΔS + 2 (∇P · ∇)S + (ΔP) S,
# and ∇P, ΔP follow from the analytic ∇α and Δα. The Δ overload registers this as Gridap's Δ.
function lap_u_ex(f::UExFunc{D}, x) where {D}
    A = f.alpha_field(x)
    grad_A = _grad_alpha_d(Val(D), f.alpha_field, x)
    lap_A = PorousNSSolver.lap_alpha(f.alpha_field, x)

    k = pi / f.L
    S = _shape_S(Val(D), k, x)
    grad_S = _shape_gradS(Val(D), k, x)
    lap_S = _shape_lapS(Val(D), k, x)                              # ΔS picks up k² from two derivative orders.

    U_a0 = f.U * f.alpha_0
    P = U_a0 / A                                                    # scalar prefactor U·α₀·α⁻¹
    grad_P = - (U_a0 / A^2) * grad_A                               # ∇P
    lap_P = - (U_a0 / A^2) * lap_A + 2.0 * (U_a0 / A^3) * (grad_A ⋅ grad_A)   # ΔP

    # 2 (∇P · ∇)S — directional derivative of S along ∇P.
    term2 = 2.0 * (grad_P ⋅ grad_S)

    return P * lap_S + term2 + lap_P * S
end
Δ(f::UExFunc) = x -> lap_u_ex(f, x)

# Exact ∇(∇·u_ex), needed by the symmetric-gradient / (in 3D) deviatoric viscous strong operator.
# The base shape S is divergence-free (∇·S ≡ 0) and z-invariant, so for u = U·α₀·α⁻¹·S the only
# divergence comes from the in-plane porosity factor: ∇·u = −U·α₀·(S·∇α)/α². Differentiating that
# scalar again (quotient/product rule) gives ∇(∇·u) in terms of S, ∂S, the analytic porosity
# gradient ∇α and Hessian ∇²α — all IN-PLANE (the z-component of ∇(∇·u) is 0 because ∇·u is
# z-invariant), so the raw 2D `grad_alpha`/`hess_alpha` are used and the result is padded to D.
# Below, φ ≡ S·∇α and φx, φy are its partials; each ∂S picks up factor k = π/L (chain rule).
function grad_div_u_ex(f::UExFunc{D}, x) where {D}
    c = f.U * f.alpha_0
    A = f.alpha_field(x)
    gA = PorousNSSolver.grad_alpha(f.alpha_field, x)   # in-plane 2-vector
    H  = PorousNSSolver.hess_alpha(f.alpha_field, x)   # in-plane 2×2
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
    return _mms_vec(Val(D), gx, gy)
end

# Build the callable exact velocity u_ex for this MMS instance (dimension D of the bundle).
# ============================================================================
# GENUINELY 3D manufactured field (audit response R6 / N19).
#
# Unlike the default z-extruded field, this one is genuinely three-dimensional:
# all velocity components depend on x, y, z and u_z ≢ 0. It is built so the
# weighted flux q = α u equals the curl of a smooth vector potential A, whence
# div(α u) = div(curl A) = 0 EXACTLY for any α. With equal wavenumber k = π/L in
# every direction (so the thin-z variation is as well resolved as the in-plane
# one) and amplitudes (a₁,a₂,a₃) = (1,2,3), v ≔ curl(A) is
#     v_x = −k c₁ s₂ s₃,   v_y = 2k s₁ c₂ s₃,   v_z = −k s₁ s₂ c₃
# (sᵢ = sin(kxᵢ), cᵢ = cos(kxᵢ)); div v = 0 and Δv = −3k² v (a Laplacian
# eigenfunction, exploited below). The exact velocity is u = U α₀ v/α, so it
# inherits div(α u) = 0. Field + derivatives are verified symbolically
# (sympy/genuine3d_mms_verification.py) and against ForwardDiff
# (test/blitz/mms_genuine3d_exactness_blitz_test.jl). [must-test][known-fragility]
# ============================================================================
struct UExFuncGenuine3D <: Function
    U::Float64
    alpha_0::Float64
    alpha_field::AbstractPorosityField
    L::Float64              # spatial frequency k = π/L (equal in x, y, z)
end

# v = curl(A) (divergence-free, genuinely 3D) and ∇v with T[i,j] = ∂ᵢvⱼ, in the same
# component order as _shape_gradS: (∂₁v₁,∂₂v₁,∂₃v₁, ∂₁v₂,∂₂v₂,∂₃v₂, ∂₁v₃,∂₂v₃,∂₃v₃).
@inline function _v3(k, x)
    s1=sin(k*x[1]); s2=sin(k*x[2]); s3=sin(k*x[3])
    c1=cos(k*x[1]); c2=cos(k*x[2]); c3=cos(k*x[3])
    VectorValue(-k*c1*s2*s3, 2.0*k*s1*c2*s3, -k*s1*s2*c3)
end
@inline function _grad_v3(k, x)
    s1=sin(k*x[1]); s2=sin(k*x[2]); s3=sin(k*x[3])
    c1=cos(k*x[1]); c2=cos(k*x[2]); c3=cos(k*x[3])
    k2 = k*k
    TensorValue( k2*s1*s2*s3,     -k2*c1*c2*s3,    -k2*c1*s2*c3,      # ∂₁v₁,∂₂v₁,∂₃v₁
                 2.0*k2*c1*c2*s3, -2.0*k2*s1*s2*s3, 2.0*k2*s1*c2*c3,  # ∂₁v₂,∂₂v₂,∂₃v₂
                 -k2*c1*s2*c3,    -k2*s1*c2*c3,     k2*s1*s2*s3)      # ∂₁v₃,∂₂v₃,∂₃v₃
end

function (f::UExFuncGenuine3D)(x)
    A = f.alpha_field(x)
    k = pi / f.L
    return f.U * f.alpha_0 * (1.0 / A) * _v3(k, x)
end

# ∇u = U α₀ [ α⁻¹ ∇v − α⁻² (∇α ⊗ v) ]  (quotient rule; ∇α padded to 3D, α z-invariant).
function grad_u_ex(f::UExFuncGenuine3D, x)
    A = f.alpha_field(x)
    grad_A = _grad_alpha_d(Val(3), f.alpha_field, x)
    k = pi / f.L
    v = _v3(k, x); grad_v = _grad_v3(k, x)
    return f.U * f.alpha_0 * ( (1.0 / A) * grad_v - (1.0 / A^2) * outer(grad_A, v) )
end
∇(f::UExFuncGenuine3D) = x -> grad_u_ex(f, x)

# Δu = P Δv + 2 (∇P·∇)v + (ΔP) v with P = U α₀ α⁻¹ and Δv = −3k² v (eigenfunction).
function lap_u_ex(f::UExFuncGenuine3D, x)
    A = f.alpha_field(x)
    grad_A = _grad_alpha_d(Val(3), f.alpha_field, x)
    lap_A = PorousNSSolver.lap_alpha(f.alpha_field, x)
    k = pi / f.L
    v = _v3(k, x); grad_v = _grad_v3(k, x)
    lap_v = -3.0 * k^2 * v
    U_a0 = f.U * f.alpha_0
    P = U_a0 / A
    grad_P = -(U_a0 / A^2) * grad_A
    lap_P  = -(U_a0 / A^2) * lap_A + 2.0 * (U_a0 / A^3) * (grad_A ⋅ grad_A)
    return P * lap_v + 2.0 * (grad_P ⋅ grad_v) + lap_P * v
end
Δ(f::UExFuncGenuine3D) = x -> lap_u_ex(f, x)

# ∇(∇·u), FULL 3D. Since div v = 0 and α is z-invariant, div u = −U α₀ (v·∇α)/α² with
# v·∇α = v_x α_x + v_y α_y (the v_z α_z term vanishes). This depends on z, so ∇(∇·u) has a
# nonzero z-component (unlike the extruded field). With φ ≔ v·∇α:
function grad_div_u_ex(f::UExFuncGenuine3D, x)
    c = f.U * f.alpha_0
    A = f.alpha_field(x)
    gA = PorousNSSolver.grad_alpha(f.alpha_field, x)   # in-plane 2-vector (α_x, α_y)
    H  = PorousNSSolver.hess_alpha(f.alpha_field, x)   # in-plane 2×2
    Ax = gA[1]; Ay = gA[2]
    Axx = H[1,1]; Axy = H[1,2]; Ayy = H[2,2]
    k = pi / f.L
    s1=sin(k*x[1]); s2=sin(k*x[2]); s3=sin(k*x[3])
    c1=cos(k*x[1]); c2=cos(k*x[2]); c3=cos(k*x[3])
    k2 = k*k
    vx = -k*c1*s2*s3; vy = 2.0*k*s1*c2*s3
    vx_x =  k2*s1*s2*s3;    vx_y = -k2*c1*c2*s3;    vx_z = -k2*c1*s2*c3   # ∂ⱼ v_x
    vy_x =  2.0*k2*c1*c2*s3; vy_y = -2.0*k2*s1*s2*s3; vy_z = 2.0*k2*s1*c2*c3   # ∂ⱼ v_y
    φ  = vx*Ax + vy*Ay
    φx = vx_x*Ax + vx*Axx + vy_x*Ay + vy*Axy
    φy = vx_y*Ax + vx*Axy + vy_y*Ay + vy*Ayy
    φz = vx_z*Ax + vy_z*Ay
    gx = -c*(φx/A^2 - 2.0*φ*Ax/A^3)
    gy = -c*(φy/A^2 - 2.0*φ*Ay/A^3)
    gz = -c*(φz/A^2)
    return VectorValue(gx, gy, gz)
end

# Genuinely-3D exact pressure p = P c₁ s₂ c₃ (zero mean on the box), analytic ∇p registered.
struct PExFuncGenuine3D <: Function
    P::Float64
    L::Float64
end
function (f::PExFuncGenuine3D)(x)
    k = pi / f.L
    return f.P * cos(k*x[1]) * sin(k*x[2]) * cos(k*x[3])
end
function grad_p_ex(f::PExFuncGenuine3D, x)
    k = pi / f.L
    s1=sin(k*x[1]); s2=sin(k*x[2]); s3=sin(k*x[3])
    c1=cos(k*x[1]); c2=cos(k*x[2]); c3=cos(k*x[3])
    return VectorValue(-f.P*k*s1*s2*c3, f.P*k*c1*c2*c3, -f.P*k*c1*s2*s3)
end
∇(f::PExFuncGenuine3D) = x -> grad_p_ex(f, x)

# Build the callable exact velocity u_ex for this MMS instance (dimension D of the bundle).
function get_u_ex(mms::PaperMMS{D}) where {D}
    if mms.field_variant == :genuine3d
        return UExFuncGenuine3D(mms.U, mms.alpha_field.alpha_0, mms.alpha_field, mms.L)
    end
    return UExFunc{D}(mms.U, mms.alpha_field.alpha_0, mms.alpha_field, mms.L)
end

# Pick physically sensible amplitudes for the manufactured fields. The velocity scale is just
# U; the pressure scale P is sized so it stays comparable to the dominant momentum term across
# regimes — viscous when Re, Da small (P ~ Uν/L) and growing with the convective/Darcy
# numbers via the (1 + Re + Da) factor. Re = UL/ν is the Reynolds number; Da = σ₀L²/(α_∞ν)
# is the Darcy number from the constant reaction coefficient σ₀ (0 if the law has no constant
# part). Returns (U_scale, P_scale).
function get_characteristic_scales(mms::PaperMMS)
    nu = mms.formulation.ν

    # Constant reaction coefficient σ₀, if the reaction law exposes one (else 0).
    sigma_0 = 0.0
    if hasproperty(mms.formulation.reaction_law, :sigma_val)
        sigma_0 = mms.formulation.reaction_law.sigma_val
    end

    L = mms.L
    alpha_infty = mms.alpha_infty
    U = mms.U

    Re = U * L / nu                          # Reynolds number
    Da = sigma_0 * L^2 / (alpha_infty * nu)  # Darcy number

    P = (1.0 + Re + Da) * U * nu / L
    return U, P
end

"""
    PExFunc{D} <: Function

Callable exact pressure field p_ex(x) = P·cos(kx₁) sin(kx₂) with manufactured frequency k = π/L and
amplitude P from `get_characteristic_scales`. Mirrors `UExFunc`: subtyping `Function` lets Gridap treat
it as a field, and the `∇` overload below registers the exact analytic gradient so the forcing oracle (and
any strong-residual evaluation) sees machine-exact ∇p instead of differentiating a closure. This is the
single source of truth for p_ex / ∇p_ex (DRIV-07 — previously re-derived inline in several places). The
pressure VALUE is dimension-independent (p_z ≡ 0); only ∇p_ex is padded to D.
"""
struct PExFunc{D} <: Function
    P::Float64              # pressure amplitude
    L::Float64              # characteristic length: spatial frequency k = π/L
end
# Default-dimension (2D) convenience constructor; 3D instances come from `get_p_ex` / PExFunc{3}.
PExFunc(P::Float64, L::Float64) = PExFunc{2}(P, L)

function (f::PExFunc)(x)
    k = pi / f.L            # spatial frequency k = π/L
    return f.P * cos(k*x[1])*sin(k*x[2])
end
# Exact pressure gradient ∇p_ex = (−P·k·sin(kx₁)sin(kx₂), P·k·cos(kx₁)cos(kx₂), [0]); registered as
# Gridap's ∇ for PExFunc (mirroring grad_u_ex / ∇(::UExFunc)).
function grad_p_ex(f::PExFunc{D}, x) where {D}
    k = pi / f.L
    return _mms_vec(Val(D), f.P * -k*sin(k*x[1])*sin(k*x[2]), f.P * k*cos(k*x[1])*cos(k*x[2]))
end
∇(f::PExFunc) = x -> grad_p_ex(f, x)

# Exact pressure p_ex(x) = P·cos(kx₁) sin(kx₂), amplitude P from get_characteristic_scales, frequency
# k = π/L. Returned as a PExFunc (one source of truth, with a registered analytic ∇).
function get_p_ex(mms::PaperMMS{D}) where {D}
    _, P_amp = get_characteristic_scales(mms)
    if mms.field_variant == :genuine3d
        return PExFuncGenuine3D(P_amp, mms.L)
    end
    return PExFunc{D}(P_amp, mms.L)
end

# Deviatoric viscous strong operator ∇·(2Aν∇ᵈu), the ONE term whose analytic form genuinely
# differs by dimension. With ∇ᵈu = ∇ˢu − (1/D)(∇·u)I:
#   ∇·(2Aν∇ᵈu) = 2ν(∇ˢu·∇A − (1/D)(∇·u)∇A) + AνΔu + Aν(1 − 2/D)∇(∇·u).
# In 2D the dilatational term vanishes (∇·(∇ᵈu) = ½Δu exactly, no Hessian of the divergence);
# in 3D it is (Aν/3)∇(∇·u). Written as two Val(D)-dispatched methods so each reproduces the
# historic per-dimension expression bit-for-bit. [must-test][known-fragility]
@inline function _deviatoric_visc(::Val{2}, nu, A, SPi_u, grad_A, div_u_val, lap_u, u_f, x)
    # ∇ᵈu · ∇A = ∇ˢu · ∇A − ½(∇·u)∇A; ∇(∇·u) term cancels in 2D.
    ViscProj_u_dot_grad_A = (SPi_u ⋅ grad_A) - 0.5 * div_u_val * grad_A
    return 2.0 * nu * ViscProj_u_dot_grad_A + nu * A * lap_u
end
@inline function _deviatoric_visc(::Val{3}, nu, A, SPi_u, grad_A, div_u_val, lap_u, u_f, x)
    ViscProj_u_dot_grad_A = (SPi_u ⋅ grad_A) - (1.0/3.0) * div_u_val * grad_A
    return 2.0 * nu * ViscProj_u_dot_grad_A + nu * A * lap_u + (nu * A / 3.0) * grad_div_u_ex(u_f, x)
end

"""
    evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)

Build the manufactured momentum forcing `f_ex` and mass source `g_ex` for this MMS instance,
returned as Gridap `CellField`s ready to be assembled as the right-hand side.

Mechanism: both are computed by a pointwise *oracle* closure that evaluates the strong PDE
residual analytically at each quadrature point, using the exact u_ex/p_ex and their closed-form
derivatives (grad_u_ex, lap_u_ex, grad_div_u_ex) plus the exact porosity α, ∇α, ∇²α. Evaluating
the continuous operators directly — rather than differentiating an interpolant — keeps the
forcing free of interpolation error, so the only residual error in the sweep is discretization
error and the O(h^{k+1}) rate is recoverable. Works in 2D and 3D via the mms's dimension `D`
(the 3D field is the z-extrusion of the 2D one). [must-test]

Momentum: f_ex = (convection) + (pressure) + (reaction) − (viscous), each term assembled from
the SAME formulation operators the solver uses (reaction law, viscous operator), so f_ex is
exact for whichever concrete formulation `mms` carries.
Mass: g_ex enforces the porosity-weighted continuity ∇·(α u) plus the PHYSICAL compressibility
term ε_phys·p (from `formulation.physical_epsilon`). The NUMERICAL penalty ε_num is deliberately
excluded — the Codina iterative penalty adds ε_num·pⁿ to the LHS and ε_num·pⁿ⁻¹ to the RHS, so
it cancels at convergence and must NOT be folded into g.

`model`, `h_cf`, `X`, `Y`, `tau_reg_lim` are accepted for harness signature parity (the exact
fields for error metrics are obtained by callers via `get_u_ex` / `get_p_ex`); `c_1`, `c_2` are
the stabilization constants threaded to the reaction speed.
"""
function evaluate_exactness_diagnostics(mms::PaperMMS{D}, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim) where {D}
    u_f = get_u_ex(mms)
    p_ex_func = get_p_ex(mms)

    alpha_field = mms.alpha_field
    nu = mms.formulation.ν
    reaction_law = mms.formulation.reaction_law
    viscous_op = mms.formulation.viscous_operator

    # The reaction term is evaluated through the SAME `reaction_speed` routine the assembly uses
    # (src/models/regularization.jl) so the manufactured forcing is exact for the nonlinear
    # Forchheimer law as well as the constant-σ law. [known-fragility] `reaction_speed` carries
    # only a constant velocity floor (no mesh-dependent ν/h term), so it is mesh-independent by
    # construction — this pointwise oracle (which has no element `h`) is therefore exact
    # regardless of `h_floor_weight`. The `h` argument below is a dummy for signature parity.
    reg_oracle = mms.formulation.regularization
    h_unused = 1.0
    dv = Val(D)

    f_ex_oracle = function(x)
        u = u_f(x)
        grad_u = grad_u_ex(u_f, x)
        lap_u = lap_u_ex(u_f, x)

        grad_p = grad_p_ex(p_ex_func, x)

        A = alpha_field(x)
        grad_A = _grad_alpha_d(dv, alpha_field, x)

        # Convective term α (u·∇)u. In Gridap, Vector ⋅ Tensor is the contraction v_i T_{ij},
        # so `u ⋅ grad_u` equals (u·∇)u; the leading α is the porosity weighting from the paper.
        conv = A * (u ⋅ grad_u)

        # Reaction term σ(α,u)·u. The two reaction laws differ here: ConstantSigmaLaw returns a
        # fixed σ₀, while the Forchheimer-Ergun law σ = a(α) + b(α)|u| (eq:DBFResistanceTerm) is
        # porosity-dependent and nonlinear in u. Both go through the same `sigma`/`reaction_speed`
        # routines the assembly uses, so the forcing stays exact for either law.
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

        # Viscous term: dispatch on the formulation's viscous operator so the manufactured
        # forcing uses the strong form that matches the weak operator the solver actually
        # assembles. [must-test] Using the wrong branch here breaks MMS exactness.
        if viscous_op isa PorousNSSolver.SymmetricGradientViscosity
            # Weak form: 2ν (α ∇ˢu ⊙ ∇ˢv), with ∇ˢu = sym(∇u). Strong divergence (dimension-
            # agnostic): ∇·(2Aν ∇ˢu) = 2ν (∇ˢu · ∇A) + Aν Δu + Aν ∇(∇·u).
            SPi_u = 0.5 * (grad_u + transpose(grad_u))     # ∇ˢu
            grad_div_u = grad_div_u_ex(u_f, x)
            visc = 2.0 * nu * (SPi_u ⋅ grad_A) + nu * A * lap_u + nu * A * grad_div_u
        elseif viscous_op isa PorousNSSolver.DeviatoricSymmetricViscosity
            # Canonical operator. Weak form: 2ν (α ∇ᵈu ⊙ ∇ˢv) with the deviatoric (trace-free)
            # symmetric gradient ∇ᵈu = ∇ˢu − (1/D)(∇·u)I. Strong form is dimension-dependent —
            # see `_deviatoric_visc` (2D drops the ∇(∇·u) term; 3D keeps (Aν/3)∇(∇·u)).
            SPi_u = 0.5 * (grad_u + transpose(grad_u))     # ∇ˢu
            div_u_val = tr(grad_u)                          # ∇·u
            visc = _deviatoric_visc(dv, nu, A, SPi_u, grad_A, div_u_val, lap_u, u_f, x)
        elseif viscous_op isa PorousNSSolver.LaplacianPseudoTractionViscosity
            # [legacy] Laplacian/pseudo-traction variant. Strong form:
            #   ∇·(αν ∇u) = ν (∇u · ∇A) + αν Δu.
            visc = nu * (grad_u ⋅ grad_A) + A * nu * lap_u
        else
            error("MMS Oracle missing native analytical derivation for $(typeof(viscous_op)).")
        end

        # Strong momentum residual rearranged as the body force: f = conv + ∇p·α + σu − visc.
        return conv + pres + rxn - visc
    end

    f_ex_raw = CellField(f_ex_oracle, Ω)

    # Mass source g_ex for the continuity equation. The porosity-weighted divergence expands as
    # ∇·(α u) = α (∇·u) + ∇α · u; the `physical_epsilon * p` term is the pseudo-compressibility
    # stabilization the formulation adds to the mass equation, so g_ex matches it exactly.
    g_ex_oracle = function(x)
        u = u_f(x)
        grad_u = grad_u_ex(u_f, x)
        A = alpha_field(x)
        grad_A = _grad_alpha_d(dv, alpha_field, x)

        div_u_val = tr(grad_u)
        return mms.formulation.physical_epsilon * p_ex_func(x) + A * div_u_val + (grad_A ⋅ u)
    end

    g_ex_raw = CellField(g_ex_oracle, Ω)

    println("Calculated MMS Exact Forcing...")
    return f_ex_raw, g_ex_raw
end
