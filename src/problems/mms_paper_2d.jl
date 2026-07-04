#=
src/problems/mms_paper_2d.jl

Method of Manufactured Solutions (MMS) for the 2D porous Navier-Stokes problem.

The MMS picks a smooth, analytically known velocity/pressure pair (u_ex, p_ex), plugs it
into the strong PDE, and reads off the body force f and mass source g that make that pair
the exact solution. Driving the solver with (f, g) and the Dirichlet data of u_ex turns the
discretization error into the only error left, so a convergence sweep over h must hit the
theoretical O(h^{k+1}) rate. This file builds the exact fields, their derivatives, and the
forcing oracle used by the MMS test harness (test/extended/ManufacturedSolutions).

Manufactured fields (frequency k = π/L):
  base shape   S = (sin(kx₁) sin(kx₂), cos(kx₁) cos(kx₂))   — divergence-free, ∇·S ≡ 0
  velocity     u_ex = U·α₀·α(x)⁻¹·S   — the α⁻¹ factor makes A·u physically meaningful in
               porous media (porosity-weighted flux) while keeping u smooth and bounded
  pressure     p_ex = P·cos(kx₁) sin(kx₂)
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

"""
    Paper2DMMS{F<:AbstractFormulation}

Bundles everything needed to manufacture an exact solution for one concrete formulation.
The forcing must be derived against the *same* viscous operator and reaction law the solver
will assemble, so the formulation is stored here and dispatched on when building the oracle.

Fields:
- `formulation`  — the `AbstractFormulation` whose strong operators define f and g (carries
                   ν, the viscous operator, the reaction law, the regularization).
- `U`            — characteristic velocity amplitude (sets the magnitude of u_ex).
- `alpha_field`  — the porosity field α(x); also supplies ∇α, ∇²α used in the exact derivatives.
- `L`            — characteristic domain length; the manufactured spatial frequency is k = π/L.
- `alpha_infty`  — upper porosity bound (used to non-dimensionalize Da in the scale estimate).
"""
struct Paper2DMMS{F<:AbstractFormulation}
    formulation::F
    U::Float64
    alpha_field::AbstractPorosityField
    L::Float64
    alpha_infty::Float64
end

"""
    check_mms_parameters(mms::Paper2DMMS)

Validate that the manufactured-solution parameters are physically admissible before any
exact field is built. Catches the silent failure modes that would otherwise surface only as
NaNs or wrong convergence rates: non-positive L/U/ν, a porosity bound outside (0, 1], a base
porosity α₀ outside (0, α_∞], and — for radial-bump porosity fields — radii that do not
satisfy 0 < r₁ < r₂ < L/2 (the bump must fit inside the domain). Throws `ArgumentError` on
the first violation.
"""
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

"""
    Paper2DMMS(form, U, alpha_field; L=1.0, alpha_infty=1.0)

Convenience constructor that builds the struct and immediately runs `check_mms_parameters`,
so an inadmissible configuration fails at construction rather than mid-sweep.
"""
function Paper2DMMS(form::AbstractFormulation, U::Float64, alpha_field::AbstractPorosityField; L=1.0, alpha_infty=1.0)
    mms = Paper2DMMS(form, U, alpha_field, L, alpha_infty)
    check_mms_parameters(mms)
    return mms
end

import Gridap.Fields: ∇, Δ

"""
    UExFunc <: Function

Callable exact velocity field u_ex(x) = U·α₀·α(x)⁻¹·S(x), where the base shape
S = (sin(kx₁) sin(kx₂), cos(kx₁) cos(kx₂)) carries the manufactured frequency k = π/L.
Subtyping `Function` lets Gridap treat it as an ordinary field and lets us attach exact
analytic ∇ and Δ overloads (below) instead of differentiating numerically.

Fields: `U` velocity amplitude, `alpha_0` base porosity, `alpha_field` the porosity field
α(x) (and its derivatives), `L` characteristic length fixing the frequency k = π/L.
"""
struct UExFunc <: Function
    U::Float64
    alpha_0::Float64
    alpha_field::AbstractPorosityField
    L::Float64              # characteristic length: spatial frequency is k = π/L
end

# 3-arg constructor for callers on the unit domain (L = 1.0); the frequency is then k = π.
UExFunc(U::Float64, alpha_0::Float64, alpha_field::AbstractPorosityField) =
    UExFunc(U, alpha_0, alpha_field, 1.0)
function (f::UExFunc)(x)
    A = f.alpha_field(x)
    k = pi / f.L            # spatial frequency k = π/L
    S = VectorValue(sin(k*x[1])*sin(k*x[2]), cos(k*x[1])*cos(k*x[2]))
    return f.U * f.alpha_0 * (1.0 / A) * S
end

# Exact velocity gradient ∇u_ex = U·α₀·∇(α⁻¹ S). By the quotient rule on α⁻¹ S this is
# α⁻¹∇S − α⁻²(∇α ⊗ S), using the analytic porosity gradient ∇α. The overload below registers
# this as Gridap's ∇ for UExFunc so the strong residual sees machine-exact derivatives.
function grad_u_ex(f::UExFunc, x)
    A = f.alpha_field(x)
    grad_A = PorousNSSolver.grad_alpha(f.alpha_field, x)
    k = pi / f.L
    S = VectorValue(sin(k*x[1])*sin(k*x[2]), cos(k*x[1])*cos(k*x[2]))

    # ∇S_ij = ∂_i S_j — each spatial derivative picks up factor k = π/L by the chain rule.
    grad_S = TensorValue(
        k*cos(k*x[1])*sin(k*x[2]), k*sin(k*x[1])*cos(k*x[2]),
        -k*sin(k*x[1])*cos(k*x[2]), -k*cos(k*x[1])*sin(k*x[2])
    )

    outer_A_S = outer(grad_A, S)
    return f.U * f.alpha_0 * ( (1.0 / A) * grad_S - (1.0 / A^2) * outer_A_S )
end
∇(f::UExFunc) = x -> grad_u_ex(f, x)

# Exact vector Laplacian Δu_ex, needed by the viscous strong operator. Writing u = P(x)·S(x)
# with the scalar prefactor P = U·α₀·α⁻¹, the product rule gives
#   Δ(P S) = P ΔS + 2 (∇P · ∇)S + (ΔP) S,
# and ∇P, ΔP follow from the analytic ∇α and Δα. The Δ overload registers this as Gridap's Δ.
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
    P = U_a0 / A                                                    # scalar prefactor U·α₀·α⁻¹
    grad_P = - (U_a0 / A^2) * grad_A                               # ∇P
    lap_P = - (U_a0 / A^2) * lap_A + 2.0 * (U_a0 / A^3) * (grad_A ⋅ grad_A)   # ΔP

    # 2 (∇P · ∇)S — directional derivative of S along ∇P.
    term2 = 2.0 * (grad_P ⋅ grad_S)

    return P * lap_S + term2 + lap_P * S
end
Δ(f::UExFunc) = x -> lap_u_ex(f, x)

# Exact ∇(∇·u_ex), needed by the symmetric-gradient viscous strong operator. The base shape
# S is divergence-free (∇·S ≡ 0), so for u = U·α₀·α⁻¹·S the only divergence comes from the
# porosity factor: ∇·u = −U·α₀·(S·∇α)/α². Differentiating that scalar again (quotient/product
# rule) gives ∇(∇·u) in terms of S, ∂S, the analytic porosity gradient ∇α and Hessian ∇²α.
# Below, φ ≡ S·∇α and φx, φy are its partials; each ∂S picks up factor k = π/L (chain rule).
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

# Build the callable exact velocity u_ex for this MMS instance.
function get_u_ex(mms::Paper2DMMS)
    return UExFunc(mms.U, mms.alpha_field.alpha_0, mms.alpha_field, mms.L)
end

# Pick physically sensible amplitudes for the manufactured fields. The velocity scale is just
# U; the pressure scale P is sized so it stays comparable to the dominant momentum term across
# regimes — viscous when Re, Da small (P ~ Uν/L) and growing with the convective/Darcy
# numbers via the (1 + Re + Da) factor. Re = UL/ν is the Reynolds number; Da = σ₀L²/(α_∞ν)
# is the Darcy number from the constant reaction coefficient σ₀ (0 if the law has no constant
# part). Returns (U_scale, P_scale).
function get_characteristic_scales(mms::Paper2DMMS)
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
    PExFunc <: Function

Callable exact pressure field p_ex(x) = P·cos(kx₁) sin(kx₂) with manufactured frequency k = π/L and
amplitude P from `get_characteristic_scales`. Mirrors `UExFunc`: subtyping `Function` lets Gridap treat
it as a field, and the `∇` overload below registers the exact analytic gradient so the forcing oracle (and
any strong-residual evaluation) sees machine-exact ∇p instead of differentiating a closure. This is the
single source of truth for p_ex / ∇p_ex (DRIV-07 — previously re-derived inline in three places).
"""
struct PExFunc <: Function
    P::Float64              # pressure amplitude
    L::Float64              # characteristic length: spatial frequency k = π/L
end
function (f::PExFunc)(x)
    k = pi / f.L            # spatial frequency k = π/L
    return f.P * cos(k*x[1])*sin(k*x[2])
end
# Exact pressure gradient ∇p_ex = (−P·k·sin(kx₁)sin(kx₂), P·k·cos(kx₁)cos(kx₂)); registered as Gridap's
# ∇ for PExFunc (mirroring grad_u_ex / ∇(::UExFunc)).
function grad_p_ex(f::PExFunc, x)
    k = pi / f.L
    return VectorValue(f.P * -k*sin(k*x[1])*sin(k*x[2]), f.P * k*cos(k*x[1])*cos(k*x[2]))
end
∇(f::PExFunc) = x -> grad_p_ex(f, x)

# Exact pressure p_ex(x) = P·cos(kx₁) sin(kx₂), amplitude P from get_characteristic_scales, frequency
# k = π/L. Returned as a PExFunc (one source of truth, with a registered analytic ∇).
function get_p_ex(mms::Paper2DMMS)
    U_amp, P_amp = get_characteristic_scales(mms)
    return PExFunc(P_amp, mms.L)
end

# Placeholder stub: the real momentum forcing is assembled analytically in
# evaluate_exactness_diagnostics (the strong residual evaluated pointwise on u_ex, p_ex).
# This returns a zero closure and is not used to drive the solver.
function build_exact_forcing(mms::Paper2DMMS, c_1, c_2)
    u_ex = get_u_ex(mms)
    p_ex = get_p_ex(mms)
    form = mms.formulation

    f_ex = x -> begin
        VectorValue(0.0, 0.0)
    end
    return f_ex
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
error and the O(h^{k+1}) rate is recoverable. [must-test]

Momentum: f_ex = (convection) + (pressure) + (reaction) − (viscous), each term assembled from
the SAME formulation operators the solver uses (reaction law, viscous operator), so f_ex is
exact for whichever concrete formulation `mms` carries.
Mass: g_ex enforces the porosity-weighted continuity ∇·(α u) (+ a pseudo-compressibility term).

`X`, `Y` are the trial/test multifield spaces (used only to interpolate u_ex, p_ex for the
error metrics); `c_1`, `c_2` are the stabilization constants threaded to the reaction speed;
`h_cf`, `tau_reg_lim` are accepted for harness signature parity.
"""
function evaluate_exactness_diagnostics(mms::Paper2DMMS, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)
    u_ex_func = get_u_ex(mms)
    p_ex_func = get_p_ex(mms)

    alpha_field = mms.alpha_field
    nu = mms.formulation.ν

    u_f = get_u_ex(mms)
    # p_ex value and ∇p_ex both come from the single PExFunc returned by get_p_ex (DRIV-07).

    reaction_law = mms.formulation.reaction_law

    # The reaction term is evaluated through the SAME `reaction_speed` routine the assembly uses
    # (src/models/regularization.jl) so the manufactured forcing is exact for the nonlinear
    # Forchheimer law as well as the constant-σ law. [known-fragility] `reaction_speed` carries
    # only a constant velocity floor (no mesh-dependent ν/h term), so it is mesh-independent by
    # construction — this pointwise oracle (which has no element `h`) is therefore exact
    # regardless of `h_floor_weight`. The `h` argument below is a dummy for signature parity.
    reg_oracle = mms.formulation.regularization
    h_unused = 1.0

    f_ex_oracle = function(x)
        u = u_f(x)
        grad_u = grad_u_ex(u_f, x)
        lap_u = lap_u_ex(u_f, x)
        
        grad_p = grad_p_ex(p_ex_func, x)
        
        A = alpha_field(x)
        grad_A = PorousNSSolver.grad_alpha(alpha_field, x)
        
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
        viscous_op = mms.formulation.viscous_operator
        if viscous_op isa PorousNSSolver.SymmetricGradientViscosity
            # Weak form: 2ν (α ∇ˢu ⊙ ∇ˢv), with ∇ˢu = sym(∇u). Strong divergence:
            #   ∇·(2Aν ∇ˢu) = 2ν (∇ˢu · ∇A) + Aν Δu + Aν ∇(∇·u).
            SPi_u = 0.5 * (grad_u + transpose(grad_u))     # ∇ˢu
            SPi_u_dot_grad_A = SPi_u ⋅ grad_A

            # ∇(∇·u) in closed form from the exact porosity gradient and Hessian.
            grad_div_u = grad_div_u_ex(u_f, x)
            visc = 2.0 * nu * SPi_u_dot_grad_A + nu * A * lap_u + nu * A * grad_div_u
        elseif viscous_op isa PorousNSSolver.DeviatoricSymmetricViscosity
            # Canonical operator. Weak form: 2ν (α ∇ᵈu ⊙ ∇ˢv) with the deviatoric (trace-free)
            # symmetric gradient ∇ᵈu = ∇ˢu − ½(∇·u)I. Strong divergence:
            #   ∇·(2Aν ∇ᵈu) = 2ν (∇ᵈu · ∇A) + 2Aν ∇·(∇ᵈu).
            # In 2D, ∇·(∇ᵈu) = ½Δu exactly, so the ∇(∇·u) dilatational term cancels — no Hessian
            # of the divergence is needed here.
            SPi_u = 0.5 * (grad_u + transpose(grad_u))     # ∇ˢu
            div_u_val = tr(grad_u)                          # ∇·u

            # ∇ᵈu · ∇A = ∇ˢu · ∇A − ½(∇·u)∇A.
            SPi_u_dot_grad_A = SPi_u ⋅ grad_A
            ViscProj_u_dot_grad_A = SPi_u_dot_grad_A - 0.5 * div_u_val * grad_A

            visc = 2.0 * nu * ViscProj_u_dot_grad_A + nu * A * lap_u
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
        grad_A = PorousNSSolver.grad_alpha(alpha_field, x)

        div_u_val = tr(grad_u)
        return mms.formulation.physical_epsilon * p_ex_func(x) + A * div_u_val + (grad_A ⋅ u)
    end

    g_ex_raw = CellField(g_ex_oracle, Ω)

    # Interpolants of the exact fields onto the FE spaces — used only for error metrics, never
    # as forcing (the forcing is the pointwise oracle above). X.spaces[1] is velocity, [2] pressure.
    u_ex_cf = interpolate_everywhere(x -> u_f(x), X.spaces[1])
    p_ex_cf = interpolate_everywhere(p_ex_func, X.spaces[2])

    println("Calculated MMS Exact Forcing...")
    return f_ex_raw, g_ex_raw
end
