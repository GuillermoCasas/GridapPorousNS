# src/solvers/convergence_criterion.jl
#=
    convergence_criterion.jl

# Role
A **scale-free stopping criterion** for the outer nonlinear (Picard / Newton) iteration, kept
deliberately SEPARATE from the iteration machinery (`nonlinear.jl`, `solver_core.jl`). The solver
algorithm decides *how* to step; this module decides *whether the current iterate is converged*, and
returns the two dimensionless residual measures it bases that on. Nothing here mutates solver state.

The spec this implements is `docs/solver/nonlinear-convergence-criterion-prompt.md`. Read it for the
full rationale. The criterion is

    converged  ⇔  ε_M ≤ tol_M   and   ε_C ≤ tol_C

with a **momentum** measure ε_M and a **mass** measure ε_C, each a residual divided by a scale that is
*measured from the current iterate and known material data* — never from an a-priori characteristic
velocity/length/pressure or a global Re/Da. The point is regime-robustness: whichever physical
mechanism dominates (viscous, convective, or porous-resistance) automatically sets the scale.

# ε_M — momentum (term-magnitude envelope)
The numerator is the norm of the assembled **stabilized** nonlinear residual the solver actually drives
to zero (velocity block); the caller passes it in as `r_M`. The denominator is the dynamic envelope of
the momentum force magnitudes,

    D_M = ‖α u·∇u‖ + ‖2 ∇·(α ν Πˢ∇u)‖ + ‖α ∇p‖ + ‖σ(α,u) u‖ + ‖f‖,           ε_M = ‖r_M‖ / D_M,

where Πˢ is the deviatoric-symmetric projector of the viscous operator. We use **Philosophy A**
(spec §3.1): every term is assembled THROUGH THE WEAK FORM (the viscous term is the integrated-by-parts
first-derivative quantity ⟨εˢ(v), 2 α ν Πˢ∇u⟩ — `weak_viscous_operator`), with the same vector norm
(Euclidean) used for the residual numerator. This (a) keeps the denominator in the same space/units as
the numerator, (b) needs no second derivatives and so works for P1/P1 with no special-casing, and
(c) drives ε_M → 0 at the discrete solution (unlike the strong-residual "Philosophy B", whose numerator
floors at O(h^k)). The sum-of-norms is a deliberately CONSERVATIVE envelope (D_M ≥ ‖Σ terms‖), so it
never under-estimates the scale and so cannot report convergence early through term cancellation.

# ε_C — mass (residual over a term-magnitude envelope)
    ε_C = ‖R_p‖ / (‖∇(α u)‖ + ‖g‖),   R_p = ε p + ∇·(α u) − g,   ∇(α u) = α ∇u + u ⊗ ∇α  (Frobenius),
the genuine mass-equation residual R_p normalized by a term-magnitude envelope. NOTE the −g: the
continuity source is SUBTRACTED, so ε_C → 0 at the discrete solution even for a forced/manufactured
problem (without it ε_C would floor at ‖g‖/‖∇(αu)‖, measuring the source instead of the residual). The
denominator mirrors the momentum envelope D_M (which carries the body force ‖f‖): the robust
flux-variation scale ‖∇(α u)‖ PLUS the source magnitude ‖g‖, so a strongly-forced mass equation is
measured against a scale that reflects its forcing. ‖∇(α u)‖ stands in for the divergence ‖∇·(α u)‖
(which can collapse for a near-incompressible flow) because boundary-layer shear keeps it bounded away
from zero. All norms are global L²(Ω), built only from `u`, `p`, and the known fields `α`, `g`. The `ε p`
penalty is kept for strictness (negligible at production ε* ~ 1e-4). When ε=0, g=0, α constant it reduces
to the textbook ‖∇·u‖/‖∇u‖. The PURE-divergence ratio ‖∇·(α u)‖/‖∇(α u)‖ carries the analytic ≤ √d bound
((tr A)² ≤ d (A:A)) and is checked separately as a quadrature/assembly self-check.

# Consistency requirements (read before wiring in)
- **Philosophy-A envelope consistency.** The sent-in `r_M` MUST be the velocity block of the SAME
  assembled residual whose terms `D_M` sums — same test space `V`, same measure `dΩ`, same sign
  convention, same (Euclidean) vector norm. Then `‖r_M‖ ≤ Σ‖term‖ + ‖stab‖` is structural and
  `ε_M ∈ [0,1]` up to the small stabilization excess (the numerator includes the VMS subscales, the
  denominator is the Galerkin force terms; spec §5.10). A *persistent* `ε_M ≫ 1` does not mean the
  flow is unconverged — it means the residual and the term decomposition have drifted apart (different
  quadrature/space/sign). That is the bug to chase, not the criterion. Log `ε_M` on a known case for
  the first few iterations and confirm it lands in `[0,1]`.
- **Require ≥ 1 completed iteration (the k≥1 rule).** Do NOT evaluate or trust the verdict at the
  trivial initial iterate. Both ratios are then roundoff/roundoff; `ε_C ≤ √d` holds analytically, but
  the COMPUTED ε_C divides a *signed* divergence sum (subject to catastrophic cancellation) by an
  all-positive sum-of-squares, so it can exceed √d numerically. The `degenerate` flag (a denominator
  sitting at the underflow floor) marks exactly this state; the √d self-check warning is suppressed
  when degenerate so it never trips on `0/0`-adjacent noise.

# Scope / what this is NOT
- It introduces NO a-priori scale (no U, L, P, Re, Da). The optional pressure-normalized fallback of
  spec §6 is intentionally NOT implemented: it divides by ‖p‖, which is a measured solution scale that
  is indeterminate under all-Dirichlet velocity BCs AND mesh-coupled through τ₁ — strictly worse than,
  and not genuinely scale-free unlike, the flux-gradient ratio.
- Denominators are guarded by a pure machine-eps underflow floor (not a problem scale), so a trivial
  all-zero state does not divide by zero. (The √d bound makes ε_C robust whenever ‖∇(α u)‖ is well
  above roundoff; the `degenerate` flag + the k≥1 rule cover the roundoff regime.)
- It is decoupled: a pure function of the iterate + material data + the sent-in residual. Wiring it
  into the stopping rule of `solve_system` / `solve_osgs_stage!` is a separate, caller-side decision.
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

"""
    ConvergenceMeasure

The result of one convergence evaluation. `converged` is the verdict; the rest are the diagnostics the
spec asks to report every iteration (ε_M, ε_C, and the per-term breakdown of the momentum envelope so
a stalled solve reveals which force balance is limiting it).

Fields:
- `eps_M`, `eps_C` — the dimensionless momentum / mass residual measures.
- `converged`      — `eps_M ≤ tol_M && eps_C ≤ tol_C`. Do not accept it when `degenerate` (see k≥1 rule).
- `degenerate`     — a denominator sat at the underflow floor (the iterate carries ~no force / flux
                     structure, e.g. the all-zero initial guess). The verdict is meaningless here.
- `r_M`, `D_M`     — momentum numerator (sent-in residual norm) and denominator envelope.
- `terms`          — NamedTuple `(convection, viscous, pressure_grad, resistance, body_force)`: the five
                     ‖·‖ contributions whose sum is `D_M`.
- `mass_num`, `mass_den` — ‖R_p‖ = ‖ε p + ∇·(α u) − g‖ and the envelope ‖∇(α u)‖_F + ‖g‖ (numerator /
                     denominator of `eps_C`).
- `sqrt_d`         — the √d ceiling for `eps_C` (self-check reference).
"""
struct ConvergenceMeasure
    eps_M::Float64
    eps_C::Float64
    converged::Bool
    degenerate::Bool
    r_M::Float64
    D_M::Float64
    terms::NamedTuple
    mass_num::Float64
    mass_den::Float64
    sqrt_d::Float64
end

# Pure machine-eps underflow floor (spec edge case 1): keeps a denominator strictly positive WITHOUT
# injecting a problem scale. It only ever bites the degenerate all-zero state; for any developed iterate
# the physical terms (and, for ε_C, the √d bound) keep the ratios well-posed.
_floor(den) = max(den, eps(Float64))

# L²(Ω) norm of a scalar/vector/tensor field `w`: sqrt(∫ w⊙w dΩ). `⊙` is the full (double) contraction,
# so this is the Frobenius L² norm for tensors and the ordinary L² norm for scalars/vectors.
_l2(w, dΩ) = sqrt(max(0.0, sum(∫(w ⊙ w)dΩ)))

"""
    momentum_force_envelope(uh, ph, α, ν, viscous_op, σ, f, V, dΩ) -> (D_M, terms)

The dynamic momentum-force envelope `D_M` (Philosophy A): assemble each physical momentum term's
velocity-block load vector through the weak form and sum their Euclidean norms. `viscous_op` is the
viscous operator instance (selects the deviatoric-symmetric projector Πˢ via `weak_viscous_operator`);
`σ` is the reaction coefficient field σ(α,u)=a(α)+b(α)|u| evaluated at the iterate; `V` is the velocity
TEST space (the same one whose DOFs the residual numerator `r_M` lives on). Returns `D_M` and the
per-term breakdown.
"""
function momentum_force_envelope(uh, ph, α, ν, viscous_op, σ, f, V, dΩ)
    convection   = norm(assemble_vector(v -> ∫( v ⋅ (α * (∇(uh)' ⋅ uh)) )dΩ, V))   # α u·∇u
    viscous      = norm(assemble_vector(v -> ∫( weak_viscous_operator(viscous_op, uh, v, α, ν) )dΩ, V))  # ⟨εˢ(v), 2ανΠˢ∇u⟩
    pressure_grad = norm(assemble_vector(v -> ∫( v ⋅ (α * ∇(ph)) )dΩ, V))           # α ∇p
    resistance   = norm(assemble_vector(v -> ∫( v ⋅ (σ * uh) )dΩ, V))               # σ u
    body_force   = norm(assemble_vector(v -> ∫( v ⋅ f )dΩ, V))                      # f
    D_M = convection + viscous + pressure_grad + resistance + body_force
    return D_M, (convection = convection, viscous = viscous, pressure_grad = pressure_grad,
                 resistance = resistance, body_force = body_force)
end

"""
    mass_criterion(uh, ph, α, eps_val, g, d, dΩ) -> (eps_C, mass_num, mass_den, div_ratio)

The scale-free mass measure ε_C = ‖R_p‖ / (‖∇(α u)‖_F + ‖g‖)  (field L² norms).

NUMERATOR — the **genuine mass-equation residual** R_p = ε p + ∇·(α u) − g: it SUBTRACTS the mass source
`g` (the continuity-equation forcing; `g ≡ 0` for the unforced physical problem, `g ≠ 0` for a
manufactured solution with a non-solenoidal flux). Without the `−g` term ε_C would measure the raw flux
divergence, which at the discrete solution equals the (nonzero) source and so floors at ‖g‖/‖∇(αu)‖
instead of → 0. With it, R_p → the discretization floor O(h^{kv}) at convergence.

DENOMINATOR — a mass-equation TERM-MAGNITUDE ENVELOPE, exactly analogous to the momentum `D_M` (which
includes the body force ‖f‖): the robust flux-variation scale ‖∇(α u)‖_F plus the source magnitude ‖g‖.
Adding ‖g‖ matches how f enters D_M — the residual is measured against the full set of balanced terms,
so a strongly-forced mass equation is normalized by a scale that actually reflects the forcing, not just
the flux gradient. ‖∇(α u)‖_F is used in place of the divergence ‖∇·(α u)‖ (which can collapse for a
near-incompressible flow) because boundary-layer shear keeps it robustly bounded away from zero. `∇·(α u)`
expands for variable porosity as `α(∇·u) + u·∇α`, and `∇(α u) = α∇u + u⊗∇α`. Reduces to ‖∇·u‖/‖∇u‖ for
constant α, ε=0, g=0.

Also returns `div_ratio = ‖∇·(α u)‖/‖∇(α u)‖`, the PURE-divergence ratio that carries the analytic
≤ √d bound (the genuine √d self-check; neither the g-subtracted numerator nor the g-augmented denominator
is √d-bounded). When g=0 and ε=0, ε_C and div_ratio coincide.
"""
function mass_criterion(uh, ph, α, eps_val, g, d, dΩ)
    div_flux  = α * (∇ ⋅ uh) + uh ⋅ ∇(α)            # ∇·(α u)
    mass_res  = eps_val * ph + div_flux - g          # R_p = ε p + ∇·(α u) − g  (genuine mass residual → 0)
    flux_grad = α * ∇(uh) + outer(uh, ∇(α))         # ∇(α u)  (second-order tensor)
    mass_num  = _l2(mass_res, dΩ)
    grad_norm = _l2(flux_grad, dΩ)                   # ‖∇(α u)‖_F — robust flux-variation scale
    g_norm    = _l2(g, dΩ)                           # ‖g‖ — mass-source magnitude (cf. ‖f‖ in D_M)
    mass_den  = grad_norm + g_norm                   # term-magnitude envelope: flux-gradient proxy + source
    eps_C     = mass_num / _floor(mass_den)
    div_ratio = _l2(div_flux, dΩ) / _floor(grad_norm) # ‖∇·(αu)‖/‖∇(αu)‖ ≤ √d  (pure-divergence √d self-check)
    return eps_C, mass_num, mass_den, div_ratio
end

"""
    evaluate_convergence(r_M, uh, ph, α, ν, viscous_op, σ, f, eps_val, g, V, dΩ, d; tol, tol_M, tol_C)
        -> ConvergenceMeasure

Evaluate the scale-free criterion at the current iterate. `r_M` is the Euclidean norm of the assembled
stabilized momentum residual's velocity block (Philosophy A numerator — what the solver drives to zero),
supplied by the caller. The remaining arguments are the iterate (`uh`, `ph`), material/forcing data
(`α`, `ν`, `viscous_op`, the reaction field `σ`, body force `f`, penalty `eps_val`, mass source `g`), the
velocity test space `V`, the measure `dΩ`, and the spatial dimension `d`. Tolerances default to `tol`.

Computes ε_M = r_M / D_M (momentum) and ε_C = ‖R_p‖/(‖∇(αu)‖+‖g‖) (mass, with R_p = εp + ∇·(αu) − g),
sets `converged`, and logs a warning if the pure-divergence ratio exceeds √d beyond a small numerical
margin (a self-check, NOT a hard assert — quadrature round-off can nudge it slightly).
"""
function evaluate_convergence(r_M::Float64, uh, ph, α, ν, viscous_op, σ, f, eps_val, g, V, dΩ, d::Int;
                              tol::Float64, tol_M::Float64 = tol, tol_C::Float64 = tol)
    D_M, terms = momentum_force_envelope(uh, ph, α, ν, viscous_op, σ, f, V, dΩ)
    eps_M = r_M / _floor(D_M)
    eps_C, mass_num, mass_den, div_ratio = mass_criterion(uh, ph, α, eps_val, g, d, dΩ)

    # Degenerate state: a denominator at the underflow floor means the iterate carries no force / no
    # flux structure (e.g. the all-zero initial guess). The ratios are then roundoff/roundoff and the
    # verdict is meaningless — the caller must apply the k≥1 rule and not accept convergence here.
    degenerate = (D_M < eps(Float64)) || (mass_den < eps(Float64))

    # √d self-check on the PURE-divergence ratio ‖∇·(αu)‖/‖∇(αu)‖ (the analytically √d-bounded quantity —
    # neither the g-subtracted numerator nor the g-augmented denominator of ε_C is √d-bounded). A genuine
    # violation on a developed iterate signals a quadrature/assembly bug — warn, do not assert. Suppressed
    # when degenerate, where a numerical √d violation is just roundoff/roundoff (signed cancellation).
    sqrt_d = sqrt(d)
    if !degenerate && div_ratio > sqrt_d * (1.0 + 1e-2)
        @warn "convergence_criterion: pure-divergence ratio exceeds the √d ceiling beyond numerical tolerance — check quadrature/assembly consistency" div_ratio sqrt_d
    end

    converged = (eps_M ≤ tol_M) && (eps_C ≤ tol_C)
    return ConvergenceMeasure(eps_M, eps_C, converged, degenerate, r_M, D_M, terms, mass_num, mass_den, sqrt_d)
end
