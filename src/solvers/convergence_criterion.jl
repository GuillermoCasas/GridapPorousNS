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

# ε_C — mass (Philosophy A, SYMMETRIC with ε_M — the GATE)
    ε_C = ‖r_C‖ / D_C,
where `r_C` is the Euclidean norm of the **pressure (q-test) block of the SAME assembled stabilized
residual vector `b`** whose velocity block gives `r_M` (the caller passes it in, exactly like `r_M`),
and `D_C` is the dynamic mass-term envelope
    D_C = ‖∫ q (α∇·u + u·∇α)‖ + ‖∫ q (ε p)‖ + ‖∫ q g‖               (assembled over the pressure test space Q).
This is the Philosophy-A analogue of ε_M (spec §3.1): the numerator is what the solver ACTUALLY drives
to zero (the weak/algebraic continuity residual, subscales included) — so, unlike the old strong-form
measure, ε_C → 0 at the discrete solution just like ε_M and can be gated at 1e-9. The mass equation is
now treated IDENTICALLY to momentum: both residuals are brought down the same way. The envelope sums the
Galerkin (physical) mass terms assembled through the weak form over Q — the porous-flux divergence
∫q∇·(αu), the compressibility penalty ∫q(εp), and the mass source ∫qg — mirroring the Galerkin momentum
terms in D_M (the VMS stabilization terms in the q-block, τ₂/OSGS-projection + iterative penalty, live in
the numerator `r_C`, not the envelope — again exactly as for momentum). `ε_C ∈ [0,1]` up to the small
stabilization excess.

# ε_C^strong — mass (Philosophy B, DIAGNOSTIC ONLY — NOT the gate)
    ε_C^strong = ‖R_p‖ / (‖∇(α u)‖ + ‖g‖),   R_p = ε p + ∇·(α u) − g,   ∇(α u) = α ∇u + u ⊗ ∇α  (Frobenius).
This is the former gate quantity, retained purely as a physically-interpretable diagnostic ("how far from
incompressible is the flow"). Its L² strong-residual numerator FLOORS at the discretization error O(h^{kv})
(the FE velocity is only weakly divergence-free), so it CANNOT be driven to zero and must NOT gate the
solve — that is precisely why the gate moved to the Philosophy-A `ε_C` above. The PURE-divergence ratio
‖∇·(α u)‖/‖∇(α u)‖ carries the analytic ≤ √d bound ((tr A)² ≤ d (A:A)) and is checked separately as a
quadrature/assembly self-check. Computed by `mass_criterion` (unchanged), reported as `eps_C_strong`.

# Consistency requirements (read before wiring in)
- **Philosophy-A envelope consistency (BOTH blocks).** The sent-in `r_M`/`r_C` MUST be the velocity/
  pressure block of the SAME assembled residual whose terms `D_M`/`D_C` sum — same test spaces `V`/`Q`,
  same measure `dΩ`, same sign convention, same (Euclidean) vector norm. Then `‖r_M‖ ≤ Σ‖term‖ + ‖stab‖`
  (and likewise for `r_C`) is structural and `ε_M, ε_C ∈ [0,1]` up to the small stabilization excess (the
  numerator includes the VMS subscales, the denominator is the Galerkin force terms; spec §5.10). A
  *persistent* `ε_M ≫ 1` (or `ε_C ≫ 1`) does not mean the flow is unconverged — it means the residual and
  the term decomposition have drifted apart (different quadrature/space/sign). That is the bug to chase,
  not the criterion. Log both on a known case for the first few iterations and confirm they land in `[0,1]`.
- **Require ≥ 1 completed iteration (the k≥1 rule).** Do NOT evaluate or trust the verdict at the trivial
  initial iterate. The gate ratios `ε_M = ‖r_M‖/D_M`, `ε_C = ‖r_C‖/D_C` are then roundoff/roundoff. The
  `degenerate` flag (either GATE denominator `D_M`/`D_C` sitting at the underflow floor) marks exactly this
  state and is never accepted as converged. Separately, the DIAGNOSTIC strong-form `ε_C^strong ≤ √d` holds
  analytically, but the COMPUTED pure-divergence ratio divides a *signed* divergence sum (catastrophic
  cancellation) by an all-positive sum-of-squares, so it can exceed √d numerically at the trivial iterate;
  the √d self-check warning is suppressed when degenerate so it never trips on `0/0`-adjacent noise.

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
- `eps_M`, `eps_C` — the dimensionless momentum / mass GATE measures (both Philosophy-A: `‖r_M‖/D_M`,
                     `‖r_C‖/D_C`; both → 0 at the discrete solution). `converged` gates on these two.
- `converged`      — `eps_M ≤ tol_M && eps_C ≤ tol_C`. Do not accept it when `degenerate` (see k≥1 rule).
- `degenerate`     — a GATE denominator (`D_M` or `D_C`) sat at the underflow floor (the iterate carries
                     ~no force / flux structure, e.g. the all-zero initial guess). Verdict meaningless.
- `r_M`, `D_M`     — momentum numerator (velocity-block residual norm) and denominator envelope.
- `terms`          — NamedTuple `(convection, viscous, pressure_grad, resistance, body_force)`: the five
                     ‖·‖ contributions whose sum is `D_M`.
- `r_C`, `D_C`     — mass numerator (pressure-block residual norm — the weak continuity residual the
                     solver drives to zero) and denominator envelope (Galerkin mass-term magnitudes).
- `mass_terms`     — NamedTuple `(divergence, penalty, source)`: the three ‖·‖ contributions summing to `D_C`.
- `eps_C_strong`   — DIAGNOSTIC ONLY (not gated): the former strong-form ‖R_p‖/(‖∇(αu)‖+‖g‖), which floors
                     at O(h^{kv}) and so must not gate the solve. A physical "how incompressible" reading.
- `mass_num`, `mass_den`, `div_ratio` — the strong-form diagnostic's ‖R_p‖, its envelope ‖∇(αu)‖_F+‖g‖,
                     and the pure-divergence ratio ‖∇·(αu)‖/‖∇(αu)‖ (the √d self-check quantity).
- `sqrt_d`         — the √d ceiling for the strong-form self-check reference.
- `tol_M`, `tol_C` — the gate tolerances this verdict was measured against (`converged = eps_M ≤ tol_M ∧
                     eps_C ≤ tol_C`). Exposed so a caller can test the momentum leg alone (`eps_M ≤ tol_M`)
                     — used by the scale-free residual-floor accept (nonlinear.jl), which certifies a solve
                     whose momentum is converged but whose mass gate `eps_C` cannot reach `tol_C` because its
                     Philosophy-A envelope `D_C` collapses for a (near-)divergence-free flow.
"""
struct ConvergenceMeasure
    eps_M::Float64
    eps_C::Float64
    converged::Bool
    degenerate::Bool
    r_M::Float64
    D_M::Float64
    terms::NamedTuple
    r_C::Float64
    D_C::Float64
    mass_terms::NamedTuple
    eps_C_strong::Float64
    mass_num::Float64
    mass_den::Float64
    div_ratio::Float64
    sqrt_d::Float64
    tol_M::Float64
    tol_C::Float64
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
    mass_force_envelope(uh, ph, α, eps_val, g, Q, dΩ) -> (D_C, mass_terms)

The dynamic mass-term envelope `D_C` (Philosophy A — the exact mass-side analogue of
`momentum_force_envelope`): assemble each Galerkin (physical) continuity-equation term's pressure-block
load vector through the SAME weak form the residual uses, and sum their Euclidean norms. `Q` is the
pressure TEST space (the same one whose DOFs the numerator `r_C` lives on). The three terms mirror the
q-tested Galerkin mass form `∫ q (ε p + ∇·(αu) − g) dΩ` (`continuous_problem.jl`,
`build_stabilized_weak_form_residual`): the porous-flux divergence `∫ q (α∇·u + u·∇α)`, the
compressibility penalty `∫ q (ε p)`, and the mass source `∫ q g`. The VMS stabilization terms in the
q-block (τ₂/OSGS-projection coupling, the Codina iterative penalty) are DELIBERATELY excluded from the
envelope — they live in the numerator `r_C = ‖b[pressure block]‖`, exactly as the momentum subscales live
in `r_M` and not in `D_M`. Returns `D_C` and the per-term breakdown.
"""
function mass_force_envelope(uh, ph, α, eps_val, g, Q, dΩ)
    divergence = norm(assemble_vector(q -> ∫( q * (α * (∇⋅uh) + uh ⋅ ∇(α)) )dΩ, Q))   # ∫ q ∇·(αu)
    penalty    = norm(assemble_vector(q -> ∫( q * (eps_val * ph) )dΩ, Q))              # ∫ q (ε p)
    source     = norm(assemble_vector(q -> ∫( q * g )dΩ, Q))                           # ∫ q g
    D_C = divergence + penalty + source
    return D_C, (divergence = divergence, penalty = penalty, source = source)
end

"""
    mass_criterion(uh, ph, α, eps_val, g, d, dΩ) -> (eps_C_strong, mass_num, mass_den, div_ratio)

DIAGNOSTIC ONLY (no longer the gate — see the module header). The strong-form mass measure
ε_C^strong = ‖R_p‖ / (‖∇(α u)‖_F + ‖g‖)  (field L² norms). Retained as a physically-interpretable
"how far from incompressible" reading and as the √d self-check host; the GATE is the Philosophy-A
`‖r_C‖/D_C` (`mass_force_envelope` + the pressure block of the residual). This quantity FLOORS at the
discretization error O(h^{kv}) and therefore cannot be driven to zero.

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
    evaluate_convergence(r_M, r_C, uh, ph, α, ν, viscous_op, σ, f, eps_val, g, V, Q, dΩ, d; tol, tol_M, tol_C)
        -> ConvergenceMeasure

Evaluate the scale-free criterion at the current iterate. BOTH residuals are treated identically
(Philosophy A): `r_M` / `r_C` are the Euclidean norms of the assembled stabilized residual's velocity /
pressure blocks — the momentum and continuity residuals the solver actually drives to zero — supplied by
the caller (the same `b`, its two field blocks). The remaining arguments are the iterate (`uh`, `ph`),
material/forcing data (`α`, `ν`, `viscous_op`, the reaction field `σ`, body force `f`, penalty `eps_val`,
mass source `g`), the velocity/pressure TEST spaces `V`/`Q`, the measure `dΩ`, and the spatial dimension
`d`. Tolerances default to `tol`.

Computes the two GATE measures ε_M = ‖r_M‖/D_M and ε_C = ‖r_C‖/D_C (both → 0 at the discrete solution),
sets `converged = ε_M ≤ tol_M ∧ ε_C ≤ tol_C`, and ALSO evaluates the strong-form DIAGNOSTIC
ε_C^strong = ‖R_p‖/(‖∇(αu)‖+‖g‖) (via `mass_criterion`) — reported but never gated — plus the √d
self-check warning on the pure-divergence ratio (NOT a hard assert; quadrature round-off can nudge it).
"""
function evaluate_convergence(r_M::Float64, r_C::Float64, uh, ph, α, ν, viscous_op, σ, f, eps_val, g, V, Q, dΩ, d::Int;
                              tol::Float64, tol_M::Float64 = tol, tol_C::Float64 = tol)
    # Momentum GATE (Philosophy A): weak velocity-block residual over the Galerkin force envelope.
    D_M, terms = momentum_force_envelope(uh, ph, α, ν, viscous_op, σ, f, V, dΩ)
    eps_M = r_M / _floor(D_M)
    # Mass GATE (Philosophy A, symmetric): weak pressure-block residual over the Galerkin mass envelope.
    D_C, mass_terms = mass_force_envelope(uh, ph, α, eps_val, g, Q, dΩ)
    eps_C = r_C / _floor(D_C)
    # Strong-form DIAGNOSTIC (not gated): floors at O(h^{kv}); hosts the √d pure-divergence self-check.
    eps_C_strong, mass_num, mass_den, div_ratio = mass_criterion(uh, ph, α, eps_val, g, d, dΩ)

    # Degenerate state: a GATE denominator (D_M or D_C) at the underflow floor means the iterate carries
    # no force / no flux structure (e.g. the all-zero initial guess). The gate ratios are then roundoff/
    # roundoff and the verdict is meaningless — the caller applies the k≥1 rule and does not accept here.
    degenerate = (D_M < eps(Float64)) || (D_C < eps(Float64))

    # √d self-check on the PURE-divergence ratio ‖∇·(αu)‖/‖∇(αu)‖ (the analytically √d-bounded quantity).
    # A genuine violation on a developed iterate signals a quadrature/assembly bug — warn, do not assert.
    # Suppressed when degenerate, where a numerical √d violation is just roundoff/roundoff (cancellation).
    sqrt_d = sqrt(d)
    if !degenerate && div_ratio > sqrt_d * (1.0 + 1e-2)
        @warn "convergence_criterion: pure-divergence ratio exceeds the √d ceiling beyond numerical tolerance — check quadrature/assembly consistency" div_ratio sqrt_d
    end

    converged = (eps_M ≤ tol_M) && (eps_C ≤ tol_C)
    return ConvergenceMeasure(eps_M, eps_C, converged, degenerate, r_M, D_M, terms,
                              r_C, D_C, mass_terms, eps_C_strong, mass_num, mass_den, div_ratio, sqrt_d,
                              tol_M, tol_C)
end
