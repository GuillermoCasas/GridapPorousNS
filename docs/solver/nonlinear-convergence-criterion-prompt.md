# Prompt: scale-free convergence criterion for the porous Navier–Stokes nonlinear solver

> **STATUS (2026-07-04) — the MASS measure in this spec has evolved.** Production now uses the
> **Route-B "Philosophy-A" algebraic** mass gate `ε_C = ‖r_C‖ / D_C` — the norm of the assembled
> stabilized mass residual over a term-magnitude envelope, **symmetric with `ε_M`** and gated at the
> same `~1e-9` level. The strong-form / flux-gradient measure specified below
> (`ε_C = ‖∇·(α u)‖ / ‖∇(α u)‖`, and the `−g`-subtracted variant) is now the **diagnostic
> `eps_C_strong`**, no longer the gate. A **`residual_floor_reached` scale-free accept** was also added:
> it accepts machine-floor-converged cells whose `ε_C` cannot reach `tol` because `D_C` collapses for
> near-divergence-free flow (the mass envelope → 0 when the flow is essentially incompressible). The
> **momentum (Philosophy A) spec below is still current and unchanged.** See
> [`docs/mms/route-b-2d-sweep-status.md`](../mms/route-b-2d-sweep-status.md).

You are implementing the **stopping criterion for the outer nonlinear (Picard / Newton) iteration**
of a stabilized finite element solver for the stationary generalized (porous) Navier–Stokes
system. Read the whole spec before writing code. The rationale notes are there so that, when you
hit an ambiguity, you resolve it the way the design intends rather than "helpfully" reintroducing
something the design deliberately excludes.

---

## 0. Summary (what to build)

Stop the outer loop when

```
converged  ⇔  max(ε_M, ε_C) ≤ tol           (or separate tol_M, tol_C)
```

with

- **Momentum**  `ε_M = ‖r_M^k‖ / D_M^k`
  where `D_M^k = ‖α u·∇u‖ + ‖2 ∇·(α ν Π u)‖ + ‖α ∇p‖ + ‖σ(u) u‖ + ‖f‖`
- **Mass**  `ε_C = ‖∇·(α u^k)‖ / ‖∇(α u^k)‖`  with the guaranteed bound `ε_C ≤ √d`

All norms are global `L²(Ω)` norms (`‖w‖ = sqrt(∫_Ω |w|² dΩ)`), accumulated by quadrature in a
single assembly pass. `Π` is the deviatoric∘symmetric projector `Π^DS` used in the viscous term;
`d ∈ {2,3}` is the spatial dimension. Report `ε_M` and `ε_C` separately every iteration.

---

## 1. Hard constraint (do not violate)

The criterion **must be computable from the current iterate `U^k = (u^k, p^k)` and known
material/mesh data alone** (`α(x)`, `ν`, `σ = a(α) + b(α)|u|`, `f`, element size `h_K`).

**Do NOT introduce, request, or hard-code any a-priori characteristic scale** — no characteristic
velocity `U`, length `L`, pressure scale `P`, nor any global Reynolds `Re` or Damköhler `Da`
number. Those quantities are bespoke to a manufactured-solution test where they happen to be known;
a production criterion cannot assume them. If at any point you find yourself needing such a scale,
you have taken a wrong turn — re-read §3.2 and §4.2, which explain how the needed scale is *measured
from the iterate* instead of supplied.

(Element-level `Re_h`, `Da_h` built from `h_K` and the iterate are allowed — they are not a-priori
scales. They appear only in the optional fallback, §6.)

---

## 2. Governing equations and residuals

Strong form (kinematic; density absorbed into `ν`, `p`, `σ`):

```
momentum:   α u·∇u − 2∇·(α ν Π∇u) + α ∇p + σ(α,u) u = f
mass:       ε p + ∇·(α u) = 0
```

with `σ(α,u) = a(α) + b(α)|u|` (Darcy + Forchheimer), `ε ≥ 0` a small compressibility (iterative
penalty), equal-order `P_k/P_k` velocity–pressure, ASGS or OSGS VMS stabilization.

- **Momentum residual** `r_M^k` = momentum residual evaluated at `U^k`.
- **Mass residual** `r_C^k = ∇·(α u^k)` (`+ ε p^k` if `ε>0`; negligible — see §4.1 / edge case 7).

---

## 3. Momentum normalization

### 3.1 Numerator `‖r_M^k‖` — pick one philosophy and use it for the denominator too

**Philosophy A — algebraic (recommended).** Measure what the solver actually drives to zero:
the norm of the assembled **stabilized** nonlinear residual *vector* (ASGS/OSGS, subscales
included), velocity block. Use a fixed vector norm (Euclidean or mass-matrix-weighted) — whichever
you pick, use the same norm for every term in `D_M`. Build the denominator terms (§3.2) by
assembling each physical term's velocity-block contribution **through the same weak form**: the
viscous term is then the integrated-by-parts `⟨∇^S v, 2 α ν Π∇u^k⟩`, a first-derivative quantity —
**no second derivatives, no special-casing by polynomial order**. This is the default.

**Philosophy B — pointwise force density (only if you specifically want a force-balance picture).**
Use `L²(Ω)` norms of the strong-form force-density fields directly. Then `r_M^k` is the strong
residual field. Caveat: the strong residual of a finite-element solution does **not** vanish at the
discrete solution — it floors at the discretization truncation level `O(h^k)` in a negative norm —
so set `tol` above that floor and don't expect machine-zero. Also the viscous term needs a second
derivative (see edge case below). Prefer A unless you have a reason.

### 3.2 Denominator `D_M^k` — dynamic term-magnitude envelope

```
D_M^k = ‖α u^k·∇u^k‖      (convection)
      + ‖2 ∇·(α ν Π∇u^k)‖ (viscous)
      + ‖α ∇p^k‖          (pressure gradient)
      + ‖σ(u^k) u^k‖       (resistance: Darcy + Forchheimer)
      + ‖f‖                (body force)
```

### 3.3 Rationale

- **Why a sum of term magnitudes is regime-robust.** At the solution the momentum terms do not
  vanish — they *balance*. Whichever physical mechanism dominates (viscous as `Re,Da→0`, convection
  as `Re→∞`, resistance as `Da→∞`) automatically enters the sum and sets the scale, with no regime
  parameter to choose. This is the same reason the construction works for ordinary Navier–Stokes;
  it carries over unchanged. (Offline check, not needed by the code: non-dimensionalizing sends the
  four operator terms to coefficients `Re, 1, (1+Re+Da), Da`, so `D_M ~ (1+Re+Da) ·
  α∞νU/L²`, i.e. the natural force envelope. The pressure-gradient term alone has coefficient
  `1+Re+Da` and is therefore leading-order in *every* regime — so `‖α∇p^k‖` by itself is already a
  valid scale once `p^k` has developed, which is why the envelope stays robust even if another term
  is momentarily mis-estimated.)
- **Why dynamic (recomputed each iteration), not a closed form.** `σ` depends on `|u^k|` through
  Forchheimer and `α` varies in space, so the true local resistance is *not* the nominal global
  `Da`. The dynamic sum reads the actual local magnitudes; a frozen analytic prefactor cannot.
- **Why include `‖f‖`.** In forced/manufactured problems `f` is one of the balanced forces and can
  be the dominant one; including it prevents the scale from collapsing if the operator terms cancel
  among themselves. When `f = 0` (e.g. inlet-driven flow) it simply contributes nothing — correct.
- **Why the viscous term is unproblematic under Philosophy A.** The weak form has already
  integrated it by parts, so it is first-order. Under Philosophy B with `P_1`, `∇·(ανΠ∇u_h)` is
  ~0 element-wise; accept the ~0 contribution — the co-dominant pressure-gradient term (which
  balances the viscous force in that regime) supplies the viscous scale. Do **not** patch this with
  an inverse-estimate factor `C_inv/h`: it overestimates the viscous force on fine meshes and makes
  `ε_M` over-optimistic.

---

## 4. Mass normalization

### 4.1 Numerator `‖∇·(α u^k)‖`

Use the divergence of the porous flux `q^k = α u^k`. This is the quantity that → 0 at the fixed
point: with the iterative penalty (the `ε p^{n−1}` term on the RHS) the compressibility terms cancel
at convergence, leaving exactly the incompressibility residual. If `ε > 0`, the consistent numerator
is `‖ε p^k + ∇·(α u^k)‖`, but `ε* ~ 1e−4` so `‖∇·(α u^k)‖` alone is equivalent and clearer.

### 4.2 Denominator `‖∇(α u^k)‖` — the flux-gradient (Frobenius) norm

```
∇(α u^k) = α ∇u^k + u^k ⊗ ∇α        (full second-order tensor; Frobenius norm)
ε_C = ‖∇·(α u^k)‖ / ‖∇(α u^k)‖ ≤ √d
```

### 4.3 Rationale

- **Why this is the right scale-free measure, and not ad hoc.** It is the pressure-normalized
  "relative pressure error from mass imbalance" with the divergence→pressure conversion factor
  *measured from the iterate instead of supplied*. The regime-robust conversion factor is
  `Φ/α∞ = ‖p^k‖ / ‖∇(α u^k)‖` (because `‖p^k‖ ~ P` and `‖∇(α u^k)‖ ~ α∞U/L`, whose ratio reproduces
  the full viscous+inertial+Darcy envelope `Φ/α∞` in every regime). Substituting it into the
  pressure-normalized criterion `ε_C = (Φ/α∞)·‖∇·(α u^k)‖/‖p^k‖` cancels `‖p^k‖` identically,
  leaving the flux-gradient ratio. So the gradient ratio *is* the pressure idea with the a-priori
  factor removed; it inherits the relative-pressure-error reading wherever the pressure scale is
  flow-consistent, but never requires `p`.
- **Why it is genuinely scale-free.** Both norms are built from `u^k` and the known field `α`. No
  characteristic scale enters. Dimensionless by construction.
- **Why the `√d` bound exists, and use it as a self-check.** `∇·q = tr(∇q)` and `(tr A)² ≤ d (A:A)`
  pointwise, so `‖∇·q‖² ≤ d ‖∇q‖_F²`. Therefore `ε_C ≤ √d` always; a computed value above `√d`
  signals a quadrature/projection/assembly bug — assert or log it.
- **Why keep the `u ⊗ ∇α` term.** It is real flux structure forced by the porosity gradient; the
  ratio then measures dilatation against the *total* flux variation. When `α` is constant (incl.
  `α ≡ 1`, classical Navier–Stokes) it vanishes and `ε_C` reduces to the textbook
  `‖∇·u‖/‖∇u‖` — graceful, no special-casing needed.
- **Why global, not pointwise.** A pointwise ratio blows up wherever `∇(α u) → 0` (near-uniform
  patches). Use domain-integrated `L²` norms. With no-slip walls and the parabolic inlet there is
  always boundary-layer shear, so `‖∇(α u^k)‖` is robustly bounded away from zero in practice.

---

## 5. Edge cases (apply all)

1. **Trivial initial guess `u^0 = 0`.** `D_M^0` and `‖∇(α u^0)‖` may be 0 → `0/0`. Guard every
   denominator with a *pure underflow floor* `den = max(den, eps(eltype))` (machine epsilon, **not**
   a problem scale), and do not declare convergence at `k = 0` — require at least one completed
   iteration. If `u ≡ 0` genuinely solves the system (zero BCs and `f = 0`), the numerators are also
   ~0 and the floor yields convergence without inventing a scale.
2. **BC-driven flow, `f = 0` (e.g. inlet-driven).** The denominator must not rely on `f`; once the
   iterate is nonzero the internal terms carry `D_M`. `‖f‖ = 0` contributing nothing is correct.
3. **Uniform porosity `α ≡ const` (incl. `α ≡ 1`).** `∇α = 0` ⇒ `‖∇(α u)‖ = α‖∇u‖`, and `ε_C`
   reduces to `‖∇·u‖/‖∇u‖`. Expected; do not special-case.
4. **Locally near-uniform flux.** Never a pointwise ratio (see §4.3). Always global `L²`.
5. **`ε_C > √d`.** Analytically impossible ⇒ treat as a bug indicator (assert/log).
6. **All-Dirichlet / pressure indeterminacy (`Γ_N = ∅`, `ε = 0`).** Pressure is defined up to a
   constant (fixed by the zero-mean condition). The flux-gradient ratio is immune — it uses no
   `‖p‖`. Do **not** build a pressure-normalized mass measure in this case.
7. **Compressibility `ε > 0` (iterative penalty).** Strict numerator `‖ε p^k + ∇·(α u^k)‖`, but
   `ε* ~ 1e−4`, so `‖∇·(α u^k)‖` is equivalent and preferred. The penalty's RHS term cancels at the
   fixed point.
8. **Newton vs Picard.** Always measure the **genuine nonlinear residual** evaluated at `U^k`
   (re-assemble the operator), never the linearized / inner linear-solve residual — the latter → 0
   *inside* each step and does not reflect outer convergence. (A common failure mode: gating on the
   inner residual makes the outer loop stop too early or behave erratically.)
9. **Tolerance vs floor.** `ε_M` and `ε_C` cannot fall below the level set by the discretization
   (the stabilized scheme permits a small nonzero `∇·(α u_h)` at the discrete solution, `O(h^k)`)
   and by `ε`. Set `tol` **above** this floor. Diagnostic: a fixed outer-iteration count that is
   independent of `tol` is the signature of either `tol` below the floor, or measuring a quantity
   that does not decrease (see edge case 8) — check both.
10. **Stabilization terms.** Numerator = the method's full residual (ASGS/OSGS, subscales included),
    for consistency with the iteration. Denominator = the Galerkin physical force terms in §3.2;
    adding the stabilization terms to `D_M` is optional and harmless (they are bounded by the
    Galerkin terms by design).
11. **Element-wise vs global accumulation.** Accumulate `‖·‖²` per element, then sum and take the
    square root (one assembly pass). The global ratio is the default. If a regime varies *strongly*
    across the mesh, the most faithful variant is an element-wise envelope (form the ratio per
    element, then reduce by `max` or an `ℓ²`-over-elements norm); offer this only if needed.

---

## 6. Optional fallback (documented, not the default): pressure-normalized mass measure, still scale-free

If a pressure-referenced mass measure is explicitly wanted (to read the criterion directly as a
relative pressure error), replace the global `Re, Da` by **element-level** `Re_h = |u^k|_K h_K / ν`
and `Da_h = σ_K h_K² / (α_K ν)` — the same quantities already inside `τ_1`. Then

```
ε_C^alt = ‖ (h²/(α_K τ_{1,K})) · ∇·(α u^k) ‖ / ‖p^k‖
        ~ ‖ (ν (1 + Re_h + Da_h)/α_K) · ∇·(α u^k) ‖ / ‖p^k‖
```

i.e. the divergence→pressure conversion factor expressed through your own stabilization parameter
`τ_1` (so still no bespoke scales — only `h_K`, the iterate, and material data). It is mesh-coupled
and fussier than the flux-gradient ratio and reduces to it in spirit. Use only if the explicit
pressure reading is required; otherwise prefer §4.

---

## 7. Implementation notes (Julia / Gridap)

- Reuse the existing residual weak form for `r_M^k` (Philosophy A) so the stabilized residual and
  the convergence measure stay consistent by construction.
- `L²(Ω)` norm of a field `w`: `sqrt(sum(∫(w⋅w)dΩ))` (use `⊙`/`inner` and Frobenius for tensors).
- `∇·(α u)` and `∇(α u)` via Gridap differential operators on the `α`-field interpolated in the
  same FE space as the solution (consistent with the paper's nodal interpolation of `α`).
- Guard denominators: `den = max(den, eps(Float64))`.
- Log `ε_M`, `ε_C`, and each `D_M` term per iteration; the per-term breakdown is the diagnostic that
  tells you which balance is limiting convergence when the loop stalls.
- Assert `ε_C ≤ sqrt(d) * (1 + tol)` as a cheap correctness check.
