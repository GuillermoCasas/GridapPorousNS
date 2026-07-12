# Audit of the stability and continuity results

**Scope.** `lemma:Stability` and its supporting development in §5 of
`article.tex`; `lemma:Continuity` as stated in the main text and its complete
proof in `continuity_appendix.tex` (assumptions `H:data`–`H:jump`,
`lem:parameters`, `lem:winv`, `lem:jump`, `lem:continuity` Steps 0–9,
`lem:continterp`, `prop:stability`, `thm:convergence`); `th:Convergence` in
the main text. Sources: July 2026 repository revision.

**Method and assurance levels.** Every claim was placed in one of three
tiers, and the tier is stated explicitly throughout:

1. **Machine-checked** — the purely algebraic content (parameter identities
   and inequalities, coefficient expansions and lower bounds, the jump
   lemma, the Cauchy–Schwarz aggregations) is proved in Coq 8.18 from the
   real-number axioms and re-verified by the trusted kernel (`coqchk`). See
   `StabilityAlgebra.v` and `ContinuityAlgebra.v` and the README crosswalk.
2. **Hand-verified** — the functional-analytic steps (testing, integration
   by parts, inverse estimates, facewise assembly, interpolation
   replacements) were re-derived line by line against the manuscript. These
   are outside the reach of stdlib Coq; the verification record is in §4
   below.
3. **Cited-standard** — the existence of the inverse-estimate and
   interpolation constants themselves (Brenner–Scott, Ern–Guermond) is taken
   on the cited literature, as is universal practice.

---

## 1. Verdicts

| Result | Verdict |
|---|---|
| §5 development (eq:StabilityEstimate → eq:StabilityEstimateFinal → coefficient bounds → `lemma:Stability`) | **Correct.** Every algebraic step machine-checked; every analytic step hand-verified. No error found. |
| `lemma:Continuity` (main text) + `app:Continuity` (full proof) | **Correct as written**, subject to two precision amendments to the hypotheses (F1, F2 below). All eighteen term bounds, the exact rewritings of Step 6, the jump treatment and the Step 9 absorption were verified; the algebraic core is machine-checked with explicit constants where the appendix writes a generic `C`. |
| `lem:parameters`, `lem:winv`, `lem:jump`, `lem:continterp` | **Correct** (`lem:winv` subject to F2; `lem:continterp` inherits F1 through identity (iii)). `lem:parameters` and `lem:jump` are machine-checked in full. |
| `th:Convergence` / `thm:convergence` | **Correct.** The triangle-inequality argument is standard and its inputs check out. |

The bottom line: **no mathematical error was found in any proof.** The
findings below are precision gaps in two hypotheses (the proofs use slightly
more than the hypotheses literally grant — both trivially satisfied in the
application), plus hygiene and presentation items, plus one proposed
strengthening remark.

---

## 2. Findings and proposed amendments

### F1 (moderate precision gap) — `H:advection` under-specifies the regularity of **a**

`H:advection` assumes only `a ∈ C(Ω̄)^d` together with the *global
distributional* identity `∇·(αa) = 0`. But identity (iii) of Step 6a
(eq:elemibp) applies the divergence theorem on each element and explicitly
invokes "`∇·(αa) = 0` **in K**", i.e. an a.e. pointwise statement on each
element. For a merely continuous `a`, `∇·(αa)` need not be a function on `K`
at all, so neither the elementwise identity nor the divergence-theorem step
is licensed by the hypothesis as written. The same identity is reused with
the interpolation error in `lem:continterp` (observation (1)).

The gap is vacuous in the application — `a` is the continuous finite element
velocity of the previous iteration, hence elementwise polynomial — but the
hypothesis should say so. Since `αa` is continuous across interior faces,
its distributional divergence has no face contributions, so the global
distributional identity does localise once elementwise weak differentiability
is granted.

**Proposed replacement** (continuity_appendix.tex, lines 170–175):

```latex
\begin{assumption}[Advection]\label{H:advection}
$\ba\in C(\overline{\Omega})^d$ with $\ba|_K\in W^{1,\infty}(K)^d$ for every
$K\in\Th$ (in the application, $\ba$ is the continuous finite element
velocity of the previous iteration, which satisfies this trivially), and
$\nabla\cdot(\alpha\ba)=0$ in the sense of distributions, i.e.\
$\int_\Omega \alpha\ba\cdot\nabla\psi \dd\Omega = 0$ for all
$\psi\in W^{1,1}_0(\Omega)$. Since $\alpha\ba$ is continuous across interior
faces, its distributional divergence carries no face contributions, and the
global identity localises: $\nabla\cdot(\alpha\ba)=0$ a.e.\ in each
$K\in\Th$, which is the form used in \eqref{eq:elemibp}.
\end{assumption}
```

No other change is needed; identities (i) and (ii) use only the global
statement and are fine as written.

### F2 (minor precision gap) — `H:porosity`'s derived bound needs element convexity

`H:porosity` derives `α_{∞,K} ≤ (1+C_α) α_{0,K}` from
`h_K‖∇α‖_{L∞(K)} ≤ C_α α_{0,K}` ("In particular…"). The derivation is the
mean-value inequality along the segment joining near-extremal points of
`α|_K`, which requires that segment to lie in `K` — i.e. convex elements (or
a chunkiness-dependent enlargement of `δ_α` for non-convex ones). Simplicial
and tensor-product elements are convex, so this is again vacuous in
practice, but the one-word gap should be closed.

**Proposed patch** (continuity_appendix.tex, line 164):

```latex
In particular, the elements being convex,
$\alpha_{\infty,K}\le \delta_\alpha\,\alpha_{0,K}$ with
$\delta_\alpha\coloneqq 1+C_\alpha$ (for shape-regular non-convex elements
the same holds with $\delta_\alpha$ enlarged by a constant depending only on
the chunkiness parameter), so that ...
```

The scalar chain itself (`α_∞ ≤ (1+C_α)α_0`, and the coefficient step
`α_∞/α_0^{1/2} ≤ δ^{1/2}α_∞^{1/2}` inside the proof of eq:winv-divvisc) is
machine-checked (`delta_alpha_bound`, `winv_ratio`).

### F3 (repository hygiene) — stale labels `eq:847` / `eq:855`

The SymPy suite (`stability_estimate_verification.py` and the scripts
README) cites the labels `eq:847` and `eq:855` for the viscous and velocity
coefficient expansions. Those labels do not exist in the current tex; the
corresponding displays (article.tex, the two equations after
eq:UpperBoundOnEpsilon, currently at lines 883–885 and 891–893) are
**unlabelled**. Since two verification suites now anchor to these displays,
they deserve stable labels.

**Proposed patch** (article.tex):

```latex
\begin{equation} \label{eq:ViscousCoefficientBound}
    \nu \tau_1 \left( \alpha_K \Big( 2 - 4 \frac{C_\text{inv}^2}{c_1} ...
```
```latex
\begin{equation} \label{eq:VelocityCoefficientBound}
    \alpha_K \tau_1 \sigma \left( \Bigl( 1 - \frac{\xi C_\text{inv}^2}{c_1} ...
```

and update the two script headers accordingly. The Coq development keeps the
historical definition names `visc_847` / `u_855` (they match the scripts);
the README records the mapping to the new labels.

### F4 (typographical) — broken-norm subscript in the main-text triple norm

The main-text definition of the working norm (article.tex, line 923) writes
`‖σ̃_α^{1/2} u_h‖²` without the broken-norm subscript `_h`, while σ̃_α is
elementwise constant; the appendix version (eq:triplenorm, line 133)
correctly carries `_h`. Numerically identical here (the elemental L² squares
sum to the global one), but the notation should match the appendix and the
declared convention for elementwise weights.

**Patch:** `\Big\| \widetilde{\sigma}_\alpha^{1/2} \boldsymbol{u}_h \Big\|_h^2`.

### F5 (hygiene) — `supplement.tex` is untouched template boilerplate

The supplement is still the SIAM template (lipsum text, "An example
theorem", example table). If no supplement is planned, drop the file from
the submission set; if one is planned, it currently contains nothing.

### F6 (proposed strengthening remark) — analysis τ₂ vs implemented τ₂

The analysis is carried out with eq:Tau2Final, i.e. dropping the `εh²` term
of the implemented eq:Tau2. This is declared, but the gap between the
analysed and implemented parameters can be closed in one sentence, because
under eq:UpperBoundOnEpsilon the two are equivalent up to the factor
`1 + C₂ < 2`:

```
τ₂ / τ₂^impl = (c₁ α_K τ_{1,NS} + ε h²) / (c₁ α_K τ_{1,NS})
             = 1 + ε τ₂  ≤  1 + C₂ ,
```

so `τ₂^impl ≤ τ₂ ≤ (1+C₂) τ₂^impl`, and every estimate of §5 and Appendix B
transfers to the implemented parameter with constants inflated by at most
`1 + C₂`. This chain is machine-checked (`tau2impl_le_tau2`,
`tau2_le_scaled_impl` in `ContinuityAlgebra.v`).

**Proposed remark** (after eq:Tau2Final, or as a remark in the appendix):

```latex
\begin{remark}
The analysis uses $\tau_2$ as in \cref{eq:Tau2Final}, i.e.\ without the
$\varepsilon h^2$ term of \cref{eq:Tau2}. Under
\cref{eq:UpperBoundOnEpsilon} the two definitions are equivalent up to the
factor $1+C_2<2$: writing $\tau_2^{\mathrm{impl}}$ for \cref{eq:Tau2},
$\tau_2/\tau_2^{\mathrm{impl}} = 1+\varepsilon\tau_2 \le 1+C_2$, so that
$\tau_2^{\mathrm{impl}}\le\tau_2\le(1+C_2)\,\tau_2^{\mathrm{impl}}$ and all
estimates hold for the implemented parameter with constants inflated by at
most $1+C_2$.
\end{remark}
```

### F7 (optional clarification, low priority) — the role of α_K = α_{∞,K}

The main text (line 819) presents `α_K = α_{∞,K}` as the natural choice
"yielding simplified estimates". More precisely: the specific inequalities
`‖αM‖²_K ≤ α_K‖α^{1/2}M‖²_K` (used in eq:BoundOfCrossedTerm) and the
appendix's `α ≤ α_K` steps hold *as written* only for `α_K = α_{∞,K}`; any
other admissible representative works after inserting factors of `δ_α`
(bounded under H:porosity). One clause to that effect would preclude a
reader trying `α_K = α_{0,K}` with the constants as printed. The appendix
already fixes `α_K := α_{∞,K}` by definition (line 110), so this concerns
the main text only.

---

## 3. What was *not* audited

The well-posedness discussion of §2 (existence/uniqueness for the continuous
problem); the numerical-examples sections; the elemental-matrices appendix
(deliberately left to the SymPy suite, as before); the constants inside the
cited interpolation/inverse-estimate literature. The Fourier design section
was checked only at the level of the definitions consumed by §5 and
Appendix B (eq:Tau1, eq:Tau2, eq:CAlpha, eq:TauNavierStokes — all consistent
with the earlier `TauDesign.v` formalisation).

---

## 4. Verification record (hand-audited steps)

For the record, the step-by-step confirmations, with the hypothesis each
step consumes. "MC" marks steps whose scalar core is machine-checked.

**§5 stability.**
Galerkin testing (eq. after line 826): convection vanishes via
`∇·(αa) = 0` + `u_h = 0` on Γ; pressure/mass pair vanishes via divergence
theorem on `αu_h p_h`; viscous term uses idempotence/symmetry of the
projector. ✔ — the difference-of-squares structure of eq:StabilityEstimate
(no cross terms between `αX` and `2∇·(ανΠ∇u)−σu`) follows from the adjoint
having momentum component `−αX(V) − G(v)`; consistent with
eq:AdjointDifferentialOperator and the appendix eq:strongop. ✔ —
eq:BoundOfCrossedTerm: expansion of the square ✔; inverse estimate applied
through the amended C̄_inv remark (line 863), now made precise by lem:winv
✔; `‖αΠ∇u‖² ≤ α_K‖α^{1/2}Π∇u‖²` needs `α_K = α_{∞,K}` (see F7) ✔; the
factor bookkeeping of lines 852–855 (the ν pulled out, the Young parameter
`ξC̄²/h²`, the exact cancellation of C̄² between the two Young branches) ✔
MC (Young identity/inequality). — eq:StabilityEstimateFinal collection ✔ MC
(coefficient expansions). — ε-chain: `ετ₂ ≤ C₂` ✔ MC; pressure coefficient
`≥ (1−C₂)ε` ✔ MC. — Viscous bracket `≥ Cν` with
`C = min{2−4C̄²/c₁, 2(1−2/ξ)}` ✔ MC (`viscous_coefficient_lower_bound`;
requires `C ≤ 2`, which holds). — Velocity bracket `≥ Cσ̃_α` with
`C = 1−ξC̄²/c₁`, and the exact slack `α_Kτ₁σ(ξC̄²/c₁)(c₂|a|/h)` ✔ MC. —
`c₁ > 2ξC̄²`, `ξ > 2` ⟹ all constants positive ✔ MC
(`stability_constants_positive`). — Assembly into
`B_S ≥ C_stab|||U_h|||²` ✔ MC (`elemental_coercivity`).

**Appendix B.**
`lem:parameters` (P1)–(P5) ✔ MC in full, including the square-root forms.
— `lem:winv`: the product-rule split, `‖∇·M‖ ≤ √d‖∇M‖`, the
`α_∞/α_0^{1/2} ≤ δ^{1/2}α_K^{1/2}` step (✔ MC), and the constant
`C̄ = √(dδ_α)C_inv + C_α` ✔ (subject to F2). — `lem:jump` ✔ MC in full with
explicit constants (`C_J/c_J` where the appendix writes `C`). —
`lem:continuity` Step 0: the 18-term expansion of eq:Bstab reproduced and
confirmed exact ✔. Step 1 ✔ (the factor-2 weighted Cauchy–Schwarz: MC,
`step1_bound`). Step 2 (`σ − σ²τ₁ = σ̃` elementwise) ✔ MC. Step 3
(`1+C₂ < 2`) ✔. Step 4: eq:keyvisc ✔ MC; T₆/T₇/T₉ ✔; the double inverse
estimate and the T₈ chain `σ(c₁να_K/h²)τ₁ ≤ σ̃` ✔ MC; T₁₂ ✔. Step 5: T₁₃
convective chain ✔ MC. Step 6a: identities (i)–(iii) ✔ subject to F1; the
five-term regrouping of `N` re-derived and confirmed, including the sign
remark on [III] vs Codina's Eq. (66) ✔. Step 6b: elementwise IBP and
facewise assembly ✔; `1−στ₁ = φ₁τ₁` ✔ MC; eq:volpart's
`φ₁τ₁ ≤ φ₁^{1/2}τ₁^{1/2}` and the P3 split ✔ MC; jump part: eq:jumpsplit ✔
MC (both the geometric-mean route and the `σ̃^{1/2}|_i τ₁^{1/2}|_j` form),
`h`-power bookkeeping `h^{d−1}·h^{−d/2}·h^{−d/2} = h^{−1}` ✔. Step 6c ✔ MC
(`φ₁^{1/2}τ₁ ≤ τ₁^{1/2}`). Step 6d: single-valuedness of the integrand
(uses continuity of `a`) ✔; the absorption
`τ₁^{1/2}|_i φ₁^{1/2}|_j ≤ C` ✔ MC, with the `h`-powers cancelling exactly
✔. Step 7 ✔ MC (`ετ₂^{1/2} ≤ C₂^{1/2}ε^{1/2}`). Step 8: all 18 terms
accounted for, none double-counted ✔. Step 9: the five absorption
coefficients ✔ MC (`absorb1`–`absorb4` cores; absorb5 is definitional), and
the squaring-and-summing ✔ MC (`norm5_absorption` with the explicit
aggregate constant). — `lem:continterp`: the replacement table's
coefficients match `lem:winv` line by line ✔ (each interpolation estimate
re-derived); the `ℓ² ⊂ ℓ¹` step in eq:Enorm ✔; the `H² ↪ C⁰` and
`k_ψ+1 > d/2` conditions for the L^∞ estimates ✔; identity reuse subject to
F1 ✔. — `thm:convergence`: consistency + coercivity + `lem:continterp` +
triangle inequality ✔.

**Cross-consistency.** The appendix's eq:Bstab was checked against the
main-text `B_S = B − Σ⟨L*V, τLU⟩`: momentum and mass rows match, and
specialising `V = U` reproduces eq:StabilityEstimate exactly (the
difference-of-squares check). The appendix definitions eq:taus/eq:phi1/
eq:sigmatilde/eq:epscond match eq:Tau1Final/eq:Tau2Final/eq:SigmaAlpha/
eq:UpperBoundOnEpsilon. ✔

---

## 5. Machine-checked coverage

`StabilityAlgebra.v` (31 statements) and `ContinuityAlgebra.v` (83
statements) carry the algebraic load described above. Beyond the algebra,
`AbstractStability.v` and `AbstractContinuity.v` (with the supporting
`AbstractSums.v` and `InnerSpace.v`) prove lemma:Stability and
lemma:Continuity as complete theorems from the named trusted base listed in
the README's scope ledger, so that the "MC" markers of §4 now extend to the
summation, Cauchy–Schwarz, Young, facewise and absorption layers of both
proofs. `AbstractInterpolation.v` and `AbstractConvergence.v` extend the
same treatment to lem:continterp and thm:convergence: the interpolation
estimates of eq:interp/eq:interpinfty enter as named hypotheses (HI_*),
the triple-norm triangle inequality is proved from the pre-Hilbert
axioms, and the convergence theorem is the literal kernel-level
composition of the two closed lemmas, glued by the consistency hypothesis
Horth and the tested identity HBS_W (the cross-module tau-formula
agreement is itself machine-checked). Only the Green-type identities, the
inverse-estimate and interpolation-estimate primitives, and the data
assumptions remain outside the kernel. The full crosswalk is in
the README; the Lean 4 plan for the residue is LEAN_ROADMAP.md. Build:
`./run_all.sh` (only `coqc` required; `coqchk` kernel re-verification
included when available).
