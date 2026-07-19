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

### F8 (defect — APPLIED) — `lem:winv`: a mislabelled estimate

In `continuity_appendix.tex`, `\label{eq:winv-conv}` sat on the **last** line
of the `lem:winv` display — the pressure-gradient estimate
`‖α∇r_h‖_K ≤ (C_inv/h_K) α_K ‖r_h‖_K` — while the **convective** line above it
(`‖α a·∇w_h‖_K ≤ (C_inv/h_K) α_K |a|_{∞,K} ‖w_h‖_K`) carried no label at all.
The proof of the lemma refers to "the estimates (eq:winv-conv)" in the plural,
so the two lines were evidently intended as one labelled group — but LaTeX
attaches the label to a single line. Consequently **two of the four references
to `eq:winv-conv` pointed at the wrong estimate**: Step 5 (line ~536, "the first
contribution", which is the *convective* term `σ(τ₁u, α a·∇v)`) and Step 9
(line ~798, "the velocity part", also convective) both resolved to the
pressure-gradient line.

Fixed by splitting the label:
  * the **convective** line now carries `\label{eq:winv-conv}` (matching the two
    references that mean it);
  * the **pressure-gradient** line carries the new `\label{eq:winv-gradp}`;
  * the proof sentence now cites both;
  * Step 9's "the pressure part uses (eq:winv-conv) directly" was re-pointed to
    `eq:winv-gradp`.

Surfaced by the Coq audit, which has to cite the two estimates *separately*
(`Hw_cxu`/`Hw_cxv` encode the convective one, `Hw_gpu` the pressure-gradient
one) and so could not use a single label for both. The paper rebuilds clean
(0 errors, 0 undefined references). No mathematical content changes.

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

**§5 stability — the two diagonal Green identities.** These now carry the
tested identity that `HBS` used to assume, so they are audited here in the
detail `HBS` previously received. Both are the `v = u` specialisations of
identities already stated in the appendix's Step 6a, and each is one line:

* `H_skew_diag` = eq:skew at `v = u`. Step 6a(i) proves
  `(v, αa·∇u) = −(u, αa·∇v)` from the pointwise product rule
  `(αa·∇u)·v + (αa·∇v)·u = αa·∇(u·v)`, the membership `u·v ∈ W^{1,1}_0(Ω)`
  and the *global distributional* form of `∇·(αa) = 0` (H:advection). Setting
  `v = u` gives `(u, αa·∇u) = −(u, αa·∇u)`, hence `= 0`. ✔ The Coq
  `H_skew_diag` sums the elemental pairings `⟨u_h, αa·∇u_h⟩_K`; the sum of the
  elemental integrals is the global one, so the global identity is the right
  one to cite. ✔
* `H_ibp_diag` = eq:globalibp (first identity) at `v = u`, i.e.
  `(u, α∇p) = −(p, ∇·(αu))`, from `α ∈ W^{1,∞}(Ω)` and `u ∈ H^1_0(Ω)^d`
  (whence `αu ∈ H^1_0(Ω)^d`, so the divergence theorem carries no boundary
  term). ✔ Same elemental-sum remark. ✔

Worth recording: **neither diagonal identity is exposed to F1.** F1 concerns
the *elementwise* identity (iii) (eq:elemibp), which needs `∇·(αa) = 0`
pointwise in `K`; identities (i) and (ii) use only the global distributional
statement and are, as F1 itself notes, fine as written. So the stability
lemma's analytic base is F1-free; F1 continues to bear on `lem:continuity`
Step 6a(iii) and on `lem:continterp` only.

Given these two, the tested identity itself (eq:StabilityEstimate) is no
longer hand-audited at all — it is `Theorem HBS` in `AbstractStability.v`,
derived from `H_skew_diag` + `H_ibp_diag` and inner-product algebra, using no
positivity. The remaining hand-audit for it is the reading obligation on the
eighteen-term `B_S`, discharged under "Appendix B, Step 0" below.

**§5 stability — the estimate.**
Viscous term uses idempotence/symmetry of the
projector ✔ — the difference-of-squares structure of eq:StabilityEstimate
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
confirmed exact ✔ — **load-bearing for both lemmas; see the note below.**
Step 1 ✔ (the factor-2 weighted Cauchy–Schwarz: MC,
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

**Appendix B, Step 0 — the reading obligation, in detail.** This audit is
now the *only* hand-verified bridge between the manuscript's `B_S` and the
number all four Coq theorems talk about, for the stability lemma as much as
for continuity, so it is set out explicitly rather than left as a tick.

The obligation is a Stratum-1 *identification*, not a hypothesis: nothing
inside Coq can be wrong about it, because `BS` is a `Definition`, not a
`Variable`. What a reader must check by eye is that the eighteen summands
`T₁…T₁₈` of `AbstractInterpolation.v` transcribe eq:T1–eq:T18 of the
appendix's Step 0, term for term — and, through them, that Step 0's
decomposition of eq:Bstab is exact (every term appearing once, none twice,
none omitted). Both halves were re-checked line by line at this revision and
all eighteen agree. Three points where the transcription is *not* a literal
symbol match, and so were checked against the atom glossary rather than read
off:

* `du ~ 2ν∇·(αΠ∇u)` — eq:T6's `−4ν²(τ₁·,·)` becomes `−(τ₁ du, dv)`, the
  `4ν²` living inside the two atoms. Corroborated independently by `S3`,
  which is eq:winv-divvisc with exactly that `2ν` in place. ✔
* `gu ~ α^{1/2}Π∇u` — eq:T1's weight `α` is split across the two slots. ✔
* `xu := cxu +ᵥ gpu ~ αX(U)` (and `xv` likewise) — this is where the `L*V`
  adjoint sign convention sits: the `+αa·∇v` convective slot of
  eq:AdjointDifferentialOperator / eq:strongop. It is what makes `T₂`, `T₇`,
  `T₉`, `T₁₀`, `T₁₁`, `T₁₃` read correctly, and it is fixed once, in the
  glossary, for all of them. This is the [known-fragility] sign whose flip is
  the "anti-SUPG" failure. ✔

The obligation did not grow when the stability file joined it: the same
eighteen-term `Definition` already stood behind the interpolation and
convergence files (it was already what `BSWW` unfolds to).
`AbstractStability.v` now *reuses* it verbatim — `AbstractInterpolation.BS`
applied at the diagonal atoms — rather than restating it, so one reading
serves all four theorems and the two encodings of `B_S` are reconciled inside
the kernel (`BSWW_is_ASBS`) rather than by a side condition.

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
Horth alone (item 32). The tested identity for the W-pair, formerly the
hypothesis HBS_W, is gone from the file entirely: abstract_stability is
applied at those atoms and carries it, and the two encodings of B_S agree
definitionally (`BSWW_is_ASBS`), as does the cross-module τ-formula
(`tau1_agree`/`tau2_agree`). Only the Green-type identities, the
inverse-estimate and interpolation-estimate primitives, and the data
assumptions remain outside the kernel. The full crosswalk is in
the README; the Lean 4 plan for the residue is LEAN_ROADMAP.md. Build:
`./run_all.sh` (only `coqc` required; `coqchk` kernel re-verification
included when available).

The trusted base currently stands at **50 named hypotheses** (24 data-type, 8
Green/consistency, 18 interpolation-type; the analytic residue is items 25–50),
enumerated row by row in Table `tab:inventory` of `coq_coverage.tex` — that
table, not this file, is the authoritative list. Two changes since the previous
revision (53 items) change what §4 above must hand-audit, and both are **trades,
not free reductions**:

- **The tested identity is no longer assumed** (item 27 of the old numbering).
  `HBS` is now a theorem of `AbstractStability.v`, proved from the two
  *diagonal* Green identities `H_skew_diag` and `H_ibp_diag` (items 28–29 in the
  new numbering — renumbered, not newly added; they were already in the base),
  which are what it silently bundled. Two honest qualifications. (a) The
  previous description of `HBS` as the development's single largest assumption
  was **wrong, and is withdrawn**: `HBS` had the shape `Variable BS : R` plus
  `Hypothesis HBS : BS = <closed expression in the other free atoms>`, which
  constrains no model — it was a definition written as a hypothesis, and
  eliminable as such. Deleting it removes a row from the count and a reading
  obligation, but no logical content. (b) `AbstractStability.v`'s *own* analytic
  base therefore went **up**, from one real fact (`S3`) to three (`S3` plus the
  two Green identities): it traded one vacuous hypothesis for two substantive
  ones. The union is what falls, because `AbstractConvergence.v` already
  carried both identities. What genuinely improves: the bespoke five-term
  tested form no longer has to be read against eq:StabilityEstimate by eye,
  and `B_S` now has one encoding instead of two, reconciled in the kernel.
  Its convergence twin `HBS_W` is gone outright — `AbstractConvergence.v`
  applies `abstract_stability` at the discrete-error atoms instead, so one
  proof and one pair of identities serve both diagonals. Net for §4: two
  one-line identities to audit (done above, and F1-free), against one bespoke
  identity and one reading obligation retired. The two new identities are
  demonstrably satisfiable rather than covertly strong: `NonVacuity.v`
  discharges both (`w_skew_diag`, `w_ibp_diag`) in its witness model and still
  exhibits `abstract_stability` with a non-trivial conclusion.
- **`IU_nonneg`/`IP_nonneg` are derived** (old items 19, 20), from `HI_uu`/
  `HI_pp` and the pre-Hilbert `nrm_nonneg`, at the price of strengthening
  `C_I ≥ 0` to `C_I > 0` (item 15, now `CI_pos`). Strictly a trade: `{C_I > 0}`
  is strictly stronger than `{C_I ≥ 0, I_U ≥ 0, I_P ≥ 0}` — the old set admits
  the degenerate model `C_I = 0`, which does not imply `C_I > 0`. Two reasons
  the strengthening costs nothing in substance: the discarded reading is
  vacuous for the manuscript (`I_U = h_K^{k+1}|u|_{H^{k+1}(K)} ≥ 0` by
  definition, and `C_I > 0` for any real interpolation estimate — a
  `C_I = 0` estimate would assert exact interpolation of every function), and
  `C_I` is monotone in the development: it occurs only in the upper bounds
  `‖·‖ ≤ C_I·I` and in the output constant `K_{PP,I} = √c₁·C_I + K_{6b}`, so
  any model of the old base survives at `C_I + 1` with a weaker final constant.
  Nothing in §4's estimates depends on the `C_I = 0` reading. This strengthening
  is now **witnessed**: `NonVacuityInterp.v` and `NonVacuityConv.v` (added
  2026-07-17) discharge `CI_pos : 0 < C_I` strictly (`C_I = 1`, `C_I = 4`) and
  jointly with the full interpolation/convergence bundles, so `0 < C_I` is
  machine-checked to be consistent with the rest, not merely argued. (It sits
  outside `NonVacuity.v`'s own witness, which instantiates `abstract_stability`
  and does not mention `C_I`; before those two files, item 15 was unwitnessed.)

One further change since the previous revision is **structural, not a change to
the trusted base at all**:

- **The nine weighted inverse estimates now share one schema.** The Class-I
  `winv` family — `Hw_gu`, `Hw_gv`, `Hw_du`, `Hw_dv`, `Hw_cxu`, `Hw_cxv`,
  `Hw_gpu`, `Hw_divu` (items 34–41) and the stability lemma's `S3` (item 33,
  `≡ Hw_du`) — is now written through the single predicate
  `winv_est C W A B := forall k, ‖A k‖ ≤ C / h_K k · W k · ‖B k‖` of the new
  file `InverseEstimates.v`, and the "double" composites (`du` bounded directly
  by `uu`, weight `√α_K·√α_K = α_K`, `h^{-1}·h^{-1} = h^{-2}`) are derived
  generically by the proved lemma `winv_compose`, retiring the byte-for-byte
  `double_inv_u`/`double_inv_v` block that stood **three times** (twice in
  `AbstractContinuity.v`, once in `AbstractInterpolation.v`). This is
  **notational, not a reduction of trust**: each hypothesis unfolds
  *definitionally* to the estimate it always stated (six exactly; the three
  `α_K|a|`/`α_K`-weight ones up to a re-association every `pose proof … nra`
  consumer absorbs), so the nine remain **nine independent named hypotheses**
  and the count stays **50** — `winv_est` names their common shape, it replaces
  none of them and adds none. It is emphatically *not*, and must not become, a
  single estimate `forall x, ‖D x‖ ≤ C/h · W · ‖x‖` quantified over an arbitrary
  vector: the discrete atoms and the interpolation-error atoms inhabit the *same*
  carrier, so a `forall`-`x` inverse estimate would license one for a
  non-polynomial interpolation error, which is false — the polynomial /
  interpolation-error firewall (§4, "true for polynomials, never for both") is
  exactly what keeps the nine from collapsing to one. `winv_compose` is a proved
  lemma, not a hypothesis (`Print Assumptions`: only the three stdlib axioms), so
  §4's "double inverse estimate" tick (Appendix B, Step 4 above) is now
  discharged through it rather than by hand, with no addition to the base.
