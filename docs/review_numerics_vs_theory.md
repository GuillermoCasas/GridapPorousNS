# Theory–numerics consistency review of `article.tex` — v2, after full theory audit

> **ARCHIVED INPUT (do not treat items below as open).** This is the external review that `paper-revision-plan.md`
> responds to point-by-point; current resolution status lives in `paper-revision-plan.md` and
> `pre-submission-checklist.md`. Notably now settled: **C7** (the 1.29 triple — genuine mesh-independent
> saturation, verified 2026-07-19), and the v2 claim that the data *"empirically discriminate in favour of the
> weighted form"* — **refuted** as logically unsound (one-sided upper bounds; see `pre-submission-checklist.md §0.2a`).

*A stabilized finite element method for incompressible, inertial flows in inhomogeneous porous media* (Casas, González-Usúa, Codina, de-Pouplana)

**Scope of v2.** This revision supersedes the first review. In addition to the main text, I have now audited the full theory bundle: the paper's three appendices (`continuity_appendix.tex`, `fourier_appendix.tex`, `elemental_matrices_appendix.tex`), and the companion notes (`osgs_reaction_note`, `c1_dimension_note` + its verification scripts' claims, `tau_saturation_note`, `scale_free_gate_note`, `centered_encoding`, `velocity_floor_regularization`, `pressure_recentering_note`, `osgs_algorithm`, and the Cocquet notes). The bundled `paper/article.tex` is byte-identical to the reviewed one, so all table references stand. The bundle **resolves every conditional** of the first review, in most cases decisively; the changes are flagged inline as **[v2]**.

**Method.** Close reading; independent recomputation of the nodal-interpolation errors of the manufactured solution on the finest 2D mesh (ℙ₁ triangles / ℚ₂ quads, degree-4/7 quadrature); step-by-step verification of the continuity-appendix proof; consistency checks of the Fourier appendix symbols; and quantitative verification of the appendix's porosity-resolution assumption for the actual manufactured field.

---

## 0. Theory-audit findings **[v2, new]**

**A. The porosity-weighted convergence theorem is proven, rigorously.** Appendix C states and proves the sharp continuity bound
\[
\BS(\ba;U_h,V_h) \le C\Bigl(\|\alpha_K\tau_2^{1/2}h^{-1}\boldsymbol u_h\|_h + \|\alpha_K\tau_1^{1/2}h^{-1}p_h\|_h\Bigr)\triplenorm{V_h}
\]
(its eq. `sharpcont`), extends it to the interpolation error (Lemma `continterp`), and concludes the porosity-weighted convergence theorem \(\triplenorm{U-U_h}\le C\sum_K (\alpha_K/h_K)(\tau_{2,K}^{1/2}\mathrm E_{\text{int},K}(\boldsymbol u)+\tau_{1,K}^{1/2}\mathrm E_{\text{int},K}(p))\) — exactly the amended main-text statement. I verified the load-bearing steps: the weighted inverse estimates (Lemma `winv`, with the correct enlarged constant \(\overline C_{\text{inv}}=\sqrt{d\delta_\alpha}\,C_{\text{inv}}+C_\alpha\)); the placement of the \(\alpha_K\) weights in Steps 6b (via \(\varphi_1^{1/2}=c_1^{1/2}\alpha_K\tau_2^{1/2}/h\)) and 6c; the jump lemma and its facewise bookkeeping; the Step-9 absorption; and the interpolation-error rerun, whose elemental coefficients match the discrete ones term by term. The convention \(\alpha_K=\alpha_{\infty,K}\) (supremum) is fixed unambiguously there — confirming that the main text's "elemental **minimum**" sentence (D1) is a leftover error.

**B. The assumptions are explicit and the manufactured problem satisfies them.** Appendix C's Assumption *Porosity* requires \(h_K\|\nabla\alpha\|_{L^\infty(K)}\le C_\alpha\,\alpha_{0,K}\) (resolved porosity), so \(\alpha_{\infty,K}\le(1+C_\alpha)\alpha_{0,K}\) elementwise. For the paper's plateau bump I compute \(\sup(|\alpha'|/\alpha)\approx 7.3\) at \(\alpha_0=0.5\) and \(\approx 40.5\) at \(\alpha_0=0.05\) (attained near \(r\approx0.265\), where \(\alpha\approx0.11\)); hence \(C_\alpha\approx 40h\) in the worst case — ≈ 0.13 on the finest mesh, ≈ 4 on the coarsest. The assumption is comfortably satisfied asymptotically, and \(\delta_\alpha\to1\), which is precisely what makes the elementwise weight \(\sqrt{\alpha_K/\alpha_0}\) of my decomposition below well-defined. (On the coarse end of the α₀ = 0.05 sequence the constants are honest but sizeable; not worth a caveat in the paper, but worth knowing.)

**C. The Fourier appendix is consistent.** I re-derived the deviatoric–symmetric viscous symbol independently: \(\widehat{\mathcal L}_\nu=\frac{\alpha\nu}{h^2}(|\boldsymbol k_0|^2\mathbb I+(1-\tfrac2d)\boldsymbol k_0\otimes\boldsymbol k_0)\) with longitudinal eigenvalue \((2-\tfrac2d)\alpha\nu|\boldsymbol k_0|^2/h^2\) — matches, including the "equals the transverse one for d = 2" statement and the knowingly dropped 4/3. The λ-choice recovers \(\tau_2 = h^2/(c_1\alpha\tau_{1,\text{NS}})\) exactly, and the identifications \(c_1=|\boldsymbol k_0|^2\), \(c_2=|\boldsymbol k_0\cos\phi|\) close the loop to the main text.

**D. Companion-note findings that change earlier conclusions** (detailed under the relevant items): the `osgs_reaction_note` proves the provable reactive coercivity ordering is OSGS ≥ ASGS (C4); the `c1_dimension_note` grounds the Kuhn-threshold remark (C6); the `tau_saturation_note` documents the mechanism behind the excluded 2D corner (C8); the `osgs_algorithm` note shows the implementation projects the *full* residual for non-constant σ, flipping D5 onto the paper's sentence.

---

## 1. What the theory of the paper actually guarantees

**(T1) Stability (Lemma 1).** \(B_\text{S}(\boldsymbol a,U_h,U_h) \ge C\,\triplenorm{U_h}^2\) in the working norm (ν-weighted projected gradient, \(\widetilde\sigma_\alpha\)-weighted velocity, ε-weighted pressure, τ₁-weighted \(\alpha X\), τ₂-weighted porous divergence), under: linearized problem, prescribed \(\boldsymbol a\) with \(\nabla\cdot(\alpha\boldsymbol a)=0\); all-Dirichlet boundary; \(c_1 > 2\xi \overline C_{\text{inv}}^2\), \(\xi>2\) **[v2: with the α-enlarged inverse constant per Appendix C, Remark `winvconst` — see D10]**; ε small per (UpperBoundOnEpsilon); ASGS only.

**(T2) Convergence (Theorem 1), porosity-weighted form — now unconditionally the operative statement [v2]:**
\[
\triplenorm{E_h} \le C \sum_K \frac{\alpha_K}{h_K}\Big(\tau_{2,K}^{1/2}\,\mathrm{E}_{\text{int},K}(\boldsymbol u) + \tau_{1,K}^{1/2}\,\mathrm{E}_{\text{int},K}(p)\Big).
\]

Structural facts governing the comparison with the tables:

1. **The theory covers the linearized ASGS method only** (the experiments: nonlinear, ASGS *and* OSGS, non-homogeneous Dirichlet). The manufactured field satisfies \(\nabla\cdot(\alpha\boldsymbol u)=0\) exactly (verified), matching the advection assumption at the exact level.
2. **At ε = 0 the working norm contains no L² pressure control**, and no L² velocity control when σ ≈ 0. All "theoretical rates" in L² columns are duality heuristics on top of the working-norm estimate. The parentheticals (velocity \(k_v{+}1/k_v\), pressure \(k_p/k_p{-}1\) in L²/H¹) are the *viscous-regime worst case*; the paper's own asymptotics predict one order more for the pressure in the reactive/convective regimes (the Da\(^{-1/2}\) mitigation is h-independent).
3. **Regime-dependent guarantees** (equal order k): viscous — velocity H¹ rate k, pressure H¹ rate k−1 (L² heuristic k), no L² velocity control; convective — X-combination and porous divergence optimal, pressure H¹ rate k if ∇e_p dominates X; reactive — velocity L² rate k+1 (via \(\widetilde\sigma_\alpha\)), pressure H¹ rate k. Observations must sit at or above this table — and they do, everywhere, on the structured 2D meshes.
4. **The α₀ bookkeeping — the internal inconsistency, now unambiguous [v2].** Section 6's \(1/\alpha_0\) prefactors are what the *unweighted* theorem yields. Redoing the same normalizations with the proven weighted theorem, the weight multiplying \(\mathrm E_{\text{int},K}(\boldsymbol u)\) is \(\sqrt{\alpha_K/\alpha_0}\): **1** on plateau elements, at most \(\alpha_0^{-1/2}\) on full-porosity ones — never \(1/\alpha_0\). Since Appendix C proves the weighted form (its closing remark even calls it "the sharpest afforded by the present argument"), §6 is now derived from a strictly weaker statement than the paper's own theorem, and the results discussion inherits the mismatch. The fix direction is settled: update §6 (or add the dual-form remark of §7.1 below); do *not* weaken the theorem.

---

## 2. A priori expectations

E1. Optimal velocity H¹ rates in all regimes; optimal L² velocity guaranteed only in the reactive regime, expected everywhere by duality.

E2. Pressure rates regime-dependent: viscous worst case (H¹ k−1, L² k); one order better under dominant reaction or convection.

E3. **The α₀-sweep is not a clean test of the stability constants.** Any manufactured velocity with \(\nabla\cdot(\alpha\boldsymbol u)=0\) and fixed flux scales as \(1/\alpha\); with \(\boldsymbol u = U(\alpha_0/\alpha)\boldsymbol\varphi\), \(|\boldsymbol u|_{H^{k+1}}\) grows as α₀ decreases. Observed FME growth must be decomposed into (solution roughness) × (method factor). The pressure — whose exact field and normalization are α₀-independent — is the clean test.

E4. The bounds are one-sided: "consistency" means observed ≥ theoretical rate and observed ≤ bound; entries far better than the bound indicate non-sharpness that the discussion should name.

E5. At Re = 10⁶ the finest mesh has \(\Reyn_h = 3125\) — deeply pre-asymptotic. At Da = 10⁶, \(\Damk_h \approx 9.8\) with the global α_∞ = 1, but ≈ 195 in the α₀ = 0.05 plateau under the elementwise convention the local asymptotics use.

---

## 3. Independent verification: the approximation floor

Nodal-interpolation errors of the exact fields on the 320×320 mesh (per unit U and unit pressure shape; my computation; normalization convention cross-checked against the `centered_encoding` note's \(U_c,\ P_c\sqrt{|\Omega|}\), \(|\Omega|=1\) **[v2]**):

| quantity                | interp. error      | matching FME entries                                                   | efficiency FME/interp |
| ----------------------- | ------------------ | ---------------------------------------------------------------------- | --------------------- |
| ℙ₁, u, H¹, α₀=0.5  | **3.499e-2** | 3.50e-2 (all rows with Re ≤ 1)                                        | **1.00**        |
| ℙ₁, u, H¹, α₀=0.05 | **1.751e-1** | 1.76e-1                                                                | **1.01**        |
| ℙ₁, u, L², α₀=0.5  | 3.259e-5           | ASGS 6.95e-5 / OSGS 3.62e-5                                            | 2.13 / 1.11           |
| ℙ₁, u, L², α₀=0.05 | 1.637e-4           | ASGS 6.84e-4 / OSGS 6.33e-4                                            | 4.18 / 3.87           |
| ℚ₂, u, L², α₀=0.5  | **2.065e-7** | 2.06e-7–2.08e-7                                                       | **1.00**        |
| ℚ₂, u, L², α₀=0.05 | **2.385e-6** | 2.39e-6–2.42e-6                                                       | **1.00**        |
| ℚ₂, u, H¹, α₀=0.5  | **4.282e-4** | 4.29e-4                                                                | **1.00**        |
| ℚ₂, u, H¹, α₀=0.05 | **4.947e-3** | 4.96e-3–4.97e-3                                                       | **1.00**        |
| ℙ₁, p, H¹            | **1.090e-2** | 1.09e-2 (high-Re and high-Da rows; also the TH rows of the DBF tables) | **1.00**        |
| ℙ₁, p, L²            | 9.837e-6           | 8.76e-6 (high-Da), 4.87e-6 (high-Re)                                   | 0.89 / 0.50           |
| ℚ₂, p, L²            | **3.848e-9** | 3.84e-9–3.94e-9 (high-Da)                                             | **1.00**        |
| ℚ₂, p, H¹            | **7.979e-6** | 7.98e-6–8.24e-6 (high-Da)                                             | **1.00**        |

(Entries slightly below the nodal floor are unproblematic: the discrete solution can beat nodal interpolation, which is not the best approximation.)

**F1. Wherever the tables show clean optimal rates, the error coincides with the best-approximation error to three significant digits** — all ℚ₂ velocity entries, all ℙ₁ H¹ velocity entries, and the pressure in the high-Re/high-Da regimes. The stabilization introduces *no measurable accuracy loss* there, a far stronger statement than "optimal rates", available for free by adding a reference row of interpolation errors (twelve numbers above).

**F2. The α₀-sweep decomposition.** Pure interpolation grows by ×5.0 (ℙ₁, both norms) and ×11.6 (ℚ₂, both norms) as α₀ goes 0.5 → 0.05; the observed velocity ratios are ×5.03 and ×11.6. Method-only degradation factors:

| quantity                           | method factor (0.5→0.05) | compare                          |
| ---------------------------------- | ------------------------- | -------------------------------- |
| ℙ₁ u H¹, ℚ₂ u L², ℚ₂ u H¹ | **1.00**            | α₀⁻¹ᐟ² = 3.16, 1/α₀ = 10 |
| ℙ₁ u L² ASGS / OSGS             | 1.96 / 3.48               | ≤ α₀⁻¹ᐟ² ≈ 3.16          |

The velocity data are **inconsistent with a sharp 1/α₀ degradation and consistent with the proven weighted theorem** (weights in [1, 3.16]). This is now the empirical companion of a proven result, not of a conjecture **[v2]**.

**F3. The pressure as the clean α₀-test.** Viscous-regime pressure ratios: ×8.3–9.5 (ℙ₁), ×10.2 (ℚ₂ ASGS), ×16.2–16.5 (ℚ₂ OSGS). The dominant channel is \(\tau_2^{1/2}\mathrm E_{\text{int}}(\boldsymbol u)\) (larger than the pressure term by L/h), whose growth is the interpolation growth of u times a weight in \([1,\alpha_0^{-1/2}]\): predicted windows [5.0, 15.9] (ℙ₁) and [11.6, 36.6] (ℚ₂) under the weighted theorem — both observed ratios inside or at the edge. Under the unweighted form the windows would start at ×50 and ×116 — clearly refuted as sharp scalings.

**F4. Passed internal audits.** Identical rows across (Re, Da) ∈ {10⁻⁶, 1}² are genuine Stokes-limit saturation (pressure FMEs differ by exactly the normalization ratio P = 2ν vs 3ν: 2.00e-3 × 2/3 = 1.33e-3, in both variants); the ℙ₁ pressure H¹ floor 1.09e-2 recurs identically for the equal-order and the Taylor–Hood ℙ₁ pressures in the DBF tables (same space, mesh, exact field); the error normalization implemented (`calculate_normalized_errors`, per the `centered_encoding` note) matches the convention my floors assume — corroborated by the twelve 3-digit matches.

---

## 4. Conflicts: expectation vs observation

### C1. The α₀-sweep discussion misattributes the observed degradation

**Expectation** (from the discussion as written): ~×10 loss for all quantities, attributed to the 1/α₀ prefactors of §6. **Observation:** velocity loss = best-approximation growth exactly (method factor 1.00 in ℙ₁-H¹ and both ℚ₂ norms; 1.96–3.48 in ℙ₁-L²); pressure loses ×8–16 in the viscous regime and nothing at high Da.

**Assessment.** The bounds hold, but "consistent with the asymptotic analysis" is achieved for the wrong reason: the ×10 the paper points at is, for the velocity, a property of the manufactured solution, and the residual method factor matches the *weighted* theorem's α₀⁻¹ᐟ², not 1/α₀. §6 is internally inconsistent with the paper's own (proven **[v2]**) theorem.

**Verdict.** Rework §6 to the weighted prefactors (or add the dual-form remark of §7.1 below, now citable to Appendix C's Remark `sharper`); rewrite the results paragraph with the decomposition; optionally add the interpolation reference row. Note the same correction propagates to the companion `tau_saturation_note`, whose Layer-2 paragraph quotes the "ubiquitous 1/α₀" claim **[v2]**.

### C2. High-Da pressure α₀-independence exceeds the estimate

Observed: identical H¹ pressure FME (1.09e-2) at both porosities, ≤ ×1.85 in L², versus the estimate's residual 1/α₀ prefactor (×10). **Explanation:** approximation floor (F1), α₀-independent because the exact pressure is. **Verdict:** not an unexplained "exception" — a positive floor result; state it with the floor value.

### C3. Pressure super-convergence and the "nominal optimal rate" terminology

ℙ₁: both variants interpolation-optimal in L² (≈2, one above the tabulated theoretical 1). ℚ₂: ASGS sits *on* the working-norm guarantee (L² ≈ 1.9, H¹ ≈ 0.8–0.9) — the only place the theoretical one-order pressure loss is actually observed, i.e. evidence of sharpness for the analysed variant; OSGS is interpolation-optimal, one orτ1sder higher. **Explanation for the order-dependence:** for ℙ₁ the adjoint viscous term vanishes elementwise and ASGS ≈ OSGS; for k ≥ 2 the orthogonal projection removes the FE-resolvable consistency pollution (cf. codina2008analysis). **Verdict:** disambiguate "optimal"; define the parenthetical convention at the first table; make the ASGS/ℚ₂ sharpness observation explicit — it is a point in the analysis's favour.

### C4. Reaction-dominated velocity: no Da¹ᐟ² effect, and an unreported ASGS/OSGS gap — **[v2: substantially revised]**

**Observation.** ASGS ℙ₁-H¹ velocity unchanged to three digits at Da = 10⁶ (still the floor); ℚ₂ grows ×3.5 only — the Da¹ᐟ² allowance of (DominantReactionVelocityGradientEstimate) never materializes. But **OSGS ℙ₁ velocity at (Da = 10⁶, α₀ = 0.5) is 1.23e-1 in H¹ (3.5× ASGS/floor) and 1.77e-4 in L² (1.65× ASGS)** — the largest velocity discrepancy between the variants in the whole campaign; it shrinks at α₀ = 0.05 (×1.2) and vanishes at Re = 10⁶. Unmentioned, and in tension with "velocity errors practically indistinguishable".

**Assessment, refined by the `osgs_reaction_note`.** The note proves (Proposition "Reactive coercivity gap") that in this very regime the *provable* reactive control ordering is **OSGS ≥ ASGS**: the constant-σ annihilation \((\mathcal I-\Pi)[\sigma\boldsymbol u_h]=0\) spares OSGS the reactive drain \(-\|\tau_1^{1/2}\sigma\boldsymbol u_h\|_h^2\) that reduces ASGS's coefficient from σ to \(\widetilde\sigma_\alpha\), with ratio \(\sim \Damk_h/(1+\Reyn_h)\). The *observed accuracy ordering at this corner is the reverse.* Both facts coexist (coercivity bounds are one-sided and do not rank error constants), but this rules out my v1 hypothesis (weaker OSGS reactive coupling) and shows the gap is an error-constant/consistency effect that neither the paper's analysis nor the note resolves. It also means the note's proposed drop-in paragraph — "By consistency the convergence rate is identical for both methods—as our experiments confirm" — is true for rates but should not be allowed to imply equal constants: the tables contradict that at this corner by ×3.5.

**Explanation for the absent Da¹ᐟ² effect** (unchanged): the term bounds contamination of the velocity by the worst-case pressure error; the computed pressure sits at its floor, so the mechanism never activates.

**Verdict.** Qualify "practically indistinguishable"; report the OSGS gap as an empirical observation (optimal rate preserved, slope 1.05), explicitly noting that the reactive-coercivity comparison, if anything, favours OSGS — so the gap is a constant effect the analysis does not capture; temper the corresponding sentence if the note's paragraph is merged.

### C5. Convection-dominated ℚ₂ sub-optimality — endorsed and quantifiable

\(\Reyn_h = 3125\) on the finest mesh: the pre-asymptotic attribution is sound; quote the number next to the existing Da_h ≈ 9.8 remark, and fix the unclear clause "converges on the finest mesh here as well".

### C6. 3D irregular-mesh sub-optimality attributed to an "element-quality tail" — **[v2: refined]**

**Assessment.** Red refinement of tetrahedra is shape-stable (bounded similarity classes), so quality does not degrade under refinement; a persistent rate deficit (ℙ₂ L² 2.55 vs 3; pressure below even the tabulated rates) on three or four coarse meshes is more plausibly pre-asymptotic, possibly compounded by the marginal c₁ and the ε-penalty/incompatible-data consistency perturbations. The `c1_dimension_note` grounds the threshold story properly: the elementwise threshold \(c_1^*=2\hat c^2\) is computed exactly per shape (tet/tri ratio ≈ 4.46 for the structured pairing, in [2.8, 4.5] across shapes), ℙ₁ is immune (\(c_1^*=0\)), and the note is explicit that the practical global threshold sits *below* the elementwise one only *empirically* — so the main-text remark is a fair summary of a documented, honestly-labelled empirical claim, and my v1 wording ("plausible but unproven") should be read in that softer light. What remains under-supported is the *quality-tail* causal attribution specifically.

**Verdict.** Hedge the quality-tail sentence toward pre-asymptotics (or substantiate with a quality histogram / error localization); temper "approach the theoretical optima" for ASGS ℙ₁-L² = 1.75. See also D10 for the \(\overline C_{\text{inv}}\) caveat that thins the c₁ margin further.

### C7. Suspicious constant in Table 3D-H¹: OSGS pressure FME = 1.29 three times

Unchanged: three different mesh/order combinations, different nonzero slopes, identical 1.29 — verify against raw data; if genuine, one sentence on the saturating feature.

### C8. The excluded corner (Re, α₀) = (10⁶, 0.05) — **[v2: mechanism identified]**

The `tau_saturation_note` documents exactly the two-layer mechanism that makes this corner fail: (Layer 1) a generic τ₁-saturation algebraic-residual floor under the tight, scale-free *absolute* stopping gate, "empirically dormant until Re ≳ 10⁶"; (Layer 2) low-α₀ aggravation of the nonlinear solve. Crucially, the note stresses this is a *stopping-criterion artifact, not an under-resolved solution* (the convergence theorem still applies). **Verdict:** add one sentence to §7.1 stating the reason in these terms — it converts a silent exclusion that invites the referee's worst reading into a documented, theory-consistent limitation. Proposed wording in §7.3-edits below.

### C9. Taylor–Hood pressure converging while its velocity does not (DBF tables)

Unchanged: internally coherent (pressure at the ℙ₁ floor 9.84e-6 despite oscillatory velocity error); one reinforcing half-sentence recommended.

---

## 5. Other defects (logic, organization, clarity)

**D1 — α_K "elemental minimum" contradiction** (§4 end vs §5 and Appendix C, which fixes \(\alpha_K=\alpha_{\infty,K}\)): confirmed a leftover **[v2]**; fix the §4 sentence.

**D2 — Local vs global α_∞ in Da_h.** Unchanged: the results discussion quotes the global Da_h ≈ 9.8 while the local convention gives ≈ 195 in the α₀ = 0.05 plateau; one clarifying clause, and unify α vs α_K across the three §6 limit subsections.

**D3 — Dimensional slip** in the intermediate factor "(Uν/h + 1)" of (DominantViscosityPressureGradientEstimate): should be \(U\nu/(Ph)\); final form correct.

**D4 — Table caption convention** defined only in the DBF captions; define at first use and add the regime-dependence caveat (the parentheticals are the viscous worst case; the reactive/convective rows meet the higher regime-specific prediction).

**D5 — Reaction terms and the OSGS projection — [v2: flipped].** The implementation (per `osgs_algorithm`, "ProjectResidualWithoutReactionWhenConstantSigma") applies the trim *only* for constant σ and reverts to projecting the full residual for the variable σ of the DBF runs — so the implementation matches (OSGSProblem) everywhere. It is the *paper's* §3.1 sentence ("we do not include the reaction terms in the calculation of the orthogonal projection") that reads as unconditional; qualify it: the exclusion applies when σ is constant (as in §7.1–7.2), where the trimmed component is identically zero; for §7.3 the full residual is projected. (The same policy trims the exact \((I-\Pi)(\varepsilon p_h)=0\) on the pressure side — worth the same parenthetical.)

**D6 — The ε_ref chain** (EpsilonRef): constants (100, 100/c₁) unverifiable as written; simplify or justify (the 3D runs need it only at Re = Da = 1).

**D7 — Scope sentence missing in §7** (linearized/ASGS theory vs nonlinear/both-variant experiments): one framing sentence.

**D8 — Minor wording**: the unclear "here as well" clause; the "nominal optimal rate" ambiguity (C3); the deferred-derivation footnote is now backed by an actual note (`c1_dimension_note`) — consider citing it as a preprint/report rather than "will be reported separately" **[v2]**.

**D9 — [v2: resolved.]** The appendices are audited; no outstanding unverifiable dependencies. (The `velocity_floor_regularization` smoothing affects only Newton tangents, not converged solutions — no bearing on reported errors; the `pressure_recentering` hardening is default-off.)

**D10 — [v2, new] The enlarged inverse constant is defined but never used.** Appendix C, Remark `winvconst`: because α is not piecewise polynomial, the α-weighted inverse estimates carry \(\overline C_{\text{inv}}=\sqrt{d\delta_\alpha}\,C_{\text{inv}}+C_\alpha\), and "all statements of the main text remain valid verbatim upon replacing \(C_{\text{inv}}\) by \(\overline C_{\text{inv}}\) in the conditions on \(c_1\)". The main text defines the macro `\Cinva` (line 184) and never uses it: the stability lemma's condition and the §7.2 Kuhn-threshold comparison are stated with the bare polynomial constant. Add the replacement remark (one sentence pointing to the appendix), and note in §7.2 that the α-enlargement thins the c₁ = 256 margin slightly further (harmless at α₀ = 0.5 with resolved gradients, but the statement should be honest).

**D11 — [v2, new] Symbol clash \(C_\alpha\).** Main text eq. (CAlpha) defines \(C_\alpha := \alpha + (h/|\boldsymbol k_0|)|\nabla\alpha|\) (a coefficient field in the τ design); Appendix C uses \(C_\alpha\) for the resolved-porosity constant of its Assumption *Porosity*. Same document, two meanings. Rename one (e.g. the appendix constant to \(c_\alpha\) or \(C_{\nabla\alpha}\)).

---

## 6. Summary of the consistency verdict

The numerical campaign is *more* than consistent with the theory, and after the theory audit the statement is stronger than in v1: the porosity-weighted convergence theorem is **proven**, its assumptions are **satisfied by the manufactured problem with explicit constants**, and the tables — once the best-approximation floor is computed — **empirically discriminate in favour of the weighted form** over the unweighted 1/α₀ bounds that §6 and the results discussion currently use. On structured 2D meshes the velocity error is best-approximation-exact in every probed regime; the pressure meets or exceeds the regime-dependent guarantees, with the ASGS/ℚ₂/viscous case confirming the sharpness of the working-norm loss; the one genuine anomaly (the OSGS reactive-regime velocity constant) runs *opposite* to the provable coercivity ordering established in the companion note and deserves to be reported; and the excluded 2D corner has a documented, theory-consistent stopping-criterion explanation that the paper should state. The remaining defects are wording, bookkeeping (α_K min/max, C_α clash, \(\overline C_{\text{inv}}\)), one table constant to verify (1.29 ×3), and the under-supported 3D quality-tail attribution.

---

## 7. Proposed modification of the discussion

### 7.1 New remark to close §6 (Robustness) — can now cite the appendix

```latex
\begin{remark}\label{rem:WeightedVsUnweighted}
The prefactor $1/\alpha_0$ appearing in the estimates of this section results from
bounding the elementwise weights of \cref{eq:ConvergenceResult} uniformly from above.
Retaining the porosity-weighted form of the theorem---which is the one actually
proved in \cref{app:Continuity} (see \cref{eq:convergence} and the closing remark
there)---the weight multiplying $\mathrm{E}_{\textup{int},K}(\boldsymbol{u})$ after
the normalizations above is $(\alpha_K/\alpha_0)^{1/2}$: it equals $1$ on the
elements where $\alpha_K \approx \alpha_0$ and never exceeds $\alpha_0^{-1/2}$.
Consequently, when the interpolation error of the exact solution concentrates in the
low-porosity region, the weighted estimates predict essentially no deterioration of
the (velocity) error beyond the growth of the interpolation error itself, and a
deterioration of at most $\alpha_0^{-1/2}$ otherwise. This distinction matters for
the interpretation of the numerical experiments: any manufactured velocity
compatible with $\nabla\cdot(\alpha\boldsymbol{u})=0$ necessarily scales as
$1/\alpha$, so that $\mathrm{E}_{\textup{int}}(\boldsymbol{u})$ itself grows as
$\alpha_0$ decreases, and observed errors must be compared against this baseline
before being attributed to the stability constants; see
\cref{sec:NumericalExamplesStationaryCase2D}.
\end{remark}
```

### 7.2 Replacement for the discussion paragraphs of §7.1

```latex
Overall, the observed rates meet or exceed the guarantees of
\cref{sec:StabilityASGS,sec:Robustness} in every case. The comparison can in fact be
made sharper than a rate check: since the exact solution is known, the nodal
interpolation error on the finest mesh provides an absolute reference. Measured
against it, the velocity error of both variants is best-approximation-exact to three
significant digits in the $H^1$-seminorm for the linear element and in both norms for
the biquadratic one, in all regimes with $\Reyn \leq 1$ (for instance,
$3.50\times10^{-2}$ against an interpolation value of $3.499\times10^{-2}$ for
$\mathbb{P}_1$ at $\alpha_0=0.5$); the stabilization thus introduces no measurable
accuracy loss there. The two methods differ mainly in the pressure, discussed below.

The effect of the minimum porosity must be read against the fact that the
manufactured velocity scales as $\alpha_0/\alpha$, so that its interpolation error
grows by factors of $5.0$ ($\mathbb{P}_1$) and $11.6$ ($\mathbb{Q}_2$) when passing
from $\alpha_0=0.5$ to $\alpha_0=0.05$. The observed velocity errors grow by exactly
these factors in the norms listed above: the degradation is entirely that of the best
approximation, and the stability constants contribute no additional loss,
consistently with the porosity-weighted form of \cref{eq:ConvergenceResult}
(\cref{rem:WeightedVsUnweighted}) and well below the uniform $1/\alpha_0$ bounds of
\cref{sec:Robustness}. The only quantity showing a residual method effect is the
$\mathbb{P}_1$ velocity in the $L^2$-norm, whose error relative to interpolation
grows by a factor between $2.0$ (ASGS) and $3.5$ (OSGS), compatible with the weighted
prediction $\alpha_0^{-1/2}\approx 3.2$. The pressure, whose exact field and
normalization are independent of $\alpha_0$, provides the clean test of the
constants: in the viscosity-dominated regime its error grows by factors of $8$--$10$
(ASGS) and up to $16$ (OSGS), which the estimates attribute to the coupling term
$\tau_2^{1/2}\mathrm{E}_{\textup{int}}(\boldsymbol{u})$, dominant in this regime and
growing precisely with the interpolation error of the velocity quoted above.

For viscosity-dominated flows ($\Reyn_h,\Damk_h\to 0$) the velocity is optimal
(indeed best-approximation-exact) in all cases and virtually identical for the two
methods. The pressure is where the analysis and the two stabilizations can be told
apart. For the biquadratic element the ASGS pressure converges at the rate predicted
by the working-norm analysis---one order below the interpolation-optimal one in both
norms ($\approx 1.9$ in $L^2$, $\approx 0.9$ in the $H^1$-seminorm)---so that the
loss of one pressure order in \cref{eq:DominantViscosityPressureGradientEstimate} is
not an artifact of the proof: it is observed, for the variant the analysis covers.
The OSGS pressure, by contrast, is interpolation-optimal, one order higher in both
norms; and for the linear element both variants are interpolation-optimal
($\approx 2$ in $L^2$). We attribute this pattern to the orthogonal projection
removing the finite-element-resolvable components of the residual that pollute the
ASGS pressure at the working-norm order for $k \geq 2$, while for $k=1$ the
corresponding (second-derivative) terms vanish elementwise~\cite{codina2008analysis}.

In convection-dominated flows ($\Reyn=10^6$; $\Reyn_h = 3125$ on the finest mesh, so
the regime is preserved throughout the sequence) the pressure recovers optimality, in
agreement with
\cref{eq:DominantConvectionXTermEstimate,eq:DominantPressureGradientXTermEstimate}:
its error reaches the approximation floor of the pressure space (e.g.\
$1.09\times10^{-2}$ in the $H^1$-seminorm for $\mathbb{P}_1$, equal to the
interpolation value). The biquadratic velocity rates sit slightly below their nominal
values ($\approx 2.8$ in $L^2$, $\approx 1.8$ in $H^1$), which we attribute to the
finest mesh not yet being within the asymptotic range at this elemental Reynolds
number.

For reaction-dominated flows ($\Damk_h\to\infty$) note first that
$\Damk_h = \Damk\, h^2/L^2 \approx 9.8$ on the finest mesh (and about $20$ times
larger within the low-porosity plateau at $\alpha_0=0.05$, where the elementwise
supremum of $\alpha$ enters its definition), so even the most reaction-dominated case
is only moderately so at that resolution. Increasing $\Damk$ from $1$ to $10^6$
improves the pressure down to its approximation floor ($8.8\times10^{-6}$ in $L^2$
and $1.09\times10^{-2}$ in $H^1$ for $\mathbb{P}_1$, at \emph{both} porosities), in
agreement with the $h$-independent mitigation factor of
\cref{eq:DominantReactionPressureGradientEstimate}; that the improvement also removes
the $\alpha_0$-dependence---which the estimate, as an upper bound, retains---is again
a floor effect. Conversely, the potential $\Damk^{1/2}$ deterioration of the velocity
gradient allowed by \cref{eq:DominantReactionVelocityGradientEstimate} does not
materialize: that term bounds the contamination of the velocity by the worst-case
pressure error, and since the computed pressure sits at its floor, the mechanism is
never activated. The ASGS velocity is unaffected to three digits; the OSGS
$\mathbb{P}_1$ velocity does show a moderate loss in the $H^1$-seminorm at
$(\Damk,\alpha_0)=(10^6,0.5)$ (a factor $3.5$ over ASGS, with the optimal rate
preserved), the only appreciable velocity discrepancy between the two variants in the
whole campaign; it vanishes at high $\Reyn_h$ and shrinks at $\alpha_0=0.05$. We note
that the provable reactive coercivity in this regime is, if anything, \emph{stronger}
for OSGS than for ASGS (for constant $\sigma$ the orthogonal projection annihilates
the reactive residual, sparing OSGS the drain that reduces the ASGS coefficient from
$\sigma$ to $\widetilde\sigma_\alpha$; cf.\ \cref{sec:LinearizationOfCoupledSystem}),
so the observed gap is an error-constant effect that our analysis does not resolve;
we report it as an observation.
```

### 7.3 Further point edits

1. §7.1, mesh-sweep sentence — state the reason for the excluded corner, e.g.: "…is excluded from the sweep: at this combination the nonlinear iteration cannot drive the (scale-free, absolute) algebraic residual to the prescribed tolerance on part of the mesh sequence — a saturation of $\tau_1$ at large $\Reyn_h$, aggravated at low porosity, that limits the attainable residual under the tight stopping gate rather than the accuracy of the converged discrete solution." *(Adjust to the actual failure mode observed; the mechanism is the one documented in the τ-saturation companion note.)*
2. Opening ¶ of §7.1: "…practically indistinguishable except in the strongly reaction-dominated cases discussed below".
3. First table caption: define the parenthetical convention and add the regime-dependence caveat (as in v1).
4. §6: fix "elemental minimum" (D1); fix the intermediate factor (D3); unify α vs α_K (D2); add the \(\overline C_{\text{inv}}\) remark near the stability lemma's condition on \(c_1\) (D10); resolve the \(C_\alpha\) symbol clash (D11).
5. §7.2: hedge the element-quality attribution toward pre-asymptotics (C6), verify the 1.29 entries (C7), temper "approach the theoretical optima" for ℙ₁-L², and consider citing the c₁-threshold note instead of "will be reported separately"; also acknowledge the α-enlarged inverse constant when comparing c₁ = 256 to the Kuhn threshold.
6. §7 opening: add the scope sentence (D7).
7. §3.1: qualify the projection-trim sentence — the reaction terms are excluded from the computed projection *when σ is constant* (their orthogonal component being exactly zero, as is the penalty term's); for the variable σ of §7.3 the full residual is projected (D5, flipped).
8. §7.3 closing: apply the same α₀-decomposition caveat to "consistent with the 1/α₀ dependence".
9. If the `osgs_reaction_note` paragraph is merged into §5, append the caveat that equal rates do not imply equal constants, citing the (Da, α₀) = (10⁶, 0.5) observation (C4).
10. Optionally add the interpolation-error reference row/efficiency column to the 2D tables (twelve numbers in §3).
