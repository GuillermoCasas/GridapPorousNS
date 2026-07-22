# chatgpt response

# Overall assessment

Your preferred option is essentially the best one, with one important correction:

> **Keep the complete statements of the principal ASGS and OSGS results, their hypotheses, the norms, and a concise proof roadmap in the main text. Move all proof details—including stability—to refereed appendices.**

I would **not** reduce the main text to merely citing appendix theorem numbers without stating the results. A reader should be able to understand exactly what has been proved, for which method, and under which assumptions without interrupting the main narrative to reconstruct the theorem from an appendix.

The revised manuscript is substantially better than the previous version. It now handles many of the scope and interpretation problems correctly. However, I would not submit it yet, primarily because:

1. the new broken-$\ell^2$ ASGS estimate is the right target, but the current continuity proof does not yet establish it as explicitly as the theorem requires;
2. the OSGS proof has a sound and valuable mathematical core, but several of its hypotheses, global estimates, and constants require correction;
3. the OSGS theorem currently analyzes a related idealized method rather than the precise method used in the numerical experiments;
4. several conclusions about stabilization constants and numerical robustness remain stronger than the available theory or evidence.

I do not see a fatal conceptual obstruction in either proof. The theory appears repairable and worth integrating. The most effective route is a **unified analysis with common lemmas**, not the addition of the standalone 19-page OSGS document to the manuscript.

---

# 1. Extent to which the revision addresses the previous audit

## Substantially addressed

The revision has successfully corrected or moderated many important points:

- The introduction now distinguishes the **linearized ASGS problem covered by the theorem** from the nonlinear ASGS and OSGS computations.
- Priority language has been moderated.
- The tensor discussion now correctly refers to removing the spherical or trace part, rather than the deviatoric part.
- The pressure-gradient scaling error has been corrected to involve
  $$
  \frac{\|\boldsymbol a\|_\infty U}{P}+1.
  $$
- Solver failure is no longer routinely identified with nonexistence of a discrete solution or a fold of the solution branch.
- The conclusion now recognizes the $\alpha_0^{-1}$ dependence that can appear in the pressure-gradient estimate.
- The previous claim that interpolation of $\alpha$ had been shown harmless has been replaced by the correct statement that this was not tested.
- The three-dimensional manufactured solution is now described as $z$-invariant or extruded, rather than fully three-dimensional in its analytical structure.
- The Taylor–Hood comparison is more carefully qualified, particularly regarding the pressure error.
- The centered parameterization now consistently acknowledges that $U$, $\nu$, and $\sigma$ vary.
- The OSGS projection indicator is no longer described uncritically as an orthogonal component when the precise implemented projection does not support that terminology.

These changes materially improve the manuscript’s credibility.

## Partially addressed

Several earlier concerns have been recognized but not fully resolved:

- The theorem hypotheses are still somewhat dispersed. A reader must gather assumptions about the mesh, porosity, pressure penalty, coefficient regularity, stabilization parameters, boundary conditions, and inverse constants from multiple places.
- The revised global convergence functional is much better, but the continuity lemma and its proof have not yet been rewritten consistently around it.
- The robustness discussion still sometimes moves between fixed-data $h\to0$ convergence and high-$\mathrm{Re}_h$ or high-$\mathrm{Da}_h$ pre-asymptotic scaling without clearly distinguishing those limits.
- The numerical method still uses pointwise stabilization parameters, while the proofs use elementwise constants.
- Important reproducibility information—raw errors, precise mesh sizes, degrees of freedom, quadrature choices, nonlinear residual definitions, stopping tolerances, and complete iteration data—remains incomplete.
- The relation between the formal OSGS projection and the implemented projection remains insufficiently explicit.

## Still requiring correction

The most important remaining manuscript-level problems are:

- The abstract says that stability and convergence of “the linearized method” are analyzed, but this is ambiguous once two stabilized methods are discussed. Before integrating OSGS, it should explicitly say **ASGS**; after integration, it should explicitly say **ASGS and OSGS**, with their distinct conclusions.
- The conclusion still presents $c_1=16k^4$ as theoretically predicted or confirmed. The present theory does not certify this value.
- A remaining broad statement that the method “attains Taylor–Hood accuracy” is still too general. It should be restricted to the reported velocity comparisons and should mention the cases where Taylor–Hood pressure is markedly more accurate.
- The phrase “interpolation floor” remains misleading. The nodal interpolation error is a useful benchmark, not a mathematical lower bound for the discrete error.
- The weak OSGS pressure-$H^1$ behavior in the three-dimensional example deserves more prominence and a more cautious interpretation.
- The DBF comparison changes more than one modeling or discretization ingredient, so it does not isolate the effect of the finite element pair.
- The claim that direct LU removes numerical-solver effects should be narrowed: it removes iterative linear-solver error, not nonlinear iteration error, discretization error, quadrature error, conditioning effects, or implementation error.
- The sources still contain numerous active revision and author-note macros. I found approximately 361 active `\amend` commands in the manuscript, 13 `\Guillermo` notes, 3 `\Joaquin` notes, and 7 further `\amend` commands in the revised continuity appendix. These must all be resolved before submission.

---

# 2. Revised ASGS continuity and convergence argument

## The new estimate is the correct direction

The manuscript now introduces the broken weighted-$\ell^2$ quantity

$$
\Psi_{\mathrm A}(h)^2
=
\sum_{K}
\alpha_K^2 h_K^{-2}
\left(
\tau_{2,K}E_{u,K}^2+
\tau_{1,K}E_{p,K}^2
\right).
$$

This is a much more natural global interpolation functional than the previous elementwise $\ell^1$ sum. It is compatible with broken finite element norms and avoids an artificial mesh-cardinality loss.

## The current proof does not yet quite prove the stated theorem

The revised continuity lemma still effectively establishes a bound in terms of the earlier $\ell^1$-type quantity, while the convergence theorem states the stronger $\Psi_{\mathrm A}$ result. The proof then says, in effect, that the stronger estimate follows “by inspection.”

That is not sufficiently rigorous. The desired result is probably already latent in the calculations, because the proof repeatedly forms broken $L^2$ norms and applies Cauchy–Schwarz. But the argument must be stated at the lemma level, rather than added retrospectively in the theorem.

The continuity lemma should directly read

$$
\left|
B_{\mathrm S}
\bigl(
\boldsymbol a;
U-I_hU,V_h
\bigr)
\right|
\le
C\,\Psi_{\mathrm A}(h)\,
\lVert V_h\rVert_{\mathrm S}.
$$

The absolute value is also necessary.

## Exact repair needed

The proof should be reorganized as follows:

1. Treat every volume residual contribution using a global discrete Cauchy–Schwarz inequality:
   $$
   \sum_K A_K B_K
   \le
   \left(\sum_K A_K^2\right)^{1/2}
   \left(\sum_K B_K^2\right)^{1/2}.
   $$

2. For each face contribution, first express the face sum through neighboring element patches.

3. Use trace inequalities, local quasi-uniformity, finite patch overlap, and local comparability of the stabilization parameters.

4. Square-sum the interpolation factors rather than summing them element by element.

5. Identify the first square root exactly with $\Psi_{\mathrm A}(h)$, up to the stated mesh and coefficient constants.

6. State the old $\ell^1$ expression only as a weaker corollary:
   $$
   \Psi_{\mathrm A}(h)
   \le
   \sum_K
   \alpha_K h_K^{-1}
   \left(
   \tau_{2,K}^{1/2}E_{u,K}
   +
   \tau_{1,K}^{1/2}E_{p,K}
   \right).
   $$

7. Derive all rate corollaries from $\Psi_{\mathrm A}$, not from the weaker majorant.

This repair should not require a numerical rerun. It is mathematical and expository work.

## Additional ASGS clarifications

The revised proof should also:

- add absolute values to all continuity inequalities;
- correct “there exist{s}”;
- state explicitly whether the discrete norm is a genuine norm or only a seminorm before boundary conditions and pressure normalization are imposed;
- include the required Korn or kernel argument;
- state the pressure normalization;
- collect the constant dependencies in one place;
- remove the assertion that a velocity obtained from a preceding finite element solve “trivially” satisfies
  $$
  \nabla\!\cdot(\alpha\boldsymbol a)=0.
  $$
  A standard discretely divergence-free velocity does not automatically satisfy this weighted condition;
- distinguish fixed-data asymptotic convergence from parameter-dependent pre-asymptotic regimes.

---

# 3. Assessment of the OSGS proof

## The proof contains genuine new mathematical content

The OSGS argument is not just an ASGS proof with notation changed. Its distinctive ingredients are:

- the weighted projection and fluctuation operators;
- the projection smoothing or approximation estimates;
- a porosity-weighted inf–sup construction;
- the special test function used to recover missing control;
- the treatment of the full reaction term in the OSGS norm;
- consistency and interpolation estimates adapted to the orthogonal-subscale structure.

This is sufficient reason to include an OSGS theorem in the paper. Doing so would also make the numerical treatment of OSGS much more balanced: the manuscript would no longer present a rigorous theorem only for one of its two central methods.

Nevertheless, the proof is not ready to be inserted unchanged.

## 3.1 The OSGS global convergence functional has the same $\ell^1$ problem

The current OSGS theorem uses a quantity of the form

$$
\mathcal E(h)
=
\sum_K
\frac{\alpha_K}{h_K}
(1+\mathrm{Da}_{h,K})^{1/2}
\left(
\tau_{2,K}^{1/2}E_{u,K}
+
\tau_{1,K}^{1/2}E_{p,K}
\right).
$$

For standard interpolation estimates, this elementwise $\ell^1$ sum may introduce an additional factor comparable to $N_h^{1/2}$, which is of order $h^{-d/2}$ on a quasi-uniform mesh. Consequently, the stated global rate corollaries do not follow from this expression without an additional and currently absent argument.

The OSGS theorem should instead use a broken weighted-$\ell^2$ functional such as

$$
\Psi_{\mathrm O}(h)^2
=
\sum_K
(c_1+\mathrm{Da}_{h,K})
\frac{\alpha_K^2}{h_K^2}
\left(
\tau_{2,K}E_{u,K}^2
+
\tau_{1,K}E_{p,K}^2
\right),
$$

up to an equivalent normalization of the harmless fixed constants.

The present $\ell^1$ quantity can remain as a convenient upper bound, but not as the principal functional from which optimal global rates are claimed.

## 3.2 The theorem does not yet analyze the exact implemented method

The most important scope issue is the distance between the proof and the numerical code.

The proof appears to use:

- a $\tau$-weighted projection;
- elementwise-constant stabilization parameters;
- a truncated or first-order residual/test structure;
- constant scalar reaction coefficients;
- a linearized Oseen–Brinkman-type problem.

The numerical experiments appear to use:

- a standard or unweighted $L^2$ projection;
- pointwise stabilization parameters;
- a fuller residual and adjoint structure;
- nonlinear DBF resistance;
- nonlinear iterations, sometimes with lagged projected quantities.

Thus, the theorem presently proves stability and convergence of a **nearby idealized OSGS method**, not automatically of the exact nonlinear implementation.

This is not disqualifying, but it must be disclosed. There are three possible responses, in descending order of rigor:

1. **Preferred:** adapt the proof to the projection and stabilization definitions actually used in the implementation.
2. Prove a rigorous equivalence or perturbation result showing that the implemented and analyzed choices differ only by controlled terms.
3. State explicitly that the theorem concerns an idealized linearized OSGS variant and that the nonlinear computations investigate a related implementation outside the theorem.

The manuscript should not silently pass from one method to the other.

A particularly important point is that the projection argument annihilating a constant reaction contribution does not transfer automatically to the nonlinear DBF term
$$
\sigma(\alpha,\boldsymbol u_h)\boldsymbol u_h.
$$

## 3.3 The porosity hypothesis must be strengthened

The OSGS proof uses a condition resembling

$$
h_K\lVert\nabla\alpha\rVert_{\infty,K}
\le C\alpha_K,
$$

where $\alpha_K$ is based on the local supremum. This does not provide the local lower-bound or patch-equivalence control subsequently needed with constants independent of the porosity contrast.

The natural replacement is either

$$
h_K\lVert\nabla\alpha\rVert_{\infty,K}
\le C\alpha_{0,K},
\qquad
\alpha_{0,K}:=\operatorname*{ess\,inf}_K\alpha,
$$

or a direct assumption that local suprema and infima are uniformly comparable on each element and its finite patch.

This should be harmonized with the corresponding ASGS assumption.

## 3.4 The mesh threshold is data-dependent

Several smoothing and perturbation estimates contain factors involving $\alpha_0^{-1}$, $\nu^{-1}$, coefficient derivatives, or related quantities. Therefore, the “sufficiently small” mesh threshold $h_0$ depends on these data unless stronger relative regularity assumptions are imposed.

The theorem should not suggest that $h_0$ is uniformly independent of viscosity, minimum porosity, or coefficient regularity.

A rigorous formulation would state:

> For fixed problem data satisfying the stated coefficient assumptions, there exists $h_0>0$, depending on the displayed coefficient and regularity constants, such that the result holds for $h\le h_0$.

## 3.5 The theoretical stabilization restrictions do not certify the numerical constants

The assumptions in the current proof lead to lower bounds of the approximate form

$$
c_1\ge 400\,C_{\mathrm{inv}}^2,
\qquad
c_2\ge20\,C_{\mathrm{inv}},
$$

depending on how the auxiliary constants are normalized. These are not compatible with treating the reported values $4k^4$, $16k^4$, and $2k^2$ as consequences of the proof.

There are three defensible approaches:

- present the theoretical restrictions as sufficient but highly non-sharp, and the numerical constants as empirical choices;
- sharpen the proof, particularly the special-test perturbation argument;
- provide local generalized-eigenvalue or inverse-constant diagnostics demonstrating that the reported values satisfy the actual mesh-dependent coercivity threshold.

The first option is the simplest and probably the most honest. The conclusion should not say the theory predicts $c_1=16k^4$.

A potentially useful route to sharper constants is to introduce the special test as

$$
W_h=U_h+\theta V_h^0
$$

and retain the favorable terms at order $\theta$, while absorbing perturbations in the form

$$
\eta D + C_\eta\theta^2 P.
$$

Choosing $\theta$ sufficiently small may replace some of the present very large sufficient constants by fixed positive lower bounds. This is a promising proof strategy, not yet a completed result.

## 3.6 Other technical corrections

Before publication, the OSGS proof should also address the following:

- The coefficient assumption presently stated in $W^{1,\infty}$ is insufficient where derivatives through order $k_u$ are used. Either require $W^{k_u,\infty}$ or formulate scaled derivative assumptions explicitly.
- One weighted approximation lemma states a regularity requirement only through $r-1$, while its proof differentiates $\alpha$ through order $r$.
- Patch equivalence of $\tau$ should be assumed or proved directly, rather than inferred from continuity alone in parameter limits where the diffusion floor can vanish.
- The OSGS energy quantity needs a kernel argument establishing that it is a norm on the stated discrete space.
- Inf–sup quotients should explicitly exclude the zero test function.
- One consistency estimate appears to turn
  $$
  \varepsilon^2\tau_2\le C_2^2\tau_2^{-1}
  $$
  into a factor $C_2^{3/2}$; after taking square roots the corresponding factor should be $C_2$.
- Projection assumption A7 is a substantive additional hypothesis, especially in its weighted form. It should be prominent and, ideally, proved for the chosen finite element spaces and weights.
- Discrete existence and uniqueness should be stated separately from consistency and convergence.
- Claims that a factor is the “exact price” or “exact reward” should be replaced by language such as “the factor produced by the present estimate.”
- The statement that the same proof covers a second method should be removed unless every projection and residual identity has actually been checked for that method.

My overall mathematical judgment is:

> **The OSGS argument is structurally credible and worth developing, but it requires a careful theorem rewrite and several nontrivial proof repairs. It should not yet be cited in the manuscript as proving the exact numerical OSGS method.**

---

# 4. Recommended integration into the paper

## Main text

The main theory section should contain approximately the following material:

### Common analytical setting

State once:

- the linearized continuous problem;
- boundary conditions and pressure normalization;
- mesh assumptions;
- porosity assumptions;
- coefficient regularity;
- finite element spaces;
- interpolation operators;
- stabilization parameters and their comparability assumptions;
- the distinction between fixed-data asymptotics and parameter-dependent regimes.

### The two stabilized formulations

Define the ASGS and OSGS bilinear forms side by side. This is the right place to expose their structural difference:

- ASGS controls the residual directly;
- OSGS controls projected fluctuations and requires a discrete inf–sup argument.

### Norms and error functionals

Define:

$$
\lVert\cdot\rVert_{\mathrm A},
\qquad
\lVert\cdot\rVert_{\mathrm O},
\qquad
\Psi_{\mathrm A}(h),
\qquad
\Psi_{\mathrm O}(h).
$$

### Full principal theorem statements

Keep in the body:

1. ASGS stability or coercivity theorem.
2. ASGS convergence theorem.
3. OSGS discrete inf–sup theorem.
4. OSGS convergence theorem.
5. Rate corollaries under explicitly stated polynomial and regularity assumptions.

Each theorem must say exactly which assumptions it uses. Avoid a theorem whose hypotheses can only be reconstructed from several appendices.

### Comparison and scope paragraph

Conclude the main theory section with a compact statement such as:

> The ASGS result follows from coercivity in the stabilization norm, whereas the OSGS result relies on projection estimates and a discrete inf–sup construction. Both results concern the stated linearized, elementwise-parameter formulations. The nonlinear implementations used in the computations incorporate additional choices identified in Section X and are not claimed to fall directly under these theorems.

That paragraph would resolve much of the current ambiguity.

## Refereed appendices

### Appendix A: Common analytical tools

Move all shared material here:

- mesh and patch notation;
- porosity comparability;
- stabilization-parameter bounds;
- interpolation estimates;
- inverse and trace inequalities;
- pressure normalization and norm-kernel lemma;
- common coefficient estimates.

### Appendix B: ASGS analysis

Include only ASGS-specific arguments:

- coercivity;
- viscous inverse estimate;
- jump or face estimates;
- corrected broken-$\ell^2$ continuity lemma;
- convergence proof and rate corollaries.

### Appendix C: OSGS analysis

Include only OSGS-specific arguments:

- projection definitions;
- weighted projection stability and approximation;
- smoothing estimates;
- proof or precise assumption for A7;
- special-test construction;
- discrete inf–sup proof;
- OSGS consistency;
- corrected broken-$\ell^2$ interpolation estimate;
- convergence theorem.

This eliminates the need to repeat the continuous problem, spaces, parameter definitions, interpolation notation, regularity assumptions, and standard inequalities in the OSGS appendix.

## What should not be added verbatim

The standalone OSGS document should not be appended in its present 19-page form. Its separate:

- abstract;
- introduction;
- literature motivation;
- repeated problem statement;
- repeated mesh and space definitions;
- repeated stabilization definitions;
- repeated interpolation material;
- repeated discussion;
- separate bibliography

should all disappear in the integrated paper.

Only the genuinely OSGS-specific mathematical core should remain.

A source-level comparison suggests that a unified presentation can remove roughly **one quarter to two fifths** of the combined theoretical material, depending on how aggressively the common coefficient estimates and interpolation lemmas are consolidated. That saving is meaningful, but it may not by itself bring the entire manuscript within a strict twenty-page target.

---

# 5. Is moving all proofs to appendices editorially sensible?

Yes. It is the strongest realistic presentation.

The essential distinction is between an **appendix that is part of the refereed article** and an unrefereed online supplement. Principal proofs should remain in the refereed document. Current SISC guidance recommends a maximum of twenty pages and says submissions above twenty-six pages may be returned without review; it also indicates that principal proofs should not be relegated to unrefereed supplementary material. ([epubs.siam.org](https://epubs.siam.org/journal/sisc/instructions-for-authors?utm_source=chatgpt.com), [epubs.siam.org](https://epubs.siam.org/journal/sisc/editorial-policy?utm_source=chatgpt.com)) SINUM similarly asks authors to limit submissions to twenty pages and treats longer manuscripts as exceptional. ([epubs.siam.org](https://epubs.siam.org/journal/sinum/instructions-for-authors?utm_source=chatgpt.com), [epubs.siam.org](https://epubs.siam.org/journal/sinum/editorial-policy?utm_source=chatgpt.com))

Therefore:

- **Main text:** theorem statements, assumptions, interpretation, comparison, and proof roadmaps.
- **Refereed appendices:** principal stability and convergence proofs.
- **Online supplement or repository:** extended elemental-matrix formulas, secondary Fourier algebra, large auxiliary tables, raw numerical data, scripts, and implementation details that are not logically necessary to validate the principal theorems.

Moving proofs to appendices improves readability, but it does not eliminate the total-length problem. The final decision must be made after compiling in the actual target journal class.

I could not obtain a reliable target-class page count from the supplied package because it is still missing `siamart190516.cls`, `shared.tex`, `figures/bump_plateau.pdf`, and a bibliography file under the referenced name `references.bib`. The standalone OSGS source compiles to nineteen pages only in generic article formatting, which is not an informative estimate of its integrated SIAM length.

If the unified manuscript remains above approximately twenty-six SISC pages after removing duplication and relocating nonessential computational material, the realistic options are:

1. split the work into a theory paper and a computational paper;
2. shorten the numerical survey substantially;
3. target a journal with a less restrictive length policy.

I would first attempt the unified paper. The ASGS–OSGS comparison is scientifically coherent and stronger than two loosely connected papers, provided the resulting manuscript remains readable.

---

# 6. Numerical reruns: what is and is not necessary

## No rerun is required merely to include the OSGS theorem

No new numerical experiments are logically necessary if the paper states honestly that:

- the theorem concerns the specified linearized OSGS formulation;
- the nonlinear numerical method contains additional implementation choices;
- the theorem supplies analytical motivation rather than a complete convergence proof for the nonlinear code;
- the reported stabilization constants are empirical rather than theoretically certified.

That is an acceptable and common scope boundary.

## Minimal bridge experiments that would materially strengthen the paper

To claim that the theory closely represents the implemented OSGS method, I recommend three small ablation studies rather than a rerun of the entire parameter grid:

1. weighted versus ordinary $L^2$ projection;
2. elementwise-constant versus pointwise stabilization parameters;
3. truncated analytical residual/test versus the full implemented residual/test.

Run these only in three representative cases:

- viscosity dominated;
- convection dominated;
- reaction dominated.

Report whether the solutions, errors, iteration counts, and projection indicators differ materially.

## Conditional reruns

Other computations are needed only if the corresponding strong claims are retained:

- pseudo-arclength continuation and Jacobian diagnostics for claims of folds or nonexistence;
- local eigenvalue or inverse-constant diagnostics if the numerical value of $c_1$ is to be theoretically certified;
- a stabilized Taylor–Hood comparator if superiority of the equal-order formulation is to be isolated from the effect of convection stabilization;
- additional meshes and projection diagnostics for a strong explanation of the three-dimensional OSGS pressure-$H^1$ behavior;
- a non-extruded manufactured solution for a strong claim of genuinely three-dimensional verification;
- analytic-versus-interpolated porosity runs if conclusions about interpolation of $\alpha$ are restored.

The simpler alternative is to narrow the associated wording.

---

# 7. Recommended sequence of work

1. **Repair the ASGS continuity lemma first.** Prove the broken-$\ell^2$ estimate explicitly, including both face terms.

2. **Rewrite the OSGS theorem around $\Psi_{\mathrm O}$.** Do not derive rate claims from the current elementwise $\ell^1$ functional.

3. **Resolve the analyzed-versus-implemented OSGS definition.** Preferably adapt the proof to the ordinary $L^2$ projection used by the code. Failing that, state the difference explicitly and add the minimal bridge experiments only where needed.

4. **Harmonize the common assumptions.** Use one mesh hypothesis, one porosity-resolution condition, one coefficient-regularity convention, one pressure normalization, and one notation for local parameter comparability.

5. **Rework the OSGS constants and dependencies.** Either sharpen the special-test proof or present the current restrictions as non-sharp sufficient conditions unrelated to the empirical $c_1$.

6. **Create the unified main theory section and three appendices.** Do not paste the standalone OSGS paper into the manuscript.

7. **Revise the abstract, introduction, numerical preamble, and conclusion only after the final theorem scope has been frozen.**

8. **Remove duplicated and nonessential material, compile in the target class, and then make the one-paper-versus-two-paper decision from the real page count.**

My realistic recommendation is therefore:

> **Integrate the OSGS results into the same paper. Keep both ASGS and OSGS theorem statements in the main text, move all stability and convergence proofs to shared and method-specific refereed appendices, and explicitly distinguish the analyzed linear formulations from the nonlinear implementations. Do not include the current standalone OSGS document verbatim.**

## Detailed deliverables

- [Full comparative audit of the revised manuscript and OSGS proof](sandbox:/mnt/data/revised_paper_osgs_audit.md)
- [Proposed theory-section and appendix architecture](sandbox:/mnt/data/theory_integration_blueprint.md)
- [Implementation instructions for the AI revision assistant](sandbox:/mnt/data/ai_revision_instructions_v2.md)

---
Powered by [AI Exporter](https://saveai.net)