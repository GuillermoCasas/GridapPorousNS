# AI implementation brief for revising the paper

## Mission

Revise the manuscript for maximum mathematical clarity and rigor. Preserve the underlying method and numerical data, but do not retain a statement stronger than the theorem or evidence supports. Keep four evidentiary levels distinct throughout:

1. **Definition/derivation:** what scheme is actually specified.
2. **Heuristic design:** what the Fourier argument motivates.
3. **Proved result:** the linearized ASGS theorem under its complete hypotheses.
4. **Empirical observation:** nonlinear ASGS/OSGS numerical behavior outside the theorem.

Do not invent data, rates, continuation diagrams, coercivity constants, or implementation details. Where a requested claim requires new computations, either insert a clearly marked placeholder for verified results or use the supplied text-only fallback.

Line numbers below refer to the uploaded source snapshot.

## Non-negotiable guardrails

- Do not call Newton/Picard failure proof of nonexistence or a branch fold.
- Do not call the nodal interpolant an error floor, lower bound, or absolute reference.
- Do not state or imply an OSGS convergence theorem.
- Do not state that the nonlinear DBF problem is covered by the linear constant-reaction theorem.
- Do not state that pointwise stabilization parameters are covered by the elementwise-constant analysis.
- Do not say porosity interpolation was tested unless new runs are supplied.
- Do not say errors degrade at worst as \(\alpha_0^{-1/2}\) without mentioning the \(1/\alpha_0\) reaction pressure-gradient exception.
- Do not claim that the equal-order method matches Taylor–Hood pressure accuracy; the tables contradict this.
- Do not use “theory confirms \(c_1=16k^4\)” unless the missing global coercivity evidence is inserted.

## Phase 0 — make the source package buildable and clean

### P0.1 Restore dependencies

Ensure the build directory contains:

- `siamart190516.cls` or the correct current class;
- `shared.tex`;
- `figures/bump_plateau.pdf`;
- the bibliography under the filename referenced by `\bibliography{references}`.

Rename `references(4).bib` to `references.bib` or change the LaTeX command consistently.

**Acceptance criterion:** a clean `latexmk -pdf` build from an empty auxiliary-file state.

### P0.2 Resolve revision markup

Accept/reject and remove all active wrappers:

- `\amend{...}`;
- `\Guillermo{...}`;
- `\Joaquin{...}`.

Remove the macro definitions after all wrappers are resolved.

**Acceptance criterion:** no active occurrences of these commands outside comments.

### P0.3 Clean source noise

- Remove the control character in the early comment of `continuity_appendix.tex`.
- Remove obsolete review comments and large dead-code blocks once no longer needed.
- Remove `\hfill\\` from section/subsection headings.
- Verify there are no unresolved references/citations after the clean build.

## Phase 1 — repair the theorem and robustness chain

This phase precedes all abstract/conclusion rewriting because it determines what can be claimed.

### T1. Replace the elementwise \(\ell^1\) theorem by the natural \(\ell^2\) estimate if the proof supports it

**Current locations:** `article:965–975`; `continuity_appendix:958–976`, `989–1017`.

Define

\[
\mathcal E_h^2 = \sum_K \alpha_K^2 h_K^{-2}
\left(\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2\right).
\]

Audit the interpolation-continuity proof term by term, especially interior-face terms. Use Cauchy–Schwarz, local quasi-uniformity, bounded face multiplicity, and parameter jump comparability to retain \(\mathcal E_h\).

If successful:

- state `|||U-U_h||| <= C \mathcal E_h` in the main theorem and appendix;
- call the old \(\ell^1\) expression a coarser corollary only;
- delete the claim that the \(\ell^1\) bound is “sharpest.”

If unsuccessful:

- retain the actual proved bound;
- carry all mesh-cardinality factors explicitly in the robustness section;
- remove claims of optimal global order not supported by that bound.

**Acceptance criterion:** the theorem, interpolation lemma, proof, and robustness section use the same globally defined quantity without an unexplained change from local to global notation.

### T2. Make the theorem self-contained

Replace the theorem's reference to the nonlinear iteration equation by a direct definition of the linear continuous and discrete problems. State:

- domain and dimension;
- homogeneous all-Dirichlet velocity conditions;
- pressure normalization;
- constants \(\nu>0\), \(\sigma\ge0\);
- advection regularity and \(\nabla\cdot(\alpha a)=0\);
- porosity positivity, \(W^{1,\infty}\) regularity, and mesh-resolution condition;
- mesh shape regularity/local quasi-uniformity;
- conforming velocity and **H1-conforming pressure** spaces and degrees;
- elementwise-constant \(\tau_i\), \(\alpha_K=\alpha_{\infty,K}\), elementwise velocity maximum;
- \(\varepsilon\) smallness condition;
- inverse-constant condition;
- jump condition when \(\sigma>0\);
- exact-solution regularity and consistency;
- constant dependencies;
- discrete existence/uniqueness.

Use the model theorem in `full_paper_audit.md`, Section 15, as the drafting basis.

### T3. Define the triple norm before the stability lemma

Move its definition before Lemma `lemma:Stability`. Add a short proof/remark that it is a norm for the chosen boundary conditions and pressure normalization, or rename it “mesh-dependent energy seminorm.”

### T4. Correct continuity statements

At `article:956–960` and the corresponding appendix lemmas:

- insert absolute values: `|B_S(a;U_h,V_h)| <= ...`;
- fix “there exist{s}” to “there exists”;
- state the jump condition directly;
- state the dependence of \(C\);
- do not make the reader infer assumptions through nested lemma references.

### T5. Correct Eq. `eq:DominantPressureGradientXTermEstimate`

Replace the factor

\[
\frac{\|a\|_\infty}{\sqrt P}+1
\]

by

\[
\frac{\|a\|_\infty U}{P}+1.
\]

Then state that it is \(O(1)\) only under the additional scaling \(P\sim U^2\) and \(\|a\|_\infty\sim U\).

### T6. Rebuild the robustness section from global weighted interpolation functionals

Introduce explicitly defined global quantities, for example

\[
\mathcal E_u^2=\sum_K\alpha_K^2h_K^{-2}\tau_{2,K}E_{u,K}^2,
\qquad
\mathcal E_p^2=\sum_K\alpha_K^2h_K^{-2}\tau_{1,K}E_{p,K}^2.
\]

For every dominant regime:

- state whether the regime is assumed elementwise throughout the mesh or only locally;
- distinguish fixed-data \(h\to0\) from parameter families where \(\mathrm{Re}_{h,K}\) or \(\mathrm{Da}_{h,K}\) is large;
- show every step that replaces \(\alpha_K\) by \(\alpha_0\) or a local number by a global one;
- distinguish a bound on a coupled residual from separate bounds on velocity and pressure terms;
- avoid calling a bound “optimal” unless the resulting global rate is demonstrated.

### T7. Correct the porosity-dependence summary

Every summary must say:

- most gradient estimates have a worst-case \(\alpha_0^{-1/2}\) factor;
- the reaction-dominated pressure-gradient estimate has \(\alpha_0^{-1}\).

Apply this to abstract, robustness section, and conclusion.

### T8. Correct the advection-assumption commentary

In `continuity_appendix:173–182`, replace the claim that the previous finite element iterate satisfies the assumptions trivially. State that continuity and elementwise regularity are automatic, while weighted solenoidality is an additional analytical assumption generally not satisfied by the discrete iterates.

### T9. Correct notation and model statements

- Replace `sigma 1_3` by `sigma I_d`.
- Correct the statement that the deviatoric part is removed; it is the spherical/trace part that is removed.
- Use one modified inverse constant consistently instead of silently reusing `C_inv` after absorbing porosity-gradient terms.
- Explain that the paper's “Damköhler” number is a reaction-number convention and is not the conventional porous-media Darcy number.

## Phase 2 — make the continuous and discrete formulations precise

### F2.1 State continuous-model scope honestly

At `article:254` and `264–268`:

- remove any suggestion that general nonlinear well-posedness is established;
- state that the general VMS derivation is formal;
- identify the special cited DBF well-posedness result and its small-data assumptions;
- state that the rigorous analysis begins only for the linearized problem.

### F2.2 Correct trace and traction spaces

Replace the single `D: X -> L2(boundary)^d` abstraction by separate Dirichlet trace and weak traction operators. Use proper `H^{1/2}`/`H^{-1/2}` pairings. Do not pair the full vector `V=[v;q]` directly with a velocity traction without extracting the velocity trace.

### F2.3 Standardize weak-form notation

Use:

- `(f,g)_omega` for L2 products;
- `<F,v>_{X',X}` for duality only;
- explicit boundary duality for traction.

### F2.4 Narrow the lifting claim

Replace “all developments apply equally” by a statement that nonhomogeneous data can be treated at the formulation level by a lifting, while the proof is only for homogeneous all-Dirichlet data and would require extra lifting/compatibility terms.

### F2.5 Separate pressure nullspace, boundary compatibility, and penalty

Rewrite the 3D explanation so that:

- zero mean fixes the constant pressure mode;
- exact discrete incompressibility with nonhomogeneous boundary values has a separate compatibility condition;
- the iterative penalty is a chosen regularization/fixed-point device.

Display the actual iterative-penalty equation and explain why the converged fixed point recovers the intended mass equation.

### F2.6 Define the actual OSGS method used in code

Introduce an explicit projection space `X_h^proj` without homogeneous boundary constraints if that is what the code uses. Rewrite the projection equation and all subsequent references accordingly.

State unambiguously:

- whether projection is tau-weighted or ordinary L2;
- which residual terms are projected in each experiment;
- whether reaction trimming changes only the nonlinear iteration or the converged scheme;
- whether advection/tau uses resolved, total, or lagged velocity.

### F2.7 Distinguish the displayed Picard algorithm from the Newton implementation

Rewrite the algorithm section as:

1. definition of the discrete nonlinear equations;
2. one possible staggered Picard solver;
3. the Newton/Newton–Krylov solver actually used in the reported runs.

Do not claim equivalence without either an algebraic argument or comparison data.

### F2.8 Reframe the Fourier section

Use “motivates,” “calibrates,” and “suggests scaling,” not “ensures” or “minimum.”

Clarify:

- generalized spectral radius assumptions;
- velocity-block embedding of reaction;
- discarded terms and O(1) factors;
- pointwise/elementwise parameter choice;
- scripts used for symbolic checks.

## Phase 3 — revise numerical reporting without new simulations

These edits are mandatory even if no reruns are performed.

### N3.1 Replace all nonexistence/fold wording

Search for:

- “has no solution”;
- “root exists from”;
- “fold”;
- “turning point”;
- equivalent wording.

Replace by:

> The tested nonlinear solvers and initializations did not converge to the prescribed tolerance on these meshes. The computations do not determine whether a discrete root is absent or merely difficult to reach because of conditioning or basin-of-attraction effects.

Retain strong branch language only when verified continuation data are inserted.

### N3.2 Fix the dimensional-encoding contradiction

At line 1118, replace “we take U=L=1 in all experiments” by:

> We fix the domain length scale \(L=1\). In the constant-reaction sweeps, the dimensional values of \(U\), \(\nu\), and \(\sigma\) are selected separately for each dimensionless cell by the centered encoding below. The DBF comparison uses the scaling stated in its own subsection.

### N3.3 Replace interpolation-floor language

Global search for “floor,” “absolute reference,” and “best approximation” near nodal interpolation discussion. Replace by:

- “nodal interpolation benchmark”;
- “same order/magnitude as the nodal interpolant”;
- “ratio to the nodal interpolation error.”

Delete any universal fixed-factor relation to best approximation unless a derivation is added.

### N3.4 Recast two-point slopes

Whenever only two converged levels exist, call the result a “two-mesh decrease estimate” or “pre-asymptotic two-point slope,” not a convergence order. Remove “superoptimal” as a theoretical descriptor.

### N3.5 Separate observations from explanations

Use this template in every results subsection:

> **Observation.** State only what the table/curve shows.
>
> **Relation to the estimate.** State whether it is consistent with a one-sided bound.
>
> **Limitation.** State what mechanism the experiment does not isolate.

Do not use “caused by,” “explained by,” “makes clear,” or “confirms” unless a controlled comparison supports it.

### N3.6 Rename the OSGS excess measure

Replace “error component” by “quadratic excess indicator,” define it only where nonnegative, and state that no orthogonality is implied. Prefer a simple ratio or signed difference.

### N3.7 Recast the 3D experiment

Replace “genuinely three-dimensional test” by “extruded three-dimensional tetrahedral discretization test.” State that the exact solution is z-invariant and has zero z-component.

### N3.8 Elevate the 3D OSGS pressure issue

Add a clear sentence in the 3D discussion and conclusion:

> The OSGS pressure H1 error decreases only weakly in several 3D cases and remains substantially larger than the ASGS error; the present analysis does not explain this behavior.

### N3.9 Correct the c1 narrative if no rerun is supplied

Use:

> The value `c1=16 k^4` was selected empirically for the tetrahedral calculations and produced stable convergent sequences in the reported tests. It lies below the local sufficient threshold discussed above, so the present coercivity proof does not certify this particular value.

### N3.10 Correct Taylor–Hood comparison language

Use:

- “velocity errors comparable in the viscosity-dominated cases”;
- “Taylor–Hood pressure is substantially more accurate in this test”;
- “the high-Reynolds comparison is against unstabilized Galerkin Taylor–Hood and therefore demonstrates the effect of residual convection stabilization, not superiority of equal-order spaces.”

Delete “matches Taylor–Hood accuracy” as a general statement.

### N3.11 Delete the unperformed porosity-interpolation conclusion

At line 1665, retain only that analysis of discrete/interpolated porosity is future work and cite the relevant literature. Delete “our numerical experiments indicate...” unless new data are inserted.

### N3.12 Clarify dimensional DBF coefficients

Insert the exact dimensional formula used in code and the units/scales. Describe the quoted Damköhler value as representative because the reaction depends on alpha and |u|.

## Phase 4 — numerical additions that require verified data

Implement only when the corresponding data/results are supplied.

### R4.1 Continuation near suspected folds

Required outputs:

- pseudo-arclength branch diagram;
- residual norm;
- Jacobian smallest singular value/eigenvalue;
- multiple initializations/directions.

Only then restore “fold/turning point.”

### R4.2 Porosity-interpolation experiment

Compare analytic alpha, P1 alpha, and degree-matched alpha on representative cases. Report positivity treatment and coefficient error.

### R4.3 Pointwise versus elementwise tau

Run one representative case per dominance regime and polynomial degree. State whether differences are within discretization variability.

### R4.4 Stabilized Taylor–Hood control

Use the same convection/reaction residual stabilization as far as mathematically appropriate. Report DOFs and costs.

### R4.5 True 3D manufactured solution

Use a full xyz-dependent weighted-divergence-free field with nonzero uz.

### R4.6 c1/coercivity study

Publish local eigenvalues, global Rayleigh quotients, and errors for at least three c1 values.

### R4.7 OSGS pressure diagnostics

Vary projection boundary space, quadrature, epsilon, normalization, and tolerances.

## Phase 5 — rewrite abstract, introduction, and conclusion

### A5.1 Abstract

Use the proposed abstract in `full_paper_audit.md`, Section 16, after updating it to match the final theorem. The abstract must include “linearized ASGS” and must identify nonlinear OSGS/DBF tests as beyond the theorem.

### A5.2 Introduction

Replace the current broad contribution paragraph by the four-item contribution paragraph in `full_paper_audit.md`, Section 17.

Remove:

- “only previously proposed stabilized method”;
- unquantified data-structure complexity claims;
- “no revision of theory was necessary”;
- any implication that numerical tests directly validate all theorem claims.

State prior literature factually:

- Codina 2001: ASGS and linearized analysis;
- Cocquet et al.: variable-porosity DBF, mixed BC, small-data well-posedness, Taylor–Hood with discrete porosity;
- Skrzypacz: variable-porosity linearized LPS/equal-order enrichment;
- Nillama et al.: stabilized equal-order Navier–Stokes–Brinkman implementation with spatially varying stabilization.

### A5.3 Conclusion

Use the proposed conclusion in `full_paper_audit.md`, Section 18. Ensure it contains:

- exact theorem scope;
- empirical results outside scope;
- alpha0 pressure exception;
- no fold/nonexistence claim without continuation;
- Taylor–Hood pressure gap;
- 3D OSGS pressure limitation;
- no porosity-interpolation claim without data.

## Phase 6 — improve tables, figures, and reproducibility

### Q6.1 Table labels

Change “theoretical rate” for OSGS rows to “nominal interpolation order” or “ASGS benchmark order.”

### Q6.2 Raw data

Add a supplement/repository containing every mesh-level error, h, DOFs, nonlinear iterations, residuals, and parameter values. In the paper, state how slopes were calculated.

### Q6.3 Reproducibility block

Add:

- Gridap/Julia versions;
- code commit/archive DOI;
- mesh-generation details;
- quadrature orders;
- exact forcing-generation method;
- nonlinear/linear solver and preconditioner;
- tolerances for momentum, mass, and projection residuals;
- hardware only together with DOFs/cost if computational limitations are discussed.

### Q6.4 Diagnostic quantities

Report per mesh:

- `R_alpha = max_K h_K ||grad alpha||_inf / alpha_0,K`;
- `max_K epsilon tau_2,K` for 3D;
- representative min/max pointwise tau values;
- element Re and reaction numbers using the defined h.

### Q6.5 Figure/caption wording

Change the porosity-figure caption to:

> Solid volume fraction \(1-\alpha\) used in the manufactured tests.

## Phase 7 — bibliography and final language pass

### B7.1 Bibliography

- Remove unused template/manual entries.
- Replace Semantic Scholar API URLs with DOI/journal metadata.
- Use bare DOI strings.
- Verify author spellings, publication type, pages/article numbers, and titles.
- Fix the question mark in the SPE conference title.
- Preserve acronym capitalization with braces.

### B7.2 Language

Global edits:

- “less versed reader” -> “readers less familiar with VMS”;
- “battery of tests” -> “numerical study”;
- “porous matrix's volume fraction” -> “solid volume fraction”;
- reduce “clearly,” “precisely,” “in fact,” and “it is interesting”;
- distinguish “stationary,” “steady,” and “time-independent” consistently;
- split paragraphs longer than roughly 200–250 words.

### B7.3 Final consistency search

Search the final source for the following terms and verify each occurrence manually:

`robust`, `optimal`, `sharpest`, `confirm`, `corroborate`, `verify`, `fold`, `no solution`, `root exists`, `floor`, `absolute reference`, `genuinely three-dimensional`, `matches`, `theoretical`, `interpolat* alpha`, `alpha_0^{-1/2}`.

Every occurrence must have scope/evidence appropriate to the final theorem and data.

## Final acceptance checklist

The revision is complete only when:

- the source builds cleanly from scratch;
- theorem and appendix state the same global error quantity;
- the robustness section follows from that quantity;
- Eq. 1052 is corrected;
- all theorem hypotheses are visible in one place;
- abstract/introduction/conclusion use identical scope language;
- OSGS/nonlinear/pointwise-tau/DBF results are explicitly outside the theorem;
- solver failure is not called nonexistence/fold without continuation evidence;
- interpolation benchmarks are not called floors;
- Taylor–Hood claims agree with both velocity and pressure tables;
- the alpha-interpolation claim is deleted or supported by new runs;
- the 3D OSGS pressure limitation is disclosed;
- revision-color macros and internal notes are gone;
- raw data/reproducibility details are available.
