# Revision instructions for an AI assistant

## Objective

Reorganize and revise the manuscript so that it contains a concise common ASGS/OSGS theory section, complete theorem statements in the main text, detailed proofs in refereed appendices, and exact distinctions between analyzed and implemented methods. Do not change numerical data unless a separate rerun package is supplied.

## Non-negotiable mathematical corrections

### 1. Repair the ASGS interpolation-continuity lemma

In `continuity_appendix(1).tex`:

1. Put absolute values around every occurrence of the continuity bilinear form.
2. Change Lemma `lem:continterp` to state
   \[
   |B_S(a;U-\widehat U_h,V_h)|\le C\Psi_A(h)\,|||V_h|||.
   \]
3. Define
   \[
   \Psi_A(h)^2=\sum_K\alpha_K^2h_K^{-2}
   (\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2).
   \]
4. Write the square-summed bounds explicitly for:
   - the interpolation triple norm;
   - the two bracketed norms in the assembled continuity estimate;
   - the pressure face term;
   - the convective face term.
5. Make the \(\ell^1\) functional a weaker corollary only.
6. Remove “sharpest” and use “stronger weighted broken-\(\ell^2\) form retained by the proof.”
7. Delete the retrospective assertion in the convergence proof that the stronger lemma follows “by inspection”; it will be unnecessary after the lemma is corrected.

Acceptance criterion: the convergence theorem follows directly by citing a lemma whose displayed conclusion already contains \(\Psi_A\).

### 2. Correct the ASGS theorem presentation

In `article(4).tex`:

1. Add absolute values in the continuity lemma.
2. Correct “there exist{s}” to “there exists.”
3. Replace the parenthetical claim that a previous finite element velocity satisfies the advection assumptions trivially. Say that finite element regularity is automatic but weighted solenoidality is an additional assumption.
4. Add a compact standing-assumptions block before the ASGS/OSGS theorem statements.
5. State constant dependencies explicitly.
6. Prove/cite norm definiteness under full Dirichlet data and pressure normalization, or call the triple quantity a seminorm.

### 3. Replace the OSGS \(\ell^1\) convergence theorem

In `osgs_convergence.tex`:

1. Replace `mathcal E(h)` as the principal error functional by
   \[
   \Psi_O(h)^2=\sum_K(c_1+\mathrm{Da}_{h,K})\frac{\alpha_K^2}{h_K^2}
   (\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2).
   \]
2. Retain every square-summed estimate in the consistency and interpolation proofs; do not enlarge it to an elementwise plain sum.
3. State the main theorem as
   \[
   |||U-U_h|||_O\le C\Psi_O(h).
   \]
4. Rewrite the ASGS-norm corollary with the same square-summed functional.
5. Rewrite the \(L^2\) velocity corollary in square-summed form and use the exact factor \(1+c_1/\mathrm{Da}_{h,K}\), unless absorption of \(c_1\) is explicitly declared.
6. State quasi-uniformity and bounded local convection factors when deducing a standard global interpolation order.

Acceptance criterion: no rate claim relies on an elementwise \(\ell^1\) sum whose number of terms grows with refinement.

### 4. Repair OSGS hypotheses

1. Replace the porosity condition based on \(\alpha_K=\sup_K\alpha\) by the common local-inf condition
   \[
   h_K||\nabla\alpha||_{\infty,K}\le C_\alpha\alpha_{0,K}.
   \]
2. Derive local element and patch comparability from that condition, or assume patch comparability directly.
3. Change the convection regularity from elementwise \(W^{1,\infty}\) to the actual order required by the consistency proof.
4. Replace or qualify the relative derivative condition on \(a\); use scaled derivatives or retain \(||a||_{W^{k_u,\infty}}\) in the constant.
5. Correct the weighted approximation lemma to require \(\alpha\in W^{r,\infty}\) when derivatives through order \(r\) are used.
6. State that the smoothing threshold \(h_0\) depends on the coefficient data under the present assumptions.
7. Display the weighted projection-stability hypothesis prominently and do not imply that the unweighted literature automatically proves it.
8. Write infimum and supremum over nonzero functions.
9. Correct the mass-consistency factor from \(C_2^{3/2}\) to the factor justified by the calculation.
10. State discrete existence and uniqueness as a consequence of the inf--sup result.

### 5. Decide how to handle \(c_1,c_2\)

Choose exactly one route and make it explicit.

#### Route A: retain current sufficient conditions

- Keep the large lower bounds.
- State that they are sufficient, not calibrated or necessary.
- State that the numerical constants have not been shown to satisfy them.
- Remove every claim that the theorem predicts or validates `c1=16 k^4`.

#### Route B: scaled-test proof

- Replace the special test by `W = U_h + theta V^0`.
- Re-estimate every perturbation as `eta D + C theta^2 P`.
- Choose `theta` after the fixed positive values of `c1,c2` are given.
- Track the resulting dependence of the inf--sup constant on `c1,c2`.
- Do not claim success until all terms and the test-norm bound are checked.

### 6. Remove OSGS overclaims

Replace:

- “exact price” by “factor introduced by the present proof”;
- “exact reward” by “stronger reaction control retained in the norm”;
- “sits exactly where the theory puts it” by “is compatible in order of magnitude with the bound”;
- “nor permitted at leading order” by no claim at all;
- “Da_h <= 1 on any reasonable mesh” by the exact local inequality;
- unproved Method-II equivalence by a prospective remark or a complete proof.

## Main-paper reorganization

### 7. Create a common main-text theory section

Keep in the main text:

1. common linearized problem;
2. analyzed ASGS and OSGS forms;
3. standing assumptions, with variant-specific items marked;
4. ASGS and OSGS norm definitions;
5. \(\Psi_A\) and \(\Psi_O\);
6. complete stability and convergence theorem statements;
7. one proof-roadmap paragraph;
8. one ASGS/OSGS comparison table;
9. one analyzed-versus-implemented scope paragraph.

Do not put line-by-line estimates in the main text.

### 8. Build three appendices

#### Common appendix

Move or consolidate:

- mesh/patch setting;
- porosity comparability;
- common parameter inequalities;
- interpolation estimates;
- Galerkin cancellation;
- norm-definiteness/Korn result;
- constant conventions.

#### ASGS appendix

Move:

- coercivity proof;
- ASGS viscous inverse estimate;
- jump lemma;
- discrete and interpolation continuity;
- ASGS convergence.

#### OSGS appendix

Move:

- weighted projection definitions and annihilations;
- smoothing and projection-stability machinery;
- inf--sup proof;
- consistency and interpolation continuity;
- OSGS convergence and essential corollaries.

Delete the standalone OSGS abstract, introduction, provenance note, duplicate setup, duplicate bibliography, and extended numerical discussion.

## Alignment of theory and computations

### 9. Insert an explicit mismatch paragraph

State that the OSGS theorem uses:

- weighted projections;
- elementwise-constant parameters;
- a first-order/truncated form;
- constant scalar reaction;
- prescribed weighted-solenoidal advection.

State that the computations use:

- ordinary \(L^2\) projection;
- pointwise parameters;
- full residual/adjoint terms;
- nonlinear reaction in the DBF case;
- nonlinear discrete iterates not exactly weighted solenoidal.

Conclude that the experiments concern a nearby but not identical method and are not a direct verification of the theorem.

### 10. Solver wording

- Report or request the full coupled nonlinear residual, including the projection equation.
- Do not say that a momentum tolerance alone proves the errors are unaffected by the solver.
- Describe lagging as an iteration strategy; at a converged fixed point, verify the projection equation rather than asserting equivalence.

## Robustness section

### 11. Separate the two limits

Create two introductory paragraphs:

1. fixed data with `h -> 0`, where `Re_h -> 0` and `Da_h -> 0`;
2. elementwise pre-asymptotic dominance, where `Re_h >> 1` or `Da_h >> 1` on the current mesh or in a simultaneous parameter--mesh limit.

For each displayed regime estimate:

- start from \(\Psi_A\) or \(\Psi_O\);
- state global quasi-uniformity or retain elementwise sums;
- show the use of \(\alpha_0\) and \(\alpha_\infty\);
- avoid “clearly optimal”; say “has the expected interpolation order under the stated scaling assumptions.”

## Numerical discussion

### 12. Apply observation--interpretation--limitation structure

Across every numerical subsection:

- replace “interpolation floor” by “nodal interpolation benchmark”;
- identify two-level slopes explicitly;
- do not use a one-sided upper bound as a verified prediction;
- discuss the 3D OSGS pressure-gradient behavior as a limitation;
- remove “purely discretization errors” after direct LU;
- qualify causal explanations;
- in the DBF comparison, state that stabilization and element-pair effects are confounded;
- change the summary before the conclusion to “comparable velocity accuracy in the reported cases.”

### 13. Stabilization-constant claims

Unless inverse constants are computed and the sufficient inequality is checked, replace every statement that theory predicts `c1=16 k^4` by:

> The sufficient stability condition motivates increasing `c1` for elements with larger inverse constants; the value used here was selected empirically and restored the observed convergence behavior.

## Abstract, introduction, conclusion

### 14. Abstract

Before OSGS integration, say “linearized ASGS method.” After integration, say “separate linearized ASGS and OSGS results.” Qualify robustness and do not imply nonlinear coverage.

### 15. Introduction

List the exact analytical scope once. Explain that the OSGS theorem concerns an analyzed variant. Avoid priority and superiority rhetoric.

### 16. Conclusion

State separately:

1. what is proved for linear ASGS;
2. what is proved for linear analyzed OSGS;
3. what is observed for nonlinear implementations;
4. what remains outside theory.

Do not claim that the OSGS theorem explains the nonlinear DBF test. Do not call proximity to the nodal interpolant a universal stronger fact.

## Reproducibility and source cleanup

### 17. Add reproducibility data

Provide or request:

- all mesh sizes and degrees of freedom;
- raw error values, not only slopes;
- quadrature orders;
- complete nonlinear and projection residual norms;
- all stopping criteria and maximum iterations;
- exact definitions of `h` in each table;
- stabilization evaluation convention;
- code/data repository and post-processing instructions.

### 18. Remove revision markup

Resolve every active `\\amend`, `\\Guillermo`, and `\\Joaquin` macro before submission. Remove stale comments and unused bibliography entries after a clean compile.

## Optional rerun package

Do not invent new numerical values. If reruns are authorized, request exactly three representative cases and compare:

1. plain versus weighted projection;
2. pointwise versus elementwise-constant `tau`;
3. full versus analyzed/truncated OSGS form.

Use one diffusion-, one convection-, and one reaction-dominated case. Report both errors and full coupled residuals.

## Final acceptance checklist

The revision is complete only when:

- both main convergence theorems are weighted broken-\(\ell^2\) estimates;
- every theorem is self-contained through a common assumption block;
- norm definiteness is established;
- OSGS `h0`, projection stability, coefficient regularity, and parameter restrictions are honest;
- the analyzed/implemented distinction is visible in the main text;
- all robustness quantifiers are correct;
- strong numerical claims are either supported by diagnostics or weakened;
- the source contains no active revision notes;
- the complete manuscript compiles in the target journal class and its page count is checked.
