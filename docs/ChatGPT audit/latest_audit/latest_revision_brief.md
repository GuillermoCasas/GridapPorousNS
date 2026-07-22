# Implementation brief for revising the latest manuscript

## Objective

Produce a clean, submission-ready manuscript in which ASGS and OSGS are treated in parallel, while every theorem states its exact hypotheses and every numerical claim remains within the evidence.

## A. Source and build

1. Rename `continuity_appendix(2).tex` to `continuity_appendix.tex`, or change the manuscript input. Confirm the compiled appendix contains the broken weighted-\(\ell^2\) function \(\Psi_{\mathrm S}\).
2. Rename `references(4).bib` to `references.bib`.
3. Add `shared.tex`, the journal class, and `figures/bump_plateau.pdf` to the submission package.
4. Flatten all `\amend`, `\Guillermo`, and `\Joaquin` markup and delete resolved comments.
5. Compile from a clean directory and check references, equations, floats, overfull boxes, bibliography, and total page count.

## B. Exact formulations

6. In the VMS section, define separately:
   - the analyzed \(\tau_i\)-weighted projection onto unconstrained spaces;
   - the ordinary \(L^2\) projection used in the nonlinear code.
7. Correct the nonlinear projection equation: either add the weighted product or label it as the implemented ordinary projection.
8. Replace the sentence saying the lagged iteration is analyzed. State that the OSGS theorem analyzes the stationary current-residual Method-I form.
9. Restrict the rigorous viscous theory to the deviatoric-symmetric projection, or add an explicit abstract Korn/coercivity hypothesis.

## C. Common assumptions

10. Split assumptions into Common, ASGS-specific, and OSGS-specific groups.
11. Move the strong ASGS penalty condition out of common data; retain only \(\varepsilon\ge0\) in common assumptions.
12. List exact assumption labels in each theorem; do not rely only on “standing assumptions.”
13. Adopt one pressure normalization in body and appendices. Prefer zero mean for the full-Dirichlet analytical problem for all \(\varepsilon\).
14. Move the OSGS pressure-mean correction into common tools and apply it in ASGS.
15. Use \(\ell_K\) for stabilization length and reserve \(h_K\) for element diameter.
16. Add the explicit relative-oscillation assumption for the continuous size surrogate.
17. State nodal Lagrange spaces if the smoothing proof remains nodal.
18. Strengthen \(\boldsymbol a\) to global \(W^{k_u,\infty}\), or replace the global best-approximation lemma by a valid broken-regularity result.
19. State explicitly that the OSGS theorem constant depends on \(\beta_0\), and that \(h_0\) depends on coefficient data.
20. Replace the patch-equivalence proof by the shared-vertex porosity argument; treat advection/\(\tau\) equivalence with a data-dependent small-mesh argument.

## D. ASGS appendix

21. Add absolute values in the continuity lemma, its sharper form, and all assembled continuity estimates.
22. After writing \(T_{13}=C+R\) and \(N=R+T_2+T_4+T_{11}\), replace every final use of \(|T_{13}|+|N|\) by \(|C|+|N|\), or regroup the complete signed sum before taking absolute values.
23. Add a common norm-definiteness lemma using the explicit deviatoric-symmetric identity.
24. State dependence of the coercivity constant on \(1-4C_{\mathrm{inv},\alpha}^2/c_1\).
25. Simplify the coercivity condition to \(c_1>4C_{\mathrm{inv},\alpha}^2\), unless a fixed \(\xi\) is needed elsewhere.
26. Display the two interpolation-error face estimates and the finite-overlap Cauchy–Schwarz step explicitly.
27. Mean-correct the pressure interpolant.
28. Rename \(\Psi\) to \(\Psi_{\mathrm S}\) and use the ASGS norm subscript throughout.
29. Remove claims that the local \(\alpha_K\) factor by itself proves improved low-porosity physical accuracy.

## E. OSGS appendix

30. Use the common norm-definiteness lemma; correct the phrase \(\varepsilon\|q\|=0\).
31. Correct the stability theorem’s dependence statement so it agrees with the smoothing proof.
32. Keep the projection-stability hypothesis prominent and explicitly conditional.
33. State \(c_1\ge1\), or replace inequalities using it by versions with \(\max\{1,c_1\}\).
34. Exclude the zero function in every inf–sup denominator.
35. Correct the explanation after the mass consistency estimate: the final step uses the parameter conversion corresponding to P5.
36. In the body, insert \(1+c_1/\mathrm{Da}_{h,K}\) in the \(L^2\)-velocity corollary.
37. Replace every varying-factor comparison \(\sqrt{1+\mathrm{Da}_h}\) by \(\sqrt{1+\mathrm{Da}_h/c_1}\) after absorbing the fixed \(c_1^{1/2}\).
38. Delete the numerical claim that the theorem quantitatively explains an error ratio near 3.5.
39. Qualify “uniform in \(\sigma\)” by the dependence on \(\beta_0\), \(h_0\), and exact-solution norms.
40. Say the coefficient of the pressure term decreases, not necessarily the physical pressure contribution.
41. Delete or shorten the unproved Method-II extension.

## F. Shared architecture and notation

42. Rename the first analytical appendix “Common analytical tools and ASGS analysis.”
43. Move common parameter inequalities, pressure mean correction, norm identity, interpolation notation, and porosity comparability to a neutral subsection.
44. Use one discrete-space notation, one error decomposition, one definition of \(\alpha_K\), and method subscripts on both norms.
45. Expand the comparison table so OSGS-specific patch/smoothing/coefficient assumptions are visible.
46. Rename `sec:StabilityASGS` to a neutral stability/convergence-analysis label.

## G. Robustness text

47. Replace unconditional robustness claims by parameter-explicit, component-specific statements.
48. Replace “verbatim” with “same explicit scaling up to fixed and projection-stability constants.”
49. Replace “exactly inflated” with “the right-hand functional carries an additional factor.”
50. Correct the body \(L^2\) equation as in item 36.
51. Clarify fixed-data \(h\to0\) versus simultaneous parameter/mesh limits.
52. Limit porosity claims to explicit displayed weights and mention hidden \(C_{\alpha,m}\), \(\beta_0\), \(h_0\), and solution-norm dependence.
53. Once the operator is fixed, state that the viscous norm controls the full \(H^1\) seminorm under full Dirichlet data.

## H. Numerical section

54. Replace “same discrete solution” by “same algebraic formulation and observed branch in the reported runs.”
55. Replace “genuine discretization errors” for LU runs by “not contaminated by iterative linear-solver stopping tolerances.”
56. Remove all fold/fold-recedes/root-exists claims. Report only the first meshes on which the tested solvers converged.
57. State explicitly that \(N=512,768\) are supplemental runs outside the declared \(10\)–\(320\) sequence.
58. Replace “fine enough to resolve” by “finer meshes on which the tested solver converged.”
59. Replace “any compatible velocity scales as \(1/\alpha\)” by a statement about the chosen manufactured field.
60. Call nodal interpolation a benchmark, not best approximation or floor.
61. Replace “loss is not an artifact” by “observed slopes coincide with the order allowed by the estimate.”
62. Label the OSGS pressure explanation as a hypothesis.
63. Remove quantitative comparison of the OSGS/ASGS ratio with the theorem factor.
64. Rewrite the 3D pressure/penalty paragraph to distinguish pressure normalization, boundary-flux incompatibility, and penalty regularization.
65. Restrict tetrahedral stabilization conclusions to the tested mesh families; do not say the weighted \(P_1\) viscous residual vanishes identically when \(\nabla\alpha\ne0\).
66. Call two-mesh quantities local observed slopes and parenthesized values nominal interpolation orders.
67. Temper causal claims from the stabilized Taylor–Hood control.
68. Add machine-readable raw data and full solver/mesh/quadrature metadata.

## I. Abstract, introduction, conclusion

69. Abstract: say “linearized ASGS and OSGS formulations”; replace “preventing” by “designed to control”; qualify robustness.
70. Introduction: state both theoretical contributions symmetrically and remove unsupported continuous uniqueness wording.
71. Conclusion: use “those theorems”; limit empirical claims to tested cases; distinguish explicit porosity weights from hidden constants.
72. Conclusion: say the OSGS bound is qualitatively compatible with, but does not quantitatively explain or prove necessary, the reaction-regime discrepancy.
73. Conclusion: restrict the tetrahedral and Taylor–Hood claims to the reported experiments.
74. Remove the untested expectation that porosity interpolation is benign.

## Acceptance checks

The revision is ready for a final audit only when:

- the compiled source contains the intended newest appendices;
- every theorem has an exact hypothesis list;
- the body and appendix OSGS formulas agree, including \(1+c_1/\mathrm{Da}_h\);
- no text claims a fold, nonexistence, exact quantitative prediction, or unconditional parameter robustness without supporting evidence;
- all revision notes are removed;
- raw numerical data and run metadata are available;
- the target-journal PDF has been checked page by page.
