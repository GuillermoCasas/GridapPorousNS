# Audit of the revised manuscript and the proposed OSGS theory

## Executive judgment

The revision is a meaningful improvement. It resolves or substantially mitigates many of the most visible scope and interpretation problems identified in the previous audit. In particular, the new text is much more honest about the distinction between the proved linearized ASGS problem and the nonlinear ASGS/OSGS computations; it corrects the pressure-gradient algebra; it no longer infers discrete nonexistence from nonlinear-solver failure; it corrects the porosity-interpolation claim; it identifies the extruded nature of the three-dimensional manufactured solution; and it narrows the Taylor--Hood comparison in several important places.

The revision is not yet ready for submission. The most important remaining mathematical issue is the new weighted broken-\(\ell^2\) ASGS convergence estimate. That estimate is probably recoverable from the existing argument, and it is the correct form to seek, but the appendix currently proves an interpolation-continuity lemma only with the coarser \(\ell^1\) functional and then asserts, in the convergence proof, that the stronger result follows by inspection. The stronger estimate must be stated and demonstrated at the continuity-lemma stage, including the two face terms. The robustness section must then be rederived from the final weighted functional with explicit mesh and asymptotic quantifiers.

The proposed OSGS analysis contains a genuine and potentially publishable theoretical contribution. Its central inf--sup construction is coherent in structure, and the distinction it reveals between ASGS and OSGS reaction control is useful. It is not ready to be inserted unchanged. The principal corrections are:

1. replace the current elementwise \(\ell^1\) convergence functional by the weighted broken-\(\ell^2\) functional that the estimates actually produce;
2. repair the porosity, coefficient-regularity, patch-equivalence, smoothing, and projection-stability hypotheses;
3. state honestly the data dependence of the mesh threshold \(h_0\);
4. address the very large sufficient lower bounds imposed on \(c_1,c_2\), which do not cover the constants used in the experiments;
5. distinguish rigorously between the analyzed OSGS method and the implemented nonlinear method;
6. remove several claims that infer a numerical mechanism or a relative performance prediction from a one-sided upper bound.

My publication recommendation is therefore:

> Retain the complete theorem statements, hypotheses, norms, error functionals, and an explanatory proof roadmap in the main body. Move all detailed ASGS and OSGS proofs, including stability/coercivity/inf--sup proofs, to refereed appendices in the same article. Organize those appendices into a common technical appendix and two short method-specific appendices. Do not put the principal proofs in unrefereed supplementary material, and do not reduce the main text to bare citations of results whose hypotheses and meaning the reader cannot see.

This is close to the option proposed by the authors, with one essential qualification: the theorem statements themselves and the exact relation between the analyzed and implemented schemes must remain visible in the main paper.

---

## 1. Extent to which the revision addresses the previous audit

### 1.1 Substantially addressed

The following changes are successful and should be retained.

#### Scope of the proved result

The introduction now says that the analysis concerns the linearized ASGS problem with constant scalar reaction and homogeneous Dirichlet data, and that the nonlinear ASGS and OSGS computations go beyond the theorem (`article(4).tex`, around lines 237--243). The numerical section repeats the distinction before presenting the experiments (around lines 1085--1096). The conclusion now begins its analytical claims with “For the linearized ASGS discretization” (around line 1658).

This is a major improvement. It eliminates the previous implication that the theorem directly covered the nonlinear OSGS simulations.

#### Pressure-gradient algebra

The convection-dominated pressure-gradient factor is now written as
\[
  \frac{\|\boldsymbol a\|_\infty U}{P}+1,
\]
and the text states the additional scalings \(P\sim U^2\) and \(\|\boldsymbol a\|_\infty\sim U\) needed to regard it as order one (`article(4).tex`, around lines 1054--1058). This directly addresses the algebraic error identified previously.

#### Solver failure versus nonexistence

The revised numerical discussion and conclusion now state that the tested nonlinear solver did not converge and explicitly decline to infer that the discrete problem lacks a solution (`article(4).tex`, around lines 1651 and 1658). This is the correct interpretation in the absence of continuation or branch diagnostics.

#### Porosity interpolation

The conclusion now acknowledges that \(\alpha\) and \(\nabla\alpha\) were evaluated analytically and that the experiments therefore do not test finite element interpolation of \(\alpha\) (`article(4).tex`, around line 1670). This corrects a direct factual contradiction.

#### Minimum-porosity dependence

The conclusion now records the \(1/\alpha_0\) exception for the reaction-dominated pressure-gradient estimate instead of saying that every quantity degrades at worst as \(\alpha_0^{-1/2}\) (`article(4).tex`, around line 1660).

#### Tensor terminology and dimensional notation

The text now says that the spherical/trace part, rather than the deviatoric part, is removed by the deviatoric-symmetric projection. The identity tensor notation was also generalized from a three-dimensional identity to \(I_d\). These are correct repairs.

#### Three-dimensional scope

The three-dimensional manufactured solution is now described as extruded or \(z\)-invariant rather than as a genuinely three-dimensional flow field. This is substantially more accurate. It remains a useful three-dimensional assembly and element test, but not a strong test of fully three-dimensional flow structure.

#### Taylor--Hood comparison

The conclusion now says that the stabilized equal-order method matches the *velocity* accuracy of Taylor--Hood in the reported DBF comparison and acknowledges the better Taylor--Hood pressure accuracy (`article(4).tex`, around line 1664). This is a material correction.

#### OSGS excess indicator

The manuscript no longer calls the square-root difference of squared ASGS/OSGS errors an orthogonal error component. That avoids assigning vector-space meaning to a scalar diagnostic for which no orthogonality has been established.

#### Centered dimensional encoding

The statement that all tests use \(U=L=1\) has been replaced by a consistent description: \(L=1\) is fixed, while \(U,\nu,\sigma\) vary according to the centered encoding (`article(4).tex`, around lines 1123 and 1136--1142).

### 1.2 Partially addressed

#### ASGS convergence estimate

The theorem now states the desired weighted broken-\(\ell^2\) estimate (`article(4).tex`, around lines 965--979), and the revised appendix defines
\[
\Psi_A(h)^2
=
\sum_K \frac{\alpha_K^2}{h_K^2}
\left(
\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2
\right).
\]
This is the right direction and resolves the conceptual problem with the old elementwise \(\ell^1\) theorem *provided the stronger estimate is actually proved*.

At present, however, the interpolation-continuity lemma still states only
\[
B_S(\boldsymbol a;U-I_hU,V_h)
\le C\,\psi_A(h)\,\lVert\!\lvert V_h\rvert\!\rVert,
\]
where \(\psi_A\) is the \(\ell^1\) majorant (`continuity_appendix(1).tex`, around lines 901--907). The convergence proof then says that the same lemma “holds with \(\Psi(h)\)” because the proof first contained mesh \(\ell^2\) norms (`continuity_appendix(1).tex`, around lines 1008--1014). That retrospective assertion is not a substitute for the lemma and its proof.

The repair is straightforward in concept but must be written explicitly; see Section 3 below.

#### Robustness analysis

The revision now defines a global interpolation quantity and uses the weighted theorem rather than the old undefined scalar notation. Nevertheless, the derivation still passes too quickly from an elementwise weighted sum to global scalar expressions. The following must be made explicit in every regime:

- whether the mesh is globally quasi-uniform or only locally quasi-uniform;
- which uses of \(\alpha_K\le1\), \(\alpha_K\ge\alpha_0\), and extremal element parameters are made;
- whether \(h\) denotes a global mesh size or an elementwise \(h_K\);
- whether the statement is a fixed-data convergence result or a pre-asymptotic parameter-regime statement.

The phrase “as \(h\to0\)” immediately before considering \(\mathrm{Re}_h\to\infty\) and \(\mathrm{Da}_h\to\infty\) remains mathematically misleading (`article(4).tex`, around lines 1000--1010). For fixed physical data, \(\mathrm{Re}_h=O(h)\) and \(\mathrm{Da}_h=O(h^2)\), so both tend to zero under refinement. Convection- and reaction-dominated element regimes are parameter-dependent pre-asymptotic regimes, not alternative fixed-data \(h\to0\) limits.

#### Numerical evidence language

The new numerical-scope paragraph correctly says that the estimates are one-sided and that an observed error below a bound does not verify sharpness (`article(4).tex`, around lines 1085--1096). Several later paragraphs still use stronger causal or predictive language. The paper should apply the same discipline consistently: observation, possible interpretation, limitation.

#### Taylor--Hood conclusion

The detailed paragraph and conclusion were narrowed to velocity accuracy, but the stand-alone summary immediately before the conclusion still says that the equal-order method “attains the accuracy” of Taylor--Hood on the DBF equations without qualification (`article(4).tex`, around line 1653). This sentence should be changed to “attains comparable velocity accuracy in the reported cases” and should repeat that the comparison does not isolate element-pair effects from convection stabilization.

### 1.3 Unresolved

Most reproducibility and source-hygiene issues were not touched by the revision. In particular:

- raw errors, mesh sizes, degrees of freedom, all nonlinear residuals, stopping criteria, quadrature orders, and post-processing definitions remain insufficiently documented for independent reproduction;
- pointwise stabilization parameters are still used in the computations while the theory needs elementwise constants, and “was observed not to alter the results” is not supported by an ablation;
- the porosity-resolution condition and the compressibility/penalty margin are not evaluated numerically;
- many rates still rely on two points or irregular pre-asymptotic subsets;
- the nodal interpolation error is still repeatedly called a “floor,” although it is only a benchmark and not a lower bound;
- the three-dimensional OSGS pressure \(H^1\)-type behavior remains a significant adverse result that deserves a dedicated discussion;
- the statement that direct LU makes errors purely discretization errors is too strong;
- the DBF comparison still conflates pressure--velocity pair choice with residual convection stabilization;
- the claimed theoretical prediction of the three-dimensional value \(c_1=16k^4\) remains unsupported by displayed inverse-constant data;
- the paper still contains active revision and author-note macros.

The current source contains approximately 361 active `\\amend` commands, 13 `\\Guillermo` notes, and 3 `\\Joaquin` notes. This alone means the source is not in submission form.

---

## 2. Remaining manuscript-level mathematical and stylistic issues

### 2.1 Abstract

The abstract says “an a priori stability and convergence analysis of the linearized method” and immediately says that two variants are considered. In the current paper this can still be read as an analysis of both variants. Before OSGS theory is integrated, write “of the linearized ASGS method.” After the OSGS theorem is integrated, write explicitly that separate results are established for the linearized ASGS and OSGS variants, and do not imply that either theorem covers the nonlinear DBF implementation.

The phrase “robust with respect to Reynolds and Damköhler numbers” should be qualified. The ASGS theorem gives parameter-weighted estimates, but some isolated controls still deteriorate with \(\mathrm{Re}_h\), \(\mathrm{Da}\), or \(\alpha_0\), and the OSGS estimate introduces an additional local \((c_1+\mathrm{Da}_{h,K})^{1/2}\) factor. A safer abstract formulation is:

> The estimates display explicitly how the stabilization, porosity, and local convection/reaction scales enter the error constants and identify the regimes in which the bounds remain uniform.

### 2.2 Standing assumptions and theorem readability

The current main-text ASGS theorem refers through several layers to the hypotheses of earlier lemmas. Referees should not have to reconstruct the theorem from scattered prose. Introduce a compact “Standing assumptions for the linear theory” block before the two theorem statements. It should state, in one place:

1. \(d\in\{2,3\}\), bounded polyhedral domain;
2. full homogeneous Dirichlet velocity boundary condition for the proof;
3. constant \(\nu>0\), constant scalar \(\sigma\ge0\), and constant \(\varepsilon\ge0\);
4. \(0<\alpha_0\le\alpha\le1\), required porosity regularity, and the local mesh-resolution condition;
5. prescribed \(\boldsymbol a\) with the exact regularity needed and \(\nabla\cdot(\alpha\boldsymbol a)=0\);
6. conforming, shape-regular, locally quasi-uniform meshes and continuous pressure spaces of the required regularity;
7. elementwise-constant stabilization parameters and their restrictions;
8. the ASGS face-comparability condition when \(\sigma>0\);
9. the OSGS weighted projection-stability hypothesis and small-mesh threshold;
10. exact-solution regularity and pressure normalization.

Then distinguish assumptions common to both variants from assumptions marked “ASGS only” and “OSGS only.”

### 2.3 The mesh-dependent “norm”

Both analyses call their triple quantities norms without proving that the deviatoric-symmetric gradient has no nontrivial kernel under the stated boundary conditions. Under full homogeneous Dirichlet data a suitable deviatoric Korn inequality should eliminate the conformal/rigid kernel, but this should be cited or proved. Then show that a vanishing residual term forces the pressure to be constant and that pressure normalization eliminates the remaining constant. Otherwise call the quantity a seminorm until the definiteness lemma is supplied.

This is particularly important for the OSGS inf--sup quotient, whose denominator must be definite.

### 2.4 Advection hypothesis

The ASGS appendix says that a previous continuous finite element velocity satisfies the advection assumptions “trivially.” Only continuity and elementwise regularity are automatic. The condition
\[
\nabla\cdot(\alpha\boldsymbol a)=0
\]
is not automatic for a generic discrete iterate. Replace the parenthetical sentence by:

> In a nonlinear iteration, the previous finite element velocity supplies the required continuity and elementwise regularity. Weighted solenoidality is an additional analytical assumption and is not generally satisfied by the discrete iterates.

The numerical-scope paragraph already makes the latter point and should be cross-referenced.

### 2.5 Continuity notation

Every continuity inequality must have an absolute value:
\[
\left|B_S(\boldsymbol a;U,V)\right|\le\cdots.
\]
This omission persists in the main text and the appendix. Also correct “there exists” in the main-text lemma.

### 2.6 Fixed-data convergence versus dominance regimes

Recast the robustness section around two logically distinct questions:

- **Fixed physical data, \(h\to0\):** then \(\mathrm{Re}_{h,K}\to0\) and \(\mathrm{Da}_{h,K}\to0\), and the eventual asymptotic regime is diffusion-scaled at the element level.
- **Parameter-dependent/pre-asymptotic regimes:** for a given mesh or a simultaneous parameter--mesh limit, some elements may satisfy \(\mathrm{Re}_{h,K}\gg1\) or \(\mathrm{Da}_{h,K}\gg1\). The formulas describe the corresponding local behavior of the estimator.

Do not write the three dominance cases as mutually exclusive \(h\to0\) asymptotics. State the quantifiers for every “uniform,” “optimal,” and “robust” claim.

### 2.7 Numerical-method description

The current text says that the accelerated Newton implementation converges to the same discrete solution as the staggered scheme and therefore does not affect the errors. At a genuine converged root this may be true if both algorithms evaluate the same nonlinear residual and projection equations. The paper should report the norm of the *full coupled residual*, including the projection equation, and state that equality of fixed points follows from the displayed algebraic systems. It should not rely on the momentum residual alone.

The OSGS theorem discussed below will make the implementation distinction even more important.

### 2.8 Numerical discussion

Adopt the following paragraph pattern throughout:

1. **Observation:** quote the actual trend, error ratio, or slope.
2. **Interpretation:** connect it cautiously to a term in an upper bound or to a known numerical mechanism.
3. **Limitation:** state confounding changes, number of data points, and whether the theorem covers the case.

Specific repairs still needed include:

- replace “interpolation floor” by “nodal interpolation benchmark/reference”;
- replace “the theory predicts \(c_1=16k^4\)” by “the sufficient condition motivates increasing \(c_1\); the value \(16k^4\) was selected empirically” unless the local inverse constants are measured;
- describe two-level slopes as two-level indicators, not established rates;
- discuss the three-dimensional OSGS pressure-gradient anomaly explicitly as a limitation;
- remove “purely discretization errors” after the direct-solver statement;
- in the DBF comparison, say that the equal-order method includes residual convection stabilization while the Taylor--Hood comparator does not, so the experiment does not isolate the finite element pair;
- delete causal claims such as “as befits a dedicated pressure space” unless supported by a controlled comparison;
- change the summary at line 1653 to velocity-specific, case-specific wording.

---

## 3. Audit of the revised ASGS continuity and convergence argument

### 3.1 What has improved

The appendix now identifies the natural weighted broken-\(\ell^2\) functional
\[
\Psi_A(h)
=
\left[
\sum_K \frac{\alpha_K^2}{h_K^2}
\left(
\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2
\right)
\right]^{1/2}
\]
and correctly notes that its elementwise \(\ell^1\) counterpart is only a majorant. This directly addresses the most serious defect in the previous theorem, because the \(\ell^1\) form can introduce a mesh-cardinality factor when converted to standard global interpolation norms.

The body of the continuity proof is already organized in broken \(L^2\) norms and uses Cauchy--Schwarz after summing the face contributions. Therefore the stronger theorem is plausible and, in my assessment, should be obtainable without a fundamentally new proof.

### 3.2 The current gap

The interpolation-continuity lemma at lines 901--907 is still stated with \(\psi_A(h)\), the \(\ell^1\) majorant. Its proof states:

- the interpolation error itself is bounded by \(\Psi_A(h)\) in the triple norm;
- the bracketed terms are bounded by \(\psi_A(h)\);
- the face estimates use the corresponding interpolation \(L^\infty\) bounds;
- the final result is \(C\psi_A(h)\).

The convergence theorem then upgrades the lemma to \(\Psi_A(h)\) by a prose assertion. That leaves the essential estimate unstated at the place where it is proved.

### 3.3 Required proof repair

Replace the interpolation-continuity lemma by
\[
\boxed{
\left|B_S(\boldsymbol a;U-I_hU,V_h)\right|
\le C\,\Psi_A(h)\,\lVert\!\lvert V_h\rvert\!\rVert .
}
\]
Then prove explicitly:

1. the interpolation triple quantity is bounded by \(C\Psi_A(h)\);
2. the two bracketed norms in the assembled continuity estimate satisfy
   \[
   \left\|\alpha_K\tau_2^{1/2}h^{-1}e_u\right\|_h
   +
   \left\|\alpha_K\tau_1^{1/2}h^{-1}e_p\right\|_h
   \le C\Psi_A(h);
   \]
3. after replacing the first-argument inverse estimates by interpolation estimates, the pressure face term is bounded by
   \[
   C\lVert\!\lvert V_h\rvert\!\rVert
   \left[
   \sum_K\frac{\alpha_K^2}{h_K^2}\tau_{1,K}E_{p,K}^2
   \right]^{1/2};
   \]
4. the convective face term is bounded by the interpolation triple quantity and hence by \(C\Psi_A(h)\);
5. only after the theorem is proved may the \(\ell^1\) expression be recorded as a weaker corollary.

The current facewise estimates strongly suggest these statements. Writing them out will make the upgrade rigorous and easy for a referee to verify.

### 3.4 Terminology

Do not call the estimate “sharpest” unless optimality in the constants or weights has been established. A safer phrase is:

> the stronger weighted broken-\(\ell^2\) form retained by the present proof.

### 3.5 Constant dependencies

The appendix says that the continuity constant is independent of \(\alpha\) and \(\boldsymbol a\), but the proof depends on named resolution, jump-comparability, mesh, and coefficient-regularity constants. State the dependency precisely:

> \(C\) is independent of \(h\) and the magnitudes of \(\nu,\sigma,\varepsilon\), but may depend on shape regularity, polynomial degree, \(c_1,c_2\), the porosity-resolution constant, the face-comparability constants, and the stated scaled regularity constants of \(\boldsymbol a\).

### 3.6 Consequence for the robustness section

Once the \(\Psi_A\) theorem is repaired, derive the regime estimates directly from \(\Psi_A\), not from an intermediate scalar heuristic. For each regime, show one line converting the weighted sum to the chosen global norm. This is where global quasi-uniformity or elementwise extrema must be declared.

---

## 4. Audit of the proposed OSGS theory

## 4.1 Overall assessment

The OSGS note is not merely a duplicate of the ASGS appendix. Its genuinely new parts are:

- the weighted projection framework;
- the patch-equivalence and smoothing construction;
- the special OSGS test function built from projected momentum and mass quantities;
- the inf--sup proof that recovers the projected components not controlled by the diagonal;
- the distinction between full reaction control in the OSGS norm and damped reaction control in the ASGS norm;
- the OSGS consistency error and its approximation by projection;
- the additional local reaction factor in the OSGS error bound.

Those results are valuable enough to integrate, after correction. The proof is structurally credible and closely follows an established OSGS strategy, but several hypotheses and rate deductions presently overstate what has been proved.

### 4.2 The analyzed OSGS method

The note analyzes
\[
B_{\mathrm{osgs}}(\boldsymbol a;U,V)
=
B(\boldsymbol a;U,V)
+
(\Pi_1^\perp[\alpha X(U)],\alpha X(V))_{\tau_1}
+
(\Pi_2^\perp[\nabla\cdot(\alpha u)],\nabla\cdot(\alpha v))_{\tau_2},
\]
where the projections are \(\tau_i\)-weighted projections onto unconstrained finite element spaces.

For constant \(\sigma\) and \(\varepsilon\), the fluctuations of \(\sigma u_h\) and \(\varepsilon p_h\) vanish exactly. This is correct for the weighted projection because these quantities belong to the corresponding finite element spaces. It justifies omitting those pieces from the projected residual for the *linear constant-coefficient problem*.

The analyzed method remains different from the numerical method in several other ways; see Section 5.

### 4.3 Positive features of the stability proof

The diagonal identity correctly controls the Galerkin viscous, reaction, and penalty terms plus the two fluctuation norms. The missing projected components are recovered through a special test function based on smoothed weighted projections. The mean correction in the pressure test is well chosen: because the weighted divergence has zero unweighted integral under full homogeneous Dirichlet velocity data, the constant shift is invisible to the relevant pairings.

The perturbation estimates have the right structure. The reaction pairing is absorbed using \(\sigma\tau_1\le1\), which explains why the OSGS norm can retain the full \(\sigma\|u_h\|^2\) term. The mass/penalty pairing is handled by \(\varepsilon\tau_2\le C_2\). The projected and fluctuation components are then combined through the projection-stability assumption.

Subject to the corrections below, this is a plausible inf--sup proof.

### 4.4 Blocker: the OSGS convergence theorem uses the wrong summation structure

The current OSGS error functional is
\[
\mathcal E(h)
=
\sum_K \frac{\alpha_K}{h_K}(1+\mathrm{Da}_{h,K})^{1/2}
\left(
\tau_{2,K}^{1/2}E_{u,K}
+
\tau_{1,K}^{1/2}E_{p,K}
\right).
\]
This is an elementwise \(\ell^1\) sum. The consistency, interpolation-continuity, and interpolation-norm estimates preceding the theorem are all obtained first as square roots of sums of squares and are enlarged to the plain sum only at their final step. The theorem should retain the square-summed form.

A natural exact functional is
\[
\boxed{
\Psi_O(h)^2
=
\sum_K
\bigl(c_1+\mathrm{Da}_{h,K}\bigr)
\frac{\alpha_K^2}{h_K^2}
\left(
\tau_{2,K}E_{u,K}^2
+
\tau_{1,K}E_{p,K}^2
\right).
}
\]
Then state
\[
\lVert\!\lvert U-U_h\rvert\!\rVert_O
\le C\Psi_O(h).
\]
If \(c_1\) is fixed, \(c_1+\mathrm{Da}_{h,K}\) may be replaced up to constants by \(1+\mathrm{Da}_{h,K}\), but the exact expression is clearer and avoids hiding dependence on \(c_1\).

The current \(\ell^1\) theorem can lose a factor comparable to \(N_h^{1/2}\sim h^{-d/2}\) on a quasi-uniform family. Consequently, the statements that the theorem gives full interpolation order do not follow from the theorem as presently written. This is not cosmetic; it is the OSGS analogue of the principal ASGS issue in the first audit.

### 4.5 Corrected velocity corollary

With \(\Psi_O\), the reaction-controlled velocity estimate should be written in square-summed form. A representative exact version is
\[
\|u-u_h\|
\le C
\left[
\sum_K
\left(1+\frac{c_1}{\mathrm{Da}_{h,K}}\right)
\left
a_K E_{u,K}^2+b_K E_{p,K}^2
\right)
\right]^{1/2},
\]
where
\[
a_K=1+\frac{c_2}{c_1}\mathrm{Re}_{h,K},
\qquad
b_K=\frac{\alpha_K}{\sigma\nu}.
\]
This follows from dividing by \(\sigma^{1/2}\) and using \(\tau_1\le1/\sigma\). The current factor \(1+\mathrm{Da}_{h,K}^{-1}\) silently absorbs \(c_1\); retain \(1+c_1/\mathrm{Da}_{h,K}\) unless the absorption is stated.

Full interpolation order then follows on a quasi-uniform mesh under fixed regularity and bounded local convection factors, provided the reaction-dominated condition holds on the elements concerned. The estimate is uniform in \(\sigma\), but not automatically in every other parameter.

### 4.6 Blocker/major issue: the porosity-resolution assumption is too weak for the patch arguments

Assumption (A3) uses
\[
h_K\|\nabla\alpha\|_{L^\infty(K)}\le C_\nabla\alpha_K,
\qquad \alpha_K=\sup_K\alpha.
\]
This does not by itself provide a lower bound comparing neighboring values of \(\alpha_K\). The patch-equivalence proof effectively needs either a small relative oscillation or a local lower-bound formulation. The fallback to the global ratio \(\alpha_\infty/\alpha_0\) also hides precisely the porosity dependence that the theorem claims to expose.

Use the same condition as in the revised ASGS appendix:
\[
h_K\|\nabla\alpha\|_{L^\infty(K)}
\le C_\nabla\alpha_{0,K},
\qquad
\alpha_{0,K}=\inf_K\alpha.
\]
Then derive \(\alpha_{\infty,K}\le\delta_\alpha\alpha_{0,K}\), and extend that comparison over a uniformly bounded patch. Alternatively, state patch comparability of \(\alpha_K\) and \(\tau_{i,K}\) as an explicit assumption and stop deriving it from an insufficient condition.

### 4.7 Major issue: the small-mesh threshold is data dependent

The smoothing proof contains factors such as \(1/(\alpha_0\nu)\) and moduli of continuity of \(\alpha\), \(\boldsymbol a\), and the mesh-size surrogate. Therefore the requirement \(\psi(h)\le\psi_0\) generally produces a threshold
\[
h_0=h_0(\alpha_0,\nu,\alpha,\boldsymbol a,\text{mesh family},\ldots).
\]
The stability constant may be independent of the magnitudes of those parameters once \(h\le h_0\), but the threshold is not uniformly parameter independent under the present assumptions.

State this honestly. For a robustness theorem uniform in the threshold as well, stronger scaled assumptions would be necessary, for example direct uniform local comparability of \(\tau_i\) over patches.

### 4.8 Major issue: the sufficient restrictions on \(c_1,c_2\) do not cover the experiments

The OSGS proof assumes
\[
c_1\ge\gamma^2 C_{\mathrm{inv}}^2,
\qquad
c_2\ge\gamma C_{\mathrm{inv}},
\]
with an explicit choice \(\gamma_0=10(1+M_2)\), hence at least of order 20 even before accounting for the porosity-gradient term. This implies very large sufficient constants, such as \(c_1\gtrsim400C_{\mathrm{inv}}^2\) and \(c_2\gtrsim20C_{\mathrm{inv}}\). The numerical choices \(c_1=4k^4\) or \(16k^4\), \(c_2=2k^2\), generally do not satisfy these displayed sufficient bounds for ordinary inverse constants.

There are three honest options:

1. retain the sufficient theorem and state that it does not certify the experimental constants;
2. rerun representative tests with constants satisfying an evaluated sufficient bound, if such constants remain numerically useful;
3. improve the proof.

A promising proof improvement is to use the scaled special test
\[
W=U_h+\theta V^0
\]
with \(0<\theta\ll1\), rather than \(W=U_h+V^0\). The productive projected terms are then of order \(\theta P_i\), while perturbations can be estimated as \(\eta D+C\theta^2P_i\). Choosing \(\theta\) sufficiently small, depending on fixed positive \(c_1,c_2\), should allow absorption without imposing enormous lower bounds. This is a plausible route, not yet a completed proof, and it must be worked through term by term. It would yield a stability constant depending on \(c_1,c_2\) rather than a uniform constant for all positive choices.

Until that refinement is completed, do not say that the OSGS theorem explains or validates the experimental parameter values.

### 4.9 Major issue: coefficient regularity

Assumption (A4) says that \(\boldsymbol a\) is elementwise \(W^{1,\infty}\) but later requires derivatives \(D^j\boldsymbol a\) for \(1\le j\le k_u\). Replace it by the actual regularity required, such as elementwise \(W^{k_u,\infty}\).

The inequality
\[
\|D^j\boldsymbol a\|_{L^\infty(K)}
\le C_D\|\boldsymbol a\|_{L^\infty(K)}
\]
is also highly restrictive and dimensionally unclear unless all variables are nondimensional. A more transparent assumption is
\[
h_K^j\|D^j\boldsymbol a\|_{L^\infty(K)}
\le C_{a,j}\|\boldsymbol a\|_{L^\infty(S_K)},
\]
or, without a relative assumption, retain \(\|\boldsymbol a\|_{W^{k_u,\infty}}\) explicitly in the error constant. The latter is less “robust” but more generally valid.

### 4.10 Error in the weighted approximation lemma hypothesis

The weighted approximation lemma says that the porosity derivative order \(m\) need only satisfy \(m\ge r-1\) in certain cases, while its proof applies the Leibniz rule through derivatives \(D^j\alpha\) for \(j=1,\ldots,r\). The lemma therefore needs \(\alpha\in W^{r,\infty}\), not merely \(W^{r-1,\infty}\). The global assumption \(m=\max(k_u,k_p)\) may already provide enough regularity in the applications, but the lemma statement is wrong and should be corrected.

### 4.11 Projection-stability assumption

Assumption (A7) is central, not technical decoration. It is a weighted analogue of a known unweighted projection condition, but the note does not prove that the weighted constant \(\beta_0\) is uniform for the chosen spaces and weights. The main theorem should display this assumption prominently.

There are two acceptable routes:

- keep it as an explicit hypothesis and state that verification for the chosen element families and weights is outside the paper;
- prove it from the unweighted result plus uniform local comparability of the weights over basis-function supports.

The second route would significantly strengthen the paper but is not necessary for a conditional theorem, provided the conditional nature is transparent.

### 4.12 Norm definiteness and inf--sup notation

Prove that the OSGS triple quantity is a norm under the stated boundary and pressure-normalization conditions, or call it a seminorm. In the inf--sup theorem, write the infimum and supremum over nonzero functions. As written, the quotient includes zero denominators.

### 4.13 Consistency estimate coefficient error

In the mass-consistency estimate, from \(\varepsilon\tau_2\le C_2\) one obtains
\[
\varepsilon^2\tau_2\le C_2^2\tau_2^{-1},
\]
so after taking square roots the factor is \(C_2\), not \(C_2^{3/2}\). Since \(C_2\le1\), the displayed \(C_2^{3/2}\) is smaller and is not implied by the preceding inequalities. Replace it by \(C_2\), or absorb the dependence into the generic constant after stating it correctly.

### 4.14 Existence and uniqueness

Once the finite-dimensional inf--sup estimate and continuity are established, state explicitly that the linear OSGS discrete problem has a unique solution. At present the note proceeds as though existence were automatic but does not record the consequence.

### 4.15 Overinterpretation of the OSGS factor

The discussion calls \((c_1+\mathrm{Da}_{h,K})^{1/2}\) the “exact price” of deleting the reactive subscale and full reaction control the “exact reward.” The proof establishes one upper bound obtained by a particular sequence of inequalities. It does not prove that the factor is necessary or sharp, nor that no alternative analysis can remove it.

Use:

> In the present proof, the conversion of the pressure and reaction interpolation terms introduces the local factor \((c_1+\mathrm{Da}_{h,K})^{1/2}\), while the OSGS norm retains full reaction control.

Similarly, the numerical comparison may say that the observed excess is compatible in scale with the bound, not that it “sits exactly where the theory puts it,” and never that an upper bound forbids a leading-order excess.

### 4.16 Method II and screening-length remarks

The Method II remark asserts the same theorem without supplying the modified proof. Either provide the necessary details in the appendix or reduce the statement to a prospective observation.

The screening-length interpretation is useful as a heuristic, but “\(\mathrm{Da}_h\le1\) on any reasonable mesh” is not a mathematical statement. State the exact condition \(h_K\lesssim(\alpha_K\nu/\sigma)^{1/2}\) and identify it as a local reaction--diffusion resolution criterion.

---

## 5. Does the OSGS theorem analyze the method used in the computations?

Not exactly. This distinction must be central in the integrated paper.

### 5.1 Projection inner product

The theorem uses \(\tau_i\)-weighted projections. The implementation uses the ordinary \(L^2\) projection. These coincide only when the relevant weight is globally constant. The weighted projection is essential to the current best-approximation argument.

### 5.2 Stabilized residual and test slot

The theorem analyzes a first-order/truncated Method-I form. The implementation displays the full residual and full adjoint test, including viscous pieces and the force contribution. Constant reaction and penalty pieces vanish under projection, but the viscous fluctuation, force fluctuation, and viscous adjoint term do not vanish in general.

### 5.3 Stabilization parameters

The proof requires elementwise-constant \(\tau_i\). The computations evaluate \(\tau_i\) pointwise. This affects both the definition of the projection and the elementwise manipulations.

### 5.4 Reaction model

The theorem assumes constant scalar \(\sigma\). In the DBF experiment, \(\sigma(\alpha,u_h)u_h\) is nonlinear and generally not an element of the finite element space, so the exact annihilation of the reaction fluctuation does not apply. The OSGS theorem therefore cannot be presented as an analysis of the DBF OSGS scheme.

### 5.5 Lagged projection

Lagging the projection is primarily a nonlinear-solver issue. If the iteration converges to a point at which the projection equation is satisfied with the converged residual, the lag disappears from the fixed-point equations. The paper should therefore report the final projection-equation residual. It should not automatically describe the lagged algorithm as a different converged discretization, nor assume equivalence without checking the full coupled residual.

### 5.6 Recommended wording

Use a prominent paragraph such as:

> The OSGS theorem concerns a linearized, first-order analyzed variant with elementwise-constant stabilization parameters and weighted orthogonal projections. The numerical implementation uses ordinary \(L^2\) projections, pointwise parameters, the full residual/adjoint form, and, in the DBF test, nonlinear reaction. The computations therefore explore a nearby but not identical scheme; they are not a direct verification of the theorem.

This is scientifically acceptable. It is much better than blurring the distinction.

---

## 6. How to integrate the ASGS and OSGS theory

### 6.1 Recommended main-text content

Keep approximately three to five journal pages of theory in the main body, containing:

1. the common linearized problem;
2. the exact analyzed ASGS and OSGS bilinear forms;
3. one compact standing-assumptions block, with variant-specific assumptions marked;
4. the definitions of the ASGS and OSGS mesh-dependent norms;
5. the definitions of \(\Psi_A(h)\) and \(\Psi_O(h)\);
6. the full ASGS stability and convergence theorem statements;
7. the full OSGS inf--sup and convergence theorem statements;
8. a short proof roadmap explaining the common and method-specific mechanisms;
9. a comparison table listing norm, consistency, extra assumptions, reaction factor, and distance from the implementation;
10. the scope paragraph distinguishing the nonlinear computations.

Do not retain long chains of estimates in the main body.

### 6.2 Recommended appendices

#### Appendix A: Common analytical tools

Include only once:

- mesh and patch assumptions;
- porosity bounds and local comparability;
- continuous problem and Galerkin cancellation identity;
- parameter definitions and common parameter inequalities;
- interpolation notation and standard estimates;
- the definiteness/Korn lemma for the mesh-dependent quantities;
- common constant-dependency convention.

#### Appendix B: ASGS analysis

Include:

- ASGS coercivity/stability proof;
- ASGS-specific weighted inverse estimate involving the viscous residual;
- reaction-dependent face-comparability lemma;
- discrete continuity proof;
- interpolation-continuity proof in the corrected \(\Psi_A\) form;
- ASGS convergence proof.

#### Appendix C: OSGS analysis

Include:

- weighted projections and the exact analyzed OSGS form;
- weighted projection-stability hypothesis;
- patch/smoothing lemma, or only the OSGS-specific part if patch comparability is in Appendix A;
- OSGS inf--sup proof;
- consistency estimate;
- interpolation-continuity estimate in the corrected \(\Psi_O\) form;
- OSGS convergence theorem and one or two essential corollaries.

Remove the standalone OSGS note’s separate abstract, long introduction, repeated continuous setting, repeated parameter derivation, extended numerical commentary, provenance note, and independent bibliography when integrating it.

### 6.3 What can genuinely be shared

The following material is common and should not be duplicated:

- domain, spaces, pressure normalization, and full Dirichlet setting;
- constant coefficients and prescribed weighted-solenoidal advection;
- mesh regularity and element/patch notation;
- porosity positivity and mesh-resolution assumptions;
- definitions and elementary bounds for \(\tau_1,\tau_2\);
- interpolation operators and local errors;
- global notation for weighted broken norms;
- exact-solution regularity;
- the Galerkin skew/cancellation identity;
- parameter and constant-dependency conventions.

The genuinely OSGS-specific material is roughly the projection framework, smoothing construction, projection-stability hypothesis, special-test inf--sup argument, consistency defect, and OSGS-specific reaction factor. The genuinely ASGS-specific material is the residual-adjoint coercivity calculation, viscous weighted inverse estimate, and reaction-dependent face terms.

### 6.4 Why theorem statements must stay in the main text

A main text that merely says “the results are proved in Appendix B/C” without displaying their hypotheses and conclusions would have three problems:

- readers could not understand what the numerical comparisons are being compared against;
- the paper’s main contribution would be hidden from the narrative;
- abstract and conclusion claims would become difficult to audit against exact results.

The theorem statements are not the source of the length problem. Duplicated setup, line-by-line algebra, extensive numerical prose, elemental matrices, and Fourier derivations are.

---

## 7. Realistic length assessment

The supplied source has approximately:

- 24,681 words in the revised main article source;
- 3,937 words in the revised ASGS continuity appendix;
- 8,265 words in the standalone OSGS note.

The standalone OSGS note compiles to 19 pages in its generic article layout. Adding it verbatim is therefore not realistic for a journal with a roughly 20-page target. A unified presentation can remove a substantial amount of duplication, plausibly around one quarter to two fifths of the combined theory material, but that will not make the total article short by itself.

A realistic compact target is:

- main theory summary: 3--5 SIAM pages;
- common analytical appendix: 3--5 pages;
- ASGS-specific proof appendix: 5--8 pages;
- OSGS-specific proof appendix: 6--9 pages.

These are estimates, not measured SIAM page counts. The complete package supplied here still lacks the SIAM class, `shared.tex`, the referenced figure, and the bibliography under the filename expected by the source, so a reliable final page count cannot be produced. The manuscript must ultimately be compiled in the target journal’s unmodified class before a submission decision is made.

For a strict SIAM-length target, the following material is better moved out of the article than the principal proofs:

- elemental matrix entries;
- detailed Fourier algebra beyond the scaling rationale;
- complete raw numerical tables or secondary parameter grids;
- code and machine-readable result data;
- implementation details that can live in a reproducibility repository.

Keep only representative tables/plots and the numerical facts needed to support the main claims.

---

## 8. Changes requiring no numerical rerun

The paper can become mathematically honest without rerunning the campaign if the authors are prepared to narrow the bridge between theory and implementation. The following are proof/text changes only:

1. repair the ASGS \(\Psi_A\) interpolation-continuity lemma;
2. rewrite the OSGS theorem in \(\Psi_O\) form and update all corollaries;
3. repair OSGS assumptions (porosity, regularity, patch comparability, \(h_0\), projection stability);
4. correct the \(C_2\) exponent and inf--sup notation;
5. prove norm definiteness or use “seminorm”;
6. state the exact analyzed/implemented mismatch;
7. present experimental \(c_1,c_2\) as empirical unless the sufficient condition is verified;
8. rederive the ASGS and OSGS robustness statements with explicit quantifiers;
9. weaken causal/predictive numerical wording;
10. narrow the DBF and Taylor--Hood claims;
11. move detailed proofs to unified appendices;
12. remove revision macros and complete reproducibility details.

Under this route, the OSGS results are still worthwhile: they explain the behavior of a clearly defined nearby linearized OSGS method and provide a theoretical comparison with ASGS.

---

## 9. Changes that may require reruns or new diagnostics

Reruns are necessary only if the paper wishes to make stronger claims linking the theorem to the exact implementation.

### 9.1 Minimal bridge study

A compact and high-value bridge study would compare, on three representative manufactured cases (diffusion-, convection-, and reaction-dominated):

1. ordinary \(L^2\) versus \(\tau\)-weighted projection;
2. pointwise versus elementwise-constant \(\tau_i\);
3. full residual/adjoint OSGS versus the analyzed first-order form.

This need not repeat the full parameter grid. It would quantify whether the analyzed and implemented variants are practically indistinguishable in the regimes used to discuss the theorem.

### 9.2 Nonlinear reaction

To claim that OSGS theory explains the DBF computations, a nonlinear extension or a dedicated linear/frozen-reaction comparison is needed. Otherwise state that the DBF experiment lies outside both linear theorems.

### 9.3 Stabilization constants

To claim that the theorem predicts \(c_1=16k^4\), compute or bound the relevant local inverse constants and show the sufficient inequality. A sensitivity study alone can show empirical necessity/usefulness but not theorem compliance.

### 9.4 Previously identified conditional reruns

The following remain conditional on retaining the corresponding strong claims:

- pseudo-arclength continuation and Jacobian diagnostics for a fold or nonexistence claim;
- a convection-stabilized Taylor--Hood control to isolate element-pair effects;
- additional three-dimensional meshes and a non-extruded manufactured solution for strong three-dimensional claims;
- targeted diagnostics for the weak OSGS pressure-gradient convergence;
- analytic-versus-interpolated-porosity tests for claims about interpolation of \(\alpha\).

---

## 10. Best realistic way forward

### Recommended route A: unified theory and computation paper

Choose this route if the unified manuscript can be brought within the target journal’s page policy after compilation.

1. Repair both weighted \(\ell^2\) convergence theorems.
2. Repair the OSGS hypotheses and decide how to handle \(c_1,c_2\).
3. Build a compact common theory section in the main text.
4. Put all proof details in common/ASGS/OSGS appendices.
5. Move elemental matrices, extended Fourier derivations, raw data, and code to supplementary/repository material.
6. Shorten the numerical narrative substantially and retain only representative evidence.
7. Compile in the exact journal class and count pages.

This is the scientifically strongest single-paper version because the paper analyzes both methods it compares numerically.

### Recommended route B: two-paper strategy

Choose this route if the integrated manuscript remains far above the journal limit or if the OSGS proof needs substantial further development.

- **Theory paper:** common porous Oseen setting; ASGS coercivity/convergence; OSGS inf--sup/convergence; comparison of norms, assumptions, and reaction behavior; a small set of validating linearized experiments.
- **Computational/method paper:** nonlinear ASGS/OSGS formulation, implementation, full parameter sweep, three-dimensional and DBF tests; cite the theory paper/preprint precisely and state which implemented variants it covers.

A computational paper should not present an unrefereed companion proof as settled theory. Ideally the theory paper is submitted concurrently or made available as a stable preprint with exact theorem numbering, and the computational paper uses appropriately conditional wording.

### My preferred decision

Try route A only after the proof corrections and an actual target-class page count. If the result exceeds a hard return-without-review threshold by more than a small amount, do not hide core proofs in supplementary material. Split the work or choose a journal with a more flexible length policy.

---

## 11. Ordered revision sequence

1. Freeze the exact analyzed ASGS and OSGS methods.
2. Repair \(\Psi_A\) and \(\Psi_O\).
3. Harmonize common assumptions, especially porosity resolution.
4. Prove/cite norm definiteness.
5. Correct OSGS projection, smoothing, regularity, \(h_0\), and constant-dependency statements.
6. Decide whether to improve the \(c_1,c_2\) proof with a scaled test or state the limitation.
7. Rewrite the main theorem statements and comparison table.
8. Reorganize the appendices around common material.
9. Rework the robustness section from the final functionals.
10. Align abstract, introduction, numerical preamble, and conclusion with the exact theorems.
11. Decide on the minimal bridge reruns.
12. Compress numerical prose and move reproducibility material to a repository/supplement.
13. Remove all amendment/author-note macros.
14. Compile in the target class and make the single-paper/split decision from the actual page count.

---

## Final assessment

The revision has addressed a substantial fraction of the previous framing and interpretation criticism, but it has not yet resolved the core convergence-proof issue or the broad numerical reproducibility issues. The new OSGS proof is worth developing and integrating. Its stability argument is promising, but its present convergence theorem, assumptions, parameter restrictions, and implementation claims require major revision.

The best presentation is not to keep stability proofs in the main text, and not to hide the theorem statements. Put concise, complete results in the main narrative and all detailed stability and convergence arguments in refereed appendices, sharing common tools aggressively. This gives the reader a coherent paper, protects rigor, and saves substantially more space than maintaining two parallel theory expositions.
