# Blueprint for integrating the ASGS and OSGS theory

## 1. Editorial decision

Use a **results-in-main-text, proofs-in-appendices** architecture.

The main text must contain the analyzed methods, standing assumptions, norms, error functionals, complete theorem statements, and the relationship to the implementation. Every detailed coercivity, continuity, inf--sup, consistency, and convergence proof may be moved to appendices.

Do not place the principal proofs only in unrefereed supplementary material. Move reproducibility assets, elemental matrices, extended Fourier calculations, and secondary numerical results there instead.

---

## 2. Proposed main-text theory section

### 2.1 Common linearized setting

Define once:

\[
\alpha\boldsymbol a\cdot\nabla\boldsymbol u
-2\nabla\cdot(\alpha\nu\Pi_{DS}\nabla\boldsymbol u)
+\alpha\nabla p+\sigma\boldsymbol u=\boldsymbol f,
\]
\[
\varepsilon p+\nabla\cdot(\alpha\boldsymbol u)=0,
\qquad
\boldsymbol u|_{\partial\Omega}=0.
\]

State that this is the linear model analyzed below and that it differs from the nonlinear numerical model.

### 2.2 Standing assumptions

Create a table or compact assumption block.

| Assumption | Common | ASGS only | OSGS only |
|---|---:|---:|---:|
| Shape-regular, locally quasi-uniform mesh | yes |  |  |
| Full homogeneous Dirichlet velocity data | yes |  |  |
| Constant \(\nu>0,\sigma\ge0,\varepsilon\ge0\) | yes |  |  |
| \(0<\alpha_0\le\alpha\le1\) and local-inf porosity resolution | yes |  |  |
| Prescribed \(\boldsymbol a\), weighted solenoidal | yes |  |  |
| Elementwise-constant \(\tau_i\) | yes |  |  |
| Exact-solution regularity | yes |  |  |
| Reaction-dependent face comparability |  | yes |  |
| ASGS coercivity lower bound on \(c_1\) |  | yes |  |
| Weighted projection stability \(\beta_0\) |  |  | yes |
| OSGS smoothing threshold \(h\le h_0(\text{data})\) |  |  | yes |

Use the stronger local condition
\[
h_K\|\nabla\alpha\|_{L^\infty(K)}
\le C_\alpha\alpha_{0,K}
\]
in both analyses.

### 2.3 Stabilization parameters

Define \(\tau_1,\tau_2\) once and list only the common inequalities needed by the theorem statements. Put derivations in Appendix A.

### 2.4 Analyzed ASGS and OSGS forms

Display both forms next to one another. For OSGS, explicitly say that the projections are weighted and that the analyzed form is first-order/truncated.

### 2.5 Norms

Define
\[
\|V\|_A^2
=
\nu\|\alpha^{1/2}\Pi_{DS}\nabla v\|^2
+\|\widetilde\sigma_\alpha^{1/2}v\|_h^2
+\varepsilon\|q\|^2
+\|\tau_1^{1/2}\alpha X(V)\|_h^2
+\|\tau_2^{1/2}\nabla\cdot(\alpha v)\|_h^2,
\]

\[
\|V\|_O^2
=
2\nu\|\alpha^{1/2}\Pi_{DS}\nabla v\|^2
+\|\sigma^{1/2}v\|^2
+\varepsilon\|q\|^2
+\|\tau_1^{1/2}\alpha X(V)\|_h^2
+\|\tau_2^{1/2}\nabla\cdot(\alpha v)\|_h^2.
\]

State a lemma that these are norms under full Dirichlet data and pressure normalization. Put the Korn/kernel proof in Appendix A.

### 2.6 Error functionals

Use the exact square-summed quantities:

\[
\Psi_A(h)^2
=
\sum_K\frac{\alpha_K^2}{h_K^2}
\left(\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2\right),
\]

\[
\Psi_O(h)^2
=
\sum_K(c_1+\mathrm{Da}_{h,K})\frac{\alpha_K^2}{h_K^2}
\left(\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2\right).
\]

Do not make the \(\ell^1\) versions the principal theorems. They may be recorded as weaker corollaries.

### 2.7 Main theorem statements

#### ASGS theorem template

> Under assumptions (H1)--(H8-A), the linear ASGS problem has a unique discrete solution. There is a constant \(C_A\), independent of \(h\) and of the magnitudes of \(\nu,\sigma,\varepsilon\), but depending on the stated mesh, porosity-resolution, face-comparability, degree, and design constants, such that
> \[
> \|U-U_h^A\|_A\le C_A\Psi_A(h).
> \]

List the exact dependence on \(\alpha_0\) only through the displayed weights and named regularity/comparability constants; do not say “independent of \(\alpha\)” without qualification.

#### OSGS theorem template

> Under assumptions (H1)--(H8-O), including weighted projection stability and \(h\le h_0(\text{data})\), the linear OSGS problem has a unique discrete solution. There is a constant \(C_O\), independent of \(h\) and of the magnitudes of \(\nu,\sigma,\varepsilon\) once the assumptions hold, but depending on the named projection, mesh, coefficient-regularity, and design constants, such that
> \[
> \|U-U_h^O\|_O\le C_O\Psi_O(h).
> \]

If the large lower bounds on \(c_1,c_2\) are retained, put them directly in the theorem. If the scaled-test proof succeeds, state instead that \(c_1,c_2>0\) are fixed and that the inf--sup constant depends on them.

### 2.8 Proof roadmap

Use one paragraph:

> ASGS stability follows from coercivity of the stabilized quadratic form; continuity requires an ASGS-specific weighted inverse estimate and reaction-dependent face control. OSGS stability is inf--sup rather than coercive in the full residual norm; a smoothed projected test recovers the projected momentum and divergence components. The convergence proofs then combine stability with variant-specific interpolation continuity; the OSGS method is weakly inconsistent because the exact residual has a nonzero orthogonal fluctuation.

### 2.9 Comparison table

| Feature | ASGS | OSGS |
|---|---|---|
| Stability mechanism | coercivity | inf--sup |
| Reaction term in norm | damped \(\widetilde\sigma_\alpha\) | full \(\sigma\) |
| Consistency | exact for analyzed form | high-order consistency defect |
| Extra assumption | face comparability when \(\sigma>0\) | weighted projection stability, smoothing threshold |
| Main error factor | \(\Psi_A\) | \(\Psi_O\), with local \(c_1+\mathrm{Da}_h\) |
| Exact implemented scheme? | linearized nearby form | linearized weighted/truncated nearby form |

### 2.10 Scope paragraph

Retain a boxed or italicized paragraph immediately after the theorems:

> These theorems concern linearized schemes with prescribed weighted-solenoidal advection, constant scalar reaction, full Dirichlet velocity data, elementwise-constant parameters, and the displayed ASGS/OSGS forms. The computations use nonlinear advection, pointwise parameters, ordinary \(L^2\) OSGS projection, and, in the DBF test, nonlinear reaction. The computations therefore investigate nearby methods beyond the theorems.

---

## 3. Appendix structure

### Appendix A. Common analytical setting

1. mesh and patch notation;
2. local porosity comparability from the local-inf resolution condition;
3. stabilization parameters and common inequalities;
4. interpolation operators and local errors;
5. Galerkin cancellation identity;
6. deviatoric Korn/kernel lemma and pressure normalization;
7. constant-dependency convention.

### Appendix B. ASGS proof

1. ASGS bilinear form;
2. coercivity calculation;
3. ASGS weighted inverse estimate for the viscous residual;
4. face-jump lemma;
5. discrete continuity;
6. interpolation continuity with \(\Psi_A\), including explicit face sums;
7. convergence and existence/uniqueness.

### Appendix C. OSGS proof

1. weighted projections and exact annihilations;
2. patch-weight/smoothing lemma;
3. projection-stability assumption or proof;
4. OSGS norm and comparison with ASGS norm;
5. scaled or unscaled special-test inf--sup proof;
6. consistency defect;
7. interpolation continuity with \(\Psi_O\);
8. convergence and essential corollaries.

---

## 4. Material to delete from the standalone OSGS note when integrating

Delete or merge:

- separate title, abstract, and introduction;
- repeated continuous problem derivation;
- repeated mesh/space definitions;
- repeated stabilization-parameter definitions;
- repeated interpolation notation;
- repeated bibliography;
- long “consistency with the numerical campaign” remark;
- provenance acknowledgment;
- speculative Method II theorem unless proved;
- repeated robustness discussion already present in the main article.

Retain only the OSGS-specific mathematical core and one short interpretive remark.

---

## 5. Space-saving priorities

Save space in this order:

1. remove duplicate ASGS/OSGS setup;
2. move elemental matrices out of the article;
3. condense the Fourier appendix to the few equations that justify scaling;
4. replace long numerical paragraphs by observation/interpretation/limitation blocks;
5. retain representative rather than exhaustive tables in print;
6. place raw errors, full grids, code, and post-processing in a repository/supplement;
7. avoid restating theorem consequences in the abstract, introduction, robustness section, numerical section, and conclusion with slightly different wording.

Do **not** save space by removing theorem hypotheses or by putting all core proofs in unrefereed supplementary material.

---

## 6. Decision gates

### Gate 1: mathematical readiness

Do not integrate until:

- \(\Psi_A\) is proved at the continuity-lemma level;
- \(\Psi_O\) replaces the \(\ell^1\) theorem;
- OSGS A3/A4, weighted approximation, \(h_0\), and \(C_2\) errors are fixed;
- the \(c_1,c_2\) strategy is chosen;
- norm definiteness is established.

### Gate 2: implementation bridge

Choose one:

- **honest nearby-method route:** no new runs; clearly delimit theorem and implementation;
- **strong bridge route:** perform the three representative ablations for projection weight, parameter variability, and residual truncation.

### Gate 3: page count

Compile the complete paper in the target journal class.

- If comfortably within the policy: submit one unified paper.
- If only slightly over: shorten numerical exposition and appendices.
- If well over a hard threshold: split into theory and computational papers or choose a more flexible journal.

---

## 7. Recommended final narrative

The paper’s main story should become:

1. a common VMS framework gives two stabilized formulations for variable-porosity flow;
2. separate linear theories establish ASGS coercivity and OSGS inf--sup stability under explicit assumptions;
3. the estimates reveal common porosity weighting and a method-specific reaction factor;
4. nonlinear computations explore how these nearby formulations behave beyond the theorem;
5. the observed results support practical robustness but do not constitute proofs of sharpness or of equivalence between analyzed and implemented variants.

That story is clearer, more rigorous, and more novel than a paper that fully proves only ASGS while numerically emphasizing OSGS.
