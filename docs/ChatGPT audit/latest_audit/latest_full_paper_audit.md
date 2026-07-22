# Full audit of the latest ASGS–OSGS manuscript

## Executive assessment

The latest version is a **substantial and successful revision**. The paper now has a coherent two-method analytical architecture: the body states parallel ASGS and OSGS stability/convergence results, the ASGS appendix proves a porosity-weighted broken-\(\ell^2\) estimate, and the OSGS appendix develops a genuinely different inf–sup argument with its own broken-\(\ell^2\) error functional. This addresses the most serious structural criticism of the previous version.

My present recommendation is nevertheless **major revision before submission**, but the reason is no longer a defective central theorem. The main mathematical mechanisms are credible. The remaining work consists principally of:

1. making the exact analyzed formulations and hypothesis sets unambiguous;
2. correcting two definite proof/bookkeeping errors and several smaller technical statements;
3. qualifying robustness claims whose constants still contain hidden parameter dependence;
4. removing numerical interpretations that go beyond the evidence; and
5. completing a clean, buildable source package.

The ASGS proof is close to publication quality after localized repairs. The OSGS proof is mathematically valuable and structurally convincing, but it remains a **conditional theorem** because of the weighted projection-stability assumption and the data-dependent smoothing threshold. That is acceptable if stated prominently and consistently.

The best presentation remains the current one in principle: full theorem statements and a compact comparison in the body, with complete proofs in refereed appendices. I recommend reorganizing the first part of the ASGS appendix as a neutral “Common analytical tools” appendix, because OSGS already imports several of its lemmas. Symmetry should mean parallel exposition and notation, not concealment of the genuinely stronger OSGS assumptions.

---

## 1. What has been successfully corrected

The revision has addressed most of the high-level defects identified previously.

- Both ASGS and OSGS are now theoretically represented in the body.
- The two convergence results use broken weighted-\(\ell^2\) functionals rather than the earlier elementwise \(\ell^1\) sum.
- The OSGS analysis is not presented as a trivial variant of ASGS: coercivity and inf–sup mechanisms are correctly distinguished.
- The manuscript explicitly states that the theory concerns linearized formulations with prescribed weighted-solenoidal advection, constant scalar reaction, full Dirichlet data, elementwise-constant parameters, and—in OSGS—the first-order residual and weighted projections.
- The nonlinear computations are now acknowledged to lie outside the theorem.
- The pressure-gradient algebra in the convection-dominated regime has been corrected.
- Fixed-data asymptotics are now distinguished from convection- and reaction-dominated pre-asymptotic regimes.
- The previous claims about porosity interpolation, fully three-dimensional exact fields, Taylor–Hood pressure equivalence, and solver failure implying nonexistence have largely been moderated.
- The OSGS proof now includes the pressure mean correction, exact consistency-defect estimate, interpolation continuity, norm comparison, and a separate reaction-robust \(L^2\)-velocity corollary.
- The numerical section now contains a stabilized Taylor–Hood control, which is a meaningful improvement over the earlier comparison.

These are major advances. The paper is now worth finishing as a unified ASGS–OSGS contribution.

---

## 2. Submission and source-package blockers

### 2.1 The latest ASGS appendix is not the file included by the manuscript

The manuscript contains

```tex
\input{continuity_appendix.tex}
```

but the newly supplied file is `continuity_appendix(2).tex`. Unless the file has been renamed outside the uploaded package, the compiled paper will use an older proof or fail to find the file. This is a **blocker**, because the latest broken-\(\ell^2\) proof may not be the one submitted.

**Required action:** rename the supplied file to `continuity_appendix.tex`, or change the input line and verify the compiled PDF contains the new lemma and theorem.

### 2.2 The bibliography name is still inconsistent

The manuscript requests `references`, whereas the supplied bibliography is `references(4).bib`.

**Required action:** normalize the file to `references.bib` and compile from a clean directory.

### 2.3 The package is not self-contained

The supplied package still lacks at least `shared.tex`, `siamart190516.cls`, and `figures/bump_plateau.pdf`. Consequently, I could not perform a reliable final typeset-page audit, check float placement, or verify the actual journal page count.

### 2.4 Revision markup and author notes remain active

Static inspection found approximately:

- 408 active `\amend{...}` commands in the main article;
- 13 active `\Guillermo{...}` notes;
- 3 active `\Joaquin{...}` notes;
- 20 further `\amend` commands in the elemental-matrices appendix;
- 3 in the Fourier appendix;
- 1 in the latest ASGS appendix.

There are no active unresolved internal references in the files inspected, but all author-note and revision-color macros must be flattened or removed before submission.

---

## 3. Exact method being analyzed: the largest remaining consistency issue

### 3.1 The body gives incompatible OSGS projection definitions

The theoretical section defines \(\Pi_i\) as a \(\tau_i\)-weighted projection onto the **unconstrained** finite-element spaces. The earlier VMS discretization, however, still presents the projection through a generic equation whose test space is described inconsistently, and the accompanying discussion/footnote is needed to explain that the implementation uses unconstrained spaces.

More importantly, the displayed projection equation

```tex
\dualpairing{W_h,\boldsymbol\pi_h}
 = \dualpairing{W_h,\mathcal R U_h}
```

is an ordinary \(L^2\) projection unless the dual pairing is explicitly weighted. It is not the \(\tau\)-weighted projection used by the OSGS theorem.

This is not a stylistic matter. Orthogonality, annihilation, the exact diagonal identity, and the projection-stability assumption depend on the product and the projection space.

**Required action:** separate three objects explicitly:

1. the formal OSGS projection used in the VMS derivation;
2. the weighted projection of the analyzed linear Method-I formulation;
3. the ordinary \(L^2\) projection used in the implementation.

A suitable body definition is:

```tex
For the analyzed OSGS formulation, let \(\Pi_i\) denote the orthogonal
projection onto the corresponding unconstrained finite-element space in the
broken weighted product
\((r,s)_{\tau_i}=\sum_K\tau_{i,K}(r,s)_K\), and set
\(\Pi_i^\perp=I-\Pi_i\). The nonlinear implementation instead uses the
ordinary \(L^2\) projection and pointwise stabilization parameters; it is a
nearby method not covered directly by the theorem.
```

The earlier nonlinear projection equation should either be weighted accordingly or be labeled explicitly as the implementation’s ordinary projection.

### 3.2 “Lagged form analyzed” contradicts the actual OSGS appendix

The linearization section says that the lagged projected residual is the form analyzed. The OSGS appendix instead analyzes a stationary bilinear form with the current unknown in the weighted projection and no nonlinear iteration index. Its consistency and inf–sup identities do not analyze the lagged Picard update.

**Required action:** use method-specific scope language. For example:

```tex
The ASGS theorem concerns the linearized residual form with prescribed
advection \(\boldsymbol a\). The OSGS theorem concerns the corresponding
stationary Method-I form with exact weighted projections of the current
finite-element residual. The lagged nonlinear iteration is a solver for the
implemented nonlinear method and is not itself the object of either theorem.
```

### 3.3 The OSGS appendix overstates agreement with the implementation

The appendix correctly notes differences in projection product and pointwise parameters, but one sentence says that the analyzed unconstrained projection “matches the implementation.” It matches only the **projection space**, not the inner product, parameters, nonlinear reaction, or residual structure.

**Required action:** replace “matches the implementation” with “matches the implementation in its use of the unconstrained projection space.”

### 3.4 The viscous operator is broader in the model than in the theory

The model permits a generic `\ViscProj`, but the appendices fix the orthogonal deviatoric-symmetric projection. Several norm-definiteness and coercivity statements rely on this choice.

Choose one of two routes:

- **Recommended:** state that the analysis is for the deviatoric-symmetric operator and that other projections require an analogous Korn/coercivity assumption.
- Or formulate a general standing hypothesis \(\|\nabla v\|\le C\|\ViscProj\nabla v\|\) on \(H_0^1\).

For the actual deviatoric-symmetric operator, the proof can be simplified by the exact full-Dirichlet identity

\[
\|\operatorname{dev}\operatorname{sym}\nabla v\|^2
=\tfrac12\|\nabla v\|^2
+\left(\tfrac12-\tfrac1d\right)\|\nabla\!\cdot v\|^2,
\]

which directly gives \(\|\nabla v\|\le\sqrt2\|\operatorname{dev}\operatorname{sym}\nabla v\|\) in \(d=2,3\). This is clearer than invoking a conformal Korn theorem with an unspecified domain constant.

---

## 4. Common assumptions and theorem hypotheses

### 4.1 “Standing assumptions” is still too ambiguous

The ASGS and OSGS theorems say “under the standing assumptions of the common setting.” The common section contains common assumptions, ASGS-specific conditions, OSGS-specific conditions, and regularity assumptions used only for convergence. Read literally, either theorem can appear to inherit the other method’s conditions.

**Required action:** state exact labels in every theorem:

- ASGS coercivity: common mesh/data/porosity/advection/spaces + ASGS coercivity/penalty assumptions.
- ASGS continuity/convergence: common assumptions + face-jump condition when needed + regularity for convergence.
- OSGS stability: common mesh/data/porosity/advection/spaces + OSGS design, projection, and patch/smoothing assumptions; no solution regularity.
- OSGS convergence: OSGS stability hypotheses + coefficient smoothness and exact-solution regularity.

Do not use a blanket phrase as the only hypothesis declaration.

### 4.2 The common pressure-penalty assumption is ASGS-specific

The common data assumptions impose the stronger ASGS pressure-penalty condition, while the OSGS proof only needs \(\varepsilon\tau_{2,K}\le C_2\). This makes the OSGS theorem inherit an unnecessary ASGS restriction and undermines the intended symmetry.

**Required action:** move the ASGS penalty upper bound into the ASGS-specific assumption group. Keep only \(\varepsilon\ge0\) in the common data, and state the OSGS bound separately.

### 4.3 Pressure normalization is inconsistent

The main weak formulation, ASGS appendix, and OSGS appendix do not use exactly the same convention for \(\varepsilon>0\). The OSGS norm proof says a constant pressure is removed by the \(\varepsilon\)-term; the main text sometimes describes a zero-mean pressure space regardless of \(\varepsilon\); elsewhere zero mean is imposed only when \(\varepsilon=0\).

**Recommended convention:** for the all-Dirichlet analytical problem, use the zero-mean pressure space for all \(\varepsilon\ge0\). When \(\varepsilon>0\), the equation itself is compatible with that normalization and the choice creates no loss. Then use the same space in both appendices and the numerical description.

If you retain the full \(L^2\) pressure space for \(\varepsilon>0\), rewrite every space definition and definiteness proof consistently.

### 4.4 ASGS pressure interpolation needs the same mean correction as OSGS

For \(\varepsilon=0\), a raw nodal/Lagrange interpolant of a zero-mean pressure need not have zero mean and therefore need not belong to \(Q_{h0}\). The OSGS appendix handles this correctly; the ASGS appendix does not.

**Required action:** reuse the OSGS mean-shift lemma in the common tools and apply it to both methods.

### 4.5 Local versus global quasi-uniformity

The common mesh hypothesis is local/patch quasi-uniformity, but the robustness section says global quasi-uniformity is “assumed throughout” when replacing local sums by a single \(h\).

**Required action:** either add global quasi-uniformity explicitly only to the simplified rate corollaries and numerical scaling discussion, or retain local \(h_K\)-weighted expressions. Do not retroactively claim it was assumed throughout.

### 4.6 The element length has two incompatible meanings

The OSGS smoothing construction uses a continuous size surrogate \(\chi_1(x,h)\), while \(h_K\) elsewhere denotes the geometric element diameter. At points the notation effectively redefines \(h_K\).

**Required action:** use \(\ell_K\) for the stabilization length and assume \(c h_K\le\ell_K\le C h_K\). Use \(\chi_1\) only to construct the nodal/smoothed \(\ell\).

### 4.7 Missing assumption on the size-surrogate oscillation

The smoothing proof invokes the fact that the relative oscillation of \(\chi_1(\cdot,h)\) over an element tends to zero, and claims \(O(h)\) under Lipschitz data. This property is not clearly included in the stated patch assumption.

**Required action:** state

\[
\max_{a\in N_K}\left|\frac{\chi_1(x_a,h)}{\ell_K}-1\right|
\le \omega_\chi(h),\qquad \omega_\chi(h)\to0,
\]

and \(\omega_\chi(h)=O(h)\) if that rate is used.

### 4.8 Coefficient smoothness assumptions are dimensionally and analytically incomplete

Bounds such as \(\|D^j\boldsymbol a\|_\infty\le C_D\|\boldsymbol a\|_\infty\) and \(1+\sum\|D^j\alpha\|/\alpha_0\) have dimensions unless the problem is declared nondimensional. More importantly, the OSGS consistency proof applies global approximation to \(\boldsymbol a\cdot\nabla u\). Elementwise smoothness plus mere continuity of \(\boldsymbol a\) is not enough to guarantee the required global \(H^{k_u}\) regularity.

**Required action:** either state that the whole analysis is written in nondimensional coordinates and strengthen \(\boldsymbol a\in W^{k_u,\infty}(\Omega)^d\), or formulate a broken-regularity approximation lemma that genuinely supports the proof used.

### 4.9 OSGS uses nodal spaces without assuming them

The smoothing operator is defined through nodal values and nodal shape functions, but the common space assumption only says “continuous polynomial spaces.”

**Required action:** state that the proof concerns standard nodal Lagrange spaces, or replace the construction by an abstract stable smoothing operator and list its properties.

### 4.10 OSGS projection stability is a strong conditional hypothesis

The hypothesis gives a constant \(\beta_0\) independent of \(h\), but the abstract and discussion sometimes treat the final constant as Reynolds- and Damköhler-uniform. This only follows if \(\beta_0\), the patch constants, and the smoothing threshold are themselves uniform in the physical parameters.

**Required action:** expose the dependence:

```tex
The hidden constant depends on the weighted projection-stability constant
\(\beta_0\). Parameter-uniform OSGS robustness therefore additionally
requires \(\beta_0\) to remain bounded away from zero over the parameter
family considered.
```

Ideally, add either a proof for the chosen spaces or a small generalized-eigenvalue diagnostic showing how \(\beta_0\) behaves in representative parameter regimes.

### 4.11 The OSGS mesh threshold is data-dependent

The smoothing proof contains \(\alpha_0^{-1}\), \(\nu^{-1}\), and coefficient moduli, so \(h_0\) is not uniform in these quantities. The appendix acknowledges this, while the body theorem initially lists a narrower dependence.

**Required action:** make the body theorem agree with the appendix: \(h_0\) depends on the coefficient data and may deteriorate as \(\alpha_0\downarrow0\) or \(\nu\downarrow0\).

### 4.12 Patch equivalence proof needs correction

The existing proof derives an additive estimate involving the maximum porosity on the patch and then claims multiplicative equivalence. That does not yield a contrast-independent result as written.

A clean proof is available on vertex patches. If \(K\) and \(K'\) share a vertex \(x_a\), the resolved-porosity condition gives

\[
\alpha_K\le\delta_\alpha\alpha_{0,K}
\le\delta_\alpha\alpha(x_a)
\le\delta_\alpha\alpha_{K'},
\]

and conversely. Extend through a uniformly bounded patch chain.

For \(|\boldsymbol a|\), multiplicative equivalence can fail near zeros. The diffusion floor in \(\tau_i\) can still give parameter equivalence for fixed \(\nu>0\) and sufficiently small \(h\), but the constants and threshold are data-dependent. State that directly.

---

## 5. VMS derivation and Fourier design argument

The VMS section is useful but should be presented as a derivation/design heuristic, not as a theorem.

### 5.1 The scale decomposition is formal

The notation \(\mathcal X_0=\mathcal X_{h0}\oplus\widetilde{\mathcal X}\) does not specify the topology or projection defining the complement. The exact Green-operator formula also assumes an invertible fine-scale operator and appropriate domains.

**Required action:** add a sentence stating that this is a formal VMS decomposition used to motivate the algebraic subscale model; the rigorous results begin with the explicitly defined discrete bilinear forms.

### 5.2 Zero subscales on element boundaries

The assumption means bubble-like, element-interior subscales whose trace is assigned to the resolved scale. It does not mean that the finite-element space resolves the physical boundary data “exactly.”

### 5.3 Fourier matching is heuristic

The mean-wavenumber replacement, neglected boundary traces, and additive combination of diffusion/convection/reaction scales are design approximations. Matching an operator norm does not uniquely identify the stabilization matrix.

**Required action:** label the section “Fourier scaling rationale” or similar, and replace statements such as “preserves stability” or “must provide the desired stability” with “motivates the scaling later subjected to variational stability analysis.”

### 5.4 Generalized spectral-radius notation

Define whether the matrix pencil is Hermitian positive semidefinite and whether \(\rho\) means the largest generalized eigenvalue or largest modulus. The current notation is hard to verify for a complex Fourier symbol.

---

## 6. ASGS appendix audit

### 6.1 Overall judgment

The ASGS appendix now has the right architecture:

1. exact consistency;
2. weighted inverse and parameter lemmas;
3. coercivity;
4. a continuity proof retaining broken \(\ell^2\) norms;
5. interpolation-error continuity;
6. the convergence theorem with \(\Psi_{\mathrm S}(h)\).

The core coercivity algebra and the global \(\ell^2\) conversion are sound in structure. The appendix is close to publishable after the following corrections.

### 6.2 Missing absolute values in the continuity lemma

The lemma states \(B_S(U_h,V_h)\le\cdots\) rather than \(|B_S(U_h,V_h)|\le\cdots\). The sharper version has the same defect, as do a few intermediate assembled estimates.

**Required action:** put absolute values around all continuity pairings and consistently estimate grouped sums rather than individual signed terms where cancellation is used.

### 6.3 Definite bookkeeping error in the \(T_{13}\) group

The proof splits \(T_{13}=C+R\), estimates \(C\), and defines

\[
N=R+T_2+T_4+T_{11}.
\]

The final assembly later writes a bound involving \(|T_{13}|+|N|\), which double-counts \(R\) and does not follow from the preceding estimates. The valid quantity is

\[
|C|+|N|,
\]

or, better, the absolute value of the complete grouped contribution.

This is a **definite proof error**, but it is local and easy to repair.

### 6.4 The ASGS norm’s definiteness should be proved once and shared

The OSGS appendix proves norm definiteness; the ASGS appendix uses the corresponding quantity as a norm without an equally explicit kernel argument. Introduce a common lemma using the explicit deviatoric-symmetric identity and pressure normalization.

### 6.5 Coercivity-constant dependence is misstated

The coercivity constant is the minimum of terms including \(1-4C_{\mathrm{inv},\alpha}^2/c_1\). It therefore depends on the **coercivity margin** and tends to zero as \(c_1\downarrow4C_{\mathrm{inv},\alpha}^2\). It is not a constant depending only generically on shape regularity and \(c_1\) without noting this margin.

The condition written as “\(c_1>2\xi C^2\) for some \(\xi>2\)” is existentially equivalent to \(c_1>4C^2\), but this is needlessly indirect. State the simple condition, or fix \(\xi\) before the theorem.

### 6.6 The continuity hypotheses include unnecessary conditions

The appendix’s global convention says the coercivity condition holds “throughout,” but continuity itself only needs positive \(c_1,c_2\), the parameter bounds, and the face condition. Separate hypotheses by lemma to avoid making the convergence theorem appear more restrictive than necessary.

### 6.7 The interpolation face estimates are too compressed

The new interpolation lemma says the jump terms have “identical powers of \(h\)” after replacing inverse estimates by \(L^\infty\) interpolation estimates. This is plausible, but it is the most delicate part of the upgrade from the discrete continuity bound to the exact interpolation-error bound.

**Required action:** display the two final face estimates explicitly and show the finite-overlap Cauchy–Schwarz step that produces \(\Psi_{\mathrm S}(h)\). This will eliminate the impression that the central improvement is being asserted “by inspection.”

### 6.8 Exact consistency requires precise local regularity

The statement \(\mathcal L_aU=F\) in \(L^2(K)\) requires sufficient elementwise \(H^2\) regularity of the velocity and \(H^1\) regularity of the pressure. State this where consistency is invoked, not only later in the interpolation section.

### 6.9 Mean-correct pressure interpolation

As noted above, use the shared mean-shift construction.

### 6.10 The low-porosity interpretation of \(\Psi_{\mathrm S}\) is too strong

The factor \(\alpha_K\le1\) makes the **formal functional** smaller for fixed interpolation errors. It does not show that the physical error improves as porosity decreases, because exact-solution norms and constants may grow with \(\alpha_0^{-1}\).

Replace “tighter in low-porosity regions, consistently with robustness” with a neutral statement such as:

```tex
The local porosity weight is retained rather than replaced by its global
worst case. Its net effect on the physical error also depends on the
porosity-dependence of the exact-solution regularity.
```

### 6.11 Notation

Rename the appendix’s \(\Psi(h)\) to \(\Psi_{\mathrm S}(h)\), use the subscripted ASGS norm throughout, and use the same interpolation/decomposition symbols as in OSGS.

---

## 7. OSGS appendix audit

### 7.1 Overall judgment

The OSGS proof is a serious and valuable addition. Its logic is coherent:

1. define weighted projections and fluctuations;
2. establish parameter conversions and smoothing;
3. assume a projected-component stability inequality;
4. use a special test function to recover components absent from the diagonal;
5. prove discrete inf–sup stability;
6. quantify the consistency defect;
7. prove interpolation continuity in \(\Psi_{\mathrm O}(h)\);
8. combine the estimates through inf–sup stability.

I found no fatal algebraic contradiction in the central special-test collection. The proof should be retained, but with the following corrections and qualifications.

### 7.2 Norm definiteness can be simplified and unified

Use the common explicit deviatoric-symmetric identity. In the current proof, the phrase \(\varepsilon\|q\|=0\) should be \(\varepsilon^{1/2}\|q\|=0\), or simply “the \(\varepsilon\|q\|^2\) term forces \(q=0\).”

### 7.3 The stability theorem understates dependencies

The theorem says \(\beta_{\mathrm{osgs}}\) and \(h_0\) depend only on a short list, but the smoothing function contains \(\alpha_0\), \(\nu\), coefficient moduli, size-surrogate regularity, and patch constants. The subsequent prose admits this.

**Required action:** make the theorem statement itself accurate. Distinguish dependence of the inf–sup constant from dependence of the admissible mesh threshold.

### 7.4 The projected stability assumption must remain prominent

The assumption is not a routine finite-element inverse inequality. It is the key nontrivial OSGS hypothesis. It should appear in the main theorem statement or immediately before it, not be buried in a long standing-assumption list.

A useful addition would be a sentence explaining whether it is known for the exact equal-order spaces and weighted products used here, or only assumed by analogy with the cited result.

### 7.5 Patch equivalence and smoothing need the repairs in Section 4

The current patch proof is not sufficient as written; the smoothing proof uses an unstated size-surrogate property and has a data-dependent threshold. These are not cosmetic because the special test must belong to the conforming finite-element space.

### 7.6 Best approximation requires stronger global product regularity

The lemma assumes \(z\in H^r(\Omega)\). Its application to \(z=\boldsymbol a\cdot\nabla u\) requires a global regularity condition on \(\boldsymbol a\), not only elementwise smoothness and continuity. Strengthen the assumption or prove a broken version.

### 7.7 Silent assumption \(c_1\ge1\)

Several estimates use \(c_1+\mathrm{Da}_{h,K}\ge1\) and \(c_1+\mathrm{Da}_{h,K}\le c_1(1+\mathrm{Da}_{h,K})\). State \(c_1\ge1\), which is harmless for all practical choices, or use \(\max\{1,c_1\}\).

### 7.8 The inf–sup quotient must exclude the zero test function

The body theorem does this; one appendix display takes the supremum over all \(V_h\in X_h\), which formally includes a zero denominator. Add `\setminus\{0\}`.

### 7.9 One consistency-proof explanation is inaccurate

After the mass consistency estimate, the prose says \(\varepsilon\tau_2\le C_2\) was applied again in the last step. The last conversion is actually the parameter inequality converting \(\tau_2^{-1/2}\) into the \((c_1+\mathrm{Da}_h)^{1/2}\alpha_Kh_K^{-1}\tau_1^{1/2}\) weight. Correct the explanation.

### 7.10 The main-text \(L^2\)-velocity formula omits a factor

The appendix corollary proves

\[
\|u-u_h\|
\le C\left[\sum_K
\left(1+\frac{c_1}{\mathrm{Da}_{h,K}}\right)
\left(
\left(1+\frac{c_2}{c_1}\mathrm{Re}_{h,K}\right)E_{u,K}^2
+\frac{\alpha_K}{\sigma\nu}E_{p,K}^2
\right)\right]^{1/2}.
\]

The corresponding body equation omits \(1+c_1/\mathrm{Da}_{h,K}\). This is a **definite mathematical inconsistency**.

**Required action:** insert the factor, or label the body expression explicitly as the asymptotic simplification valid only when \(\mathrm{Da}_{h,K}\gg c_1\).

### 7.11 The normalized OSGS excess factor is misstated

Relative to ASGS and after absorbing the fixed \(c_1\) into the generic constant, the extra varying factor is

\[
\sqrt{1+\mathrm{Da}_{h,K}/c_1},
\]

not \(\sqrt{1+\mathrm{Da}_{h,K}}\). Statements comparing the observed factor \(3.5\) with \(\sqrt{1+\mathrm{Da}_h}\approx3\) therefore do not follow from the theorem. For \(c_1=4\) and \(\mathrm{Da}_h\approx10\), the normalized factor is only \(\sqrt{3.5}\approx1.87\), and untracked constants dominate any quantitative comparison.

**Required action:** remove the numerical “explanation.” Retain only the qualitative statement that the theorem permits a reaction-dependent pre-asymptotic excess in energy/gradient control.

### 7.12 “Uniform in \(\sigma\)” needs qualifications

The algebra of the \(L^2\)-velocity corollary is correct, but a truly parameter-uniform statement also requires:

- \(\beta_0\) bounded away from zero;
- a mesh satisfying the data-dependent threshold;
- exact-solution Sobolev norms not growing with \(\sigma\);
- the reaction-dominated condition \(\mathrm{Da}_{h,K}\gtrsim c_1\) where the displayed simplification is used.

State “explicitly free of positive powers of \(\sigma\), conditional on the theorem constants and solution norms” rather than an unconditional robust estimate.

### 7.13 The pressure contribution does not necessarily decrease physically

Its **coefficient** decreases like \((\sigma\nu)^{-1/2}\), but \(\|p\|_{H^{k_p+1}}\) may depend on \(\sigma\). Use “the displayed coefficient decreases” rather than “the pressure contribution decreases.”

### 7.14 Remove or shorten the speculative Method-II extension

The Method-II remark is not proved and adds length to an already long appendix. Unless it is central to the paper, reduce it to one sentence in future work.

---

## 8. Symmetry and sharing between ASGS and OSGS

### 8.1 What works

The current body achieves useful symmetry:

- parallel theorem statements;
- explicit norm comparison;
- parallel error functionals;
- a comparison table;
- a clear statement that ASGS uses coercivity and OSGS inf–sup stability.

### 8.2 Where symmetry is currently artificial

The table’s row “Extra assumption” lists one item per method, but OSGS actually needs projection stability, patch structure, smoothing/size-surrogate control, and additional coefficient smoothness. ASGS needs the face-jump condition and weighted second-derivative inverse control. The two hypothesis sets are not naturally equal in size.

**Required action:** use a row labeled “Method-specific analytical ingredients” with multiple items, or split the table into stability and convergence assumptions.

### 8.3 Recommended appendix architecture

Rename and reorganize as follows:

#### Appendix A: Common analytical tools

- mesh/patch notation and overlap;
- pressure normalization and mean-correct interpolation;
- explicit deviatoric-symmetric norm identity;
- stabilization-parameter identities common to both methods;
- porosity comparability;
- common interpolation notation and error decomposition.

#### Appendix B: ASGS analysis

- ASGS coercivity;
- weighted inverse bound for the viscous residual;
- face-jump estimates;
- ASGS continuity and interpolation continuity;
- ASGS convergence.

#### Appendix C: OSGS analysis

- weighted projections and smoothing;
- projected stability assumption;
- special-test inf–sup proof;
- consistency defect;
- OSGS interpolation continuity and convergence;
- norm and \(L^2\)-velocity corollaries.

This would save space and make the logical dependency honest: OSGS already imports parameter lemmas from the appendix currently titled “ASGS convergence analysis.”

### 8.4 Notation to unify

Use the same notation in both proofs:

- \(X_h\) or \(\mathcal X_{h0}\), not both `\Xh` and `\Xhz`;
- \(\|\!|\!|\cdot\|\!|\!|_{\mathrm S}\) and \(\|\!|\!|\cdot\|\!|\!|_{\mathrm O}\) everywhere—no bare ASGS norm;
- \(\Psi_{\mathrm S}\) and \(\Psi_{\mathrm O}\), not appendix \(\Psi\);
- the same interpolation split, e.g. \(e=\eta+\phi_h\), for both methods;
- the same definition of \(\alpha_K\) globally (preferably \(\sup_K\alpha\));
- either use \(2\nu\) in both norms or explicitly state that the ASGS factor \(\nu\) is a harmless normalization chosen after coercivity.

---

## 9. Robustness analysis

### 9.1 The abstract’s robustness claim is too broad

The ASGS and OSGS bounds do not support an unconditional statement that “the error estimates are robust with respect to Reynolds and Damköhler numbers.” The estimates are component-, norm-, and regime-specific; OSGS includes a reaction-dependent energy factor and a projection-stability constant; the admissible \(h_0\) is data-dependent.

**Suggested wording:**

```tex
We derive parameter-explicit estimates and identify which components remain
uniform in the viscous, convective, and reactive regimes. The OSGS energy
estimate contains an additional elemental reaction factor, while a separate
\(L^2\)-velocity corollary is free of positive powers of the reaction
coefficient under the stated projection-stability and mesh-resolution
assumptions.
```

### 9.2 The OSGS comparison is not “verbatim”

The body says viscous and convection estimates hold verbatim for OSGS. Up to a fixed constant and conditional theorem constant this is asymptotically reasonable, but the OSGS norm has a different reaction term and the projection stability constant remains present.

Use “have the same \(h\)- and explicit parameter scaling up to fixed and projection-stability constants.”

### 9.3 The reaction-regime comparison says “exactly inflated” when it is not exact

The norms differ, the fixed \(c_1\) normalization matters, and component extraction is not identical. Replace “exactly” by “the right-hand error functional carries the additional factor.”

### 9.4 The body \(L^2\) equation must be corrected

Insert the missing \(1+c_1/\mathrm{Da}_{h,K}\) factor as noted above.

### 9.5 The macroscopic Damköhler discussion is confused

The statement that macroscopic \(\mathrm{Da}\) is “not expected to be relevant except perhaps in extremely reaction-dominated flows” is misleading: it is the chosen macroscopic reaction/diffusion ratio and affects constants even at fixed \(h\). What is true is that fixed-data \(h\to0\) convergence order is controlled by \(\mathrm{Da}_h=O(h^2)\), whereas simultaneous parameter limits can destroy uniformity.

### 9.6 The “mesh-independent mitigation factor” wording is awkward

The factor \(\mathrm{Da}^{-1/2}\) is mesh-independent because \(\mathrm{Da}\) is macroscopic; rewriting it using \(\mathrm{Da}_h\) introduces an \(h\) that cancels. Say this directly rather than saying it is independent “since \(\mathrm{Da}_h\propto h^2\).”

### 9.7 Porosity dependence is understated

The conclusion says constants degrade at worst as \(\alpha_0^{-1/2}\), except one \(\alpha_0^{-1}\) estimate. This ignores hidden dependence through:

- \(C_{\alpha,m}\), which contains \(\alpha_0^{-1}\);
- the OSGS projection constant \(\beta_0\);
- the data-dependent mesh threshold;
- exact-solution regularity.

Say “the explicit principal weights displayed in the component estimates” rather than “the constants.”

### 9.8 The viscous control is not weak for the chosen operator

Once the theorem is explicitly restricted to the deviatoric-symmetric operator and full Dirichlet data, its norm controls the full \(H^1\) seminorm by the identity above. Remove the statement that control depends vaguely on the operator, or retain it only for a separately stated generalized operator setting.

---

## 10. Numerical-method description and interpretation

### 10.1 Solver statements are too strong

“Converges to the same discrete solution” assumes uniqueness of the nonlinear algebraic root. Say “solves the same nonlinear algebraic formulation and, in the reported runs, converged to the same observed branch.”

Likewise, a direct LU factorization removes iterative **linear** solver tolerance; it does not eliminate nonlinear iteration error, quadrature error, conditioning, implementation error, or discretization error. Replace “genuine discretization errors” with “not contaminated by iterative linear-solver stopping tolerances.”

### 10.2 Definite contradiction: no fold diagnostics versus asserted fold

The two-dimensional section first correctly says that no continuation or Jacobian diagnostics were performed and that the computations cannot decide whether a root is absent. It later states that the discrete branch “folds,” that “the fold recedes,” and gives mesh thresholds where a root exists.

This is logically inconsistent. Newton/Picard failure on one mesh and success on a finer mesh do not establish a fold.

**Required replacement:**

```tex
The omitted low-porosity, high-Reynolds corner exhibits nonlinear-solver
nonconvergence on the coarser meshes tested. Converged solutions were obtained
from approximately \(N=512\) for the \(P_1\) sequence and from \(N=160\) for
the \(Q_2\) sequence. These observations locate the first successful runs of
the tested solvers; they do not establish a fold or the absence of a discrete
root on coarser meshes.
```

Also explain that the \(P_1\) runs with \(N=512,768\) are supplemental runs outside the previously declared \(10\)–\(320\) sequence.

### 10.3 “Once fine enough to resolve the solution” is an inference

Solver convergence does not prove resolution. Use “on the finer meshes where the tested solver converged.”

### 10.4 The porosity statement is false as written

“Any velocity compatible with \(\nabla\cdot(\alpha u)=0\) scales as \(1/\alpha\)” is false. Your manufactured family is constructed with that scaling; many compatible fields are not.

Replace with “the manufactured velocity is constructed with an \(\alpha_0/\alpha\) factor.”

### 10.5 Interpolation benchmark is not best approximation or a floor

The nodal interpolant is a useful benchmark, not generally the best approximation and not a lower bound. Statements that the degradation is “entirely best approximation,” that the stability constant contributes “no additional loss,” or that an error “sits on the floor” should be replaced by empirical language:

> Within the reported precision, no additional loss relative to the chosen nodal-interpolation benchmark is measurable in these cases.

### 10.6 Observed rates do not prove a theoretical loss is sharp

The ASGS \(Q_2\) pressure slope being one order below interpolation is compatible with the working-norm estimate. It does not prove that the theoretical loss “is not an artifact” or that it is necessary. The finite range could be pre-asymptotic, and the theorem is only an upper bound.

Use “the observed slopes coincide with the order allowed by the estimate.”

### 10.7 Attribution of OSGS pressure improvement is speculative

The explanation that orthogonal projection removes resolvable residual components is plausible and consistent with the method design, but the campaign does not isolate that mechanism. Label it a hypothesis or interpretation.

### 10.8 The reaction-factor comparison is quantitatively invalid

The discussion attributes a factor near \(3.5\) to \(\sqrt{1+\mathrm{Da}_h}\). As explained in Section 7.11, the normalized theorem factor is \(\sqrt{1+\mathrm{Da}_h/c_1}\), and all theorem constants are untracked. Remove the numerical match and the claim that the factor “makes clear” the mechanism. The observed excess indicator is descriptive but not a projection-orthogonal component.

### 10.9 “Accelerating relative rate” is overfitted

A sequence of a few slopes increasing from approximately 0.6 to 1.0 suggests a changing pre-asymptotic rate; it does not establish acceleration as a law. Use descriptive wording and report the raw values.

### 10.10 Three-dimensional pressure/penalty explanation is incorrect

The manuscript says that with full Dirichlet conditions pressure is determined only up to a constant, that boundary incompatibility makes the \(\varepsilon=0\) problem ill-posed, and that \(\varepsilon>0\) both restores well-posedness and fixes pressure through zero mean.

These are distinct issues:

- pressure is determined up to a constant for the incompressible problem, and a zero-mean constraint fixes it;
- an imposed discrete boundary trace with nonzero total porous flux can make the exact discrete mass constraint incompatible;
- \(\varepsilon>0\) relaxes/regularizes that constraint;
- the iterative penalty adds a correction so the target manufactured solution is recovered at convergence.

**Suggested replacement:**

```tex
On the irregular meshes, interpolation of the manufactured Dirichlet trace
does not preserve the zero total porous flux exactly. With \(\varepsilon=0\),
testing the discrete mass equation with the constant pressure mode would then
impose an incompatible exact flux constraint. We therefore use a small
pressure-penalty parameter \(\varepsilon>0\), together with the iterative
penalty correction, to relax this compatibility defect without changing the
target solution at convergence. The pressure normalization is imposed
separately by the zero-mean convention.
```

### 10.11 Three-dimensional coercivity claims need narrowing

The local generalized-eigenvalue and global coercivity diagnostics are valuable. However:

- “\(P_1\) viscous residual vanishes identically” is true for the unweighted second derivative on an affine element, not for \(\nabla\cdot(\alpha\ViscProj\nabla u_h)\) when \(\nabla\alpha\ne0\);
- the experiments show that the tested \(P_2\) tetrahedral families needed a larger constant, not that all quadratic tetrahedra universally require it;
- “restored optimal rates” should exclude the OSGS pressure-\(H^1\) stagnation and should be limited to the quantities actually recovered.

### 10.12 “Rates established” is too strong for short sequences

The irregular \(P_2\) sequence has only three meshes, and many slopes use only the two finest levels. Say “the last two levels exhibit the reported local slopes,” not that convergence rates are established and further meshes would only reconfirm them.

### 10.13 The Taylor–Hood interpretation needs updating

The added stabilized Taylor–Hood control is useful. The current text still overstates it in two directions:

- It says the control “confirms” that the gain comes from stabilization. It supports that attribution for the tested case; it does not prove it generally.
- It says the stabilization is a redundant perturbation because the stabilized Taylor–Hood error is larger in the viscous case. That is a plausible interpretation, not established causally.

Also replace “Taylor–Hood velocity fails to converge” with “its error stagnates over the tested mesh range.”

### 10.14 Table captions call nominal rates “theoretical”

For the nonlinear methods and the reported \(L^2\) norms, the parenthesized values are best described as nominal interpolation orders. The theorem does not provide every \(L^2\) rate in the tables, and it does not cover the nonlinear implementation.

### 10.15 Reproducibility remains incomplete

Add supplementary machine-readable data containing, for every run:

- mesh size convention, element count, and degrees of freedom;
- raw errors before normalization;
- interpolation benchmark errors;
- exact nonlinear residual norm and stopping tolerance;
- iteration counts and failed-run status;
- quadrature order;
- stabilization constants and whether parameters are pointwise or elemental;
- projection type;
- solver/preconditioner settings;
- actual ranges of \(\mathrm{Re}_{h,K}\) and \(\mathrm{Da}_{h,K}\).

This is especially important because many claims use two-mesh slopes and selected FME ratios.

---

## 11. Abstract, introduction, and conclusion

### 11.1 Abstract

The abstract should say “linearized ASGS and OSGS formulations,” not singular “linearized method.” Replace “preventing instabilities” by “designed to control instabilities.” Replace unconditional robustness language by the parameter-explicit qualification proposed above.

### 11.2 Introduction

The introduction now motivates both methods, but the first statement of contributions should list ASGS and OSGS symmetrically from the outset. The section label `sec:StabilityASGS` should be renamed to a neutral `sec:StabilityAnalysis` because it now contains both analyses.

The statement that the continuous nonlinear model is unique for sufficiently large viscosity/minimum porosity should either be supported by a precise theorem/citation or reduced to “we assume a sufficiently regular solution exists.”

### 11.3 Conclusion

The conclusion should be revised in the following ways:

- “that theorem” should be plural or “those theorems” when referring to both methods;
- “robust under extreme variations” should become “displayed stable convergence over the tested parameter grid, except for reported solver failures”;
- the explicit \(\alpha_0\) statement must be limited to displayed principal weights and acknowledge hidden constants;
- “errors sit essentially on the nodal interpolant” must be limited to the identified manufactured cases and norms;
- “residual growth is inherited rather than introduced” is causal and should become “tracks the growth of the interpolation benchmark”;
- the OSGS reaction discrepancy is qualitatively compatible with the new bound, so say the theory does not **quantitatively explain or establish the necessity** of the observed factor;
- the tetrahedral claim must be restricted to the tested mesh families and error components;
- the Taylor–Hood conclusion should say the stabilized equal-order velocity was comparable in the tested cases, while pressure could be substantially less accurate in viscosity-dominated cases;
- remove the expectation that interpolated porosity is benign, since it was not tested and the positivity/weighted-norm issue may be nontrivial.

---

## 12. Recommended replacement paragraphs

### 12.1 Scope paragraph after the theorem comparison table

```tex
The ASGS and OSGS results have parallel statements but different analytical
hypotheses. The ASGS proof uses coercivity, a weighted inverse estimate for
the viscous residual, and a face-jump condition when the reaction is present.
The OSGS proof instead uses weighted projections, a projected-component
stability assumption, and a data-dependent smoothing argument. Both theorems
concern the stated linearized, elementwise-parameter formulations. The
nonlinear computations use pointwise parameters and ordinary \(L^2\)
projections and are therefore interpreted as experiments for related methods,
not as direct verification of the theorems.
```

### 12.2 OSGS robustness paragraph

```tex
After the fixed factor \(c_1^{1/2}\) is absorbed into the generic constant,
the OSGS error functional differs from the ASGS functional by the varying
factor \((1+\mathrm{Da}_{h,K}/c_1)^{1/2}\). Thus the two functionals have the
same fixed-data asymptotic order because \(\mathrm{Da}_{h,K}=O(h_K^2)\), while
the OSGS energy estimate permits a reaction-dependent pre-asymptotic excess.
This is a one-sided allowance rather than a quantitative prediction of the
OSGS/ASGS error ratio.
```

### 12.3 Correct body \(L^2\)-velocity estimate

```tex
\[
\|\boldsymbol e_u\|
\lesssim
\left[
\sum_K
\left(1+\frac{c_1}{\mathrm{Da}_{h,K}}\right)
\left(
\left(1+\tfrac{c_2}{c_1}\mathrm{Re}_{h,K}\right)
\mathrm E_{\mathrm{int},K}(\boldsymbol u)^2
+
\frac{\alpha_K}{\sigma\nu}
\mathrm E_{\mathrm{int},K}(p)^2
\right)
\right]^{1/2}.
\]
```

### 12.4 Numerical solver-failure paragraph

```tex
At \((\mathrm{Re},\alpha_0)=(10^6,0.05)\), the nonlinear solvers tested do
not converge on the coarser meshes, even from the exact manufactured field.
Converged runs were obtained only on finer meshes (from approximately
\(N=512\) for the reported \(P_1\) continuation of the sequence and from
\(N=160\) for \(Q_2\)). These thresholds identify solver success for the
algorithms and initializations tested; without pseudo-arclength continuation
or Jacobian diagnostics they do not establish a fold or absence of a discrete
root.
```

### 12.5 Three-dimensional penalty paragraph

Use the replacement in Section 10.10.

---

## 13. Changes requiring no numerical rerun

The following are text/proof/source corrections only:

- correct the appendix input and bibliography names;
- reconcile projection space/product definitions;
- separate stationary theorem from lagged nonlinear iteration;
- fix theorem hypothesis lists and pressure normalization;
- share the pressure mean correction and norm-definiteness lemma;
- repair ASGS absolute values and the \(T_{13}\) bookkeeping;
- repair OSGS patch equivalence, size-surrogate assumption, coefficient regularity, and theorem dependencies;
- add the missing \(1+c_1/\mathrm{Da}_h\) factor;
- correct the normalized OSGS factor;
- remove fold/nonexistence wording and other causal overinterpretations;
- rewrite the 3D penalty explanation;
- revise abstract, introduction, robustness discussion, and conclusion;
- flatten revision markup and compile a clean package.

---

## 14. Changes that may justify limited reruns

No full numerical campaign is required merely to submit the two theorems honestly. Limited diagnostics would materially strengthen particular claims:

1. **Projection-stability constant:** compute the generalized eigenvalue defining the OSGS projected-component stability constant over representative meshes and parameter regimes. This would support the claimed parameter robustness of the OSGS theorem.
2. **Theory/implementation bridge:** compare weighted versus ordinary projection and elemental versus pointwise \(\tau\) in one viscous, one convective, and one reactive case.
3. **Fold claim:** only pseudo-arclength continuation and Jacobian singular-value diagnostics can justify fold language. Otherwise delete it.
4. **Three-dimensional coercivity:** retain the existing local/global coercivity diagnostic, but report its exact definition and values in supplementary material. Recompute it with the weighted viscous residual if the text claims the porosity-weighted operator is what vanishes for \(P_1\).
5. **Rate claims on short sequences:** one or two additional meshes for irregular \(P_2\) and the OSGS pressure-\(H^1\) anomaly would make the discussion more convincing, but narrowing the wording is an acceptable alternative.

---

## 15. Prioritized implementation sequence

### Priority 1: definite correctness blockers

1. Include the correct latest ASGS appendix and normalize filenames.
2. Fix the OSGS projection definition and lagged/stationary scope contradiction.
3. Correct ASGS continuity absolute values and the \(T_{13}\) grouping.
4. Insert the missing \(1+c_1/\mathrm{Da}_h\) factor in the body.
5. remove all fold assertions unsupported by continuation.
6. rewrite the 3D pressure/penalty paragraph.

### Priority 2: theorem rigor and transparency

7. Give exact hypotheses for each theorem.
8. separate common, ASGS-specific, and OSGS-specific penalty assumptions.
9. correct patch equivalence and add the size-surrogate assumption.
10. strengthen advection regularity or prove a broken approximation lemma.
11. expose \(\beta_0\) and data-dependent \(h_0\) in all robustness claims.
12. unify pressure normalization and interpolation mean correction.

### Priority 3: interpretation and presentation

13. normalize the OSGS reaction factor and remove quantitative error-ratio predictions.
14. temper interpolation, best-approximation, solver, and Taylor–Hood claims.
15. reorganize common analytical material and unify notation.
16. revise abstract/introduction/conclusion after the theorem scope is frozen.
17. provide raw numerical data and diagnostics as supplementary files.
18. flatten annotations, compile in the target class, and perform a final page/layout audit.

---

## Final recommendation

The paper should continue as a **single symmetric ASGS–OSGS article**. The two methods now have sufficiently substantial and complementary theory to justify joint treatment. Keep the complete theorem statements and a concise mechanism comparison in the main body, and retain all detailed stability/convergence proofs in refereed appendices. Reorganize the shared lemmas into a neutral common appendix, but do not obscure the stronger OSGS assumptions in the name of symmetry.

After the definite corrections above, the mathematical core appears publishable. The largest residual risk is not the ASGS/OSGS convergence architecture; it is overclaiming parameter uniformity and numerical mechanisms beyond what the conditional constants and finite computations establish.
