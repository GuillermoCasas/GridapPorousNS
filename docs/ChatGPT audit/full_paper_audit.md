# Full audit of the variable-porosity VMS paper

**Files audited**

- `article(3).tex` — main manuscript, 1,687 lines
- `continuity_appendix.tex` — detailed continuity and convergence appendix, 1,019 lines
- `fourier_appendix.tex` — Fourier-symbol calculations, 121 lines
- `elemental_matrices_appendix.tex` — elemental matrices, 377 lines
- `references(4).bib` — bibliography database, 59 entries

Line references in this report refer to this uploaded snapshot. I did **not** attempt to re-prove every estimate or check every elemental matrix entry; I audited the logical scope of the results, the theorem statements and hypotheses, the consistency of the analysis, the interpretation of the numerical evidence, and the manuscript's clarity and rigor.

## 1. Overall assessment

The paper contains a potentially valuable contribution: a residual-based VMS formulation for a generalized variable-porosity stationary flow model, a Fourier-motivated design of stabilization parameters, a detailed ASGS stability/continuity analysis for a linearized problem, and a broad numerical study of ASGS and OSGS variants.

In its present state, however, it is **not yet submission-ready**. The main reasons are not the elemental algebra. They are:

1. The principal convergence statement and the subsequent robustness discussion are not aligned mathematically. The appendix derives an elementwise weighted \(\ell^2\) interpolation quantity and then weakens it to an \(\ell^1\) sum; the manuscript nevertheless calls the latter “sharpest” and later treats the local interpolation errors as if a global norm estimate had been proved.
2. Several numerical claims are stronger than the reported evidence. In particular, failure of Newton/Picard iterations is repeatedly interpreted as nonexistence of a discrete solution and as a fold of the solution branch.
3. The introduction, abstract, and conclusions describe a substantially broader validation than the analysis actually proves. The theorem is for the **linearized ASGS method**, under homogeneous all-Dirichlet conditions, constant viscosity and scalar reaction, prescribed weighted-divergence-free advection, mesh-resolved positive porosity, elementwise-constant stabilization parameters, and additional mesh/jump assumptions. The numerical study solves nonlinear ASGS and OSGS problems, uses pointwise stabilization parameters, and includes a nonlinear DBF reaction.
4. There are direct contradictions in the numerical setup and conclusions, including the statement that all tests use \(U=L=1\) despite a later parameter encoding in which \(U\) changes, and a conclusion about interpolation of \(\alpha\) although every test evaluates \(\alpha\) analytically.
5. The source is still in an internal revision state: hundreds of colored amendment commands and author-note commands remain, and the uploaded package is not build-complete.

My recommendation is **major revision**, with the central theorem/robustness issue addressed before polishing the numerical narrative. The paper can become much clearer and more rigorous if it separates, every time, among:

- what is **proved** for the linear ASGS model;
- what is **motivated heuristically** by Fourier analysis;
- what is **observed empirically** for nonlinear ASGS/OSGS implementations; and
- what is only a **hypothesis suggested by solver behavior**.

## 2. Logical map of the paper

The manuscript currently follows this chain:

1. Introduce a generalized stationary porous Navier–Stokes model with porosity-weighted convection, viscous stress, pressure gradient, reaction/resistance, and a pressure penalty.
2. Recast the model as a generic convection–diffusion–reaction system.
3. Derive a VMS formulation and define ASGS and OSGS variants.
4. Use a Fourier-symbol scaling argument to motivate \(\tau_1\) and \(\tau_2\).
5. Prove coercivity/stability and continuity for a **linearized ASGS** bilinear form.
6. Derive an a priori interpolation-based error estimate.
7. specialize the estimate to viscous-, convection-, and reaction-dominated regimes.
8. Test nonlinear ASGS and OSGS schemes in 2D, an extruded 3D setting, and a DBF manufactured comparison with Taylor–Hood.
9. Conclude that the original Navier–Stokes convergence behavior is essentially preserved and that the method is robust over extreme parameter ranges.

The weak point is the transition from steps 6–7 to steps 8–9. The numerical experiments are useful, but they do not directly validate the theorem because they differ from it in method, nonlinearity, reaction law, stabilization evaluation, and in the discrete advection constraint. The paper partly acknowledges this at `article:1080–1091`; that disclaimer is good and should become the organizing principle of the abstract, introduction, numerical discussion, and conclusions.

## 3. What is actually proved: hypotheses that must be stated prominently

The main theorem at `article:965–975` is not self-contained. It refers to the stability and continuity lemmas, which in turn refer to conditions scattered between the main text and `continuity_appendix.tex`. A reader should not have to reconstruct the following list.

### 3.1 Continuous/linearized problem

The proved analysis concerns an Oseen-type, linearized problem with:

- a polyhedral domain in \(d=2\) or \(3\);
- homogeneous Dirichlet velocity data on the whole boundary;
- constant kinematic viscosity \(\nu>0\);
- a constant scalar isotropic reaction \(\sigma\ge 0\), not the nonlinear DBF resistance;
- prescribed advection \(\boldsymbol a\), rather than the solved nonlinear velocity;
- \(\nabla\!\cdot(\alpha\boldsymbol a)=0\) in the distributional sense;
- a pressure penalty \(\varepsilon\ge0\) satisfying a mesh- and parameter-dependent upper bound;
- sufficient regularity for strong element residuals and interpolation estimates.

### 3.2 Porosity

The appendix assumes:

- \(\alpha\in W^{1,\infty}(\Omega)\);
- \(0<\alpha\le1\), hence a strictly positive lower bound on the compact closure if continuity is intended;
- the mesh-resolution condition
  \[
  h_K\|\nabla\alpha\|_{L^\infty(K)}\le C_{\nabla\alpha}\,\alpha_{0,K};
  \]
- consequent local comparability of the infimum and supremum of \(\alpha\) on each element.

This is much stronger and more precise than the introduction's wording “smooth and bounded gradients that are sufficiently resolved.” It should be given a named standing assumption in the main text.

### 3.3 Mesh and spaces

The analysis uses:

- a shape-regular, locally quasi-uniform mesh family;
- neighboring element-size comparability and uniformly bounded face counts;
- conforming continuous finite element spaces;
- pressure functions with an elementwise gradient, so the actual pressure space used by the stabilized form is an \(H^1\)-conforming subspace, not merely an unspecified subspace of \(L^2\);
- polynomial inverse estimates, with a modified inverse constant that also absorbs the porosity-gradient condition;
- homogeneous discrete boundary conditions;
- zero-mean pressure when \(\varepsilon=0\) and all boundary conditions are Dirichlet.

### 3.4 Stabilization parameters

The theorem uses:

- elementwise-constant \(\tau_{1,K}\) and \(\tau_{2,K}\);
- \(\alpha_K=\alpha_{\infty,K}\);
- an elementwise maximum of \(|\boldsymbol a|\);
- \(c_1>2\xi C_{\mathrm{inv}}^2\) for some \(\xi>2\);
- the reduced \(\tau_2\) without the \(\varepsilon h^2\) term, justified only under the stated smallness condition on \(\varepsilon\);
- when \(\sigma>0\), a cross-face comparability/jump condition for the parameter \(\varphi_{1,K}\).

The simulations instead evaluate stabilization parameters pointwise. This is a meaningful implementation variant, not a detail that should be claimed harmless without either an argument or an ablation.

### 3.5 Method and result

The proof covers **ASGS only**, with the orthogonal projection set to zero. It does not establish stability or convergence for OSGS. It also does not cover the nonlinear reaction used in the DBF example.

The theorem should state the precise continuous variational problem, the precise discrete ASGS problem, existence/uniqueness of the discrete solution as a consequence of coercivity, pressure normalization, and the dependence/independence of the constant \(C\).

## 4. Highest-priority mathematical issues

### M1. The convergence theorem is stated with an \(\ell^1\) element sum, while the appendix first derives a sharper \(\ell^2\) quantity

**Locations:** `article:965–975`; `continuity_appendix:958–976`; `continuity_appendix:989–1017`.

The main theorem states

\[
|||E_h|||\le C\sum_K\frac{\alpha_K}{h_K}
\left(\tau_{2,K}^{1/2}E_{u,K}+\tau_{1,K}^{1/2}E_{p,K}\right).
\]

The appendix first obtains

\[
|||E|||\le C\left[\sum_K\alpha_K^2h_K^{-2}
\bigl(\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2\bigr)\right]^{1/2},
\]

and then uses \(\ell^2\subset\ell^1\) to weaken it. It later calls the weakened bound “the sharpest afforded by the present argument.” That statement is internally false: the displayed \(\ell^2\) quantity is strictly sharper than its \(\ell^1\) upper bound.

More importantly, the robustness section silently replaces local \(E_{\mathrm{int},K}\) terms and their element sum by global quantities \(E_{\mathrm{int}}(u)\) and \(E_{\mathrm{int}}(p)\). That passage does not follow from the stated \(\ell^1\) theorem without additional mesh-cardinality factors. On a quasi-uniform mesh, an \(\ell^1\) sum of local interpolation norms can lose a factor comparable to the square root of the number of elements, i.e. \(h^{-d/2}\), relative to the natural global \(\ell^2\) norm. This can alter the claimed order and therefore affects the paper's central “optimal/robust” narrative.

**Required correction:**

1. Define
   \[
   \mathcal E_h^2 := \sum_K\alpha_K^2h_K^{-2}
   \left(\tau_{2,K}E_{u,K}^2+\tau_{1,K}E_{p,K}^2\right).
   \]
2. Revisit the interpolation-continuity proof, especially all face terms, and use Cauchy–Schwarz and bounded face multiplicity to retain \(\mathcal E_h\) rather than switching to an \(\ell^1\) quantity.
3. If this succeeds, state the theorem with \(\mathcal E_h\) and describe the \(\ell^1\) version only as a coarser corollary.
4. Re-derive every equation in the robustness section from the corrected global weighted estimate.
5. If a genuinely unavoidable face term forces an \(\ell^1\) estimate, remove “sharpest,” “optimal,” and the present global robustness formulas unless the resulting mesh-cardinality dependence is carried explicitly.

This is the single most important revision.

### M2. The robustness section is not rigorous with the theorem as stated

**Locations:** `article:996–1077`.

The section uses \(E_{\mathrm{int}}(u)\) and \(E_{\mathrm{int}}(p)\) as undefined global scalar errors, although the theorem defines only local \(E_{\mathrm{int},K}\). It also moves from elementwise values \(\alpha_K\), \(\tau_{i,K}\), and local regimes to global worst-case factors without stating the inequalities used.

**Required correction:** after resolving M1, introduce global weighted interpolation functionals explicitly. For example,

\[
\mathcal E_u:=\left(\sum_K\alpha_K^2h_K^{-2}\tau_{2,K}E_{u,K}^2\right)^{1/2},\qquad
\mathcal E_p:=\left(\sum_K\alpha_K^2h_K^{-2}\tau_{1,K}E_{p,K}^2\right)^{1/2}.
\]

Then derive regime-specific upper bounds for these functionals. Do not jump from local asymptotics to a global formula without stating whether the regime is assumed on every element or only on a subregion.

### M3. Equation `eq:DominantPressureGradientXTermEstimate` contains an algebraic/dimensional error

**Location:** `article:1043–1053`, especially line 1052.

From Eq. 1043, if the pressure-gradient term dominates, multiplication by \(\|\boldsymbol a\|_\infty\) gives a coefficient

\[
\frac{\|\boldsymbol a\|_\infty U}{P}+1,
\]

when the bound is expressed using \(E_{\mathrm{int}}(p)=P E^*_{\mathrm{int}}(p)\). The manuscript instead writes \(\|\boldsymbol a\|_\infty/\sqrt P+1\), which does not follow from Eq. 1043 and is dimensionally suspect unless an unmentioned nondimensionalization is imposed.

**Replace with:**

\[
\|\nabla e_p\|_h\lesssim \alpha_0^{-1/2}
\left(\frac{\|\boldsymbol a\|_\infty U}{P}+1\right)
\frac{E_{\mathrm{int}}(p)}{h},
\]

followed by the conditional scaling \(P\sim U^2\) and \(\|\boldsymbol a\|_\infty\sim U\).

### M4. The limiting-language in the robustness section mixes mesh refinement with parameter families

**Locations:** `article:996–1004`, `1030`, `1054`, `1071–1077`.

For fixed physical data, \(\mathrm{Re}_h\propto h\) and \(\mathrm{Da}_h\propto h^2\), so both tend to zero as \(h\to0\). Statements such as “as \(h\to0\)” followed by “\(\mathrm{Re}_h\to\infty\)” or “\(\mathrm{Da}_h\to\infty\)” cannot describe a fixed-problem asymptotic refinement limit. They describe parameter-dependent/pre-asymptotic regimes or families in which physical coefficients vary with \(h\).

**Required correction:** call these “element-level dominance regimes” and state the quantifiers. For example: “Assume that on the elements of interest \(\mathrm{Re}_{h,K}\gg1\) while the other ratios remain bounded.” Reserve “as \(h\to0\)” for fixed-data convergence.

The sentence at line 1077 saying the mitigation factor is independent of mesh size “since \(\mathrm{Da}_h\propto h^2\)” also needs rewriting: the conclusion depends on substituting the relation between a fixed macroscopic number and its element number; it is not compatible with simultaneously treating \(\mathrm{Da}_h\to\infty\) under fixed data.

### M5. The theorem and lemmas are not self-contained enough

**Locations:** `article:938–970`.

Problems include:

- the triple norm is used in Lemma 1 before it is defined;
- the continuity inequality lacks an absolute value around \(B_S\);
- the same omission appears in the appendix continuity statements;
- “there exists” is mistyped as “there exists{s}”;
- the jump condition is referenced indirectly as “analogous to Eq. (39)” rather than stated;
- the theorem defines the exact problem by reference to an iteration equation for the nonlinear algorithm;
- constant dependencies are not stated;
- the precise finite element degrees and pressure regularity are not integrated cleanly into the statement;
- consistency and discrete existence/uniqueness are not stated.

**Required correction:** create one “Standing assumptions for Section X” block, define the norm before use, state the jump condition in the paper, and replace the convergence theorem with a self-contained version. A proposed statement is supplied in Section 12 below.

### M6. The main proof assumption is not satisfied “trivially” by a previous finite element velocity

**Location:** `continuity_appendix:173–182`.

Continuity and elementwise \(W^{1,\infty}\) regularity are automatic for a conforming finite element velocity. The condition \(\nabla\cdot(\alpha\boldsymbol a)=0\) is not. The numerical section itself concedes that discrete iterates do not satisfy it (`article:1086–1088`).

**Required correction:** replace “which satisfies this trivially” by a precise separation:

> A conforming finite element iterate satisfies the continuity and elementwise regularity requirements. The weighted-divergence-free condition is an additional analytical assumption and is generally not satisfied exactly by the discrete nonlinear iterates.

### M7. The claim that the error deteriorates “at worst like \(\alpha_0^{-1/2}\)” is contradicted by the paper's own pressure estimate

**Locations:** `article:1075–1077`; conclusion `article:1655`.

The pressure-gradient estimate in the reaction-dominated regime carries \(1/\alpha_0\), and the manuscript explicitly notes that it is the unique exception. The conclusion nevertheless says the constants degrade at worst like \(\alpha_0^{-1/2}\).

**Required correction:** state the general \(\alpha_0^{-1/2}\) dependence and the explicit \(1/\alpha_0\) pressure-gradient exception every time the result is summarized, including the abstract if porosity robustness remains a headline claim.

### M8. The “norm” status of the triple quantity should be justified

**Location:** `article:944–948`.

Depending on \(\ViscProj\), \(\varepsilon\), \(\sigma\), the boundary conditions, and pressure normalization, it is not completely immediate that zero triple quantity implies \(U_h=0\). If this is intended as a norm rather than a seminorm, add a short proposition or remark identifying the kernel and explaining why homogeneous Dirichlet data plus pressure normalization eliminate it. If only a seminorm is needed, call it a mesh-dependent energy seminorm.

## 5. Continuous model and weak formulation

### C1. Correct the description of the deviatoric projection

**Location:** `article:256–268`.

The manuscript defines \(\DPi\) as the projector that **extracts the deviatoric part**, but later says that \(\DPi\SPi\) removes the deviatoric part from the viscous term. The displayed formula shows the opposite: it retains the deviatoric symmetric gradient and removes the spherical/trace part.

**Replace the sentence at line 268 by:**

> For \(\ViscProj=\DPi\SPi\), the viscous stress retains the deviatoric symmetric part of the velocity gradient and removes its spherical part. Establishing coercivity for this operator requires an appropriate deviatoric Korn inequality and compatible boundary conditions.

### C2. State the porosity assumptions consistently

**Locations:** `article:249`, `467`, and the appendix standing assumptions.

“Differentiable with uniformly bounded gradient” is less precise than the later \(W^{1,\infty}\) assumption. Use one statement throughout:

\[
\alpha\in W^{1,\infty}(\Omega),\qquad 0<\alpha_0\le \alpha(x)\le1\quad\text{a.e.}
\]

Then distinguish the continuous positivity condition from the additional mesh-resolution condition used only in the discrete analysis.

### C3. Do not imply general well-posedness of the nonlinear model

**Locations:** `article:254`, `264–268`.

The text says the penalty “helps to ensure well-posedness” and then proceeds by assuming that the solution “always exists” and uniqueness holds for sufficiently large \(\nu\) and \(\inf\alpha\). This is too vague for a rigorous paper and risks suggesting a theorem that is not supplied.

**Required wording:**

- identify the continuous nonlinear well-posedness as outside the scope;
- say exactly which cited special case has a theorem and under what small-data assumptions;
- state that the subsequent VMS derivation is formal for the general nonlinear model;
- state that the rigorous analysis is for the specified linearized problem.

A suitable replacement is:

> We do not establish existence or uniqueness for the full nonlinear model (1)–(2). The derivation below is therefore formal at this level. The rigorous stability and error analysis in Section X concerns the linearized problem under the hypotheses stated there. For the DBF specialization with mixed boundary conditions, Cocquet et al. prove well-posedness under smallness assumptions on the data.

### C4. The trace operator is typed incorrectly

**Locations:** `article:296–304`, `357–378`, `461–465`.

A generic trace of an \(H^1\) velocity belongs naturally to an \(H^{1/2}\)-type boundary space, not simply \(L^2(\partial\Omega)^d\). Traction belongs to a negative-order dual space. The single abstract operator \(\mathcal D:\mathcal X\to L^2(\partial\Omega)^d\) conflates Dirichlet trace and Neumann traction.

**Required correction:** define two operators:

- \(\gamma_D:\mathcal V\to H^{1/2}(\Gamma_D)^d\), the velocity trace;
- \(\mathcal T_N(U)\in H^{-1/2}(\Gamma_N)^d\), the weak traction when defined.

Then write the weak boundary pairing as \(\langle \gamma_N v,t_N\rangle_{H^{1/2}_{00},H^{-1/2}}\). Do not write \(\langle V,\mathcal DU\rangle_{\Gamma_N}\) unless the trace of the velocity component of \(V\) is explicitly meant.

### C5. The notation for inner products, integrals, and duality is overloaded

**Location:** `article:457–479`.

The manuscript uses \((\cdot,\cdot)\) for an \(L^2\) inner product and \(\langle\cdot,\cdot\rangle\) both for an ordinary product integral and for duality. This makes the weak form harder to verify.

**Recommended convention:**

- \((f,g)_\omega\) for \(L^2\) products;
- \(\langle F,v\rangle_{X',X}\) only for duality;
- explicit boundary duality brackets for tractions.

### C6. \(\alpha\in W^{1,\infty}\) alone does not make every nonlinear weak-form term bounded

**Location:** `article:467–469`.

Boundedness also requires regularity/growth assumptions on the advecting field, reaction tensor, and solution spaces. Replace the blanket claim with a list of the assumptions needed for the particular form. In the linear analysis, state them directly; in the general nonlinear derivation, say the forms are defined for sufficiently regular arguments.

### C7. Clarify nonhomogeneous boundary data

**Location:** `article:378`.

“All developments apply equally” is too broad. A lifting can treat nonhomogeneous Dirichlet data at the weak-form level, but the stability identity uses homogeneous test/trial functions and \(\nabla\cdot(\alpha\boldsymbol a)=0\); a lifting introduces additional source terms and compatibility issues.

**Replace by:**

> Nonhomogeneous Dirichlet data can be treated by a lifting at the formulation level. The stability and convergence analysis below is stated only for homogeneous all-Dirichlet data; extending it requires tracking the lifting terms and weighted-divergence compatibility.

### C8. Clarify pressure normalization and penalty

**Locations:** `article:459`, `1431–1439`.

The zero-mean condition fixes the pressure nullspace for an incompressible all-Dirichlet problem. A small \(\varepsilon\) is a different device. In the 3D section the manuscript combines the pressure constant, boundary-data compatibility, and well-posedness into one statement.

Separate these points:

1. For \(\varepsilon=0\), impose a zero-mean pressure to remove the constant nullspace.
2. A nonhomogeneous discrete velocity trace must satisfy the discrete flux compatibility needed by an exactly divergence-free mixed problem; if it does not, use a compatible boundary projection/lifting or a penalty formulation.
3. The iterative penalty should be written as an explicit fixed-point scheme and its limiting equation stated.

### C9. Use \(I_d\), not \(\boldsymbol 1_3\)

**Location:** `article:833`.

The analysis covers \(d=2,3\), so write \(\boldsymbol\sigma=\sigma I_d\).

### C10. Distinguish a reaction number from the conventional porous-media Darcy number

**Locations:** `article:986–994` and throughout the robustness section.

The quantity \(\sigma L^2/(\alpha_\infty\nu)\) is a diffusion-to-reaction ratio inverse, and in a Darcy interpretation it is closely related to an inverse Darcy number. Calling it a Damköhler number is defensible in a generic reaction–diffusion framework, but porous-media readers may expect “Da” to mean permeability divided by \(L^2\). Add one sentence defining the convention and warning that it is a **reaction/Damköhler-type number**, not the conventional Darcy number.

## 6. VMS derivation and definition of the discrete schemes

### V1. Separate bubble-subscale assumptions from boundary representation

**Locations:** `article:535–540`.

Assumption A.1 says \(\mathcal D_K\widetilde U=0\) and concludes both that the subscales vanish on element boundaries and that finite element functions resolve physical boundary conditions exactly. These are different assumptions.

Split into:

- an element-bubble/localization assumption for subscales;
- a separate assumption about how essential boundary data are represented in the resolved space.

A.2 then becomes partly redundant with A.1. Define precisely whether values, normal derivatives, or flux contributions of the subscales are neglected on interelement faces.

### V2. Avoid claiming that the exact eliminated fine-scale term “must provide” stability

**Location:** `article:510–518`.

Exact equivalence to a well-posed continuous problem does not by itself mean that the isolated second term admits the desired computable coercivity estimate in the chosen discrete norm. Replace “must provide the desired stability” with “encodes the effect of unresolved scales whose approximation is used to recover discrete stability.”

The sentence “all VMS-stabilized methods are characterized by the way in which these two operators are approximated” is also too sweeping. Qualify it as a description of the class considered here.

### V3. Define ASGS more precisely

**Location:** `article:570–574`.

“The SGS space is the space of finite element residuals, and thus the projection is the identity” is imprecise. In ASGS, the unresolved residual is algebraically approximated by the full strong residual elementwise. State this directly rather than attributing it to a rigorously defined residual space.

### V4. The formal OSGS scheme and the implemented OSGS scheme use different projection spaces

**Locations:** `article:575–610`.

The equations define the projection onto the constrained space \(\mathcal X_{h0}\), while the footnote says the implementation projects onto unconstrained velocity and pressure finite element spaces. That is a different method, especially at the boundary.

**Required correction:** move the implemented projection space into the main definition of the scheme. Introduce a symbol such as \(\mathcal X_h^{\mathrm{proj}}\) and define whether it contains unconstrained boundary degrees of freedom. Then use that symbol consistently in the projection equation and implementation description. If the analysis does not cover this OSGS choice, say so.

### V5. The projection inner product is changed without clearly defining the final method

**Locations:** `article:579–587`.

The text first defines a \(\tau\)-weighted projection, then replaces it by the ordinary \(L^2\) projection. The final equations should state which projection is actually used. Treat the weighted version as motivation and the \(L^2\) projection as the scheme definition, or analyze both separately.

### V6. Clarify whether the advecting velocity is the resolved or total velocity

**Locations:** `article:514`, `556–559`, `600`, `619–634`.

The text says both \(\tau_K\) and the residual depend on the SGS because the total velocity is \(U_h+\widetilde U\), but the later algorithm often evaluates them from the previous resolved iterate. Define, for every displayed problem, whether convection and stabilization use \(u_h\), \(u_h+\widetilde u\), or a lagged field. The notation \(\mathcal L_{\boldsymbol u}\) is currently too easy to misread.

### V7. The constant-reaction trimming makes the implemented OSGS algorithm differ from the displayed algorithm during iteration

**Location:** `article:638`.

At a converged state, a constant reaction applied to an FE function lies in the projection space and cancels in an exact \(L^2\) orthogonal residual. During the lagged iteration, however, the manuscript explicitly alters that term to improve nonlinear convergence. This is an algorithmic modification, not merely an implementation detail.

**Required correction:** provide the actual residual projected in each experiment and distinguish:

- the converged discrete OSGS equations;
- the nonlinear iteration used to reach them;
- the conditions under which omitting the reaction projection leaves the converged equation unchanged.

### V8. The numerical solver does not match the displayed Picard algorithm closely enough for reproducibility

**Locations:** `article:621–634`, `1093`.

The manuscript derives a staggered Picard iteration but reports using a globalized Newton/Newton–Krylov variant that includes projection derivatives. The assertion that this converges to “the same discrete solution” is plausible but not demonstrated by the text.

Report:

- the exact nonlinear residual vector;
- which dependence of \(\tau\) is differentiated;
- how the projection is eliminated/differentiated;
- globalization/line search or trust-region rule;
- linear and nonlinear tolerances;
- preconditioner;
- stopping norm for momentum, mass, and projection equations;
- a direct comparison on at least one representative problem if equivalence is a claim.

### V9. Do not say all VMS variants preserve the exact stability properties

**Location:** `article:518`.

The text should say that the approximations are *designed* to recover suitable stability, which must then be established for each formulation. Only ASGS receives an analysis in this paper.

## 7. Fourier-motivated stabilization design

The Fourier section is useful as a design heuristic, but several sentences present it as stronger than it is.

### F1. Label the section explicitly as a calibration/motivation, not a proof

**Locations:** `article:692–829`; `fourier_appendix.tex`.

Use wording such as “symbol-based scaling argument” or “Fourier calibration.” Stability is established later and only for the simplified ASGS parameters.

### F2. Explain the transition from a one-sided estimate to a calibration equality

The argument bounds the inverse symbol in one direction and then selects parameters by matching scales. State that equality is an engineering calibration criterion, not a necessary or sufficient stability condition.

### F3. Restrict the generalized spectral-radius definition

If \(\rho_{\Lambda^{-1}}(A)\) is defined through a generalized eigenproblem, state that the matrices involved are symmetric/Hermitian positive semidefinite in the applications where a real maximum is used. For a general nonsymmetric matrix, “maximum eigenvalue” is not well-defined as a real order statistic.

### F4. Resolve the type and embedding of \(\boldsymbol\sigma\)

**Location:** around `article:796–812`.

The reaction tensor acts on the velocity block, while the Fourier system has \(d+1\) variables. Define the block embedding and show that for \(\boldsymbol\sigma=\sigma I_d\), the expression reduces to the scalar \(\sigma\) used in the theorem.

### F5. Remove “minimum still ensuring” language

The text introduces safety factors such as 5 and later acknowledges that the assembled \(\tau\) does not enforce the original symbol condition except in limiting regimes. Present the final formula as a robust interpolation of dominant-term scalings, not the minimal parameter satisfying a proved inequality.

### F6. Clarify the status of \(\lambda\) and discarded terms

The scaling \(\lambda\) is introduced self-referentially and then an \(O(1)\) factor and a term are discarded. Spell out the asymptotic assumptions under which this is done. Avoid “therefore” when the transition is a modeling choice.

### F7. The mean-value-theorem/wave-number interpretation is heuristic

Do not say the mean value theorem “predicts” the characteristic unresolved wave number. Say it motivates the expected \(h^{-1}\) scaling.

### F8. Supply the symbolic-verification scripts or soften the claim

The Fourier appendix says calculations were symbolically verified, but no scripts are included. Archive the script with the reproducibility package, or replace the statement by a less audit-dependent description.

### F9. Distinguish the three-dimensional matrices from the general-\(d\) claims

The main matrix display is for \(d=3\). State how the 2D case is obtained and whether the appendix calculations were carried out independently in both dimensions.

## 8. Numerical study: global audit of soundness and reproducibility

The numerical campaign is broad and contains useful observations, but the discussion often moves too quickly from a computed pattern to a causal explanation or a mathematical conclusion. The following changes would markedly improve rigor.

### N1. Solver nonconvergence does not establish nonexistence of a discrete solution or a fold

**Locations:** `article:1142`, `1154`, `1646`, `1653`; relevant table captions.

The evidence described is failure/stalling of a nonlinear solver, even when initialized from sampled exact data. This establishes only that the tested algorithm and initialization did not reach the requested residual tolerance. It does not distinguish among:

- absence of a discrete root;
- a root with a small basin of attraction;
- an ill-conditioned Jacobian;
- a distinct nearby branch;
- failure of the globalization or preconditioner;
- an implementation or quadrature issue.

A fold/turning point is an assertion about the solution branch and requires branch tracking, not merely Newton failure.

**Text-only fallback:** replace every such statement with:

> On the coarser meshes, the nonlinear solvers and initializations tested here did not converge to the prescribed tolerance. The present computations do not distinguish nonexistence of a discrete root from poor conditioning or loss of the solver's basin of attraction.

**Evidence needed to retain “fold” or “no solution”:** pseudo-arclength continuation (or an equivalent branch-tracking method), a branch diagram, residuals, smallest Jacobian singular values/eigenvalues near the turning point, and tests from multiple initial guesses/directions.

### N2. The setup contradicts itself about the characteristic velocity

**Locations:** `article:1118` and `1131–1137`.

Line 1118 says all experiments use \(U=L=1\). The centered encoding later sets

\[
U=\frac{\mathrm{Re}}{\sqrt{\alpha_\infty\mathrm{Da}}},
\]

which varies by cell. Both statements cannot be true.

**Required correction:** say that \(L=1\) is fixed, while \(U,\nu,\sigma\) are selected by the centered dimensional encoding for the constant-reaction sweeps. Explain separately how the DBF cases are scaled, because the nonlinear resistance does not fit the same constant-\(\sigma\) parameterization.

### N3. Report raw errors, mesh measures, and more reliable rate estimates

**Locations:** all numerical tables.

Almost every reported slope uses only the two finest meshes. This is especially unstable when the mesh-size ratio is small, as the paper itself notes for 3D \(\mathbb P_2\). A two-point slope is not enough to establish an asymptotic rate, and “superoptimal” slopes at a solver threshold are plainly pre-asymptotic.

**Required reporting:**

- raw error at every mesh level;
- the precise \(h\) used in each rate calculation;
- degrees of freedom;
- local rates between consecutive meshes;
- a least-squares slope over at least the finest three monotone points when available;
- a clear flag where fewer than three converged levels exist.

The summary tables may remain in the main paper, but the raw tables should be in an appendix or data repository.

### N4. Define the mesh size unambiguously

The text alternates among nodal distance, target size, element size, and structured-grid count \(N\). State whether \(h_K\) is element diameter, maximum edge length, inradius-based size, or nodal spacing, and what global \(h\) is used for rates and element Reynolds/Damköhler numbers.

### N5. Report quadrature and manufactured forcing generation

The paper states that Gridap evaluates forcing and errors, but not:

- quadrature degree for each element order and nonlinear term;
- whether derivatives are generated symbolically, by automatic differentiation, or manually;
- quadrature degree for the error norms;
- whether the pointwise \(\tau\) is recomputed at every quadrature point and nonlinear iteration.

Without these details, high-order and nonlinear convergence claims are not fully reproducible.

### N6. A momentum-residual tolerance alone is insufficient

**Location:** `article:1093`.

Report a norm of the complete stabilized residual, including mass conservation and, for OSGS, the projection equation. Also report relative/absolute tolerances, maximum iterations, and whether errors are insensitive to further tightening for representative difficult cells—not only higher-order elements.

### N7. Quantify the porosity-resolution hypothesis

The analysis assumes \(h_K\|\nabla\alpha\|_\infty/\alpha_{0,K}\) is uniformly controlled. Compute and report

\[
R_\alpha:=\max_K\frac{h_K\|\nabla\alpha\|_{L^\infty(K)}}{\alpha_{0,K}}
\]

for every mesh family, especially the coarser low-porosity cases. This is directly relevant to the interpretation that coarse-mesh failure is a resolution phenomenon.

### N8. Verify the actual \(\varepsilon\) stability margin

For the 3D runs, report

\[
\max_K \varepsilon\tau_{2,K}
\quad\text{and}\quad
\frac{\varepsilon}{c_1\min_K(\alpha_K^2\tau_{1,K}/h_K^2)}
\]

on every mesh, rather than only an inequality containing unexplained constants. This connects the experiment to the theorem much more directly.

### N9. Pointwise and elementwise-constant stabilization parameters are different schemes

**Locations:** `article:828`, `1086`.

The statement that pointwise evaluation “was observed not to alter the results” needs evidence. At minimum, add a small ablation for representative viscous-, convective-, and reaction-dominated cases comparing:

- pointwise \(\tau\);
- \(\tau\) based on element maxima, as in the theorem.

If no rerun is made, say only that pointwise evaluation was used and lies outside the proof.

### N10. The nodal interpolant is a benchmark, not an “absolute reference” or error floor

**Locations:** table captions and numerical discussion, especially `article:1152–1158`, `1164`.

The nodal interpolant is neither the best approximation nor a lower bound for the discrete error. The manuscript itself reports finite element errors below the nodal interpolation error. Therefore terms such as “floor,” “absolute reference,” and “sits on the floor” are mathematically misleading.

**Use instead:** “nodal interpolation benchmark,” “same asymptotic scale as the nodal interpolant,” or “within X% of the nodal interpolation error.” If a best-approximation comparison is intended, compute an \(L^2\) or energy projection.

### N11. Do not claim a universal factor between nodal and best approximation without a derivation

Any statement that the \(\mathbb P_1\) nodal \(L^2\) error exceeds the best approximation by a fixed factor such as \(\sqrt6\) must be derived for this exact function, norm, and mesh family, or removed. Such a factor is not generic.

### N12. “Any velocity compatible with \(\nabla\cdot(\alpha u)=0\) scales as \(1/\alpha\)” is false

Compatibility constrains the weighted flux \(\alpha u\), but does not uniquely fix its amplitude or spatial dependence. The chosen manufactured field has \(u=(\alpha_0/\alpha)\,u_0\). Attribute low-porosity interpolation growth to this particular construction and its gradients, not to every compatible velocity.

### N13. One-sided upper bounds cannot be “corroborated” merely because errors lie below them

**Location:** numerical discussions and conclusion.

The disclaimer at `article:1089–1091` is correct. Apply it consistently. Replace “confirms,” “verifies,” or “corroborates the estimate” by “is consistent with the estimate” unless the data discriminate the predicted scaling from plausible sharper alternatives.

### N14. Avoid causal claims not isolated by the experiment

Several explanations are plausible but not established:

- that the OSGS reaction-dominated velocity gap is governed by \(\mathrm{Da}_h\);
- that a projection mechanism causes a particular pressure optimum;
- that pressure improvement with reaction is caused by the derived mitigation factor;
- that irregular-mesh rate loss is solely a property of the mesh sequence;
- that low-porosity pressure growth is carried by a specific coupling term.

Phrase these as hypotheses unless supported by controlled ablations.

### N15. The “OSGS excess component” has no orthogonality basis

**Location:** `article:1158`.

The quantity \(\sqrt{e_{\mathrm{OSGS}}^2-e_{\mathrm{ASGS}}^2}\) is called an error component, but the ASGS and OSGS error vectors are not shown to be orthogonal. It is only a scalar quadratic difference and may be undefined when OSGS is smaller.

**Use instead:** ratios, signed differences, or call it a “quadratic excess indicator” with an explicit warning that it is not a vector component.

### N16. Separate rate and magnitude when comparing ASGS and OSGS

In several tables OSGS pressure has a higher observed rate but a larger absolute error. State these two facts separately. “Better pressure convergence” is ambiguous and should not be used without saying whether it means rate, constant, or finest-mesh error.

### N17. The 3D pressure behavior is a limitation, not a minor curiosity

**Locations:** `article:1441` and `tab:3DH1`.

The OSGS pressure \(H^1\) error near 1.29 on several cases and the low rates are striking. Elevate this to a clear limitation in the conclusions and, ideally, diagnose it through:

- raw error curves;
- pressure normalization checks;
- projection-space/boundary treatment ablation;
- quadrature/tolerance checks;
- an alternative manufactured pressure with genuine \(z\)-dependence.

### N18. Direct sparse LU does not eliminate nonlinear or quadrature error

**Location:** `article:1427`.

It eliminates iterative linear-solver tolerance in each linear solve. It does not by itself prove that the final error is purely discretization error. Report nonlinear stopping criteria and quadrature accuracy.

### N19. Do not call the extruded manufactured problem “genuinely three-dimensional”

**Location:** `article:1427`.

The exact fields are independent of \(z\) and have zero exact \(z\)-velocity. An irregular tetrahedral mesh may generate a nonzero \(z\)-error, but this tests three-dimensional assembly and mesh anisotropy, not a genuinely three-dimensional solution.

**Text-only correction:** call it an “extruded three-dimensional discretization test.”

**Stronger rerun:** add a manufactured field with nonzero \(u_z\) and all components depending on \(x,y,z\), while satisfying \(\nabla\cdot(\alpha u)=0\).

### N20. The 3D stabilization-constant claim is not adequately documented

**Locations:** `article:1429`, `1659`.

The text says \(c_1=16k^4\) lies below the elementwise sufficient threshold but is admissible because global conformity makes the effective condition milder. That is plausible but no global theorem, eigenvalue table, or reproducible calculation is supplied. The conclusion then says the theory predicts and experiments confirm the choice.

To retain this claim, provide:

- the local generalized eigenproblem and reference-element values;
- the global discrete coercivity diagnostic and normalization;
- values versus mesh for \(c_1=4k^4\), \(16k^4\), and a value above the local sufficient threshold;
- associated error curves.

Otherwise, describe \(16k^4\) as an empirically successful 3D choice and remove the claim that it is validated by the proved condition.

### N21. The 3D incompatibility/ill-posedness statement is too compressed

**Location:** `article:1431`.

A pressure constant nullspace is fixed by a zero-mean constraint; it does not make the problem intrinsically ill-posed. Boundary trace incompatibility with exact discrete mass conservation is separate. Explain the discrete compatibility condition and why the chosen trace fails it. Consider a compatible boundary lifting/projection as a cleaner control.

### N22. The inequality defining \(\varepsilon_{\mathrm{ref}}\) needs a derivation

**Location:** `article:1433–1437`.

The constants \(c_2/c_1\) and \(100/c_1\) appear without derivation. State how they follow from the selected parameter ranges, definition of \(h\), and bounds on \(\alpha_K\), or move the derivation to an appendix.

### N23. “The reported sequences already establish the rates” is contradicted by the data

**Location:** `article:1427`.

Some irregular-mesh and pressure rates are substantially below nominal values, and the \(\mathbb P_2\) regular slope uses a small mesh-ratio. Say that the available data show monotone reduction and approximate trends, with unresolved asymptotic behavior in the noted cases.

## 9. Audit of the 2D manufactured-results discussion

The reported arithmetic checks that can be verified from the text are mostly coherent: excluding the \((\mathrm{Re},\alpha_0)=(10^6,0.05)\) corner leaves 15 of 18 combinations; \(\mathrm{Re}_h\approx10^6/320\approx3.1\times10^3\); and the global \(\mathrm{Da}_h\approx10^6/320^2\approx9.8\). The concerns are primarily interpretive.

### 9.1 Convection-dominated quadratic pressure

The data support the observation that the quadratic pressure error can remain substantially above the nodal interpolation benchmark while retaining a high observed rate. They do **not** prove that a particular theorem term is the cause. Use “consistent with the pressure-control structure of the estimate” rather than “the estimate explains.”

### 9.2 “Superoptimal” rates at the excluded corner

Rates near 3 for a \(\mathbb P_1\) velocity based on \(N=512\) and 768 are pre-asymptotic two-point slopes. Do not call them a convergence rate of the method. Report the two raw errors and describe the decrease without assigning theoretical significance.

### 9.3 Reaction-dominated pressure improvement

The pressure errors decrease as the reaction number rises in the reported cells. That is a sound observation. The claim that the mechanism is the \(\mathrm{Da}^{-1/2}\) factor is not isolated, because changing \(\mathrm{Da}\) also changes dimensional coefficients and potentially solver conditioning. Phrase it as consistency with the bound, not proof of the mechanism.

### 9.4 OSGS reaction-dominated velocity gap

This is one of the paper's most interesting empirical anomalies. It deserves a compact dedicated subsection with plots of raw errors versus \(h\) and \(\mathrm{Da}_h\), rather than a very long paragraph of post hoc explanation. The manuscript is right to say the coercivity argument does not resolve it. Stop there unless an ablation identifies the cause.

### 9.5 Table captions should not call OSGS rates “theoretical”

No OSGS convergence theorem is proved. Parenthetical rates may be called “nominal interpolation orders” or “ASGS benchmark orders,” not theoretical OSGS rates.

## 10. Audit of the 3D discussion

### 10.1 Scope

Present this as a test of the implementation on tetrahedral meshes and of sensitivity to mesh regularity. It is not an independent test of three-dimensional physical variation.

### 10.2 Mesh interpretation

If the nodal interpolation rate is also depressed on the irregular sequence, it is evidence that the chosen mesh sequence and \(h\)-definition contribute. It does not prove that the method has no additional mesh sensitivity. Compare computed error divided by interpolation error at each level, not just their fitted slopes.

### 10.3 Pressure anomaly

The repeated OSGS \(H^1\) pressure errors around 1.29 should be treated as a red flag. The current footnote explaining that the repeated value is not a typo is useful but not enough. It needs a diagnostic or a clear limitation statement.

### 10.4 Stabilization constant

Because \(c_1=16k^4\) is below the stated elementwise sufficient threshold, the present theorem does not certify it. Either use a theorem-certified value and rerun, or separate “sufficient theoretical bound” from “empirical global value.”

## 11. Audit of the DBF/Taylor–Hood comparison

### D1. The dimensional resistance formula is unclear

**Locations:** `article:1545–1550`.

Equations define \(a(\alpha)\) and \(b(\alpha)\) using dimensionless constants, while the prose says the dimensional coefficients carry a factor of \(\nu\). The actual formula implemented in the dimensional solver is not displayed with units.

**Required correction:** write the exact dimensional formula used by the code, including factors of \(\nu\), \(L\), and \(U\), and verify that each term in \(\sigma(\alpha,u)u\) has acceleration units. Then state the representative definition of \(\mathrm{Da}\) for a velocity-dependent \(\sigma\).

### D2. The nonlinear reaction makes the Damköhler number spatial- and solution-dependent

A single quoted \(\mathrm{Da}\) evaluated at \(\alpha_0\) and \(|u|=U\) is a representative maximum/scale, not a global constant coefficient. Say so explicitly.

### D3. This is inspired by Cocquet et al., not a reproduction of their benchmark

The paper uses different manufactured fields and all-Dirichlet data. Describe it as a comparison using a DBF specialization related to the cited work. Do not imply direct replication of the reference's benchmark or numerical constants.

### D4. “Matches Taylor–Hood accuracy” is false for pressure

**Locations:** `article:1644`, `1648`, `1659` and tables.

For \(\mathrm{Re}=1\), quadratic equal-order velocity errors are indeed comparable to Taylor–Hood. Pressure errors are often orders of magnitude larger. For example at \(\alpha_0=0.5\), the \(L^2\) pressure FME is about \(1.8\times10^{-4}\) for stabilized \(\mathbb P_2/\mathbb P_2\) and \(4.17\times10^{-6}\) for Taylor–Hood. At \(\alpha_0=0.1\), the gap is larger.

**Required wording:**

> In the viscosity-dominated cases, the stabilized quadratic equal-order method gives velocity errors comparable to Taylor–Hood, while Taylor–Hood is substantially more accurate for pressure in this manufactured test.

### D5. The high-Reynolds comparison conflates inf–sup stability and convection stabilization

Taylor–Hood is inf–sup stable but the comparator is an **unstabilized Galerkin** method. The equal-order method includes pressure and convection/reaction stabilization. Therefore the experiment shows the value of residual stabilization at high convection; it does not show that equal-order spaces are superior to Taylor–Hood.

To isolate the space-pair question, add a convectively stabilized Taylor–Hood control using the same residual stabilization with only the pressure-related terms adjusted as appropriate.

### D6. Report degrees of freedom and cost

\(\mathbb P_2/\mathbb P_2\) and \(\mathbb P_2/\mathbb P_1\) do not have the same pressure degree or unknown count. A claim of avoiding “complexity” needs DOFs, assembly/solve time, memory, and solver details. Otherwise remove the complexity rhetoric.

### D7. Do not explain Taylor–Hood pressure accuracy by a “dedicated pressure space” without analysis

A lower-order pressure space is not inherently a dedicated accuracy mechanism. Simply report the observation. The persistence of accurate pressure while velocity stagnates may be specific to the manufactured forcing and should not be generalized.

### D8. The high-Reynolds low-porosity slopes are pre-asymptotic

The stabilized iteration converges on only the finest levels. Do not describe slopes around 3 for \(\mathbb P_1\) as optimal/superoptimal evidence. They are threshold-crossing two-point slopes.

### D9. Verify “standard Forchheimer value” precisely

Identify the exact convention and equation in the cited source for \(C_b=1.75\). Forchheimer/Ergun constants depend on nondimensionalization; “standard” is too vague.

## 12. Introduction, abstract, and conclusion consistency

### 12.1 Claim-by-claim consistency matrix

| Current claim | Where | Audit result | Required revision |
|---|---|---|---|
| Error estimates are robust with respect to Reynolds and Damköhler numbers | Abstract, line 204 | Too strong. Bounds retain explicit \(\mathrm{Re}_h\), \(\mathrm{Da}\), and porosity factors, and the global derivation needs repair | Use “parameter-explicit estimates” and describe the regimes/limitations |
| Essentially the same stability and convergence properties are preserved for smooth resolved porosity | Introduction, line 237 | Omits linearization, ASGS-only, constant scalar reaction, all-Dirichlet, weighted-solenoidal advection, and elementwise-constant \(\tau\) | State the full scope in one sentence |
| Numerical battery backs up the analytical conclusion | Introduction, line 237 | Experiments are outside the theorem in several respects | Say the tests explore behavior beyond the proved setting |
| The cited Navier–Stokes–Brinkman work needed no revision of the original theory | Introduction, line 237 | Overbroad and unnecessary; depends on detailed formulation differences | Remove or narrow to a factual comparison of model terms |
| LPS is the only previously proposed stabilized FEM for variable porosity | Introduction, line 241 | Priority claim is difficult to substantiate and not needed | Remove “the only”; describe it as a relevant prior method |
| Inf–sup pairs cause increased data-structure complexity | Introduction, line 241 | Rhetorical and unsupported | Remove or quantify with DOFs/cost |
| The convergence analysis validates the stabilization parameters | Introduction, line 243 | It validates a simplified ASGS, elementwise-constant form under narrow assumptions | State that it supports the asymptotic scaling in that setting |
| The general method remains as robust in extreme parameter variation as the original method | Conclusion, line 1653 | Too broad; one theorem setting and several numerical anomalies | Replace by a scoped proved result plus empirical observations |
| The discrete problem has no solution on coarse meshes | Conclusion, line 1653 | Unsupported by solver failure | Use solver-nonconvergence wording or provide continuation evidence |
| Porosity constants degrade at worst as \(\alpha_0^{-1/2}\) | Conclusion, line 1655 | Contradicted by the reaction pressure-gradient estimate \(1/\alpha_0\) | State the exception |
| Errors sit on the nodal interpolation error, a stronger fact than optimality | Conclusion, line 1655 | Nodal interpolation is not a lower bound; not uniform across all tables | Say many velocity errors are of comparable magnitude to the nodal benchmark |
| 3D theory predicts and experiments confirm \(c_1=16k^4\) | Conclusion, line 1659 | The chosen value is below the stated local sufficient threshold; supporting data are not shown | Supply diagnostics/reruns or call it empirical |
| Equal-order method matches Taylor–Hood accuracy | Conclusion, line 1659 | False for pressure in viscosity-dominated DBF cases | Restrict to velocity accuracy and state the pressure gap |
| Interpolation of \(\alpha\) does not spoil convergence | Conclusion, line 1665 | No experiment interpolates \(\alpha\); line 1118 and 1552 say it is evaluated analytically | Delete or run an interpolation study |

### 12.2 Recommended abstract structure

The abstract should contain four clearly separated statements:

1. **Formulation:** residual-based VMS derivation and ASGS/OSGS variants.
2. **Parameter design:** Fourier-symbol argument motivates porosity-dependent stabilization parameters.
3. **Proved result:** linearized ASGS theorem with its main restrictions.
4. **Empirical study:** nonlinear ASGS/OSGS calculations beyond the theorem, including both successes and observed pressure/3D limitations.

A proposed abstract is given in Section 15.

### 12.3 Recommended introduction contribution list

Replace the current broad narrative by an explicit list such as:

1. We derive ASGS and OSGS residual-based formulations for a stationary variable-porosity generalized flow model.
2. We give a Fourier-symbol scaling argument for the stabilization parameters.
3. For the ASGS discretization of a linearized problem with constant viscosity and scalar reaction, homogeneous Dirichlet data, prescribed weighted-solenoidal advection, mesh-resolved positive porosity, and elementwise-constant parameters, we prove coercivity and an a priori estimate in a porosity-weighted stabilization norm.
4. We numerically investigate nonlinear ASGS and OSGS implementations, including pointwise parameters and a nonlinear DBF resistance; these tests explore behavior beyond the proved setting.

This immediately prevents scope drift.

### 12.4 Recommended conclusion structure

Use four short paragraphs:

1. Exactly what was formulated and proved.
2. What the numerical experiments observed, with rate and magnitude distinguished.
3. Limitations/anomalies: no OSGS theorem, nonlinear and pointwise-\(\tau\) cases outside theory, 3D OSGS pressure behavior, solver nonconvergence not nonexistence, pressure porosity exception.
4. Specific future work: OSGS/nonlinear analysis, steep/discontinuous porosity, discrete porosity, continuation near difficult parameter corners, and genuinely 3D tests.

## 13. Literature positioning

The introduction should be more factual and less priority-driven.

- Codina's 2001 paper itself formulates ASGS for generalized stationary incompressible flows and analyzes the **linearized** equations; this reinforces the need to describe the present theorem as a linear ASGS extension rather than a validation of all nonlinear variants.
- Cocquet et al. analyze a variable-porosity DBF model with mixed boundary conditions, prove uniqueness under small-data assumptions, and prove convergence/optimal estimates for a Taylor–Hood approximation with interpolated porosity. The present paper should describe this accurately when contrasting scope.
- Skrzypacz's LPS work treats a linearized Brinkman–Forchheimer–Darcy equation with nonconstant porosity, equal-order interpolation with enrichment, and numerical support for optimal error bounds. This makes the “only previous stabilized method” wording both unnecessary and risky.
- Nillama et al. present a stabilized equal-order Navier–Stokes–Brinkman method with spatially varying stabilization and mesh-resolution studies. It is safer to state the concrete similarities/differences than to claim the older theory applies “without revision.”

The bibliography should cite these works by their precise contribution, not use them as rhetorical foils.

## 14. Source-package, bibliography, and style audit

### 14.1 Build completeness

The uploaded package does not compile as supplied because it lacks:

- `siamart190516.cls`;
- `shared.tex`;
- `figures/bump_plateau.pdf`;
- a file named `references.bib` (the supplied file is `references(4).bib`).

Therefore I could not perform a final page-layout, float-placement, overfull-box, or rendered-equation audit. Restore/rename these dependencies and compile from a clean directory before submission.

### 14.2 Internal structural checks

After excluding commented-out material, the source has:

- 358 active labels, all unique;
- no active unresolved internal references found by static scan;
- 38 cited bibliography keys, all present in the supplied `.bib` file;
- 59 unique bibliography entries;
- 21 unused entries.

This is good structural hygiene, but the final clean build is still necessary.

### 14.3 Revision markup must be resolved

The package contains approximately:

- 374 active `\amend{...}` commands;
- 13 active red `\Guillermo{...}` commands;
- 3 active blue `\Joaquin{...}` commands.

These are not merely comments; they affect the rendered manuscript. Accept or reject every revision and remove the author-note macros before submission.

### 14.4 Stale comments and control character

- Remove obsolete commented-out derivations and old review notes once the revision is stable.
- `continuity_appendix.tex` contains an escape/control character in an early comment. Remove it to avoid tooling problems.
- Avoid leaving comments that contain stale labels/citations, because future automated checks become noisy.

### 14.5 Bibliography cleanup

The `.bib` file contains template/demo entries unrelated to the paper and many metadata inconsistencies. Recommended actions:

- delete unused style/manual/tutorial entries;
- replace Semantic Scholar API URLs with DOI fields or authoritative journal metadata;
- store DOI values as bare DOI strings rather than full URLs;
- standardize journal capitalization and en-dashes in page ranges;
- verify author spellings and titles against Crossref/publisher records;
- fix the question mark in the `smith2004investigation` booktitle;
- add DOIs to the principal numerical-analysis references where available;
- verify entries such as `quarteroni2009numerical`, `grillo2014darcy`, and conference-paper types/fields;
- ensure titles preserve acronyms such as VMS, ASGS, OSGS, DBF, Navier–Stokes, and Darcy–Brinkman–Forchheimer.

### 14.6 Style and readability

The paper is often hardest to read in the numerical discussion, where paragraphs extend for hundreds of words and combine data, interpretation, caveats, and mechanisms.

**Recommended edits:**

- split long numerical paragraphs into “Observation,” “Interpretation,” and “Limitation” units;
- use small auxiliary tables for ratios to interpolation benchmarks instead of embedding many numbers in prose;
- remove repeated intensifiers such as “clearly,” “precisely,” “in fact,” and “it is interesting”; 
- replace “battery of tests” by “numerical study”;
- replace “less versed reader” by “readers less familiar with VMS”;
- replace “porous matrix's volume fraction” by “solid volume fraction \(1-\alpha\)”;
- remove layout hacks such as `\hfill\\` after section headings;
- standardize \(k_u\) versus \(k_v\), \(I_d\), \(\alpha_K\), and the notation for macro/element reaction numbers;
- avoid using the same symbol \(U\) for the combined unknown and a velocity scale without repeated reminders; preferably rename the scale \(U_0\) or \(u_{\mathrm{ref}}\);
- use “stationary” consistently rather than switching among stationary, steady, and time-independent without purpose;
- shorten footnotes that carry essential method definitions by moving them into the main text.

### 14.7 Package hygiene

Review whether both `mathtools` and duplicate `amsmath` loading are needed through `shared.tex`/class. The `mathabx` package can alter standard symbols and conflict with other math packages; retain it only if a specific symbol requires it.

## 15. Proposed corrected theorem statement

The following is a model statement; equation labels and exact function-space notation should be adapted to the final proof.

> **Theorem (ASGS error estimate for the linearized problem).** Let \(\Omega\subset\mathbb R^d\), \(d\in\{2,3\}\), be a bounded polyhedral domain, and let \(\{\mathcal T_h\}\) be a shape-regular, locally quasi-uniform mesh family. Assume \(\alpha\in W^{1,\infty}(\Omega)\), \(0<\alpha_0\le\alpha\le1\), and
> \[
> h_K\|\nabla\alpha\|_{L^\infty(K)}\le C_\alpha\alpha_{0,K}
> \quad\forall K\in\mathcal T_h.
> \]
> Let \(\nu>0\) and \(\sigma\ge0\) be constants, and let \(\boldsymbol a\in C(\overline\Omega)^d\) with \(\boldsymbol a|_K\in W^{1,\infty}(K)^d\) and \(\nabla\cdot(\alpha\boldsymbol a)=0\) in distributions. Impose homogeneous Dirichlet velocity data on \(\partial\Omega\), and use zero-mean pressure when \(\varepsilon=0\). Let \(\mathcal V_{h0}\subset H_0^1(\Omega)^d\) and \(\mathcal Q_{h0}\subset H^1(\Omega)\cap L_0^2(\Omega)\) [modified when \(\varepsilon>0\)] be conforming polynomial spaces of degrees \(k_u,k_p\ge1\). Define elementwise-constant \(\tau_{1,K},\tau_{2,K}\) by Eqs. (...) with \(\alpha_K=\alpha_{\infty,K}\) and \(|\boldsymbol a|_{\infty,K}\). Assume the pressure-penalty condition (...), the inverse-constant condition \(c_1>2\xi\overline C_{\mathrm{inv}}^2\) for some \(\xi>2\), and, if \(\sigma>0\), the stated face-jump comparability condition.
>
> Let \((\boldsymbol u,p)\) solve the linearized variational problem and let \((\boldsymbol u_h,p_h)\) be the consistent ASGS solution. Assume \(\boldsymbol u\in H^{k_u+1}(\Omega)^d\), \(p\in H^{k_p+1}(\Omega)\), and sufficient elementwise regularity for the strong residual. Then the discrete problem has a unique solution and
> \[
> |||U-U_h|||\le C
> \left[
> \sum_{K\in\mathcal T_h}\alpha_K^2h_K^{-2}
> \left(
> \tau_{2,K} E_{u,K}^2+
> \tau_{1,K} E_{p,K}^2
> \right)
> \right]^{1/2},
> \]
> where \(E_{u,K}=h_K^{k_u+1}|\boldsymbol u|_{H^{k_u+1}(K)}\) and \(E_{p,K}=h_K^{k_p+1}|p|_{H^{k_p+1}(K)}\). The constant \(C\) is independent of \(h\), \(\nu\), \(\sigma\), and \(\varepsilon\), but may depend on the mesh regularity, polynomial degrees, \(c_1,c_2\), the porosity-resolution/comparability constants, and the face-jump constants.

**Important:** use this \(\ell^2\) statement only after confirming that every interpolation-continuity and face term retains the same global \(\ell^2\) structure. If not, state the actual weaker estimate and adjust all robustness claims.

## 16. Proposed replacement abstract

> We derive residual-based variational multiscale formulations for a stationary generalized Navier–Stokes model with spatially varying porosity and consider algebraic and orthogonal subgrid-scale variants. A Fourier-symbol scaling argument motivates porosity-dependent stabilization parameters for diffusion-, convection-, and reaction-dominated regimes. For the ASGS discretization of a linearized problem with prescribed weighted-divergence-free advection, constant viscosity and scalar reaction, homogeneous Dirichlet data, mesh-resolved positive porosity, and elementwise-constant stabilization parameters, we prove coercivity and an a priori estimate in a porosity-weighted stabilization norm. We then investigate nonlinear ASGS and OSGS implementations numerically in two-dimensional manufactured tests, extruded three-dimensional tetrahedral tests, and a manufactured Darcy–Brinkman–Forchheimer comparison. These computations explore settings beyond the proved result, including pointwise stabilization parameters and nonlinear resistance. They show stable velocity convergence over broad parameter ranges while also revealing pressure sensitivity at low porosity and in the three-dimensional OSGS tests. The resulting estimates are parameter-explicit; their dependence on the minimum porosity, including the stronger reaction-limit pressure-gradient factor, is stated explicitly.

This abstract should be updated after the theorem is finalized. If the \(\ell^2\) estimate cannot be retained, remove “a priori estimate” claims of optimal global order unless the actual result supports them.

## 17. Proposed replacement contribution paragraph for the introduction

> The contributions of this work are fourfold. First, we derive residual-based ASGS and OSGS formulations for a stationary generalized flow model with an externally prescribed, spatially varying porosity. Second, we use a Fourier-symbol scaling argument to motivate stabilization parameters that interpolate among diffusion-, convection-, and reaction-dominated regimes. Third, for the ASGS discretization of a linearized problem with constant viscosity and scalar reaction, homogeneous Dirichlet data, prescribed weighted-divergence-free advection, mesh-resolved positive porosity, and elementwise-constant stabilization parameters, we establish coercivity and an a priori error estimate in a porosity-weighted mesh-dependent norm. Fourth, we study nonlinear ASGS and OSGS implementations numerically, including pointwise parameter evaluation and a nonlinear Darcy–Brinkman–Forchheimer resistance. The latter experiments deliberately extend beyond the setting of the theorem and are interpreted as empirical evidence rather than as a direct verification of it.

## 18. Proposed replacement conclusion

> We have derived residual-based ASGS and OSGS formulations for a stationary generalized flow model with spatially varying porosity. A Fourier-symbol argument was used to motivate the scaling of the stabilization parameters. For the ASGS discretization of a linearized problem, under homogeneous Dirichlet conditions, constant viscosity and scalar reaction, prescribed weighted-divergence-free advection, mesh-resolved positive porosity, and elementwise-constant stabilization parameters, we established coercivity and an interpolation-based a priori estimate in a porosity-weighted stabilization norm. The parameter dependence of that result is explicit; most gradient controls carry a worst-case \(\alpha_0^{-1/2}\) factor, while the reaction-dominated pressure-gradient estimate carries the stronger \(\alpha_0^{-1}\) factor.
>
> The numerical study considered nonlinear ASGS and OSGS problems and therefore went beyond the proved setting. Across the two-dimensional manufactured tests, both variants generally produced velocity errors with the expected rates and often with magnitudes comparable to the nodal interpolation benchmark. Pressure behavior depended more strongly on the regime and the formulation. In particular, the OSGS linear-element velocity error increased in the reaction-dominated tests, and the three-dimensional OSGS pressure \(H^1\) error showed weak convergence. These observations are not explained by the present ASGS analysis and should be regarded as limitations and directions for further study.
>
> In the manufactured DBF comparison, the stabilized quadratic equal-order method gave velocity accuracy comparable to the unstabilized Taylor–Hood method in the viscosity-dominated cases, while Taylor–Hood was substantially more accurate for pressure. At high Reynolds number, the unstabilized Taylor–Hood velocity stagnated whereas the residual-stabilized equal-order methods converged on the reported meshes. This comparison demonstrates the importance of convection stabilization; it does not by itself isolate the effect of the velocity–pressure pair. In the difficult high-Reynolds/low-porosity cases, the nonlinear solvers failed on some coarse meshes. Without branch continuation, these failures should not be interpreted as proof that the discrete equations have no solution or possess a fold.
>
> Future work should address an OSGS and nonlinear error analysis, stabilization parameters varying within elements, discontinuous or under-resolved porosity, finite element interpolation of the porosity, compatible treatment of nonhomogeneous boundary data, continuation near difficult parameter corners, and manufactured solutions with genuinely three-dimensional variation.

## 19. Changes that may require reruns or new computations

The following items require new numerical work **only if the corresponding current claim is to be retained**. Each item includes a text-only fallback.

### R1. Establish or withdraw the discrete-fold/nonexistence claim — **required to retain the claim**

**Run:** pseudo-arclength continuation in a meaningful continuation parameter (mesh resolution can be awkward; a physical coefficient or forcing amplitude is preferable), from both sides of the suspected turning point. Record:

- solution norm and selected observables along the branch;
- complete residual norm;
- smallest singular value or relevant eigenvalue of the Jacobian;
- Newton iteration counts and step sizes;
- results from multiple initial guesses and continuation directions.

**Minimum acceptable output:** a branch diagram and Jacobian diagnostic showing a turning point or a documented exhaustive failure that is described more cautiously.

**Fallback:** replace “no solution,” “root exists from \(N\),” and “branch fold” by solver-nonconvergence wording.

### R2. Test interpolation of porosity — **required to retain the conclusion at line 1665**

**Run:** repeat representative low/high porosity cases with:

- analytic \(\alpha\) and \(\nabla\alpha\);
- nodal \(P_1\) interpolation of \(\alpha\);
- interpolation matching the solution degree;
- a positivity-preserving treatment if needed.

Report solution error relative to the exact PDE solution and the model/discretization difference between analytic and interpolated coefficients.

**Fallback:** delete the claim that the experiments show porosity interpolation does not spoil convergence.

### R3. Validate the 3D \(c_1\) narrative — **required to retain “theory predicts and experiments confirm”**

**Run:** for representative regular and irregular tetrahedral meshes, compare at least \(c_1=4k^4\), \(16k^4\), and a value above the local sufficient threshold. Report:

- local reference-element generalized eigenvalues;
- global minimum Rayleigh quotient/coercivity diagnostic;
- errors and convergence curves;
- sensitivity to mesh refinement.

**Fallback:** describe \(16k^4\) as an empirically selected value and remove claims of theorem-based certification.

### R4. Compare pointwise and elementwise-constant stabilization parameters — **strongly recommended**

Choose at least one viscosity-, convection-, and reaction-dominated cell for each polynomial order. Compare errors, residuals, and nonlinear convergence. This directly measures the gap between the theorem and implementation.

**Fallback:** make no claim that pointwise evaluation leaves results unchanged; state only that it is the implemented choice outside the proof.

### R5. Add a stabilized Taylor–Hood comparator — **required for a fair space-pair superiority claim**

Apply an appropriate convection/reaction stabilization to the Taylor–Hood pair and compare:

- unstabilized Taylor–Hood;
- stabilized Taylor–Hood;
- stabilized equal-order ASGS/OSGS.

Report DOFs and cost. This separates pressure-pair stability from high-Péclet stabilization.

**Fallback:** restrict the conclusion to the fact that residual stabilization succeeds where the *unstabilized* Galerkin comparator fails.

### R6. Add a genuinely three-dimensional manufactured solution — **recommended if 3D is a headline contribution**

Construct a weighted divergence-free flux \(q=\alpha u\) with all three components and full \(x,y,z\) dependence, for example through a vector potential, then set \(u=q/\alpha\). Use a nontrivial 3D pressure. Test both mesh families.

**Fallback:** call the existing case an extruded 3D discretization test.

### R7. Supply raw convergence data and regression slopes — **strongly recommended**

No new solve is needed if raw outputs exist. Publish errors for every mesh and compute local and multi-point slopes. If raw outputs are unavailable, rerun to regenerate them.

### R8. Verify Newton/Picard discrete-solution equivalence — **recommended**

On representative ASGS and OSGS cells, solve with both the displayed staggered Picard scheme and the Newton implementation to tight tolerance. Compare nodal vectors, complete residuals, and projected residuals.

**Fallback:** describe Newton as the implementation used and remove the unverified claim of equivalence to the displayed iteration.

### R9. Report/check the \(\varepsilon\) condition numerically — **recommended**

Compute the actual elementwise stability margins on every 3D mesh. This may require only post-processing, not new solves.

### R10. Diagnose the 3D OSGS pressure plateau — **strongly recommended**

Repeat selected cases while varying:

- constrained versus unconstrained projection space;
- quadrature order;
- pressure normalization;
- nonlinear tolerance;
- \(\varepsilon\);
- regular versus irregular mesh;
- a genuinely 3D pressure field.

At minimum, provide raw curves and state that the cause is unresolved.

### R11. Test the proposed OSGS reaction-gap mechanism — **optional but needed for the present causal explanation**

Vary \(\sigma\) continuously at fixed \(\nu\), \(u\), mesh, and porosity; compare full versus reaction-trimmed projection; plot the ASGS/OSGS error ratio against \(\mathrm{Da}_h\). Without this, retain only the observed gap.

### R12. Quantify porosity resolution — **post-processing or light rerun**

Report \(R_\alpha\) per mesh. If the coarse meshes strongly violate the hypothesis, consider one additional mesh family that resolves the transition layer independently of the global \(N\).

## 20. Changes requiring mathematical reworking but no numerical rerun

These are not merely prose edits, but they do not inherently require new simulations.

1. **Repair the convergence theorem/global norm issue** (M1–M2). Retain the \(\ell^2\) quantity if the face estimates allow it; otherwise propagate the true weaker result.
2. **Re-derive the robustness section** from the corrected global estimate, with explicit local/global quantifiers.
3. **Correct Eq. 1052** and every downstream statement that uses it.
4. **Rewrite the theorem, stability lemma, and continuity lemma as self-contained statements**, including all hypotheses and constant dependencies.
5. **Insert absolute values in continuity inequalities** and correct “there exists{s}.”
6. **Prove or rename the triple norm** as a seminorm/energy quantity.
7. **State the exact linear continuous and discrete ASGS problems** rather than referring to the nonlinear iteration equation.
8. **Clarify the pressure space as \(H^1\)-conforming** for residual-gradient terms.
9. **State and use the jump condition directly** in the main text.
10. **Correct the advection assumption commentary**: a finite element iterate is not automatically weighted-divergence-free.
11. **Separate fixed-data convergence from high-element-number parameter regimes.**
12. **Derive or remove the 3D \(\varepsilon_{\mathrm{ref}}\) inequality.**
13. **Correct the deviatoric-projection statement** and identify the relevant Korn-type requirement.
14. **Retype the weak boundary operators and pairings** in appropriate trace spaces.
15. **Define the actual OSGS projection space and residual** used in the implementation.
16. **Clarify the exact dimensional DBF reaction formula** and representative reaction number.

## 21. Changes that can be made by text/source editing only

These changes require no new mathematics or simulations if claims are weakened as indicated.

1. Replace broad “robust” claims by “parameter-explicit” or precisely scoped statements.
2. State at the start of the abstract/introduction/conclusion that the proof is linear ASGS only.
3. Distinguish heuristic Fourier design, proved ASGS analysis, and empirical nonlinear/OSGS results.
4. Replace all “no solution/fold/root exists” language by solver-nonconvergence language unless R1 is completed.
5. Fix the \(U=L=1\) contradiction.
6. Delete the porosity-interpolation conclusion unless R2 is completed.
7. State the \(1/\alpha_0\) reaction pressure-gradient exception.
8. Replace “genuinely three-dimensional” by “extruded three-dimensional discretization test” unless R6 is completed.
9. Restrict the Taylor–Hood claim to comparable velocity accuracy in the viscosity-dominated cases; acknowledge much better Taylor–Hood pressure accuracy.
10. State that the high-Reynolds comparison is against unstabilized Galerkin Taylor–Hood.
11. Replace “nodal interpolation floor/absolute reference” by “nodal interpolation benchmark.”
12. Remove or qualify causal explanations not isolated by an experiment.
13. Rename the OSGS “excess component” as a non-orthogonal scalar indicator or remove it.
14. Remove the priority claim “only previously proposed stabilized method.”
15. Remove unsupported data-structure complexity rhetoric.
16. Replace “theoretical” OSGS table orders by “nominal interpolation/ASGS benchmark orders.”
17. Break long numerical paragraphs into observation/interpretation/limitation units.
18. Resolve all `\amend`, `\Guillermo`, and `\Joaquin` markup.
19. Restore missing build dependencies and normalize filenames.
20. Clean the bibliography and remove unused/template entries.
21. Fix wording/typography: “there exists,” `I_d`, “solid volume fraction,” “readers less familiar,” and section-heading layout hacks.
22. Add a limitations paragraph before future work.

## 22. Recommended order of revision

1. Decide whether the \(\ell^2\) convergence estimate can be retained; repair the theorem and robustness section.
2. Freeze the exact scope of the proved contribution.
3. Decide which strong numerical claims are worth rerunning to retain; otherwise weaken them immediately.
4. Rewrite abstract, introduction contribution paragraph, numerical preamble, and conclusion using the same scope language.
5. Add reproducibility details/raw data and revise all numerical interpretations.
6. Clean the source package, bibliography, revision markup, and style.
7. Compile from a clean environment and perform a final rendered-page audit.

## 23. Bottom-line publication risk

The central risk is not that the method is necessarily wrong. It is that the manuscript currently claims more than its stated theorem and reported experiments establish. A rigorous revision should make the contribution **narrower but stronger**:

- a clearly stated linear ASGS theorem;
- a correctly assembled global error estimate;
- a parameter-regime discussion derived from that estimate;
- nonlinear ASGS/OSGS experiments presented as exploratory evidence;
- honest treatment of anomalies and solver limitations.

That version would be substantially clearer, more credible, and easier for a reader or reviewer to verify.
