> **SUPERSEDED IN PART (2026-07-29, later the same day).** The block this spec places in an
> Appendix C subsubsection `\label{sec:projectors}` was subsequently RELOCATED to a new
> main-text subsection `\label{sec:ViscousProjector}` ("The viscous projector", Sec. 2.1),
> duplicated verbatim in BOTH mains, together with the display `eq:PiKorn` (moved out of the
> (A3) item, which now states the bare conditions only). Do not re-apply the Appendix C
> placement described below; see `theory/README.md` and the header of
> `theory/paper/asgs_convergence.tex`.
>
> **ALSO SUPERSEDED (2026-07-31): the macro BODIES quoted below.** The `\newcommand` snippets in
> this brief still define `\ViscProj` and friends as `\overset{...}{\Pi}`. That notation was replaced
> across `theory/` on 2026-07-31: the projector is now sans-serif with the instance named by the
> subspace it keeps (`\ViscProj`=𝖯, `\SPi`=𝖯_Sym, `\DPi`=𝖯_Dev, `\DSPi`=𝖯_DS, plus `\IPi`=𝖨 and
> `\AnyProj`=𝖰). The macro *names* and their semantics are unchanged, so every instruction below that
> refers to a macro by name still applies verbatim — only the four bodies are stale. Rationale is in
> the preamble comment of both mains; see also `theory/README.md`.
>
> **ALSO SUPERSEDED (2026-07-30): the `\amend{...}` wrapping.** Step 7 below and every verbatim
> snippet wrap new prose in `\amend{...}`. That macro no longer exists — all 1011 wrappers were
> unwrapped and the `\newcommand` deleted (`pending-tasks.md` §1c). Read those `\amend{…}` as plain
> text; re-applying them would inject an undefined control sequence.

# Implementation brief — generalize the viscous projector in the a priori theory of `article_v2.tex`

**Role.** You are editing the LaTeX sources of the SIAM paper *"A stabilized finite element method for incompressible, inertial flows in inhomogeneous porous media"* (Casas, González-Usúa, Codina, de-Pouplana), repository directory `theory/`. Your task is to generalize the stability and convergence theory from the specific deviatoric–symmetric viscous projector to an axiomatically defined family of projectors, following this brief **exactly**. Every mathematical statement you will insert has been proved and numerically verified in advance (see §9); do not weaken, strengthen, or "improve" the mathematics. Where this brief gives verbatim LaTeX, use it (adapting only spacing/line breaks and any macro-name collisions you detect); where it gives instructions, follow them conservatively.

**Prime directive.** This is a *scope generalization with zero quantitative cost*: the proofs of coercivity, continuity, inf–sup stability, consistency and convergence are **not to be re-derived or restructured**. The only mathematical point where the projector's specific form is currently used is the definiteness of the working norms (Lemma `lem:definiteness` in `paper/asgs_convergence.tex`, equation `eq:devsymidentity`, and its delegation from the OSGS appendix). Everything else uses only two abstract properties, which you will now make standing assumptions. If at any point an edit seems to require touching the body of a proof other than as specified below, **stop and report** instead of improvising.

---

## 0. Objective

The paper's model section (§2, `sec:TheContinuousProblem`) already introduces the viscous term
`-2∇·(αν Π ∇u)` with `Π := D̂Ŝ` a product of commuting orthogonal projections, and explicitly
lists alternative instances from the literature: `D̂ ≡ I, Ŝ = sym` (the symmetric-gradient
operator of Cocquet et al., DBF) and `D̂ = Ŝ ≡ I` (full gradient, Skrzypacz). The a priori
analysis, however, is currently stated and proved only for the deviatoric–symmetric instance.
Moreover, the DBF comparison experiment (`sec:NumericalExampleFromLiterature`) **runs the
stabilized ASGS/OSGS methods with `Π = Ŝ`**, so the theory as written does not literally cover
one of the paper's own experiments.

The generalization:

1. introduces a new **common standing assumption** `H:projector` (with display `eq:PiKorn`)
   stating the two properties the proofs actually use — (Π1) constant-coefficient orthogonal
   projection, (Π2) a Korn-type compatibility inequality on `H_0^1(Ω)^d`;
2. adds to Appendix C (`paper/asgs_convergence.tex`) a short new subsubsection with:
   a **sufficient algebraic criterion** ("range contains the deviatoric–symmetric tensors" ⇒
   (Π2) with the uniform constant √2), the **verification for the four canonical instances**
   `{I, Ŝ, D̂, D̂Ŝ}` with sharp constants `{1, √2, √(d/(d−1)), √2}`, and a scoping **remark**
   giving the sharp algebraic characterization of (Π2);
3. rewrites `lem:definiteness` to cite the assumption (one-paragraph proof; the
   deviatoric–symmetric identity `eq:devsymidentity` **moves** into the new lemma, keeping its
   label so all existing cross-references remain valid);
4. makes small, anchored wording edits so that the model section, the τ-design section, the
   Fourier appendix, the strong-form matrices, the OSGS appendix and the numerical sections
   are consistent with the generic projector, with the deviatoric–symmetric operator clearly
   flagged as the default instance used in the experiments (and `Ŝ` as the instance used in
   the DBF comparison, now covered by the theory).

No estimate, no constant and no hypothesis of the five main results
(`lemma:Stability`, `lemma:Continuity`, `th:Convergence`, `th:StabilityOSGS`,
`th:ConvergenceOSGS`) changes quantitatively; the Korn constant `C_K` enters only the
definiteness lemma.

---

## 1. Ground truth: the audit of where the projector enters the proofs

You do not need to re-do this audit, but you must not contradict it. Every use of `\ViscProj`
in the two analysis appendices falls into exactly one of these classes:

| # | Property used | Where (labels/files) |
|---|---|---|
| 1 | **Self-adjoint idempotent (Frobenius), pointwise, constant coefficients** ⇒ the weak viscous form is `2ν(Π∇v, αΠ∇u)`; Galerkin diagonal `eq:GalerkinCoercivityIdentity` / `oa:eq:CoercivityIdentity`; global IBP in the OSGS consistency proof (`oa:lem:consistency`). | main texts; `asgs_convergence.tex` `eq:Bstab`; both OSGS appendix files |
| 2 | **Non-expansiveness** `\|ΠT\| ≤ \|T\|` (consequence of #1). | `eq:winv-grad`; interpolation replacements `eq:interpdivvisc`–`eq:interpgrad`; OSGS stability Step 3 (term T1); OSGS (I1) |
| 3 | **Constant coefficients ⇒ `Π∇w_h` inherits the FE structure elementwise**, so the plain inverse estimate applies (with the *same, projector-independent* constant `C_inv`, since each component of `Π∇w_h` is a fixed constant combination of the `∂_j w_i`). | `lem:winv` proof (sentence "acts pointwise with constant coefficients"); `eq:doubleinv` |
| 4 | **Major symmetry** `P_{aibj} = P_{bjai}` (equivalent to #1) ⇒ `K_ji^⊤ = K_ij`, the symmetry used implicitly in the adjoint `eq:AdjointFlux`/`eq:AdjointDifferentialOperator`. | main texts, abstract-form section |
| 5 | **Korn compatibility** `‖∇v‖ ≤ C_K‖Π∇v‖` on `H_0^1(Ω)^d` — used *only* for definiteness of the triple norms. | `lem:definiteness` (via `eq:devsymidentity`); delegated by `oa:lem:definiteness` (both OSGS appendix copies) and discussed in the pedagogy box of `osgs_appendix_commented.tex` |
| 6 | **Instance-specific material that stays instance-specific** (no theorem depends on it): the DS closed forms in `eq:matrices_stationary_strong_problem` and the elemental-matrices appendix; the DS Fourier symbol `eq:ftViscClosed`/`eq:ftViscEig`/`eq:ftTauNu` and the "for d=2 there is nothing to drop" sentence; the 3D `P2`-tetrahedron threshold footnote (`c_1 = 16k^4`); the Neumann trace `eq:neumann_bc` (generic as written). | main texts, `fourier_appendix.tex`, `elemental_matrices_appendix.tex` |

Classes 1–4 are consequences of assumption (Π1) below; class 5 is assumption (Π2); class 6
gets scoping sentences only.

---

## 2. Hard build constraints (read before editing anything)

1. **`paper/asgs_convergence.tex` is `\input` by BOTH `paper/article.tex` (v1) and
   `paper/article_v2.tex` (v2).** Its own header comment says it may cite only labels defined
   in **both** builds. The new assumption `H:projector` and its display `eq:PiKorn` must
   therefore be added to **both** main files (v1 defines its H:\* labels inside its
   `sec:StabilityASGS` assumption list; v2 inside `sec:CommonSetting`). Any text you add to
   `asgs_convergence.tex` or `fourier_appendix.tex` (also shared) may cite only: labels those
   files define themselves, `H:*` labels present in both mains, and main-text labels already
   cited by those files (which therefore exist in both). **Never** cite from shared files:
   `sec:CommonSetting`, `sec:TheContinuousProblem` (verify existence in v1 first; if absent,
   write "of the main text" instead of a `\cref`), `app:osgs`, `sec:StabilityOSGS`,
   `eq:TripleNormOSGS`, any `oa:*` label, `H:design`…`H:advectionsmooth`.
2. **Update the fragility header** of `asgs_convergence.tex`: add `H:projector` and
   `eq:PiKorn` to its list of labels guaranteed to exist in both builds, and note the date.
3. **`osgs_appendix_commented.tex` is the copy `article_v2.tex` inputs**;
   `osgs_appendix.tex` is its uncommented sibling. Apply the OSGS edits **identically to
   both** (the pedagogy-box edit exists only in the commented copy).
4. **Insertion point of the new assumption**: immediately **after `H:data`** in both mains.
   Rationale: several hypothesis lists cite ranges (`\crefrange{H:mesh}{H:regularity}`,
   `\cref{H:mesh}--\cref{H:spaces}` in `th:StabilityOSGS`/`oa:th:stability`,
   `\cref{H:mesh}--\cref{H:regularity}` throughout both OSGS appendix copies). Placing the new
   item after `H:data` keeps it inside *every* such range with **zero** edits to those
   statements. Do not place it after `H:regularity` or the OSGS ranges will silently exclude
   it. After inserting, `grep` both mains and all appendices for literal strings `(A1)`
   … `(A7)`, `(S1)`, `(O1)` outside labels to confirm no hard-coded item numbers exist in
   prose (references are all `\cref`-based; the only known occurrence is a preamble *comment*
   in v2 listing "(A1)-(A6)", which you should update to "(A1)-(A7)").
5. **Do not touch** the bodies of the coercivity, continuity, convergence, inf–sup,
   consistency or interpolation proofs except at the precise anchors below.
6. **Compile discipline**: after all edits, build **both** `article.tex` and `article_v2.tex`
   from `theory/paper/` with the local `latexmkrc`; require exit 0, zero undefined
   references/citations, zero multiply-defined labels, in both builds.
7. American (SIAM) spelling in new paper prose; wrap new **main-text** prose in `\amend{...}`
   (the 2026 amendment marker, which renders flat); appendix additions go unwrapped, matching
   the existing appendix style.

---

## 3. Notation and macros

Current state: v2 defines, at the top of `sec:TheContinuousProblem`,

```latex
\newcommand{\ViscProj}{\overset{\scriptscriptstyle{\mathrm{DS}}}{\Pi}}
\newcommand{\DPi}{\overset{\scriptscriptstyle{\mathrm{D}}}{\Pi}}
\newcommand{\SPi}{\overset{\scriptscriptstyle{\mathrm{S}}}{\Pi}}
```

v1 has a parallel definition block (locate it by grepping `newcommand{\ViscProj}` in
`article.tex`). The standalone notes (`continuity appendix/`, `osgs_a_priori/`) define their
own copies — out of scope except as noted in §7.

**Recommended macro plan (Option R — honest notation; use this unless the user has said
otherwise):**

* In **both** mains, change the decoration of the generic projector from `DS` to `v`
  (viscous), and add a dedicated macro for the deviatoric–symmetric instance:

  ```latex
  \newcommand{\ViscProj}{\overset{\scriptscriptstyle{\mathrm{v}}}{\Pi}}   % generic viscous projector
  \newcommand{\DSPi}{\overset{\scriptscriptstyle{\mathrm{DS}}}{\Pi}}      % deviatoric-symmetric instance
  ```

  Semantics after the change: `\ViscProj` = the generic projector of `H:projector`
  (this is what *every* occurrence in the two analysis appendices, the working norms, the
  Galerkin identity, the Neumann trace and the robustness section now means — **no
  search-and-replace needed there**); `\DSPi` = the default instance. Only the spots listed in
  §6 that *specifically mean the DS instance* switch to `\DSPi` or get a scoping sentence.
* Korn constant: `C_{\mathrm K}` (subscript "K" for Korn). Do **not** use `C_\Pi` (`Π_1`,
  `Π_2`, `Π_h` are taken by the OSGS machinery).

**Fallback (Option K — only if the user prefers zero glyph churn):** keep the `DS` decoration
on `\ViscProj`, still add `\DSPi` (then identical in appearance), and add one footnote at the
first analysis use stating that, by an abuse of notation, the symbol denotes any projector
satisfying `H:projector`. Everything else in this brief is unchanged. Flag clearly in your
final report which option you implemented.

---

## 4. LaTeX payload

### P1 — the new standing assumption (v2 wording; insert as a new `\item` right after the `H:data` item in `sec:CommonSetting`)

```latex
  \item\label{H:projector} \emph{Viscous projector.} The operator $\ViscProj$ of
  \cref{eq:StrongMomentumEquation} is a linear map on second-order tensors, acting pointwise
  with constant coefficients, that is an orthogonal projection with respect to the Frobenius
  inner product: $\ViscProj\ViscProj=\ViscProj$ and
  $(\ViscProj\mathbf{A})\!:\!\mathbf{B}=\mathbf{A}\!:\!(\ViscProj\mathbf{B})$ for all tensors
  $\mathbf{A},\mathbf{B}$; in particular $|\ViscProj\mathbf{T}|\le|\mathbf{T}|$ pointwise.
  Moreover, $\ViscProj$ is \emph{Korn{\hyp}compatible}: there is a constant
  $C_{\mathrm{K}}\ge1$, independent of the mesh and of the physical parameters, such that
  \begin{equation}\label{eq:PiKorn}
    \lVert\nabla\boldsymbol{v}\rVert
    \le C_{\mathrm{K}}\,\bigl\lVert\ViscProj\nabla\boldsymbol{v}\bigr\rVert
    \qquad\forall\,\boldsymbol{v}\in H^1_0(\Omega)^d .
  \end{equation}
  Every orthogonal projection whose range contains the deviatoric--symmetric tensors
  satisfies \cref{eq:PiKorn} with $C_{\mathrm{K}}=\sqrt2$; in particular so do all four
  members $\ViscProj\in\{\mathbb{I},\,\SPi,\,\DPi,\,\DSPi\}$ of the family introduced in
  \cref{sec:TheContinuousProblem} (\cref{lem:projfamily,lem:projinstances} of
  \cref{app:Continuity}, where the sharp constants are also given). The constant
  $C_{\mathrm{K}}$ enters the analysis only through the definiteness of the working norms
  (\cref{lem:definiteness}); no other constant below depends on the choice of $\ViscProj$.
```

For **v1**, insert the same item after its `H:data` item, with two adaptations: (i) if v1 has
no `\label{sec:TheContinuousProblem}`, replace that `\cref` by "the model section"; (ii) keep
the rest verbatim (v1 also inputs `asgs_convergence.tex`, so `app:Continuity`,
`lem:projfamily`, `lem:projinstances`, `lem:definiteness` resolve).

### P2 — new subsubsection for `paper/asgs_convergence.tex`

Insert **inside** `\subsection{Technical results}\label{sec:auxiliary}`, as a new
subsubsection placed **after** "Two global integration-by-parts identities" and **before**
"The ASGS triple norm":

```latex
\subsubsection{Admissible viscous projectors}\label{sec:projectors}

The results of this appendix hold for any viscous projector satisfying the standing
assumption \cref{H:projector}; the deviatoric--symmetric choice used in the numerical
experiments is one instance. This subsubsection collects the algebra behind that assumption:
a sufficient criterion with a uniform Korn constant (\cref{lem:projfamily}) and its
verification for the projector family introduced in the main text
(\cref{lem:projinstances}). Throughout, $\SPi\mathbf{T}=\operatorname{sym}\mathbf{T}
\coloneqq\tfrac12(\mathbf{T}+\mathbf{T}^{\mathsf T})$,
$\DPi\mathbf{T}\coloneqq\mathbf{T}-\tfrac1d(\operatorname{tr}\mathbf{T})\,\mathbf{I}$, and
$\DSPi\coloneqq\DPi\SPi$ is the deviatoric--symmetric projection.

\begin{lemma}[A sufficient criterion for Korn compatibility]\label{lem:projfamily}
Let $\Pi$ be a constant-coefficient orthogonal projection on second-order tensors whose
range contains the deviatoric--symmetric tensors, i.e., $\Pi\mathbf{T}=\mathbf{T}$ whenever
$\mathbf{T}=\mathbf{T}^{\mathsf T}$ and $\operatorname{tr}\mathbf{T}=0$. Then, pointwise,
\begin{equation}\label{eq:projmonotone}
  \bigl|\Pi\mathbf{T}\bigr|^2
  = \bigl|\DSPi\,\mathbf{T}\bigr|^2 + \bigl|(\Pi-\DSPi)\mathbf{T}\bigr|^2
  \;\ge\; \bigl|\DSPi\,\mathbf{T}\bigr|^2
  \qquad\text{for every tensor }\mathbf{T},
\end{equation}
and, for every $\bv\in H^1_0(\Omega)^d$,
\begin{equation}\label{eq:devsymidentity}
  \bigl\lVert\DSPi\nabla\bv\bigr\rVert^2
  = \tfrac12\lVert\nabla\bv\rVert^2
  + \Bigl(\tfrac12-\tfrac1d\Bigr)\lVert\nabla\cdot\bv\rVert^2
  \ \ge\ \tfrac12\lVert\nabla\bv\rVert^2 .
\end{equation}
Consequently $\lVert\nabla\bv\rVert\le\sqrt2\,\lVert\Pi\nabla\bv\rVert$ for all
$\bv\in H^1_0(\Omega)^d$: every such $\Pi$ satisfies \cref{eq:PiKorn} with
$C_{\mathrm{K}}=\sqrt2$, uniformly over the family and over the domain.
\end{lemma}

\begin{proof}
Since the range of $\Pi$ contains that of $\DSPi$ and both are orthogonal projections,
$\Pi\DSPi=\DSPi\Pi=\DSPi$, so $\Pi-\DSPi$ is itself an orthogonal projection, with range
orthogonal to that of $\DSPi$; \cref{eq:projmonotone} is the resulting pointwise Pythagoras
identity. For \cref{eq:devsymidentity}, write
$\DSPi\nabla\bv=\operatorname{sym}\nabla\bv-\tfrac1d(\nabla\cdot\bv)\,\mathbf{I}$; since the
deviatoric and spherical parts are orthogonal,
$\lVert\DSPi\nabla\bv\rVert^2=\lVert\operatorname{sym}\nabla\bv\rVert^2
-\tfrac1d\lVert\nabla\cdot\bv\rVert^2$. For $\bv\in H^1_0(\Omega)^d$, integration by
parts---first for $\bv\in C_c^\infty(\Omega)^d$ and then by density---gives
\begin{equation}\label{eq:crossibp}
  \int_\Omega\nabla\bv:\nabla\bv^{\mathsf T}\dd\Omega
  = \int_\Omega(\nabla\cdot\bv)^2\dd\Omega ,
\end{equation}
whence $\lVert\operatorname{sym}\nabla\bv\rVert^2
=\tfrac12\lVert\nabla\bv\rVert^2+\tfrac12\lVert\nabla\cdot\bv\rVert^2$; combining the two
identities yields \cref{eq:devsymidentity}, the final inequality following from
$\tfrac12-\tfrac1d\ge0$ for $d\ge2$. Integrating \cref{eq:projmonotone} with
$\mathbf{T}=\nabla\bv$ and chaining with \cref{eq:devsymidentity} gives the Korn bound. The
constant $\sqrt2$ is sharp already for $\Pi=\DSPi$: divergence-free fields turn
\cref{eq:devsymidentity} into an equality.
\end{proof}

\begin{lemma}[Instances]\label{lem:projinstances}
The four members of the family introduced in the main text,
$\Pi\in\{\mathbb{I},\ \SPi,\ \DPi,\ \DSPi\}$, are constant-coefficient orthogonal
projections whose ranges contain the deviatoric--symmetric tensors, and therefore satisfy
\cref{H:projector} with $C_{\mathrm{K}}=\sqrt2$. The sharp constants are, respectively,
\begin{equation}\label{eq:projconstants}
  C_{\mathrm{K}}(\mathbb{I})=1,\qquad
  C_{\mathrm{K}}(\SPi)=\sqrt2,\qquad
  C_{\mathrm{K}}(\DPi)=\sqrt{\tfrac{d}{d-1}},\qquad
  C_{\mathrm{K}}(\DSPi)=\sqrt2 .
\end{equation}
\end{lemma}

\begin{proof}
Idempotency and Frobenius self-adjointness are immediate for each operator (for $\SPi$ they
follow from $\mathbf{A}^{\mathsf T}\!:\!\mathbf{B}=\mathbf{A}\!:\!\mathbf{B}^{\mathsf T}$,
and for the spherical part $\mathbb{I}-\DPi$ from
$(\operatorname{tr}\mathbf{A})\,\mathbf{I}\!:\!\mathbf{B}
=(\operatorname{tr}\mathbf{A})(\operatorname{tr}\mathbf{B})$), and each fixes every
symmetric traceless tensor, so \cref{lem:projfamily} applies. For the sharp constants:
$C_{\mathrm{K}}(\mathbb{I})=1$ trivially; for $\SPi$, \cref{eq:crossibp} gives
$\lVert\SPi\nabla\bv\rVert^2=\tfrac12\lVert\nabla\bv\rVert^2
+\tfrac12\lVert\nabla\cdot\bv\rVert^2\ge\tfrac12\lVert\nabla\bv\rVert^2$, with equality for
divergence-free fields, and the same fields make \cref{eq:devsymidentity} an equality for
$\DSPi$. For $\DPi$, the pointwise orthogonal splitting gives
$\lVert\DPi\nabla\bv\rVert^2=\lVert\nabla\bv\rVert^2
-\tfrac1d\lVert\nabla\cdot\bv\rVert^2$, and \cref{eq:crossibp} with the Cauchy--Schwarz
inequality yields $\lVert\nabla\cdot\bv\rVert^2\le
\lVert\nabla\bv\rVert\,\lVert\nabla\bv^{\mathsf T}\rVert=\lVert\nabla\bv\rVert^2$, whence
$\lVert\DPi\nabla\bv\rVert^2\ge\bigl(1-\tfrac1d\bigr)\lVert\nabla\bv\rVert^2$. Sharpness
follows by taking $\bv=\nabla\varphi$ with $\varphi\in C_c^\infty(\Omega)$: then
$\nabla\bv$ is symmetric and
$\lVert\nabla\cdot\bv\rVert=\lVert\Delta\varphi\rVert
=\lVert D^2\varphi\rVert=\lVert\nabla\bv\rVert$, by a double integration by parts.
\end{proof}

\begin{remark}[Scope of the assumption]\label{rem:projscope}
Containment of the deviatoric--symmetric tensors in the range is sufficient, not necessary,
for \cref{eq:PiKorn}. The sharp characterization is algebraic: \cref{eq:PiKorn} holds---for
every domain, with $C_{\mathrm{K}}\le c_\Pi^{-1}$---if and only if the kernel of $\Pi$
contains no nonzero rank-one tensor, where
$c_\Pi\coloneqq\min\{|\Pi(\bv\otimes\boldsymbol{k})| : |\bv|=|\boldsymbol{k}|=1\}$.
Sufficiency follows from Plancherel's theorem applied to the extension of $\bv$ by zero,
since $\lVert\Pi\nabla\bv\rVert^2_{L^2(\mathbb{R}^d)}
=\int_{\mathbb{R}^d}\bigl|\Pi\bigl(\widehat{\bv}\otimes\boldsymbol{k}\bigr)\bigr|^2
\dd\boldsymbol{k}\ge c_\Pi^2\lVert\nabla\bv\rVert^2$; necessity, from profiles
$\bv_\lambda=\bv_0\,\varphi(\boldsymbol{x})\sin(\lambda\boldsymbol{k}_0\cdot\boldsymbol{x})$
with $\varphi\in C_c^\infty(\Omega)$ oscillating along a rank-one kernel direction
$\Pi(\bv_0\otimes\boldsymbol{k}_0)=0$, for which
$\lVert\nabla\bv_\lambda\rVert/\lVert\Pi\nabla\bv_\lambda\rVert\to\infty$. Thus, for
instance, the complementary projector $\mathbb{I}-\DSPi$ (spherical plus skew-symmetric
part) is also admissible---with $C_{\mathrm{K}}=\sqrt2$ for $d=2$---even though its range
meets the deviatoric--symmetric tensors only at the origin, whereas the skew-symmetric and
spherical parts alone are not (their kernels contain the rank-one tensors
$\bv\otimes\bv$ and $\boldsymbol{e}_1\otimes\boldsymbol{e}_2$, respectively; gradient
fields $\bv=\nabla\varphi$ and divergence-free fields realize the degeneracy in
$H^1_0(\Omega)^d$). The physically motivated choices all lie in the family of
\cref{lem:projinstances}, which is why we state the sufficient criterion rather than the
characterization.
\end{remark}
```

### P3 — rewritten definiteness lemma (replaces the current `lem:definiteness` **and its proof** in `asgs_convergence.tex`; the surrounding "The ASGS triple norm" prose stays)

```latex
\begin{lemma}[Definiteness of the working norm]\label{lem:definiteness}
Let \cref{H:data,H:projector,H:porosity,H:spaces} hold. Then $\triplenorm{\cdot}$ of
\eqref{eq:triplenorm} is a norm on $\Xh$.
\end{lemma}
\begin{proof}
Homogeneity, symmetry and the triangle inequality being immediate, only definiteness
remains. Suppose $\triplenorm{V_h}=0$. Since $\nu>0$ (\cref{H:data}) and
$\alpha\ge\alpha_0>0$ (\cref{H:porosity}), the first term forces $\ViscProj\nabla\bvh=0$,
hence $\nabla\bvh=0$ by the Korn compatibility \cref{eq:PiKorn} of \cref{H:projector}
(available with $C_{\mathrm{K}}=\sqrt2$ for the whole family of \cref{lem:projinstances},
by \cref{lem:projfamily}) and $\bvh=\boldsymbol 0$ by Poincar\'e. Then $\X(V_h)=\nabla p_h$,
and the $\tau_1$ term---whose weights $\tau_{1,K}$ are positive---forces $\nabla p_h=0$, so
that $p_h$ is constant on the connected domain $\Omega$
($\mathcal{Q}_{h0}\subset H^1(\Omega)$ by \cref{H:spaces}). That constant vanishes: through
the term $\varepsilon\norm{p_h}^2$ when $\varepsilon>0$, and through the zero{\hyp}mean
constraint defining $\mathcal{Q}_0$ when $\varepsilon=0$.
\end{proof}
```

Also adjust the sentence *introducing* the lemma (currently ending "…is the content of the
following lemma."): no change needed, but delete any leftover in-lemma derivation of the DS
identity — it now lives in P2.

### P4 — setting sentence of the ASGS appendix

In `asgs_convergence.tex`, `\subsection{The linearized problem and the ASGS method}`, replace

> `$\varepsilon\ge 0$ the penalty parameter, and $\ViscProj$ the (pointwise, orthogonal) projection onto the deviatoric--symmetric part of a second-order tensor, so that $\bigl|\ViscProj\mathbf{T}\bigr|\le|\mathbf{T}|$ pointwise for every tensor $\mathbf{T}$.`

by

```latex
$\varepsilon\ge 0$ the penalty parameter, and $\ViscProj$ the viscous projector of
\cref{H:projector}: a pointwise, constant-coefficient orthogonal projection of second-order
tensors---so that $\bigl|\ViscProj\mathbf{T}\bigr|\le|\mathbf{T}|$ pointwise for every
tensor $\mathbf{T}$---which is Korn-compatible in the sense of \cref{eq:PiKorn}. The
deviatoric--symmetric choice $\ViscProj=\DSPi$ used in the numerical experiments is one
instance; see \cref{sec:projectors}.
```

### P5 — Fourier appendix (shared by both builds)

(a) In `\subsection{Viscous term…}` (`sec:ftViscous`), after the sentence introducing
`eq:ftViscSymbol` ("The viscous part is second order, so its symbol is…"), insert the scoping
clause: change "Inserting the velocity block of $\mathbf{K}_{ij}$ from
\cref{eq:matrices_stationary_strong_problem}, whose entries…" to
"Inserting the velocity block of $\mathbf{K}_{ij}$ from
\cref{eq:matrices_stationary_strong_problem} \amend{(the default instance
$\ViscProj=\DSPi$; see \cref{rem:ftGenericPi} below for the general case)}, whose entries…".

(b) Append at the end of `sec:ftViscous` (after the sentence "…consistent downstream."):

```latex
\begin{remark}[General viscous projectors]\label{rem:ftGenericPi}
The closed form \cref{eq:ftViscClosed} is the deviatoric--symmetric instance of a
projector-uniform structure. For any $\ViscProj$ satisfying \cref{H:projector}, the viscous
symbol on the velocity block is
$\widehat{\mathcal{L}}_{\nu}=(2\alpha\nu/h^2)\,\mathbf{T}_{\ViscProj}(\boldsymbol{k}_0)$,
where $\mathbf{T}_{\ViscProj}(\boldsymbol{k})$ is the symmetric positive-semidefinite
matrix with quadratic form
$\boldsymbol{v}^{\mathsf T}\mathbf{T}_{\ViscProj}(\boldsymbol{k})\,\boldsymbol{v}
=|\ViscProj(\boldsymbol{v}\otimes\boldsymbol{k})|^2$. Non-expansiveness gives
$\mathbf{T}_{\ViscProj}(\boldsymbol{k}_0)\le|\boldsymbol{k}_0|^2\,\mathbb{I}_d$, and for
every projector whose range contains the deviatoric--symmetric tensors
(\cref{lem:projfamily} of \cref{app:Continuity})---in particular for the whole family of
\cref{lem:projinstances}---also
$\mathbf{T}_{\ViscProj}(\boldsymbol{k}_0)\ge\tfrac12|\boldsymbol{k}_0|^2\,\mathbb{I}_d$.
The spectral radius entering $\tau_{\nu,1}^{-1}$ therefore lies between
$\alpha\nu|\boldsymbol{k}_0|^2/h^2$ and $2\alpha\nu|\boldsymbol{k}_0|^2/h^2$ for every
admissible projector: the factor is $2-\tfrac2d$ for $\DSPi$ (the value displayed in
\cref{eq:ftTauNu}) and $2$ for $\mathbb{I}$ and for $\SPi$. Dropping this $O(1)$ factor, as
done below \cref{eq:StabilizationParameters}, is thus uniform over the family, and the
parameters \cref{eq:Tau1,eq:Tau2} require no modification.
\end{remark}
```

Before finalizing (b), verify that `eq:StabilizationParameters`, `eq:Tau1`, `eq:Tau2` are
defined in **both** mains (the appendix already cites them, so they should be); if any is
v2-only, drop that `\cref` and refer to "the main text" instead.

### P6 — model section (v2; mirror in v1 where the parallel text exists)

(a) At the end of the paragraph following `eq:StrongMassEquation` (the one ending "…see
\cref{sec:NumericalExamplesStationaryCase3D}." — note the commented-out sentence
"`%. $\scriptstyle{\ViscProj}$ is an orthogonal linear projection operator.`" nearby), append:

```latex
\amend{The operator $\ViscProj$ is a fixed linear map on second-order tensors, acting
pointwise with constant coefficients; the precise properties the analysis requires of it are
collected in the standing assumption \cref{H:projector} of \cref{sec:CommonSetting}.}
```

(v1: point instead to its assumption list — e.g. "…are collected in the standing assumption
\cref{H:projector} of \cref{sec:StabilityASGS}." after checking the v1 label of that section.)

(b) Rewrite the opening of the family paragraph. Replace

> `By defining $\scriptstyle{\ViscProj} \displaystyle{\coloneqq} \scriptstyle{\DPi\SPi}$, where $\scriptstyle{\DPi}$ and $\scriptstyle{\SPi}$ are \amend{commuting} orthogonal linear projection operators, and considering different versions of the latter two, we obtain alternative formulations found in different contexts in the literature. Our particular choice for these operators in the examples presented, corresponds to taking $\scriptstyle\DPi$ and $\scriptstyle\SPi$ as the operators that extract, respectively, the deviatoric and symmetric components of the tensor upon which they act. This yields`

by

```latex
By considering $\scriptstyle{\ViscProj}\displaystyle{\,=\,}\scriptstyle{\DPi\SPi}$, where
$\scriptstyle{\DPi}$ and $\scriptstyle{\SPi}$ are \amend{commuting} orthogonal linear
projection operators\amend{---each equal either to the identity or to the operator that
extracts, respectively, the deviatoric or the symmetric component of the tensor upon which
it acts---}we obtain alternative formulations found in different contexts in the
literature. \amend{Our default choice, denoted $\DSPi$ and used in all the numerical
examples except where stated (cf.\ \cref{sec:NumericalExampleFromLiterature}), takes both
projections nontrivially, $\ViscProj=\DSPi\coloneqq\DPi\SPi$.} This yields
```

(the displayed decomposition equation that follows stays as is; with the default just fixed
in words, it remains literally correct).

(c) At the end of the "As mentioned, other combinations are possible…" paragraph (after the
Cocquet/Skrzypacz sentence and its footnote), append:

```latex
\amend{All these combinations---and in fact any projector satisfying the standing assumption
\cref{H:projector} below---fall within the scope of the a priori analysis of
\cref{sec:StabilityASGS}: the stability and convergence results there are stated and proved
for a generic admissible $\ViscProj$, with the concrete instances verified in
\cref{lem:projinstances} of \cref{app:Continuity}.}
```

(v1: same, with `sec:StabilityASGS` as its analysis section; drop the mention of
`sec:NumericalExampleFromLiterature` in (b) if v1 lacks that section.)

### P7 — strong-form matrices and adjoint symmetry (v2 `sec:AbstractReformulation`; mirror in v1)

Immediately after the sentence following `eq:matrices_stationary_strong_problem` ("where
$\delta_{\bullet\bullet}$ is the Kronecker delta…brevity."), append:

```latex
\amend{The displayed $\mathbf{K}_{ij}$ is the instance corresponding to the default
$\ViscProj=\DSPi$. For a general projector satisfying \cref{H:projector}, with fourth-order
representation $(\ViscProj\mathbf{M})_{ai}=P_{aibj}M_{bj}$, the velocity block reads
$[\mathbf{K}_{ij}]_{ab}=2\nu\alpha\,P_{aibj}$; the Frobenius self-adjointness of
$\ViscProj$ is the major symmetry $P_{aibj}=P_{bjai}$, which gives
$\mathbf{K}_{ji}^{\top}=\mathbf{K}_{ij}$, the symmetry used in
\cref{eq:AdjointFlux,eq:AdjointDifferentialOperator}.}
```

(If, in v2 or v1, this sentence would precede the *definitions* of `eq:AdjointFlux` /
`eq:AdjointDifferentialOperator`, keep the forward `\cref`s — cleveref handles them.)

### P8 — OSGS appendix edits (apply to **both** `osgs_appendix_commented.tex` and `osgs_appendix.tex`; the pedagogy edit only exists in the commented copy)

(a) In the proof of `oa:lem:definiteness`, replace

> `it is there that the full Dirichlet data enter, forcing $\bv \equiv 0$ through the deviatoric{\hyp}symmetric identity \cref{eq:devsymidentity} and $q \equiv 0$ through the pressure normalization`

by

```latex
it is there that the full Dirichlet data enter, forcing $\bv \equiv 0$ through the Korn
compatibility \cref{eq:PiKorn} of \cref{H:projector} (cf.\ \cref{lem:projfamily,%
eq:devsymidentity}) and $q \equiv 0$ through the pressure normalization
```

(the closing parenthetical warning about the seminorm degeneration is already generic —
leave it).

(b) **Commented copy only** — in the pedagogy box "Anatomy of the norm; the Korn subtlety; a
one-way comparison", replace the passage from "On definiteness: the point needing proof…" to
"…$\bv\equiv0$." by:

```latex
On definiteness: the point needing proof is that the viscous term sees only the projected
gradient $\ViscProj\nabla\bv$. Under homogeneous Dirichlet data this needs no genuine Korn
inequality: the standing assumption \cref{H:projector} supplies
$\lVert\nabla\bv\rVert\le C_{\mathrm{K}}\lVert\ViscProj\nabla\bv\rVert$ on $H^1_0$, and for
the whole projector family of \cref{lem:projinstances} this follows from the elementary
identity \cref{eq:devsymidentity} of \cref{app:Continuity} together with the pointwise
monotonicity \cref{eq:projmonotone}, with the uniform constant $C_{\mathrm{K}}=\sqrt2$
(sharp for the deviatoric--symmetric and symmetric instances). So the only field with
$\ViscProj\nabla\bv\equiv0$ in $H^1_0$ is $\bv\equiv0$.
```

---

## 5. File-by-file edit plan

Anchors are verbatim substrings of the current sources; verify uniqueness before replacing.

**E1 — `paper/article_v2.tex`**
1. Macro block in `sec:TheContinuousProblem` (anchor:
   `\newcommand{\ViscProj}{\overset{\scriptscriptstyle{\mathrm{DS}}}{\Pi}}`): apply §3
   Option R (or K).
2. P6(a), P6(b), P6(c) in the model section.
3. P7 after `eq:matrices_stationary_strong_problem`.
4. Design section, anchor `so that for $d=2$ there is nothing to drop, the retained
   expression being exact` (and the parallel sentence near `eq:StabilizationParameters`,
   anchor `we will ignore the $O(1)$ factor`): append to the latter:
   `\amend{(these statements concern the default $\DSPi$; for a general admissible projector
   the dropped factor lies in $[1,2]$, see \cref{rem:ftGenericPi})}`.
5. `sec:CommonSetting`: insert P1 after the `H:data` item. Update the preamble *comment*
   "(A1)-(A6)" → "(A1)-(A7)".
6. 3D constants footnote (anchor `These thresholds are the largest eigenvalues of a local`):
   change `the operator $\boldsymbol{u}\mapsto\nabla\cdot(\ViscProj\nabla\boldsymbol{u})$`
   to `the operator
   $\boldsymbol{u}\mapsto\nabla\cdot(\DSPi\nabla\boldsymbol{u})$\amend{, for the
   deviatoric--symmetric projector employed in the experiments}`.
7. `sec:NumericalExampleFromLiterature` (anchor
   `we take $\scriptstyle{\ViscProj} \displaystyle{=} \scriptstyle{\SPi}$`): keep the
   formula (with Option R it now reads "generic = SPi", which is exactly the statement that
   for this experiment the instance is `Ŝ`), and append after the closing parenthesis of
   `(i.e., $\scriptstyle{\DPi}\displaystyle{\equiv}\mathbb{I}$, the full symmetric gradient;
   cf.\ the discussion around \cref{eq:DBFResistanceTerm})`:
   `\amend{---an instance covered by the a priori analysis, cf.\ \cref{H:projector} and
   \cref{lem:projinstances}---}`.
8. Optional (recommended; wrap in `\amend{}`): one clause in the introduction where the
   analysis contributions are listed (anchor `our analysis shows that essentially the same
   stability and convergence properties are preserved`), e.g. insert after "preserved":
   `\amend{---for any viscous projector in a family that includes the full-gradient,
   symmetric-gradient and deviatoric--symmetric operators (\cref{H:projector})---}`.
   Similarly one sentence in the Conclusions if a parallel list exists.

**E2 — `paper/article.tex` (v1)**
1. Locate its `\newcommand{\ViscProj}` block and its model-section paragraphs (mirrors of
   P6 anchors) and apply the same macro plan and P6 edits (adapted per §4 notes).
2. Insert P1 (v1 variant) after its `H:data` item (anchor `\item\label{H:data}
   \emph{Coefficients.}`).
3. Apply P7's sentence after its copy of `eq:matrices_stationary_strong_problem` if present
   (grep the label; if v1 lacks it, skip and report).
4. Apply E1.4's design-section clause if the parallel sentence exists (grep
   `ignore the $O(1)$ factor`); otherwise skip and report.
5. Do **not** add OSGS- or DBF-comparison-related edits to v1.

**E3 — `paper/asgs_convergence.tex`**
1. Update the `[known-fragility]` header per §2.2.
2. P4 (setting sentence).
3. P2 (new subsubsection) between "Two global integration-by-parts identities" and "The ASGS
   triple norm".
4. P3 (replace `lem:definiteness` + proof). **Check**: the old proof contained the only
   in-file derivation of `eq:devsymidentity` and `\operatorname{sym}`-manipulations; after
   the replacement, grep the file for `operatorname{sym}` and `tfrac1d` — the only hits must
   be inside P2.
5. Internal reference repair: the old proof's phrase `hence $\nabla\bvh=0$ by
   \eqref{eq:devsymidentity}` is gone; confirm no other `\eqref{eq:devsymidentity}` remains
   in this file outside P2 (the sentence after `lem:coercivity`, "Because $\triplenorm\cdot$
   is a norm…", is unaffected).

**E4 — `paper/osgs_appendix_commented.tex` and `paper/osgs_appendix.tex`**
Apply P8(a) to both; P8(b) to the commented copy. Grep both for `deviatoric` afterwards: the
remaining hits must be instance-scoped prose (e.g. `oa:rem:analyzed`(ii)'s
`2\nu(\ViscProj\nabla\bu_h)\nabla\alpha` is generic and stays).

**E5 — `paper/fourier_appendix.tex`**
Apply P5(a) and P5(b). The header comment says the appendix is sympy-verified; append to that
comment a line noting that `rem:ftGenericPi` was added (statement-level, covered by the
verification script of §9, not by the sympy file).

**E6 — `paper/elemental_matrices_appendix.tex`**
Insert one scoping sentence at the top of the appendix body:
```latex
\amend{Throughout this appendix the viscous projector is the default instance
$\ViscProj=\DSPi$ of \cref{H:projector}, the one implemented and used in the numerical
experiments.}
```

---

## 6. What must NOT change

* The statements and proofs of `lem:parameters`, `lem:winv`, `lem:jump`, `lem:globalid`,
  `lem:coercivity`, `lem:continuity`, `lem:continterp`, `thm:convergence`, and of every
  `oa:*` result except the two sentences in P8. Their constants are projector-uniform; in
  particular the ranges/values in `H:coercivity`, `H:design`, `eq:coerconstant`, the
  `γ₀ = 10(1+M₂)` bookkeeping, and both error functionals `Ψ_A`, `Ψ_O` stay verbatim.
* `eq:devsymidentity` keeps its **label and its exact content** (it merely moves into
  `lem:projfamily`); the OSGS citations to it must still resolve.
* The τ formulas `eq:Tau1`, `eq:Tau2`, `eq:Tau1Final`, `eq:Tau2Final`, `eq:TauNavierStokes`.
* The Neumann trace `eq:neumann_bc` and the robustness section (its sentence "…depending on
  the particular form of the $\ViscProj$…" is already generic).
* Anything in `src/` or the numerical campaign: this is a theory-only change.

---

## 7. Optional secondary syncs (do only if asked, or list as follow-ups)

* `theory/continuity appendix/continuity_appendix.tex` (standalone source of App. C): its
  setting sentence describes the DS projector; a minimal sync is P4's sentence with the
  `\cref{H:projector}` replaced by prose ("a constant-coefficient orthogonal projection that
  is Korn-compatible on $H^1_0$"), since the standalone lacks the main-text labels.
* `theory/osgs_a_priori/*`: superseded working notes; at most add a header line pointing to
  the generalized treatment in the paper.
* `theory/paper novelties/paper_novelties.tex`: add a novelty bullet — the a priori theory
  covers a family of viscous operators (full-gradient / symmetric / deviatoric /
  deviatoric–symmetric) under a single Korn-compatibility assumption, closing the scope gap
  with the DBF comparison experiment.
* `theory/README.md`: one line noting the generalized-projector assumption.

---

## 8. Acceptance checklist (run all; report each)

1. `latexmk` from `theory/paper/` for **both** `article.tex` and `article_v2.tex`: exit 0;
   zero undefined references/citations; zero multiply-defined labels; page counts within a
   page or two of the pre-edit builds.
2. `grep -n "eq:devsymidentity" paper/*.tex` → definitions: exactly one (`asgs_convergence`,
   inside `lem:projfamily`); references: the two OSGS copies + any P8(b) mention; none
   dangling.
3. `grep -n "H:projector\|eq:PiKorn" paper/*.tex` → defined once per main file; cited from
   `asgs_convergence.tex`, `fourier_appendix.tex`, both OSGS copies, and main texts only.
4. Shared-file label audit: in `asgs_convergence.tex` and `fourier_appendix.tex`, confirm no
   new `\cref`/`\eqref` targets a v2-only label (forbidden list in §2.1).
5. Semantic audit of App. C and App. D: outside `sec:projectors`, no occurrence of
   `\operatorname{sym}`, `\tfrac1d`-type deviatoric algebra, or the words
   "deviatoric–symmetric" attached to `\ViscProj` in a *proof* (instance-scoped prose in
   remarks is fine).
6. Hypothesis-range audit: confirm `th:StabilityOSGS` / `oa:th:stability`
   (`H:mesh`–`H:spaces`) and all `H:mesh`–`H:regularity` ranges now include `H:projector`
   (they do automatically iff the item sits after `H:data`; verify the rendered (A·) numbers).
7. Cross-check the inserted constants against §9's table (√2, √(d/(d−1)), the symbol bounds
   [1,2], the d=2/d=3 eigenvalue lists). Any mismatch ⇒ stop and report.
8. Diff review: no edits outside the anchors of §5.

---

## 9. Verified numerical facts (for your cross-checking only — do not re-derive)

All of the following were machine-verified (script `verify_projector_claims.py`, shipped
alongside this brief; rerun with `python verify_projector_claims.py` if you modify any
constant):

* `I`, `Ŝ`, `D̂`, `D̂Ŝ` are idempotent, Frobenius-self-adjoint, non-expansive, and fix every
  deviatoric–symmetric tensor (d = 2, 3). The complement `I − D̂Ŝ`, the skew part and the
  spherical part do **not** fix them.
* Nested-range Pythagoras `|ΠT|² = |QT|² + |(Π−Q)T|²` for all nested pairs among
  `{I, Ŝ, D̂, D̂Ŝ}`.
* Major symmetry `P_{aibj} = P_{bjai}` and `K_ji^⊤ = K_ij` for all instances; the d = 3
  `D̂Ŝ` fourth-order tensor reproduces the paper's displayed `K_ij` entries exactly.
* On random `H_0^1` trig fields (Gauss quadrature, machine precision):
  `∫∇v:∇vᵀ = ∫(div v)²`;
  `‖Ŝ∇v‖² = ½‖∇v‖² + ½‖div v‖²`;
  `‖D̂Ŝ∇v‖² = ½‖∇v‖² + (½−1/d)‖div v‖²`;
  `‖D̂∇v‖² = ‖∇v‖² − (1/d)‖div v‖²`;
  `‖(I−D̂Ŝ)∇v‖² = ½‖∇v‖² + (1/d−½)‖div v‖²`; and `‖div v‖ ≤ ‖∇v‖`.
* Korn ratios observed ≤ claimed bounds: `√2` (Ŝ, D̂Ŝ, both d; attained by divergence-free
  fields), `√(d/(d−1))` (D̂; attained by gradient fields), `√2` (complement, d = 2, identity),
  `√3` bound (complement, d = 3).
* Failure witnesses: `skew∇(∇φ) ≡ 0` for `φ ∈ C_c^∞`; `sph∇v ≡ 0` for divergence-free `v`.
* Fourier symbols: `v᙭T_Π(k)v = |Π(v⊗k)|²`; closed forms
  `T_Ŝ = (|k|²I + kkᵀ)/2`, `T_{D̂Ŝ} = (|k|²I + (1−2/d)kkᵀ)/2`, `T_D̂ = |k|²I − (1/d)kkᵀ`,
  `T_I = |k|²I`; family bounds `½|k|²|v|² ≤ v᙭T_Π v ≤ |k|²|v|²` for all four instances;
  `eig(2T_{D̂Ŝ}) = {1×(d−1), 2−2/d}` matching `eq:ftViscEig`.
* Ellipticity constants `c_Π = min|Π(v⊗k)|`: `1` (I), `1/√2` (Ŝ, D̂Ŝ; and D̂ at d = 2),
  `√(1−1/d)` (D̂), `0` (skew: kernel `e₁⊗e₁`; spherical: kernel `e₁⊗e₂`).

---

## 10. Final report format

Report back with: (i) macro option implemented (R or K); (ii) the list of anchors that did
not match verbatim and how you resolved them (quote before/after); (iii) any v1 parallels
that were absent (per E2.3/E2.4/P6); (iv) the acceptance-checklist results 1–8; (v) a unified
diff of all touched files.
