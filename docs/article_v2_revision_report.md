# Referee-style revision report — `article_v2.tex` and appendices

**Manuscript:** *A stabilized finite element method for incompressible, inertial flows in inhomogeneous porous media* (Casas, González-Usúa, Codina, de-Pouplana)
**Files audited:** `article_v2.tex` (2 238 lines), `elemental_matrices_appendix.tex` (App. A), `fourier_appendix.tex` (App. B), `asgs_convergence.tex` (App. C), `osgs_appendix_commented.tex` (App. D), `genuine3d_table.tex`, `shared.tex`, `references.bib`; cross-checked against `article.tex` (v1) where sections are duplicated.
**Ground rules:** formal proofs assumed correct, as instructed. Everything else — statements, definitions, remarks, notation, cross-references, numerical claims, prose–table consistency, dimensional consistency, and exposition — was scrutinised. Every quantitative claim in the prose was re-derived or recomputed from the tabulated data; every design formula and asymptotic estimate outside the proofs was re-derived independently.
**Compilation:** `latexmk` completes cleanly; zero undefined references, zero undefined citations, zero multiply-defined labels, zero BibTeX warnings. (Compile required stubbing `soul`/`ulem`/`bbm` and commenting `mathabx` in the sandbox; not manuscript issues.)

**Verdict.** No P0 findings. I found no mathematical errors in the non-proof content: every design derivation, robustness estimate, and asymptotic claim I re-derived checks out, and the prose–table cross-checks are almost uniformly exact. The findings below are five P1 accuracy/consistency items, a block of P2 items dominated by bibliography corrections, and P3 polish. Fixing the P1 set and the bibliography puts the manuscript in genuinely submittable shape on the axes I was asked to audit.

---

## Part I — Findings

### P1 — statements that are wrong or materially misleading as written

**F1. Internal contradiction in the reaction-dominated 3D discussion (`article_v2.tex`, ~l. 1665).**
The sentence says the OSGS error drops "to the reference itself---1.21e-5---for the reaction-dominated $Da=10^6$ cell", but the same sentence states the L² nodal-interpolation reference as 2.85e-5. The value 1.21e-5 is *below* the stated reference — it sits at the best-approximation level (2.85e-5/√6 ≈ 1.16e-5, i.e. the familiar √(2d) nodal-vs-L²-projection gap). As written the sentence contradicts itself. Suggested fix: "dropping *below* the reference, to 1.21e-5 — at the level of the L²-best-approximation error — for the reaction-dominated $Da=10^6$ cell", or simply "dropping below the reference (1.21e-5)".

**F2. Overstatement of what the OSGS analysis covers (`article_v2.tex`, l. 887, with ll. 825, 837, 882).**
Line 887 claims: "This lagged, linearized form is the one on which the stability and convergence analysis of \cref{sec:StabilityOSGS} is carried out." That is not accurate as stated. The analysed system differs from the displayed lagged form in exactly the respects your own App. D honestly catalogues (`oa:rem:analyzed`, items i–iii): the analysis uses the τ-*weighted* projection, the *un-truncated* (not first-order) form, and the *converged* rather than lagged projection. Compounding this, l. 825 says "the analysis retains the weighted product throughout", yet the displayed projections in `eq:NonlinearResidualProjection` (l. 837) and `eq:DiscretizedNonlinearResidualProjection` (l. 882) are written with the plain L² pairing. The appendix resolves all of this correctly; the main text should not overclaim. Suggested fix for l. 887: "The analysis of \cref{sec:StabilityOSGS} addresses this system up to the modifications catalogued in \cref{oa:rem:analyzed} (weighted projection, no truncation, converged projection)"; and either display the weighted pairing in the two equations or add a clause noting the analysed variant weights the pairing.

**F3. False optimal-rate claim for P2/P1 ASGS velocity at Re = 10⁵ (`article_v2.tex`, l. 2186).**
The text asserts the stabilized P2/P1 (Taylor–Hood-interpolation) ASGS control "converges at optimal velocity and pressure rates at $Re=10^5$". The pressure claim is supported, but the velocity is not: the tables give L² velocity rate 2.41 at the finest refinement for $(10^5, 0.5)$ (optimal is 3), and H¹ rates 1.26 and 1.65 (optimal is 2) at the last tabulated refinements for $\alpha_0 = 0.5$ and $0.1$ respectively; only the $(10^5, 0.1)$ L² velocity reaches 3.00. Suggested fix: "converges, with optimal pressure rates and near-optimal (pre-asymptotic) velocity rates, at $Re=10^5$" — which is also the honest formulation given the pre-asymptotic-factor narrative of the paper.

**F4. Stale closing sentence of the DBF comparison (`article_v2.tex`, l. 2188).**
The section's final sentence — "This comparison does not isolate the effect of the element pair…" — reads as a leftover from before the P2/P1-ASGS control experiment was added two sentences earlier (l. 2186), whose stated conclusion is precisely that "the high-Reynolds gain is therefore attributable to the convection stabilization itself". As it stands, the paragraph asserts an isolation and then denies having one. Suggested rewrite: "The pairwise stabilized-versus-TH comparison alone does not isolate the effect of the element pair from that of the stabilization, which is why the stabilized Taylor–Hood-interpolation control above was included."

**F5. Notation collision on γ within the same assumption framework (`article_v2.tex`, l. 1108 / l. 1138 / l. 1622; App. C; App. D).**
γ is used for three unrelated objects: (a) the mesh quasi-uniformity constant in (A1)/`H:mesh`; (b) the OSGS design margin in (O1)/`H:design`, used pervasively in App. D (e.g. $\gamma_0 = 10(1+M_2)$, $c_1 \geq \gamma^2 C_{\mathrm{inv}}^2$); (c) the bump exponent γ(r) in `eq:Gamma`. (a) and (b) live in the *same* common-setting assumption list, which a referee will flag. (c) is localised to the manufactured solution and is lower risk. Suggested fix: rename the quasi-uniformity constant (e.g. $C_{\mathrm{qu}}$ or $\varrho$); leave γ to the design margin, which is entrenched across App. D.

### P2 — should be fixed before submission

**F6. Non-reproducible excess figure (`article_v2.tex`, ~l. 1669).**
The quoted excess 9.6e-3 cannot be reproduced from the rounded table values, which give 9.28e-3. Presumably it was computed from unrounded data. Either state that ("computed from unrounded errors"), quote ≈1e-2, or recompute from the printed values so a referee can reproduce it.

**F7. "Brackets the porosity-weighted prediction" undersells the discrepancy (`article_v2.tex`, 3D section).**
The observed OSGS/ASGS pressure ratios 3.5–4.1 are compared against $\alpha_0^{-1/2}$; at $\alpha_0 = 0.1$ that benchmark is √10 ≈ 3.16, so the observed range sits 11–30 % *above* it, not bracketing it. "Marginally above" (used nearby) is already generous. State the benchmark value explicitly (≈3.16) and characterise the range as sitting somewhat above it.

**F8. Codina–Blasco condition named but the original never cited (`article_v2.tex`, l. 1138; App. D).**
The text names a "Codina–Blasco-type" condition at (O2) and repeatedly in App. D, and `CodinaBlasco1997` exists in `references.bib` — but is never cited. Cite it at the first naming. (This also removes it from the uncited-entries list in F9.)

**F9. Bibliography corrections (all verified against publisher/author sources this session or the previous one).**
`references.bib` needs a pass. Confirmed errors:

1. `quarteroni2009numerical` — "Quarteroni, Silvia" is the book's *translator*, not a co-author. Author should be Alfio Quarteroni alone (with Sacco and Saleri if that is the intended book — check which Quarteroni text you mean).
2. `brenner2008mathematical` — missing co-author L. Ridgway Scott. The cited [Thm 4.5.11] is indeed the Brenner–Scott inverse-estimates theorem, so the content of the citation is right; the author field is not.
3. `grillo2014darcy` — "Carfagnay" and "Federicoz" are affiliation-superscript scrape artifacts. Correct authors: Alfio Grillo, Melania Carfagna, Salvatore Federico.
4. `Masud2002ASM` — author "A. K. M. Masud" is wrong; it is Arif Masud (and T. J. R. Hughes, if this is the 2002 Darcy paper — verify the co-author field against the actual paper).
5. `smith2004investigation` — stray "?" at the end of the booktitle ("…Exhibition?").
6. `hornung1997homogenization` — Hornung is the *editor* of the volume, not its author; use `editor = {Hornung, Ulrich}`.
7. `Codina2015OnSM` — a `@misc` with only a Semantic Scholar CorpusID URL. I could not locate an archival venue; the abstract identifies it as an informal summary document ("In this document we summarize some of our recent work…"). Either complete the metadata (it may be lecture notes or a SeMA-type bulletin item) or — cleaner — replace it in the intro citation list (l. 265) with `badia2010stabilized`, which is already in the .bib, is archival (CMAME 199(25–28):1654–1667, 2010), and covers exactly the claimed ground (stabilized FEM for Darcy).
8. `badia2020gridap`, `verdugo2022gridap` — `doi` fields contain full URLs; SIAM style wants the bare DOI.
9. `Hughes1998TheVM` — "Feijoo" missing accent (Feijóo); "Jean Baptiste" → "Jean-Baptiste" (Quincy).
10. `Roa2016VariationalMF` — surname parsing: should be "Bayona-Roa, Camilo Andrés" (currently renders as "Roa").
11. `skrzypacz` — data verified correct (AIP Conf. Proc. **1880**, 060010, 2017). It is a conference proceedings paper, so `@inproceedings` is more accurate than `@article` (though AIP Conf. Proc. is routinely cited as a journal); add `doi = {10.1063/1.5000664}`.
12. `liu2008instability` — missing pages/article number.
13. `Hamdan1994SinglephaseFT` — missing colon in title ("channels: a review").
14. `codina2018variational` — a book chapter cited as `@article`; use `@incollection` with the book title.
15. `bear2012introduction` — the original is the 1990 Kluwer edition; 2012 is the Springer reprint. Cite whichever you actually mean, consistently.
16. `Juanes2005AVM` — data verified correct (FEAD **41**(7–8):763–777, 2005). Cosmetic: add the issue number and replace the Semantic Scholar URL with `doi = {10.1016/j.finel.2004.11.008}` (verify the DOI string when editing).
17. `Braack2011EqualorderFE` — data verified correct (CMAME **200**(9–12):1126–1136, 2011). Cosmetic: add issue number and `doi = {10.1016/j.cma.2010.06.034}`; drop the Semantic Scholar URL.
18. `nillama2022explicit` — title spelling "stabilised" matches the published (British-spelled) title exactly; keep it. Volume/article number/year verified. Minor: publisher metadata gives the surname as the compound "Balazi Atchy Nillama" (rendering "L. Balazi Atchy Nillama"), whereas the current field order renders "L. B. A. Nillama". Consider `author = {Balazi Atchy Nillama, Loic and Yang, Jianhui and Yang, Liang}`.
19. Purge before submission: SIAM-template leftover entries (`KoMa14`, `siam`, `Hi14`, `PeKoPi14`, `WoZhMeSh05`, `Ne03`, `clawpack`, `AMSMSC2010`, `La86`, `MiGo04`, `GoVa13`, `CalcI`, `amsmath`, `shortmath`, `pgfplots`) and the genuinely uncited research entries `Codina2004ApproximationOT` and `auriault2005filtration` (unless you intend to cite them; `CodinaBlasco1997` is handled by F8).

**F10. Spelling stragglers against the manuscript's (SIAM/American) convention.**
The manuscript is dominantly American; residual British forms to unify: `asgs_convergence.tex` l. 624 "renormalisation" → "renormalization"; `fourier_appendix.tex` l. 51 "analysed" → "analyzed"; `osgs_appendix_commented.tex` l. 741 "neighbour-size" → "neighbor-size"; `article_v2.tex` l. 1108 footnote "neighbouring" → "neighboring"; and l. 1098 "modelled on" vs l. 265 "modeled" — unify to "modeled".

### P3 — polish, optional

**F11. Borderline prose approximations (all defensible, flag only).** "≈1.9 in L², ≈0.9 in H¹" for ASGS Q2 pressure (actual 1.81–1.90, 0.79–0.93); "≈2 in L²" for ASGS P1 pressure at $\alpha_0=0.05$ (actual 1.76–1.79); "about twenty times" for the Q2 H¹ plateau (actual 23–24); "approach" where the P2 rates in fact slightly exceed the benchmark. None is wrong at the stated precision; tighten only if you want referee-proofing.

**F12. Dimension-specific constant in App. B (`fourier_appendix.tex`, l. 146).** "Ignoring the coefficient 4/3 of $\tau_{\nu,1}^{-1}$" hard-codes d = 3, while the amended derivation earlier uses the general $(2-2/d)$. Harmonise ("the coefficient $2-2/d$, equal to 4/3 in three dimensions").

**F13. Small logical gap in App. D (`osgs_appendix_commented.tex`, l. 1447).** "Valid since $c_1 \geq 1$ under `H:design`" — `H:design` gives $c_1 \geq \gamma^2 C_{\mathrm{inv}}^2$, which implies $c_1 \geq 1$ only if $C_{\mathrm{inv}} \geq 1/\gamma$. Harmless in practice (γ ≥ γ₀ ≫ 1), but either add "since we may assume $C_{\mathrm{inv}} \geq 1$" or use $\max(c_1, 1)$.

**F14. Minor symbol reuse, distant contexts.** η as Young parameter (coercivity proof) vs radial coordinate in `eq:Gamma`; ψ as interpolated field in `eq:InterpolationError` (l. 1239) vs smoothing error ψ(h) in App. D. Both are far apart and locally defined; note only.

**F15. Housekeeping before submission.** `shared.tex` retains `\usepackage{lipsum}`; `\headers` is commented out (SIAM production wants it); caption wording is inconsistent across the four 2D tables ("worst-case rates of the analysis" vs "theoretical convergence rates" — pick one); large editorial comment blocks remain (`article_v2.tex` ll. 317–334, 848–862, plus v1/v2 sync notes); one `\sout` survives inside a comment (l. 111). The duplicated `sec:ViscousProjector` in v1 vs v2 differs only in the flagged "method"→"methods" sentence, so the files are in sync.

**F16. Unverifiable-from-tables accuracy factors (`article_v2.tex`, l. 2186).** The stabilized-TH accuracy factors 1.2/1.5 rely on TH errors at N = 160, which the tables do not report (only N = 320 for the FME). They are consistent with back-extrapolation at the observed rates, but a reader cannot check them. A one-line footnote ("TH errors at N = 160 obtained by extrapolation at the tabulated rates" or by adding the row) closes the gap.

---

## Part II — Verification inventory (checked and found correct)

For your records, the following were independently re-derived or recomputed and found sound. I list them so you know what has *positive* verification behind it, not merely absence of complaint.

**Formulation and design (main text §§2–5).** Dimensional consistency throughout, including the f-scaling of the dimensionless momentum equation. The elemental matrices $K_{ij}$, $A_c$, $A_f$, $S$ against the momentum/mass equations; the adjoint operator and the Neumann boundary term. The full Fourier design chain: the factor $2-2/d$, the design window [1, 2], $\tau_c$, $\tau_b$, $\tau_\sigma$, $\tau_{\nabla\alpha}$, their assembly into `eq:Tau1`/`eq:Tau2`, and the factor 5. The Galerkin coercivity identity, the $\tilde\sigma_\alpha$ identity, and the asymptotics in `eq:GeneralAsymptoticBehaviourOfParameters`.

**Robustness estimates (§6).** All three regime estimates re-derived from scratch, including the $\alpha_0$ exponents and mitigation factors: `eq:DominantReactionOSGSVelocity` via $(\alpha_K/\nu)\tau_2 = 1 + (c_2/c_1)Re_h$; `eq:DominantConvectionEstimate`; `eq:DominantConvectionXTermEstimate` (the $1 + U/\|a\|$ factor); `eq:DominantPressureGradientXTermEstimate` (the $\|a\|U/P + 1$ factor); the $P \sim U^2$ convection claim; the `eq:OSGSCollected` coefficient bound ≥ 1/8 (numerics re-audited: $P_1 \geq 0.14$, $P_2 \geq 0.21$); and the `eq:EpsilonRef` chain. The manufactured solution satisfies $\nabla\cdot(\alpha u) = 0$ as claimed.

**Numerical sections (§7).** Every prose-versus-table cross-check in the 2D, 3D, genuine-3D and DBF-comparison sections was recomputed: growth factors 5.0/11.6; pressure ranges 8.3–12.4 and 10.2–18.8; ASGS 1.3–2.0 vs OSGS 3.5–4.1; $Re_h \approx 3.1\times10^3 / 1.3\times10^3$; $Da_h \approx 9.8$; the ×20 plateau; 3D pairs 1.85/4.55e-2 vs 1.37/2.87e-2; genuine-3D P1 rates 1.88–1.95/0.98 and P2 3.53–3.55/2.34–2.35; OSGS P2 H¹ 2.00 vs 1.77; H¹ pressure 1.22–2.97. In the DBF section: $Da(\alpha_0) = C_a((1-\alpha_0)/\alpha_0)^2 + C_b(1-\alpha_0)/\alpha_0$ gives ≈2 and ≈40 as claimed with $C_a = 0.30$, $C_b = 1.75$; velocity growth factors 3.5–4.0 (P1) and 6.4–7.1 (P2) track the interpolation; the "factor of six" L² pressure growth is 6.2–6.5; "some 2.8×" against $\alpha_0^{-1/2} \approx 2.2$ checks (2.77–2.91); TH pressure sits exactly on the interpolation benchmark 9.84e-6 at slope 2.00; equal-order is optimal at $(10^5, 0.5)$; the non-converged velocities are O(1) as stated; and the 1.2/1.5 stabilized-TH factors are consistent with back-extrapolated TH at N = 160 (see F16).

**Appendix A (elemental matrices).** All displayed terms `eq:ViscousTerm` … `eq:ReactionTerm` re-derived; `eq:StabilizationLVLU` fully re-derived, including the adjoint $q\nabla\alpha$ cancellation, the sign pattern, and the amended factor 2 on the $\nu D S \Pi \nabla v \nabla\beta$ term; $\bar\varphi = -\pi_{\mathrm{mass}}$; the iterative-penalty remark; all sampled submatrix entries ($G_S$, $D_{\nu D}$, $V$, $R_\sigma$, $A_A/A_L/A_C/A_{G\beta}/A_{D\beta}$, $L_A$, $C_A$, $C_C$, $G_{\beta A}$); and the 40-term count (36 momentum + 4 from $\tau_2$).

**Appendix B (Fourier).** `eq:ftViscClosed` in general d; the eigenpairs; the `ftBdesign` product/ratio conditions proven equivalent to the main-text Λ-metric spectral-radius criterion; `ftTauGradA`, `ftTauSigma`, and the assembly; `rem:ftGenericPi` ($T_\Pi \geq \tfrac12 |k_0|^2 I$ by monotonicity, lower bound $2c_\Pi^2$, consistency with `cor:ProjectorASGS`).

**Appendix C (ASGS convergence).** Strong operator and $X_G$ match the main text; the $B_{\mathrm{stab}}$ expansion; the four equivalent forms of $\tilde\sigma$; the $\delta_\alpha = 1 + C_{\nabla\alpha}$ derivation; (P1)–(P5); `lem:winv` with $\bar C_{\mathrm{inv}} = \sqrt{d\delta_\alpha}\,C_{\mathrm{inv}} + C_{\nabla\alpha}$ and the abuse-of-notation remark; the jump lemma's consistency with `H:jump`; the triple norm; definiteness; the coercivity collection identical to the main-text `eq:CollectedCoercivity`, the optimal η = 4 via $(\eta-4)(t\eta+1) = 0$, $C_{\mathrm{coer}}$, and the equivalence $c_1 > 2\xi\bar C^2$ (ξ > 2) ⟺ $c_1 > 4\bar C^2$; the sharpened elemental threshold $c_1 > 2(1+\sigma\tau_1)\bar C^2$ consistent with the 3D section; the continuity lemma (including the $\alpha_K$-sharpened form) and its 18-term ledger against (42)–(59) of the cited Codina 2001 paper; the interpolation setup, mean-shift, and the $\ell^2 \subset \ell^1$ step with its $h^{-d/2}$ caveat; and the convergence theorem's match with `th:Convergence`.

**Appendix D (OSGS).** Projections as τ-weighted L² onto unconstrained spaces, consistent with the main text; the annihilation lemma; the honest analysed-vs-implemented ledger (`oa:rem:analyzed`); (P1)–(P6), (K1)–(K3); `oa:eq:Tau2Expanded`; the patch lemma ($\delta_\alpha^2$ porosity comparability, advective-over-viscous floor) and its grading remark against (O3); the smoothing lemma ψ(h); the best-approximation lemma; the OSGS norm, definiteness, and the one-way norm comparison; the stability theorem's constants ($\gamma_0 = 10(1+M_2)$, $M_2 = 1 + C_{\nabla\alpha}/C_{\mathrm{inv}}$, $\psi_0 = \min\{1/8, \beta_0^2/16\}$) and the "$c_1 \gtrsim 400 C_{\mathrm{inv}}^2$, $c_2 \gtrsim 20 C_{\mathrm{inv}}$" honesty paragraph, all matching the main text; the Step-4 collected display matching `eq:OSGSCollected` exactly (with $M_1 = (1+\psi)/\gamma$); the $\Psi_O/\mathcal{E}$ functionals; the consistency lemma ($S_2$ vanishing at ε = 0); the (I1)–(I7) interpolation ledger, including the (I2) split yielding $(c_1 + Da_h)$; the convergence theorem's match with `th:ConvergenceOSGS`; both corollaries re-derived (including $\sigma^{-1}(c_1+Da)(\alpha_K/h)^2 = (1+c_1/Da)\,\alpha_K/\nu$); the mechanism/dispensed/method-II/robustness remarks ($\ell_\sigma$ screening length, $Da_h = (h/\ell_\sigma)^2$); and the numerics remark's factor-two L² claim (1.65 and 2.22 from the tables), the 3.5 H¹ figure, and $(1 + Da_h/c_1)^{1/2} \approx 1.9$ at $Da_h \approx 10$, $c_1 = 4$.

**Cross-file conventions.** The "never $|||\cdot|||_S$" convention is respected in the appendices; `\Xh`/`\Dah` usage is consistent; the abstract's claims (coercivity for ASGS, discrete inf-sup for OSGS, parameter-explicit estimates, porosity-weighted norm, the beyond-hypotheses caveat for the nonlinear experiments) all match the body; the intro's citation load is appropriate once F9.7 is resolved.

---

## Part III — Suggested order of operations

Fix F1–F5 first (they are the only items a referee could call *errors of statement*), then the F9 bibliography pass and F8 in one sitting, then F6–F7 and F10. The P3 items can ride along with whatever final polish pass precedes submission; F15's template leftovers (`lipsum`, commented `\headers`, editorial blocks) should be cleared in any case because SIAM production will query them.
