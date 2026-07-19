# Pre-submission checklist — `article.tex`

**Purpose.** The final read-through checklist before submitting *"A stabilized finite element method for
incompressible, inertial flows in inhomogeneous porous media."* Built 2026-07-18 from the living docs
(`paper-revision-plan.md`, `review_numerics_vs_theory.md`, `open-questions.md`, `pending-tasks.md`,
`findings.md`, `theory-code-map.md`, `part_i_erratum.md`), the newest results, and a full theory-vs-numerics
re-derivation of the α₀-exponent estimates.

Each item: **severity** (🔴 blocker / 🟠 important / 🟡 nice-to-have), **status** (open / verify / likely-done),
and the source. Work the 🔴 first. "verify" = believed done but must be eyeballed in the final PDF/data.

---

## 0. The α₀-exponent inconsistency in §6 — RE-DERIVED, ✅ APPLIED 2026-07-19

This is the item flagged in `paper-revision-plan.md` (S6-4) plus a second, previously-unflagged slip found on
re-derivation. Full working in the conversation log; summary and decision below.

**✅ APPLIED (2026-07-19).** All §6 estimates re-derived independently and the weighted form adopted throughout
(displays now print the worst-case `α₀^{-1/2}`; `rem:WeightedVsUnweighted` compressed; prose at 1021 fixed;
S6-1 closed in the numerics prose). Build green (66 pp / 722 newlabels / 0 unresolved). The §6 rewrite is
justified on **internal consistency with the proven weighted theorem (App. C `eq:convergence`)**, *not* on
empirical discrimination (the "data discriminate in favour of the weighted form" claim is logically unsound —
one-sided upper bounds; paper keeps the "cannot discriminate — both upper bounds" caveat). **One correction to
this checklist's own prescription:** the reaction pressure-gradient `eq:DominantReactionPressureGradientEstimate`
keeps its **outer `1/α₀`** — that factor is *legitimate*, not the coarse double-count: its pressure-gradient
LHS control coefficient itself carries `α₀^{1/2}` (τ₁∼1/σ, weakest control), so isolating `‖∇e_p‖` costs a
genuine further `α₀^{-1/2}`. The fix there is the **inner** `(α_∞/α₀)^{1/2} → α_∞^{1/2}` (which also removes a
mixed `α₀^{-3/2}` vs `α₀^{-1}` split *inside that one line*). A one-line explanation was added at 1081.

**Root cause.** In the working norm, the porous-divergence term carries the weight `τ₂ = h²/(c₁α_Kτ₁,ₙₛ) ∝ 1/α_K`
— the *only* term whose weight **grows** as porosity drops, so `min_K τ₂^{1/2}` is attained at `α_∞=1` (not at
`α₀`). Against the α₀-based normalization `N`, the divergence-control coefficient is therefore deterministically
**`α₀^{-1/2}`** (×`(1+Re_h)^{1/2}` where not absorbed), in *every* limit.

| Limit | correct porous-divergence LHS coeff | paper prints | status |
|---|---|---|---|
| viscous ([article.tex:1019](../theory/paper/article.tex#L1019)) | **α₀^{-1/2}** | ~~`1/α₀`~~ → `α₀^{-1/2}` | ✅ FIXED (was S6-4) |
| convection ([article.tex:1043](../theory/paper/article.tex#L1043)) | **α₀^{-1/2}** | ~~`1`~~ → `α₀^{-1/2}` | ✅ FIXED (was newly-flagged) |
| reaction ([article.tex:1067](../theory/paper/article.tex#L1067)) | `(1+Re_h)^{1/2}α₀^{-1/2}` | `(1+Re_h)^{1/2}/α₀^{1/2}` | ✅ already correct (unchanged) |

The reaction subsection is already right and is exactly what the other two reduce to. This LHS-coefficient
error is **independent of the weighted/unweighted RHS choice** — it is fixed by the working norm and `N` alone.

**The RHS prefactor (weighted vs unweighted).** The theorem actually *proved* (App. C, `thm:convergence`
`eq:convergence`) is the **porosity-weighted** form. Its correct normalization gives the elementwise weight
**`√(α_K/α₀) ∈ [1, α₀^{-1/2}]`** (=1 on the low-porosity plateau), **never `1/α₀`**. The current `1/α₀`
prefactors descend from the *coarser* `α_K→1` bound and are strictly weaker than the paper's own theorem; they
also produce **mixed exponents inside one estimate** (e.g. `eq:DominantReactionVelocityGradientEstimate` has a
`1/α₀` term next to an `α₀^{-1/2}` term).

**DECISION (adopt the weighted form throughout §6).** It is the only presentation "obviously fully consistent
with the rest of the theory": every estimate then descends by one normalization step from the proven theorem,
with no worst-case-then-walk-back detour and no mixed exponents. Concretely:

- ✅ Fixed the porous-divergence LHS coefficient to `α₀^{-1/2}` at 1019 (S6-4) **and** 1043 (convection analog).
  Reaction already correct.
- ✅ Replaced the coarse `1/α₀` RHS prefactor by the worst-case `α₀^{-1/2}` in the displayed estimates
  (1019, 1027, 1031, 1043, 1047, 1052, 1056, 1068, 1073); the elementwise `√(α_K/α₀)` (=1 on plateau,
  ≤`α₀^{-1/2}`) is defined once in the compressed remark. Displays use the worst-case constant `α₀^{-1/2}`
  (a global norm inequality cannot carry a free elementwise `K` index). **1079 is the exception** — see the
  APPLIED banner: its outer `1/α₀` is legitimate and stays; only the inner `(α_∞/α₀)^{1/2}→α_∞^{1/2}` changed.
- ✅ Fixed `eq:DominantReactionVelocityGradientEstimate` (1073): first term `1/α₀ → α₀^{-1/2}` so both terms
  are `α₀^{-1/2}` (no mixed exponents).
- ✅ Collapsed `rem:WeightedVsUnweighted` to the `√(α_K/α₀)` definition + worst-case `α₀^{-1/2}` + the coarse
  `1/α₀` contrast + the honest "cannot discriminate — both one-sided upper bounds" caveat.
- ✅ Fixed the prose at 1021: the porous-divergence control is **α₀-optimal** (same `α₀^{-1/2}` on both sides ⇒
  no loss as α₀→0); the velocity/pressure gradients degrade at most `α₀^{-1/2}`, none on the plateau. Also noted
  the reaction pressure-gradient `1/α₀` exception up front so the reader is prepared for it.

**Reconciliation with the numerics** (why the weighted form is what the data show): velocity method factor ≈1.00
across the α₀-sweep (interpolation error concentrates on the plateau, where `√(α_K/α₀)≈1`); the one residual
method effect, ℙ₁ u L², is 1.96 (ASGS)/3.48 (OSGS) at the weighted worst case `√10≈3.16` (OSGS marginally over
— review S6-1, a real but small OSGS constant, not a rate loss); viscous pressure grows ×8–16, inside the
weighted windows [5, 15.9]/[11.6, 36.6], far below the unweighted [50, 116]. The unweighted `1/α₀` would
predict ×10/×50/×116 — refuted as *sharp*.

**Caveat (kept honest):** these α₀ exponents are exact and independent of the enlarged inverse constant
`C̄_inv`; `C̄_inv` only affects the `c₁ > 2ξC̄_inv²` coercivity margin (a separate axis, §1 below), not any α₀ power.

---

## 1. Theory / a-priori claims

- 🟠 **verify — the c₁ story is told truthfully and consistently.** Paper: `c₁=4k⁴` in 2D
  ([937](../theory/paper/article.tex#L937)), `c₁=16k⁴` in 3D "just below the elementwise Kuhn threshold"
  ([1450](../theory/paper/article.tex#L1450), [1680](../theory/paper/article.tex#L1680)), `c₁` element-dependent
  via `C_inv`. Ensure **no withdrawn framing survives anywhere** in the paper or shipped notes: "Gridap↔paper
  discrepancy", "c₁ masks a bug", the clean-room/NumPy element-family verdict (all refuted — Kratos runs the
  full subscale optimally at paper c₁ on tets). Src: `findings.md §3`, `theory-code-map.md §2.5`.
- 🟠 **open — S45-3: Lemma 1 (Stability, [943](../theory/paper/article.tex#L943)) hypotheses are insufficient
  for its own proof.** It omits `eq:SmallPorosityGradient` (resolved porosity) and mesh-nondegeneracy, which its
  proof (~line 880) and App. C `prop:stability` use; the convergence theorem
  ([960](../theory/paper/article.tex#L960)) lists them but the stability lemma does not. Coordinated Lemma 1 +
  Lemma 2 hypothesis surgery. Src: `paper-revision-plan.md §7 (S45-3)`.
- 🟠 **verify — divergence ledger walk.** Confirm each documented code↔paper divergence is honestly stated or
  intentionally-silent-and-defensible: (a) `(1/α)∇·(αa)v` omitted from the adjoint (line ~800 justifies);
  (b) positive-sign convective adjoint for the `A²−B²` symmetry; (c) `τ₁/τ₂` simplified forms dropping `εh²`
  and the `C_α` gradient term (lines 762/764 justify); (d) OSGS projection on unconstrained `V_free/Q_free`;
  (e) reaction-projection trim only for constant σ. Src: `theory-code-map.md §2.1–2.7`.
- 🟠 **verify — §3.1 projection-trim sentence (D5).** Must be qualified: reaction terms excluded from the
  orthogonal projection **only when σ is constant** (orthogonal component exactly zero); for the variable-σ DBF
  runs of §7.3 the **full** residual is projected. Code is correct
  (`ProjectResidualWithoutReactionWhenConstantSigma`, OSGS-only, `Constant_Sigma`-gated). Src: review D5.
- 🟡 **open — S6-3: `Da = Da_h L²/h²`** ([1177](../theory/paper/article.tex#L1177)) is imprecise under the
  elementwise `α_∞` convention — `Da/Da_h = (L²/h²)(α_K/α_∞)`. State the convention once or write the exact
  identity. Src: `paper-revision-plan.md §6/§7 (S6-3)`.
- 🟡 **verify — best-approximation claim scoped to H¹ only** ([1169](../theory/paper/article.tex#L1169)): keep
  "best-approximation-exact in the H¹-seminorm for both elements" with the `√6` ℙ₁-L² caveat; ensure the
  falsified "and in both norms for the biquadratic one" (ℚ₂-L² gap 0.16–2.5%) does **not** survive. Src: §0.2c.
- 🟡 **verify — §7 scope sentence (D7)** and the `C_inv` vs weighted `C̄_inv` convention (D10, `rem:winvconst`
  reworded — no double-count; `\Cinva` macro is USED 16× by the appendix, do **not** delete). Src: review D7/D10.

## 2. Numerics / results

- ✅ **CLOSED (2026-07-19) — verified against the on-disk DBs; one final recheck at the very end (below).**
  E1 was the risk that a 3D-table slope was computed partly from an interpolant (a stalled solve returning the
  exact-guess field). Checked directly against the four canonical DBs
  (`results/k{1,2}/TET/{structured,nested_red}/`, re-run 2026-07-14/15, after the direct-LU fix): **zero
  `success=false`** in any ladder, and every finest cell moved off the interpolant (none byte-identical). The
  honest-exit gate demonstrably fires — the pre-fix backup (`pre_p2_LU.bak`, 2026-06-20) recorded the
  irregular-ℙ₂ stalls as `success=false` (`iters=18`, max-iter thrash); post-fix the same cells converge in
  2–5 iters. The stall/interpolant risk was **irregular-mesh only**; structured Kuhn (ℙ₁ and ℙ₂) never carried
  a `success=false` in any snapshot, so no published slope is part-interpolant. Src: on-disk DBs;
  `paper-revision-plan.md §0.4b/E1/§8`; `findings.md §3`.
  - 🟡 **Final recheck — do this at the very end, right before submission** (and after any 3D re-run). Confirm
    on the four canonical DBs: (i) no `success=false` anywhere; (ii) each finest cell's error ≠ its
    interpolation reference — in particular the two irregular finest cells a stall would camouflage:
    K1-nested_red ℙ₁ ASGS (`iters=1`, err 1.10e-3 = 1.62× interp 6.82e-4 → moved) and K2-nested_red ℙ₂ OSGS
    (`iters=5`, err 2.02e-4 = 1.005× interp 2.01e-4 → on the velocity floor but genuinely iterated). Both pass
    today; re-confirm they still do.
- 🟠 **verify — C7 "1.29 triple"** ([1462](../theory/paper/article.tex#L1462)): the footnote claims OSGS
  pressure-H¹ FMEs 1.292/1.289/1.290 on three different mesh/order rows. Check the three underlying values
  against the certified DB (genuine mesh-independent saturation vs transcription coincidence). Src: C7/§8.
- 🟠 **verify — S6-1 closed.** OSGS ℙ₁-L² method factor (3.5–4.1) exceeds `α₀^{-1/2}=3.16` in most viscous
  rows; the corrected prose ([1171](../theory/paper/article.tex#L1171)) must not contain any "of the order of
  the weighted prediction" overclaim for OSGS. Ties to §0 above. Src: `paper-revision-plan.md §0.2a`.
- 🟠 **verify — Cocquet magnitude honesty.** `tab:CocquetMMSL2/H1` must trace to certified runs; **no** claim
  that Gridap FMEs reproduce the historical Kratos magnitudes (they are ~30–300× off while rates agree; the
  convergence cap is mesh-topology, not the formulation; modified-corner is falsified). Src: `findings.md §6`.
- 🟠 **verify — no Kratos magnitude-reproduction claim in 2D either.** Normalized FMEs are ~3–12× larger than
  the paper's Kratos values (norm-dependent), an open code-vs-code calibration question; rates agree. The paper
  now says experiments were run in Gridap ([1124](../theory/paper/article.tex#L1124)). Src: `open-questions.md §2`.
- 🟡 **verify — 2D tables** match the certified sweep computed with the **tight k=2 gate**
  (`eps_tol_momentum=1e-9`, `ftol=1e-12`) that recovers finest-segment O(h³)/O(h²); read per-pair ratios, not
  `analyze_results`' over-flagging verdict. Src: `findings.md §1`.
- 🟡 **verify — OSGS reaction-dominated velocity gap** reported as a pre-asymptotic, Da_h-governed transient
  that recovers (rate 1.60 by N=640), **not** an order ceiling ([1177–1179](../theory/paper/article.tex#L1177)).
  Src: `findings.md §4`.

## 3. Figures / tables

- 🔴 **verify — cell-by-cell audit of the auto-transcribed 3D tables (commit 638a298)** against the certified
  DB: every slope, every FME in `tab:3DL2`/`tab:3DH1`, including the interpolation-reference rows. `article.tex`
  does **not** `\input` the generator, so any transcription slip ships silently. Src: `paper-revision-plan.md §0.4d`.
- 🟠 **open — wire tables to the generator (D4c)** OR record a completed cell-by-cell reconciliation. Hand-copied
  tables are the systemic drift mechanism behind the C7- and E1-class risks. Src: `paper-revision-plan.md D4c`.
- 🟡 **verify — all interpolation-reference rows** (12 in 2D, 8 in 3D, plus Cocquet ℙ₁/ℙ₂) match
  `interp_reference.h5` to printed precision, use the two-finest-mesh slope rule stated in the caption, and share
  the finest mesh + the shared `calculate_normalized_errors` functional with the data rows. Src: §5/§8/B1.
- 🟡 **verify — first table caption (`tab:Linear2DL2`)** defines the parenthetical "(theoretical rate)"
  convention + the regime-dependence caveat (parentheticals = viscous worst case). Src: review D4.
- 🟡 **open — results-section figures.** `bump_plateau.pdf` is referenced
  ([1151](../theory/paper/article.tex#L1151)) but `open-questions §4` records a standing `\Guillermo{Add figures}`
  — convergence results are tables-only. Author decision: add convergence-plot figures or confirm tables-only;
  verify the one figure renders (caption/axes/legend). Src: `open-questions.md §4`.

## 4. Editorial / prose

- 🔴 **open — finalize ALL review markup.** True counts (verified): `\Guillermo`=14, `\Joaquin`=4 (=**18**
  spans), `\amend` used **279×**. Two distinct tasks: (i) flatten the 279 `\amend` + 18 author spans (redefine
  as `{#1}` for the final build); (ii) **resolve the macros that contain open TODOs**, not just decolor them —
  e.g. `\Guillermo{CITATIONS FOR THIS STRATEGY}` ([~659](../theory/paper/article.tex#L659)) will **not** be
  caught by "0 undefined citations" (there is no `\cite` yet) — and the "Add figures" note. Src: critique;
  `article.tex:115–122`.
- 🟠 **open — `supplement.tex` / `\externaldocument{supplement}`** ([22](../theory/paper/article.tex#L22)):
  SIAM boilerplate (`\lipsum`, `thm:bigthm`). Write the real supplement or remove the line and any `\cref` to it.
- 🟡 **verify — `C_α` symbol clash (D11)** resolved (appendix constant → `C_{∇α}`; main text/Fourier keep `C_α`).
- 🟡 **verify — Part I erratum fix in the submitted appendix**: phantom `I`/`G_β`/`D_β` removed, Neumann load
  `V_T` restored; `assembly_consistency_verification.py` green (110/110); the `\amend V_T` marker accepted for
  final. Note: the display should read `P + G_αP` (Galerkin flux block renamed to break the `G_P` collision,
  commit 8a644d2) — update `part_i_erratum.md §3/§5` accordingly. Src: `part_i_erratum.md`.
- 🟡 **open — author decision on `centered_encoding.tex`** (self-describes "not yet merged"; defines the
  error-normalization convention the tables use). Merge or keep as companion note. Src: `open-questions.md §4`.
- 🟡 **verify — notation D1/D2** (`α_K` "elemental minimum" → `α_{∞,K}` supremum at
  [842](../theory/paper/article.tex#L842); `α` vs `α_K` unified across the §6 limits) and the Fourier appendix
  A16/S45-1 fixes (general-`d` `K_ij`, `(2−2/d)` noting =1 for d=2). Src: review D1/D2, plan A16.
- 🟡 **open — companion-note fixes (A17)** if the notes ship: `osgs_reaction_note` stray `α_K` in `τ₁`;
  `velocity_floor_regularization §4` corrected `u_base` claim. Src: `paper-revision-plan.md A17`.

## 5. Build / LaTeX

- 🔴 **verify + commit — the `\@ifpackageloaded{lineno}{}{\allowdisplaybreaks}` guard**
  ([article.tex:56–64](../theory/paper/article.tex#L56), currently uncommitted). Without it, `[review]` +
  `allowdisplaybreaks` collapses display math to ~22 pp with all-`??` refs on TeX Live 2023/macOS. Src: MEMORY
  `paper-build-fragilities`.
- 🔴 **verify — clean `latexmk` in BOTH review and final (review-off) mode**, each with **0 unresolved refs and
  0 undefined citations**. Healthy final build = **66 pp / 722 newlabels / 0 unresolved**. Reconcile the stale
  "43 pp" in `theory/README.md:11` and `open-questions.md §4`.
- 🔴 **open — produce the submission build with review markup OFF** (`\documentclass` without `[review]`,
  [article.tex:2](../theory/paper/article.tex#L2)) so `lineno` is off, `allowdisplaybreaks` activates,
  pagination is correct, and colors resolve. Verify: no line numbers, no colored text in the final PDF.
- 🟠 **verify + commit — the `latexmkrc` fix** (uncommitted, +47/−10): works around latexmk 4.79 not expanding
  `%B` in `$aux_dir`, uses `@ARGV` basename, keeps SyncTeX on. Src: MEMORY `paper-build-fragilities`.
- 🟠 **open — commit the two modified files** (`theory/paper/article.tex`, `theory/paper/latexmkrc`) so the exact
  healthy-build sources are under version control. Src: `git status`.

## 6. Provenance / reproducibility

- 🔴 **verify — the entire 3D section traces to a certified, committed config+result at `c₁=16k⁴`.** The
  original `c1x4` raw data was lost (gitignored `results/`); D1c added `c1_multiplier` to schema/config, D3c
  committed the nested_red base mesh. Confirm every 3D number came from the certified re-run through the official
  path (not the lost data), and that **both** the regular-Kuhn and irregular drivers/configs are committed
  (`pending-tasks §6e`). Src: `paper-revision-plan.md §0.4c/C1r/D1c/D3c`.
- 🔴 **verify — every reported number (2D/3D/Cocquet) via the official test path**, single canonical results
  leaf, no forked `*_corner` side-DBs merged, no plotter/analyzer reading non-official files. `c₁=16k⁴` must have
  a production config representation (`get_c1_c2` is dimension-blind `4k⁴`; `16k⁴` arrives via `c1_multiplier`).
  Src: CLAUDE.md reproducible-results; official-results-path rule.
- 🟠 **verify — 3D mesh reproducibility (D3c):** committed `nested_red_base_lc0.200_alg1.msh` + gmsh 4.9.3
  provenance; `load_or_build_base_mesh` prefers the committed file; family regenerates deterministically
  (425→3400→27200); regular Kuhn family is code-generated. Src: `paper-revision-plan.md §0.4c/D3c`.
- 🟡 **likely-done — `ε_M`/`ε_C` persisted per mesh** (D6c, applied, inert). Src: `paper-revision-plan.md D6c`.

## 7. References / citations

- 🟠 **open — the c₁-derivation footnote** "a detailed derivation will be reported separately"
  ([1450](../theory/paper/article.tex#L1450)). The derivation exists
  (`theory/numerical_constants/c1_dimension_note.tex` → validated `element_c1.jl`, reproduces the `C_inv²` table
  incl. Kuhn 214). Cite it as a preprint/report/companion, or defend the forward-reference. Src: review D8.
- 🟠 **verify — 0 undefined citations** in the final build; all cited works resolve in `references.bib` (codina
  2001/2008/2018, villota2019, cocquet2021, badia2020/verdugo2022 gridap, codina1993, nillama2022,
  hughes2007). Src: `theory/README.md`.

## 8. Formal proof (Coq)

- 🟠 **verify — the paper's stability condition matches the machine-checked margin.** `StabilityAlgebra.v`
  proves the *sharp* positivity threshold is `c₁ > ξC̄_inv²` (a factor 2 below the paper's `c₁ > 2ξC_inv²`
  at [932](../theory/paper/article.tex#L932)/[943](../theory/paper/article.tex#L943) — so the paper's condition
  is sufficient, not necessary); `C_stab_margin` needs `c₁ > 2ξC̄_inv²` for a `C̄_inv`-free floor, with the
  **weighted** `C̄_inv = √(dδ_α)C_inv + C_α` the relevant constant. Reconcile with "16k⁴ sits just below the
  Kuhn threshold". Src: `findings.md §8`; `coq_coverage.tex`.
- 🟠 **verify — if any `.v` was touched**, run `./run_all.sh`: ZERO `Admitted`, ZERO `Axiom`, `Print
  Assumptions` returning only the 3 stdlib axioms. Any "machine-checked" claim must be scoped to the 3-of-4
  non-vacuity-witnessed theorems (`abstract_stability/continterp/convergence`) with `abstract_continuity`'s
  witness gap disclosed. Src: `findings.md §8`; CLAUDE.md Coq gate.
- 🟡 **verify — amendment F8** (the `eq:winv-conv` label moved to the convective line; `eq:winv-gradp` added; 4
  call sites re-pointed) is in the submitted appendix. Src: `AUDIT.md F8`.
- 🟡 **verify — no over/under-claim of implemented-vs-analyzed τ₂** (S45-2: `eq:Tau2` full vs `eq:Tau2Final`
  analyzed+implemented; Coq `abstract_convergence_implemented` covers it) and that σ=0 (pure NS) is admissible.

## 9. Reviewer-demand gaps (from the adversarial critique — add these)

- 🟠 **open — code/data-availability statement is ABSENT.** No `github`/`zenodo`/`availab`/`reproducib`
  statement in `article.tex`, though the thesis is reproducibility and there is a large public code + Coq base.
  SIAM (SISC RCR/badges; SINUM/SIMAX) expects one. Add a code/data-availability statement.
- 🟠 **verify — the conclusion's TH-vs-VMS claim** ([1680](../theory/paper/article.tex#L1680)): "the method
  remains convergent in the convection-dominated regime in which the unstabilized Galerkin Taylor–Hood velocity
  does not" must be backed by data in the paper and consistent with `cocquet-form-mms-status.md §4.3` (TH
  velocity flat, rate 0 at the corner). A referee will demand substantiation.
- 🟡 **nice — MSC codes:** `65M60` (evolution equations) is odd for a stationary problem; consider a 76-series
  porous/fluid code (e.g. 76S05). Funding/acknowledgments and other MSC codes are present (verified).
- 🟡 **verify — solver-disclosure sentence** ([1124](../theory/paper/article.tex#L1124)): the Newton–Krylov
  acceleration "does not affect any of the reported errors" — stays true given the OSGS-P2 finest cells needed
  `osgs_p2_precond_c1_mult=4` (preconditioner-only, root-preserving), but pre-empt the referee probe.

---

### Suggested sequencing

1. **Provenance/build blockers** (§5, §6): commit the two files, confirm the certified 3D re-run and official
   path. Everything numeric depends on these.
2. **3D table transcription audit** (§3): cell-by-cell check vs the certified DBs, settles C7. (E1 finest-cell
   certification is already **closed** — §2 — modulo the one final recheck at the very end.)
3. **α₀-exponent rewrite** (§0) and the theory-claims pass (§1).
4. **Prose/markup/refs** (§4, §7), then the **final review-off build** (§5).
5. **Reviewer-demand gaps** (§9).
