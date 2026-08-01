# Pre-submission checklist — `article.tex`

**Purpose.** The final read-through checklist before submitting *"A stabilized finite element method for
incompressible, inertial flows in inhomogeneous porous media."* Built 2026-07-18 from the living docs
(`paper-revision-plan.md`, `archive/review_numerics_vs_theory.md`, `open-questions.md`, `pending-tasks.md`,
`findings.md`, `theory-code-map.md`, `part_i_erratum.md`), the newest results, and a full theory-vs-numerics
re-derivation of the α₀-exponent estimates.

Each item: **severity** (🔴 blocker / 🟠 important / 🟡 nice-to-have), **status** (open / verify / likely-done),
and the source. Work the 🔴 first. "verify" = believed done but must be eyeballed in the final PDF/data.

> **2026-07-21 audit-response addendum.** A systematic external audit (`docs/ChatGPT audit/`,
> per-issue disposition in `validity_verdicts.csv`) was validity-checked and applied as `\amend` changes; many
> items below are now DONE (paper builds clean, ~69 pp, 0 undefined refs). Cross-cutting completions:
> **M03** (§6 `eq:DominantPressureGradientXTermEstimate` factor `‖a‖/√P`→`‖a‖U/P` — fixed + the eight §6
> isolation displays machine-checked); **M01** (convergence theorem promoted to the sharp ℓ² form Ψ(h), ℓ¹ as
> corollary — matches the Coq); **coverage audit** (all 369 displays checked, 0 further errors, SymPy 242/242);
> **N01** (fold/"no solution"→solver non-convergence); **C01** (deviatoric→spherical wording); **M09** (conclusion
> now states the `1/α₀` reaction-pressure exception → resolves F10); **D04** (Taylor–Hood match is velocity-only
> → the §9 TH item); **I07** (α-interpolation claim → future work; the 9b MMS-α item; R2 ablation now runnable);
> **C09/C10** (σ𝟏₃→σ𝕀; Damköhler≠Darcy → F3); **N02** (U=L=1 reconciled); **N10/N13/N15/N19/S03/S07** wording.
> Still open: raw-data + reproducibility supplement, bibliography clean-up. (The `\amend`/author-note strip
> was done 2026-07-30 — §4.) The 3D-OSGS 1.29 (C7/E1) is confirmed a *genuine* under-stabilization (rerun R10),
> not stale data.

> **2026-07-23 addendum (commented-appendix audit — applied).** A third external audit
> (`ChatGPT audit/audit_commented_article.md`, verified disposition there) drove a batch of definite fixes,
> now applied and built clean (v2 + v1 + note, 0 undefined refs; SymPy **262/262**): the **Codina citation**
> (`Codina2008FiniteEA`→`codina2008analysis` in both appendices + `:897`); three trusted-base proof gaps
> (`ω_χ→0` smooth-grading hypothesis on `H:patch`; global `a∈W^{kᵤ,∞}` in `H:advectionsmooth`; the
> contrast-free patch-equivalence chain); a **dev-sym definiteness lemma** in App C; abs-value bars on the
> ASGS continuity displays; mean-corrected pressure interpolant; the §4.5 technicals; the §4.7 numeric
> over-claim removal; the A17 inf–sup domain; and fold/`1/α` prose tempering. **Do NOT re-fix** the verified
> false positives (§4.6 L² factor, §6.4 P₁ residual, §2.4 `m≥2`, §10 bib). Two new gates:
> `proof_verification/sympy/theorem_statement_verification.py` (in `run_all.py`) and
> `proof_verification/hypothesis_transcription_audit.md`. Still open: the `:274` Fourier cite decision and the
> broader β₀/h₀ and robustness wording (author judgment). (The `\amend`/author-note flatten was done
> 2026-07-30 — §4.)

> **2026-07-30 addendum (fourth external audit — verified, applied, and the dump deleted).** A
> fourth external audit (produced from a Repomix export of `theory/` + `proof_verification/`, so it
> saw neither `src/`, nor `article.tex`, nor the commented App-D twin) was verified item by item
> against primary sources by a 10-cluster + adversarial-refutation workflow. **All 46 items adjudicated:
> four verified FALSE (S4, S7, N4, P6), one stale quotation (C1's footnote half), one cluster that is an
> artifact of the packed export rather than of the repo (C12/T9's "missing files" and the build failure),
> a dozen already done, five declined with reasons, and the rest applied here.** Both papers rebuild clean
> (**v2 116 pp / 976 newlabels, v1 80 pp / 776 newlabels, 0 undefined, 0 multiply-defined,
> 0 overfull > 1 pt**), SymPy **615/615 across 23 scripts**, Coq **25/25 + coqchk + only the 3 stdlib
> axioms**, Blitz **306/306**. (Counts regenerated from the live logs at the end of the pass — the
> in-pass figures 114/974 and 608 were superseded by the later reread fixes.) The audit file
> itself is gone (routed per `.agents/skills/external-audit-intake/`); its IDs are kept below so
> nothing reads as missed.
>
> **✅ APPLIED — proof artefacts.** **C2:** `AbstractContinuity.v` now concludes on `Rabs BS`
> (`abstract_continuity`, `abstract_continuity_sharp`), matching the printed `|B_ASGS| ≤ …`; the
> signed bounds survive as `*_signed` corollaries. **C3:** the OSGS Coq error functional is now the
> sharp broken-ℓ² `Ψ_O` (`PsiO`/`PsiTerm`), with the ℓ¹ majorant `E(h)` kept as `Eh` and the printed
> relation `Ψ_O ≤ c₁^{1/2}E(h)` itself kernel-proved (`PsiO_le_Eh`). Both are transcription repairs,
> not new mathematics — see `findings.md` §9 for why. Coverage wording synced in `coq_coverage.tex`
> (4 loci incl. the trust-stratum table), `coq-formal/README.md`, `EQUATION_COVERAGE_LEDGER.md`,
> `hypothesis_transcription_audit.md` (row 6 + the diagnosis paragraph), and
> `theorem_statement_verification.py`'s header.
>
> **✅ APPLIED — paper (both mains unless noted).** **C1/N1/S2** the OSGS projection display: the
> unconstrained target now has its own symbol `𝒳_h^proj := 𝒱_h × 𝒬_h`, defined once in
> §`sec:ScaleSplitting` and used at all three projection sites (the OSGS bullet, `Π_{τh}`, and the
> `W_h` quantifier of `eq:OSGSProblem`) — the display is no longer contradicted by its own footnote;
> the formal scale-splitting `Π_h` is now explicitly distinguished from `Π_{τh}`, and a footnote
> records that `𝒳_h^proj ⊄ 𝒳_0`, so the identification is at the level of the subscale *model*.
> App-D item (iv) (both twins) no longer documents a mismatch that is gone. **C7** the advection is
> "externally prescribed — modelled on, but not identified with, the frozen velocity", plus a §7
> sentence that a `k_u≥2` Lagrange iterate satisfies neither `∇·(αu_h)=0` nor global `W^{k_u,∞}`.
> **C8** the boundary operator's codomain is a generic trace space, with the classical/weak split
> spelled out. **C9** Carathéodory-type conditions for the reaction pairing + the DBF drag's
> velocity regularization (`(|u|²+u_ε²)^{1/2}`, mesh-independent, `u_ε/U ∈ [8·10⁻⁷, 1.2·10⁻⁴]`
> across the four §7.3 cells — verified against the configs). **C10** the porosity-oscillation step
> `α_{∞,K} ≤ δ_α α_{0,K}` is stated in (A4) and derived — with the element convexity and `h_K` =
> diameter it needs, plus the non-convex `1+C_geo·C_∇α` and surrogate `1+χ₀·C_∇α` variants — in the
> App-C remark `rem:porosityresolution`, which (A4) and both App-D twins cite instead of restating
> it (relocated 2026-07-30). **C11** abstract: "robust w.r.t. Re and Da" → "parameter-explicit"; conclusion: the
> `α₀^{-1/2}` claim is scoped to the *displayed* weights, naming `C_{∇α}`, `C_{α,m}`, `β₀`, `h₀`.
> **T1** abstract states the experiments lie beyond the hypotheses. **T3** — the one *factual* find:
> §7 printed the **`centered`** encoding (`L=1`, `ν=1/√(α_∞Da)`) while **every official 2D sweep ran
> `minmax`** (`L=√(α_∞Da)`, `U=(Re²/(α_∞Da))^{1/4}`, `ν=σ=(α_∞Da/Re²)^{1/4}`) — corrected, together
> with the invariance claim (analytic, not finite-precision) and a new sentence justifying
> `Re_h = Re h/L = Re/N`. **T4** the two 3D captions and both Cocquet captions now say *worst-case
> bound rate* instead of "theoretical"/"optimal". **T5** the "finer meshes would only reconfirm
> them" sentence is retracted (it contradicted the two-mesh-slope hedge four lines below). **N2**
> the projector-independence claim now names the Korn constant `C_K` for the full-gradient
> conversion. **N3** a σ-notation sentence in (A2). **P4** two over-strong verbs softened. **P5**
> App-C definiteness made verbatim-parallel with App-D; the pressure gauge named at (A6) and at the
> 2D `ε=0` statement (**superseded 2026-07-30**: the 2D and DBF campaigns do not run at `ε=0` at all —
> both harnesses carry a literal `1e-8`; see `findings.md` §9.5 and `pending-tasks.md` §4g. The paper
> now says so, and the recorded provenance attributes read the assembled value). **S1** the §5 opener
> now states the scope before the first interpretation, and
> the exact-α statement (α is not interpolated onto a finite element space) is recorded — since
> 2026-07-30 in the App-C remark `rem:porosityresolution` cited from (A4), not in (A4) itself. **T9** a *Code and data availability* section was added to
> both mains.
>
> **✅ APPLIED — companion notes.** **T8** `cocquet_form_mms_manufactured_solution.tex` no longer
> calls a P2/P1 **ASGS** test "matching Cocquet": it is an *operator-matched stabilized diagnostic*,
> and "reproduction" is reserved for the pure-Galerkin TH run (its stale reproduce command fixed
> too). **N10** `centered_encoding.tex`'s three mutually inconsistent status statements are replaced
> by one provenance table (strategy per artefact, where it is recorded, commit), the false
> "master HDF5 produced under centered" claim removed, and the roundoff attribution narrowed per
> `lessons_learned.md`. **N5/N6** Coq module count 24→25 and the stale `theory/verification scripts/`
> path fixed. **N9** the bump-smoothness coverage sentence corrected (Coq `C¹`, SymPy `C²`, nothing
> beyond). **C4** the "entire chain machine-checked" headline is now qualified *in the bold*, in both
> `coq_coverage.tex` and the Coq README. **N8** one submission source of truth named
> (`article_v2.tex`; `article.tex` = the code-transcription anchor).
>
> **🔴 STILL OPEN from this audit.** (i) **T9** — the *Code and data availability* section now names
> the repository, `\url{https://github.com/GuillermoCasas/GridapPorousNS}` (verified public on
> 2026-07-30: the unauthenticated GitHub API returns `private: false`). What is **still missing and
> cannot be produced from inside the repo** is the *archival* half: a Zenodo (or equivalent) deposit
> of the exact snapshot, its DOI, and the commit hash of that snapshot. Two conditions to check before
> submitting: the working tree of this pass is **not yet pushed**, so the revision the paper describes
> is not on the remote until it is; and the statement's specific promises (a config file per reported
> case, the mesh generators and committed base meshes, the table-regeneration scripts, the verification
> suites) are true of the tracked tree — 39 tracked configs, 12 tracked meshes,
> `make_results_tables.py`, `make_3d_tables.py`, `sympy/run_all.py`, `coq-formal/_CoqProject` all
> git-tracked — but the *result databases* are gitignored by design, so either ship them with the
> deposit or keep the statement's wording ("the scripts that regenerate the tables … from the raw
> result databases") backed by §6g's regeneration recipe. (ii) **T7** — ✅ **resolved in this
> same pass** (see the RESOLVED T7 block below): the author decision came out *add-alongside*, and
> `genuine3d_table.tex` is now `\input` after `tab:3DH1` in `article_v2.tex`. This sentence recorded it
> as open because it was written before the decision landed; `pending-tasks.md` §7f and
> `open-questions.md` §4 are closed to match. (iii) **C12/P2/S6**
> — the release bundle (pinned TeX/Coq/Python/Julia versions, one build command, one gate command,
> archived logs with hashes, and the URL/DOI above): `pending-tasks.md` §6g. (iv) **T6** — a
> per-attempt terminal-status supplement for the omitted cells: `pending-tasks.md` §5c. (v) **N7/P1**
> — the `\amend`/author-macro flatten: **✅ DONE 2026-07-30**, 1011 wrappers unwrapped across both mains
> and App. A/B/C and the three `\newcommand`s deleted, gated output-neutral (identical page/label counts,
> zero-line PDF text diffs, SymPy 636/636). Detail: `pending-tasks.md` §1c.
>
> **✅ RESOLVED 2026-07-30 (was Tier-2, author-reserved twice) — the global-quasi-uniformity tension.**
> (A1) asked for *local* quasi-uniformity, (O3) added *patch* quasi-uniformity, and §6 invoked *global*
> quasi-uniformity "assumed throughout", which nothing supplied — flagged by two prior audits (A01) and
> deferred both times. Settled by **developing the theory first** in
> `theory/mesh_regularity_note/` (4 pp, builds clean): global QU **implies** the local and patch
> versions (so adopting it shortens the hypothesis list), it does **not** imply smooth grading (kept,
> and free for an admissible constant surrogate), the paper's former "`ω_χ=O(h)` when quasi-uniform"
> claim was **false** (corrected to the Lipschitz-size-function condition), and global QU is needed
> **only** for §6's single-`h` statements — because `h` appears on both sides of them — and equally for
> both variants, the appendices needing nothing beyond the weaker versions. (A1) now states global
> quasi-uniformity with a footnote recording exactly what it buys and how to weaken it; §6's invocation
> now has a referent. Argument and counterexamples: `findings.md` §9.4.
>
> **✅ RESOLVED 2026-07-30 — T7, the genuinely-3D table is now in the paper.** `genuine3d_table.tex` is
> `\input` after `tab:3DH1` (v2 only; v1 has no such rows, as with the R5 control), with a paragraph
> that says what the data actually shows: the velocity rates approach or exceed the optimal exponents
> for both orders and both variants, every pressure rate exceeds the worst-case bound rate, the OSGS ℙ₂
> pressure reaches `2.00` in `H¹` (against `1.77` on the extruded field) — **but the absolute `H¹`
> pressure errors remain `O(1)` (1.22–2.97), as on the extruded field**, so the near-stagnant
> *magnitude* is *not* an artifact of the extruded field's degeneracy. That **sharpens** the adverse
> 3D finding rather than dissolving it, which is why including the table is the more honest option:
> the data exists, is certified (all 16 levels `success=true`, `c1_mult=4`), and every printed value
> was re-derived from the record. The drop-in file's own suggested sentence was *not* used — it claimed
> the OSGS pressure "converges rather than saturating", which the `O(1)` errors do not support.
>
> **✅ RESOLVED 2026-07-30 — the 3D `c₁` convention, and the ℙ₁ claim.** The runs' element length is now
> stated: `h_K = (6√2|K|)^{1/3}`, the edge of the equal-volume regular tetrahedron
> (`smoke3d.jl:176`, `h_conv="regular_tet"` — **not** the `shortest_edge` of `base_config.json`, which
> the 3D harness never reads; a reread finding that assumed otherwise was a false positive). The
> "sits just below the elementwise Kuhn threshold" sentence is now explicitly *relative* to the
> two-dimensional value, with the absolute margin disclaimed as requiring both a fixed `ξ` and a common
> length convention across element families — which is the standing ruling's own position. The
> conclusions' ℙ₁ half no longer claims experimental support it does not have: the elementwise threshold
> **vanishes identically** for ℙ₁, so no increase is required of it for coercivity, and that is now the
> stated reason (all 3D runs used `16k⁴`; no ℙ₁ run at `4k⁴` is reported, and the repo's own control
> found `4k⁴` *less* accurate for ℙ₁).
>
> **DECLINED, with reason (do not re-raise without new evidence).** **S3** per-theorem
> "theorem / formal coverage / implemented" boxes in the article — the manuscript makes **zero**
> formal-verification claims, so a coverage row would *add* a claim; rows 1 and 3 already exist
> (per-result hypothesis labels + `tab:ASGSvsOSGS`; `:1456-1463`, §7 preamble, `oa:rem:analyzed`).
> **S5** a six-column regime-map table — its content is already carried by the fixed-data-vs-
> pre-asymptotic paragraph, `tab:ASGSvsOSGS`'s reaction row and the table-caption convention; a new
> wide table is page budget and overfull-box risk for no new information. **C5** relocating the
> theory/implementation distance list *before* the theorems — the paragraph exists after them and is
> reinforced in §7; the honest register entry is "we decline the relocation", not "the auditor missed
> it". **T2** the 3D `c₁` "admissible because global continuity weakens the elemental threshold"
> clause — standing ruling, recorded twice (this file §1 and `open-questions.md` §3, R3 dropped
> 2026-07-21); the ξ imprecision is already disclosed there. **P7** date stamps in `.tex` comments —
> all 13 are on comment lines, invisible in the PDF; fold into the flatten pass if ever.
>
> **✅ APPLIED 2026-08-01 (both mains + both App-D twins).** Two follow-ups to **C1/N1/S2**.
> (a) *Unconstrained projection space, said once.* The rationale (target is `𝒳_h^proj`, whose members
> need not vanish on `Γ_D`, so `π_h` is not pinned to zero there; projecting onto `𝒳_{h0}` instead
> would drop (O2) but generates spurious near-wall layers, as in `codina2008analysis` Remark 1) was
> stated in three places; it now lives in one footnote after `eq:L2InnerProduct`, the OSGS-space
> bullet defers to it, the footnote to `eq:OSGSProblem` is gone, and `oa:sec:method` + `oa:rem:analyzed`(iv)
> cite it instead of restating it — (iv) now reads "no difference". (b) *(A7)/(A9) gets a referent.*
> The regularity assumption said "*the* exact solution" with nothing establishing that one exists:
> §2's existence hedge is about the **nonlinear, mixed-BC, general-σ** problem, which `sec:WeakForm`
> then explicitly declines to assert well-posedness for. Well-posedness of the *linearized* problem
> is not a hypothesis but a two-line theorem — coercivity (viscous by (A3)'s Korn condition,
> convection skew since `∇·(αa)=0`, reaction `≥0`) plus the α-weighted inf-sup, inherited from `α≡1`
> through the isomorphism `v ↦ v/α` of (A4) — so it is now *stated* in a footnote to the assumption,
> which assumes only the extra smoothness, genuinely unavailable on a polyhedral `Ω`.
>
> **VERIFIED FALSE / ARTIFACT — do NOT "fix".** **N4** (Damköhler≠Darcy is already stated at the
> definition of `Da`), **S7** (there is no verification-process history in the article body),
> **P6** (no ε/machine-ε collision inside the paper; only in one standalone note),
> **C1's footnote half** (the audit quoted the pre-2026-07-27 footnote; the live one already
> concedes Codina's hedge, and the audit's replacement would be a *regression* — the repo's
> `projection_space_note` proves a **degree-dependent** two-sided result, optimal for `k=1` and
> `Θ(h^{3/2})` for `k≥2`, which is *stronger* than the audit's suggested wording), **C6/S4** (the
> non-certification of `c₁=4k⁴/16k⁴`, `c₂=2k²` is already stated in the main text immediately after
> `th:StabilityOSGS`), **C12's build-failure claim** (both mains build clean here; the auditor hit
> the known `ntheorem`×`cleveref≥0.21` clash inside the bundled 2019 SIAM class — a TeX-distribution
> portability issue, tracked as part of §6g), **C12/T9's "missing files"** (`article.tex`,
> `osgs_appendix_commented.tex`, `figures/bump_plateau.pdf`, the 25 `.vo` objects, `src/`, the
> configs and meshes all exist; they were absent from the *pack* by its own exclusion rules).
>
> **T10 (the audit's "positive alignment worth retaining")** needs no action and is not a compliment to
> bank: it lists six caveats the numerical narrative already carries (fixed-data vs pre-asymptotic;
> upper bounds permit but do not predict the OSGS reaction excess; pressure *rate* vs pressure *error*;
> solver failure ≠ nonexistence; a 3D discretization test vs a genuinely 3D field; the stabilized-TH
> comparison not isolating the element pair). Every one of them is still in the paper after this pass —
> several were *strengthened* here (T4, T5, C11, P4) — and the T7 decision above is precisely the fifth
> one's remaining half. Do not let a later prose pass quietly remove them.
>

> **✅ DISCHARGED 2026-07-31 — was: 🔴 NEW BLOCKER found by the 2026-07-30 critical reread, the DBF table's
> stabilized-Taylor--Hood rows mix meshes and databases.** *Everything in this block describes the state
> as found; the resolution is the last bullet.* Verified directly against the DBs at the time: `cocquet_form_mms_taylorhood_stabilized.h5`
> stops at **N=160** (and at **N=80** for the `(10⁵,0.1)` cell, the rest NaN), while the equal-order and
> unstabilized-TH rows are at **N=320**; and the `(10⁵,0.1)` `P₂/P₁ ASGS` row printed in the tables
> (`2.18e-5` velocity, `4.81e-6` pressure) comes from **`results/debug_results/cocquet_stabth_corner.h5`**,
> a forked side-DB — which `.agents/rules/official-results-path.md` forbids for published numbers.
> Consequences, and what was done now:
> - The claim that the stabilized control is "about an order of magnitude *less* accurate than the
>   unstabilized one" in the viscous regime was an **artifact of comparing N=160 against N=320**:
>   `3.12e-6 / 2.85e-7 = 10.9`, whereas at the common mesh N=160 the ratio is **1.16×** (α₀=0.5) and
>   **1.47×** (α₀=0.1). ✅ **Corrected in the prose** to the like-for-like comparison.
> - ✅ **Disclosed in both DBF captions** (which mesh each `P₂/P₁ ASGS` FME is on) and in the §7.3 preamble
>   (four discretizations, not three; the ladder is not common to all of them).
> - 🔴 **The proper fix is still open:** re-run the stabilized-TH control to N=320 at all four cells
>   *through the official path* (`data/cocquet_form_mms_taylorhood_stabilized.json`, extend the ladder;
>   archive the current DB to `previous_results/` first) so that every FME in the two tables is on one
>   mesh and no row is sourced from `debug_results/`. Until then the tables carry the caveat above.
>   **Correction (2026-07-30, later pass): the earlier claim here that "v1 is unaffected — it has no
>   `P₂/P₁ ASGS` rows" was false.** `article.tex` carries the same four-method DBF tables
>   (`article.tex:1793`, `:1802`, …), so every item in this block applies to it too. v1 had been left
>   with the *uncorrected* prose — the refuted "order of magnitude less accurate", "three
>   discretizations", the `Q_0` pressure-gauge clause — and with **no** mesh caveat in either caption.
>   All now mirrored — but only after a second parity sweep, which found that the first pass had updated
>   v1's "We compare **four** discretizations" preamble and left "the three discretizations behave alike"
>   standing in the viscosity paragraph, plus four more v1-only drifts the audit batch had not touched:
>   the 3D penalty paragraph still called `ε = 10⁻⁴ε_ref` a *compressibility* (contradicting the same
>   subsection's new "kept at exactly `ε = 0`"), the 3D regular-mesh claim still said "the observed rates
>   approach the theoretical optima" (true of the velocity only, and the captions no longer print an
>   optimum there), the `c₁` remark still asserted an absolute margin below the Kuhn threshold without the
>   element-length convention, and the `u_ε` drag-regularization disclosure was missing entirely.
>   The general lesson (now in `lessons_learned.md`): a claim that one main is exempt must be checked
>   against that file, and a parity sweep must cover the passage the correction *implies*, not just the
>   sentence that was quoted.
>   **Status: ✅ DISCHARGED 2026-07-31.** The official config completed its full `[10..320]` ladder for all
>   four cells; all eight `P₂/P₁ ASGS` rows are requoted in both mains, both caption caveats and the §7.3
>   ladder caveat are deleted, and the corner row is now officially sourced — the re-run reproduces the
>   side-DB's `N=320` values *exactly*, so those numbers were right and only their provenance was wrong.
>   See `pending-tasks.md` §7h.
>
> **✅ Also corrected by the reread (3D subsection, both verified against the result records).** The
> claim that both mesh families are "generated with `Gridap`" was wrong — the irregular base mesh is
> generated by **gmsh** (`mesh3d.jl` drives `GridapGmsh.gmsh`, `Algorithm3D`), only the red refinement is
> Gridap. And "the reported errors are computed with a direct (sparse LU) solver" was true of the **ASGS
> rows only**: all four OSGS 3D records carry `jfnk: true` (`jfnk_restart` 30/80, and
> `jfnk_precond_c1_mult: 4.0` for irregular ℙ₂), i.e. GMRES preconditioned by that LU factorization. Both
> sentences now say what was actually done. *A gmsh citation should be added to `references.bib`* (the
> canonical reference is Geuzaine & Remacle, IJNME 79(11), 2009) — the sentence names the tool but cites
> nothing, since inventing bib fields is worse than leaving the citation to the author.

> **One status correction to this file.** §10.A recorded the RW-3 batch as "ALL 15 APPLIED &
> GREP-VERIFIED"; RW-3 was applied as an *added* clause (the two-mesh-slope hedge) without
> retracting the sentence it contradicted. That sentence is now retracted (T5 above), so the status
> line is finally accurate.

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

> **§1 AUDITED 2026-07-19** by a 9-agent verify+adversarial workflow (each item read against `article.tex`, the
> continuity appendix, the Coq `abstract_stability` base, and the docs). Result: **6 of 7 already clean/applied;
> S45-3 (Lemma 1) APPLIED this pass; S6-3 has a ready one-line fix awaiting go-ahead.** Details per item below.

- ✅ **VERIFIED CLEAN (2026-07-19) — the c₁ story is told truthfully and consistently.** Paper: `c₁=4k⁴` in 2D
  ([937](../theory/paper/article.tex#L937)), `c₁=16k⁴` in 3D "just below the elementwise Kuhn threshold"
  ([1441](../theory/paper/article.tex#L1441), [1671](../theory/paper/article.tex#L1671)), `c₁` element-dependent
  via `C_inv`; Cocquet uses the triangular `4k⁴` ([1564](../theory/paper/article.tex#L1564)). **No withdrawn
  framing survives anywhere** in `theory/` — "Gridap↔paper discrepancy", "c₁ masks a bug", the clean-room/NumPy
  element-family verdict are all absent (grep-verified; the only `element-family` hit is a neutral reference in
  `pressure_recentering_note`). The "4× < 4.46× ⇒ just below" arithmetic checks out `((100+5√2)/24=4.461)` and the
  footnote already hedges the absolute margin via the global-vs-elementwise mildness argument. Src: `findings.md §3`,
  `theory-code-map.md §2.5`. *(No paper change. Doc nit for later: `findings.md`/`theory-code-map.md` still cite
  "article.tex line 910" for the element-aware-c₁ remark whose actual line is 937.)*
- ✅ **APPLIED (2026-07-19) — S45-3: Lemma 1 (Stability, [943](../theory/paper/article.tex#L943)) hypotheses were
  insufficient for its own proof.** The proof used the mesh-nondegeneracy inverse estimate `eq:InverseEstimateFiniteOrderNorm`
  ([888](../theory/paper/article.tex#L888)) and the porosity-resolution condition `eq:SmallPorosityGradient`
  ([827](../theory/paper/article.tex#L827)/[890](../theory/paper/article.tex#L890) — the enlarged `C̄_inv` in the
  retained `c₁>2ξC_inv²` is defined *from* it), but the lemma stated neither. **Fix:** added exactly two clauses to
  Lemma 1 — "the family of meshes is non-degenerate (…so `eq:InverseEstimateFiniteOrderNorm` holds)" and "the porosity
  field is resolved by the mesh in the sense of `eq:SmallPorosityGradient`". This makes Lemma 1's hypotheses **equal
  to** App. C `prop:stability` (`H:data`–`H:mesh` + `c₁>2ξC̄²`) and to the machine-checked `abstract_stability`
  trusted base (`{H_skew_diag, H_ibp_diag, S3, Heps}` + sharp `c₁>ξC̄²`, whose `S3` weighted inverse estimate is
  precisely what packages mesh-nondegeneracy + resolution into the enlarged `C̄_inv`). **Minimality proven by the
  Coq (adversarial pass):** `0<α≤1` and `a`-regularity were **deliberately NOT added** — `StabilityAlgebra.v` uses
  only `0<αK` (never `α≤1`) and stability needs only `|a|_{∞,K}<∞`; those belong to continuity (Lemma 2), which
  already lists `a∈W^{1,∞}`. `ν>0`, `σ≥0`, `α>0` stay in the standing prose ([837](../theory/paper/article.tex#L837)/[842](../theory/paper/article.tex#L842)).
  **Coordinated Lemma 2 edit** ([960](../theory/paper/article.tex#L960)): dropped its now-redundant restatement of
  `eq:SmallPorosityGradient` and instead noted in the "sufficient but not minimal" parenthetical that mesh-nondegeneracy
  and resolution *are still needed* (only the coercivity threshold `c₁>2ξC_inv²` is relaxed for continuity). Theorem
  `th:Convergence` inherits automatically — no edit. **Build re-verified green: 66 pp / 722 newlabels / 0 unresolved;
  both new `\cref`s resolve (4.27, 5.8).** Src: `paper-revision-plan.md §7 (S45-3)`.
- ✅ **VERIFIED (2026-07-19) — divergence ledger walk.** All five honestly stated or intentionally-silent-and-defensible:
  (a) `(1/α)∇·(αa)v` omitted from the adjoint — disclosed at [866](../theory/paper/article.tex#L866); (b) positive-sign
  convective adjoint for the `A²−B²` symmetry — paper-faithful (`B_S` subtracts `L*`), disclosed via `X(U_h)`;
  (c) `τ₁/τ₂` simplified forms dropping `εh²`/`C_α` — justified at [824](../theory/paper/article.tex#L824)/[826](../theory/paper/article.tex#L826)
  and the remark at [845](../theory/paper/article.tex#L845); (e) reaction-projection trim only for constant σ —
  disclosed at [642](../theory/paper/article.tex#L642). **The one genuinely paper-silent item is (d)** — the OSGS
  projection is computed on unconstrained `V_free/Q_free` while the paper defines `π_h` on the Dirichlet-constrained
  `X_{h0}`; this lived only in `theory-code-map.md §2.6`. **APPLIED (2026-07-19):** added a disclosure footnote at
  `eq:OSGSProblem` ([614](../theory/paper/article.tex#L614)) stating that the projection is computed with `W_h`
  ranging over the FE spaces *without* their Dirichlet constraints (so the projection is not forced to vanish on the
  boundary), and that projecting on `X_{h0}` instead would introduce an `O(1)` boundary residual spoiling the optimal
  `O(h^{k+1})` convergence — matching `theory-code-map.md §2.6`. The paper is now self-contained on this point.
  Src: `theory-code-map.md §2.1–2.7`.
- ✅ **VERIFIED (2026-07-19) — §3.3 projection-trim sentence (D5)** ([642](../theory/paper/article.tex#L642)).
  Both qualifications present: reaction terms excluded from the orthogonal projection **only when σ is constant**
  (the 2D/3D MMS), and for the velocity-dependent σ of §7.3 the **full** residual is projected and the implementation
  coincides with `eq:OSGSProblem`. Code confirmed (`run_simulation.jl:56-58` double-gate; `CocquetFormMMS/run_test.jl:149`
  full residual on the DBF branch). *(Checklist said "§3.1"; it is actually §3.3 — locator drift only.)* Src: review D5.
- ✅ **APPLIED (2026-07-19) — S6-3: `Da = Da_h L²/h²`** was imprecise under the elementwise convention. Derivation
  (both agents, direction confirmed): with `Da=σL²/(α_∞ν)` global and `Da_h=σh²/(α_Kν)` elementwise (the τ₁ asymptotic
  at [1003](../theory/paper/article.tex#L1003) *forces* the `α_K`), the exact relation is `Da = Da_h (L²/h²)(α_K/α_∞)`.
  The bare form is used at **both** [1075](../theory/paper/article.tex#L1075) **and** [1081](../theory/paper/article.tex#L1081)
  (the mitigation-factor identity), so I fixed the **root** rather than one display: at
  [1008](../theory/paper/article.tex#L1008) — where the `h`-subscript convention is defined — I now state that "domain
  of interest = element" swaps **both** `L→h` **and** `α_∞→α_K`, give the exact `Da = Da_h (L²/h²)(α_K/α_∞)`, and note
  the ratio `α_K/α_∞ ≤ 1` is bounded and `h`-independent, so the compact `Da = Da_h L²/h²` used at 1075/1081 is
  licensed "up to this fixed factor". This is the checklist's "state the convention once" option; it keeps 1075 and
  1081 mutually consistent (a 1075-only edit would have contradicted 1081). Low severity confirmed — all *scaling*
  conclusions (`Da_h∝h²`, `h`-independent mitigation) are untouched. Build green. Src: `paper-revision-plan.md §6/§7 (S6-3)`.
- ✅ **VERIFIED (2026-07-19) — best-approximation claim scoped to H¹ only** ([1160](../theory/paper/article.tex#L1160)):
  "sits on the interpolation error in the H¹-seminorm for both elements" + the `√6` ℙ₁-L² caveat are present; the
  falsified "and in both norms for the biquadratic one" is **gone**. The surviving "in both norms for the biquadratic
  element" at [1164](../theory/paper/article.tex#L1164) is the *separate, true* ASGS-vs-OSGS method-agreement claim —
  correctly retained, do not touch. Src: §0.2c.
- ✅ **VERIFIED (2026-07-19) — §7 scope sentence (D7)** ([1102-1113](../theory/paper/article.tex#L1102)) and the
  `C_inv` vs weighted `C̄_inv` convention (D10, `rem:winvconst` [345-351] — no double-count; `\Cinva` USED 16× in the
  appendix, do **not** delete). **Checklist itself is wrong on one point:** it asks the scope sentence to state a
  "Neumann outlet in the DBF benchmark" — but **no experiment in this paper uses a Neumann outlet**; all three
  families (2D/3D/DBF) are all-Dirichlet manufactured solutions ([1557](../theory/paper/article.tex#L1557)), and the
  paper explicitly declines Cocquet's Neumann tube flow ([1555](../theory/paper/article.tex#L1555)). The sentence is
  correct *because* it omits that claim — **do not add it**. Src: review D7/D10.

## 2. Numerics / results

> **§2 AUDITED & CLEARED 2026-07-19** — all seven items verified **without re-running**, against the on-disk
> certified DBs and the paper prose (data checks via `make_3d_tables.py --check`, direct JSON/HDF5 reads, and
> 2D-table regeneration-and-diff; text checks via a 4-agent workflow). **Result: 7/7 verified; no paper change
> needed.** Details per item below.

- ✅ **CLOSED + FINAL-RECHECK DONE (2026-07-19) — E1** (no published 3D slope is part-interpolant). Confirmed
  on the four canonical DBs (`results/k{1,2}/TET/{structured,nested_red}/convergence3d_results.json`): **zero
  `success=false`** across all **30 solver levels**, and the two cells a stall would camouflage both moved off
  the interpolant with `success=True` — K1-nested_red ℙ₁ ASGS `l2u`=1.103e-3 vs interp 6.823e-4 = **1.617×**
  (iters=1), K2-nested_red ℙ₂ OSGS `l2u`=2.023e-4 vs interp 2.012e-4 = **1.006×** (iters=5, on the velocity
  floor but genuinely iterated). Matches the expected 1.62×/1.005× exactly. The final-recheck-at-the-very-end
  is now performed; **re-confirm once more only if the 3D sweep is ever re-run.** Src: on-disk DBs; `findings.md §3`.
- ✅ **VERIFIED (2026-07-19) — C7 "1.29 triple"** ([1453](../theory/paper/article.tex#L1453)):
  `make_3d_tables.py --check article.tex` reports *"every solver + interp \num value in tab:3DL2 and tab:3DH1
  matches the data"*, and the three raw OSGS pressure-H¹ FMEs read from the DBs are **1.29198 / 1.28894 /
  1.28954** → the footnote's 1.292/1.289/1.290, all rounding to 1.29. Genuine mesh/order-independent saturation,
  not a transcription coincidence. **This `--check` also clears the §3 🔴 3D-table cell-by-cell audit** (it diffs
  *every* slope/FME/interp value in both 3D tables against the DBs). Src: on-disk DBs.
- ✅ **VERIFIED (2026-07-19) — S6-1** (workflow PASS). The corrected prose ([1162](../theory/paper/article.tex#L1162))
  reports the true spread (ASGS 1.3–2.0, OSGS 3.5–4.1), calls OSGS **"marginally above"** the `α₀^{-1/2}` weighted
  prediction (not "of the order of"), and keeps the honest "both upper bounds, cannot select between them" caveat;
  `rem:WeightedVsUnweighted` ([1083-1099]) carries no OSGS overclaim. Grep confirms "of the order of the weighted
  prediction" has **0 hits** in `article.tex`. Src: `paper-revision-plan.md §0.2a`.
- ✅ **VERIFIED (2026-07-19) — Cocquet magnitude honesty** (workflow PASS + provenance trace). Prose: the section
  is now a manufactured-solution comparison against the paper's **own** Taylor–Hood ([1555](../theory/paper/article.tex#L1555));
  "Kratos" and "modified corner" appear **0×** in `article.tex`; the only non-convergence (the (10⁵,0.1) corner) is
  attributed to a coarse-mesh fold that recedes with refinement, not a formulation defect. Provenance: every
  spot-checked value in `tab:CocquetMMSL2/H1` traces to the on-disk DBs (`cocquet_form_mms_{vms,taylorhood}.h5`) —
  e.g. (1,0.5) ℙ₁ OSGS 5.631e-5, ℙ₂ ASGS 3.424e-7; the TH **n.c.** entries are genuine rate-0 stagnation
  (5.140e-1 / 4.024e-1 / 6.338 / 4.963). Src: `findings.md §6`.
- ✅ **VERIFIED (2026-07-19) — no Kratos magnitude-reproduction claim in 2D** (workflow PASS). "Kratos"/"Multiphysics"
  appear **0×** in `article.tex`; the paper states the experiments were run in Gridap at
  [1115](../theory/paper/article.tex#L1115) (not the checklist's old ~1124), and benchmarks only against its own
  nodal-interpolant reference. (The former Gridap-vs-Kratos magnitude-offset open question was **removed 2026-07-19**
  — Kratos is not part of the paper.) Src: workflow audit 2026-07-19.
- ✅ **VERIFIED (2026-07-19) — 2D tables match the certified sweep.** Regenerated all four tables from the DBs
  (`make_results_tables.py`) and numerically diffed: **all 15 solver data rows × 4 tables are identical** to the
  paper's `tab:Linear2D*`/`tab:Quadratic2D*`; velocity rates recover finest-segment O(h³)/O(h²) (Q2 L²≈3.00,
  H¹≈2.00), with the documented pre-asymptotic dip (2.82/1.81) only in the Re=10⁶ rows. The k2/QUAD DB config
  confirms the **tight gate `eps_tol_momentum=1e-9`** (⚠️ but `ftol=1e-10`, **not** the `1e-12` this checklist
  stated — minor doc drift, rates are correct either way). Interp rows (not emitted by the generator, D4c open)
  are the already-triple-verified B1 references. Src: `findings.md §1`.
- ✅ **VERIFIED (2026-07-19) — OSGS reaction-dominated velocity gap** (workflow PASS). Framed at
  [1158](../theory/paper/article.tex#L1158)/[1170](../theory/paper/article.tex#L1170) as a **larger error at a
  preserved rate**, a pre-asymptotic effect governed by `Da_h` that decays under refinement at an accelerating
  rate — explicitly *not* an order ceiling ("the gap is an accuracy effect our analysis does not resolve"). The
  paper does not tabulate the N=640 recovery it did not run (honest); its own ℙ₁ H¹ Da=10⁶ rows already show the
  rate preserved (~1.05) with the FME elevated ~3.5×. Src: `findings.md §4`.

## 3. Figures / tables

- ✅ **DONE (2026-07-31) — the L²/H¹ table pairs merged, one table per example; FME precision cut by one
  significant figure.** Each example used to print two tables (one per norm) over the same row keys; they are
  now **one** table whose data columns are `L²-norm` then `H¹-seminorm`. v1 81→77 pp, v2 117→113 pp.
  **Label map — every `…L2`/`…H1` label above and in the other docs resolves through this table:**

  | was | is now |
  |---|---|
  | `tab:Linear2DL2`, `tab:Linear2DH1` | `tab:Linear2D` |
  | `tab:Quadratic2DL2`, `tab:Quadratic2DH1` | `tab:Quadratic2D` |
  | `tab:3DL2`, `tab:3DH1` | `tab:3D` |
  | `tab:CocquetMMSL2`, `tab:CocquetMMSH1` | `tab:CocquetMMS` |

  **Three space levers, applied in this order — the font is the LAST resort, not the first.** (i) The
  column-header deck is emitted **once per table**, not once per velocity/pressure block; the block-specific
  worst-case bound rates moved into the block band row as an (L², H¹) pair (`velocity (2, 1)`,
  `pressure (1, −)`), the same idiom `tab:3D` uses in its element labels. (ii) Every **FME** lost one
  significant figure (solver rows 3 s.f.→2, `tab:3D` interpolant rows 4→3); **slopes are untouched**.
  (iii) `tab:CocquetMMS` moved `Re` and `α₀` out of two columns into spanning subheadings.
  Only then the font: **`\footnotesize` + `\tabcolsep` 2pt, uniform across all five convergence tables**
  (the four merged ones and `tab:Genuine3D`, which was `\scriptsize`). Measured against `\textwidth` =
  370.4pt: the 11-column 2D table is 348.0pt, `tab:3D` 335.7pt, `tab:CocquetMMS` 363.1pt,
  `tab:Genuine3D` 355.3pt. Before levers (i)–(iii) the same tables needed `\scriptsize` (2D was 409.0pt at
  `\footnotesize`/4pt), so **the levers bought back a whole font step** — this matters because the
  2026-07-30 caption re-scope below had removed a `\footnotesize` override precisely to stop one table being
  smaller than the rest. **Do not restore a larger font for one of these five, and do not re-inflate the FME
  precision without re-checking the widths.**

  **Fidelity.** The merge was generated by parsing the old tables and matching rows on their key columns,
  then checked by a separately-written verifier that re-parses the merged tables and compares every cell to
  the pre-merge original (keys, methods, reference rates, slopes exact; FMEs exactly one s.f. shorter).
  `tab:3D`'s element labels became **pairs** — `ℙ₁ (2, 1)` = bound rate in L² then H¹ — and its H¹ dash is
  now `$-$`. Prose that quoted a tabulated FME was moved to the new precision; the two passages that argue
  at the *third* digit (`3.52e-2` vs `3.499e-2`, ratio 1.01) keep their digits and now say they are quoted
  to more digits than the table carries, and the recurring-value footnote reads `1.3` (…"the same two
  significant figures").
- 🟡 **OPEN (opened 2026-07-31) — 24 FMEs were DOUBLE-ROUNDED and should be re-derived from the DBs.**
  Dropping a significant figure was done by re-rounding the *printed* 3-s.f. value, because the 2D sweep
  HDF5s and the Cocquet DBs are not on this disk. That is exact except when the printed value ends in 5,
  where the true value can fall either side: **28 FME literals end in 5**, and 4 of them (all `tab:3D`
  regular-mesh) were resolved against `convergence3d_results.json` — which caught a **real** error:
  `1.85e-4` (regular ℙ₂ velocity, L² ASGS) is `1.9e-4` by half-up but **`1.8e-4`** from the data, now
  corrected. The remaining **24** are unverified against full precision:
  `tab:Linear2D` 10 (`6.95e-5`, `1.75e-5`×2, `1.85e-5`, `2.85e-5`, `2.35e-3`, `8.75e-4`, `3.65e-2`×2,
  `1.75e-1`), `tab:Quadratic2D` 9 (`1.05e-4`, `1.95e-4`, `9.75e-6`, `4.45e-7`, `3.85e-9`×3, `4.95e-3`×2),
  `tab:CocquetMMS` 3 (`1.85e-6`, `2.85e-7`, `3.45e-7`), `tab:3D` 2 (irregular `7.25e-1`, `8.415e-2`).
  Each could be off by one in the last digit. **Durable fix, already in place:** the generators now format
  at the new precision (`make_3d_tables.py` `fmt_fme` `%.1e` / `fmt_fme4` `%.2e`; `make_results_tables.py`
  `_fmt_fme` `%.1e`), so regenerating from restored DBs produces the direct-rounded values and
  `make_3d_tables.py --check` diffs them. **Re-run both checks once the DBs are back, before submission.**
- ✅ **DONE (2026-07-19) — cell-by-cell audit of the auto-transcribed 3D tables (commit 638a298).**
  `make_3d_tables.py --check article.tex` diffs **every** slope, FME, and interpolation-reference `\num` in
  `tab:3DL2`/`tab:3DH1` against the certified DBs + `interp_reference3d.json` and returns *"every solver + interp
  \num value matches the data"*. No transcription slip. (`article.tex` still does not `\input` the generator, so
  the **drift risk recurs on any future 3D edit** — re-run `--check` before submission, or close D4c below.)
  Src: `paper-revision-plan.md §0.4d`.
- 🟠 **reconciliation DONE (2026-07-19); durable fix D4c still open.** The "record a completed cell-by-cell
  reconciliation" branch is satisfied — `make_3d_tables.py --check` (3D) and `make_results_tables.py`
  regenerate-and-diff (2D) both match every row (§2). **D4c proper — `\input` the generator instead of
  hand-copying — remains open**; until it lands the C7-/E1-class drift risk recurs on every table edit, so
  re-run both checks before submission. Src: `paper-revision-plan.md D4c`.
- ✅ **VERIFIED (2026-07-19) — all interpolation-reference rows.** 3D (8): matched by `make_3d_tables.py --check`
  against `interp_reference3d.json`. 2D (12) + Cocquet ℙ₁/ℙ₂ × α₀∈{0.5,0.1}: **freshly regenerated**
  (`run_interpolation_reference.jl` — a pure interpolate-and-integrate pass, no solver) and matched to printed
  precision — 16/16 for the main 2D set + Cocquet@0.5, plus the 4 Cocquet@0.1 velocity cells (ℙ₁ 1.13e-4 / 1.21e-1,
  ℙ₂ 1.82e-6 / 4.19e-3) via a temp α₀=0.1 variant (valid because the Cocquet MMS reuses the main field — the @0.5
  reference matched to the digit). The two-finest-mesh slope rule (stated in the caption **and** used in the
  computation), the shared finest mesh (N=320 ladder), and the shared `calculate_normalized_errors` functional
  (`mms_error_norms.jl`, D5c) are all confirmed. Src: §5/§8/B1.
- ✅ **VERIFIED (2026-07-19); caption re-scoped 2026-07-30 — first table caption (`tab:Linear2DL2`,
  [1387](../theory/paper/article.tex#L1387) / [1675](../theory/paper/article_v2.tex#L1675)).** It now states only
  what the table needs to be read unambiguously — element and norm, the two-finest-mesh slope rule, and that the
  parenthetical is the **worst-case rate of the analysis** — so that the table could drop the `\footnotesize` +
  `\tabcolsep` overrides that made it the one convergence table in a smaller font (at normalsize the five-line
  caption ran the float 26.27pt past the page). Nothing was orphaned: the symbolic convention (`k_v+1`/`k_p` in L²,
  `k_v`/`k_p−1` in H¹) is restated in place by every caption that defers to it (`tab:3DL2`, `tab:3DH1`, and
  `tab:Genuine3D` in v2); the "dash = no positive rate" gloss moved into `tab:Linear2DH1`, the only 2D table that
  prints one; the regime-dependence caveat (reaction/convection attain one order more for the pressure) is in the
  results prose; and the interpolant rows are self-labelled in the table body. Src: review D4.
- ✅ **VERIFIED / tables-only settled (2026-07-19) — results-section figures.** The paper has exactly one figure —
  `bump_plateau.pdf` (the 1−α porosity field), referenced at [1142](../theory/paper/article.tex#L1142) and present
  on disk; it **renders correctly** (a labeled 3D surface, z-axis 0 / 1−α₀ / 1, caption matches). No standing
  `\Guillermo{Add figures}` note remains and no convergence figures were added — **tables-only** is the settled
  decision (the build succeeds at 66 pp with the figure included). Src: `open-questions.md §4`.

## 4. Editorial / prose

- ✅ **DONE (2026-07-30) — ALL review markup finalized.** Both halves are now closed. Decoloring had
  already been done (the three macros were identity `{#1}`); the remaining textual unwrap removed **1011**
  wrappers — `article.tex` 461 `\amend` + 8 `\Guillermo` + 3 `\Joaquin`, `article_v2.tex` 495 + 8 + 3,
  `asgs_convergence.tex` 6, `elemental_matrices_appendix.tex` 22, `fourier_appendix.tex` 5 — and deleted the
  three `\newcommand`s. (Earlier counts here — 279, then 328 — were snapshots each later prose pass
  invalidated; the file-set also grew by `article_v2.tex`.) The four TODO-bearing author notes
  (`REVIEW: CHECK`, `JUSTIFY`, `CITATIONS`, `CITATIONS FOR THIS STRATEGY`) sat on commented-out draft
  paragraphs, and those paragraphs were then deleted outright by the dead-comment sweep below — so the notes
  no longer exist in the source at all. Gated output-neutral and it held: identical page counts (81/118), newlabel counts (778/980),
  0 undefined/multiply-defined/overfull>1pt, and **zero-line** `pdftotext` diffs on both PDFs; SymPy 636/636.
  Detail + the tooling change (`make_3d_tables.py` emitted *and* parsed `\amend`): `pending-tasks.md` §1c.
- ✅ **DONE (2026-07-30) — commented-out draft fragments deleted from both mains.** Eight blocks, **116 lines
  per file** (identical in v1 and v2), plus one dead trailing fragment: the abandoned Darcy-threshold
  paraphrase (whose opening sentence was already broken) and its non-Darcy aside; the transient/initial-condition
  aside; the alternative-`A_U`-definition remark; the entire dropped "Linearization of the differential operator"
  subsection (generalized fixed point, contraction condition, Picard, Newton–Raphson — 75 lines); the
  `eq:AdjointFlux_commented` equation (**this closes the "commented duplicate label" cosmetic residue** flagged
  in `ChatGPT audit/latest_audit_response.md`); the superseded `eq:VMSWeakFormSystem_commented`; the dropped
  Projection / SGS-linearization subsubsections; and the `%\input{variational_crimes}` placeholder for a section
  never written (no such file exists). Verified first that **no live text references any label defined only
  inside them** and that every removed line was a comment. Structural comments were deliberately kept: preamble
  and macro notes, the SIAM `% REQUIRED` markers, the `v1/v2 DIVERGENCE` flags, the viscous-projector relocation
  note, the authoritative notation convention, the appendix-ordering note, and the commented `\input` that
  documents the App-D twin switch. Same output-neutral gate, same result: 81/118 pp, 778/980 newlabels, 0
  undefined/multiply-defined/overfull>1pt, **zero-line** `pdftotext` diffs; SymPy 636/636; `make_3d_tables.py
  --check` green.
- ✅ **DONE (2026-07-19) — supplement.tex removed.** It was pure SIAM template boilerplate (`\lipsum`,
  "An Example Article", `thm:bigthm`, `tab:foo`) and `article.tex` made **no** `\cref` to any supplement label.
  Removed the `\externaldocument{supplement}` line + comment from `article.tex`, dropped `supplement.tex` from
  `latexmkrc` `@default_files` and the README dependency list, and **deleted `supplement.tex`**. Build re-verified
  green (66 pp / 722 newlabels / 0 unresolved, no xr/supplement warning).
- ✅ **VERIFIED (2026-07-19) — `C_α` symbol clash (D11) resolved.** In the compiled paper: field `C_α` = `eq:CAlpha`
  in `article.tex` + `fourier_appendix.tex`; the porosity-resolution **constant** is `C_{∇α}` in
  `continuity_appendix.tex` (no bare `C_α` there — grep-confirmed). Disjoint, no in-document collision. (The
  standalone companion note + `c1_dimension_note` still use bare `C_α`, but they are separate documents not
  `\input` by the paper — out of scope.)
- ✅ **DONE (2026-07-19) — Part I erratum.** The submitted appendix is correct: phantom `I`/`G_β`/`D_β` removed,
  `V_T` restored (`\amend`), display reads `P + G_αP`. **But** commit 8a644d2's `G_P→G_αP` rename wrapped the
  `G_αP`/`Q_φ` definition LHSs in `\amend{…}`, defeating `assembly_consistency_verification.py`'s parser (it had
  dropped to 3/4 → suite 109/110). **Fixed:** the script now unwraps `\amend{…}` before parsing (one-line `re.sub`,
  invariant unchanged) → back to **4/4 / 110/110** (re-ran). Updated `part_i_erratum.md §3` (`G_P`→`G_αP`), §4
  (rename note), §5 (collision marked **RESOLVED**). Src: `part_i_erratum.md`.
- ✅ **DONE (2026-07-19) — centered_encoding: short section added** (author-directed). A new "Centered dimensional
  encoding" paragraph in §7 ([~1151](../theory/paper/article.tex#L1151)) explains, at reproducibility level, that
  each `(Re,Da,α₀)` cell has a free dimensional scale; a naive `U=1` drives `σ` to ~`10^12` (double-precision edge),
  so the harness centers the coefficients (`√(νσ)=1`, `L=1` ⇒ `ν=1/√(α_∞Da)`, `σ=√(α_∞Da)`, `U=Re/√(α_∞Da)`), a
  strict reparametrization that leaves the normalized errors unchanged. Full `centered_encoding.tex` stays a
  companion note. Src: `open-questions.md §4`.
- ✅ **VERIFIED (2026-07-19) — notation D1/D2 + Fourier (A16/S45-1).** `α_K = α_{∞,K}` (supremum) at
  [842](../theory/paper/article.tex#L842), no "minimum" leftover; every elementwise `τ₁/τ₂/σ̃_α` in the three §6
  limits reads `α_K` (reaction `τ₁∼1/σ` correctly α-free). Fourier appendix gives `K_ij` for general `d` (1/3, 2/3
  relegated to the labeled d=3 instance) and `τ_{ν,1}^{-1}=(2−2/d)…` with the note it equals 1 for d=2;
  `eq:StabilizationParameters` uses `(2−2/d)`, not a bare 4/3. Src: review D1/D2, plan A16.
- ✅ **VERIFIED (2026-07-19) — companion-note fixes (A17).** `osgs_reaction_note` `eq:asymp` reads `τ₁∼1/σ` (no stray
  `α_K`); `velocity_floor_regularization §4` correctly states the harness sets `h_floor_weight=0` and inherits
  `u_base=1e-4` (not 0), making `ε_d` a no-op — confirmed against the code (`SmoothVelocityFloor` call sites,
  `base_config.json`). Src: `paper-revision-plan.md A17`.

## 5. Build / LaTeX

- ✅ **committed — the `\@ifpackageloaded{lineno}{}{\allowdisplaybreaks}` guard** (present at `article.tex:60`;
  the working tree is clean, so it is under version control). Without it, `[review]` + `allowdisplaybreaks`
  collapses display math to ~22 pp with all-`??` refs on TeX Live 2023/macOS. Src: MEMORY `paper-build-fragilities`.
- 🔴 **verify — clean `latexmk` in BOTH review and final (review-off) mode**, each with **0 unresolved refs and
  0 undefined citations**. Healthy final build = **68 pp / 722 newlabels / 0 unresolved** (was 66; the 2026-07-19 review pass added
  ~2 pp of `\amend` prose — the page count drifts with prose, but **722 newlabels / 0 unresolved** is the invariant). Reconcile the stale
  "43 pp" in `theory/README.md:11` and `open-questions.md §4`.
- 🔴 **open — produce the submission build with review markup OFF** (`\documentclass` without `[review]`,
  [article.tex:2](../theory/paper/article.tex#L2)) so `lineno` is off, `allowdisplaybreaks` activates,
  pagination is correct, and colors resolve. Verify: no line numbers, no colored text in the final PDF.
- 🟠 **verify + commit — the `latexmkrc` fix** (uncommitted, +47/−10): works around latexmk 4.79 not expanding
  `%B` in `$aux_dir`, uses `@ARGV` basename, keeps SyncTeX on. Src: MEMORY `paper-build-fragilities`.
- 🟠 **open — commit the two modified files** (`theory/paper/article.tex`, `theory/paper/latexmkrc`) so the exact
  healthy-build sources are under version control. Src: `git status`.

## 6. Provenance / reproducibility

- ✅ **VERIFIED (2026-07-19, §B pass) — the entire 3D section traces to a certified, committed config+result at `c₁=16k⁴`.**
  `make_3d_tables.py --check` matches every `tab:3DL2`/`tab:3DH1` value to the on-disk DBs; the drivers, `element_c1.jl`,
  and the `nested_red_base_lc0.200_alg1.msh` mesh are git-tracked (the DBs stay gitignored/regenerable, by design). The
  original `c1x4` raw data was lost (gitignored `results/`); D1c added `c1_multiplier` to schema/config, D3c
  committed the nested_red base mesh. Confirm every 3D number came from the certified re-run through the official
  path (not the lost data), and that **both** the regular-Kuhn and irregular drivers/configs are committed
  (`pending-tasks §6e`). Src: `paper-revision-plan.md §0.4c/C1r/D1c/D3c`.
- ✅ **VERIFIED (2026-07-19, §B) — every reported number (2D/3D/Cocquet) via the official test path** (3D `--check` OK; 2D `run_test.jl`+`test_config.json` and Cocquet `run_test.jl`+3 configs git-tracked, DBs present on disk), single canonical results
  leaf, no forked `*_corner` side-DBs merged, no plotter/analyzer reading non-official files. `c₁=16k⁴` must have
  a production config representation (`get_c1_c2` is dimension-blind `4k⁴`; `16k⁴` arrives via `c1_multiplier`).
  Src: CLAUDE.md reproducible-results; official-results-path rule.
- ✅ **VERIFIED (2026-07-19, §B) — 3D mesh reproducibility (D3c):** committed `nested_red_base_lc0.200_alg1.msh` (git-tracked, confirmed) + gmsh 4.9.3
  provenance; `load_or_build_base_mesh` prefers the committed file; family regenerates deterministically
  (425→3400→27200); regular Kuhn family is code-generated. Src: `paper-revision-plan.md §0.4c/D3c`.
- 🟡 **likely-done — `ε_M`/`ε_C` persisted per mesh** (D6c, applied, inert). Src: `paper-revision-plan.md D6c`.

## 7. References / citations

- ✅ **DONE (2026-07-19) — the c₁-derivation footnote no longer refers to a separate document.** Per author
  direction (no unpublished separate documents referenced in the paper), the forward-reference *"a detailed
  derivation will be reported separately"* ([1445](../theory/paper/article.tex#L1445)) was removed and replaced by
  an explanation of the **checks** confirming ℙ₂ tetrahedra need the larger c₁: a numerical evaluation of the
  discrete coercivity constant on the Kuhn meshes is negative at `c₁=4k⁴` (deepening under refinement) and positive
  at `16k⁴`, and the convergence study stalls above the interpolation floor at `4k⁴` but recovers the optimal rates
  at `16k⁴`. No absolute `ĉ²` values were added (they would complicate the "sits just below" framing).
  `c1_dimension_note.tex` stays a companion note, unreferenced by the paper. **Whole-paper forward-reference scan:**
  this was the *only* reference to unpublished separate work; the remaining "future work" mentions
  ([237](../theory/paper/article.tex#L237) transient case, [826](../theory/paper/article.tex#L826),
  [1681](../theory/paper/article.tex#L1681)) are standard and cite only published works. Build green. Src: review D8.
- ✅ **VERIFIED (2026-07-19) — 0 undefined citations.** The build reports 0 undefined citations, and a key-by-key
  reconciliation shows **34 cited keys ↔ 34 `\bibitem`s, exact match** — nothing cited-but-missing, nothing orphaned
  (the 7 extra `\cite` matches were commented-out lines — **as of 2026-07-30 those lines are gone**, so a bare
  `\cite` grep now matches the real cites exactly). All listed works resolve (codina 2001/2008/2018,
  villota2019, cocquet2021, badia2020/verdugo2022 gridap, codina1993, nillama2022, hughes2007). Src: `theory/README.md`.

## 8. Formal proof (Coq)

- ✅ **VERIFIED (2026-07-19, §B) — the paper's stability condition matches the machine-checked margin.** Confirmed
  consistent: `StabilityAlgebra.v` has the coercivity coefficient `C := 1 − ξC_inv²/c₁` (positive iff `c₁ > ξC_inv²`,
  the sharp threshold); the paper's `c₁ > 2ξC_inv²` ([928](../theory/paper/article.tex#L928), with ξ>2) is the stronger
  *sufficient* choice, forcing `C > ½` (a C_inv-free floor). `StabilityAlgebra.v`
  proves the *sharp* positivity threshold is `c₁ > ξC̄_inv²` (a factor 2 below the paper's `c₁ > 2ξC_inv²`
  at [932](../theory/paper/article.tex#L932)/[943](../theory/paper/article.tex#L943) — so the paper's condition
  is sufficient, not necessary); `C_stab_margin` needs `c₁ > 2ξC̄_inv²` for a `C̄_inv`-free floor, with the
  **weighted** `C̄_inv = √(dδ_α)C_inv + C_α` the relevant constant. Reconcile with "16k⁴ sits just below the
  Kuhn threshold". Src: `findings.md §8`; `coq_coverage.tex`.
- ✅ **DONE — ran `./run_all.sh`: the whole tree compiles + coqchk kernel re-check pass, ZERO `Admitted`, ZERO
  `Axiom`.** The tree is now **24 modules** (the 2026-07-19 note said 18 — the **OSGS Coq chain** `Osgs*.v` +
  `NonVacuityOsgs.v` was added alongside the article_v2 App-D integration), so there are now **eight** headline
  abstract theorems — the four ASGS (`abstract_stability/continuity/continterp/convergence`) **and** the four OSGS
  (`abstract_osgs_stability/continterp/consistency/convergence`). `Print Assumptions` on the ASGS headlines returns
  only the 3 stdlib axioms (`sig_not_dec`, `sig_forall_dec`, `functional_extensionality_dep`). "Machine-checked"
  non-vacuity is witnessed for 3-of-4 ASGS theorems (`abstract_continuity`'s witness gap disclosed) plus the OSGS
  chain (`NonVacuityOsgs.v`). **Two notes:** (i) the tree compiles clean under **Rocq 9.1.1** (this environment,
  not the Coq 8.18 named in CLAUDE.md — only a `From Coq`→`From Stdlib` deprecation warning); (ii) `run_all.sh` had
  a **bash-3.2 portability bug** (`mapfile` is bash≥4; macOS ships 3.2) — **fixed** with a portable `while read`
  fill. Src: `findings.md §8`; CLAUDE.md Coq gate.
- 🟡 **verify — amendment F8** (the `eq:winv-conv` label moved to the convective line; `eq:winv-gradp` added; 4
  call sites re-pointed) is in the submitted appendix. Src: `AUDIT.md F8`.
- 🟡 **verify — no over/under-claim of implemented-vs-analyzed τ₂** (S45-2: `eq:Tau2` full vs `eq:Tau2Final`
  analyzed+implemented; Coq `abstract_convergence_implemented` covers it) and that σ=0 (pure NS) is admissible.

## 9. Reviewer-demand gaps (from the adversarial critique — add these)

- 🟠 **open — code/data-availability statement is ABSENT.** No `github`/`zenodo`/`availab`/`reproducib`
  statement in `article.tex`, though the thesis is reproducibility and there is a large public code + Coq base.
  SIAM (SISC RCR/badges; SINUM/SIMAX) expects one. Add a code/data-availability statement.
- ✅ **VERIFIED (2026-07-19, §B) — the conclusion's TH-vs-VMS claim** ([1675](../theory/paper/article.tex#L1675)/[1664](../theory/paper/article.tex#L1664)):
  "remains convergent in the convection-dominated regime in which the unstabilized Galerkin Taylor–Hood velocity
  does not" **is backed by the `tab:CocquetMMS` *n.c.* rows** — the TH velocity is non-convergent, O(1) (FME
  5.14e-1 / 4.02e-1 in L², 6.34 / 4.96 in H¹ at Re=10⁵), while the equal-order VMS converges. Consistent with
  `cocquet-form-mms-status.md §4.3`.
- 🟡 **nice — MSC codes:** `65M60` (evolution equations) is odd for a stationary problem; consider a 76-series
  porous/fluid code (e.g. 76S05). Funding/acknowledgments and other MSC codes are present (verified).
- 🟡 **verify — solver-disclosure sentence** ([1124](../theory/paper/article.tex#L1124)): the Newton–Krylov
  acceleration "does not affect any of the reported errors" — stays true given the OSGS-P2 finest cells needed
  `osgs_p2_precond_c1_mult=4` (preconditioner-only, root-preserving), but pre-empt the referee probe.

---

## 10. External AI revision (`docs/archive/final_AI_revision.md`, 2026-07-19) — per-point assessment + new items

A second external AI reviewed a **~2-h-stale** version of `article.tex`. Every point was re-verified against the
**current** paper by a 19-agent workflow (each claim read against the source + appendices, plus an independent
recheck pass that re-derived the math/arithmetic and read the driver code); the two math/numeric findings were
additionally adjudicated by hand. **Headline: no new blockers.** The AI's two "blocking" items dissolve — the
"missing Fourier appendix" is a non-issue (the reviewer lacked the file) and the "Damköhler off by 10⁵" is
**false** of the current code (it describes an already-fixed bug); what survives there is an important *presentation*
fix. Net: **~16 important + ~20 nice-to-have genuinely-new items** below; the rest were already tracked, already
fixed, or invalid (§10.C).

**✅ APPLIED 2026-07-19 (this session).** All 🟠 items in §10.A and the safe 🟡 items in §10.B
were applied to `article.tex`/appendices, each change wrapped in `\amend{}`; build re-verified
green (**68 pp** / 722 newlabels / 0 unresolved / 0 undefined citations / 0 bib warnings — the 66→68 bump is the
added prose). **F1 and F2 are now additionally machine-checked** by the new
`proof_verification/sympy/display_consistency_verification.py` (suite **242/242 across 17 scripts (2026-07-21 coverage audit; zero further algebra errors across all 369 displays)**); why the
existing machinery missed them and how to close the gap is in
`proof_verification/verification-gap-coverage.md`. **Second no-re-run pass (2026-07-19) — now applied:**
IA-5e (`Π^S∇u=∇^S u`), F14d (`λ`→`μ` eigenvalue), F14e (`U` disambiguated from the combined unknown), 9b (α and
∇α evaluated analytically at the quadrature points in the 2D/3D MMS — code-verified in `run_test.jl` /
`run_simulation.jl`), and M5 (retyped `Codina2015OnSM` as `@misc`; the empty-booktitle bib warning is gone).
**F14c assessed and left unchanged** — the pull-out `Π̃[τ⁻¹Ũ]=τ⁻¹Ũ` is *exact* for ASGS (Π̃=I) and the step is an
explicitly heuristic *motivation* ([article.tex L720]), so weakening it to `≈` is unwarranted (the AI's "≈ under
either convention" is wrong for ASGS). **Still deferred, needing author data (not re-runs):** M6 (DOIs — SIAM
tolerates their absence) and M8 (Codina/de-Pouplana emails). (F9a and F9c — the τ
theory/practice gap and the stopping-tolerance sentence — **were applied** after an accuracy re-check of this banner.)

### 10.A New — important (🟠) — ✅ ALL 15 APPLIED & GREP-VERIFIED 2026-07-19 (none needs a re-run)

> Every item below is a paper/prose edit built from the *existing* table data — no simulation re-run
> required — and each was confirmed present in the committed `theory/paper/article.tex` by grepping its
> distinctive phrase (F1 sign, F2 `\amend{2}\nu` ×3, F3, F5, F9a, F9c, F10, F11a/b, IA-1, IA-2, C2, C3,
> C4, RW-3). Build green (68 pp / 722 newlabels / 0 unresolved / 0 undefined).

- 🟠 **F1 — sign typo, eq:weak_form_eliminated_subscales ([512](../theory/paper/article.tex#L512)).** The subscale
  term is printed with a **minus** `- ⟨𝓛𝓛̃⁻¹𝓡U_h, V_h⟩`; the correct sign is **plus** (substituting
  `Ũ = 𝓛̃⁻¹𝓡U_h` into eq:weak_form_resolved gives `+`, matching the very next equation eq:simplified_weak_form_resolved
  ([525](../theory/paper/article.tex#L525)) and eq:OSGSProblem). Motivational-only — does **not** propagate to the
  method — but L518 asserts it "does not entail any approximation", so a referee will read it. Fix `−`→`+`. (Both
  the primary and the independent-recheck agent, plus a hand derivation, agree.)
- 🟠 **F2 — missing factor 2 in two appendix displays** (`elemental_matrices_appendix.tex`). In eq:StabilizationLVLU
  the cross-terms `ν ϕ ∇v ∇β` (L14) and `ν ϕ ∇u ∇β` (L16), and the test-slot `ν ϕ ∇v ∇β` in eq:StabilizationLVF
  (L28), should each read `2ν …`: from `2∇·(ανϕ∇u)=α[ν Δ̄u + 2ν ϕ∇u·∇β]` with `∇α=α∇β`, and `Δ̄` (L19) already
  carries its 2's while `ϕ∇u` carries a ½. **Verified DISPLAY-ONLY** (hand-checked the assembly): the assembled
  entries `A_Gβ` (L144, symmetrized two-term `(∂_i N^b ∂_j α + ∂_m α ∂_m N^b δ_ij)` = `2ν Π^S∇u∇β`), `A_Dβ` (L145,
  the `2/3` deviatoric coefficient), and the whole `G_β/D_β` family **do** carry the factor 2 — so the
  implementation/reference is correct; fix the 3 display coefficients only. *(The workflow recheck agent's claim
  that the assembly is also wrong is itself a misread of the symmetrization — do not act on it.)*
- 🟠 **F3 — DBF Da/σ presentation** ([1566](../theory/paper/article.tex#L1566); eq:DBFResistanceTerm
  [266](../theory/paper/article.tex#L266); eq:CocquetMMSReaction [1563](../theory/paper/article.tex#L1563) vs
  `Da=σL²/(α_∞ν)` [988](../theory/paper/article.tex#L988)). **The code is correct** — the harness scales the Ergun
  coefficients by ν (`a_scale=ν/L²`, `b_scale=ν/(U L²)` in `CocquetFormMMS/run_test.jl`), so `Da(α₀)≈2, 40` is
  genuinely Re-independent (the AI's "actual Da≈2e5/4e6" describes a *fixed* bug). Two real presentation defects
  survive: (i) the printed `σ=a(α)+b(α)|u|` with dimensionless `C_a=0.30` **omits the ν-scaling** the code applies,
  so a literal reader recovers the ν-free case where `Da∝Re` — state that `C_a,C_b` are dimensionless Damköhler
  coefficients / that the dimensional drag carries ν; (ii) the clause "unlike the reference, in which `a` carries a
  1/Re factor, we fix `C_a=0.30` so that Re and Da can be varied independently" is **backwards** — it is the
  *retained* ν(∝1/Re) factor, not its removal, that keeps Da Re-independent; the real contrast with the reference is
  the fixed numeric constant. Flagship comparison; a numerical-analysis referee will probe this.
- 🟠 **F10 — DBF closing sentence overstates the α₀ attribution** ([1660](../theory/paper/article.tex#L1660)).
  "consistent with the α₀^{-1/2} dependence" — but the L² pressure FME grows **~6.2–6.5×** as α₀:0.5→0.1 (ASGS 6.26,
  OSGS 6.52, P2 6.45/6.21 — verified), exceeding **both** α₀^{-1/2}=2.24 **and** 1/α₀=5; the pressure interpolant
  reference is α₀-independent so nothing absorbs the excess (unlike velocity, whose reference grows 3.47×). Also,
  lowering α₀ here simultaneously raises Da ~2→40 (eq:CocquetMMSReaction), so the two rows **conflate α₀ and Da**.
  Reword to the one-sided-bound spirit (bounded by, not equal to) OR note the α₀↔Da confound; the velocity part is fine.
- 🟠 **F5 — define the OSGS/ASGS "excess"** ([1174](../theory/paper/article.tex#L1174)). "its absolute size agrees
  to within 1% at α₀=0.5 and 0.05" holds **only** for the *quadrature* excess `√(e_OSGS²−e_ASGS²)` (0.1179 vs 0.1167,
  ratio 1.010 — verified from tab:Linear2DH1); the naive difference is 0.0878 vs 0.035 (ratio 2.5), so a referee
  subtracting sees an apparent falsehood. Add a one-clause definition, e.g. "defining the excess as `(e_OSGS²−e_ASGS²)^{1/2}`".
- 🟠 **F9a — τ theory/practice gap missing from the scope paragraph** (~[1098-1113](../theory/paper/article.tex#L1098)).
  The proofs assume τ₁ **elementwise-constant** (`continuity_appendix.tex:460`; the interior-face jump hypothesis at
  [956](../theory/paper/article.tex#L956)); the runs use τ **variable within elements** ([828](../theory/paper/article.tex#L828)).
  Both disclosed separately, but the delimitation list — whose job is to enumerate exactly these idealizations — omits
  it. Add one clause.
- 🟠 **F9c — nonlinear stopping tolerance not reported.** The solver paragraph (~[1124](../theory/paper/article.tex#L1124))
  quotes no residual/stopping tolerance, yet FMEs are compared to interpolation references at the **3rd significant
  digit** ([1164](../theory/paper/article.tex#L1164): "3.50 vs 3.499"). State the tolerance so the 3-sig-fig
  comparisons are not attributable to solver noise. (Fits with the §9 solver-disclosure item.)
- 🟠 **F11a — 3D irregular-mesh attribution** ([1457](../theory/paper/article.tex#L1457)). Replace/augment the
  unverified "element-quality tail" with the paper's own stronger **in-table** argument: the nodal interpolant shows
  the *same* depressed rates on the irregular sequence (P1 L²=1.83, H¹=0.71; P2 L²=2.67, H¹=1.52 vs regular
  1.90/0.94/3.20/2.22) and the solver slopes track it — so the depression is a property of the mesh **sequence**, not
  the formulation. Point at the interpolant rows of tab:3DL2/3DH1.
- 🟠 **F11b — 3D OSGS-vs-ASGS pressure direction reversal** ([1457](../theory/paper/article.tex#L1457) + conclusions
  [1673](../theory/paper/article.tex#L1673)). Unlike 2D, in 3D the OSGS pressure carries **larger** absolute errors
  than ASGS despite comparable/better rates (regular P1 L²: OSGS 4.55e-2 vs ASGS 2.87e-2), its H¹ saturating at the
  O(1) 1.29 floor. Add one honest sentence in §7.2-3D, and qualify the conclusions' "somewhat better pressure
  convergence" with the matching 3D caveat so it does not read as unconditional.
- 🟠 **IA-1 — abstract undersells** ([204](../theory/paper/article.tex#L204)). Methods-only: omits the
  porosity-weighted a-priori stability/convergence analysis (robust in Re & Da), the 3D tetrahedral campaign, and the
  equal-order-vs-Taylor-Hood DBF comparison. Add a clause each — those are the SIAM-reader selling points.
- 🟠 **IA-2 — unhedged universal claim** ([241](../theory/paper/article.tex#L241)). "the only precedent of a
  stabilized finite element method: the Local Projection Stabilization" contradicts the paper's own `nillama2022`
  citation two paragraphs earlier (VMS *is* a stabilized FEM for porous NS). Hedge ("to our knowledge") and scope to
  the **variable-porosity** problem.
- 🟠 **C2 — conclusion robustness vs excluded corner** ([1669](../theory/paper/article.tex#L1669) vs
  [1160](../theory/paper/article.tex#L1160)). "just as robust … as it is well-established to be" is unqualified while
  §7 excludes the (Re,α₀)=(10⁶,0.05) corner as a coarse-mesh fold with **no discrete solution** on the coarser meshes.
  Add one clause acknowledging the coarse-mesh solvability limit (framed as a resolution limit, per §7).
- 🟠 **C3 — conclusions omit two headline contributions** ([1666-1673](../theory/paper/article.tex#L1666)). Add
  (i) the empirical headline — both variants' velocity error sits on the nodal interpolant where no term dominates
  ([1164](../theory/paper/article.tex#L1164), "stronger than optimality"), low-α₀ degradation inherited from the
  exact solution; and (ii) the porosity-weighted elementwise `(α_K/α₀)^{1/2}` estimate as a **named** theoretical
  result. L1671's "absolute errors are very stable" is only a vague gesture at (i).
- 🟠 **C4 — conclusion velocity claim needs its exception** ([1673](../theory/paper/article.tex#L1673)). "the two
  variants behave very similarly for the velocity" — but §7 ([1174](../theory/paper/article.tex#L1174)) calls the
  reaction-dominated P1 ASGS/OSGS H¹ gap (~3.5×) "the largest velocity discrepancy … in the campaign" and leaves it
  unresolved. Add one clause.
- 🟠 **RW-3 — two-mesh-slope fragility not acknowledged** ([1457](../theory/paper/article.tex#L1457)). All tabulated
  slopes are two-finest-mesh estimates; the regular 3D family's two finest meshes have **h-ratio ≈1.2** (verified
  0.05964/0.04970), amplifying slope noise ~5.5×, so "essentially optimal L²/H¹ orders" for P2 velocity rests on a
  fragile estimate. Either add a pre-asymptotic/ratio-sensitivity clause, or (cheaper, stronger) reframe L1457 around
  the slope-noise-free FME-vs-interpolant match already in the tables.

### 10.B New — nice-to-have (🟡) — ✅ APPLIED 2026-07-19 (F14c assessed → no change; M6/M8 need author data)

> Two passes: the first applied F4, F6, F7a/b, F13-2, F14a/b/f/g, IA-3/4/5/7, C1/5/6, M7, M9, f8-4, RW-5; the
> second (no re-runs) applied IA-5e, F14d, F14e, 9b, M5. F14c left as-is (assessed defensible). Only M6 (DOIs)
> and M8 (author emails) remain — both need author-supplied data, not re-runs.

- 🟡 **F4 — "optimal" for superoptimal slopes.** (a) fold para [1160](../theory/paper/article.tex#L1160): "converges
  at the optimal rates (P1 2.99–3.03 L²)" — P1-L² optimum is 2, so this is *superoptimal* pre-asymptotic; say "at or
  above the optimal rates". (b) DBF corner [1662](../theory/paper/article.tex#L1662): "recovers accurate,
  optimally-converging solutions" sits one sentence before the pre-asymptotic hedge — drop the rate qualifier.
  (c) 3D [1457](../theory/paper/article.tex#L1457): "essentially optimal" is actually *accurate* (at/above optima) —
  no fix needed, optional strengthening only (note the P2 velocity FMEs sit on the interpolant).
- 🟡 **F6 — three-sig-digits scope + table precision** ([1164](../theory/paper/article.tex#L1164)). "agree to three
  significant digits at α₀=0.5 … for both elements" overclaims for Q2 (4.29e-4 vs interp 4.28e-4, 3rd digit differs,
  0.23%) — scope "three significant digits" to P1 (or "a fraction of a percent"). Separately, the quoted 3.499e-2
  carries one more digit than the table prints (3.50e-2) — round it or note it is the unrounded reference.
- 🟡 **F7a — projection wording** ([638](../theory/paper/article.tex#L638)). "their projection is exactly zero" is
  imprecise: for constant σ the reaction term σu_h ∈ FE space, so `Π_h(σu_h)=σu_h`; what vanishes is the fluctuation
  `(I−Π_h)(σu_h)`. Reword to "annihilated by (I−Π_h) …". Meaning is recovered by the adjacent clauses → precision-only.
- 🟡 **F7b — footnote wording** ([610](../theory/paper/article.tex#L610)). "force the boundary DOFs to mirror the
  prescribed Dirichlet data": by the paper's notation `𝒳_{h0}` (subscript 0) is the **homogeneous** zero-trace space,
  so constraining forces the projection's velocity trace to **zero** on Γ_D, not to the prescribed data. Reword; the
  O(1)-boundary-residual consequence stands.
- 🟡 **F13-2 — Neumann-datum trace wording** ([465](../theory/paper/article.tex#L465)). "`g∈H^{-1/2}(Γ_N)^d`, dual of
  the traces on Γ_N of H¹(Ω)" — on a proper boundary piece the constrained trace space is `H^{1/2}_{00}(Γ_N)`; the
  `H^{-1/2}` label is acceptable shorthand but tighten it once the F13 `V₀` fix lands.
- 🟡 **F14 batch — small alignment items.** (a) [1162](../theory/paper/article.tex#L1162) "the one appreciable
  exception" undercounts — [1174](../theory/paper/article.tex#L1174) names **two** appreciable ASGS/OSGS velocity
  discrepancies; reword to "the main exception". (b) [1164](../theory/paper/article.tex#L1164) "the strongly
  reaction-dominated **column**" → "rows"/"regime" (tables vary Da down rows). (c) [556](../theory/paper/article.tex#L556)
  the internal "=" in `ϕ[τ_K⁻¹Ũ]=τ_K⁻¹Ũ` is exact only for ASGS; for OSGS use the paper's "≈" (footnote L539).
  (d) λ overloaded — Λ-weight scale ([711](../theory/paper/article.tex#L711)) vs eigenvalue in `spec_{Λ⁻¹}`
  ([731](../theory/paper/article.tex#L731)); rename the eigenvalue (μ). (e) U overloaded — scalar velocity scale
  ([990](../theory/paper/article.tex#L990)/[1114](../theory/paper/article.tex#L1114)) vs combined unknown `U=[u;p]`;
  disambiguate. (f) [1114-1115](../theory/paper/article.tex#L1114) `\text{sin}/\text{cos}` → `\sin/\cos` (both occur).
  (g) preamble [115](../theory/paper/article.tex#L115) amendcolor comment says "dark green" but `rgb{0.58,0,0.83}` is
  violet — fix alongside §4 flattening.
- 🟡 **IA-3 — intro advantages list** ([241](../theory/paper/article.tex#L241)) omits the key selling point:
  residual-based stabilization cures LBB **and** convection-dominance simultaneously (an inf-sup pair alone controls
  only LBB) — foreshadow §8's O(1)-stagnant TH velocity ([1662](../theory/paper/article.tex#L1662)).
- 🟡 **IA-4 — dead stabilized-Darcy citations.** The commented sentence at [218](../theory/paper/article.tex#L218) is
  the *only* use of {Masud2002ASM, Juanes2005AVM, Codina2015OnSM, Braack2011EqualorderFE}. Decide one way: reinstate
  (adds Darcy context + activates 4 entries — these authors referee such papers) or drop the 4 from `references.bib`.
- 🟡 **IA-5 — intro/abstract grammar batch.** "concentrate in"→"on" (L237); trim redundant "no need for revisiting the
  theory was required" (L239); "associated to"→"associated with" (L204, L579); "(commutative)"→"commuting" (L256); the
  operator/action conflation `Π^S = ∇^S u` (L262, L264); clarify "in its most recent form" (L243).
- 🟡 **IA-7 — keywords** ([208-210](../theory/paper/article.tex#L208)): add "porous media" (domain/title) and "ASGS"
  (currently only OSGS listed, though both variants are tested throughout).
- 🟡 **C1 — conclusions calque** ([1669](../theory/paper/article.tex#L1669)): "robust **in front of** extreme
  variations" (Spanish/Catalan *frente a*/*davant de*) → "under" / "in the face of".
- 🟡 **C5 — 3D caveat on OSGS pressure** ([1673](../theory/paper/article.tex#L1673)): "in some regimes" partly covers
  it, but note that in 3D "better convergence" means **rate**, not error (OSGS pressure FME larger; H¹ barely
  converges — see F11b).
- 🟡 **C6 — sharpen "absolute errors are very stable"** ([1671](../theory/paper/article.tex#L1671)) with the
  interpolation-reference statement from §7 ([1164](../theory/paper/article.tex#L1164)).
- 🟡 **9b — MMS α evaluation unspecified.** In the 2D/3D runs α is analytic (eq:PlateauBumpFunction); state whether
  α/∇α are evaluated exactly at quadrature or interpolated (the Cocquet run says nodal interpolation,
  [1568](../theory/paper/article.tex#L1568); the conclusion [1681](../theory/paper/article.tex#L1681) leans on
  "interpolation of α does not spoil convergence").
- 🟡 **M5 — `Codina2015OnSM` booktitle.** If the Darcy sentence (L218) is reinstated, add a `booktitle` to
  `@inproceedings{Codina2015OnSM}` in `references.bib` (currently title/author/year/url only).
- 🟡 **M6 — DOIs sparse.** Only ~5 `references.bib` entries carry a DOI. SIAM tolerates; add opportunistically for the
  camera-ready.
- 🟡 **M7 — lone `\eqref`.** The single `\eqref{eq:DimensionlessMomentumEquation}` at
  [990](../theory/paper/article.tex#L990) amid 203 `\cref`/`\Cref` uses → change to `\cref` for uniformity.
- 🟡 **M8 — author emails.** In `shared.tex:37-40` only Casas/González have `\email`; Codina and de-Pouplana have none.
  Add them or designate a corresponding author (SIAM convention).
- 🟡 **M9 — copy-edit.** "spatially-inhomogeneous"→"spatially inhomogeneous" (L249); "such term"→"such a term" (L862);
  "applied and further developed to"→"extended to" (L697); rewrite the "since, given that … it is only important that"
  stack (L828); complete the fragment footnote (~L226).
- 🟡 **f8-4 — cleanup.** Delete the orphaned `theory/paper/supplement.pdf` (143 KB) + stale
  `theory/paper/latex compilation/supplement/` intermediates left after `supplement.tex` was removed — untracked,
  referenced by nothing, purely cosmetic.
- 🟡 **RW-4 — response-letter prep (no paper change).** Pre-empt two likely referee objections: (a) the campaign is
  entirely manufactured-solution based (an "engineering relevance" objection given the intro's applications framing);
  (b) no computational-cost comparison (OSGS overhead vs ASGS / vs Taylor–Hood) — §7.3's practical conclusion (L1675)
  is accuracy-only. Optional to address in-paper; sufficient to have a prepared response.
- 🟡 **RW-5 — optional future-work sentence** (~[1682](../theory/paper/article.tex#L1682)): a *discriminating*
  bound-sharpness test — support an oscillatory error component where α=1 (off the low-porosity plateau) — would turn
  the "cannot select between the weighted and uniform α₀ bounds" remark into a decidable experiment for one
  manufactured solution.

### 10.C Assessed — no new action (invalid / moot / already tracked)

- **Fourier appendix "missing" (AI finding 8): INVALID / moot.** `fourier_appendix.tex` exists (121 ln),
  `\label{app:FourierTau}` resolves (App. B, p.52 in `article.aux`), and `\externaldocument{supplement}` is already
  removed (§4). Only residual is the orphaned `supplement.pdf` → f8-4 above.
- **F3 "Da actually ≈2e5/4e6, off by 10⁵": FALSE** of the current code (describes a *fixed* bug). Only the
  presentation fix (§10.A F3) survives.
- **~17 unused `.bib` keys "will ship" (AI): INVALID.** The paper uses BibTeX `\bibliography{references}` (L1693),
  which emits only cited entries; the ~18 template leftovers never enter the `.bbl`. Optional cosmetic `.bib` tidy only.
- **MSC codes (IA-6): already §9** — enrich that bullet to *replace* 65M60 (evolution — the paper is stationary) with
  76S05 (flows in porous media) + 76M10 (FEM in fluids); consider 65N15; keep 65N30/65N12.
- **Code/data-availability (9d): already §9.**  **34 cited ↔ 34 bibitems, 0 undefined (M4): already §7.**
- **Analysis-scope weakness (linearized / constant-σ / all-Dirichlet / ASGS; ASGS pressure one order suboptimal):
  already stated in the paper (delimitation [1098-1109](../theory/paper/article.tex#L1098)) and tracked (§1 D7).**
- **Reorg of §7 / add a convergence figure / move tables to a supplement (AI): moot** — tables-only is the settled
  decision (§3) and `supplement.tex` is removed (§4).

### 10.D Full-paper consistency + prose re-read (2026-07-19) — ✅ DONE

An 8-agent section-by-section re-read (consistency vs tables/cross-refs + prose flow/elegance/clarity) *after* all
the §10.A/§10.B edits. **Two real bugs in the review edits were caught and fixed:**
- **F10** ([1660](../theory/paper/article.tex#L1660)) was self-contradictory ("*exceeding* both bounds … which it
  therefore *obeys*"); rewritten to "grows more than the α₀^{-1/2} prefactor alone, the remainder via the
  velocity-error coupling; not a clean α₀ probe given the α₀↔Da confound."
- **RW-5** ([1683](../theory/paper/article.tex#L1683), the discriminating-test future-work sentence) had **inverted**
  α-geometry — the weighted factor √(α_K/α₀) is **1 on the plateau** and **α₀^{-1/2} where α=1**, so my "plateau: bounds
  coincide / α=1: weighted predicts no loss" was backwards and contradicted `rem:WeightedVsUnweighted`. Rewritten
  correctly (plateau → weighted=1 vs uniform 1/α₀, both upper bounds; α=1 → weighted α₀^{-1/2} vs uniform 1/α₀, decidable).

**Pre-existing consistency issues fixed:** L619 (τ_K nonlinearity wrongly scoped to OSGS-only — τ depends on u in both
variants, only π_h is OSGS-specific), L820 (λ mislabeled "length scale" — it is velocity²), L1174 ("provable for OSGS"
→ "the same coercivity argument would give for OSGS", since no OSGS estimate is proved). **Plus ~15 prose/clarity fixes**
(test-first argument order in eq:weak_form_eliminated_subscales, R-operator argument, redundancies, the "In all rigor"
calque, over-precise claims softened to what the tables support). Build green (68 pp / 722 / 0 unresolved / 0 undefined
/ 0 bib warnings). **Still open (from the re-read):** the `\Guillermo`/`\Joaquin` macros wrap *retained* prose that
renders red/blue — handled by the §4 flatten (redefine to `{#1}`), not a separate fix.

---

### Suggested sequencing

1. **Provenance/build blockers** (§5, §6): commit the modified files, confirm the certified 3D re-run and official
   path. Everything numeric depends on these.
2. ~~**3D table transcription audit** (§3)~~ **✅ DONE (2026-07-19)** — `make_3d_tables.py --check` matches every
   3D `\num`; **C7 and E1 both CLOSED** (§2). The full §2 numerics audit (2D tables, Cocquet, S6-1, reaction-gap)
   is likewise cleared. Re-run the `--check`/regenerate-diff only if a sweep is ever regenerated.
3. **α₀-exponent rewrite** (§0) and the theory-claims pass (§1) — both **✅ DONE** (§0, §1).
4. **Prose/markup/refs** (§4, §7), then the **final review-off build** (§5).
5. **Reviewer-demand gaps** (§9).
6. **External-revision items** (§10): work the 🟠 first — the two typos (F1 sign, F2 factor-2 display), the
   presentation fixes (F3 Da/σ, F10 pressure attribution, F13 function space), and the abstract/conclusions
   (IA-1/IA-2, C2/C3/C4) — before the final review-off build (§5); batch the 🟡 into the copy-edit + markup-flatten
   pass (§4). Note §10.A F2 is **display-only** (assembly/implementation verified correct).

---

## 11. Referee-style revision report on `article_v2` (intake 2026-07-30) — ✅ ALL 16 ADJUDICATED

Source: `docs/article_v2_revision_report.md` (untracked dump, sha256 `48d70da4a2bb99149fe877e8d032a7e6f2191557e1b390baa4c6ab3b1e46c5ee`,
committed verbatim then deleted per `.agents/skills/external-audit-intake`). Items **F1–F16**. Each was verified
against primary sources (live `.tex`, the raw result HDF5s, CrossRef/DOI) by one agent and then adversarially
re-checked by a second whose job was to refute the first — that second pass caught **five** defects in the
proposed patches and is the reason the applied text differs from the report's suggestions in several places.

### 11.A Applied — errors of statement (the report's P1 set)

- **F1 ✅ (real, P1).** §7.1's corner paragraph said the OSGS L² velocity drops "to the reference itself—1.21e-5"
  two clauses after quoting that reference as 2.85e-5. Raw data (`corner_tri_k1_a005.json`, N=768):
  FME = 1.2056e-5 = **0.42×** the floor, i.e. 3.6% above the L²-best-approximation level 2.85e-5/√6 = 1.16e-5.
  Rewritten to "dropping *below* it … a factor 0.42", tied to the √6 caveat the paragraph already carries.
  **Two extras the report missed:** the trailing pressure clause ("about a factor 4 above it in L²") is likewise
  true only at Da≤1 — at Da=1e6 the pressure is 9.10e-7 = 0.53× its 1.71e-6 floor — now stated; and
  `docs/mms/convergence-2d.md`, the doc the paper's wording was transcribed from, carried the same slip (fixed).
- **F2 ⚠️→✅ (report's quote was STALE; a different, real defect underneath).** The report quotes l.887 as citing
  `sec:StabilityOSGS`; the live text says `sec:StabilityASGS` in **both** mains and always has (`git log --all -S`
  is empty). But in **v2 only**, `sec:StabilityASGS` now labels the *umbrella* §5 "…the ASGS **and OSGS** methods",
  so the `\cref` swept the OSGS analysis into a claim true only of ASGS. Retargeted to `sec:StabilityASGSvariant`
  and followed by one sentence naming the three departures (converged not lagged projection, τ-weighted product,
  first-order truncation) with a pointer to `oa:rem:analyzed`. **v1 deliberately unchanged** — there the label is
  ASGS-only, so the plain sentence is correct and `oa:*` would be undefined.
- **F3 ✅ (real, P1).** "[the stabilized P₂/P₁ control] converges at optimal velocity and pressure rates at
  Re=10⁵" is false for velocity in **3 of 4** (norm, porosity) slots: 2.41 vs 3 (L²) and 1.26 vs 2 (H¹) at
  α₀=0.5, 1.65 vs 2 (H¹) at α₀=0.1; only (10⁵,0.1) L² reaches 3.00. Pressure *is* optimal throughout. All slopes
  re-derived from the HDF5. The existing two-mesh disclaimer does **not** cover it — it is scoped to the
  (10⁵,0.1) corner, which is the one cell where L² velocity *is* optimal, while the worst shortfall sits at
  (10⁵,0.5) on a full 6-rung ladder. Rewritten to state the observed rates against their references.
- **F4 ✅ (clarity, not error).** The closing "This comparison does not isolate the effect of the element pair"
  is *literally true* (its antecedent is the pairwise contrast in the sentence it closes — the adversarial pass
  correctly downgraded the report's "flat contradiction"), but landing it two sentences after "the high-Reynolds
  gain is therefore attributable to the convection stabilization itself" reads as self-denial. Rescoped to the
  pairwise contrast and pointed forward to the control.
- **F5 ✅ (real, P1).** γ meant three things: (A1) quasi-uniformity, (O1) OSGS design margin, and the bump
  exponent γ(r). (A1) and (O1) sit in the same subsection. The (A1) γ is **write-only** — 2 tokens, one line,
  never used again — so it was renamed to `C_qu` (1 line per main). The (O1) γ is entrenched across 82 appendix
  lines + `coq_coverage.tex` + ~103 Coq identifiers and was left alone; γ(r) likewise (PlateauBump.v).

### 11.B Applied — accuracy and hygiene

- **F7 (PARTIAL — the report's own reasoning was wrong, the underlying ambiguity was real).** The report placed
  this in the 3D section, about pressure ratios, and concluded "not bracketing". All three are wrong: it is the
  2D section, about L² *velocity* growth, and the pair **does** bracket (ASGS 1.3–2.0 below, OSGS 3.5–4.1 above).
  What is real is that the paper wrote the benchmark as a bare `α₀^{-1/2}`: for a *growth factor between two
  porosities* the prediction is the **ratio** √(0.5/0.05)=√10≈3.2, but a reader substituting α₀=0.05 gets 4.47
  and reaches the opposite conclusion. Benchmark now stated numerically in all **three** places carrying it —
  including §7.3's `α₀^{-1/2}≈2.2`, which was **literally false** as printed (α₀^{-1/2} at 0.1 is 3.16; 2.2 is
  the ratio √5) and which the report never patched.
- **F11 ✅ partially applied.** (iv) "rates approach the interpolation-optimal exponents" → "approach **or
  exceed**" (3D P2 gives 3.33/3.34 against an interpolant row of 3.20). (i) Q2 ASGS pressure "≈1.9 / ≈0.9" →
  ranges "1.8–1.9 / 0.8–0.9" (actual 1.81–1.90 / 0.79–0.93). (iii) "about twenty times" → "some twenty to
  twenty-five times": the sentence quotes **H¹ first**, where the ratio is 23.3–24.3, not the 19.7–20.5 of L².
  (ii) "≈2" for P1 pressure left as is — a joint two-variant claim whose interpolant reference is exactly 2.00.
- **F12 ✅ (real).** App. B l.146 hard-coded "the coefficient 4/3 of τ_{ν,1}^{-1}" while its own eq:ftTauNu gives
  2−2/d and its l.76 says the factor is **1 in 2D** with "nothing to drop". Now states the general coefficient —
  and, per the adversarial pass, keeps the **projector qualifier** both mains carry (2−2/d for `\DSPi`; 2 for
  `I`, `\SPi`, `\DPi`, cf. `rem:ftGenericPi`), marked `\amend{}` like every other correction in that file.
- **F13 ✅ (real).** App. D's "valid since c₁ ≥ 1 under (O1)" does not follow: (O1) gives c₁ ≥ γ²C_inv², so
  c₁ ≥ 1 needs C_inv ≥ 1/γ. Replaced by the unconditional `max{c₁,1}` form, with the parenthetical recording
  that the maximum is c₁ in practice. **Both App. D twins edited** (the SymPy twins gate compares them).
- **F10 ✅ (9 sites, not the report's 5).** The report missed the App. D **clean twin** (`osgs_appendix.tex:408`),
  **v1** (`article.tex:1069`), and three word families it did not scan (`manoeuvre`×2, `cancelling`, `labelled`×3).
  An exhaustive 40-family sweep of all nine typeset files now returns zero British forms in typeset text.
  Deliberately untouched: `nillama2022explicit`'s published title ("stabilised"), "CERCA programme", "Severo
  Ochoa Centre" (proper names), and the `Behaviour` inside a label name (never typeset).
- **F15 ✅ (a), (c), (e-verified); (d) DECLINED.** `\usepackage{lipsum}` removed (zero uses). All **six** table
  captions unified to "worst-case bound rates in parentheses" — note the *first* unification attempt used a
  longer phrase and pushed `tab:Linear2DH1` **4.27pt over the page**, tripping hygiene rule H4; the shorter
  phrasing restores 0 oversized floats. `\headers` **enabled with a shortened running title**: the full title
  overflows every odd page by 146.5pt (measured independently in a minimal document), against a gate budget of 0.
- **F8 ✅ + F9 ✅ (bibliography, 20 entries).** All corrections re-verified against CrossRef/DOI, not taken on
  trust — and every proposed DOI was resolved before use. Highlights: `CodinaBlasco1997` was in the `.bib` but
  never cited (now cited at the first naming, with wording that claims only what is verifiable); the two Gridap
  `doi` fields held **full URLs**, which `siamplain.bst` prepends `https://doi.org/` to — i.e. two **broken
  links in the shipped PDF**, the only defect in the set that reached the reader; Brenner was missing Scott;
  Quarteroni's *translator* was credited as co-author; "Carfagnay"/"Federicoz" were affiliation-marker scrape
  artifacts (confirmed from the publisher PDF *and* independently from the authors' other joint papers — note
  CrossRef itself propagates the error); Hornung is the volume's **editor**; Masud/Hughes, Feijóo (accent +
  middle initial) and "Jean-Baptiste" (which changes the printed initials to J.-B.) all corrected; Bayona Roa's
  compound surname was parsing as "Roa" and Balazi Atchy Nillama's as "Nillama". **All 9 Semantic Scholar
  CorpusID stubs replaced by resolving DOIs** — including `Codina2008FiniteEA`, which the report did not list.
  The 15 SIAM-template leftovers were deleted (verified uncited); genuine but uncited research entries were
  **kept** (they never enter the `.bbl`, and `Codina2004ApproximationOT` is cited by `paper novelties/`).

### 11.C Verified FALSE or declined — **do NOT re-fix**

- **F6 — FALSE. The 9.6e-3 excess is correct.** The report recomputed it from the 3-digit table entries and got
  9.28e-3. From the unrounded errors (`k1/TRI/results.h5`: ASGS 3.526348e-2, OSGS 3.654739e-2) it is
  **9.6020e-3**, which is what the paper prints. The excess is a difference of nearly equal squares, so a
  half-ULP in each table entry moves it ±4%; the other three excess figures (1.18e-1, 1.17e-1, 1.5e-2) *are*
  reproducible from the rounded values, which is why only this one looked wrong. **Changing it to 9.28e-3 would
  make the paper less accurate.** A clause noting the unrounded evaluation was added so the next reader does not
  repeat the round trip.
- **F9(13) — FALSE.** `Hamdan1994SinglephaseFT` does *not* need a colon: the publisher's registered title is
  "…porous channels a review of flow models…", colonless. Adding one would diverge from the record.
- **F9(19) purge — PARTIAL/declined in part.** Uncited entries never reach the `.bbl` (BibTeX emits only cited
  keys), so this is cosmetic, as already ruled in §10.C. Template stubs deleted; research entries kept.
- **F14 — declined (symbol reuse ψ, η).** All four objects are bound variables defined at their point of use and
  no display mixes two meanings. A rename would desync the byte-identical interpolation-error paragraph across
  the mains or touch ~105 lines across the two App. D twins plus `coq_coverage.tex`, for no gain.
- **F15(d) — declined (the "large editorial comment blocks").** `article_v2.tex:317–334` and `848–862` are
  **live in-source documentation**, not author notes: the first names its enforcing gate (rule D12 of
  `projector_algebra_verification.py`) and the relocation to `theory/viscous_projector_note/`; the second is the
  authoritative notation convention, duplicated verbatim in v1. Deleting them would be a regression. (LaTeX
  comments are not typeset, so "SIAM production will query them" does not apply.)
- **F16 — PARTIAL; the report's diagnosis was wrong.** The 1.2/1.5 accuracy factors are **measured**, not
  back-extrapolated: TH at N=160 is 2.685e-6 / 1.437e-5 in the HDF5, giving 1.162 and 1.472. A footnote saying
  "obtained by extrapolation" would have stated a falsehood. Instead the four errors are now quoted inline so a
  reader can divide them. The report's proposed parenthetical was also **factually wrong** (it claimed the
  control's FMEs are all at N=160; at (10⁵,0.1) it is N=320) and was not used.

### 11.D Still open (unchanged by this intake)

- 🔴 The stabilized-TH control's ladder is still **paused at 23/24 rungs** (§2 / `pending-tasks.md`), so the
  printed `P₂/P₁ ASGS` block is transcribed from `previous_results/` + `debug_results/` rather than from the
  official DB. **The F3 wording fix is correct either way** — completing the run moves (10⁵,0.5) to 2.36 (L²) /
  1.32 (H¹), still sub-optimal.
- 🟡 **One editorial call for the authors.** `Codina2015OnSM` (an unciteable `@misc`: no venue, Semantic Scholar
  stub only) was **replaced** in the intro citation list by the archival `badia2010stabilized` (Badia & Codina,
  CMAME 199, 2010), which supports the same claim and is by the same co-author. If R. Codina prefers to keep the
  2015 document, revert that one `\cite` and supply its venue; the entry is still in the `.bib`.
