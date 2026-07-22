# Response to the 2026-07-22 external audit (`latest_audit/`) — verified & enriched revision plan

**What this is.** A point-by-point verification of every item in `latest_audit/` against the *current* source
(`article_v2.tex`, `continuity_appendix.tex` = App. C, `osgs_appendix.tex` = App. D, appendices A/B), plus the
coverage-gap closure and a full two-paper coherence read. Each audit item is classified
**CONFIRMED** / **PARTIAL** / **MISREAD** / **ALREADY-DONE**, tagged **ALG** (equation/estimate/proof error) /
**SCOPE** (hypothesis precision) / **PROSE** (wording/over-claim) / **BUILD** (source hygiene), and — where it
touches the paper's mathematics — cross-checked against the Coq/SymPy machinery.

**Headline verdict (concur with the audit).** *Major revision, no fatal theorem defect.* The two-method
ASGS/OSGS architecture is sound; the genuine equation-level defects are small and local; the bulk of the
register is scope-precision and over-claim tempering (text-only). Three of the four "Blockers" (B01–B03) are
**repo-clean non-issues** (download-export artifacts on the auditor's side).

**Applied this session** (the two items the audit + our own verification tie directly to the proof-verification
system, plus the coverage closure the third user instruction asked for):
- **M14 (T13 double-count) — FIXED** in `continuity_appendix.tex` (`|T₁₃|+|N| → |T₁₃^c|+|N|`); both papers
  rebuild clean (v1 70 pp / 726 labels / 0 undefined; v2 90 pp / 0 undefined).
- **Coverage gaps closed** — `sympy/osgs_display_consistency_verification.py` (6) + `continuity_grouping_verification.py`
  (5); suite **242 → 253/253 across 19 scripts**. Recorded in `lessons_learned.md §1` (2026-07-22), `AUDIT.md`
  (F9), `verification-gap-coverage.md` (addendum), `EQUATION_COVERAGE_LEDGER.md`, `sympy/README.md`,
  `coq_coverage.tex`.

Everything else below is staged for the **author** (theorem-statement scope and manuscript prose are author
judgment; the proof-gap repairs require writing new mathematical content). Line numbers are `article_v2.tex`
unless prefixed; they drift with the `\amend` prose — resolve by the quoted label.

---

## 0. Source / build (Group 4)

| ID | Verdict | Note |
|---|---|---|
| **B01** | **MISREAD (repo-clean)** | `article_v2.tex:1848` inputs `continuity_appendix.tex` (correct name); it **is** the broken-ℓ² proof (`Ψ(h)`=`eq:psih`). "(2)" = a download artifact. |
| **B02** | **MISREAD (repo-clean)** | `:1840` = `\bibliography{references}`; `references.bib` present. "(4)" = artifact. |
| **B03** | **MISREAD (repo-clean)** | `shared.tex`, `siamart190516.cls`, `siamplain.bst`, `figures/bump_plateau.pdf` all present; both papers compile. |
| **B04** | **CONFIRMED · BUILD** | Flatten before submission: `article_v2` has 408 `\amend` + 13 `\Guillermo` + 3 `\Joaquin`; +24 `\amend` in appendices (incl. the new F9/T13 `\amend`s). |

---

## 1. Definite algebraic / proof-level fixes (Group 1 — the only true equation-level items)

| ID | Sev | Verdict | Location & fix |
|---|---|---|---|
| **M14** T₁₃ double-count | Major | **✅ APPLIED (ALG)** | `continuity_appendix.tex` `eq:groupstep`: `|T₁₃|+|N| → |T₁₃^c|+|N|`. Result unaffected. Guarded by `continuity_grouping_verification.py`. |
| **M13** missing abs-values | Major | **CONFIRMED · ALG(statement)** | Add `|·|` to `eq:continuity` (:447), `eq:sharpcont` (:455–460), `eq:assembly` (:830). The *proof* already bounds `Σ|T_j|`; only the lemma displays drop the bars. Body `lemma:Continuity`/`lem:continterp` already carry them. |
| **M11** patch equivalence | Major | **CONFIRMED · ALG(gap)** | `osgs_appendix.tex:366–370` derives an *additive* Lipschitz estimate then asserts *multiplicative* equivalence (only valid if `C·C_∇α<1`). Replace with the audit's shared-vertex resolved-porosity chain `α_K≤δ_α α_{0,K}≤δ_α α(x_a)≤δ_α α_{K'}`; treat `|a|`/τ equivalence with a data-dependent small-mesh argument. |
| **M12** `a·∇u` global regularity | Major | **CONFIRMED · ALG(gap)** | `osgs_appendix.tex:899,906–908` applies patch best-approx to `z=a·∇u`, needing global `H^{k_u}(S_K)`; `H:advectionsmooth` gives only elementwise smoothness + `a∈C⁰`. Strengthen to `a∈W^{k_u,∞}(Ω)` (or prove a broken-regularity approximation lemma). |
| **M08** ASGS pressure mean-shift | Major | **CONFIRMED · ALG(gap)** | `continuity_appendix.tex:915–924` interpolates without mean correction; for ε=0 a raw interpolant of a zero-mean `p` need not lie in `Q_{h0}`. Reuse the OSGS `oa:rem:meanshift` in the common tools and apply to both methods. |
| **A07** ASGS norm definiteness | Moderate | **CONFIRMED · ALG(gap)** | App. C uses the ASGS working quantity as a norm without a kernel argument; App. D has `oa:lem:definiteness`. Add a shared definiteness lemma (use the explicit dev-sym identity `‖dev sym ∇v‖²=½‖∇v‖²+(½−1/d)‖∇·v‖²`). |
| **A11** face estimates compressed | Moderate | **CONFIRMED · ALG(exposition)** | `continuity_appendix.tex:1041–1044` asserts "identical powers of h" — the delicate ℓ² upgrade. Display the two face bounds + the finite-overlap Cauchy–Schwarz step explicitly. |
| **A17** inf-sup sup missing `\{0}` | Moderate | **CONFIRMED · ALG(notation)** | `osgs_appendix.tex:805` `\sup_{V_h∈X_h}` → add `\setminus\{0\}` (theorem statements `oa:eq:InfSup` and body `eq:InfSupMain` already have it). Trivial. |
| **A16** silent `c₁≥1` | Moderate | **CONFIRMED · ALG(minor)** | `osgs_appendix.tex:863,931` use `c₁+Da_h≤c₁(1+Da_h)` and `≥1`. State `c₁≥1` (harmless under `H:design`) or use `max{1,c₁}`. |
| **A06** `ε‖q‖=0` | Moderate | **CONFIRMED · PROSE(notation)** | `osgs_appendix.tex:523`: the norm term is `ε‖q‖²`, so → `ε^{1/2}‖q‖=0` (or "the `ε‖q‖²` term forces `q=0`"). |
| **M16** normalized √ factor | Major | **CONFIRMED · ALG(interp.)** | `osgs_appendix.tex:1264–1268` matches observed ≈3.5 to `√(1+Da_h)≈3`; the *bounds* are correct but the varying factor after absorbing fixed `c₁^{1/2}` is `√(1+Da_h/c₁)≈1.87`. **Remove the numerical match**, keep the qualitative "reaction-dependent pre-asymptotic excess." (Body :1313 is already careful — it says "our analysis does not resolve" the gap; the appendix remark is where the invalid match lives.) |

> **M14 detail (applied):** Step 5 now writes `T₁₃=T₁₃^c+R` with `T₁₃^c` the convective contribution (named to
> avoid the generic-constant `C` clash), `R` "absorbed into `N`"; `eq:groupstep` LHS = `|T₁₃^c|+|N|`, with a
> parenthetical noting `T₁₃+T₂+T₄+T₁₁=T₁₃^c+N`. See `AUDIT.md` F9.

---

## 2. Hypothesis / scope precision (Group 2 — author-owned theorem statements)

| ID | Verdict | Fix |
|---|---|---|
| **M04** operator scope | CONFIRMED·SCOPE | Model permits generic `ViscProj` (`:277–287`) but proofs fix deviatoric-symmetric (`continuity_appendix.tex:24–26`, `osgs_appendix.tex:518` Korn). Restrict the theorem to dev-sym, or add an abstract `‖∇v‖≤C‖ViscProj∇v‖` hypothesis. |
| **M07** pressure normalization | CONFIRMED·SCOPE | ASGS (`continuity_appendix.tex:62`) full-L² for ε>0; OSGS (`osgs_appendix.tex:36`) zero-mean always; body `H:spaces` a third phrasing. Adopt **one zero-mean convention for all ε≥0** across body + both appendices. |
| **M06** common penalty is ASGS-specific | CONFIRMED·SCOPE | Common `H:data` (:870) imposes the strong `eq:UpperBoundOnEpsilon`; OSGS needs only `ετ₂≤C₂`. Keep `ε≥0` common; move the strong bound to ASGS-specific. |
| **M05** blanket "standing assumptions" | CONFIRMED·SCOPE | Body OSGS theorems (`th:StabilityOSGS` :1053, `th:ConvergenceOSGS` :1073) cite only "the standing assumptions" — list exact labels (A1–A6 + O-set), as the ASGS lemmas and all appendix theorems already do. |
| **M01/M02/M03** projection & lagged-form | CONFIRMED·SCOPE | Separate the three projection objects (formal VMS / τ-weighted analyzed / implemented L², spread across :611/:615/:630/:641); change ":665 lagged form analyzed" to "stationary current-residual Method-I form"; scope `osgs_appendix.tex:132` "matches the implementation" → "…in the projection space." (M03 is PARTIAL — `oa:rem:analyzed` already enumerates every difference; only :132 needs the clause.) |
| **M09/M10/A14** β₀, h₀ dependence | CONFIRMED·SCOPE/PROSE | Surface β₀ (h-uniform only, `H:projection`) and data-dependent h₀ in the *body* `th:StabilityOSGS` and in abstract/robustness/conclusion; reconcile the "depends only on…" line (:1053) with the "likewise depends on coefficient data" caveat (:1068). |
| **A01** local vs global QU | CONFIRMED·SCOPE | *(Also coherence issue #1, see §5.)* `:1148` "global quasi-uniformity … assumed throughout" contradicts (A1) "locally quasi-uniform"; App. D deliberately uses the ℓ² functional to avoid needing global QU. Scope global QU to the single-`h` rate corollaries, or add it to (A1). |
| **A02** ℓ_K vs h_K | CONFIRMED·PROSE | `H:patch` sets `h_K:=χ₁(x_K,h)` (:903) — redefines diameter. Introduce `ℓ_K` with `c h_K≤ℓ_K≤C h_K`. |
| **A03** size-surrogate oscillation | CONFIRMED·SCOPE | State `ω_χ(h)→0` (`=O(h)` under Lipschitz data) used in the smoothing proof (`osgs_appendix.tex:425–427`). |
| **A04** dimensional consistency | CONFIRMED·PROSE | Declare nondimensional coordinates (or insert `L^j`) for `‖D^j a‖`, `‖D^j α‖` bounds. |
| **A05** nodal Lagrange spaces | CONFIRMED·SCOPE | Smoothing uses nodal shape functions; `H:spaces` says only "continuous polynomial." State nodal Lagrange (or abstract the smoothing operator). |
| **A10/A12** per-lemma hypotheses / consistency regularity | PARTIAL | Lemma statements are careful, but the global "coercivity throughout" convention (:148–165) and the forward-referenced local `H²/H¹` (`continuity_appendix.tex:97–99`) should be stated per-lemma / at the point of use. |

---

## 3. Over-claim / wording tempering (Group 3 — author's results narrative)

**Numerical logic / results (§7):** **M17** delete the fold/root-exists language at `:1309` (contradicts `:1297`
"cannot decide … would require branch-continuation and Jacobian diagnostics") — report solver non-convergence +
first successful meshes only. **M18** rewrite the 3D penalty paragraph (`:1586`) to separate the four issues
(pressure-up-to-constant / boundary-flux incompatibility / ε-regularization / iterative-penalty correction).
**N01** ":1248 same discrete solution" → "same algebraic formulation and observed branch." **N02** narrow
"genuine discretization errors" → "not contaminated by iterative linear-solver tolerances." **N03** flag
`N=512,768` as supplemental (outside the declared 10–320 sequence). **N04** `:1303` "any velocity … scales as
1/α" is **false in general** → "the manufactured field is constructed with an α₀/α factor." **N05/N06/N07/N09**
interpolation-benchmark / "not an artifact" / OSGS-pressure mechanism / "accelerating rate" — empirical language,
label hypotheses. **N10** *(PARTIAL/mostly-MISREAD)* the "P₁ viscous residual vanishes" claim (`:1584` footnote,
`:1305`) is about the **unweighted affine** reference operator and is correct as scoped; add a one-line clarifier.
**N11–N16** restrict tetrahedral/rate/Taylor–Hood claims to tested families; supply raw-data metadata.

**Robustness / porosity (§6, §9):** **M19/M20** temper the abstract/conclusion robustness and the `α₀^{-1/2}`
claim to the *displayed principal weights* (hidden `C_{α,m}`, β₀, h₀, solution norms). **A13/A19/A20/A21/A22/A23/A25**
appendix over-claims (α_K "tightens" low-porosity; "uniform in σ"; pressure "decreases"; the Method-II remark;
the one-item symmetry table; the ASGS-titled shared appendix; ν vs 2ν norm normalization).

**M15 (body L² display):** **PARTIAL** — the body reaction-limit display drops `(1+c₁/Da_h)`, but is derived
in-context as the `Da_h≫c₁` asymptotic and cross-referenced to `oa:cor:Ltwo`. It sits on the audit's accepted
escape hatch; optional one-clause caveat at the equation. **Now guarded** by `osgs_display_consistency_verification.py`.

**Abstract/intro/conclusion (I01–I09) & style (S01–S04):** name both methods / qualify robustness (I01); state
both contributions symmetrically + rename `sec:StabilityASGS`→neutral (I02/I03); "that theorem"→"those theorems"
(I05); limit empirical claims to tested cases (I06/I08); OSGS reaction discrepancy = qualitatively compatible,
not quantitatively necessary (I07); drop the untested "interpolated porosity is benign" (I09); replace
"exactly/confirms/optimal"-overloading with estimate-language (S01–S04).

---

## 4. Already handled / audit MISREAD — do **not** "fix"

- **B01, B02, B03** — repo is build-complete (upload artifacts).
- **I04** — continuous uniqueness is already stated as an *assumption* (`:299`).
- **A15** — non-sharpness / "not a calibration" already stated (`osgs_appendix.tex:572–580`, body :1067).
- **A09** — the `c₁>2ξC²` ↔ `c₁>4C²` equivalence is already stated (`continuity_appendix.tex:432–433`); cosmetic.
- **A08** — the coercivity margin `1−4C̄²/c₁` *is* shown in the proof (`continuity_appendix.tex:434–438`); could
  be emphasized in the statement, not an error.
- **M03, A18, N10, N15, I07, I09** — the audit **overstates**; the current source is largely correct (A18: the
  step *is* P5 and P5 *is* named + Coq-verified — only the "ετ₂≤C₂ applied once more" clause is misplaced; N15:
  `:1809` already says "error stagnates at an O(1) value").

---

## 5. Coverage-gap closure (user instruction 3 — "correct it and learn") — ✅ DONE

The audit's four algebraic/near-algebraic items were checked against SymPy + Coq for whether the verification
system *should* have caught them:

| # | Item | Verdict | Action |
|---|---|---|---|
| 1 | **T₁₃ double-count** | **GENUINE GAP** (low sev.) | No App. C term-accounting existed (only App. A, in `assembly_consistency`). **Fixed the paper** + added `continuity_grouping_verification.py` (5 checks, discriminating). |
| 2 | **OSGS L² display factor** | **GENUINE GAP** (top) | The entire OSGS appendix had **zero** SymPy display coverage; Coq proves `oa:cor:Ltwo` only *abstractly* (`abstract_osgs_convergence_Ltwo: √σ‖e_u‖≤Cconv·Ψ_O`, never unfolding `(1+c₁/Da_h)`). Added `osgs_display_consistency_verification.py` (6 checks, discriminating). No paper change (body form is a legitimate asymptotic). |
| 3 | **√(1+Da_h/c₁) vs √(1+Da_h)** | **OUT OF SCOPE** | The factor is literally `Ψ_O`'s weight `(c₁+Da_h)^{1/2}`; the error is a *numerical over-claim* (M16, §1) no symbolic check can adjudicate. |
| 4 | **A18 P5 attribution** | **OUT OF SCOPE** | The step *is* P5, already Coq-verified (`OsgsParameters.P5_prod`/`P5_sqrt`); a prose label mismatch only. |

**Lesson recorded** (`lessons_learned.md §1`, 2026-07-22): the body↔appendix display-consistency net (F1/F2)
stopped at §4/App. A and must extend to **every** appendix; every collected/grouped proof display needs
term-accounting like `assembly_consistency`. Suite **253/253 across 19 scripts**. Ledger updates: `AUDIT.md` F9,
`coq_coverage.tex` (delegation + OSGS-conv boundary + F9 row), `verification-gap-coverage.md` (addendum),
`EQUATION_COVERAGE_LEDGER.md`, `sympy/README.md`.

---

## 6. Coherence read of both papers (user instruction 2 — "a coherence read … slowly through everything")

Both papers were read in order (main text + appendices), compiled, and every `\cref/\ref/\eqref` checked against
the `.aux` label reality.

**`article.tex` (v1) — clean.** Rebuild exit 0, **0 undefined references, 0 multiply-defined labels**, 70 pp /
726 labels. The v2 appendix restructuring did **not** break the shared-appendix v1 build: `continuity_appendix.tex`
cites **no** v2-only label (`sec:CommonSetting`, `app:osgs`, `th:StabilityOSGS`, `oa:*` all absent), and v1
correctly references no OSGS appendix. Only cosmetic: review markup + one *inert* commented-out
`\cref{app:NewtonRaphsonSubscales}` (`article.tex:665`) inside a commented block.

**`article_v2.tex` (v2) — highly coherent.** Rebuild exit 0, **0 undefined, 0 multiply-defined**, 90 pp, 466
labels. Verified coherent: every body→appendix pointer resolves to the right object *and* appendix (App. A/B/C/D =
elemental/Fourier/continuity/OSGS; `lemma:Stability`→`lem:coercivity`, `th:StabilityOSGS`→`oa:th:stability`,
`th:ConvergenceOSGS`→`oa:th:convergence`, `oa:cor:Ltwo`, etc.); §5.1 "Common setting" defines (A1)–(A6)/(S1)–(S2)/(O1)–(O5)
before use, each theorem invoking only its own set; notation consistent across body+appendices (`X_h`/`\Xh`/`\Xhz`
one object; `C_inv` vs `C̄_inv` reconciled; `⦀·⦀_O` identical body↔App D); ε and pressure-normalization conventions
per-experiment consistent; `c₁=4k⁴` (2D) vs `16k⁴` (3D) each stated in the right place.

**The one substantive incoherence:** *local (A1) vs global quasi-uniformity (§6 `:1148`)* — **= audit A01** (§2).
Two independent passes (the external audit and the coherence read) flag it. It is a **hypothesis-scoping decision**
(scope global QU to the single-`h` rate corollaries, or add it to A1), so it is **left for the author**, not
edited unilaterally. Cosmetic residue (v2): bare `⦀·⦀` in App. C vs `⦀·⦀_S` in body/App. D (shared appendix, an
optional one-line "≡" note); a commented duplicate label `eq:AdjointFlux_commented` (:505/:507); orphaned
`genuine3d_table.tex` (not `\input`, tracked in `pending-tasks §7f`); orphaned App. C proof-step labels; review
markup (B04).

---

## 7. Recommended implementation sequence (author)

**P1 — definite correctness (Group 1):** M14 ✅done; M13, M11, M12, M08, A07, A17, A16, A06, A11; M16 (remove the
numerical √ match). **P2 — theorem rigor/scope (Group 2):** M04, M07, M06, M05, M01/M02, M09/M10, A01–A05, A10/A12.
**P3 — interpretation/prose (Group 3):** M17, M18, M19/M20, N01–N16, I01–I09, S01–S04; M15/N10 clarifiers.
**P4 — hygiene:** B04 (flatten `\amend`/author notes), then a page-by-page final PDF check. **Optional diagnostics
(no full rerun):** the β₀ generalized-eigenvalue over representative regimes (strengthens M09); a weighted-vs-ordinary
projection A/B (M01 bridge); pseudo-arclength continuation *iff* fold language is retained (else delete per M17).
