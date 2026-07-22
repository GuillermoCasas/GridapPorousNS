# Revision plan (v2 audit response) — validity verdict, Coq analysis, and 3-part program

**Date:** 2026-07-21
**Responds to** the four v2 audit files added this round: `Paper-audit-request_v2.md` (ChatGPT top-level response), `revised_paper_osgs_audit.md` (detailed audit), `theory_integration_blueprint.md` (architecture), `ai_revision_instructions_v2.md` (implementation instructions).
**Status of this doc:** planning + record of the error-fixes (§3) and the Part 1/2/3 implementation (§IMPL). Decisions locked: **c₁,c₂ = Route A** (sufficient/non-sharp, no new math).

---

## IMPL — Implementation status (2026-07-21)

**Parts 1 & 2 applied in place; Part 3 produced as a new integrated paper `theory/paper/article_v2.tex` (the original `article.tex` is left intact for comparison).** All artifacts compile clean.

| Part | Item | Status | File |
|---|---|---|---|
| 1 | 1A ASGS continuity lemma → ℓ² at lemma level + `|·|` bars + soften "sharpest" | ✅ | `continuity_appendix.tex`, `article.tex` |
| 1 | 1B Taylor–Hood summary → velocity-qualified + confounding caveat | ✅ | `article.tex` |
| 1 | 1C c₁=16k⁴ framing (empirical, not theory-predicted) | ✅ | `article.tex` |
| 1 | 1D "interpolation/approximation floor" → benchmark/reference (all sites) | ✅ | `article.tex` |
| 1 | 1E 3D OSGS pressure-H¹ labeled an open adverse result | ✅ | `article.tex` |
| 1 | 1F robustness two-regime (fixed-data h→0 vs pre-asymptotic) paragraph | ✅ | `article.tex` |
| 1 | 1G wholesale `\amend`/`\Guillermo`/`\Joaquin` removal | ⏸ deferred | kept as change-tracking for review; resolve for submission |
| 1 | reproducibility data table | ⏸ deferred | needs author input (data repository) |
| 2 | D1 `C₂^{3/2}`→`C₂`, C6 `m≥r-1`→`m≥r`, dropped `α_∞` @:670 | ✅ | `osgs_convergence.tex` |
| 2 | 2A convergence theorem + corollaries + intro/abstract → ℓ² `Ψ_O` | ✅ | `osgs_convergence.tex` |
| 2 | 2A Coq ℓ² mirror | ⏸ deferred | `.v` tree green (proves valid ℓ¹ corollary); documented in `coq_coverage.tex`; mirror = redefine `Eh` + `wEh_val` witness + `run_all.sh` |
| 2 | 2B L² velocity corollary `1+c₁/Da` (square-summed) | ✅ | `osgs_convergence.tex` |
| 2 | 2C Route A (sufficient/non-sharp c₁,c₂; h₀ data-dependent) | ✅ | `osgs_convergence.tex` |
| 2 | 2D existence/uniqueness; 2E definiteness (Korn) + inf/sup nonzero; 2F soften overclaims; 2H Method-II → prospective | ✅ | `osgs_convergence.tex` |
| 3 | 3A main-text OSGS section (form, norm, Ψ_O, both theorem statements, comparison table, scope paragraph) | ✅ | `article_v2.tex` |
| 3 | 3B OSGS proof as refereed appendix (`\cref{app:osgs}`, labels namespaced `oa:`) | ✅ | `article_v2.tex` + `osgs_appendix.tex` |
| 3 | 3C abstract/intro/conclusion realigned for two-method scope | ✅ | `article_v2.tex` |
| 3 | common Appendix A consolidation (OSGS appendix currently self-contained, duplicates continuous problem/mesh/τ) | ⏸ deferred | noted in the appendix intro |
| 3 | 3D move elemental-matrices/Fourier to supplement + target-class page count / one-vs-two-paper decision | ⏸ author | needs target-journal compile |

**Compile state:** `article.tex` (Part 1) → 70 pp, 0 errors, 0 undefined. `osgs_convergence.tex` (Part 2) → 20 pp, clean. **`article_v2.tex` (integrated) → 88 pp, 0 errors, 0 undefined references, 912 labels resolved.** Coq `.v` tree untouched (green).

New support files: `osgs_appendix.tex` (namespaced OSGS proof), 3 bib entries added to `references.bib` (`AinsworthOden`, `ScottZhang`, `CodinaBlasco1997`), macro block + `\usepackage{amssymb}` in `article_v2.tex`.

Line numbers below are hints; each item also names the LaTeX `label` / phrase so it stays findable as the source drifts.

---

## 0. TL;DR

- The v2 audit is **competent and grounded** (its macro/word counts and "substantially-addressed" inventory match the current source). Its integration architecture — *theorem statements in main text, proofs in refereed appendices, common tools shared* — is correct and adopted.
- **But it must not be implemented blindly.** Three of its specific technical demands are **wrong** (would break or misstate the proof): porosity `sup→inf` (C1), advection regularity (C4), screening-length (D7). Two "typos" it flags are already fixed in-source (A4, A5). Its headline `ℓ¹→ℓ²` alarm is **overstated** — the ℓ² content is already proved (and, for ASGS, already machine-checked in Coq); it is a statement-level cleanup, not a proof hole.
- **The Coq question (see §2):** the two "surprising" errors are not a Coq failure. The Coq proves the deductive **skeleton** of each theorem from a **trusted base** of ~50 hand-transcribed analytic hypotheses; the errors live *inside* that trusted base (per-term constants / regularity indices the Coq abstracts as hypotheses). Coq could not catch them by construction. Both are **harmless to the theorems**.
- **Errors found and fixed (§3):** three hand-computation slips in `osgs_convergence.tex`, all in the trusted base, all harmless, all now corrected and the note recompiles clean (19 pp). The Coq needs **no change**. The Coq-covered ASGS appendix and the T1–T7 stability estimates came back **clean**.

---

## 1. Validity verdict on the audit

### 1.1 ⚠️ REFUTED — do NOT implement (audit is wrong)

| Ref | Audit demand | Why it's wrong |
|---|---|---|
| **C1** | Replace porosity resolution `h_K‖∇α‖≤C·α_K` (sup) with `α_{0,K}` (inf) | **Backwards / proof-breaking.** Stability Step 3 (`osgs_convergence.tex:891`) needs `α_K` as a pointwise *upper* bound; `lem:patch` (eq:PatchEquivalence, :606) already proves two-sided comparability unconditionally via `α_∞/α_0`. The inf-form would break coercivity. |
| **C4** | Strengthen advection regularity `W^{1,∞}→W^{k_u,∞}` | **Misreading.** (A4) at :490 already carries `‖D^j a‖≤C_D|a|` for `1≤j≤k_u`. |
| **D7** | Replace "Da_h≤1 on any reasonable mesh" with the exact screening-length condition | **Misreading.** The exact `h_K≲(α_K ν/σ)^{1/2}` with `Da_h=(h_K/ℓ_σ)²` is stated in the *same* remark; the colloquial phrase backs it, not replaces it. |

### 1.2 🟡 Already handled in-source (no action / cosmetic only)

- **A4 "there exist{s}" typo** — already fixed: `\amend{s}` renders "exists". Audit misread the revision markup.
- **A5 "velocity trivially satisfies ∇·(αa)=0"** — REFUTED. `article.tex:1091-1093` states outright it holds "at the continuous level, though not for the discrete iterates." Opposite of the audit's claim.
- **E10 "purely discretization errors"** — overstated; the actual text is scoped ("…rather than an artifact of an iterative solver's stopping tolerance").

### 1.3 🔵 Framing correction: ℓ¹→ℓ² is a statement cleanup, not a hole

The audit's loudest concern (ASGS §3 and OSGS §4.4): the convergence estimates are elementwise `ℓ¹` sums that can't give full interpolation order. **The abstract math is correct** (ℓ¹ loses up to `h^{-d/2}`) **but does not wound the proofs**: every step is derived as a broken-`ℓ²` (root-of-sum-of-squares) bound first and only relaxed to `ℓ¹` at the last line. For ASGS the Coq proves the sharp `ℓ²` form directly (`abstract_continterp`, `coq_coverage.tex:707-715`). So this is *restate-at-the-lemma-level*, low-risk — not re-prove.

### 1.4 ✅ SOLID — worth acting on

`B1/B4` (OSGS theorem is ℓ¹ → rewrite as ℓ² Ψ_O, proof already supports it); `B3` (L² velocity factor `1+c₁/Da` not `1+Da⁻¹`); `A1/A2/A3` (restate ASGS continuity lemma in ℓ²); `C3` (the `c₁,c₂` sufficient bounds ≈`400 C_inv²`,`20 C_inv` don't cover experimental `4k⁴/16k⁴`,`2k²` — **disclose, Route A**); `C6`/`D1` (real slips — **fixed, §3**); `C7`/`C2`/`C5` (assumptions to keep prominent — already disclosed in the note); `D3` (norm definiteness); `D4` (existence/uniqueness one-liner); `D2` (inf/sup over nonzero); `D5`/`E8` (soften genuine overclaims); `E6`/`E9`/`E11` (manuscript items); macros + reproducibility.

Full per-claim verdicts (adversarially cross-checked, 0 overturns) are in the workflow transcript; the substantive ones are folded into Parts 1–3 below.

---

## 2. The Coq question, answered

> *"How can there be a mistake in a proof that was Coq-covered? I assume Coq makes the proofs 100% correct."*

**Half right.** The Coq development uses a **trusted-base + machine-checked-assembly** architecture (documented as such in `proof_verification/coq_coverage.tex`). Per theorem:

1. **Assembly / deductive skeleton** — how the named intermediate estimates combine into the conclusion (triangle inequalities, ring regroupings, Cauchy–Schwarz collection, Céa / inf–sup composition). **Machine-checked in full**, zero `Admitted`/`Axiom` (beyond 3 stdlib axioms). This is where "100% correct" holds.
2. **Trusted base** — ~50 named hypotheses transcribing the paper's *per-term analytic estimates and modeling assumptions*, fed to Coq as **hypotheses with abstract constants**. Coq proves "*if* these hold *then* the conclusion follows"; it does **not** re-derive the hand-arithmetic behind each one.

Verified concretely in `OsgsInterpolation.v`: the consistency theorem `abstract_osgs_consistency` is literally *"given `|S₁|≤kS1·(E·NV)` and `|S₂|≤kS2·(E·NV)` with `kS1,kS2` abstract nonnegative reals, then `|B|≤(kS1+kS2)·(E·NV)`"* — a 5-line triangle inequality. The note computes `kS2=C₂^{3/2}`, but **Coq never sees that number** (`kS2` is a free variable). The exponent slip is *by construction* outside Coq's verified boundary. Same for the `m≥r−1` regularity index (in `lem:bestapprox`, a trusted-base input).

**Therefore:**
- **Coq did not err.** It proved the skeleton it was given; the slips are in hand-derived *inputs* it legitimately abstracts. `coq_coverage.tex` is explicit that the soundness of that base is what a human/LLM audit is for.
- **This is the definition of the guarantee, not a hole.** No formalization of a PDE a-priori estimate verifies the functional analysis from first principles (that needs formalized Sobolev/interpolation/trace theory). Every such proof is "skeleton modulo a trusted transcription."
- **Both slips are harmless to the theorems**, and the corrected constants remain valid instances of what Coq assumes — so **no Coq change is needed**.
- Scope note: the **ASGS chain (18 files)** and **OSGS chain (6 files)** are both Coq-covered at this level. For ASGS the Coq holds the sharp `ℓ²` form; for OSGS the Coq's error function is genuinely `ℓ¹` (matching the note), so upgrading OSGS to `ℓ²` is a *strengthening* requiring coordinated note+Coq work — not a bug fix.

---

## 3. Errors found and FIXED (this round) ✅

Focused adversarial re-derivation of every hand-computed constant / regularity index in the OSGS + ASGS trusted base (the layer Coq does **not** protect). Three genuine slips, all in `osgs_convergence.tex`, all harmless (absorbed by the generic `C` / covered by A3's global smoothness), all now corrected; note recompiles clean (19 pp, no undefined refs). **No Coq change required.**

| # | Location (label) | Was | Now | Type / why harmless |
|---|---|---|---|---|
| D1 | `:1140`, `lem:consistency` S₂ slot | `C₂^{3/2}` | `C₂` | exponent slip (`ετ₂≤C₂`, `C₂≤1` ⇒ only `C₂` justified; displayed value strictly smaller = "too tight"). Absorbed by generic `C`; corrected `C₂` is still a valid `kS2` for the Coq assembly. |
| C6 | `:698`, `lem:bestapprox` hyp | `m ≥ r−1 wherever r−1 ≥ 2` | `m ≥ r wherever r ≥ 2` | off-by-one regularity (proof Leibniz-differentiates `α` through order `r`). Harmless: (A3) globally supplies `m=max(k_u,k_p)` and every use has `r≤m`. |
| NEW | `:670`, `lem:smoothing` proof | `(c₂/(c₁α₀ν))·h·ω_a` | `α_∞·(c₂/(c₁α₀ν))·h·ω_a` | dropped `α_∞` factor (present at :666, lost at :670). Feeds only `ψ(h)→0`; absorbed by generic `C`. |

**Clean (re-derived, no slip):** the ASGS `continuity_appendix.tex` in full (0 slips — and it is the Coq-covered ℓ² chain); the OSGS stability estimates T1–T7 + Step-4/5 collection (0 slips); all parameter inequalities P1–P6/K1–K3; the interpolation lemma I1–I7; `lem:interpsize`; `lem:normcomparison`; `lem:patch`.

---

## 4. PART 1 — Improvements to the paper *as it is* (no OSGS integration; no rerun)

Edits to `article.tex` / `continuity_appendix.tex`.

- **1A. ASGS continuity lemma → ℓ² at the lemma level** `[proof-repair, low-risk]`. Restate `lem:continterp` (`continuity_appendix.tex:903-905`) directly as `|B_S(a;U−Î_hU,V_h)| ≤ C·Ψ(h)·|||V_h|||`; move the `Ψ→ψ` (ℓ²→ℓ¹) relaxation to a corollary; delete the retrospective "holds with Ψ(h)… by inspection" at `:1010-1014`; add `|·|` bars to every continuity inequality; drop/qualify "sharpest". (Content already machine-checked in ℓ² — this is presentation.)
- **1B. Taylor–Hood summary** `[text]`. `article.tex:1653` still says the method "attains the accuracy of Taylor–Hood on the DBF equations" with no velocity qualifier (while :1649/:1664 are correctly narrowed). Change to "comparable *velocity* accuracy in the reported cases," repeat pressure caveat.
- **1C. `c₁=16k⁴` framing** `[text]`. Fix the parenthetical so "predicts" cannot attach to the value. Present it as: sufficient condition motivates a *larger* `c₁` for higher inverse constants; value chosen empirically, confirmed numerically. (Matches our own element-aware-c₁ findings; the theory does not certify it.)
- **1D. "interpolation floor" residuals** `[text]`. Rename surviving occurrences (`:1651, :1434, :1157, :1159`) to "interpolation benchmark/reference."
- **1E. 3D OSGS pressure-H¹ anomaly** `[text; rerun optional]`. Acknowledged (footnote + "barely converging" + conclusion) but under-analyzed. Add one mechanistic paragraph or label it explicitly open.
- **1F. Robustness quantifiers** `[text]`. Split "fixed data, h→0 (⇒ Re_h,Da_h→0)" from "pre-asymptotic element dominance"; start each regime from Ψ_A; state quasi-uniformity where used.
- **1G. Source hygiene** `[mechanical]`. Resolve 362 `\amend` / 14 `\Guillermo` / 4 `\Joaquin` in `article.tex` and 7 `\amend` in the appendix. Add reproducibility data (mesh sizes, DOFs, raw errors, quadrature, full coupled residual incl. projection eq., stopping tolerances).

---

## 5. PART 2 — Improvements to the OSGS proofs (`osgs_convergence.tex`)

- **2A. Rewrite convergence theorem in Ψ_O form** `[proof-repair, low-risk]` (B1/B4). Replace `ℓ¹` `E(h)` (`eq:ErrorFunction :1054`; intro/main copies `:66`,`:105`) by `Ψ_O(h)² = Σ_K (c₁+Da_{h,K})(α_K²/h_K²)(τ₂E_u² + τ₁E_p²)`; keep the square-summed form through `lem:consistency`/`lem:interpolation`/`lem:interpsize`; `ℓ¹` becomes a corollary. **Coordinated Coq update** (`OsgsInterpolation.v`/`OsgsConvergence.v`): redefine `Eh` as ℓ² and add an `Eh_l2 ≤ Eh` (ℓ²⊂ℓ¹) corollary, mirroring the existing ASGS `abstract_continterp` / `abstract_continterp_l1` pair; re-run `./run_all.sh`.
- **2B. L² velocity corollary factor** `[proof-repair, tiny]` (B3). `cor:Ltwo`/`eq:LtwoVelocity`: write `1 + c₁/Da_{h,K}`, or explicitly declare the `c₁` absorption (~`:1288`).
- **2C. `c₁,c₂` constant gap — Route A (locked)** `[text/decision]` (C3). Keep the sufficient bounds (A6: `c₁≥γ²C_inv²`, `c₂≥γC_inv`, `γ≥γ₀=10(1+M₂)`) but state them as **sufficient, non-sharp**; state the numerical constants are *not shown* to satisfy them; remove any "theorem predicts/validates the constants" language (note + article). No new math.
- **2D. Existence/uniqueness one-liner** `[text]` (D4). State discrete existence+uniqueness as an immediate consequence of inf-sup on the square system.
- **2E. Norm definiteness** `[proof-repair, standard]` (D3). Add a Korn / conformal-Killing lemma (full Dirichlet + pressure normalization; needed because A2 allows `σ=0`, leaving only the deviatoric symmetric gradient), or call the triple quantity a *seminorm*. Also write inf/sup over **nonzero** functions (D2).
- **2F. Soften genuine overclaims** `[text]` (D5). "exact price"/"exact reward"/"sits exactly where the theory puts it" → "factor produced by the present estimate" / "stronger reaction control retained in the norm" / "compatible in order of magnitude." (`:137` and `rem:mechanism`/`rem:numerics`.)
- **2G. Keep prominent (already disclosed, no new math)** (C7/C2/C5). A7 weighted projection stability (`:503-517`, assumed), `h₀` data-dependence, the relative-derivative condition as standard-but-restrictive — surface these in the theorem statement on integration.
- **2H. Method-II remark** `[decision]` (D6). Either write out the modified proof or demote `rem:methodII` to a prospective observation.
- **✅ Already fixed:** D1, C6, NEW-670 (§3).

---

## 6. PART 3 — Integration tasks (OSGS into `article.tex`)

Architecture: **results-in-main-text, proofs-in-refereed-appendices, common tools shared.** Do **not** paste the 19-page standalone note.

- **3A. New/expanded main-text theory section** (grow §`sec:StabilityASGS`, `article.tex:830`): common linearized problem; ASGS + OSGS analyzed forms side-by-side; one **standing-assumptions block** (Common / ASGS-only / OSGS-only); both norm defs; `Ψ_A`, `Ψ_O`; **full ASGS + OSGS theorem statements**; one proof-roadmap paragraph; an ASGS-vs-OSGS **comparison table** (coercivity vs inf-sup; damped `σ̃_α` vs full `σ`; extra assumptions; reaction factor); a boxed **analyzed-vs-implemented scope paragraph** (lift the note's honest-scoping text, `osgs_convergence.tex:161-178`, which is currently invisible to the main-paper reader).
- **3B. Restructure appendices into A/B/C.**
  - *A (common):* mesh/patch, porosity comparability (the **sup** form via `lem:patch` — *not* the audit's inf), τ inequalities, interpolation estimates, Galerkin cancellation, the Korn/definiteness lemma (2E), constant conventions.
  - *B (ASGS):* coercivity, viscous weighted inverse estimate, face-jump lemma, discrete + interpolation continuity in Ψ_A, convergence. (≈ today's `continuity_appendix.tex`.)
  - *C (OSGS):* weighted projections + annihilation, patch/smoothing, A7, inf-sup special-test proof, consistency defect, interpolation continuity in Ψ_O, convergence + corollaries.
  - *Strip on import:* the note's title/abstract/intro, duplicated continuous problem, duplicated mesh/space/τ/interpolation defs, separate bibliography, provenance note, extended numerical commentary.
- **3C. Abstract/intro/conclusion realignment** `[text, do last]`. Abstract → "separate linearized ASGS *and* OSGS results," qualify "robust." Conclusion → four separate statements (proved-linear-ASGS / proved-linear-OSGS / observed-nonlinear / outside-theory). Never claim either theorem covers the nonlinear DBF test.
- **3D. Length/venue gate** `[decision]`. Compile the *unified* manuscript in the target SIAM class (currently missing `siamart` class/`shared.tex`/figure/bib — neither reviewer could get a real page count). SISC/SINUM ≈20 pp target, ≈26 pp return-without-review. Move `elemental_matrices_appendix.tex` and most Fourier algebra to a supplement/repository first. If still well over → **two-paper split** (theory + computational).
- **3E. Optional bridge study** `[rerun, optional]`. Only if claiming the theory represents the *implemented* OSGS: 3 ablations (weighted vs L² projection; elementwise-const vs pointwise τ; truncated vs full residual) on one diffusion-, one convection-, one reaction-dominated case. Otherwise the "nearby method" disclosure is sufficient and standard.

---

## 7. Sequencing & open decisions

1. **Cheap, unambiguous fixes first** (proof already supports them): 2A(note side), 2B, 2D, 1A. (2C wording — Route A — anytime.)
2. **2E** (definiteness) and the **2A Coq coordination** (`run_all.sh` must stay green: 0 `Admitted`/`Axiom`, only 3 stdlib axioms).
3. **Manuscript text:** 1B–1F, 2F, 2G.
4. **Integration** 3A–3C; then macros/reproducibility 1G.
5. **Compile in target class → split decision (3D) → align abstract/conclusion (3C) last.**

**Open decisions for the author:**
- **3D one-paper vs split** — depends on the real target-class page count (unknown until the missing class/assets are supplied).
- **2H Method-II** — prove or demote.

**Effort reality:** the audit's tone implies heavy proof repair; the verification shows most of it is **restatement + disclosure** (the ℓ² content already exists, and for ASGS is machine-checked). The only genuinely open mathematical choice — `c₁,c₂` — is settled as **Route A** (no new math).

---

## 8. Do-NOT list (audit demands to reject)

- Do **not** switch porosity resolution to `inf_K α` (C1) — breaks stability Step 3.
- Do **not** "strengthen" (A4) advection regularity (C4) — already sufficient.
- Do **not** replace the screening-length statement (D7) — already exact in the note.
- Do **not** treat A4's `exist{s}` or A5's solenoidality as uncorrected errors — already handled.
- Do **not** frame ℓ¹→ℓ² as a missing proof — it is a statement-level cleanup (ASGS already ℓ² in Coq).
