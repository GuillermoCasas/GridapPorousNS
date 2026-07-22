# Harmonization & reorganization plan for `article_v2.tex`

**Date:** 2026-07-21 · **Target:** `theory/paper/article_v2.tex` + its appendices (`continuity_appendix.tex` = ASGS, `osgs_appendix.tex` = OSGS).
**Status:** in execution (compile-verified stages). See §EXEC for progress + the element audit.

---

## EXEC — execution log & element-by-element audit (2026-07-21)

**Done (compiles clean, 89 pp):**
- ✅ **Harmonious titles** — both appendices retitled *"Convergence analysis of the ASGS/OSGS method"*.
- ✅ **ASGS coercivity proof written out** (`lem:coercivity` in `continuity_appendix.tex`): Galerkin diagonal (convective skew + pressure cancellation) + exact stabilization-diagonal expansion + reaction renormalization to `σ̃_α` (`σ−σ²τ₁=σ̃_α`) + penalty absorption via (P5) + viscous-residual absorption via the weighted inverse estimate under `c₁>2ξC̄_inv²`. Removed the redundant `prop:stability` "recall". Now symmetric with the OSGS inf-sup proof.
- ✅ **Overflowing equations** — 21 display equations multi-lined (continuity_appendix 5→0, osgs_appendix 16→0; 1 residual in main text, outside scope). Pure line-break/environment reformatting.
- ✅ **Proof-preservation check PASSED** (scripts + baselines in scratchpad `proof_baselines/`): (1) vs pre-harmonization baseline — ASGS 7/7 and OSGS 15/15 proofs token-for-token IDENTICAL ⇒ the reorg altered no proof math; (2) OSGS 15/15 IDENTICAL to `osgs_convergence.tex` (a priori OSGS work); (3) ASGS IDENTICAL to git-HEAD original except three approved changes only — the new `lem:coercivity`, the Part-1 ℓ² line `≤Cψ→=CΨ`, and the `prop:stability→lem:coercivity` citation. Re-run after each future de-dup stage.

**Element-by-element audit (answers "is each element needed here?"):**

*Common → single location (a common appendix / main-text block):* parameter inequalities (`lem:parameters` ≡ `oa:lem:parameters`), interpolation estimates (`eq:interp`/`eq:interpinfty` ≡ `oa:eq:NodalInterp`), inverse estimate (`eq:inverse` ≡ `oa:eq:InverseEstimate`), Galerkin skew/cancellation identity (`eq:skew` ≡ `oa:lem:coercivity`), patch/porosity comparability (`oa:lem:patch`), Korn norm-definiteness (`oa:lem:definiteness`), the `φ₁`/`σ̃_α`/`τ₂`-expanded param algebra.

*Strong-form / setup re-exposition → REMOVE (reference main text):* ASGS `eq:strongop`, FE-space prose, `eq:taus`, `eq:triplenorm`, `eq:Eint`; OSGS `oa:eq:StrongProblem`(+Momentum/Mass/BC), `oa:eq:GalerkinForm`, `oa:eq:Xdef`, `oa:eq:InverseEstimate`, `oa:eq:TauDefs`, `oa:eq:TripleNorm`, `oa:eq:EintDef`, the provenance/acknowledgment footer.

*Standing assumptions → MAIN-TEXT block:* ASGS `H:data/H:porosity/H:advection/H:mesh`; OSGS `oa:sec:hypotheses` (A1–A6, A8). **Stay in appendices:** `H:jump`/`lem:jump`/`lem:winv` (ASGS-only), A7 `oa:eq:Hstab` (OSGS-only).

*Error functionals → MAIN TEXT:* `eq:psih` (Ψ_A/ψ), `oa:eq:ErrorFunctionL2`/`oa:eq:ErrorFunction` (Ψ_O/ℰ).

*KEEP (method-specific proofs — these do NOT move, so proofs stay identical):* ASGS `eq:Bstab`, `eq:consistency`, `lem:winv`, `lem:jump`, `lem:coercivity`, `lem:continuity`, `lem:continterp`, `thm:convergence`; OSGS annihilation, smoothing, best-approx, inf-sup stability, consistency, interpolation continuity, convergence + corollaries, discussion remarks.

**Structural de-dup — executed (compile-verified after each edit, 90 pp, 0 undefined; proof gate re-run — all proof math preserved):**
- ✅ **Main-text §5.1 "Common setting and standing assumptions"** — states the linearized problem once + a standing-assumptions block (Common / ASGS-only / OSGS-only).
- ✅ **Strong-form re-exposition removed from both appendices** — OSGS strong-problem block (`oa:eq:StrongProblem`…) and ASGS strong-operator (`eq:strongop`), now pointing to §5.1; 5 in-proof references remapped (pointer updates only).
- ✅ **ASGS stabilized-stability *proof* fully moved out of the main text** (per the note): `eq:StabilityEstimate`, `eq:BoundOfCrossedTerm`, `eq:StabilityEstimateFinal` and the coefficient-positivity algebra are all removed, pointing to `lem:coercivity` in `app:Continuity`. **Kept in the main text** (as requested): the unstabilized Galerkin coercivity (motivation), the four fundamental definitions referenced throughout — inverse estimate `eq:InverseEstimateFiniteOrderNorm`, ε-bound `eq:UpperBoundOnEpsilon`, `σ̃_α` `eq:SigmaAlpha`, `c₁`-condition `eq:conditions_on_num_param` — the `‖·‖_h`/`X` definitions + the adjoint-term modeling remark, and the `lemma:Stability` statement + gist. Flow connectors reworded ("we have just shown"→pointer; dangling "is met." fixed). Page count 90→89.
- ✅ **`lem:coercivity` made fully explicit** — the cross-term Young bound (`eq:crossbound`) and the collected coefficients (`min{2−4C̄²/c₁, 2(1−2/ξ)}·ν`, `(1−ξC̄²/c₁)·σ̃_α`, `(1−C₂)·ε`) now written out in the appendix (transcribed from the paper's own `eq:BoundOfCrossedTerm`/`eq:StabilityEstimateFinal`), replacing the earlier citation.
- ✅ **Proof gate re-run:** original proofs preserved — OSGS 15/15 identical to `osgs_convergence.tex` except the one benign consistency-proof reference descriptor; ASGS all identical to git-HEAD except the approved Part-1 ℓ² line; new `lem:coercivity` excluded as approved-new.
- ✅ **Consistency review (4 reviewers) done + all fixes applied:** (a) added a `\subsection{The ASGS variant…}` heading so the ASGS analysis no longer sits under "Common setting" (parallel to §5.2 OSGS); (b) trimmed the redundant assumption restatement that duplicated §5.1; (c) fixed the now-false numerical-scope claim ("no estimate covers OSGS") to cite `th:StabilityOSGS`/`th:ConvergenceOSGS` (also gives those theorems their citation); (d) OSGS-appendix intro now points to `sec:StabilityOSGS`+theorems, not the section header; (e) notation unified — `C_∇→C_{∇α}` across `osgs_appendix.tex` **and** `osgs_convergence.tex` (kept proof-gate clean), triple-norm `⦀·⦀` disambiguated in the OSGS appendix, `C̄_inv` vs `C_inv` and `γ` clarified in §5.1, `σ𝟙→σ𝕀`, "that symmetry" antecedent named, section intro broadened to both variants, representative porosity noted.
- ✅ **Shared-appendix regression fixed:** `continuity_appendix.tex` is `\input` by both `article.tex` and `article_v2.tex`; the two article_v2-only refs I'd added (`sec:CommonSetting`, `app:osgs`) were remapped to labels present in both, so `article.tex` compiles clean again (71 pp, 0 undefined).
- ✅ **Final state:** `article_v2.tex` 89 pp, `article.tex` 71 pp, `osgs_convergence.tex` 20 pp — all 0 errors / 0 undefined refs / 0 multiply-defined. Proof gate: original proofs preserved (only the 2 benign/approved differences). Residual: 2 main-text display-overfulls >5 pt, and the shared-lemma common-appendix consolidation (still deferred).
- ✅ **Proof gate (math-only) after de-dup:** ASGS appendix proofs ALL IDENTICAL to the pre-dedup baseline; OSGS IDENTICAL except a single benign reference descriptor ("the linearized problem") in the consistency proof (99.11% byte-identical) from the strong-form remap. No proof mathematics altered.

**Remaining (recommended as a focused next pass):** the common-appendix consolidation of the *shared lemmas* (parameter inequalities, interpolation estimates, Galerkin identity, Korn definiteness) referenced from both appendices (~30 ref remaps); and optionally relocating the appendices' full assumption lists (now that §5.1 carries the overview).

**⚠ Correct execution order for the remaining structural de-dup** (discovered during execution): the appendices' *linearized* problem (frozen `a`, constant `σ`) is **not** the main text's `eq:StrongMomentumEquation` (which is the *nonlinear* `σ(α,u)u`). So the removal must be preceded by a **main-text §5.1 "Common linear setting"** that states the linearized problem + the standing-assumptions block + labels the two forms/norms/`Ψ` — *then* the appendices reference it. This is a deliberate re-labeling + ~50-ref-remap pass; recommended as the next focused stage to keep it compile-clean and proof-preserving.

---


## Goal (from the request)

1. **Symmetric appendices** — each appendix holds the *full convergence proof* of its method (**including the stability proof**); the main text keeps only the theorem statements + the gist of each proof.
2. **Common assumptions and the main context** of both theorems → **main text**; the technical proofs → appendices.
3. **No OSGS self-containment**: shared technical lemmas live in *one* appendix and are *referenced* from the other (de-duplicate).
4. **Harmonious appendix titles** (parallel for ASGS and OSGS).
5. **Fix overflowing equations** in the appendices (multi-line environments).

---

## 1. Current state — the asymmetry to remove

| | **ASGS** | **OSGS** |
|---|---|---|
| Main text (§5) | §5.1: `lemma:Stability`, `lemma:Continuity`, `th:Convergence` — **statements only** | §5.2: form + norm + `Ψ_O` + `th:StabilityOSGS`, `th:ConvergenceOSGS` (statements) + comparison table + scope |
| Appendix | `continuity_appendix.tex` — *"A continuity proof for the stabilized bilinear form $B_S$"*: setting, `lem:parameters`, `lem:winv`, `lem:jump`, `lem:continuity`, interpolation, `lem:continterp`, convergence. **Stability only *recalled*** (`prop:stability`) | `osgs_appendix.tex` — *"Stability and a priori error analysis of the OSGS variant"*: **fully self-contained** — continuous problem, meshes, τ, hypotheses (A1–A8), annihilation, `lem:parameters`, `lem:patch`, `lem:smoothing`, `lem:bestapprox`, norm + `lem:definiteness`, **full inf–sup stability proof** (`th:stability`), consistency, interpolation, convergence + corollaries, discussion |

**Four concrete asymmetries:**
- **(a) ASGS stability proof is missing.** `lemma:Stability` (`B_S(U_h,U_h) ≥ C‖U_h‖²`) is a bare statement; its only justification is the prose "following Lemma 2 of `codina2001stabilized`". The OSGS side, by contrast, has a full written inf–sup proof. → **the ASGS coercivity proof must be written out** (§4 below).
- **(b) OSGS appendix duplicates the setup.** Its "Continuous problem", "Meshes/FE/interpolation", "Stabilization parameters" and "Hypotheses" subsections restate material already in the article's §2–§4 and (soon) §5.1.
- **(c) Duplicated technical lemmas across the two appendices:** `lem:parameters` (both), interpolation estimates (both), norm-definiteness (`lem:definiteness` in OSGS, but the ASGS norm needs the same Korn argument), patch/porosity comparability, the Galerkin cancellation identity.
- **(d) Titles are non-parallel** ("A continuity proof…" vs "Stability and a priori error analysis…").

---

## 2. Target architecture

### Main text §5 — "Stability and convergence for the linearized problem"
- **§5.1 Common linear setting** (new): the linearized (frozen-advection) problem; **one Standing-Assumptions block** consolidating `H:data…H:mesh` (ASGS) and `A1–A8` (OSGS), each item tagged *Common / ASGS-only / OSGS-only*; the two analyzed bilinear forms `B_S`, `B_osgs` side by side; the two mesh-dependent norms; the error functionals `Ψ_A`, `Ψ_O`.
- **§5.2 The ASGS method**: coercivity (stability) + continuity + convergence **theorem statements** + one-paragraph proof gist → *Appendix B*.
- **§5.3 The OSGS method**: inf–sup stability + convergence **statements** + gist + the ASGS↔OSGS comparison table + the analyzed-vs-implemented scope paragraph → *Appendix C*.

### Appendices — parallel titles
- **Appendix A — Common analytical tools** *(new)*: mesh/patch notation; porosity resolution + local/patch comparability (from `lem:patch`); the **common** τ-parameter inequalities; interpolation estimates (nodal / Scott–Zhang); the Galerkin skew/cancellation identity; the **norm-definiteness (Korn) lemma** stated once for both triple norms; the constant-dependency convention.
- **Appendix B — Convergence analysis of the ASGS method** *(from `continuity_appendix.tex`)*: **ASGS coercivity/stability proof (written out, §4)**; ASGS-specific weighted inverse estimate `lem:winv`; reaction face-jump `lem:jump`; discrete continuity `lem:continuity`; interpolation-continuity `lem:continterp` (in `Ψ_A`); ASGS convergence. References Appendix A for the shared lemmas.
- **Appendix C — Convergence analysis of the OSGS method** *(from `osgs_appendix.tex`, de-duplicated)*: projections + `lem:annihilation`; OSGS smoothing (`lem:smoothing`, the OSGS-specific part); inf–sup stability `th:stability`; consistency; interpolation-continuity (in `Ψ_O`); convergence + corollaries. References Appendix A for parameters, interpolation, definiteness, patch.

**Harmonious titles:** *A. Common analytical tools* · *B. Convergence analysis of the ASGS method* · *C. Convergence analysis of the OSGS method*.

---

## 3. What moves where (mapping)

| Item | Now | Goes to |
|---|---|---|
| Standing assumptions (`H:*` + `A1–A8`, merged & tagged) | scattered in both appendices | **main §5.1** |
| Linearized problem + both forms `B_S`,`B_osgs` + both norms + `Ψ_A`,`Ψ_O` | main + both appendices | **main §5.1** (stated once) |
| Common τ inequalities (`lem:parameters`, shared parts) | both appendices | **App A** |
| Interpolation estimates (nodal/Scott–Zhang) | both appendices | **App A** |
| Norm-definiteness / Korn (`lem:definiteness`) | OSGS appendix | **App A** (both norms) |
| Patch equivalence + porosity comparability (`lem:patch`) | OSGS appendix | **App A** |
| Galerkin cancellation identity | OSGS appendix ("coercivity identity") | **App A** |
| ASGS coercivity proof | *nowhere (cited)* | **App B (write out)** |
| `lem:winv`, `lem:jump`, `lem:continuity`, `lem:continterp`, ASGS convergence | continuity_appendix | **App B** (unchanged) |
| Annihilation, OSGS smoothing, inf–sup proof, consistency, interp, OSGS convergence + corollaries | osgs_appendix | **App C** (reference A for shared) |
| OSGS "continuous problem / meshes / τ / hypotheses" subsections | osgs_appendix | **delete** (now in §5.1 + App A) |
| Discussion remarks (`rem:mechanism` … `rem:open`), provenance note | osgs_appendix | keep in **App C** (trim provenance) |

---

## 4. The one *writing* task — ASGS coercivity proof

`lemma:Stability` currently has **no proof**; it defers to `codina2001stabilized`. To make the appendices symmetric (both = full stability + convergence), write the coercivity argument in **Appendix B**: the diagonal of `B_S(U_h,U_h)` gives the viscous + damped-reaction + penalty + stabilization terms; the residual cross-terms are absorbed via `lem:winv` (weighted inverse estimate) under `c₁ > 2ξ C_inv²` (the condition already in `prop:stability`), with the porosity amendments of `rem:winvconst`. **Decision point:** write it out (recommended — it is what symmetry requires and mirrors the OSGS `th:stability`), or keep it as a cited result with the amendments made explicit (less work, but leaves the asymmetry the request targets).

---

## 5. Overflowing equations (checked — 22 display + 1 paragraph)

Cause: the OSGS note was written for `article`-class 2.7 cm margins; in the narrower SIAM text block its long single-line displays overflow. **Fix:** convert each to a multi-line environment — `split`/`multline` for one long expression, `align`/`aligned` for a `=…≤…=` chain, breaking at `=`, `\le`, `\ge`. Do this as each equation is relocated. Full list (file : source-line : overflow):

**`osgs_appendix.tex` → Appendix C (16):** 1135 (**140pt**), 418 (65pt), 990 (57pt), 1206 (55pt), 1100 (49pt), 814 (46pt), 868 (44pt), 616 (38pt), 1153 (31pt), 1123 (26pt), 218 (14pt), 1236 (12pt), 736 (11pt), 999 (9pt), 1112 (6pt), 790 (4pt).
**`continuity_appendix.tex` → Appendix B (4):** 894 (**71pt**), 1007 (16pt), 138 (5pt), 328 (3pt).
**main `article_v2.tex` (2):** 1069 (12pt), 354 (~0pt, ignore).

(Line numbers will shift during the reorg; re-collect `Overfull \hbox … detected at line` from the build log after each move and clear them — target: 0 display overflows.)

---

## 6. Labels & cross-references
- OSGS appendix labels are namespaced `oa:`. On merge: shared lemmas moving to **App A** get canonical labels (e.g. `app:common:parameters`, `app:common:interp`, `app:common:definiteness`, `app:common:patch`) referenced by **both** B and C; drop the duplicate ASGS-side statements.
- `continuity_appendix` labels stay for App B; `oa:`-labels stay for App C minus the deleted setup.
- New main-text labels for the standing-assumptions block and the two forms/norms.

---

## 7. Execution order (compile-verify after each step)
1. **App A**: create `appendix_common.tex`; move the shared lemmas (parameters, interpolation, definiteness, patch, Galerkin identity) into it with canonical labels.
2. **App B**: trim `continuity_appendix.tex` to reference App A; **add the coercivity proof**; retitle; fix its 4 overfull equations.
3. **App C**: trim `osgs_appendix.tex` (delete duplicated setup, reference App A); retitle; fix its 16 overfull equations.
4. **Main §5.1**: add the common linear setting + standing-assumptions block + both forms/norms/`Ψ`; reduce §5.2/§5.3 to statements + gist.
5. Wire `\input{appendix_common.tex}` before B and C; harmonize the three titles.
6. Compile `article_v2.tex`; drive overfull-display count to 0; resolve refs.

## 8. Effort / risk
- **New math:** only the ASGS coercivity write-out (§4) — bounded, follows a known template.
- **Everything else** is relocation + de-duplication + equation multi-lining — mechanical but touch-heavy; compile-verify at each step (watch for label clashes and the `-a` grep on the binary log, per the v2 build gotchas).
- Net effect: shorter combined length (de-dup), symmetric presentation, and a main text that states both theories with their shared assumptions before sending the reader to parallel appendices.

---

## 9. Execution log — C/D consolidation round (2026-07-22)

**Realized approach (differs from §7's `appendix_common.tex` plan).** The common home is the
**main text** (`sec:CommonSetting` already collects the standing assumptions and splits
*common / ASGS-only / OSGS-only*), plus Appendix C for the ASGS-side technical lemmas. No
separate common appendix was created. Reason — hard constraint: `continuity_appendix.tex`
(App C) is `\input` by **both** `article.tex` and `article_v2.tex`, so it may cite only
labels present in both; it therefore stays the self-contained ASGS reference (the user
confirmed a self-contained appendix, stability included, is fine even for `article.tex`).
All consolidation concentrated in `osgs_appendix.tex` (App D, `v2`-only, free to reference
the main text and App C).

**Edits (all in App D unless noted), reference-and-signpost only — no proof step altered:**
1. Intro rewritten: shared material (continuous problem, Galerkin form + coercivity identity,
   parameters, mesh/interp setting, common assumptions) is *referenced*, not re-derived.
2. **Galerkin coercivity identity** (`oa:lem:coercivity`) → removed as a lemma+proof; now
   quoted from the new main-text label `eq:GalerkinCoercivityIdentity` (the reader-facing
   "why not quote it?" item). Its one internal use (`oa:eq:Diagonal`) re-pointed to the
   quoted identity `oa:eq:CoercivityIdentity`.
3. `oa:sec:meshes`: lead paragraph flags common (nondegeneracy, inverse estimate
   `eq:InverseEstimateFiniteOrderNorm`) vs OSGS-specific (patches, Scott–Zhang, size
   surrogates); inverse-estimate display annotated as the main-text one specialized.
4. `oa:sec:parameters`: parameters cited as `eq:Tau1Final,eq:Tau2Final`; only `Re_h`, `Da_h`
   and the τ-weighted products flagged as OSGS-specific.
5. `oa:sec:hypotheses`: framing paragraph — (A1)–(A5) = `sec:CommonSetting`; (A6)–(A8) +
   consistency-only smoothness = OSGS-specific; (A7) vs ASGS face-jump ⇒ non-nested sets.
6. `oa:sec:norm`: OSGS triple norm recalled from `eq:TripleNormOSGS`.
7. `oa:lem:parameters`: signpost that (P1)–(P3) = App C `lem:parameters`; (P4)–(P6),(K1)–(K3)
   OSGS-specific.
8. Consistency-proof opening synced with the a-priori note (`osgs_convergence.tex`): both now
   read "solves the linearized problem of \cref{…}", differing only in the cref target.

**Verification.** `article_v2` 90 pp, 0 undefined, 0 errors; `article` clean; note 20 pp
clean. **Proof gate:** 14/14 retained OSGS proofs byte-identical to `osgs_convergence.tex`
(1 intentional consolidation = the quoted Galerkin identity); ASGS side unchanged from its
prior accepted state (all identical to git HEAD except the approved Part-1 ℓ² line in
`lem:continterp` and the two intended coercivity proofs).

### 9b. Deeper D.1 consolidation (2026-07-22, follow-up)

User pushback: D.1.1/D.1.2/D.1.3 were still "largely general" — reference-and-signpost was
not enough; the common material had to be *removed* from D, not annotated in place.
Clarified: putting common material in App C (or the main text) does **not** hurt `article.tex`
— the shared-C constraint is only about *forward-citing* `v2`-only labels, not about content;
the common setup is general (ASGS needs it) and self-contained.

Removed the four duplicated setup displays from D and redirected their few proof/text
citations to the common home (a `\cref` redirect is invisible to the proof gate):
- `oa:eq:GalerkinForm` (Galerkin form $B$) → `eq:BilinearForm` (1 cite).
- `oa:eq:Xdef` ($X$ notation) → App C `eq:XG` (0 cites).
- `oa:eq:InverseEstimate` → `eq:InverseEstimateFiniteOrderNorm` (3 cites).
- `oa:eq:TauDefs` ($\tau$ definitions) → `eq:TauNavierStokes,eq:Tau1Final,eq:Tau2Final` (4 cites).

D.1.1 is now a recall paragraph + the quoted coercivity identity; D.1.2 keeps only the three
OSGS-specific tools (patch quasi-uniformity, Scott–Zhang, χ size-surrogates) with the common
mesh/inverse/spaces referenced; D.1.3 recalls $\tau$ from the main text and keeps only
$\Reh$, $\Dah$ and the τ-weighted products. Verify: `article_v2` 90 pp, 0 undefined, 0 errors,
0 orphan refs; proof gate still 14/14 byte-identical.

### 9c. Assumptions → main text; full ASGS proof → appendix (2026-07-22)

Governing principle (user): main text = **assumptions + statements + salient/interpretive
elements only**; proofs, technical lemmas and local nomenclature → appendices. Assumptions
presented as **numbered lists**, grouped common / ASGS-only / OSGS-only, in full (theorems
hold only under them).

**Done, both papers:**
- Assumptions moved into the main text as numbered lists. `article_v2` §5.1 `sec:CommonSetting`:
  `(A1)–(A8)` (common + OSGS-only `(A6),(A7)`, keeping the exact numbering App D cites as plain
  text) + `(S1),(S2)` ASGS-only; `article.tex` §5: `(A1)–(A8)` ASGS-only. `H:*` labels carried
  by the items; `eq:resolved`+`δ_α` and `eq:ProjectionStability`+`oa:eq:Hstab` relocated here.
- Appendices hold proofs + technical lemmas + local nomenclature only. App C `sec:assumptions`
  → short "Constants and conventions" note; App D `oa:sec:hypotheses` → pointer; App D stale
  "Acknowledgment of provenance" deleted. App C internal cites redirected
  (`eq:epscond→eq:UpperBoundOnEpsilon`, `eq:inverse→eq:InverseEstimateFiniteOrderNorm`,
  `eq:jumpcond→eq:JumpCondition`; `eq:resolved` label relocated, no redirect).
- **`article.tex` inline stability derivation removed** (was `eq:StabilityEstimate…
  eq:StabilityEstimateFinal`); now statement + gist → full proof in App C, matching v2.

**2-agent full read-through then fixes applied:**
- **[bug] cleveref A/S collision**: `\cref{…,H:jump}` in v2 silently dropped the jump because
  `(S2)` reuses `enumi=2`, colliding with `(A2)=H:data`. Fixed by referencing the jump in a
  separate `\cref` in App C's continuity/continterp lemmas. (Same lemmas' `\crefrange{H:data}
  {H:mesh/jump}` had also become reversed/over-broad after the relabel — replaced with explicit
  `\cref` lists.)
- `\Eint` macro made subscript in `article.tex` (matched §5 and App C; was superscript).
- OSGS theorem/lemma statements re-pointed from "assumptions of App D" to `sec:CommonSetting`.
- Added `Da_h` gloss and the `‖·‖_{τ_i}` definition in §5.3; `C_inv`/`C_2` first-use clarified;
  lemma hypotheses de-duplicated against the `(A#)` list; `eq:SmallPorosityGradient` (heuristic)
  → `eq:resolved`/`H:porosity` (sharp) drift fixed; redundant `α_{0,K}` re-definition trimmed.

**Verify:** `article_v2` 89 pp, `article` 70 pp, both exit=0, 0 undefined, 0 SIAM-label errors.
**Proof gate:** OSGS 14/14 byte-identical to `osgs_convergence.tex`; ASGS unchanged from HEAD
except the approved Part-1 ℓ² line and the two intended coercivity proofs.

### 9d. Symmetric assumption numbering (2026-07-22)

User: the OSGS-only assumptions were `(A6),(A7)` (inside the common `(A#)` list) while ASGS-only
were the separate `(S1),(S2)` — asymmetric. Homogenized §5.1 (`article_v2`) into **three
parallel groups, each its own numbering**, in order:
- **Common `(A1)–(A6)`**: mesh, coefficients, porosity, advection, discrete spaces, regularity.
- **ASGS-only `(S1),(S2)`**: coercivity, jump.
- **OSGS-only `(O1),(O2)`**: design constants, projection stability (pulled out of the common list).

App D references renumbered to match: `(A6)design→(O1)`, `(A7)projection→(O2)`,
`(A8)regularity→(A6)`; range statements `(A1)--(A8)→(A1)--(A6),(O1),(O2)` and
`(A1)--(A7)→(A1)--(A5),(O1),(O2)`. App C unaffected (it only `\cref`s mesh/data/porosity/
advection = `(A1)-(A4)` and coercivity/jump = `(S1),(S2)`, whose numbers didn't change).
`article.tex` unchanged (ASGS-only, no OSGS group; its `(A1)–(A8)` stays).

Gate: the a-priori note (`osgs_convergence.tex`) keeps its own `(A#)` scheme; rather than
re-scheme it, the gate's normalizer now **strips assumption-ref tokens** `(A\d)/(O\d)/(S\d)`
(they are labels, like crefs), so App D's `(O1)` vs the note's `(A6)` compare equal on the math.
Verified neutral on the pre-renumber state, then: OSGS 14/14 byte-identical, ASGS 4/4.
Both docs exit=0, 89/70 pp, 0 undefined, 0 SIAM-label errors.
