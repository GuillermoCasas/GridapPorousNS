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
- ✅ **VERIFIED (2026-07-19) — first table caption (`tab:Linear2DL2`, [1176](../theory/paper/article.tex#L1176)).**
  Defines the parenthetical "(theoretical rate)" convention (`k_v+1`/`k_p` in L², `k_v`/`k_p−1` in H¹, dash = no
  guaranteed rate; same convention in the other three 2D tables), the **regime-dependence caveat** ("these are the
  *viscosity-dominated worst case*; the reaction/convection regimes attain one order more for the pressure"), the
  two-finest-mesh slope rule, and the interpolation-reference-row convention. Src: review D4.
- ✅ **VERIFIED / tables-only settled (2026-07-19) — results-section figures.** The paper has exactly one figure —
  `bump_plateau.pdf` (the 1−α porosity field), referenced at [1142](../theory/paper/article.tex#L1142) and present
  on disk; it **renders correctly** (a labeled 3D surface, z-axis 0 / 1−α₀ / 1, caption matches). No standing
  `\Guillermo{Add figures}` note remains and no convergence figures were added — **tables-only** is the settled
  decision (the build succeeds at 66 pp with the figure included). Src: `open-questions.md §4`.

## 4. Editorial / prose

- 🔴 **open — finalize ALL review markup.** True counts (verified): `\Guillermo`=14, `\Joaquin`=4 (=**18**
  spans), `\amend` used **328×** (recorded 279 — **stale by ~49**; per-file: article 156, elemental 15, continuity 7,
  fourier 3; audit 2026-07-19). Two distinct tasks: (i) flatten the 328 `\amend` + 18 author spans (redefine
  as `{#1}` for the final build); (ii) decolor the 18 author spans. **CORRECTION (audit 2026-07-19):** the
  TODO-bearing macros (`REVIEW: CHECK` L398, `JUSTIFY` L402, `CITATIONS` L644, `CITATIONS FOR THIS STRATEGY` L655)
  are **all on commented-out lines** and will not ship, and the `\Guillermo{Add figures}` note **no longer exists**
  (already removed — see §3) — so flattening/decoloring is sufficient; no TODO-authoring or citation work is needed. Src: critique;
  `article.tex:115–122`.
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

- 🔴 **verify + commit — the `\@ifpackageloaded{lineno}{}{\allowdisplaybreaks}` guard**
  ([article.tex:56–64](../theory/paper/article.tex#L56), currently uncommitted). Without it, `[review]` +
  `allowdisplaybreaks` collapses display math to ~22 pp with all-`??` refs on TeX Live 2023/macOS. Src: MEMORY
  `paper-build-fragilities`.
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
  (the 7 extra `\cite` matches were commented-out lines). All listed works resolve (codina 2001/2008/2018,
  villota2019, cocquet2021, badia2020/verdugo2022 gridap, codina1993, nillama2022, hughes2007). Src: `theory/README.md`.

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

## 10. External AI revision (`docs/final_AI_revision.md`, 2026-07-19) — per-point assessment + new items

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
green (**67 pp** / 722 newlabels / 0 unresolved / 0 undefined citations — the 66→67 bump is the
added prose). **F1 and F2 are now additionally machine-checked** by the new
`proof_verification/sympy/display_consistency_verification.py` (suite **115/115**); why the
existing machinery missed them and how to close the gap is in
`proof_verification/verification-gap-coverage.md`. **Deferred (flagged, not applied):** the four
fragile notation nits — IA-5e (`Π^S=∇^S u`), F14c (`=`→`≈` in the SGS pull-out, exact for ASGS),
F14d (`λ` eigenvalue rename), F14e (`U` scalar-vs-vector) — each in delicate `\scriptstyle`
scoping or a multi-use symbol where forcing a change risks a build break or overriding a
deliberate convention; and M5 (`Codina2015OnSM` booktitle), M8 (Codina/de-Pouplana emails),
M6 (DOIs), which need bibliographic/contact data that must not be fabricated; and the nice-to-have 9b
(how α is evaluated at quadrature in the MMS runs). These await author input. (F9a and F9c — the τ
theory/practice gap and the stopping-tolerance sentence — **were applied** after an accuracy re-check of this banner.)

### 10.A New — important (🟠)

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

### 10.B New — nice-to-have (🟡)

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
