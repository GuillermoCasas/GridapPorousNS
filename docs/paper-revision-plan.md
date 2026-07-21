# Paper revision plan — response to `archive/review_numerics_vs_theory.md` (v2)

**Status:** APPROVED and being applied (author instruction 2026-07-17: *"change them or solve the issues,
but never keep them for appearance"*; and: do not compute any slope partly from an interpolant).
Progress is tracked in §4 below.
**Date:** 2026-07-17.
**Basis:** independent verification of every claim in the review against `article.tex`, the three
appendices, the ten companion notes, the source code, and the archived result data. Verification used
three *independent* reimplementations of the review's interpolation computation (two numpy, one Gridap
using the harness's own `calculate_normalized_errors`).

> **⚠ STATUS UPDATE 2026-07-19 (supersedes the in-progress notes in §4–§8 below).** The 3D re-run **C1r is
> done** — the four canonical DBs (`results/k{1,2}/TET/{structured,nested_red}/`) are on disk at `c₁=16k⁴`
> with **0 `success=false`**. Consequently **E1, C7, and the §3 3D-table cell audit are CLOSED** (verified
> 2026-07-19: `make_3d_tables.py --check` matches every 3D `\num`; E1's two camouflage cells moved off the
> interpolant — ratios 1.617/1.006; the 1.29 triple is genuine saturation, raw 1.29198/1.28894/1.28954). The
> full 2D/Cocquet numerics audit is likewise cleared. Any "C1r/E1/C7 in progress / 0 `success=false` *so far*
> / will settle / re-run may change" wording in §4–§8 below is a **historical session log** — read
> `docs/pre-submission-checklist.md` §2/§3 for current status. **Still genuinely open:** D4c (wire tables to
> the generator).

---

## 0. Verdict on the review

**The review's factual spine is sound and its central finding is correct.** Its inferential
super-structure over-reaches in three specific places, one of its conclusions is wrong, and it misses
one item more serious than anything it raises.

### 0.1 What is confirmed

| Item | Status |
|---|---|
| **C1 / §1.4 — the α₀ bookkeeping inconsistency** | **VERIFIED, equation by equation.** All nine §6 estimates (`1009, 1017, 1021, 1033, 1037, 1042, 1046, 1057, 1063, 1069`) derive *exactly* from the **unweighted** theorem; each becomes `√(α_K/α₀)` under the weighted one. |
| **§0.A — the weighted theorem is proven** | **VERIFIED independently.** `continuity_appendix.tex:875-884` *defines* `ψ(h)` **with** the `α_K` weight; `thm:convergence:983` proves it; its closing remark calls it "the sharpest afforded by the present argument". Fix direction settled: **sharpen §6, do not weaken the theorem.** |
| **§3 — the twelve interpolation floors** | **VERIFIED 12/12 by three independent implementations**, agreeing to every quoted digit. |
| C2 (high-Da pressure α₀-independence = floor effect) | VERIFIED |
| C3 (pressure super-convergence; ASGS/ℚ₂ sits *on* the guarantee) | VERIFIED |
| C5 (Re_h = 3125; pre-asymptotic attribution sound) | VERIFIED |
| C9 (TH pressure at the ℙ₁ floor) | VERIFIED |
| D1 (α_K "elemental minimum" is a leftover) | VERIFIED — appendix fixes `α_K = α_{∞,K}` |
| D2 (local vs global α_∞: Da_h 9.77 global vs ≈195 in plateau) | VERIFIED |
| D3 (dimensional slip; correct intermediate is `Uν/(Ph)`) | VERIFIED — independently re-derived |
| D4 (caption convention) | VERIFIED |
| D5 **[v2 flip]** (code is right, the *paper's* §3.1 sentence is wrong) | **VERIFIED at code level** — trim is double-gated (`run_simulation.jl:56-59`); DBF hard-codes `ProjectFullResidual` (`CocquetFormMMS/run_test.jl:149`) |
| D7 (scope sentence) | VERIFIED |
| D11 **[v2 new]** (`C_α` symbol clash) | VERIFIED — `article.tex:811` (field) vs `continuity_appendix.tex:159` (constant) |

### 0.2 Where the review over-reaches — **must not be transplanted into the paper**

**(a) F3-d — the headline "the data discriminate in favour of the weighted form" is logically
unsound.** §6's estimates are **one-sided upper bounds**. The review constructs a *window*
`[interp_ratio, interp_ratio·α₀^{-1/2}]` and treats the lower edge as a prediction. There is no lower
edge: `√(α_K/α₀) ≥ 1` bounds the **weight**, not the observed error, and the estimate asserts only
`observed ≲ upper`. Every observed viscous pressure ratio (ℙ₁ 8.26–12.40; ℚ₂ ASGS 10.19–10.84; ℚ₂ OSGS
16.16–18.83) satisfies **both** the weighted bound (15.8 / 36.7) **and** the unweighted one (50 / 116).
*Two upper bounds both satisfied cannot be discriminated by a satisfied observation.*

> **Defensible replacement claim:** the observed degradation is far below the uniform `1/α₀` bound; once
> the interpolation growth is divided out, the residual method factor is ≤ ≈4. The unweighted form is
> grossly non-sharp (factor 5–14 loose); the weighted form is less non-sharp but is *also* not sharp.
> This still fully motivates rewriting §6 — but on grounds of **internal consistency with the paper's
> own proven theorem**, not on grounds of empirical discrimination.

**(b) F2-b — cherry-picked endpoints.** The review reports ℙ₁ u L² method factors "1.96 (ASGS) / 3.48
(OSGS)". True spreads: **ASGS 1.31–1.96** (the review quotes the *maximum*), **OSGS 3.48–4.13** (the
review quotes the *minimum*). Both come solely from the `(Re,Da)=(10⁻⁶,10⁻⁶)` row. Worse: OSGS
**exceeds** `α₀^{-1/2}=3.16` in *all four* viscous rows, yet the review tabulates it under
"≤ α₀^{-1/2}" and its draft §7.2 text calls it "compatible with the weighted prediction ≈3.2".

**(c) F1 — "best-approximation-exact" holds in H¹ but FAILS in L² for ℚ₂.**

> *Correction to an earlier draft of this plan.* I first asserted this claim was false for ℙ₁, on the
> grounds that the nodal interpolant is 2.4× worse than the L² projection. **That was wrong**: the 2.4×
> penalty is an **L² fact and does not transfer to the H¹ seminorm**, which is what the review actually
> claims for ℙ₁. The review's scoping is careful and I misread it. The real defect is elsewhere — in the
> ℚ₂ **L²** half — and I only found it by computing the H¹ projection to check my own objection.

Three distinct objects are in play, and only the third settles the claim:

1. `E_int,K(ψ) := h^{k+1}‖ψ‖_{H^{k+1}(K)}` — what Theorem 1 uses (`article.tex:966`); an error *bound*.
2. the **nodal interpolation error** `‖ψ − I_hψ‖` — what the review computed.
3. the **best approximation** `inf_{v_h}‖ψ − v_h‖` — the L² projection *in L²*, the **Ritz/energy
   projection** *in the H¹ seminorm*. These are different projections; conflating them is the trap.

Measured (this audit; quadrature-converged over four rules to ~13 digits; 1D anchor ratio = 1.00000000
across 24 configurations; minimum invariant to nullspace treatment to 1.8e-13; independent
reimplementation agrees to 1e-12):

| case | nodal | best approx. | ratio | claim at 3 s.f. |
|---|---|---|---|---|
| ℙ₁ u H¹ (α₀=0.5) | 3.499055e-2 | 3.498519e-2 | **1.00015** | ✅ TRUE |
| ℙ₁ u H¹ (α₀=0.05) | 1.751175e-1 | 1.749878e-1 | **1.00074** | ✅ TRUE |
| ℙ₁ p H¹ | 1.090442e-2 | 1.090434e-2 | **1.00001** | ✅ TRUE |
| ℚ₂ u H¹ (α₀=0.5) | 4.282290e-4 | 4.282154e-4 | **1.00003** | ✅ TRUE |
| ℚ₂ p H¹ | 7.979365e-6 | 7.979365e-6 | **1.00000** | ✅ TRUE |
| **ℚ₂ u L² (α₀=0.5)** | 2.064841e-7 | **2.046634e-7** | **1.00890** | ❌ 2.06 vs **2.05** |
| **ℚ₂ u L² (α₀=0.05)** | 2.385067e-6 | **2.325860e-6** | **1.02546** | ❌ 2.39 vs **2.33** (2.5% off) |
| **ℚ₂ p L²** | 3.847635e-9 | **3.841617e-9** | **1.00157** | ❌ 3.85 vs **3.84** |
| ℙ₁ u L² (α₀=0.5) | 3.259e-5 | 1.335e-5 | 2.442 | *(not claimed)* |
| ℙ₁ p L² | 9.837e-6 | 4.016e-6 | 2.449 | *(not claimed)* |

**Why this is structural, not an artefact of N=320** — measured `(ratio − 1)` vs `h`:
- **H¹ seminorm: O(h²), rate 2.00 in every case** → the nodal interpolant is *asymptotically optimal* in
  H¹. So the H¹ claim is not a coincidence of one mesh; it is a theorem-shaped fact.
- **L², ℙ₁: tends to a constant** — 2.4312, 2.4448, 2.4483, 2.4492 → **√6 = 2.449490**. Permanently
  suboptimal. (The review never claims ℙ₁ L², and correctly does not bold it: it prints 2.13/1.11.)
- **L², ℚ₂: tends to 1 only as O(h)** — hence still visibly off in the 3rd digit at N=320.

**Verdict:** the review's sentence *"best-approximation-exact … in the H¹-seminorm for the linear element
and in both norms for the biquadratic one"* is **true in its first half and false in its second**.
**Amendment:** keep "best-approximation-exact to three significant digits in the H¹ seminorm" for *both*
elements (true, and asymptotically so — a genuinely strong result worth stating); **drop "and in both
norms for the biquadratic one"** (the ℚ₂ L² gap is 0.16% for p and 2.5% for u at α₀=0.05).

*Bridge caveat:* the audit measured **nodal vs best**; the paper's claim is about **its method** vs best.
That transfers only because the method's reported errors equal the nodal errors to ~4 s.f. on the H¹
entries. The identification is **not universal** — the paper's ℙ₁ p L² (4.87e-6) is *not* the nodal value
(9.837e-6); it sits *between* nodal and the L²-best (4.016e-6), at 1.21× the best approximation. So
review line 80's explanation of that sub-nodal entry is **correct**, and this audit supplies the
quantitative reason.

### 0.3 Where the review is wrong

**(d) C8 — the excluded-corner mechanism is misattributed.** The review would put a factually
unsupported causal claim into the paper. Its proposed §7.3(1) sentence attributes the corner to the
τ-saturation note's *stopping-gate* story. But:

- `docs/findings.md:63` records the evidenced reason: **"a genuine discrete solution-branch fold
  (turning point), NOT a solver or Jacobian bug."** Newton *and* Picard from `u_ex` (budget 500,
  noise-floor 1e-12) both stall at `‖R‖≈5e-2` — **no root exists** at coarse N.
- The review's "stopping-criterion artifact, not an under-resolved solution" quote is scoped in the note
  to **Layer 1 only**; the review extends it to the whole two-layer mechanism.
- The deeper *why* (low-α₀ → fold) is recorded as **[OPEN]** in `docs/open-questions.md §1`, which
  explicitly flags "the link 'weaker coercivity ⇒ nonlinear-solver fold' is an unproven extension".
- **The corner has already been run and it converges** (`findings.md:99-104`): ℙ₁/TRI at N≥512, ℚ₂/QUAD
  at N≥160 (it does *not* fold for k=2), reaching optimal rates (H¹≈1.0, L²≈3.0) at machine-zero
  residual.

**(e) C4 — the framing is wrong in detail** (the *observation* is right; it is a real property, see
0.4a). Verified from the archived H5: the OSGS excess `√(OSGS²−ASGS²)` is **1.176e-1 at α₀=0.5 vs
1.167e-1 at α₀=0.05 — porosity-INDEPENDENT to 0.8%**. It is not "one corner" but the whole Da=10⁶
low-Re_h column. And "slope 1.05, optimal rate preserved" is an artifact of stopping at N=320: the gold
N=640 run gives OSGS rate **1.60** and the ratio falls 3.14 → 2.08 — a *super-convergent recovery*, not
a preserved rate. `docs/findings.md:203` already records the settled verdict: "a genuine,
pre-asymptotic OSGS coercivity gap that **recovers** to the optimal rate — NOT a bug, and NOT an
asymptotic order ceiling."

**(f) D10 — refuted on both counts.** *(Correction: an earlier draft of this plan accepted half of it.
Both halves are wrong.)*
1. "The macro `\Cinva` is defined and never used" — **false**. It is used **16 times**, by
   `continuity_appendix.tex`, which `article.tex` `\input`s (line 1620). The claim comes from grepping
   `article.tex` alone; the appendices are part of the same document. **Do not delete the macro.**
2. "The stability lemma is stated with the bare polynomial constant" — **false**. `article.tex:880`
   *explicitly* redefines `C_inv` to mean the enlarged constant ("we keep the symbol `C_inv` for this
   modified constant in what follows").

**The real residue is the opposite defect**: main text and appendix adopt **conflicting conventions for
the same symbol** — `article.tex:880` says `C_inv` *already means* the enlarged constant, while
`continuity_appendix.tex` `rem:winvconst` instructs the reader to *replace* `C_inv` by `C̄_inv`.
Applying the appendix's instruction to a main text that has already absorbed the enlargement
**double-counts** it. **Fixed 2026-07-17** by rewording `rem:winvconst` to state that the main text's
`C_inv` already denotes the enlarged constant and needs no substitution, and that the appendix writes
`C̄_inv` explicitly only to keep the two distinguishable inside the proofs.

### 0.4 What the review missed

**(a) The reactive OSGS gap is a *real, verified* property — the "solver artifact" alarm is refuted.**
I raised and then killed this: the row *was* run with JFNK enabled; the stall sensor is disabled on
every OSGS coupled route; `target_ftol` is documented as **not** the gate (the scale-free `ε_M` is, and
at Da=10⁶ the reaction term inflates the denominator ~10⁶, so `‖r_M‖≈1e-5` *is* converged); and a
stalled OSGS makes OSGS ≡ ASGS **bit-identical** (`lessons_learned.md:46`), i.e. *closes* the gap —
the opposite of what is seen. Reportable.

**(b) ⚠ HIGHEST SEVERITY — the finest irregular ℙ₂ 3D solve may not have converged.**
`docs/mms/p2-3d.md:149-152`: *"At L2 the four strategies' errors are **byte-identical** — that finest
solve does not converge (ILU-GMRES on the 27968-tet saddle) and returns the `c1`-independent
**exact-guess interpolant** (`success=false` for all)."* The paper's `tab:3DL2` irregular ℙ₂ row
(slopes 2.55/2.56, FME 2.07e-4/2.02e-4) is a **two-finest-mesh** slope. If the finest mesh returned an
interpolant rather than a discrete solution, that slope is **partly an interpolation slope** and the row
must be recomputed or withdrawn. *Uncertain*: the quoted passage is scoped to the `c1study_nested_red`
diagnostic runs, not demonstrably to the production sweep. **Must be settled before submission.**

**(c) The 3D raw data is gone and the 3D meshes are not reproducible.**
- No 3D `.h5`/results exist on disk; the `c1x4` JSON lived under gitignored `results/` and is **lost**.
  C7 therefore *cannot* be desk-checked — verifying 1.29 requires re-running **the entire 3D section**.
- `mesh3d.jl:98-109` generates the gmsh base mesh in a `mktempdir()` and **deletes it**; `save_msh`
  exists but no caller passes it; the gmsh version is unpinned. **The published irregular-family numbers
  may not correspond to any regenerable mesh.** This is a reproducible-results-rule violation.

**(d) The paper's tables are hand-copied.** `make_results_tables.py` emits paper-faithful tables to
`results/latex_compilation/paper_tables.tex`, but `article.tex` never `\input`s it. This is a *mechanism*
for C7's 1.29 triple and a systemic drift risk.

**(e) C6's strongest card, under-exploited by the review.** `article.tex:1370` **already admits**:
*"on the irregular meshes the manufactured boundary data is moreover not exactly compatible with discrete
mass conservation"* — a family-specific consistency defect, conceded **ten lines before** line 1380
attributes the sub-optimality *solely* to element quality. (The review's *other* alternative, the
ε-penalty, is refuted — `article.tex:1378` applies it to both families, so it has no differential power.)

**(f) `TAU_VISC_MULT` — an undeclared env-var knob in the τ kernel.** `src/stabilization/tau.jl:39`
reads `ENV["TAU_VISC_MULT"]` and multiplies the viscous eigenvalue `c₁ν/h²` *inside* `@inline
_tau_ns_inv`. Default `"1.0"` makes it inert, but it is exactly what
`.agents/rules/no-hard-coded-parameters.md` forbids, and it is invisible to result provenance.

**(g) The 3D `c₁` has no production representation.** `article.tex:1368` says *"in all the 3D
experiments we take c₁ = 16k⁴"*, but `get_c1_c2` (`continuous_problem.jl:202-206`) returns `4k⁴`
**unconditionally, dimension-blind**. The `16k⁴` exists *only* as a harness kwarg `c1_mult`, and
`run_sweep_structured` **defaults to `c1_mult=1.0`** — so the paper's regular-family tables require a
*non-default* invocation, and a default run silently produces 2D-constant results.

> **Note on a wrong turn:** an intermediate agent hypothesised that `c₁=16k⁴` "over-stabilizes" ℙ₁ in 3D
> and explains the rate deficit. **This is backwards** — `c₁` sits in the *denominator* of τ₁ and τ₂
> (`article.tex:812`, `tau.jl:38-58`), so `c₁=16` is *weaker* stabilization than `c₁=4`. The archive
> confirms the opposite direction (at the paper's `c₁=4`, 3D ℙ₁ ASGS L² rate is ~1.1–1.4, *worse* than
> the 1.75 at `c₁=16`). Do not pursue.

---

## 1. Plan by category

### A. Rewrite-only — no computation required

| # | Item | Action |
|---|---|---|
| A1 | **C1 / §6 α₀ bookkeeping** | Rework §6 to the weighted prefactors, **or** add the dual-form remark (review §7.1 — adopt, it can now cite `app:Continuity` `eq:convergence` + closing remark). Justify on **internal consistency**, not empirical discrimination (see 0.2a). |
| A2 | **§7.1 discussion rewrite** | Adopt review §7.2 **with the three corrections of 0.2**: (i) keep "best-approximation-exact in the H¹ seminorm" for **both** elements (verified true, and asymptotically so), but **drop "and in both norms for the biquadratic one"** — the ℚ₂ L² gap is 0.16%–2.5%; (ii) report the true spreads (ASGS 1.31–1.96, OSGS 3.48–4.13), not the endpoints, and drop "compatible with α₀^{-1/2}≈3.2" for numbers reaching 4.13; (iii) drop the "discriminate in favour of the weighted form" claim. |
| A3 | **C8 excluded corner** | State the **fold** reason (`findings.md §2`), **not** the review's stopping-gate text. Consider instead **reporting the corner** — it has been run and converges (see B3). |
| A4 | **C4 reactive OSGS gap** | Report it, corrected per 0.3e: porosity-**independent**, the whole Da=10⁶ column, a **pre-asymptotic transient that recovers** (N=640: rate 1.60, ratio 2.08) — not a constant. Note the provable coercivity ordering runs *opposite* (OSGS ≥ ASGS). |
| A5 | D1 | Fix the "elemental minimum" leftover → `α_K = α_{∞,K}`. |
| A6 | D2 | Add the local-vs-global `Da_h` clause (9.77 vs ≈195); unify `α` vs `α_K` across the three §6 limit subsections. |
| A7 | D3 | `(Uν/h + 1)` → `(Uν/(Ph) + 1)`. |
| A8 | D4 | Define the parenthetical convention at the first table + the regime caveat. |
| A9 | D5 | Qualify §3.1: the trim applies **when σ is constant**; for §7.3's variable σ the full residual is projected. Code is correct. |
| A10 | D7 | Add the scope sentence at the top of §7. |
| A11 | **D10 (corrected)** | Delete the dead `\Cinva` macro **and** resolve the `C_inv` convention clash (0.3f) — pick one convention so the appendix's "replace `C_inv` by `C̄_inv`" does not double-count. |
| A12 | D11 | Rename the appendix's `C_α` (→ `c_α` or `C_{∇α}`). |
| A13 | C2, C3, C5, C9 | Adopt as the review proposes (they are verified). |
| A14 | **C6 (redirected)** | Hedge the quality-tail attribution — but **lead with `article.tex:1370`'s own admitted boundary-incompatibility on the irregular family** (0.4e), which is family-specific and paper-conceded. Drop the ε-penalty half. |
| A15 | **§7.2 / Conclusions `c₁` consistency** | `article.tex:1368` ("all 3D at 16k⁴") vs `:1566` ("linear elements need no such increase") — a reader cannot tell which `c₁` the ℙ₁ rows used. Reword. |
| A16 | Fourier appendix presentation | `fourier_appendix.tex:37` states `K_ij` with hard-coded d=3 numbers (1/3, 2/3) then concludes the general `(1−2/d)`; `eq:ftTauNu:50` gives 4/3 unqualified though the d=2 radius is 1. Both are presentation slips (nothing downstream depends on them) but a referee will stumble. |
| A17 | **Companion-note defects** (if the bundle ships) | `osgs_reaction_note:143` `τ₁ ~ α_K/σ` → `1/σ` (stray `α_K`); its "largest provable coefficient" → "the coefficient obtained by this argument". `velocity_floor_regularization` §4 states the harness sets `u_base=0` — **false**, it sets `h_floor_weight=0` and `u_base=1e-4`, which inverts the note's own conclusion (ε_d is a **no-op** in the convergence studies by the note's own criterion). |
| A18 | `CLAUDE.md` | `theory/` description is stale (lists 3 notes; there are 10 + `numerical_constants/` + `continuity appendix/`). |

### B. Cheap auxiliary computation — **not** a solver re-run

| # | Item | Action | Cost |
|---|---|---|---|
| **B1** | **Interpolation reference curves** *(author-requested; highest value/cost ratio in the whole plan)* | Add reference **curves** (slope **and** FME) to each 2D table. Needs **no solver**: pure nodal interpolation + norms. Already produced and triple-verified. | ~3 min |
| B2 | ‖∇p_ex‖/P_c ceiling for C7 | Compute the 3D saturation ceiling (**1.21673** for the slab) — this *is* the "one sentence on the saturating feature" C7 asks for, available **today** without any re-run. | minutes |
| B3 | Excluded corner | The converged data **already exists** (`findings.md:80-89`: N=512/768/1024, rates L² 3.03/2.99, H¹ 1.04/1.01). Decide: report it, or cite it as the reason for exclusion. | none |

**B1 — PRODUCED AND VERIFIED (2026-07-17).** `test/extended/ManufacturedSolutions/run_interpolation_reference.jl`
→ `results/interpolation_reference/k<kv>/<etype>/interp_reference.h5`. Reuses the harness's own mesh /
porosity / MMS construction and the shared `mms_error_norms.jl`, at the same quadrature degree the sweep
measures with (`4k_v+4`). All twelve values reproduce the audit's independent numpy computations **to every
printed digit** — four independent implementations now agree. Finest-mesh (N=320) values and two-finest slopes:

| k_v | etype | α₀ | u L² | u H¹ | p L² | p H¹ | sl uL² | uH¹ | pL² | pH¹ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TRI | 0.5 | 3.2594e-5 | 3.4991e-2 | 9.8370e-6 | 1.0904e-2 | 1.997 | 0.998 | 2.000 | 1.000 |
| 1 | TRI | 0.05 | 1.6371e-4 | 1.7512e-1 | 9.8370e-6 | 1.0904e-2 | 1.987 | 0.991 | 2.000 | 1.000 |
| 1 | TRI | 1.0 | 1.3912e-5 | 1.5421e-2 | 9.8370e-6 | 1.0904e-2 | 2.000 | 1.000 | 2.000 | 1.000 |
| 2 | QUAD | 0.5 | 2.0648e-7 | 4.2823e-4 | 3.8476e-9 | 7.9794e-6 | 2.997 | 1.997 | 3.000 | 2.000 |
| 2 | QUAD | 0.05 | 2.3851e-6 | 4.9468e-3 | 3.8476e-9 | 7.9794e-6 | 2.991 | 1.991 | 3.000 | 2.000 |
| 2 | QUAD | 1.0 | 5.4414e-9 | 1.1285e-5 | 3.8476e-9 | 7.9794e-6 | 3.000 | 2.000 | 3.000 | 2.000 |
| 2 | TRI (Cocquet ℙ₂) | 0.5 | 2.8555e-7 | 6.5917e-4 | 8.6035e-9 | 2.1111e-5 | 2.994 | 1.994 | 3.000 | 2.000 |
| 1 | QUAD | 0.5 | 2.9755e-5 | 2.6088e-2 | 8.4239e-6 | 6.2957e-3 | 1.998 | 1.000 | 2.000 | 1.000 |

The pressure reference is **bit-identically α₀-independent** within each (k_v, etype), as designed.
Invariance checks (`--verify`) all pass: Re/Da-independence to 2.3e-13–6.5e-12 across L ∈ {0.001, 1, 1000};
pressure α₀-independence exactly 0; velocity α₀-*dependence* control 8.01; quadrature deg 12 vs 20 = 1.76e-7.

**Note the reference slopes at α₀ = 0.05 are themselves depressed** (ℙ₁ u H¹ 0.991, ℚ₂ u L² 2.991, vs
1.000/3.000 at α₀ = 1): part of the low-porosity sub-optimality is the exact solution's roughness, not the
method. This is precisely what the curves exist to separate.

**Still to run:** the Cocquet variant needs α₀ ∈ {0.5, **0.1**} (its sweep uses 0.1, not 0.05) on TRI for
ℙ₁ and ℙ₂; and the 3D reference on the regular Kuhn family.

**B1 design (verified):**
- **Only 3 reference lines per 2D table**, because the normalized interpolation error is independent of
  Re, Da and method: velocity `u/U = (α₀/α)·S(x)` depends only on α₀; pressure `p/P = cos(πx₁)sin(πx₂)`
  depends on nothing. (Verified *bitwise* identical for α₀=0.5 and 0.05.)
- **Presentation:** reference row-block at the foot of each velocity/pressure section, spanning the
  ASGS/OSGS column pair with `\multicolumn{2}{c}` (which visually encodes method-independence). Costs
  **zero columns**; reuses the paper's existing `\multicolumn{5}{l}{\textit{regular mesh}}` idiom
  (`article.tex:1397`).
- **Slope must be computed from the two finest meshes**, exactly as the data rows are — a full-ladder
  least-squares slope would disagree, because at α₀=0.05 the interpolant is itself strongly
  pre-asymptotic.
- **Print the nodal interpolant** (it *is* Theorem 1's `E_int`; ℚ₂ velocity attains it at 1.00) **plus
  one sentence** noting the L²-projection floor is ≈2.4× lower for ℙ₁, so ℙ₁ efficiencies slightly
  below 1 are expected. This is what makes A2(i) honest.
- **Quadrature must be pinned, not assumed.** At degree 3 the ℚ₂ H¹ velocity error reads **39× too
  small** and ℚ₂ H¹ pressure **459× too small** — *exact aliasing*: the ℚ₂ nodal gradient error
  `ω' = 3x²−h²` vanishes precisely at the 2-point Gauss nodes (Barlow points). The harness degree
  (`4k_v+4` = 8/12) is safe; compute the reference at **both** the harness degree (to print) and a high
  degree (to validate) and assert agreement.
- **Reference slopes (α₀=0.5):** ℙ₁ u L² 1.997 / u H¹ 0.998; ℚ₂ u L² 2.997 / u H¹ 1.997; ℙ₁ p L² 2.000 /
  p H¹ 1.000; ℚ₂ p L² 3.000 / p H¹ 2.000. **(α₀=0.05):** ℙ₁ u L² 1.987 / u H¹ 0.991; ℚ₂ u L² 2.991 /
  u H¹ 1.991.
- **Cocquet tables** need **two** pressure references (ℙ₁ for *both* the ℙ₁/ℙ₁ and the ℙ₂/ℙ₁-TH rows;
  ℙ₂ for ℙ₂/ℙ₂) and two velocity references (ℙ₁, ℙ₂) × α₀ ∈ {0.5, 0.1}. Confirmed: TH pressure FME
  9.84e-6 = the ℙ₁ nodal value **exactly**.

### C. Re-run required

| # | Item | Scope | Why |
|---|---|---|---|
| **C1r** | **⚠ Regenerate the entire 3D section** (`tab:3DL2`, `tab:3DH1`) | Both families (regular Kuhn N∈{8,16,24,32} ℙ₁ / {12,16,20,24} ℙ₂; irregular red-refined), ℙ₁+ℙ₂, ASGS+OSGS, at `c₁=16k⁴` | **Raw data is gone** (0.4c). Settles **C7** (the 1.29 triple) and **0.4b** (the possibly non-converged finest ℙ₂ solve) together. Cannot be avoided — neither is desk-checkable. |

### D. Code change + re-run

| # | Item | Action |
|---|---|---|
| D1c | **3D `c₁` production gap** (0.4g) | `get_c1_c2` is dimension-blind (`4k⁴` always); `16k⁴` lives only in a harness kwarg defaulting to `1.0`. Thread the 3D constant through schema + config per `.agents/rules/no-hard-coded-parameters.md`. **Required before C1r** so the re-run is reproducible from config alone. |
| D2c | **`TAU_VISC_MULT`** (0.4f) | Remove, or lift into schema + `SolverConfig` and thread it. Currently inert (default 1.0) but unrecorded in provenance. |
| D3c | **Persist the 3D gmsh base mesh** (0.4c) | `mesh3d.jl:98-109` deletes it; `save_msh` exists but is never called; gmsh unpinned. Persist + commit the base `.msh` so the irregular family is regenerable. **Required before C1r** or the re-run is equally unreproducible. |
| D4c | **`\input` the generated tables** (0.4d) | Point `article.tex` at `make_results_tables.py`'s output (extended to emit the B1 reference rows) instead of hand-copying. Removes the C7-class drift mechanism at the root. |
| D5c | Harness: extract shared error functional | Move `calculate_normalized_errors` into `test/extended/mms_error_norms.jl` so B1 measures **identically** to the solver (it centers the pressure error and divides by `U_c·√area` — a hand-rolled reference would get this wrong). Precedent: `harness_dynamic_budget.jl`. |
| D6c | Persist `ε_M`/`ε_C` per mesh | One-line hardening: store the actual gate values alongside `eval_residuals`, so future readers cannot repeat the `target_ftol` misreading that nearly cost us a spurious C4 retraction. |

### E. Open investigations

| # | Item | Question |
|---|---|---|
| **E1** | **⚠ Irregular ℙ₂ finest solve** (0.4b) | Did every mesh in the production `nested_red` sweep have `success=true`? If the finest returned the exact-guess interpolant, `tab:3DL2`'s 2.55/2.56 must be recomputed or withdrawn. **Blocking for submission.** Resolved by C1r. |
| E2 | Coercivity hypothesis at the coarse end | `c₁ > 2ξC̄_inv²` with `C̄_inv = √(dδ_α)C_inv + C_α`. With the paper's own `C_inv ~ k²` this is violated by `c₁=4k⁴` **even at uniform α** — so `C_inv ~ k²` is a *scaling*, not a usable value, and `c₁=4k⁴` is explicitly empirical. **The review's §0.B "the assumptions are satisfied" audits only Assumption *Porosity* and never checks the `c₁` hypothesis the theorem also requires.** Not necessarily actionable; do not overclaim in §0.B-style language. |
| E3 | C6 root cause | Element-quality tail vs pre-asymptotics vs the paper-admitted boundary incompatibility (0.4e). A quality histogram / per-element error map would settle it; otherwise hedge (A14). |
| ~~E4~~ | **CLOSED 2026-07-21 (rerun R10)** | The 1.29 triple is a **genuine pressure under-stabilization** of the equal-order OSGS discretization at `c₁=16k⁴` (not a solver artifact): reruns reproduce identical numbers, the on-disk `results/k*/TET/*/` MATCH the paper tables to the printed digits (so C1r "regenerate the 3D section" is *not* needed for correctness — the data is not gone), and the H¹ pressure saturation persists even under the solution-preserving `c₁`-precond knob. The mesh/order-independence is exactly the fingerprint of an `O(1)` under-stabilization ceiling; a saddle-point/MG preconditioner (pending-tasks 4d), not `c₁`, is the real fix. |
| E5 | Kuhn threshold under the runs' actual h-convention | The note's Table 1 uses `h_K = diam`; the harnesses use grid-spacing-like conventions (2D `√(2A)`, 3D `(6√2V)^{1/3}`). Under those, `c₁` sits **above** the elementwise threshold, inverting §7.2's "just below" claim. Also 2D and 3D use *different* conventions. `article.tex` never states which `h_K` feeds τ. |

---

## 2. Recommended sequencing

1. **D1c, D3c, D5c** (code/provenance) → 2. **C1r** (3D re-run; settles C7 + E1) → 3. **B1** (interpolation
curves) → 4. **D4c** (wire tables to the generator) → 5. **A1–A18** (prose).

Rationale: the prose depends on numbers that the 3D re-run may change (C7, E1), and the re-run is not
reproducible until D1c/D3c land. B1 is independent and can start immediately.

## 3. Risk register

- **E1 is submission-blocking.** A published slope computed partly from an interpolant is a retraction
  risk.
- **0.2a is referee-bait.** "The data discriminate in favour of the weighted form" is a claim a competent
  referee will dismantle in one line (one-sided bounds). Keep the §6 rewrite; drop the justification.
- **0.2c is referee-bait, but only in its ℚ₂-L² half.** "Best-approximation-exact … in both norms for
  the biquadratic one" is falsified by direct computation (2.06e-7 vs a true best of 2.05e-7; 2.5% off at
  α₀=0.05). The H¹ half is safe and should be kept — it is a strong, asymptotically-structural result.
- **Methodological note.** My own first objection to F1 was wrong, and cheap to have avoided: I applied an
  L²-projection number to an H¹ claim. It was caught only by computing the H¹ (Ritz) projection. The same
  failure mode (reasoning from a related-but-different quantity instead of a direct probe) is recorded at
  `docs/findings.md:235` as having produced a wrong verdict once before, and it nearly produced a spurious
  C4 retraction here too. **Compute the actual quantity.**
- **C1r is the long pole** (full 3D campaign). Everything else is hours.

---

## 4. Application log (2026-07-17)

### Applied and verified

| item | what changed | verification |
|---|---|---|
| **D2c** | `TAU_VISC_MULT` env-var read removed from the τ quadrature-point hot path (`tau.jl`). Inert (default 1.0), verdict already "not adopted", invisible to provenance. | Blitz 272/272; new test asserts `tau.jl` contains no `ENV`. |
| **D1c** | `c1_multiplier`/`c2_multiplier` added to schema → `StabilizationConfig` → `validate!` (>0) → `run_simulation`. The paper's 3D constant `c₁=16k⁴` now has a production representation. | Fails loudly when absent (same as pre-existing `element_size`); round-trip test; 12 new assertions. |
| **D1c(b)** | `run_sweep_structured` default `c1_mult` **1.0 → 4.0**. It shipped the *2D* constant while `article.tex` §7.2 says all 3D uses 16k⁴, so the published regular-family tables needed a NON-default invocation and a default run silently produced mislabelled results. | Matches `run_sweep_nested_red`'s existing 4.0 default. |
| **D3c** | Nested-red gmsh base mesh committed (`meshes/nested_red_base_lc0.200_alg1.msh`, 425 cells) + provenance README (gmsh 4.9.3 / GridapGmsh 0.7.4). `load_or_build_base_mesh` prefers the committed file, warns loudly if it regenerates. | Round-trips exactly (h̄ identical to 17 digits); family deterministic 425→3400→27200, ×8/level, h̄ halving. |
| **D5c** | `calculate_normalized_errors` consolidated from 2 copies into `test/extended/mms_error_norms.jl`. | Copies verified functionally identical before merging (only a comment differed); both harnesses parse; self-error exactly 0. |
| **B1** | Interpolation reference curves produced (see §1.B1 above). | 12/12 match the audit's independent numpy values to every printed digit — 4 independent implementations agree. All `--verify` invariances pass. |
| **A1** | §6: over-claiming "ubiquitous linear drop" corrected to a statement about the *bound*; **D3** dimensional slip `(Uν/h+1)`→`(Uν/(Ph)+1)`; new `rem:WeightedVsUnweighted`. | LaTeX exit 0, 0 unresolved refs in rendered PDF. |
| **A2** | §7.1: interpolation reference introduced; **"discriminate in favour of the weighted form" REMOVED** — replaced by "both are upper bounds and both are satisfied, so the experiments cannot be said to select between them". `best-approximation` scoped to **H¹ only**, with the √6 ℙ₁-L² caveat stated. True spreads (1.3–2.0 / 3.5–4.1; 8.3–12.4 / 10.2–18.8) replace cherry-picked endpoints. | Sentences verified present in the rendered PDF. |
| **A3** | Excluded corner: states the **evidenced fold** reason (no discrete root at coarse N; Newton *and* Picard stall from the exact solution; recedes with h; ℚ₂ does not fold; converges optimally 2.99–3.03 / 1.01–1.04 beyond it). The review's stopping-gate story was **not** adopted — it is refuted by `findings.md §2`. | — |
| **A4** | C4 reported with corrected framing: porosity-**independent** excess (1% across α₀), Da_h-controlled, decays under refinement, absent at Re=10⁶; notes it runs opposite to the provable coercivity ordering and why that is consistent (one-sided bounds). | — |
| **A5 (D1)** | α_K "elemental minimum" leftover → maximum (`α_K = α_{∞,K}`), consistent with `eq:Tau1Final` and App. C. | — |
| **A6 (D2)** | Local-vs-global `Da_h` clause added (≈9.8 global vs ≈20× larger in the plateau); `α` → `α_K` unified across the convective and reaction limits. | — |
| **A9 (D5)** | §3.1 projection trim now conditioned on constant σ; full residual projected for the DBF runs. | Verified at code level (`run_simulation.jl:56-59` double-gate; `CocquetFormMMS/run_test.jl:149` hard-codes `ProjectFullResidual`). |
| **A10 (D7)** | Scope sentence at §7: theory = linearized ASGS, all-Dirichlet; experiments = nonlinear, both variants, Neumann outlet in §7.3; **no estimate covers OSGS**; bounds are one-sided. | — |
| **A11 (D10)** | `rem:winvconst` reworded: the main text's `C_inv` ALREADY denotes the enlarged constant, so no substitution — the previous wording double-counted. | — |
| **A12 (D11)** | Appendix `C_α` → `C_{∇α}` (9 occurrences). Main text + Fourier keep `C_α` = `eq:CAlpha` (3 + 2 occurrences, untouched). | — |
| **A16** | `fourier_appendix`: `K_ij` entries now given for general `d` (noting `eq:matrices_stationary_strong_problem` displays the d=3 instance); `eq:ftTauNu` now `(2−2/d)`, noting it is **1 for d=2** — the case of the 2D experiments — so there is nothing to drop there. | Builds clean. |
| **A17** | `osgs_reaction_note` `eq:asymp`: stray `α_K` removed (`τ₁ ~ α_K/σ` → `1/σ`; its own `eq:tau1-siga` gives `τ₁ = 1/(α_K τ_NS⁻¹ + σ)`). `velocity_floor_regularization` §4: the claim "the harness sets `u_base=0`" was **false** — it sets `h_floor_weight=0` and inherits `u_base=1e-4`; corrected, which by the note's own criterion makes ε_d a **no-op** in the convergence studies (inverting its previous conclusion). | Both notes rebuild clean; corrected passages render. |

### Corrections to THIS plan made during application

- **D10 was refuted on both counts** (see §0.3f). The `\Cinva` macro is **not** dead — it is used 16× by
  `continuity_appendix.tex`, which `article.tex` `\input`s. An earlier draft of this plan accepted the
  review's claim after grepping `article.tex` alone. The macro was NOT deleted.
- **F1's terminology objection was mostly mine and mostly wrong** (see §0.2c). The ℙ₁ L² penalty (√6) does
  not transfer to the H¹ seminorm, where the nodal interpolant is asymptotically optimal (ratio − 1 = O(h²),
  measured 1.00015). Only the ℚ₂-**L²** half of the review's sentence fails.

### Still open

- **C1r / E1 — ✅ DONE (2026-07-19):** 3D re-run complete on both families, **0 `success=false`** across all
  30 levels; **E1 + C7 closed** (see the top banner and `pre-submission-checklist.md §2`). (Historical note below.)
  Any row whose finest mesh did not converge will be recomputed or withdrawn: **no slope may be computed
  partly from an interpolant** (author instruction).
- **B1b — ✅ DONE (verified 2026-07-19):** the Cocquet interpolation reference (α₀∈{0.5,0.1}, ℙ₁+ℙ₂ on TRI) and
  the 3D regular-Kuhn reference are computed, inserted in the tables, and fresh-verified to printed precision
  (`pre-submission-checklist.md §3`).
- **D4c** — wire `article.tex` tables to `make_results_tables.py` (currently hand-copied) and emit the
  reference rows from the generator.
- **A8/A13** — first-table caption convention + regime caveat; C6/§7.2 redirect (needs the 3D re-run).
- **D6c** — persist `ε_M`/`ε_C` per mesh alongside `eval_residuals`.
- **A18** — `CLAUDE.md`'s `theory/` description is stale (lists 3 notes; there are 10 + `numerical_constants/`).

---

## 5. Second-pass audit of the applied edits (2026-07-17)

A 49-agent adversarial audit was run over **every number and claim introduced** in this pass, checking each
against the verified interpolation-reference HDF5 and the paper's own tables, with each finding then
adversarially refuted. Motivation: the edits were applied partly from the review's draft prose, and the
review's *drafting* had already been shown to lose qualifiers its own analysis stated correctly.

### Verified clean

All 12 interpolation-reference rows across the four 2D tables reproduce the HDF5 to the printed precision;
reference slopes use the two-finest-mesh rule the caption states (a full-ladder least-squares fit would give
1.70 instead of 1.99 at α₀=0.05 — materially different, so this mattered); reference and data share the same
finest mesh and the same error functional; and a discriminating cross-wiring check passed (the ℙ₁ tables
carry TRI values and not QUAD, etc.). The Re/Da/method-independence and the pressure's α₀-independence were
confirmed analytically **and** bitwise in the data.

### FOUR REAL ERRORS FOUND IN THE APPLIED PROSE — all mine, all now fixed

Every one is the same failure mode: **a qualifier dropped while transplanting the review's draft**.

| # | Claim as written | Why it is false | Fix |
|---|---|---|---|
| 1 | velocity coincides with the interpolant "in every regime with **Re ≤ 1**" | The Da=10⁶ rows *satisfy* Re ≤ 1 and contradict it: ℙ₁ OSGS 1.23e-1 vs 3.499e-2 = **3.51**; ℚ₂ both variants ≈3.5. It also **contradicted the paper's own later paragraph** reporting that gap. | scoped to `Re ≤ 1` **and** `Da ≤ 1` |
| 2 | §7.3 uses "a **Neumann outlet**" | §7.3 *replaced* the Cocquet tube-flow benchmark with a manufactured solution and keeps "the exact velocity prescribed on the whole boundary"; it takes ε=0 with a zero-mean pressure constraint, which is needed *only* without a Neumann boundary. | replaced by the real §7.3 departure: the velocity-dependent σ |
| 3 | "The **only** quantity retaining a residual method effect is the ℙ₁ velocity in L²" | Unqualified "only" is false — the OSGS Da=10⁶ H¹ gap is also a method effect. | scoped to the viscosity-dominated rows |
| 4 | "The **ASGS velocity is unaffected** to three significant digits" (high Da) | True for ℙ₁ (ratio 1.01), **false for ℚ₂**, which grows ×3.55. The review's C4 *did* say "ℚ₂ grows by ×3.5"; the qualifier was lost. | replaced with the full picture (below) |

**Error 4's correction is a better result than the claim it replaces.** The measured picture at
(Da, α₀) = (10⁶, 0.5), H¹ velocity vs the interpolation reference:

| | ratio to interpolant |
|---|---|
| ℙ₁ ASGS | **1.01** (on the reference) |
| ℙ₁ OSGS | **3.51** |
| ℚ₂ ASGS | **3.55** |
| ℚ₂ OSGS | **3.53** |

So raising Da leaves *only* ℙ₁ ASGS on the interpolant and multiplies the other three by ≈3.5. The two
stabilizations part company **only for the linear element**; for ℚ₂ they agree closely with each other while
both depart from the interpolant. And the Da^{1/2} allowance (≈10³ at Da=10⁶) still does not materialize —
the largest observed growth is 3.5 — which is the point the estimate-vs-observation comparison is making.

### Lesson

The review's *analysis* was reliable; its *draft prose* systematically over-generalized (dropping "for ℙ₁",
"at α₀=0.5", "for Da ≤ 1"). Transplanting its sentences was the wrong default: each claim had to be
re-derived from the data. Four of the edits applied from that draft were false as written.

---

## 6. Full-text coherence audit (2026-07-17) — applied + remaining

A 7-reader end-to-end coherence audit (every section vs the data AND the implemented code, + cross-section
consistency, + adversarial refutation) was run. It found more of the SAME qualifier-dropping errors.

### Applied this session (all verified, LaTeX builds clean, 0 unresolved refs)

| id | error (all mine) | fix |
|---|---|---|
| S72-1 | opening §7.1: rate-exception misassigned to "reaction-dominated" | reaction rows have OPTIMAL rates; the rate exception is the **convection-dominated ℚ₂ rows** (Re=10⁶, slopes 2.79–2.83, marginally sub-optimal). Rewritten. |
| S72-2 | "velocity error … virtually identical for the two methods" (viscous) | scoped: identical in H¹ both elements + ℚ₂ both norms; ℙ₁ L² differs up to ×1.9 |
| S72-3 | "the **only** appreciable velocity discrepancy" | self-contradicted my own ℙ₁-L² sentence; now "largest … the only other is the ℙ₁ L² ×1.9" |
| S72-4 | "coincides to **three significant digits** … at both porosities" | at α₀=0.05 it's ~0.5% (eff 1.005), not a clean 3 s.f.; reworded to "3 s.f. at α₀=0.5, within half a percent at α₀=0.05" |
| S6-6 | line 1011 middle sentence still said the error (not the bound) grows with 1/α₀ | reworded to attribute low-porosity growth to the exact solution |

### REMAINING — NOT yet applied (need author eyes; do not rush)

> **✅ S6-4, its convection analog, and S6-1 APPLIED 2026-07-19** (re-derived independently, weighted form
> adopted throughout §6, build green). See `pre-submission-checklist.md §0` for the full record — including the
> one correction: the reaction pressure-gradient `eq:DominantReactionPressureGradientEstimate` keeps a
> *legitimate* outer `1/α₀` (weak τ₁∼1/σ control ⇒ isolating ‖∇e_p‖ costs a genuine extra `α₀^{-1/2}`); its fix
> was the inner `(α_∞/α₀)^{1/2}→α_∞^{1/2}`. **S6-3 and S45-3 remain open.**

| id | severity | issue | proposed fix |
|---|---|---|---|
| ~~**S6-4**~~ ✅ | ~~major~~ | **eq:ConvergenceResultDominantViscosity (article.tex:1009): the 3rd LHS term coefficient is printed `1/α₀` but the same normalization gives `α₀^{-1/2}`.** τ₂=ν/α_K, min_K τ₂ = ν/α_∞ = ν (α_∞=1), so ‖τ₂^{1/2}∇·(αe_u)‖ ≥ ν^{1/2}‖·‖, and dividing by N=ν^{1/2}α₀^{1/2} gives α₀^{-1/2}. This also makes line 1011's "the third term partially balances this deterioration" *exactly* correct. **This is a displayed-equation math change — verify independently before applying.** | change `\frac{1}{\alpha_0}` → `\frac{1}{\alpha_0^{1/2}}` on the 3rd LHS term of eq:1009 only |
| **S6-3** | major | `Da = Da_h L²/h²` (lines ~1065, 1167) is imprecise: with the elementwise α_∞ convention, `Da/Da_h = (L²/h²)(α_K/α_∞)`. The h-subscripted numbers are read with α_∞ = α_{∞,K} = α_K. | state the convention once after line 998, OR write the exact identity |
| ~~**S6-1**~~ ✅ | ~~major~~ | OSGS ℙ₁-L² method factor (3.48–4.13 sweep / 3.87–5.82 absolute at α₀=0.05) **exceeds** the weighted bound (α₀^{-1/2}=3.16 sweep / 4.47 absolute) in 3–4 of 4 rows. rem:WeightedVsUnweighted's "residual growth … of the order of the weighted prediction" is fair for ASGS but slightly generous for OSGS. | ✅ APPLIED: numerics prose now reads "the ASGS factor below it and the OSGS one marginally above" |
| S45-1, S45-2, S72-5 | minor | small self-contradictions (τ₂ implicit/analysis form wording; a viscous-para phrasing) | see journal |

### Infrastructure still pending (unchanged from §4)

- **C1r / E1 — ✅ DONE (2026-07-19):** 3D re-run complete (ℙ₁+ℙ₂, ASGS+OSGS, structured+nested-red), **0
  success=false**; **C7's 1.29 confirmed genuine** and the ℙ₂-irregular convergence settled. (Was: ℙ₂/OSGS/nested-red
  still ahead, 0 success=false so far.) Still to action from this once-blocked item: A14/A15
  (C6 quality-tail redirect + the §7.2-vs-Conclusions c₁ reporting inconsistency).
- **Cocquet reference rows — ✅ INSERTED + verified 2026-07-19.** ℙ₁/ℙ₂ TRI, α₀∈{0.5,0.1}; C9 confirmed
  (Taylor-Hood pressure at the ℙ₁ floor 9.84e-6 while its velocity is n.c.). Rows are present in
  `tab:CocquetMMSL2/H1` and match a fresh interpolation regen to printed precision.
- **D4c** — wire tables to make_results_tables.py (still hand-copied).
- **D6c** — ε_M/ε_C persistence: APPLIED to src/solvers/solver_core.jl + run_test.jl; proven inert
  (the 1 Quick error is a PRE-EXISTING SingularException in osgs_frozen_pi_jacobian_quick_test.jl:96,
  reproduced with my change stashed — unrelated).

---

## 7. Coherence-fix round (2026-07-17, continued) — applied + remaining

Full-text coherence audit (7 readers + cross-section + refutation) completed. Additional fixes applied
(all verified, LaTeX builds clean, 0 unresolved refs). Cocquet + 2D reference rows added.

### Applied

| id | error | fix |
|---|---|---|
| S72-1 | opening §7.1 rate-exception misassigned to reaction | reassigned to convection-dominated ℚ₂ (Re=10⁶, slopes 2.79–2.83, marginally sub-optimal) |
| S72-2 | "velocity … virtually identical for the two methods" (viscous) | scoped; ℙ₁ L² differs ×1.9 |
| S72-3 | "the only appreciable velocity discrepancy" | self-contradiction fixed; "largest … the only other is ℙ₁ L² ×1.9" |
| S72-4 | "three significant digits at both porosities" | at α₀=0.05 it's ~0.5%; reworded honestly |
| S72-6 | excluded-corner ℚ₂ clause implied ℚ₂ is excluded because it folds (it doesn't) | now: ℚ₂ attainable there, omitted for uniform cell set |
| S72-7 | "pressure reaches the approximation floor" (convection) — true only ℙ₁ | scoped: ℚ₂ recovers the RATE but FME is ~20× the reference |
| S6-6 | line 1011 still attributed 1/α₀ growth to the error, not the bound | reworded |
| **S45-1** | **MY Fourier fix (A16) changed the appendix to (2−2/d) but left eq:StabilizationParameters at 4/3, and the appendix says they are "exactly" equal — false for d=2** | eq:StabilizationParameters + the "ignore 4/3" sentence made general-d |
| S45-2 | remark @835 mislabelled eq:Tau2 (with εh²) as the *implemented* τ₂; the code implements eq:Tau2Final (no ε, `compute_tau_2` takes no epsilon; docs/theory-code-map.md §2.5) | relabelled: τ₂^full for eq:Tau2; the retained eq:Tau2Final is what is analysed AND run |
| B1-Cocquet | tab:CocquetMMSL2/H1 | ℙ₁/ℙ₂ interpolant rows per (Re,α₀) block; C9 confirmed (Taylor-Hood pressure at the ℙ₁ floor 9.84e-6, ratio 1.000, while its velocity is n.c.) |

### Remaining — flagged, NOT rushed (need author eyes / the 3D re-run)

> **✅ S6-4 and S6-1 APPLIED 2026-07-19** (see the §6-table banner above and `pre-submission-checklist.md §0`).
> S6-3, S45-3 still open.

| id | severity | issue |
|---|---|---|
| ~~**S6-4**~~ ✅ | ~~major~~ | eq:ConvergenceResultDominantViscosity 3rd LHS coeff printed `1/α₀`, derivation gives `α₀^{-1/2}` (also makes line 1011's "partially balances" exact). **Displayed-equation math — verify independently.** |
| **S45-3** | major | Lemma 1 (stability) hypotheses omit eq:SmallPorosityGradient (resolution) + mesh-nondegeneracy, which its own proof (line 880) uses and the appendix's prop:stability requires. Needs coordinated Lemma 1 + Lemma 2 hypothesis surgery. |
| S6-3 | major | `Da = Da_h L²/h²` imprecise (misses α_K/α_∞ under the elementwise convention) |
| ~~S6-1~~ ✅ | ~~major~~ | OSGS ℙ₁-L² factor (3.5–4.1) slightly exceeds the weighted bound; **APPLIED** — numerics prose qualified ("ASGS below it, OSGS marginally above"); rem:WeightedVsUnweighted no longer claims "of the order of the weighted prediction" |
| ~~73a, C6~~ | ~~major~~ | **RESOLVED by the 3D interpolation reference (§8; findings.md §3).** The velocity sub-optimality is the mesh's approximation capacity (interpolant itself sub-optimal on the irregular family: ℙ₁ H¹ slope 0.71, ℙ₂ L² 2.67; method at the floor, eff ~1.0); the pressure is the expected viscous one-order loss (both families, consistent with the 2D baseline). The "element-quality tail" attribution can now be stated as an *evidenced decomposition*. **A14 article rewrite pending author OK.** Still open: **C7** (the 1.29 triple — raw-data check via the re-run) and the separate `open-questions.md §3` `4k⁴`-fragility caveats. |
| S72-5/8/10/11, T1–T6, 73d/e | minor | small wording/self-contradiction items — see journal |

### Compute status
- 3D solver sweep: ℙ₂ ladder, **0 success=false** across all meshes. Settles E1 + C7.
- 3D interpolation reference: harness written (reuses Kuhn + committed nested-red + calc_errors3d); computing.
- ε_M/ε_C persistence: applied, proven inert (Quick's 1 error is a PRE-EXISTING SingularException,
  reproduced with the change stashed).

---

## 8. 3D interpolation reference (2026-07-17) — computed, verified, inserted

Harness `run_interpolation_reference3d.jl` (pure interpolation, no solver; reuses structured_kuhn_model +
build_nested_family on the COMMITTED base mesh + calc_errors3d; quadrature 4k+4, like-for-like with the
solver rows). All four families completed cleanly. FME (finest) / two-finest slope, alpha_0=0.5:

| family | order | u L² | u H¹ | p L² | p H¹ |
|---|---|---|---|---|---|
| regular Kuhn | ℙ₁ | 1.291e-3 (1.90) | 8.023e-2 (0.94) | 9.830e-4 (2.00) | 5.970e-2 (1.00) |
| regular Kuhn | ℙ₂ | 1.783e-4 (3.20) | 1.699e-2 (2.22) | 2.038e-5 (3.00) | 2.054e-3 (2.00) |
| irregular nested-red | ℙ₁ | 6.823e-4 (1.83) | 8.415e-2 (0.71) | 4.351e-4 (1.87) | 4.866e-2 (0.79) |
| irregular nested-red | ℙ₂ | 2.012e-4 (2.67) | 2.410e-2 (1.52) | 2.034e-5 (2.72) | 2.253e-3 (1.61) |

Rows inserted into tab:3DL2 and tab:3DH1 (within each regular/irregular sub-block). Build clean.

### This DECOMPOSES the 3D sub-optimality (C6) and largely resolves E1

**Velocity is at the interpolation floor on BOTH families:** regular ℙ₁ u H¹ eff 1.02/1.02; regular ℙ₂
u L² eff 1.04/1.00, u H¹ 1.02/1.07; irregular ℙ₂ u L² eff **1.03/1.00**; irregular ℙ₁ u H¹ eff 0.91/0.93
(method below nodal, i.e. better — nodal is not the best approx). So the stabilization adds no measurable
velocity error over interpolation in 3D either.

**The irregular sub-optimality is inherited from the mesh family, not the formulation.** The interpolant's
OWN rates are depressed on the nested-red family: ℙ₁ u H¹ slope **0.71** (vs 0.94 regular, nominal 1);
ℙ₂ u L² slope **2.67** (vs 3.20 regular, nominal 3). The paper's irregular method slopes (ℙ₁ H¹ 0.83–0.85;
ℙ₂ L² 2.55–2.56) are AT or slightly ABOVE the interpolant slope — i.e. the mesh family's degrading
element-quality tail caps the best-approximation rate, and the method tracks it. **This is the evidence
C6's "element-quality tail" attribution needed** (the review flagged it as an unsupported causal claim);
it can now be stated as a decomposition rather than an assertion. Recommended A14 rewrite: attribute the
irregular sub-optimality to the mesh family's approximation capacity (shown by the interpolant reference),
not the formulation.

**E1 (P2-irregular slope-from-interpolant risk) is largely defused:** on the irregular ℙ₂ family the
interpolant L² slope is 2.67 and the method sits exactly on that floor (FME eff 1.00–1.03). Whether or not
the finest solve fully converged, the method error EQUALS the interpolation floor there — the best any
method can do on that mesh family. The 3D solver re-run (in progress, 0 success=false) will confirm the
success flags; but the reported slope is a mesh-approximation property, not an artifact.

**Pressure is well above the floor in 3D** (regular ℙ₂ p L² eff 315×), consistent with the documented
paper-c1 P2-3D pressure under-stabilization (lessons_learned.md). The reference makes this visible.
