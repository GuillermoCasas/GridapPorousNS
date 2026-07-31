# Pending tasks — backlog (by block)

**Purpose.** The single living backlog for this solver. Every actionable next step, grouped by the area
of work it touches, each pointing at the evidence/doc so it can be picked up cold. This file is the
*only* place open code-correctness issues live now (the former `known_issues.md` was folded in here on
2026-07-10; its resolved items moved to [`findings.md`](findings.md)).

**Blocks**

1. [Theory](#1-theory) — derivations, notes to write, paper-math questions.
2. [Code–theory consistency](#2-code-theory-consistency) — where code and paper/algorithm must be kept aligned.
3. [Formulation](#3-formulation) — the weak form, τ, viscous/reaction operators, quadrature.
4. [Solver / numerics](#4-solver--numerics) — Newton/Picard, gates, preconditioning, acceleration.
5. [Post-processing](#5-post-processing) — plotting, rate computation, honest reporting of results.
6. [Input / output & provenance](#6-input--output--provenance) — config schema, result writing, traceability.
7. [Tests & validation sweeps](#7-tests--validation-sweeps) — guard rails and the measurements that close open questions.
8. [Cleanups](#8-cleanups) — dead config/code, hygiene.

Nothing here blocks the headline results (2D k1/k2 sweeps optimal; OSGS-P1-3D solved; P2-3D
**solver-convergence** and **accuracy** verdicts resolved — see [`findings.md`](findings.md) §3). These are
refinements, completeness measurements, and hygiene. Open *questions* (as opposed to tasks) live in
[`open-questions.md`](open-questions.md); settled results in [`findings.md`](findings.md).

---

## 1. Theory

### 1a. Write the coercivity-margin / element-aware `c₁` note (LaTeX)
The 3D-P2 resolution rests on permanent theory currently stranded in docs: the viscous 2nd-derivative
subscale is anti-coercive by construction (`B_S` carries `−τ‖𝓛_visc V‖²`), dominated by `c₁ > 2ξ·C_inv²`;
`C_inv²` is mesh-independent and element-type-dependent (Kuhn TET **214** vs Q2 quad **60**), so the paper's
`4k⁴` is under-margined for high-`C_inv` tets and the remedy is an element-aware `c₁` (`article.tex` line 910).
Write this as a `theory/coercivity_margin_note/` LaTeX note (with the `C_inv²` table and the independent
clean-room re-derivation that reproduced 214/60/48/180), and cross-link from `article.tex` line 910.
Evidence: [`findings.md`](findings.md) §3, [`mms/p2-3d.md`](mms/p2-3d.md) §A.

### 1b. (Deferred / theory-completeness) CocquetFormMMS low-α fold — clean σ̃_α isolation
The exact fold mechanism is **OPEN** ([`open-questions.md`](open-questions.md) §1). The leading σ̃_α /
reaction-in-stabilization hypothesis is paper-grounded but unconfirmed — the `STRIP_REACTION_FROM_STAB`
A/B was confounded by τ₁ entanglement (stripping σ also enlarges τ₁). The clean isolation: **strip σ from
`𝓛U`/`𝓛*V` only, holding τ₁ physical**. Theory-completeness, not a deliverable prerequisite (§4.1
convergence above the fold does not depend on it). Alternatives: (a) finish the OSGS trim-vs-full A/B at
low α; (b) for ASGS, a code change to strip σu from the stabilization residual. Evidence:
[`cocquet/cocquet-form-mms-status.md`](cocquet/cocquet-form-mms-status.md) §4.3.

### 1c. Paper editorial items needing author judgment
All items resolved 2026-07-19: (`centered_encoding.tex` — a short reproducibility-level section was added to
`article.tex`, the full note kept as a companion; `supplement.tex` — removed, SIAM boilerplate with no article
cross-reference; results-section figures — **tables-only settled**, the `\Guillermo{Add figures}` note gone, one
porosity figure kept; the "Kratos Multiphysics" claim — the paper reads Gridap, 0 Kratos mentions). Full list:
[`open-questions.md`](open-questions.md) §4.

**Editorial-markup flatten (prior-audit B04; re-raised as N7/P1 by the 2026-07-30 audit) — ✅ DONE
2026-07-30.** All **1011** wrappers unwrapped to their plain bodies and the three `\newcommand`s
deleted: `article.tex` 461 `\amend` + 8 `\Guillermo` + 3 `\Joaquin`, `article_v2.tex` 495 + 8 + 3,
`asgs_convergence.tex` 6, `elemental_matrices_appendix.tex` 22, `fourier_appendix.tex` 5 (App. D had
none). Done with a brace-balancing scan honouring escapes and comments, not a regex — 22 bodies span
line breaks and many nest braces. Four commented-out draft paragraphs in each main carried
`\Guillermo{REVIEW: CHECK}`-style author notes; those paragraphs were removed entirely by the
dead-comment sweep below, so the notes are gone from the source.

**Dead commented-out draft fragments deleted (same session).** Eight blocks, **116 lines per main**
(byte-identical block set in v1 and v2), plus one dead trailing fragment on the `\ViscProj` paragraph:
the abandoned Darcy-threshold paraphrase + non-Darcy aside; the transient/initial-condition aside; the
alternative-`A_U` remark; the whole dropped *"Linearization of the differential operator"* subsection
(generalized fixed point → contraction condition → Picard → Newton–Raphson, 75 lines); the
`eq:AdjointFlux_commented` equation; the superseded `eq:VMSWeakFormSystem_commented`; the dropped
Projection / SGS-linearization subsubsections; and `%\input{variational_crimes}`, a placeholder for a
section never written (no such file exists). Located by first/last-line text rather than line number
(the two mains are offset), with every intervening line asserted to be a comment and one adjacent
blank absorbed so paragraph spacing is untouched. **Checked first that no live text `\cref`s any label
defined only inside them** — `sec:LinearizationDifferentialOperator`, `eq:ContractingMapping`,
`eq:TotalDerivativeOfResidual`, `eq:AdjointFlux_commented`, `eq:VMSWeakFormSystem_commented`,
`eq:simplified_weak_form_subscales_OSGS{,_static_projection}` occur nowhere else in the repo. Removing
`eq:AdjointFlux_commented` also closes the *"commented duplicate label"* cosmetic residue in
`ChatGPT audit/latest_audit_response.md`, and the source now has **zero** commented-out `\cite` lines,
so a bare `\cite` grep matches the real citation set exactly.

**Kept deliberately** — these are load-bearing and must not be swept in a future pass: the preamble
and macro notes, the SIAM template `% REQUIRED` markers, the `v1/v2 DIVERGENCE` flags, the
viscous-projector relocation note (which names the guarding gate rule and says what not to
reintroduce), the authoritative `%%` notation convention, `shared.tex`'s `[known-fragility]` nameref
patch rationale, the appendix-header provenance blocks, the appendix-ordering note, and the commented
`\input{osgs_appendix.tex}` that documents the App-D twin switch.

**Gated output-neutral after each of the two edits, and it held exactly both times:** `article` 81 pp /
778 newlabels, `article_v2` 118 pp / 980 newlabels, 0 undefined + 0 multiply-defined + 0 overfull >1pt
in both — all identical to the pre-flatten build — and `pdftotext -layout` diffs of **zero lines** for
both PDFs. SymPy suite 636/636 (incl. `document_hygiene` 63/63 off the fresh logs;
`assembly_consistency` 4/4 with all 11 anchor definitions still detected, which is the check that the
previously `\amend`-wrapped `G_αP`/`Q_φ` LHSs still parse); `make_3d_tables.py --check` green.

One code dependency moved with it: `test/extended/ManufacturedSolutions3D/make_3d_tables.py` both
*emitted* and *re-parsed* `\amend`-wrapped interp rows, so its `interp_row` writer and `irx` check
regex were updated in the same change; `--check` re-passes against the flattened `article.tex`. The
tolerant `\amend` unwraps in `sympy/assembly_consistency_verification.py:178` and
`latex_index_notation.py:79` were left in place — harmless, and they keep the parsers robust if
markup is ever reintroduced.

*Still open:* **P7** (12 relocation-provenance date stamps in `.tex` comments → neutral dependency
statements; keep `shared.tex`'s `(verified <date>)`, which is the evidence a `[known-fragility]`
workaround was reproduced rather than guessed) — it was bundled with this task in the plan but is an
independent edit and was not done here.
*Declined:* keeping an annotated copy on a branch — git already holds it, and a second annotated
appendix would recreate the un-gated-twin blind spot `sympy/appendix_twins_verification.py` exists to
prevent.

---

### 1d. Cover the paper's *intermediate* derivations in the verification suite — App. A **DONE**

**Spec / inventory:** [appendix-a-intermediate-coverage-spec.md](appendix-a-intermediate-coverage-spec.md)

**✅ DONE 2026-07-29 for Appendix A** (and for App. C's Step-6 η-optimisation). The gate *parses*
App. A rather than transcribing it: all 79 component displays pass an index census taken after full
distribution plus symbolic differentiation of the printed intermediate against the printed result;
the nine historical defects are re-injected verbatim as negative controls, and five index-balanced
mutations prove the differentiation arm works independently of the census. Suite 498/498 → 608/608.

**Still open:** the same blind spot everywhere else in the paper. The spec's §6 now carries the
inventory from a six-agent survey — App. C's Step 0–9 bound chains, App. D's inf-sup Step 2–4
chains, App. B's symbol chain, the main-text↔appendix duplicated *collections*, and the documents
no script opens at all — each with a difficulty estimate and the specific gate that would close it.
Two live defects found while inventorying (a script anchored to labels that exist in no `.tex`, and
a `0 == 0` check) are fixed and recorded in [`lessons_learned.md`](lessons_learned.md).

## 2. Code–theory consistency

### 2a. TAU-02 — σ inside τ₁ evaluated at the floored `kin.mag_u`, not the physical reaction speed
`src/stabilization/tau.jl` evaluates the τ₁ reaction via `sigma(rxn_law, kin, med, mag_u)` at `kin.mag_u`
(the τ velocity-floor state), while the physical Forchheimer drag in the residual uses the mesh-independent
reaction speed (the `2026-05-21` [`lessons_learned.md`](lessons_learned.md) fix). **Re-check** whether this
σ-in-τ₁ choice is an intentional divergence against [`theory-code-map.md`](theory-code-map.md) §2 before
"fixing" — it may be deliberate. (Identifiers renamed since the audit: `effective_speed`/`reaction_speed` →
the `KinematicState` `kin.mag_u` path. audit Appendix 2, TAU-02.)

### 2b. Latent hardcoded-`ASGS` dispatch in CocquetFormMMS
[`test/extended/CocquetFormMMS/run_test.jl`](../test/extended/CocquetFormMMS/run_test.jl) hardcodes
`"method" => "ASGS"` in the per-cell config dict — the same pattern fixed in the main MMS harness on
2026-05-26 ([`lessons_learned.md`](lessons_learned.md) §5). Not triggered today (that test iterates only
`{ASGS, Galerkin}`) but it would mislabel runs if `OSGS` is added.

### 2c. Verify-first: 2026-07-09 external 3D-audit code observations
An external clean-room audit (its `C_inv` re-derivation is trustworthy and folded into
[`findings.md`](findings.md) §3) also flagged four **unverified** code observations — check each against the
tree before acting:
- **Joint c₁/c₂ scaling in the "c₁ multiplier" hooks** — `smoke3d.jl:235` (`c_1 *= c1_mult; c_2 *= c1_mult`),
  `CocquetFormMMS/run_test.jl:559-561` (`C1_MULT`), `osgs_solver.jl:362-363`. Benign at Re=1, but a
  referee-proof *element-aware c₁* claim needs a **c₁-only** hook. (At Cocquet Re=1e5 the c₂ scaling
  strengthened stabilization — a c₂ experiment may have been read as c₁.)
- **`tau_reg_lim` added to dimensionally-inconsistent quantities** in `tau.jl` (`_tau_ns_inv` [1/time], τ₁
  denom [1/time], τ₂ denom `c₁·α·τ_{1,NS}` [time]) — possibly the root of the marginal OSGS encoding-covariance
  failure (§4f); test a relative/dimensionally-consistent floor before relaxing that threshold.
- **3D harness bypasses the production element-size machinery** — `smoke3d.jl` builds `h_cf` by hand with
  `h_conv='regular_tet'≈1.12a` while `src/geometry.jl`'s production default is `:shortest_edge`; a ~26%
  c₁-equivalent 2D/3D inconsistency that confounds the code-to-paper c₁ comparison (ties to 7e).
- **`TAU_VISC_MULT` parsed per quadrature point** in `_tau_ns_inv` (comment says "read ONCE per assembly") —
  correctness-neutral, hoist it (→ §8 cleanup).

---

## 3. Formulation

### 3a. (Superseded by the RESOLVED verdict) 3D-P2 element-aware `c₁` remedy
The P2-3D catastrophe is RESOLVED: `4k⁴` under-margins high-`C_inv` structured tets, remedy = element-aware
`c₁` (`article.tex` line 910). This is a **formulation-research direction, not a code change and not a
`c1_multiplier` mask.** If pursued: compute per-element `C_inv²` and set `c₁ ≈ 2ξ·C_inv²` per element type
(Kuhn TET ≈ 3.6× the quad). Ties to task 1a (write the note first). See
[`mms/p2-3d.md`](mms/p2-3d.md) §A. Diagnostic hooks: `tau.jl` `TAU_VISC_MULT`, `smoke3d.jl` `h_conv`.

### 3b. (Deferred) CocquetFormMMS — k=2 `c₁` analog probe
The c₁×4 probe was **k=1**, where the viscous 2nd-derivative subscale is identically zero (c₁ acts only via
τ_NS — partial help, not a fix). The faithful test of whether any c₁ mechanism transfers is the **k=2 analog**
(`cocquet_form_mms_vms_k2.json`), where that subscale exists. The `C1_MULT` env-var hook is committed
(default-off, byte-identical). See [`cocquet/cocquet-form-mms-status.md`](cocquet/cocquet-form-mms-status.md) §4.2.

### 3c. FORM-04 (low) — τ rational / Forchheimer `|u|` under-integration
The τ rational and the Forchheimer `|u|` nonlinearity are under-integrated by the polynomial quadrature
rule — `src/formulations/continuous_problem.jl` (quadrature-degree helpers). (audit Appendix 2, FORM-04.)

### 3d. MODE-04 (low) — porosity logistic guard + α is 2D-only
The porosity logistic saturation guard uses hard-coded ±100 thresholds, and α is silently 2D-only —
`src/models/porosity.jl:70-74`. (audit Appendix 2, MODE-04.)

---

## 4. Solver / numerics

Scope: `src/solvers/nonlinear.jl` (safeguarded Newton), `solver_core.jl` (orchestrator),
`osgs_solver.jl` (`solve_osgs_stage!`). Read [`lessons_learned.md`](lessons_learned.md) before editing —
the safeguards are intentional design (CLAUDE.md: *"do not weaken them in pursuit of speed"*).

### 4a. Idea 1 — cross-check "converges in 2 iterations" (verification, likely no change)
Cross-tabulate per (cell, method, N): iterations-to-converge vs observed MMS rate. A tolerance is only too
loose if a cell converges **fast** *and* underperforms `h^{kv+1}`. **Likely conclusion: no change needed**
(fast convergence on a mild cell is correct-by-design). Do the cross-check once a sweep lands so it is not
re-litigated.

### 4b. Idea 2 tier 1 — enable the ASGS stall guard (cheap, reversible, config-only)
The no-progress bail (`"no_progress_stall"`, [nonlinear.jl:698](../src/solvers/nonlinear.jl#L698)) fires only
when `stall_window > 0`, **off in production**. Set `stall_window ≈ 2` + a sensible `stall_min_rel_improvement`
so Newton bails fast into Picard. Adopt only behind a measured A/B: iteration counts **and** final MMS rates
unchanged-or-better. Caveat: don't bail genuine quadratic descent (interacts with 4a).

### 4c. Idea 2 tier 2 — Newton↔Picard ping-pong: remaining A/B
The ping-pong itself has **landed** (`_pingpong_cascade!`, gated `pingpong_enabled`). Remaining: the A/B
measurement (iteration counts + final MMS rates unchanged-or-better). Do not relabel a Picard step as
Exact-Newton (CLAUDE.md invariant).

### 4d. A real saddle-point / MG preconditioner for the OSGS coupled tangent — OPTIONAL upgrade
Downgraded 2026-07-09 (was "REQUIRED for P2-OSGS-3D"): P2-OSGS-3D is solved by the preconditioner-only c₁×4
inflation (`osgs_jfnk_precond_c1_mult`, [`findings.md`](findings.md) §5). _Re-motivated 2026-07-21 (rerun R10):
this is now the ONLY route to fix the 3D-OSGS H¹-pressure ≈1.29 saturation — R10 confirmed that defect is a
genuine, reproducible pressure under-stabilization at `c₁=16k⁴` (paper tables match on-disk; the solution-preserving
`c₁` knob cannot cure it), so a real block/Schur preconditioner or more pressure stabilization is the actual fix._
A real **block/Schur (PCD/LSC/SIMPLE)
or Vanka/MG** preconditioner stays an optional upgrade for guess-independent robustness and to remove the
c₁-tuning. `τ₁` already seeds a discrete pressure-Laplacian in the (2,2) block; `τ~h` ⇒ rediscretize per MG
level. Do **not** pre-build it for 2D (equal-order 2D needs none — Phase-0 verdict). See
[`solver/jfnk-phase0-preconditioner-gate.md`](solver/jfnk-phase0-preconditioner-gate.md) "3D watch item".

### 4e. Anderson acceleration — broader sweep + tuning
Anderson is landed (`osgs_anderson_enabled`, default OFF, bit-identical), cuts staggered outer count
≈1.4–2.2×. Remaining: tune `depth`/`relaxation`/`safety` and sweep across reaction-/convection-dominated
regimes where the linear rate is the bottleneck. Does **not** rescue P2-OSGS-3D (solved by the c₁-inflated
preconditioner). See [`findings.md`](findings.md) §5 (Anderson).

### 4f. 🟠 Both MMS harnesses hardcode `physical_epsilon = 1e-8` in the per-cell config (2026-07-30)

`ManufacturedSolutions/run_test.jl:780` and `CocquetFormMMS/run_test.jl:513` build each cell's config with
a **literal** `"physical_epsilon" => 1e-8`; the swept factor `physical_properties.physical_epsilon = [0.0]`
that every config declares is never read into that dict, so it does not override the literal. So both 2D
campaigns run at a dimensionless `ε̂ = 1e-8`, not `0` — see [`findings.md`](findings.md) §9.5 for the full
argument and why nothing computed is affected (the MMS mass source carries the matching `ε·p_ex`).

**Done:** the recorded provenance is fixed (both harnesses now read `physical_epsilon` and
`numerical_epsilon_coeff` off the assembled formulation and add a `dimensionless_epsilon` attribute), and
both mains now describe `ε` honestly in §7.1, §7.3 and the §3 trim paragraph.

**The defect is broader than one literal — there are three unread swept factors and three writers:**

| declared in the config | read by the harness? | what actually runs |
|---|---|---|
| `physical_properties.physical_epsilon = [0.0]` | **no** | `1e-8` (literal at `run_test.jl:780` / `:513`) |
| `physical_properties.numerical_epsilon_coefficient = [0.0001]` | **no** | `0.0` (never passed to `PaperGeneralFormulation`, whose kwarg defaults to `0.0`) |
| — | — | the `(10⁶,0.05)` corner runs via `probe_stiff_diagnose.jl` `build_cell`, a **fourth** literal `1e-8` |

So the shipped configuration states the opposite of what runs, in both directions, and a re-run *driven
from the config* would not reproduce the paper's `ε`. That is a direct tension with the
no-implicit-defaults hard rule ("Fail loudly on missing input"), not merely untidy.

**Done in this pass (provenance only, no numeric change):** all three writers now record what was
assembled — `run_test.jl` in both harnesses reads `form.physical_epsilon` / `form.numerical_epsilon` and
adds `dimensionless_epsilon`; `merge_corner_results.py` records the corner driver's literal instead of a
false `0.0`.

**Still open — a judgment call:** whether the literals become `0.0` (matching the configs, and restoring
the simpler "`ε = 0`" statement in the paper) or the config factors get wired through (the right shape — a
swept factor the harness ignores is a trap regardless of its value). Either changes every 2D cell's
numbers at the `1e-8` level and so invalidates the committed DBs: it is a **re-run, not an edit**, and
must wait until the submission ladder is frozen. Do not "fix" it silently. Settle together with `6f` (the
production single-run path carries the same fixed `physical_epsilon`), and note that whichever way it
goes, `ε̂ = 1e-8` is *consistent* today — the MMS mass source carries the matching `ε·p_ex`
(`mms_paper.jl:602`), so the current numbers are exact for the problem actually solved.

### 4g. Encoding-invariance test — OSGS-covariance floor (threshold relaxed to 5e-8 — DECISION NEEDED)
`test/quick/encoding_invariance_quick_test.jl`: OSGS `err_u_l2` cross-encoding covariance sits at
reldiff ≈ 1.378e-8 (the other 5 metrics pass ~1e-10). The gate `_INV_RTOL` was **relaxed 1e-8 → 5e-8**
(commit `3e59810`, *"relax scale-covariance tolerance to 5e-8 for Philosophy-A mass envelope"*) so the test
now passes. **Open decision:** is 5e-8 a legitimately re-derived roundoff floor for this cell, or does it mask
a residual OSGS-covariance defect in the staggered map? The repo's no-relax-a-threshold-to-go-green rule
(`.agents/rules/`) says this must be re-derived (or the covariance tightened), not left at a hand-tuned bound.
### 4h. Lower-priority solver audit findings (all LOW)
From the audit Appendix 2 (each verified by an independent code-grounding skeptic):
- **NONL-04**: Anderson `update!` has no zero/near-zero residual guard before the least-squares solve (now
  reachable via `osgs_anderson_enabled`) — `src/solvers/accelerators.jl:53-127`.
- **NONL-06**: Picard line-search reuses Armijo `c1` as a multiplicative residual-reduction factor
  (`1 − c1·α`), a different mathematical role — `src/solvers/nonlinear.jl:434-438`.
- **SOLV-03**: one-way ASGS Picard success uses scalar ℓ∞ `ftol`, not the scale-free `cascade_step_outcome`
  the ping-pong path and Newton stage use — `src/solvers/asgs_solver.jl:144`.
- **SOLV-05**: `discrete_l2_projection` re-solves a fresh `allocate_in_domain` RHS each residual eval;
  `b_vec`/`x_solve` scratch re-allocated — `src/solvers/osgs_solver.jl:59-64`.

---

## 5. Post-processing

### 5a. Flag OSGS-degenerated-to-ASGS results distinctly (audit B.3 / B.6) — PARTIALLY done
A failed OSGS coupled solve leaves the iterate at the ASGS Stage-I boot ([solver_core.jl:511](../src/solvers/solver_core.jl#L511))
and reports **ASGS's error under the OSGS label**. The `osgs_short_circuited_on_entry` recording guard
(landed `efd0372`) already flags the *non-advancing* degeneration. **Remaining:** give such results a distinct
*label* (e.g. `method="OSGS(degenerated→ASGS)"`), add a diagnostic that red-flags identical ASGS/OSGS error
tuples, and surface per-level mesh quality (min dihedral / radius-ratio). (§5b — plotter now gates on
`success` via `_level_success`, `7d670d6` — is DONE.)

### 5b. CONV-04 (low) — sub-optimal-rate budget uses a bare power of `h`
The sub-optimal-rate budget uses a bare power of h with no reference-error normalization (scale-dependent) —
`src/solvers/mms_verification.jl:130-131`. (audit Appendix 2, CONV-04.)

### 5c. Per-attempt terminal-status supplement for the omitted cells (audit T6, 2026-07-30)

§7 already discloses *which* cell is omitted (`(Re,α₀)=(10⁶,0.05)`), *which* meshes recover it
(ℙ₁ from N≈512, ℚ₂ from N=160), the *initialization* (exact solution), the *exit* (Newton and Picard
both stall at a finite residual) and the non-claim (nonexistence is not established) — so the audit's
"a reader cannot reconstruct" is mostly false of the manuscript. What is genuinely missing is a
**per-attempt terminal-status record**: no residual values, no per-mesh pass/fail row, and the
failing coarse-mesh corner attempts are in no official DB because the config skips them.

Cheap, no re-run: the data is on disk. Extend
`test/extended/ManufacturedSolutions/make_results_tables.py` (reuse `_mesh_success()`,
`load_corner()`, `load_iters_traces()`, which already read `overall_verification_success`,
`eval_residuals`, `eval_iters` and the sidecars' `final_residual`/`success`) to emit **one CSV row
per `(Re, Da, α₀, element, method, N)`** with initialization, terminal residual and gate outcome.
Ship it with the release bundle (§6g) rather than promising a supplement in the paper — the paper
sentence goes in only once the artifact exists.

---

## 6. Input / output & provenance

### 6a. Efficiency Idea 6 — enrich saved data for self-contained OSGS diagnosis
Store per mesh, in the HDF5 group attrs + JSON trace sidecar: `tau1`/`tau2` (min/max/representative over Ω),
`sigma`, `|u|_max`, `encoding_strategy`, and the `L`/`U` scale factors; optionally surface `tau1` and the
σ-share-of-`(1/tau1)` on the trajectory plot. The solver already has these at each mesh — **only plumbing**.
Payoff: OSGS rate diagnosis becomes readable from saved data without re-running.

### 6b. Schema `method` enum vs loader mismatch (confirmed code bug)
`porous_ns.schema.json` allows `method ∈ {ASGS, OSGS, VMS, Galerkin}` (lines ~195-199), but
[config.jl:148](../src/config.jl#L148) asserts `method ∈ {ASGS, OSGS}`. A `VMS`/`Galerkin` config passes
schema validation then dies at the assertion. **Fix:** trim the schema enum to `{ASGS, OSGS}` (or implement
the others).

### 6c. Config-strictness / schema `required` gaps (audit A.4 / DRIV-03)
The schema declares `required` for exactly one object (`linear_solver`); add `required` arrays mirroring the
`@kwdef` structs. `PhysicalProperties` docstring at [config.jl:23](../src/config.jl#L23) still mislabels the
pressure-penalty ε as *"porosity ε (>0)"* (it may be 0). Schema `additionalProperties: true` on
`physical_properties` (~line 111) admits unknown keys against the strict-config intent (soft-guarded by a
loader `@warn`, [config.jl:168](../src/config.jl#L168)). Also **CONV-05**: MMS-verifier numerical params
(`tau_err`, `eps_*`, `max_extra_cycles`, `require_consecutive_passes`, `rate_check_factor`) are hard-coded —
`test/extended/CocquetFormMMS/run_test.jl:401-412`.

### 6d. DRIV-05 (low) — production `export_results` writes no provenance
`export_results` writes no provenance (Re/Da/α₀/params) into/alongside VTK — `src/io.jl:28-51`. The MMS/Cocquet
harnesses already embed provenance; production does not.

### 6e. Restore/commit the exact 3D driver that produced the committed results (audit B.4)
`convergence3d_results_frontal_c1x1_20260623.json` records `mesh_algorithm = "gmsh_Frontal_alg4_independent_remesh"`
and a 6-level P1 ladder, but **no function in `smoke3d.jl` now produces that string/ladder** — violating the
parameters→results traceability rule. Also `build_config` sets `eps_val = 1e-8` but `solve_one` builds with
`eps_phys = 0.0` (the stored config value is dead). Restore/commit the exact driver (or re-run with a committed
named function + archive the config snapshot), and make `eps_val`/`eps_phys` agree (or document why). Overlaps 7d.

### 6f. Single-run path uses a fixed `physical_epsilon` (minor)
`run_simulation` injects a fixed dimensional `physical_epsilon` rather than the per-encoding covariant value
the MMS harness derives. Harmless for a single run (no encoding sweep) but inconsistent with the harness
([`lessons_learned.md`](lessons_learned.md) §4, 2026-06-02).

### 6g. Submission release bundle (audit C12/P2/S6/T9, 2026-07-30)

The audit's C12 is **five-sixths a packing artifact** — `article.tex`,
`osgs_appendix_commented.tex`, `figures/bump_plateau.pdf`, the 25 Coq `.vo` objects, `src/`, 39
tracked configs and 12 tracked meshes all exist; they were absent from `repomix-theory-proofs-under-1m.md`
only by its own exclusion rules. Its build-failure claim is **refuted** (both mains build clean here:
v2 116 pp / 976 newlabels, v1 80 pp / 776; the auditor hit the known `ntheorem`×`cleveref≥0.21` clash
inside the bundled 2019 `siamart190516.cls`, a TeX-distribution portability issue). What is real:

1. **Pin the build environment.** The class/`cleveref` clash means "builds clean" is currently a
   statement about *this* TeX Live. Record TeX Live year + `cleveref`/`ntheorem` versions (and
   Coq/Rocq, Python+SymPy, Julia) in a lockfile or container.
2. **One build command, one gate command**, with expected exit codes and expected summary counts —
   this is what replaces the prose "verified clean". Keep the counts regenerated from the live build,
   never hand-edited: `grep -a "Output written" "latex compilation/<base>/<base>.log"` and
   `grep -ac newlabel "latex compilation/<base>/<base>.aux"`.
3. **Archived logs with hashes**, and a source-coupled check that the file linted is the file built
   (the App-D twin asymmetry is already gated by `sympy/appendix_twins_verification.py`; the same
   idea should cover the rest).
4. **The archival DOI** for the *Code and data availability* section now present in both mains. The
   URL half is done — the section names `https://github.com/GuillermoCasas/GridapPorousNS`, verified
   public 2026-07-30. Remaining: deposit the exact snapshot (Zenodo or equivalent), add its DOI and the
   commit hash to the statement, and **push** the revision the paper describes (this pass is
   uncommitted). Note the result DBs are gitignored by design, so the deposit should either include
   them or rely on the regeneration recipe of items 2-3 above.
5. Ship the per-attempt status CSV of §5c and the raw result DBs (or their regeneration recipe)
   alongside.

Do **not** describe a Repomix export with deliberate exclusions as self-contained — that framing is
what produced four of this audit's items.

---

## 7. Tests & validation sweeps

### 7a. CocquetFormMMS — k=2 corner to N=160 (cheap, firms a rate)
The α=0.1 × Re=1e5 corner is RESOLVED for **k=1** (FE-optimal above the fold: H¹u ≈ 1.07/1.10, L²u ≈ 3.0 at
N=[160,320]). The **k=2 corner already has clean roots at N=40 & N=80** (clears the fold ~2× earlier); extend
to N=160 to firm the rate. Config: extend the `cocquet_form_mms_vms_k2.json` mesh ladder. See
[`cocquet/cocquet-form-mms-status.md`](cocquet/cocquet-form-mms-status.md) §4.1.

### 7b. CocquetFormMMS — optional k=1 corner to N=640 (3-point slope)
The k=1 corner rate rests on a **2-point** slope (N=160→320). N=640 gives a 3-point slope. To make it
*official*: extend the `data/cocquet_form_mms_vms.json` ladder and re-run through the harness, archiving the
prior official DB into `previous_results/` first (per the official-results-path rule). Not needed for the
deliverable.

### 7c. 3D full OSGS structured-Kuhn sweep — honest mass-gate + P2 finer-mesh confirmation
Route B + JFNK make the OSGS-P1-3D success flag *honest* (`ftol_reached` on both ε_M and ε_C). Run the full
structured-Kuhn OSGS sweep to confirm honest convergence at every cell — and, with the c₁-inflated JFNK
preconditioner (4d), to **close the finer-mesh confirmation of OSGS-P2** (verified only on (12,12,3) so far).
Watch OSGS-P2 `success`/`eps_used` and the L²p column across (12,12,3)→(16,16,4)→(20,20,5): is the paper-c₁ P2
pressure defect (L²p=0.045) *uniform* or does it *converge*? Run:
`julia --project=. test/extended/ManufacturedSolutions3D/smoke3d.jl sweep_structured 3`. See
[`mms/p2-3d.md`](mms/p2-3d.md) §C and [`open-questions.md`](open-questions.md) §3.

### 7d. 3D structured-Kuhn control — re-baseline after the ILU-GMRES honesty fix (C.1)
C.1 (`GMRESNotConvergedError`, landed 2026-06-26) changes only the 3D fine-mesh `ILU_GMRES` path (2D uses
`LUSolver`, inert). It is **expected to flip which fine-mesh OSGS solves report success** — that is the point.
Re-run the 3D structured-Kuhn control before/after to record the flip.

### 7e. CocquetTubeTest — remaining unified variants (LOW PRIORITY; behavior-preservation already verified)
The 2026-07-08 refactor unified the nine sibling Cocquet tube tests into one config-driven harness.
Behavior-preservation is **verified**: `structured` reproduces the baseline byte-for-byte (2026-07-20,
after the interpolation-floor diagnostic was added — FE errors within ≤1.7e-4 rel.), and `unstructured_frontal`
(the new best-quality variant) ran the full 3-way sweep. The remaining variants (`alpha_one`, `deviatoric`,
`linear_reaction`, `all_dirichlet`, `modified_corner`, `unstructured_gmsh`, `freefem_meshes`,
`freefem_divisions`, `literal_picard`) are the **historical S5 diagnostic siblings whose verdicts are already
settled** in [`cocquet/investigation-synthesis.md`](cocquet/investigation-synthesis.md) — re-running them is
optional provenance bookkeeping, not a blocker. Command if ever needed: `run_convergence.jl data/<name>/…`.

### 7f. Audit-response reruns — grids done; paper integration partial (2026-07-22) 🟠 mostly done
Three audit-driven reruns were run and analyzed; verdicts recorded in [`findings.md`](findings.md) §8. Paper
integration status:
- **R5 — stabilized Taylor–Hood P2/P1 control** (audit D05): **✅ DONE + ported into `article_v2.tex`** as the
  `P2/P1 ASGS` rows of `tab:CocquetMMS` (commit `384362f`), **and into `article.tex` v1 on 2026-07-30** —
  v1 always carried the same four rows, so the earlier "not yet ported / v1 has no such rows" note was wrong
  (see the checklist correction and `lessons_learned.md` 2026-07-30 (d)). Verdict: it converges at Re=10⁵ where
  the unstabilized-TH velocity stagnates, isolating the high-Re gain to the **stabilization** rather than the
  space pair. The viscous-regime penalty is **not** ~10×: that figure compared `N=160` against `N=320`. On the
  completed ladder (§7h, ✅ done) the like-for-like factor at `N=320` is **1.2×** (α₀=0.5) and **1.4×**
  (α₀=0.1); both mains print it and the mesh caveats are gone.
- **R6 — genuinely-3D MMS** (audit N19): grid ran (`results/k*/TET/genuine3d/`), verdict optimal rates for both
  orders + OSGS pressure H¹ converges (slope 2.0, unlike the extruded field's 1.29 plateau).
  **✅ DONE + ported:** `theory/paper/genuine3d_table.tex` is `\input` after `tab:3D` in
  `article_v2.tex` (v2 only, as with the R5 control), with a paragraph stating what the data shows.
  The author decision came out *add-alongside*: the genuine-3D `H¹` pressure errors stay `O(1)`
  (1.22–2.97) exactly as on the extruded field, so they **sharpen** the adverse 3D finding rather than
  dissolving it (`open-questions.md` §4, `pre-submission-checklist.md` RESOLVED T7).
- **R2 — α-interpolation ablation** (audit I07): **✅ analyzed.** P1-α is benign for P1 elements (~1.1×) but caps
  P2 convergence (48–73× worse) — FE interpolation of α preserves convergence only when interpolated at (≥) the
  velocity order. Refines the conclusion's I07 claim (currently softened to future work).
- Dropped (analysis, no run): **R10** (3D-OSGS pressure = genuine under-stab, see 4d/7c), **R1** (fold
  continuation — wording softened), **R3** (c₁-eigenvalue study — see `open-questions.md` §3), **R4** (pointwise
  vs elementwise τ — N09 text fallback stands; a moderate code change if ever pursued).

### 7g. Constrained-projection OSGS — measure the P2 MMS rate (settles the L²/energy-norm split)
The companion note [`theory/projection_space_note/`](../theory/projection_space_note/projection_space_note.pdf)
proves the OSGS **constrained** residual projection (onto `X_{h0}` instead of the implemented unconstrained
`V_free`/`Q_free`) is degree-dependent in the **energy** norm: optimal for k=1, but a boundary-strip
consistency defect `η₀ = Θ(h^{3/2})` provably degrades the rate to 3/2 for k≥2 (two-sided; under `a=0` on
`Γ_D` and the viscous regime `τ₂=Θ(1)`). The remaining open strand is the **plain-L²/MMS gate** (the norm the
paper footnote's `O(h^{k+1})` and the ε_M/ε_C indicators measure), where an Aubin–Nitsche duality *may*
recover part of the loss (the defect enters quadratically). **Decisive check:** flip the OSGS projection
target from the unconstrained spaces to the Dirichlet-constrained `X_{h0}` and re-run the standard **P2** MMS
sweep — a rate collapse to ≈3/2 confirms the theorem reaches the L² gate; an optimal `h³` (k=2, L²) would show
the energy-norm defect stays out of the L² gate. Touch point: the `V_proj`/`Q_proj` selection in
[`src/solvers/osgs_solver.jl`](../src/solvers/osgs_solver.jl) (currently `V_free`/`Q_free` — see
[`theory-code-map.md`](theory-code-map.md) §2). This is a **debug/A-B** run — route output to
`results/debug_results/`, do not disturb the official DBs. Derivation:
`theory/projection_space_note/projection_space_note.pdf` §6 (the degree-dependent theorem); Codina (2008)
Remark 1.

### 7h. ✅ DONE (2026-07-31) — stabilized-Taylor--Hood control re-run to N=320 through the official path

The `P₂/P₁ ASGS` rows of `tab:CocquetMMS` were not on the published ladder: three cells stopped at
`N=160` and the `(10⁵,0.1)` row came from the forked side-DB `results/debug_results/cocquet_stabth_corner.h5`,
which `.agents/rules/official-results-path.md` forbids for published numbers. Mixing `N=160` against the other
rows' `N=320` also manufactured a false headline ("about an order of magnitude less accurate").

**Resolved by re-running the unmodified official config to its full declared ladder** — no fork, no filter, no
merge. Two sessions (7 h, then a ~2.5 h resume through the harness's own `[RESUME]` path). All four cells now
hold `[10,20,40,80,160,320]`.

Landed:
- **All eight `P₂/P₁ ASGS` rows requoted** in *both* mains from the official DB, using the **two-finest-mesh**
  slope `log2(e[-2]/e[-1])` — *not* the HDF5 `rate_*` attribute, which is a whole-ladder fit differing by up to
  0.21 and would have silently changed every slope while looking faithful. Six of eight rows moved; the
  corner's two were already correct.
- **Both caption caveats and the §7.3 ladder caveat deleted** — the ladder is now common to all four methods.
- **§7.3's convection claim requoted** at `N=320` (`2.36`/`1.32` replacing `2.41`/`1.26`), keeping the previous
  doubling's values in text as evidence the shortfall is stable rather than pre-asymptotic.
- **§7.3's viscous factor** is now like-for-like at `N=320`: `1.2×` (α₀=0.5) to `1.4×` (α₀=0.1), replacing the
  `N=160` common-mesh workaround.
- **Corner artefacts retired without deletion.** The official run reproduces the side-DB's `N=320` values
  **exactly** (`0.00e+00` relative difference in all four norms) — the side-DB's numbers were correct; only
  their provenance was wrong. Per `.agents/rules/reproducible-results.md` the forked config is kept as the
  record of a real run and carries an additive `_superseded` note; no parameter was touched. Nothing in the
  repo reads either artefact.
- `erase_past_results` restored to `true` on the official config.

Verified: both mains build clean (article 81 pp / 778 labels, article_v2 118 pp / 980 labels, 0 undefined,
0 multiply-defined, 0 overfull > 1 pt); SymPy 636/636.

## 8. Cleanups

### 8a. Retire dead OSGS config (post coupled-only leaning)
The 2026-06-08 coupled-only leaning left these keys **ignored by the coupled path** but still in
schema/`StabilizationConfig`/configs: `osgs_projection_coupling`, `osgs_freeze_after_k`, `osgs_stopping_mode`,
`osgs_state_drift_scale`, `osgs_projection_tolerance`, `osgs_warmup_*`, the ping-pong knobs, `ablation_mode`,
and the inert off-switches. Retire them. See
[`archive/coupled-only-leaning-and-jfnk-plan.md`](archive/coupled-only-leaning-and-jfnk-plan.md) §3.

### 8b. `_inv_centered.json` latent fragility
`test/quick/encoding_invariance_quick_test.jl` reads a config it must generate first — fine today, but a stale
leftover can confuse a clean checkout.

### 8c. Low-priority audit cleanups
- **CONV-02** (low): inline `1e-2` √d self-check margin — `src/solvers/convergence_criterion.jl:267` (now on the
  `eps_C_strong`/`div_ratio` diagnostic path, not the gate).
- **PROJ-02** (low): `ProjectResidualWithoutPressurePenalty` is never instantiated in production — dead policy
  reachable only from tests — `src/stabilization/projection.jl:34,121-143`.

### 8d. 🟡 Eight note `latexmkrc` files still use the broken literal-`%B` `$aux_dir`

`theory/<note>/latexmkrc` in **eight** directories still sets `$aux_dir = 'latex compilation/%B'`.
TeX Live 2023's latexmk 4.79 does **not** expand `%B` there, so every build routes into a directory
literally named `%B` while a months-old log sits in the correctly-named sibling. That is not cosmetic:
`document_hygiene_verification.py` globs for `latex compilation/*/<base>.log`, and `%B` sorts *before*
any letter, so the old `sorted(...)[0]` read the **stale** log. It hid a 344.92 pt overfull table
(~12 cm past the margin) in `centered_encoding.tex` for the whole of 2026-07-30.

Fixed in this pass: `centered_encoding/` and `cocquet/` (migrated to the `@ARGV` form, stale dirs
deleted, both added to the gate's `DOCS`), and the gate now selects the **newest** log and fails a new
rule `H8` when more than one aux directory holds a `<base>.log`.

Remaining (mechanical, ~5 min each — copy `theory/paper/latexmkrc`'s `@ARGV` block, set
`@default_files`, drop the `./siam//` paths, `rm -rf 'latex compilation/%B'`, rebuild):
`continuity appendix/`, `osgs_algorithm/`, `osgs_reaction_note/`, `pressure_recentering_note/`,
`projection_space_note/`, `scale_free_gate_note/`, `tau_saturation_note/`,
`velocity_floor_regularization/`. Each should then be added to `DOCS` with its current overfull debt —
none of them is gated today, so each may be carrying the same class of defect unseen.

---

## Superseded / done (do not re-open)

- **3D MMS test config-driven + official 3D-MMS extended test** — **✅ DONE 2026-07-09.** Oracle unified into
  the shared dimension-generic [`src/problems/mms_paper.jl`](../src/problems/mms_paper.jl); study params lifted
  into [`data/smoke3d_p1.json`](../test/extended/ManufacturedSolutions3D/data/smoke3d_p1.json);
  official guard [`test/extended/mms3d_config_smoke_extended_test.jl`](../test/extended/mms3d_config_smoke_extended_test.jl)
  (14/14 GREEN, ~24.5 min).
- **Document the 3D P1-ASGS L²-order deficiency** — recorded as a method property in [`findings.md`](findings.md)
  §3 (OSGS is the load-bearing 3D method); 3D ASGS optimality remains a formulation research question, not c₁ tuning.
- **OSGS-3D-P2 "good solution, ok=false" blocker** — **RESOLVED 2026-07-09** (preconditioner-only c₁×4
  inflation `osgs_jfnk_precond_c1_mult`). See [`findings.md`](findings.md) §5, [`mms/p2-3d.md`](mms/p2-3d.md) §C.
- **Efficiency Idea 4** (short-circuit OSGS plateau-verification) — **SUPERSEDED** by the coupled-only leaning.
- **Efficiency Idea 5** (`freeze_after_k` warm-up-then-freeze) — **REVERTED**; a coupling-equivalence oracle
  proved it diverges in the reaction corner. See [`archive/coupled-only-leaning-and-jfnk-plan.md`](archive/coupled-only-leaning-and-jfnk-plan.md) §2.
- **JFNK for the OSGS coupled solve** — Phase-0 gate PASSED, Phase-1 **LANDED** (`osgs_jfnk_enabled`).
- **CocquetFormMMS α=0.1×Re=1e5 k=1 corner** — **DONE 2026-07-07** (FE-optimal above the fold, N=[160,320]).
- **2D k1 & k2 QUAD sweeps** — **DONE 2026-07-03** under Route-B, behavior-preserving.
- **`cfg.phys.f_x`/`f_y` crash; `base_config.json` missing field (`eps_val`→`physical_epsilon`); dead
  `_resolve_solution_scale_per_field` helper** — all **RESOLVED**; see [`findings.md`](findings.md) §7.
- Many audit findings (A.1/A.2/A.3, C.1–C.5, all of Part D, F1–F4) — **RESOLVED** (provenance: the landing
  commits; [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) retains the faithful-transcription
  verdict + results-forensics after its resolved-ledger was trimmed 2026-07-11).
