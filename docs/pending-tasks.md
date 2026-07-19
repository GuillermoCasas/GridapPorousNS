# Pending tasks — backlog (by block)

**Purpose.** The single living backlog for this solver. Every actionable next step, grouped by the area
of work it touches, each pointing at the evidence/doc so it can be picked up cold. This file is the
*only* place open code-correctness issues live now (the former `known_issues.md` was folded in here on
2026-07-10; its resolved items moved to [`findings.md`](findings.md)).

**Blocks**

1. [Theory](#1-theory) — derivations, notes to write, paper-math questions.
2. [Code–theory consistency](#2-codetheory-consistency) — where code and paper/algorithm must be kept aligned.
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
Mechanical-but-author-gated: merge `centered_encoding.tex` into `article.tex` (self-describes "not yet
merged"; strip standalone preamble + `\end{document}`, drop `\Reyn`/`\Damk`/`\code` `\newcommand`s); decide
`supplement.tex` (SIAM boilerplate — replace or drop the `\externaldocument{supplement}` line); add the
results-section figures (`article.tex` line 1436, currently tables only). (The "Kratos Multiphysics"
implementation claim is **✅ resolved 2026-07-19** — the paper now reads Gridap, 0 Kratos mentions.) Full
list: [`open-questions.md`](open-questions.md) §4.

---

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
inflation (`osgs_jfnk_precond_c1_mult`, [`findings.md`](findings.md) §5). A real **block/Schur (PCD/LSC/SIMPLE)
or Vanka/MG** preconditioner stays an optional upgrade for guess-independent robustness and to remove the
c₁-tuning. `τ₁` already seeds a discrete pressure-Laplacian in the (2,2) block; `τ~h` ⇒ rediscretize per MG
level. Do **not** pre-build it for 2D (equal-order 2D needs none — Phase-0 verdict). See
[`solver/jfnk-phase0-preconditioner-gate.md`](solver/jfnk-phase0-preconditioner-gate.md) "3D watch item".

### 4e. Anderson acceleration — broader sweep + tuning
Anderson is landed (`osgs_anderson_enabled`, default OFF, bit-identical), cuts staggered outer count
≈1.4–2.2×. Remaining: tune `depth`/`relaxation`/`safety` and sweep across reaction-/convection-dominated
regimes where the linear rate is the bottleneck. Does **not** rescue P2-OSGS-3D (solved by the c₁-inflated
preconditioner). See [`findings.md`](findings.md) §5 (Anderson).

### 4f. Encoding-invariance test — OSGS-covariance floor (threshold relaxed to 5e-8 — DECISION NEEDED)
`test/quick/encoding_invariance_quick_test.jl`: OSGS `err_u_l2` cross-encoding covariance sits at
reldiff ≈ 1.378e-8 (the other 5 metrics pass ~1e-10). The gate `_INV_RTOL` was **relaxed 1e-8 → 5e-8**
(commit `3e59810`, *"relax scale-covariance tolerance to 5e-8 for Philosophy-A mass envelope"*) so the test
now passes. **Open decision:** is 5e-8 a legitimately re-derived roundoff floor for this cell, or does it mask
a residual OSGS-covariance defect in the staggered map? The repo's no-relax-a-threshold-to-go-green rule
(`.agents/rules/`) says this must be re-derived (or the covariance tightened), not left at a hand-tuned bound.
*(Tension surfaced 2026-07-11 during the docs audit — resolve or ratify the 5e-8 floor.)*

### 4g. Lower-priority solver audit findings (all LOW)
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

### 7e. CocquetTubeTest — run the remaining unified variants end-to-end (relocated 2026-07-11)
The 2026-07-08 refactor unified the nine sibling Cocquet tube tests into one config-driven harness, but only
the `structured` variant has been run end-to-end (it reproduces the baseline: L²u 3.01e-4 / L²p 2.31e-5). The
other variants — `alpha_one`, `deviatoric`, `linear_reaction`, `all_dirichlet`, `modified_corner`,
`unstructured_gmsh`, `freefem_meshes`, `freefem_divisions`, `literal_picard` — still need a full run to
confirm behavior-preservation (each: `julia --project=. test/extended/CocquetTubeTest/run_convergence.jl data/<name>/…`).
See [`cocquet/cocquet-tube-test-unification-status.md`](cocquet/cocquet-tube-test-unification-status.md).

---

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
