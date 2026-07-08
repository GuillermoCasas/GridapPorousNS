# Pending tasks — backlog

**Purpose.** The single living backlog for this solver: concrete, actionable next steps distilled from the
"next steps / follow-up / open" sections scattered across the investigation docs. Each task points at the
relevant file or doc so it can be picked up cold. Grouped by kind:

1. [Sweeps to run](#1-sweeps-to-run) — measurements that close an open question.
2. [Tests to promote / add](#2-tests-to-promote--add) — guard rails and official-test gaps.
3. [Solver efficiency ideas](#3-solver-efficiency-ideas) — Newton/Picard scheduling, gates, preconditioning.
4. [Open code-correctness issues to fix](#4-open-code-correctness-issues-to-fix) — from `known_issues.md` + audit appendix.
5. [Cleanups](#5-cleanups) — dead config, provenance, hygiene.

Nothing here is a *blocker* for the headline results (2D k1/k2 sweeps optimal; OSGS-P1-3D solved; the
P2-3D verdict is RESOLVED — `4k⁴` under-margined for high-`C_inv` structured tets, see
[`mms/3d-p2-instability-investigation.md`](mms/3d-p2-instability-investigation.md)). These are refinements,
completeness measurements, and hygiene.

---

## 1. Sweeps to run

### 1a. CocquetFormMMS — k=2 corner to N=160 (cheap, firms a rate)
The α=0.1 × Re=1e5 corner is RESOLVED for **k=1** (FE-optimal above the fold: H¹u ≈ 1.07/1.10, L²u ≈ 3.0
at N=[160,320], see [`cocquet/cocquet-form-mms-status.md`](cocquet/cocquet-form-mms-status.md) §4.1). The
**k=2 corner already has clean roots at N=40 & N=80** (it clears the fold ~2× earlier); extend it to N=160
to firm the rate. Config: extend the designed `test/extended/CocquetFormMMS/data/cocquet_form_mms_vms_k2.json`
mesh ladder. (§5, first bullet.)

### 1b. CocquetFormMMS — optional k=1 corner to N=640 (3-point slope)
The k=1 corner rate rests on a **2-point** slope (N=160→320). Optional N=640 gives a 3-point slope for
extra confidence. To make it *official*, extend the designed `data/cocquet_form_mms_vms.json` ladder
(`convergence_partitions` → N=640) and re-run through the harness, archiving the prior official DB into
`previous_results/` first (§5). Not needed for the deliverable — the cold exact-guess already reached N=320.

### 1c. 3D full OSGS structured-Kuhn sweep — honest mass-gate confirmation (measurement follow-up)
Route B + JFNK make the OSGS-P1-3D success flag *honest* (`ftol_reached` on both ε_M and ε_C, not a
budget-exhausted soft stall). The full structured-Kuhn OSGS sweep is the remaining **measurement** to
confirm honest convergence at every cell (first OSGS-P1 cell verified: `ε_C 0.97→6e-9`, L²p dropped ~0.83%
— the soft-stall under-convergence tail now drained). See [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md)
§F3/§F4 remaining-follow-up bullets and §B.1 Route-B update. Harness: `test/extended/ManufacturedSolutions3D/smoke3d.jl`.

### 1d. 3D structured-Kuhn control — re-baseline after the ILU-GMRES honesty fix (C.1)
C.1 (`GMRESNotConvergedError`, landed 2026-06-26) changes only the 3D fine-mesh `ILU_GMRES` path (2D uses
`LUSolver`, inert). It is **expected to flip which fine-mesh OSGS solves report success** — that is the
point. Re-run the 3D structured-Kuhn control before/after to record the flip.
See [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §F3 remaining-follow-up.

### 1e. (Deferred / theory-completeness) CocquetFormMMS low-α fold mechanism — clean σ̃_α isolation
The exact fold mechanism is **OPEN**; the leading σ̃_α / reaction-in-stabilization hypothesis is
paper-grounded but **not confirmed** — the direct `STRIP_REACTION_FROM_STAB` A/B was confounded by τ₁
entanglement (stripping σ also enlarges τ₁). The clean isolation: **strip σ from the adjoint/operator
`𝓛U`, `𝓛*V` only, holding τ₁ physical**. This is *theory-completeness*, not a prerequisite for the
deliverable (§4.1 convergence above the fold does not depend on it). The `STRIP_REACTION_FROM_STAB` gate was
reverted (not paper-faithful); §4.3 documents the method fully enough to re-derive it. See
[`cocquet/cocquet-form-mms-status.md`](cocquet/cocquet-form-mms-status.md) §4.3 / §5.
- Alternative sub-tasks if pursued: (a) finish the **OSGS** trim-vs-full A/B at low α (the valid version —
  the trim is OSGS-only — was killed before completing); (b) for **ASGS**, a code change to strip σu from
  the stabilization residual (the projection trim won't do it for ASGS).

### 1f. (Deferred) CocquetFormMMS — k=2 c₁ analog probe
The c₁×4 probe was **k=1**, where the viscous 2nd-derivative subscale is identically zero, so c₁ acts only
through τ_NS (partial help, not a fix). The faithful test of whether any c₁ mechanism transfers is the
**k=2 analog** (`cocquet_form_mms_vms_k2.json`), where that subscale exists. The `C1_MULT` env-var hook is
committed (default-off, byte-identical) so the run is ready. Deferred. See
[`cocquet/cocquet-form-mms-status.md`](cocquet/cocquet-form-mms-status.md) §4.2 closing bullet.

### 1g. (Superseded by the RESOLVED verdict) 3D-P2 element-aware c₁ remedy
The P2-3D catastrophe is RESOLVED: `4k⁴` is under-margined for high-`C_inv` structured tets, and the
theory-sanctioned remedy is an **element-aware c₁** (article.tex line 910). This is a formulation-research
direction, **not** a code change and **not** a `c1_multiplier` mask. If pursued: compute the per-element
`C_inv²` and set `c₁ ≈ 2ξ·C_inv²` per element type (Kuhn TET needs ≈3.6× the quad). See
[`mms/3d-p2-instability-investigation.md`](mms/3d-p2-instability-investigation.md) §3.2 and its
[dossier](mms/3d-p2-coercivity-resolution-dossier.md) for the full argument, caveats, and what would
overturn the verdict. Diagnostic hooks live behind `tau.jl` `TAU_VISC_MULT`, `smoke3d.jl` `h_conv`.

---

## 2. Tests to promote / add

### 2a. Make the 3D MMS test config-driven + add an official 3D-MMS extended test (audit F6)
`test/extended/ManufacturedSolutions3D/smoke3d.jl` is a hand-edited driver with hard-coded study params
(`RE`/`DA`/`ALPHA0`/`R1`/`R2`/`L`, `mesh_sequence`, `c1_mult`, `kv`/`method` loops) — the 3D analogue of
the 2D `run_test.jl` but with **no committed config JSON and no automated guard** (the source of the
deleted one-off scrap). Recipe (from [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §F6):
1. Lift the study knobs into a JSON config (mirror `ManufacturedSolutions/data/*.json` + the 3D-specific
   `mesh_sequence` / `c1_mult` / slab z-extent).
2. Add a `config <path>` entry mode to `smoke3d.jl` driving the existing `solve_one`/`run_sweep*` machinery.
3. Commit a lean smoke config (`data/smoke3d_p1.json`: §5.2, structured-Kuhn, 2–3 coarse levels, k=1,
   ASGS+OSGS, paper c₁).
4. Wire a lean **official** extended test into `test/run_extended_tests.jl` that runs it and asserts the
   solve succeeds and the P1 rate is within tolerance of optimal.

### 2b. Fix the 3D plotter/rate computation to gate on `success` (audit B.2)
[`plot_convergence3d.py`](../test/extended/ManufacturedSolutions3D/plot_convergence3d.py) `_plot_by_kv` /
`_seg_slope` compute slopes across **every** consecutive pair with **no `success` filter**, and the
top-level error arrays drop the per-level `success` flag (kept only in `levels[]`). This plots failed solves
as data and annotates their wild slopes as convergence rates — the origin of the overstated "P2 diverges"
headline. Action: exclude / visibly flag `success=False` levels; never compute a rate across a non-converged
solve. See [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §B.2.

### 2c. Flag OSGS-degenerated-to-ASGS results distinctly (audit B.3 / B.6)
A failed OSGS coupled solve leaves the iterate at the ASGS Stage-I boot state
([solver_core.jl:511](../src/solvers/solver_core.jl#L511)) and reports **ASGS's error under the OSGS label**
— the byte-identical ASGS/OSGS error tuples at shared-failed levels. Action: when the OSGS coupled solve
does not reach its own convergence verdict, mark the result distinctly (e.g. `method="OSGS(degenerated→ASGS)"`
or a flag) so a sweep cannot record an ASGS error in an OSGS column; add a diagnostic that flags identical
ASGS/OSGS error tuples as a red flag, and surface per-level mesh quality (min dihedral / radius-ratio). See
[`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §B.3 / §B.6.

### 2d. Document the 3D P1-ASGS L²-order deficiency honestly (audit B.1)
3D P1-**ASGS** is genuinely sub-optimal (L²u ≈ 1.2, not 2) even on a perfect uniform Kuhn mesh — a
method-intrinsic Aubin–Nitsche (L²-extra-order) defect, not mesh quality; **OSGS recovers it** (L²u ≈ 2.0
on the *same* mesh; ratio-to-interpolant pins at ~0.9 for OSGS, drifts 1.47→3.03 for ASGS). Actions:
(1) document it as a method property, not a mesh defect; (2) treat OSGS as the load-bearing 3D method (the
code already leans that way); (3) 3D ASGS optimality, if wanted, is a genuine formulation research question
(the orthogonal-projection consistency term) — a paper footnote, not a c₁ tuning. See
[`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §B.1.

### 2e. Efficiency idea 6 — enrich saved data for self-contained OSGS diagnosis
Store, per mesh, in the HDF5 group attrs + JSON trace sidecar: `tau1`/`tau2` (min/max/representative over Ω),
`sigma`, `|u|_max`, `encoding_strategy`, and the `L`/`U` scale factors; optionally surface `tau1` and the
σ-share-of-`(1/tau1)` on the trajectory plot. The solver already has all of these at each mesh — it is
**only plumbing** into the `run_test.jl` HDF5 write and the trace sidecar. Payoff: an OSGS rate diagnosis
(e.g. the high-Da coercivity-gap cases) becomes readable from saved data without re-running the solver. See
[`solver/efficiency-ideas.md`](solver/efficiency-ideas.md) Idea 6.

---

## 3. Solver efficiency ideas

Scope: `src/solvers/nonlinear.jl` (safeguarded Newton), `src/solvers/solver_core.jl` (orchestrator),
`src/solvers/osgs_solver.jl` (`solve_osgs_stage!`). Read [`lessons_learned.md`](lessons_learned.md) before
editing — the safeguards are intentional design (CLAUDE.md: *"do not weaken them in pursuit of speed"*).
Full context: [`solver/efficiency-ideas.md`](solver/efficiency-ideas.md).

### 3a. Idea 1 — cross-check "converges in 2 iterations" (verification, likely no change)
Cross-tabulate, per (cell, method, N): iterations-to-converge vs. observed MMS rate (`err_u_l2`,
`err_u_h1`, `err_p_l2` slopes). A tolerance is only too loose if a cell converges **fast** *and* its slope
**underperforms** `h^{kv+1}`. Flag only cells that are both. **Likely conclusion: no change needed** — fast
convergence on a mild/near-linear cell is correct-by-design (the gate targets `c_sf·h^{kv+1}`, ~1% of
discretization error; oversolving cannot reduce the O(h^{kv+1})-floored solution error). Do the cross-check
once a sweep lands; documented so the question is not re-litigated. See Idea 1.

### 3b. Idea 2 tier 1 — enable the ASGS stall guard (cheap, reversible, config-only)
The no-progress bail (`"no_progress_stall"`, [nonlinear.jl:698](../src/solvers/nonlinear.jl#L698)) fires
only when `stall_window > 0`, and is **off in production** (`stall_window == 0`), so a stagnating Newton
grinds its entire `max_iters` budget. Set `stall_window ≈ 2` with a sensible `stall_min_rel_improvement` so
Newton bails fast and the existing ping-pong cascade hands off to Picard — a documented config control, not
a safeguard weakening. Caveat: tune so genuine quadratic descent (a one-step stall then quadratic drop) is
not bailed prematurely (interaction with Idea 1). Adopt only behind a measured A/B: iteration counts **and**
final MMS rates unchanged-or-better. See Idea 2.

### 3c. Idea 2 tier 2 — the full Newton↔Picard ping-pong: remaining A/B
The ping-pong itself **has landed** as `_pingpong_cascade!` (gated on `pingpong_enabled`; returns to Newton
after Picard gains `pingpong_picard_gain_orders`; drives the ASGS Stage-I boot and the coupled Stage-II OSGS
solve). Remaining work is the **A/B measurement**: iteration counts + final MMS rates must be
unchanged-or-better. Do not relabel a Picard step as Exact-Newton (CLAUDE.md invariant). See Idea 2 tier 2 /
the suggested order of work.

### 3d. A real saddle-point / MG preconditioner for the OSGS coupled tangent — REQUIRED for P2-OSGS-3D
The JFNK "3D watch item" **fired**, and it is P2-OSGS-3D. Traced (Route-B structured-Kuhn P2-OSGS sweep):
the frozen-π preconditioner gives the inner GMRES no traction on the indefinite (u,p) saddle point —
GMRES(30) reaches rel-res 0.0102 (2% near-miss) on the *easiest* iterate, and **stagnates at rel-res 0.24
after 80 iterations** on a developed iterate → C.1 rejects → frozen-π fallback depletes its line search →
solve fails. **More Krylov vectors will not scale** (classic saddle-point stagnation). Verdict: a real
**block/Schur (PCD/LSC/SIMPLE) or Vanka/MG** preconditioner is required. `τ₁` already seeds a discrete
pressure-Laplacian in the (2,2) block; `τ~h` ⇒ rediscretize per MG level. Do **not** pre-build it for 2D
(equal-order 2D cells need no saddle-point preconditioner — Phase-0 verdict). See
[`solver/jfnk-phase0-preconditioner-gate.md`](solver/jfnk-phase0-preconditioner-gate.md) "3D watch item",
[`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §F7 (TRIGGERED), and
[`mms/3d-iterative-penalty-fix-and-osgs-coupling.md`](mms/3d-iterative-penalty-fix-and-osgs-coupling.md) §5.

### 3e. OSGS-3D-P2 solver robustness (the "good solution, ok=false" blocker)
The **correct OSGS-P2-3D root IS reachable at paper c₁** (JFNK+boot-skip+penalty on (12,12,3) reaches
L²u=0.0012187 — exactly the c₁×4 value), but the solve reports **`ok=false`**: purely a convergence-DETECTION
/ robustness problem, not the discretization. Ranked sub-tasks (from
[`mms/3d-iterative-penalty-fix-and-osgs-coupling.md`](mms/3d-iterative-penalty-fix-and-osgs-coupling.md) §5):
1. **OSGS far-guess robustness via homotopy descent** — if `eps_pert=1` fails, descend (0.1, 0.01, 0) and
   record the largest survivor; make doomed attempts fail fast (small JFNK budget / divergence-patience guard).
2. **A real saddle-point/MG preconditioner** (= 3d above) so JFNK's GMRES converges from any guess — the
   principled, Kratos-matching fix.
3. **Plain/damped staggered π-iteration** (no Anderson over-extrapolation, relaxation < 1) — cheaper than
   JFNK; test whether it makes the π-update contractive in 3D (a manual plain-staggered inner solve converged
   outer-1). Note: the *damped staggered* idea is **dead for P2** per the canonical instability doc (ρ≈8–65,
   every ω diverges) — this sub-task is the milder P1-informed variant; treat as exploratory.
4. Confirm the paper-c₁ OSGS converged root is optimal once a robust solver reaches it (it should be —
   ASGS at paper c₁ is optimal and the discretization is shared).

### 3f. (Infra, tuning open) Anderson acceleration — broader sweep + tuning
Anderson is **landed** behind `osgs_anderson_enabled` (default OFF, bit-identical), and verified to cut
staggered outer-iteration count ≈1.4–2.2× (MILD 33→15, STIFF 19→12; deeper history helps). Remaining:
tuning (`depth`/`relaxation`/`safety`) and a broader sweep across the reaction-/convection-dominated regimes
where the linear rate is the bottleneck. Note it does **not** rescue P2-OSGS-3D (ρ≫1 there). See
[`solver/osgs-anderson-acceleration.md`](solver/osgs-anderson-acceleration.md) "Caveats / scope".

---

## 4. Open code-correctness issues to fix

From [`known_issues.md`](known_issues.md) (report-only) and the audit's still-open appendix. Verified against
the working tree; severity is the author's call.

### 4a. Schema `method` enum vs loader mismatch (confirmed)
`porous_ns.schema.json` allows `method ∈ {ASGS, OSGS, VMS, Galerkin}` (lines ~195-199), but
[config.jl:148](../src/config.jl#L148) asserts `method ∈ {ASGS, OSGS}`. A `VMS`/`Galerkin` config passes
schema validation then dies at the assertion. **Fix:** trim the schema enum to `{ASGS, OSGS}` (or implement
the others). See [`known_issues.md`](known_issues.md) "Confirmed".

### 4b. Latent hardcoded-`ASGS` dispatch in CocquetFormMMS (confirmed)
[CocquetFormMMS/run_test.jl:466](../test/extended/CocquetFormMMS/run_test.jl#L466) hardcodes
`"method" => "ASGS"` in the per-cell config dict — the same pattern fixed in the main MMS harness on
2026-05-26. Not triggered today (that test only iterates `{ASGS, Galerkin}`), but it would mislabel runs if
`OSGS` is ever added. See [`known_issues.md`](known_issues.md).

### 4c. Gridap-vs-Kratos MMS magnitude offset (open, 2026-06-17)
When the high-Re/low-α corner cells are reproduced in Gridap (`run_corner_article.jl`/`run_corner_osgs.jl`),
normalized FME come out **~3–12× larger** than the paper's Kratos values, norm-dependent (vel L² ~7×,
pressure L² ~5×, H¹ ~2.4×). Convergence *rates* agree (≈2–3) and Gridap TRI matches Gridap QUAD to ~2%, so
the discretization is internally consistent — the offset is a **code-vs-code calibration** question
(candidates: characteristic-scale `U_c`/`P_c` normalization, porosity-field definition, MMS amplitude). It
affects how literally `results/paper_tables.tex` reads against the paper. Not a convergence failure. Detail:
[`mms/fold-recovery.md`](mms/fold-recovery.md). See [`known_issues.md`](known_issues.md).

### 4d. Config-strictness / schema `required` gaps (audit A.4 / DRIV-03)
The JSON schema declares `required` for exactly one object (`linear_solver`); add `required` arrays mirroring
the `@kwdef` structs. Also `PhysicalProperties` docstring at [config.jl:23](../src/config.jl#L23) still
mislabels the pressure-penalty ε as *"porosity ε (>0)"* (it may be 0; the inline field comment is correct).
Schema `additionalProperties: true` on `physical_properties` (~line 111) admits unknown keys against the
strict-config intent (soft-guarded by a loader `@warn`, [config.jl:168](../src/config.jl#L168)). See
[`known_issues.md`](known_issues.md) "Minor / cleanup" and [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §A.4.

### 4e. Encoding-invariance test marginal OSGS-covariance failure (pre-existing, NOT Route-B)
`test/quick/encoding_invariance_quick_test.jl`: the OSGS `err_u_l2` cross-encoding covariance is at
reldiff ≈ 1.378e-8 > the `_INV_RTOL = 1e-8` threshold, failing identically on HEAD (the other 5 metrics pass
at ~1e-10). **Do not** relax the threshold merely to go green (repo rule) — diagnose first: either tighten
the OSGS encoding covariance (real work) or, if it is genuinely at the roundoff floor for this cell, revisit
whether `1e-8` is the right OSGS threshold. See [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md)
Minor list.

### 4f. Lower-priority audit appendix findings (fragility/inconsistency, all LOW unless noted)
From [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) Appendix 2 (each verified by an
independent code-grounding skeptic):
- **CONV-05** (fragility, MED): MMS-verifier numerical params (`tau_err`, `eps_*`, `max_extra_cycles`,
  `require_consecutive_passes`, `rate_check_factor`) hard-coded — `test/extended/CocquetFormMMS/run_test.jl:401-412`.
- **CONV-02** (low): inline `1e-2` √d self-check margin — `src/solvers/convergence_criterion.jl:267` (now on
  the `eps_C_strong`/`div_ratio` diagnostic path, not the gate).
- **CONV-04** (low): sub-optimal-rate budget uses a bare power of h with no reference-error normalization
  (scale-dependent) — `src/solvers/mms_verification.jl:130-131`.
- **DRIV-05** (low): production `export_results` writes no provenance (Re/Da/α₀/params) into/alongside VTK —
  `src/io.jl:28-51`.
- **FORM-04** (low): τ rational / Forchheimer `|u|` nonlinearity under-integrated by the polynomial quadrature
  rule — `src/formulations/continuous_problem.jl` (quadrature-degree helpers).
- **MODE-04** (low): porosity logistic saturation guard uses hard-coded ±100 thresholds; α is silently
  2D-only — `src/models/porosity.jl:70-74`.
- **NONL-04** (low): Anderson `update!` has no zero/near-zero residual guard before the least-squares solve —
  now reachable via `osgs_anderson_enabled` — `src/solvers/accelerators.jl:53-127`.
- **NONL-06** (low): Picard line-search reuses Armijo `c1` as a multiplicative residual-reduction factor
  (`1 − c1·α`), a different mathematical role — `src/solvers/nonlinear.jl:434-438`.
- **PROJ-02** (low): `ProjectResidualWithoutPressurePenalty` is never instantiated in production — dead policy
  reachable only from tests — `src/stabilization/projection.jl:34,121-143`.
- **SOLV-03** (low): one-way ASGS Picard success uses scalar ℓ∞ `ftol`, not the scale-free
  `cascade_step_outcome` the ping-pong path and Newton stage use — `src/solvers/asgs_solver.jl:144`.
- **SOLV-05** (low): `discrete_l2_projection` re-solves a fresh `allocate_in_domain` RHS each residual eval;
  `b_vec`/`x_solve` scratch re-allocated — `src/solvers/osgs_solver.jl:59-64`.
- **TAU-02** (low): σ inside τ₁ is evaluated at `effective_speed` (mesh-dependent diffusive floor), not the
  physical `reaction_speed` used in the residual — `src/stabilization/tau.jl` (`Tau1Op`). **Re-check** vs
  [`solver/paper-code-divergences.md`](solver/paper-code-divergences.md) — the σ-in-τ₁ choice may be intentional.

---

## 5. Cleanups

### 5a. Retire dead OSGS config (post coupled-only leaning)
The 2026-06-08 coupled-only leaning left these config keys **ignored by the coupled path** but still in the
schema / `StabilizationConfig` / configs (so nothing breaks): `osgs_projection_coupling`,
`osgs_freeze_after_k`, `osgs_stopping_mode`, `osgs_state_drift_scale`, `osgs_projection_tolerance`,
`osgs_warmup_*`, the ping-pong knobs, `ablation_mode`, and the inert off-switches. Retire them in a
follow-up. See [`solver/coupled-only-leaning-and-jfnk-plan.md`](solver/coupled-only-leaning-and-jfnk-plan.md) §3.

### 5b. Restore/commit the exact 3D driver that produced the committed results (audit B.4)
`convergence3d_results_frontal_c1x1_20260623.json` records `mesh_algorithm =
"gmsh_Frontal_alg4_independent_remesh"` and a 6-level P1 ladder, but **no function now in `smoke3d.jl`
produces that string or ladder** (the sweep paths write `mesh="nested_red"`; `build_sequence("frontal")`
defines only 3 lcs). All of `mesh3d.jl`/`smoke3d.jl`/`plot_convergence3d.py` are uncommitted-modified — the
exact driver is not recoverable from HEAD+worktree, violating the parameters→results-traceability rule. Also
`build_config` sets `eps_val = 1e-8` but `solve_one` builds with `eps_phys = 0.0` (the stored config value
is dead for the solve). Action: restore/commit the exact driver (or re-run with a committed named function
and archive the config snapshot), and make `build_config`'s `eps_val` and `eps_phys` agree (or document
why). See [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §B.4 (overlaps F6, task 2a).

### 5c. `_inv_centered.json` latent fragility
`test/quick/encoding_invariance_quick_test.jl` reads a config it must generate first — fine today, but a
stale leftover can confuse a clean checkout. See [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) Minor list.

---

## Superseded / done (do not re-open)

- **Efficiency Idea 4** (short-circuit OSGS plateau-verification at the machine floor) — **SUPERSEDED** by the
  coupled-only leaning; the staggered outer loop it optimized was deleted.
- **Efficiency Idea 5** (`freeze_after_k` warm-up-then-freeze) — **REVERTED**; a coupling-equivalence oracle
  proved it diverges in the reaction corner. See [`solver/coupled-only-leaning-and-jfnk-plan.md`](solver/coupled-only-leaning-and-jfnk-plan.md) §2.
- **JFNK for the OSGS coupled solve** — Phase-0 gate PASSED and Phase-1 **LANDED** (`osgs_jfnk_enabled`).
  Remaining is only the 3D saddle-point preconditioner (task 3d). See
  [`solver/jfnk-phase0-preconditioner-gate.md`](solver/jfnk-phase0-preconditioner-gate.md).
- **CocquetFormMMS α=0.1×Re=1e5 k=1 corner** — **DONE 2026-07-07** (FE-optimal above the fold, N=[160,320]).
- **2D k1 & k2 QUAD sweeps** — **DONE 2026-07-03** under the Route-B algebraic mass gate, behavior-preserving.
  See [`mms/route-b-2d-sweep-status.md`](mms/route-b-2d-sweep-status.md).
- Many audit findings (A.1/A.2/A.3, C.1–C.5, all of Part D, F1–F4) — **RESOLVED**; see the
  [audit §0 Resolved ledger](formulation-audit-2026-06-24.md#resolved-ledger).
