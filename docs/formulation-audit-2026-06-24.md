# Formulation & Solver Audit — 2026-06-24

**Scope.** A deep, independent re-audit of the whole `src/` tree against the theory
(`theory/paper/article.tex` and the `theory/*` notes), plus a forensic re-examination of the
convergence *results* and the test harness/reporting that produces them. Three questions:

1. Is the theory faithfully transcribed into the code? Where not, what are the remaining inconsistencies and the recommended fixes?
2. Where does the formulation/solver make the algorithm **fragile or inaccurate**, and what is a viable alternative?
3. Where can it be **simplified / reorganised** for clarity, efficiency, or elegance?

**Method.** (a) A line-by-line independent read of every core file in `src/`. (b) A 9-domain
multi-agent audit of `src/` against the paper, with every finding adversarially re-verified by an
independent code-grounding skeptic *and* a theory-grounding skeptic (53 raw findings → 34 upheld, 19
refuted/trivial). (c) An independent recomputation of the 2D and 3D convergence rates straight from the
raw result files (HDF5 + JSON), *not* trusting the summary docs. (d) A fresh control experiment
(`/tmp/audit_3d_structured.jl`, log under `results/debug_results/`) run on a structured Kuhn tet mesh of
uniform, refinement-invariant quality, to separate "mesh quality" from "method/formulation" as the cause
of the 3D rate anomalies.

> **Provenance of this document.** Every code finding below was confirmed by reading the cited
> `file:line`. Every results claim was recomputed from the raw error arrays. The control-experiment
> numbers come from the run logged at
> `test/extended/ManufacturedSolutions3D/results/debug_results/audit_3d_structured.log`.

---

## 0. Status & what remains

> **This is the trimmed working copy (last updated 2026-07-01).** Resolved findings have been condensed
> into the "Resolved ledger" below (one line + commit each — provenance preserved); the body now holds
> **only the still-open items**. The full original audit (executive summary, all 34-finding write-ups, the
> complete resolved detail) is preserved verbatim at commit `a31f191` and its follow-up commits.
>
> **2026-07-01 update:** **C-3 / F4 RESOLVED** (Route B — the mass gate is now the Philosophy-A algebraic
> measure, symmetric with the momentum gate; the strong-form measure is demoted to a diagnostic). See the
> Resolved ledger, the F4 checklist item, and §C.3. A trace-grounded **correction** to the old C.3
> hypothesis is recorded there (the 0.8 gate was NON-binding, not a cause of the P2 pressure errors).
>
> **2026-07-04 update:** (1) **F4 companion `3b76864`** — a scale-free residual-floor accept fixed the
> Route-B tight-gate false-failures on high-Re/fine cells; the **full 2D k2 sweep then completed**
> behavior-preserving (`bf41727`; F4 ledger). (2) **B-5 / F5 measurement COMPLETE** — the mesh-family
> control confirms c₁×2 *masks* / c₁×4 *fixes* (ratio-to-interpolant) for **both** ASGS and OSGS, clean-room
> diagnosis committed (`b20cb78`); but the **c₁ fix is NOT adopted** (author prefers root-cause resolution
> over a multiplier), so F5's *measurement* is closed while its *fix* stays open. (3) **New landed
> hardening** — gauge-free pressure re-centering behind a default-off flag (`6551709`; Minor list), the
> iterated penalty being Codina's but the re-centering **our** proposal.

**Headline that still stands.** The continuous VMS formulation is faithfully transcribed — the strong
residual, both Jacobians, the adjoint sign conventions, the deviatoric/symmetric viscous expansions
(2D & 3D), τ₁/τ₂, the OSGS projection policies, σ(α,u), the porosity field + all four derivatives, and
both MMS oracles each match the paper. **No correctness bug was found in the weak-form assembly.** The
remaining work is in (i) a couple of doc↔code contracts, (ii) the convergence *results* and their
harness/reporting, and (iii) hygiene/fragility gaps.

### Open items at a glance

| # | Finding | Severity | Kind |
|---|---|---|---|
| **B-1** | 3D P1-**ASGS** is genuinely sub-optimal (L²u ≈ 1.2, not 2) even on a perfect uniform Kuhn mesh — method-intrinsic, not mesh quality; OSGS recovers it. **Action: document honestly** (still open). | High | results/theory |
| **B-5 / F5** | P2-3D under-stabilization at paper c₁=4k⁴ on a uniform mesh. **Measurement complete (2026-07-03):** c₁×2 *masks* (ratio-to-interpolant drifts, H¹u stalls), **c₁×4 *fixes*** (ratio pins →1, both ASGS & OSGS) ⇒ threshold c₁* ∈ ×(2,4); clean-room diagnosis committed (`b20cb78`). **Fix decision OPEN** — the c₁ multiplier is confirmed effective but **not adopted** (author prefers root-cause resolution over a multiplier). | High | results/theory |
| **B-2** | The committed "3D-P2 divergence" tables mix `success=False` solves into the rate table; the plotter doesn't gate on `success`. **Action: gate/flag failed levels.** | High | reporting |
| **B-3 / B-6** | A failed OSGS solve silently reports the ASGS Stage-I boot state under the OSGS label (byte-identical error tuples at shared-failed levels). **Action: mark OSGS-degenerated-to-ASGS distinctly.** | Med | reporting/harness |
| **B-4** | The most recent committed 3D result files can't be reproduced from the current harness (`mesh_algorithm`/ladder match no function in `smoke3d.jl`). **Action: restore/commit the exact driver** (overlaps F6). | Med | reproducibility |
| **A-4** | Config-strictness gaps: no `required` arrays in the schema (except `linear_solver`), `base_config.json` omits `eps_val`, and the `eps_val` docstring still mislabels it "porosity ε (>0)". | Low | doc↔code |
| **F6** | The 3D MMS harness (`smoke3d.jl`) is a hand-edited driver with no committed config JSON and no automated guard. | Med | cleanup/test |

*(C-3 / F4 — the loose mass gate — was **RESOLVED 2026-07-01** via Route B; moved to the Resolved ledger.)*

### Resolved ledger (provenance — do not re-open without reading the cited commit)

- **A-1 / F1** ✅ 2026-06-25 (`c010f98`) — `dynamic_*` Re/Da budget knobs relocated out of production
  `SolverConfig`/schema/`base_config.json` into harness-frame `test/extended/harness_dynamic_budget.jl`.
  Behavior-preserving (2D MMS bit-identical). CLAUDE.md + docstrings corrected.
- **A-2** ✅ 2026-06-25 (`2f50d6d`) — the scale-free ε_M/ε_C gate is documented everywhere as the
  authoritative production success test (was mislabeled "diagnostic / not yet wired in"); per-iteration
  cost documented as intentional.
- **C-2 / SOLV-04 / MODE-02 / MODE-03** ✅ 2026-06-25 — load-time `validate!` now fails loud on
  `eps_tol_momentum/mass > 0`, σ SPSD (`sigma_constant/linear/nonlinear ≥ 0`), strictly-positive velocity
  floor, `0 < alpha_0 ≤ 1`, `r_1 < r_2`, and `bounding_box` even-length parity.
- **C-4 / FORM-02 / MODE-01 / TAU-01 / DRIV-08** ✅ 2026-06-25 (`e842de9`) — the copy-pasted `1e-12`
  `|u|`-floor became the config input `velocity_magnitude_derivative_floor` (one source of truth, threaded
  through `DSigOp`/`DTau1Op`/`DTau2Op`). Documented in `theory/velocity_floor_regularization/`.
- **C-5 / PROJ-01** ✅ 2026-06-25 (Batch 1) — the §4.4 constant-σ trim now gates on an `is_sigma_constant`
  trait (true only for `ConstantSigmaLaw`), not a concrete-type check, so a new nonlinear law is rejected
  by default.
- **NONL-05** ✅ 2026-06-25 (Batch 1) — `ILUGMRESSolver` keyword constructor no longer backfills
  magic-number defaults.
- **DRIV-02** ✅ Batch 1 — dead `diagnostics_helpers.jl` (referenced a removed API) deleted.
- **Part D (all)** ✅ 2026-06-25 (`e842de9`, `7195581`, + 2026-06-26):
  - **FORM-01** — `build_picard_jacobian` is now a one-line wrapper over the general builder in
    `PicardMode()`, guarded by `test/blitz/picard_jacobian_equivalence_blitz_test.jl` (byte-equality).
  - **FORM-03** — the duplicated `dL_du_star_v` recomputation deleted (built once, reused).
  - **FORM-05** — the σ/τ₁/τ₂ + setup-scalar boilerplate extracted to
    `_build_stabilization_coefficients` (shared by the residual and both Jacobians).
  - **TAU-05** — the τ_{1,NS}⁻¹ denominator extracted to one `_tau_ns_inv` helper used by all four
    τ functions (primal + derivatives) — math re-verified.
  - **VISC-02** — the `∇(∇·u)` contraction unified into one `_grad_div` helper; dead
    `gridap_extensions.jl` deleted.
  - **VISC-01** — `EvalDivDevSymOp`/`EvalStrongViscSymOp` collapsed to single D-generic methods; the
    deviatoric grad-div coefficient is computed as `0.5 − 1/D` from the Hessian's type parameter
    (byte-identical to the old per-dimension `0.0`/`0.5−1/3`), so a new dimension cannot drift it.
  - **VISC-03** — tracked analytic regression test for the 3D deviatoric grad-div coefficient (=1/6),
    strong + adjoint, in `test/quick/viscous_operators_quick_test.jl`.
  - **NONL-03** — `accelerators.jl` (`AndersonAccelerator`) wired into the OSGS stage behind the opt-in
    `osgs_anderson_enabled` (OFF by default).
  - **DRIV-07** — the MMS oracle's inline `p_ex`/`∇p_ex` unified into a `PExFunc` with a registered ∇.
- **C-1 / F3 / NONL-01 / NONL-02** ✅ 2026-06-26 — ILU_GMRES honesty contract
  (`src/solvers/linear_solvers.jl`): `solve!` runs `gmres!(…; log=true)` and throws `GMRESNotConvergedError`
  if it does not reach `rel_tol` within `maxiter` (a non-converged step is never returned as exact); the
  ILU→identity fallback is gated by the new required config `allow_unpreconditioned_fallback` (default
  `false` ⇒ `ILUFactorizationFailure`, fail loud; `true` ⇒ identity `Pl` + loud warning). The cryptic
  `println` fallback warning is replaced by an explicit trace line; `eval_linear_system_resolution!` traces
  a distinct `[Linear Solve NOT CONVERGED]` vs `[Linear Solve Exception]`; `stop_reason` renamed
  `linear_solve_nan` → `linear_solve_failed`; both route to honest cascade rollback → Picard. Deterministic
  guard `test/blitz/linear_solver_honesty_blitz_test.jl` (6 cases). Verified Blitz 219/219, Quick 57/57.
  The 2D harnesses use `LUSolver`, so the change is inert there (behavior-preserving); it affects only the
  3D fine-mesh `ILU_GMRES` path — re-baselining the 3D structured-Kuhn control is the remaining *follow-up
  measurement* (which fine-mesh OSGS solves now report honest failure), not open implementation.
- **A-3** ✅ 2026-06-26 — OSGS coupled Newton tangent now uses the **live π** (`src/solvers/osgs_solver.jl`,
  `live_pi!`): the `dτ_1·(R−π)` / `dL*·(R−π)` Exact-Newton product-rule terms previously got a literal ZERO
  π (silently `R` instead of `R−π`). Passing the live π makes the assembled matrix the **exact frozen-π
  tangent** (verified: it matches a finite-difference of the frozen-π residual to ~1e-11, vs a ~3% error
  for the old zero-π tangent; `test/quick/osgs_frozen_pi_jacobian_quick_test.jl`). π is computed once per
  iterate and shared by residual+Jacobian via a DOF-keyed cache, so the exact tangent costs no extra
  projection. Converged solution unchanged (R−π→0 at the fixed point); convergence is no worse (same root,
  residual ≤ zero-π after a fixed budget) — the rate stays linear because the **intentionally** dropped
  `∂π/∂u` dominates it (that is the JFNK frontier, out of scope). The false tex claim ("a zero placeholder
  is the correct argument") was corrected in `theory/osgs_algorithm/osgs_algorithm.tex`.
- **C-3 / F4** ✅ 2026-07-01 (Route B) — the mass gate is now the **Philosophy-A algebraic** `ε_C = ‖r_C‖/D_C`
  (pressure block of the same residual `b` that gives `r_M`, over a Galerkin mass envelope), SYMMETRIC with
  `ε_M` and → 0 at the discrete solution, gated at `tol_C = eps_tol_momentum`. The strong-form
  `‖εp+∇·(αu)−g‖/(‖∇(αu)‖+‖g‖)` — which FLOORS at O(h^{kv}) and forced the loose `0.8` — is demoted to the
  DIAGNOSTIC `eps_C_strong` (traced, never gated). `convergence_criterion.jl` (`mass_force_envelope`, new
  `ConvergenceMeasure` fields `r_C`/`D_C`/`eps_C_strong`), `solver_core.jl` probe, `nonlinear.jl` trace;
  config `base_config.json` `0.8→1e-6` + symmetric `phase1_quad_k2.json`/`smoke3d.jl` + schema/`config.jl`.
  No tangent change (the coupled/JFNK Jacobian was already the full momentum+mass tangent; only the
  read-only stopping test changed). Verified Blitz 240/240, Quick convergence test 26/26; 2D A/B and 3D
  ASGS behavior-preserving; traces show `ε_C` driven `1e-3→1e-12` while `eps_C_strong` floors at O(h). The
  original C.3 "loose gate causes the P2 pressure errors" hypothesis is **refuted** by traces (§C.3).
  **Companion fix `3b76864` (2026-07-02):** the tight `tol_C=1e-9` gate false-failed high-Re / fine-mesh
  cells whose residual is at the machine floor but whose Philosophy-A `ε_C` floors ~1e-8 (its `D_C`
  envelope collapses for near-divergence-free flow), and each false failure burned the whole
  eps_pert-homotopy fallback. A scale-free residual-floor accept (`residual_floor_reached`: `ε_M ≤ tol_M`
  **and** the per-field residual ≤ `k_nf·ftol`, guarded by `!degenerate`; reuses the existing
  `noise_floor_success_max_ftol_multiple`, no new magic number) accepts them. With it the **full 2D k2 QUAD
  sweep completed behavior-preserving** (`bf41727`; ASGS+OSGS, N10→N320, 0 NaN, median L²u rate 3.00,
  vs pre-Route-B baseline median rel Δe_u 1.97e-11) — see
  [`docs/mms/route-b-2d-sweep-status.md`](docs/mms/route-b-2d-sweep-status.md). Full 3D-OSGS sweep remains
  the measurement follow-up. See §C.3 + the F4 checklist item.

---

## Deferred follow-ups — implementation checklist (open)

The **landed** items (Batch 1 `2f50d6d`; the dedup/config-ify batch `e842de9`; F1 `c010f98`; Anderson +
DRIV-07 `7195581`; F2/FORM-01 + VISC-01 D-generic coefficient) are recorded in the **Resolved ledger** in
§0. The items below remain open; each is self-contained and ordered roughly by value.

### F3 — C.1: ILU-GMRES honesty (behavior-CHANGING, 3D-only) → §C.1
✅ **Landed 2026-06-26** (see §0 Resolved ledger / §C.1): `gmres!(…; log=true)` + `isconverged` check →
`GMRESNotConvergedError`; the ILU→identity fallback is gated by the required `allow_unpreconditioned_fallback`
(schema + `LinearSolverConfig` + JSON), default `false` = fail-loud `ILUFactorizationFailure`; distinct
non-convergence vs exception traces; deterministic blitz guard. Blitz 219/219, Quick 57/57.
- **Remaining follow-up (measurement, not implementation):** re-baseline the 3D structured-Kuhn control
  before/after — it is *expected* to flip which fine-mesh OSGS solves report success (that is the point).
  The 2D harnesses use `LUSolver`, so they are unaffected by C.1; a post-C.1 k=2 QUAD MMS rerun is in
  progress as the 2D behavior-preservation check (completed cells show the expected optimal k=2 rates).

### F4 — C.3: the mass gate → §C.3
✅ **Landed 2026-07-01 (Route B)** — resolved not by tuning `eps_tol_mass` down but by **replacing the
gated quantity**. The investigation ("why can't `eps_tol_mass` be < 0.8") answered itself: the old gate's
strong-form L² mass residual FLOORS at O(h^{kv}) (empirically ~0.11 P1 / ~0.2–0.5 coarse-3D), so *no*
fixed tolerance below that floor is satisfiable — the `0.8` was "safely above every mesh's floor," i.e. a
non-binding rubber-stamp. Fix: gate the **Philosophy-A algebraic** mass residual `ε_C = ‖r_C‖/D_C` (the
pressure block of the same assembled residual `b` the momentum gate reads, already passed into the probe
as `field_blocks[2]`), which → 0 at the discrete solution and is gate-able at `tol_C = eps_tol_momentum`,
symmetric with `ε_M`. The strong-form measure + the `‖∇·(αu)‖/‖∇(αu)‖ ≤ √d` self-check are retained as the
DIAGNOSTIC `eps_C_strong` (traced, never gated). See §0 Resolved ledger / §C.3.
- **Remaining follow-up (measurement, not implementation):** the full 3D-OSGS structured-Kuhn sweep is
  re-running to confirm the honest mass-gate convergence at each cell (first OSGS-P1 cell: `ε_C 0.97→6e-9`
  via JFNK, honest `success`, pressure error L²p 0.44108→0.4374 ≈ −0.83% — the under-convergence tail the
  old soft-stall left, now drained). The 2D `LUSolver` harnesses are behavior-preserving (A/B to the root).

### F5 — element-type-aware c₁ for P2 tets (the B.5 fix) → §B.5
- **Status (2026-07-03): MEASUREMENT COMPLETE; FIX DECISION OPEN.** The Gridap §5.1 mesh-family control
  (§B.5) confirms the mechanism for BOTH methods: c₁×2 *masks* (ratio-to-interpolant drifts 1.34→1.94, H¹u
  stalls), c₁×4 *fixes* (ratio pins 1.14→1.06, tracking the interpolant), so the h-robust threshold is
  c₁* ∈ ×(2,4) — higher than the clean-room report's single-mesh ×(1.5,2). The clean-room diagnosis is
  committed at [`docs/convergence_problems_audit/`](docs/convergence_problems_audit/) (`b20cb78`) and
  cross-linked as **contested-pending-confirmation** to the canonical
  [`docs/mms/3d-p2-instability-investigation.md`](docs/mms/3d-p2-instability-investigation.md) (⚠ note), which
  the §5.1 sweep now supplies. **The c₁ multiplier is confirmed effective but NOT adopted** — the author
  prefers resolving the root cause over shipping an ad-hoc multiplier (even an element-aware one). So the
  item below is the *option on the table*, not a committed direction.
- (Option, if adopted) Add a config-driven `c1_multiplier` (or a per-`(element, order)` table), explicitly
  `[code-actual]`, so P2 tetrahedra use a larger c₁; 2D/quad stays paper-faithful at `4k⁴`. Reconcile the
  canonical doc's TL;DR #2 ("not c₁") with the confirmed c₁-coercivity finding, and measure the actual
  `C_inv` for P2 Kuhn vs gmsh tets to calibrate the multiplier.

### F6 — make the 3D MMS test config-driven + add an official 3D-MMS test
- **Why:** `test/extended/ManufacturedSolutions3D/smoke3d.jl` is a manual sweep driver with hard-coded
  study params (`RE`/`DA`/`ALPHA0`/`R1`/`R2`/`L`, the `mesh_sequence`, `c1_mult`, the `kv`/`method` loops),
  the 3D analogue of the 2D `run_test.jl` — but unlike the 2D side it has **no committed config JSON and no
  automated guard**, so every 3D study is a hand-edited driver (the source of the deleted one-off scrap).
- **Recipe:** (1) lift the hard-coded study knobs into a JSON config (mirror the 2D
  `ManufacturedSolutions/data/*.json` shape, plus the 3D-specific `mesh_sequence` / `c1_mult` / domain slab
  z-extent); (2) add a `config <path>` entry mode to `smoke3d.jl` that reads it and drives the existing
  `solve_one`/`run_sweep*` machinery (reuses `mesh3d.jl`'s `structured_kuhn_model` / `build_sequence`);
  (3) commit a lean smoke config (`data/smoke3d_p1.json`: §5.2, structured-Kuhn, 2-3 coarse levels, k=1,
  ASGS+OSGS, paper c₁); (4) wire a lean **official** extended test into `test/run_extended_tests.jl` that
  runs that config and asserts the solve succeeds and the P1 rate is within tolerance of optimal — making
  the 3D MMS path the official debug reference, with the bigger studies (structured control, c₁×4) as
  committed configs instead of ad-hoc drivers.
- **Verify:** the new extended test (a 3D solve — minutes of compile+solve), plus a manual config run
  reproducing one cell of the committed §5.2 numbers.

### F7 — JFNK for the OSGS coupled solve (Phase-0 gate PASSED → GO; Phase-1 open) → §A.3
- **Why:** the A.3 frozen-π tangent drops the dense `∂π/∂u = ∫L*τ·Π(dR·dU)` coupling, which sets a slow
  *linear* rate that on stiff/convective cells makes the coupled inexact-Newton **diverge**. JFNK recovers it
  matrix-free (`J·v ≈ [F(U+εv)−F(U)]/ε`, the residual already re-projects π) preconditioned by the
  already-factored frozen-π Jacobian. See `theory/osgs_algorithm/osgs_algorithm.tex` §`sec:jfnk`.
- ✅ **Phase-0 gate PASSED 2026-06-26 — decision GO** (full writeup + tables:
  `docs/solver/jfnk-phase0-preconditioner-gate.md`). The throwaway `ρ_prec ≈ 1e5–1e9` was 100% the
  constant-pressure null mode (`eps_val=0` contamination; worst-eigvec overlap 1.0, ρ_defl 0.74). Clean
  re-measure (production `ε_phys`): the **free** frozen-π preconditioner (ε_num=0) drives the inner GMRES to
  **1–4 iters (mild) / ≤16–21 peak (stiff)**; `N_j=2–3` (quadratic) vs `N_c=60` (non-converging/diverging);
  beats Anderson (32–418 fact, non-converging on stiff). Two-ε finding: adding `ε_num` to the preconditioner
  **hurts** — use ε_num=0, rely on residual-consistent `ε_phys`.
- ✅ **Phase-1 LANDED 2026-06-26** — opt-in `osgs_jfnk_enabled` (default false; behavior bit-identical when
  off), Krylov knobs (`osgs_jfnk_gmres_rel_tol`/`_maxiter`/`_restart`, `osgs_jfnk_fd_epsilon`) in
  schema/`SolverConfig`/`base_config` with fail-loud `validate!` + a mutual-exclusion assert vs Anderson.
  Realised NOT as a from-scratch outer loop but as a drop-in matrix-free `JFNKLinearSolver`
  (`src/solvers/linear_solvers.jl`) plugged into the existing `SafeNewtonSolver` by `_osgs_jfnk_solve!`
  (`src/solvers/osgs_solver.jl`) — the theory's "change exactly one thing: the inner linear solve" — so the
  outer Newton, Armijo/merit line search, divergence/stall guards, and the **C.1 honesty** contract
  (`GMRESNotConvergedError` → roll back → fall back to the frozen-π coupled solve) are all inherited
  unchanged, with zero re-implemented safeguards. The FD base point x_k is threaded in via a Ref written by
  a thin wrapper around `jac_fn_coupled` (Gridap evaluates the jacobian at the iterate right before the inner
  solve). Verified: Blitz 240/240, Quick 76/76 (incl. `osgs_jfnk_quick_test.jl`: matrix-free action == true
  full tangent; production solve → same root as a full-Jacobian Newton to 1e-6; starved-GMRES honesty), and
  `jfnk_equivalence_extended_test.jl` (ASGS byte-identical; OSGS same MMS root; JFNK iters ≤ frozen-π).
  Line-search note: reuses the exact-slope (D=−2Φ) test verbatim — at η≈1e-2 the inexact (1−η) relaxation is
  a 1% conservative effect (safe). **3D watch item (open):** if the inner `G` blows up in 3D (C.1 will flag
  it), that is the trigger to add a real saddle-point preconditioner (block/Schur — PCD/LSC/SIMPLE — or
  Vanka/MG) — do not pre-build it.
- **[TRIGGERED 2026-07-01 — the 3D watch item just fired, and it is P2-OSGS.]** The Route-B structured-Kuhn
  P2-OSGS sweep isolates the P2-3D failure to **exactly this inner-solve frontier**, not the mass gate or c₁:
  - **Mechanism (traced, `…_TET_OSGS_N10.json`):** at `eps_pert=0` the JFNK inner GMRES **does not converge**
    — full GMRES(30) reaches rel residual **0.0102 > rel_tol 0.01** (a 2% near-miss) → C.1 rejects the step
    (`stop=linear_solve_failed`); the **frozen-π fallback then depletes its line search** (`linesearch_failed`
    after 1 step, `ε_M` stuck at 7.9e-2); homotopy is already at `eps_pert=0` ⇒ solve fails. Reproduced on
    BOTH coarse levels (12,12,3 ndof≈12.3k and 16,16,4 ndof≈30k; `linsolver=LU` outer, so this is the JFNK
    *inner* Krylov solve with the frozen-π/ILU preconditioner, not the outer factorization).
  - **Route B is orthogonal here:** the P2-OSGS results are **byte-identical to the committed baseline**
    (success=False, eps_used=0, same errors to ≥4 figs) — the solve never reaches the convergence gate, so
    the mass-gate change cannot and does not affect it. This *rules the gate/criterion out* of the P2-OSGS-3D
    problem and confirms F7's prediction: the bottleneck is the **preconditioner**.
  - **Cheap-lead probe — RESOLVED: it STAGNATES (budget is not the fix).** Re-running the coarsest cell
    with full GMRES(80): the *first* Newton step's inner GMRES now converges (< 0.01) and a step is accepted,
    but the **second** step's GMRES **stagnates at rel residual 0.24 after all 80 iterations** → rejected →
    frozen-π fallback → `linesearch_failed` → same byte-identical failure (success=False, eps_used=0). A 24%
    residual from an 80-vector Krylov space is classic saddle-point stagnation: the 2% "near-miss" at
    maxiter=30 was only the *easiest* iterate (the exact-guess start); once the iterate develops, the
    frozen-π preconditioner barely dents the residual regardless of budget. **Verdict: F7 go — a real
    saddle-point preconditioner (block/Schur — PCD/LSC/SIMPLE — or Vanka/MG) is REQUIRED for P2-OSGS-3D; more
    Krylov vectors will not scale.** (Probe: `routeb_3d_p2osgs_maxiter.jl`; trace `…_p2mi/…OSGS_N10.json`.)
  - **Contrast (same sweep):** 3D ASGS-P2 is behavior-preserving under Route B (matches baseline to 3–4 figs)
    and *succeeds* via the eps_pert homotopy — so the P2-3D *ASGS* story stays the c₁/B.5 track, while the
    P2-3D *OSGS* story is this inner-solve/preconditioner track. Two distinct P2-3D problems, now separated.

### Minor / opportunistic
- `_inv_centered.json` latent fragility: the official `test/quick/encoding_invariance_quick_test.jl` reads
  a config it must generate first — fine today, but a stale leftover can confuse a clean checkout.
- **[NEW 2026-07-01] `encoding_invariance_quick_test.jl` fails marginally on HEAD (pre-existing, not a
  Route-B regression).** A worktree A/B (HEAD vs Route B) run on 2026-07-01 shows the test's **OSGS
  `err_u_l2`** cross-encoding covariance at **reldiff ≈ 1.378e-8 > the `_INV_RTOL = 1e-8` threshold** —
  failing identically on HEAD (Route B is byte-comparable, 1.372e-8; the other 5 metrics pass at ~1e-10).
  So the single "1 failed" in the current Quick suite is this pre-existing marginal OSGS-covariance issue,
  independent of Route B. Action: either tighten the OSGS encoding covariance (real work) or, if it is
  genuinely at the roundoff floor for this cell, revisit whether `1e-8` is the right OSGS threshold —
  **do not** relax it merely to go green (repo rule); diagnose first.
- **NONL-04** (Anderson `update!` has no zero/near-zero residual guard before the least-squares solve).
  Now reachable: `accelerators.jl` is wired into the OSGS stage behind the opt-in `osgs_anderson_enabled`
  (NONL-03, OFF by default), so this guard matters on that path when enabled. (NONL-01 — the ILU-GMRES
  convergence check — is resolved under C.1; see §0 ledger.)
- **[NEW 2026-07-04] Pressure-mean drift under the iterated penalty — hardening LANDED (`6551709`).** In the
  all-Dirichlet gauge-free case the iterated penalty pins the pressure constant to the *previous* pass, not to
  zero mean; a fixed boundary-flux residue `ρ` shifts the mean by `−ρ/(ε_num|Ω|)` per pass (A/B: raw mean 230
  against a true `‖p‖≈0.7` on a coarse P1 mesh — it dominates). Fix: opt-in
  `recenter_pressure_between_penalty_passes` (default OFF ⇒ bit-identical) re-centers both the lag and the
  returned field; mean-removed errors A/B byte-identical, stored mean → machine zero. `validate!` guards it to
  the gauge-free case (`iterative_penalty_enabled` + `eps_val==0`). **Provenance:** the iterated penalty is
  Codina's; the re-centering is *our* proposal, **not** attributed to him — full derivation + provenance in
  [`theory/pressure_recentering_note/`](theory/pressure_recentering_note/pressure_recentering_note.tex).

**Bit-identity verification recipe (reused for the dedup/config-ify batches):** capture a clean pre-change baseline by
`git stash push -- src/`, run a couple of 2D cells
(`run_test.jl phase1_quad_k1.json --filter kv=1,kp=1,etype=QUAD,Re=1.0,Da=1.0,alpha0=0.5 --max-N 40 --h5 debug_results/baseline_pre.h5`),
`git checkout -- src/` to restore edits, re-run to `baseline_post.h5`, and compare the `err_*` arrays for
exact equality.

---

## Part A — Theory ↔ code consistency

### A.0 What is faithful (re-verified, not assumed)

- **Strong momentum/mass residual** (`continuous_problem.jl` `eval_strong_residual_u/p`) matches
  `eq:StrongMomentumEquation`/`eq:StrongMassEquation` term-by-term, including the IBP pressure form
  `−p(α∇·v + ∇α·v)` (= `−∫ p ∇·(αv)`) and `∇·(αu) = α∇·u + u·∇α`.
- **Adjoint sign discipline.** `strong_adjoint_momentum` returns `+α(∇v)'·u` and `B_S` subtracts the
  adjoint; the `σv` term is subtracted. Matches Eq.39/Eq.50 and the documented A²−B² symmetry. The
  `(1/α)∇·(αa)v` omission is the paper's own simplification (article.tex ~L800).
- **Viscous operators** (`viscous_operators.jl`). `∇·ε^d(u)=½Δu+(½−1/d)∇(∇·u)` with coefficient `0` in
  2D and `+1/6` in 3D; `∇·ε(u)=½Δu+½∇(∇·u)`. The 2αν factor (μ=αν), the weak Jacobian (linear ⇒ `du`
  for `u`), and the self-adjoint reuse on `v` are all correct. The 3D MMS oracle (`mms3d.jl`) uses the
  matching `∇·(2αν∇ᵈu)=2ν(∇ˢu·∇α−⅓(∇·u)∇α)+ανΔu+(αν/3)∇(∇·u)`.
- **τ₁/τ₂** (`tau.jl`) implement `eq:Tau1Final`/`eq:Tau2Final` exactly: `τ₁=1/(α·τ_NS⁻¹+σ)`,
  `τ₂=h²/(c₁ α τ_NS)`, `τ_NS=(c₁ν/h²+c₂|w|/h)⁻¹`, `c₁=4k⁴, c₂=2k²`. The dropped `εh²` and `C_α` terms
  are the paper's own §4.2 simplifications. The dτ/du derivatives are the correct chain rule.
- **OSGS projection policies** (`projection.jl`) implement the §4.4 `(1−Π)` trim correctly; the
  unconstrained `V_free/Q_free` projection space matches the divergences-ledger requirement.
- **Reaction** (`reaction.jl`): `σ=a(α)+b(α)|u|`, `a=σ_lin((1−α)/α)²`, `b=σ_nonlin(1−α)/α`; `dσ/du`
  is the exact `b·(u·du)/|u|`.
- **Porosity** (`porosity.jl`): `α`, `∇α`, the scalar Laplacian, and the full Hessian were each derived
  by hand and match; the Hessian trace equals the Laplacian (mutually consistent), and the logistic is
  genuinely C∞ across the annulus boundaries. The MMS oracles are exact for the same operators the
  solver assembles.
- **ε_num** is Jacobian-pressure-block only in *both* Jacobian builders and absent from the residual and
  the VMS subscale — cancels in the residual, vanishes at convergence, as documented.

> **A.1 [HIGH] and A.2 [HIGH] are ✅ RESOLVED** — see the Resolved ledger in §0 (A.1: `dynamic_*` knob
> relocation, `c010f98`; A.2: scale-free ε_M/ε_C gate documented as authoritative, `2f50d6d`).

> **A.3 [LOW] is ✅ RESOLVED 2026-06-26** — the OSGS coupled Newton Jacobian now passes the **live π**
> (`osgs_solver.jl` `live_pi!`), so the `dτ_1·(R−π)` / `dL*·(R−π)` terms use `(R−π)` instead of the old
> zero-placeholder `R`. The assembled matrix is now the **exact frozen-π tangent** (FD-verified to ~1e-11;
> the old zero-π tangent had a ~3% error) at no extra projection cost (DOF-keyed per-iterate π cache). The
> converged solution is unchanged and convergence does not degrade; the rate stays linear because the
> *intentionally* dropped `∂π/∂u` dominates it (the JFNK frontier). The false tex claim was corrected.
> See §0 ledger, `test/quick/osgs_frozen_pi_jacobian_quick_test.jl`, and `theory/osgs_algorithm/`.

### A.4 [LOW] Config-strictness gaps relative to the repo's own hard rule

- The JSON schema declares `required` for exactly one object (`linear_solver`); presence is enforced in
  practice by the no-default `@kwdef` structs, but the schema does not encode it
  ([porous_ns.schema.json](config/porous_ns.schema.json)). Add `required` arrays mirroring the structs.
- `config/base_config.json` omits the now-required `eps_val` and fails `load_frozen_config`
  (documented; **DRIV-01**). Either add `eps_val` to the canonical example or document that it is
  intentionally incomplete.
- Doc bug: the `PhysicalProperties` struct docstring ([config.jl:23](src/config.jl#L23)) calls `eps_val`
  *"porosity ε of the medium (>0)"* — it is the compressibility/pressure-penalty ε (α is porosity), and
  it may be 0. The inline field comment is correct; the struct docstring is not.

---

## Part C — Fragility / inaccuracy, with viable alternatives

> **C.1 [Med] is ✅ RESOLVED 2026-06-26** (with NONL-01 / NONL-02 / F3) — `ILU_GMRES.solve!` now verifies
> GMRES convergence (`gmres!(…; log=true)` → `isconverged`) and throws `GMRESNotConvergedError` rather than
> returning a non-converged step as exact; the ILU→identity fallback is gated by the required config
> `allow_unpreconditioned_fallback` (default `false` ⇒ fail-loud `ILUFactorizationFailure`), replacing the
> silent `println`. Traces distinguish `[Linear Solve NOT CONVERGED]` from `[Linear Solve Exception]`; both
> roll the cascade back to Picard. See §0 ledger + `src/solvers/linear_solvers.jl` +
> `test/blitz/linear_solver_honesty_blitz_test.jl`. The 3D structured-Kuhn re-baseline is a remaining
> *measurement* follow-up (§F3), not open implementation.

> **C.2 [Med] is ✅ RESOLVED** (MODE-03 / MODE-02 / SOLV-04 + domain bounds) — `validate!` now fails loud
> on σ SPSD (`sigma_constant/linear/nonlinear ≥ 0`), strictly-positive velocity floor, `eps_tol_momentum/
> mass > 0`, `0 < alpha_0 ≤ 1`, `r_1 < r_2`, and `bounding_box` even-length parity. See §0 ledger.

### C.3 [Med] The mass-convergence gate — ✅ RESOLVED 2026-07-01 (Route B)

> **Resolution (Route B).** The gate no longer measures the strong-form residual at all. `ε_C` is now the
> **Philosophy-A algebraic** measure `‖r_C‖/D_C` (pressure block of the assembled residual `b`, over the
> Galerkin mass envelope), symmetric with `ε_M`, → 0 at the discrete solution, gated at
> `tol_C = eps_tol_momentum`. The strong-form quantity below is kept as the diagnostic `eps_C_strong`.
> See the §0 Resolved ledger and the F4 checklist item for the full write-up and verification.
>
> **Trace-grounded correction to the original finding.** Two claims in the original text (kept below for
> provenance) are **refuted by measurement**:
> 1. *"0.8 … plausibly contributes to the large converged P2 pressure errors."* **False.** At accepted 3D
>    k=2 iterates the strong-form `ε_C` sits at ~0.2–0.5 — well **below** 0.8 — so the 0.8 gate was
>    **non-binding**; the momentum gate (1e-9) was the sole operative constraint. The large P2 pressure
>    errors belong to the **B.5 c₁/under-stabilization** track, not the mass gate.
> 2. *"add a separate tighter check on `‖∇·(αu)‖/‖∇(αu)‖ ≤ √d`."* That ratio is an analytic identity that
>    holds for *any* field and → `‖g‖/‖∇(αu)‖ ≠ 0` under a forced MMS, so it is **not** a convergence
>    check. The correct fix was to gate a quantity that genuinely → 0 — the weak/algebraic pressure-block
>    residual — which the momentum gate already had in hand and was discarding.
>
> The real defect the 0.8 hid was that continuity was **un-gated** (a rubber-stamp), and that
> `eps_C_strong` FLOORS at O(h^{kv}) (empirically ~0.11 P1 / ~0.2–0.5 coarse-3D) so *no* fixed sub-floor
> tolerance is satisfiable — which is why it had to be loose. Route B removes the magic number and makes
> continuity an honest gate. Original finding preserved verbatim below.

ε_C = ‖εp+∇·(αu)−g‖ / (‖∇(αu)‖+‖g‖). A tolerance of **0.8** accepts an iterate whose mass-equation
residual is 80% of the flux-gradient envelope. The momentum gate `eps_tol_momentum=1e-9` is tight; the
mass gate is not. This plausibly contributes to the large "converged" P2 pressure errors (H¹p ≈ 2.3 at
the coarsest converged level). The looseness is intentional (the k=2-gate lesson cites mass-residual
scale sensitivity), but 0.8 is a wide margin to leave on the *only* check of the pressure/continuity
balance.

**Alternative.** Investigate why ε_C cannot be tightened (is the `‖∇(αu)‖` envelope the right scale for
the P2 pressure block?) rather than accept 0.8 as a fixed constant; if it genuinely must stay loose,
add a separate, tighter check on the pure-divergence ratio `‖∇·(αu)‖/‖∇(αu)‖` (already computed as the
√d self-check) so continuity is still gated.

> **C.4 [Low] and C.5 [Low] are ✅ RESOLVED** — C.4: the `1e-12` `|u|`-floor became the config input
> `velocity_magnitude_derivative_floor` (`e842de9`); C.5: the §4.4 constant-σ trim now gates on an
> `is_sigma_constant` trait. See §0 ledger.

---

## Part D — Simplification / reorganisation — ✅ ALL RESOLVED

> Every Part-D item has landed (see the §0 Resolved ledger for commits): FORM-01 (`build_picard_jacobian`
> → one-line wrapper + byte-equality guard test), FORM-03 (duplicate `dL_du_star_v` deleted), FORM-05
> (`_build_stabilization_coefficients` helper), TAU-05 (`_tau_ns_inv` helper, math re-verified), VISC-01
> (deviatoric grad-div coefficient computed as `0.5 − 1/D` from the Hessian's type parameter — single
> D-generic method, byte-identical to the old per-dimension constants), VISC-02 (`_grad_div` unifies the
> contraction; dead `gridap_extensions.jl` deleted), VISC-03 (tracked 3D grad-div analytic regression test
> in `viscous_operators_quick_test.jl`), NONL-03 (`AndersonAccelerator` wired into OSGS, opt-in/off),
> DRIV-07 (`PExFunc` with registered ∇), and DRIV-02 (dead `diagnostics_helpers.jl` deleted).

---

## Part B — Convergence results vs theoretical expectations

> This is the part where the user's intuition ("the clues are in the results") pays off. **Do not trust
> the summary docs** — every number below was recomputed from the raw error arrays, and the headline
> conclusions of the existing 3D investigation are re-tested against a clean control mesh.

### B.0 2D (k=1 QUAD) is honest and optimal — recomputed

Recomputing finest-pair rates straight from
`previous_results/validated_k1_quad_N640/phase1_quad_k1.h5` (48 configs): **every** cell has L²u ≥ 1.93
(median 2.03), H¹u ≥ 1.00 (median 1.01), L²p median 2.01. Zero sub-optimal at the finest pair. The
HDF5-stored `rate_u_l2` attribute is a *global* least-squares fit (≈1.7–1.96, dragged down by coarse
meshes) and is systematically lower than the finest-pair rate — so a report's headline depends on which
estimator it cites — but the method is genuinely optimal. **The 2D story is sound; the problem is
3D-specific.** (Minor reporting note: state which rate estimator a table uses.)

### B.1 [HIGH] 3D P1-ASGS is sub-optimal even on a perfect uniform mesh — method-intrinsic, not mesh quality

**The control experiment.** The structured Kuhn simplex mesh has uniform, refinement-invariant element
quality (constant inverse-estimate constant `C_inv` across levels), so it removes "mesh quality" as a
variable. Driving the *same* paper §5.2 case (Deviatoric, Constant-σ, Re=Da=1, α₀=0.5, paper c₁) through
the *same* `solve_one` the committed sweeps use:

| method | h-pair | L²u rate | H¹u rate | L²p rate |
|---|---|---|---|---|
| **P1 ASGS** | 0.149→0.099 | 1.24 | 0.92 | 0.84 |
| | 0.099→0.075 | 1.06 | 0.67 | 0.83 |
| | 0.075→0.060 | 1.27 | 0.88 | 0.95 |
| **P1 OSGS** | 0.149→0.099 | **2.21** | **0.99** | 1.66 |
| | 0.099→0.075 | **1.77** | 0.85 | 1.93 |

(Optimal: L²u = 2, H¹u = 1, L²p = 1.) On the **identical mesh and c₁**, ASGS sits at L²u ≈ 1.2 across
all three pairs while OSGS is at 2.21. ASGS's H¹u (≈ 0.9) is near-optimal — so the deficiency is
specifically the **L²-velocity extra (Aubin–Nitsche) order**, which ASGS fails to achieve in 3D and OSGS
recovers. This reproduces the committed `frontal_c1x1` numbers (recomputed: OSGS L²u ≈ 2.0 optimal,
ASGS L²u ≈ 1.3 sub-optimal) — but the structured-Kuhn control proves the gap **cannot be a mesh-quality
artifact**, because a same-mesh ASGS↔OSGS difference is by construction method-dependent.

> **Prior art / corroboration.** A separate structured-Kuhn control run earlier on 2026-06-24 (recorded
> in project memory and already reflected in [docs/README.md](README.md) line 24, which states the
> control "falsifies the old gmsh mesh-quality hypothesis") reached the same P1 conclusion: ASGS L²u
> ≈ 1.24, OSGS ≈ 1.91. **This run independently reproduces those numbers** (different driver, fresh
> meshes) — strong corroboration — and **completes the P2 case** (B.5), which the prior run could not
> (its P2 coarse solve failed and its fine P2 OOM'd). Note the *canonical* doc
> [3d-p2-convergence-investigation.md](mms/3d-p2-convergence-investigation.md) still leads with the
> mesh-quality hypothesis in its TL;DR, contradicting its own README pointer — that doc needs updating
> (see B.5 recommended action).

**Interpretation.** This is consistent with standard VMS theory: OSGS stabilises only the part of the
strong residual *orthogonal* to the FE space, removing the FE-projectable "consistency error" that ASGS's
full-residual stabilisation retains. That retained error is adjoint-inconsistent and caps the L²
superconvergence at the H¹ order. In 2D the codebase shows ASGS *is* optimal (B.0, median L²u 2.00); the
3D loss appears tied to the genuinely-3D tetrahedral discretisation of the (z-extruded) solution and the
deviatoric trace coupling — but the precise mechanism is secondary to the reproducible empirical fact.

**Why this matters for the existing narrative.** `docs/mms/3d-p2-convergence-investigation.md` and the
project memory frame 3D sub-optimality as *mesh quality / `C_inv` vs the c₁=4k⁴ budget*, "fixed by the
Frontal mesh." That framing is correct **for OSGS** (a better mesh lifts OSGS to optimal) but **wrong for
ASGS**: the ASGS L²-order loss is method-intrinsic and survives a perfect uniform mesh. Reports that
present "3D P1 optimal" should qualify it as *OSGS-optimal, ASGS-suboptimal*.

**Re-confirmation (2026-07-04, ratio-to-interpolant — the cleaner discriminator).** The ASGS-vs-OSGS
comparison above is strengthened by measuring each against the *nodal interpolant* (the best the FE space
allows) on the structured Kuhn ladder:

| method | ratio→interp: (8,8,2)→(16,16,4)→(24,24,6) | L²u rate vs interp rate |
|---|---|---|
| **P1 ASGS** | 1.47 → 2.40 → **3.03** (grows) | 1.2 vs interp **1.85** |
| **P1 OSGS** | 0.93 → 0.84 (pinned ~1) | tracks interp |

This is decisive on two fronts. (i) The *interpolant itself* only reaches rate ≈1.85 (not 2) at these
coarse meshes (the α-annulus preasymptotics — same effect as P2), so the raw ASGS <2 rate is **not all
defect**; but (ii) ASGS lags *even the preasymptotic interpolant* and the gap **widens** (ratio 1.47→3.03),
while OSGS *pins* at the interpolant (~0.9) on the **same mesh**. So the space demonstrably admits the
optimal order (OSGS gets it) and ASGS specifically loses it — a genuine L²-order (Aubin–Nitsche) defect,
not a mesh or preasymptotic artifact. (The earlier ASGS-vs-OSGS framing understated the preasymptotic
share; ratio-to-interpolant separates the two cleanly.)

**Recommended action.** (1) Document the ASGS 3D L²-order deficiency honestly as a method property, not a
mesh defect. (2) Treat OSGS as the load-bearing 3D method (as the code already leans). (3) If ASGS 3D
optimality is wanted, that is a genuine formulation research question (the orthogonal-projection
consistency term), worth a paper footnote — not a c₁ tuning.

> **Solver note (same runs).** Both P1-OSGS control solves hit the **iteration cap (20)** and were
> accepted as success via the OSGS soft-stall policy (`accept_soft_stall=true`,
> [osgs_solver.jl:28](src/solvers/osgs_solver.jl#L28)), *not* by reaching the ε_M/ε_C gate. The accepted
> iterate is nonetheless optimal-rate and more accurate than ASGS — but "success=True" here means
> "budget-exhausted soft stall accepted," not "ε-converged." This is the documented OSGS linear-rate
> coupling (the JFNK motivation), but it means the OSGS success flag is weaker than it looks and the
> iteration cap (a config value) is silently load-bearing for OSGS accuracy.
>
> **[Update 2026-07-01 — Route B + JFNK make this success honest, preliminary.]** The 2026-06-24 control
> above predates both the JFNK recipe and Route B. Re-running the structured-Kuhn sweep with `osgs_jfnk`
> **and** the Route-B algebraic mass gate, the first 3D OSGS-P1 cell reaches `success=true` via
> `ftol_reached` on **both** gates (`ε_M 1.8e-2→3e-10`, `ε_C 0.97→6.2e-9`) — i.e. genuinely ε-converged,
> **not** a budget-exhausted soft stall. So the "success flag is weaker than it looks" concern is being
> retired for the JFNK+Route-B path: the mass gate now forces an honest verdict (converge the pressure
> block or fail), and the same cell's L²p error dropped ≈0.83% (the soft-stall under-convergence tail,
> now drained). Full-sweep confirmation is the F4 measurement follow-up.

### B.2 [HIGH] The committed "3D-P2 divergence" is dominated by `success=False` solves plotted as valid data

Recomputing the committed P2 rates straight from
`previous_results/convergence3d/convergence3d_results_frontal_c1x1_20260623.json` (Frontal mesh, paper
c₁):

| level | h | success | L²u | H¹u | L²p |
|---|---|---|---|---|---|
| 0 | 0.123 | **True** | 0.0272 | 0.905 | 0.268 |
| 1 | 0.094 | **True** | 0.0123 | 0.501 | 0.126 |
| 2 | 0.078 | **False** | 0.0620 | 2.863 | 0.656 |

The level 0→1 rates (both `success=True`) are **L²u = 2.92, H¹u = 2.17, L²p = 2.77 — fully optimal for
P2** (opt 3/2/2). The "divergence" is entirely the level-2 step, which **did not converge**
(`success=False`) and whose error values (0.062, 2.863) are whatever the failed solve happened to stop
at. The `pre_frontal` (Delaunay) file is the same story inverted: its level-1 solve *fails* and spikes,
while the *finer* level-2 solve (`success=True`) recovers to L²u 0.0062.

**The reporting does not gate on `success`.** [plot_convergence3d.py](test/extended/ManufacturedSolutions3D/plot_convergence3d.py)
`_plot_by_kv` reads the top-level `hs`/`l2u`/`h1u`/`l2p` arrays and `_seg_slope` computes a slope across
*every* consecutive pair, with no `success` filter; the top-level error arrays in the JSON drop the
per-level `success` flag (kept only in `levels[]`). So a failed solve is plotted as a data point and its
wild slope is annotated as a convergence rate. The "P2 diverges under refinement" headline is largely an
artifact of mixing non-converged solves into the rate table.

This does **not** fully exonerate the formulation. The committed files conflate two effects: (i) the
failed-solve contamination above, and (ii) a **genuine** under-stabilization defect. The "optimal"
frontal pair (level 0→1, L²u ≈ 2.9) is at *coarse* h (0.12→0.094); the clean-mesh control (B.5) goes
finer and shows the P2 error **turning around and growing** on converged fine pairs — the classic
loss-of-coercivity signature (error optimal-looking at coarse h, then diverging as h→0). So the right
reading is: **the reporting overstates the divergence (failed solves), but a real P2-3D coercivity defect
exists underneath it** — quantified and resolved in **B.5**. The fix is the c₁ budget, not the mesh.

**Recommended action.** (1) Make the plotter and any rate computation **exclude or visibly flag**
`success=False` levels (the data is in `levels[]`). (2) Re-derive the "errors grow under refinement"
claim using only converged levels on a fixed-quality mesh before attributing it to coercivity. (3) Never
compute a convergence rate across a non-converged solve.

### B.3 [Med] ASGS and OSGS report byte-identical errors at every shared-failed level

Across both committed P2 files, the *only* levels where ASGS and OSGS agree to all 16 digits are exactly
the levels where **both** report `success=False`:

```
frontal   lvl2: ASGS(False, l2u=0.06198525425452791) == OSGS(False, l2u=0.06198525425452791)
pre_frontal lvl1: ASGS(False, l2u=0.06196690717763982) == OSGS(False, l2u=0.06196690717763982)
```

At every mixed- or both-success level the two methods differ (as independent solves must). Two
independent nonlinear solves with different stabilisation cannot produce bit-identical errors unless they
returned the *same field* — i.e. a shared failed/fallback state (most likely a degenerate independent
remesh at that level on which both solves fail identically, before the method-specific terms can
diverge), or a data-handling artifact. Either way the failed-level numbers are doubly untrustworthy.

**Recommended action.** Add an assertion/diagnostic that flags identical ASGS/OSGS error tuples as a
red flag; surface the per-level mesh quality (min dihedral/radius-ratio) so a bad remesh is visible; and
(with B.2) exclude these levels from the rate table.

### B.4 [Med] The most recent committed 3D results are not reproducible from the current harness

The committed `convergence3d_results_frontal_c1x1_20260623.json` records
`mesh_algorithm = "gmsh_Frontal_alg4_independent_remesh"` and a 6-level P1 ladder, but **no function now
in `smoke3d.jl` produces that string or that ladder**: the sweep paths use `build_nested_family` and
write `mesh="nested_red"`; `build_sequence("frontal")` ([mesh3d.jl](test/extended/ManufacturedSolutions3D/mesh3d.jl))
defines only three lcs `[0.137,0.098,0.070]`. All of `mesh3d.jl`, `smoke3d.jl`, `plot_convergence3d.py`
are uncommitted-modified in the working tree. So the exact driver that produced the most recent committed
numbers is not recoverable from `HEAD`+worktree — a violation of the project's own
parameters→results-traceability rule.

A related minor provenance smell: `smoke3d.jl`'s `build_config` sets `physical_properties.eps_val = 1e-8`,
but `solve_one` builds the formulation with `eps_phys = 0.0` (default), so the formulation's ε is 0 and
the config's `1e-8` is dead for the solve — the stored config value is not the one used.

**Recommended action.** Restore/commit the exact driver (the Frontal independent-remesh sweep function)
that produced the committed 3D files, or re-run them with a committed, named function and archive the
config snapshot alongside, per `reproducible-results.md`. Make `build_config`'s `eps_val` and the
formulation's `eps_phys` agree (or document why they differ).

### B.5 [HIGH] P2-3D at paper c₁ has a genuine under-stabilization defect (confirmed on a clean uniform mesh) — and it is **not** mesh quality; c₁×4 fixes it

The structured-Kuhn control (uniform, refinement-invariant quality) ran P2 ASGS+OSGS at paper c₁ and
ASGS at c₁×4. Results (paper §5.2 case, all on the same mesh ladder):

**P2 ASGS, paper c₁ (=4k⁴=64):**

| h | success | L²u | H¹u | L²p |
|---|---|---|---|---|
| 0.149 | **False** | 9.58e-2 | 2.341 | 0.557 |
| 0.099 | True | 4.93e-2 | 1.816 | 0.464 |
| 0.075 | True | 1.09e-2 | 0.562 | 0.106 |
| 0.060 | True | **1.31e-2** | **0.871** | **0.136** |

The finest **two `success=True`** levels (0.075→0.060) give rates **L²u = −0.82, H¹u = −1.96,
L²p = −1.14 — the errors GROW under refinement, on converged solves, on a perfect uniform mesh.** Most
coarse P2 solves at paper c₁ also fail outright (`linesearch_failed` / `no_progress_stall`). So the
P2-3D defect is **real**, not merely the B.2 reporting artifact.

**P2 ASGS, c₁×4 (=256):** every level `success=True`, errors decrease monotonically, rates
**L²u = +1.67, +2.67, +2.30** (climbing toward the optimal 3), H¹u → +1.26, L²p → +1.48 — and the
**absolute errors are ≈ 40× smaller** than at paper c₁. The c₁ knob is exactly the coercivity lever
(`c₁ > 2ξ C_inv²`, article.tex `eq:conditions_on_num_param`).

**Masking vs fixing (2026-07-03 mesh-family control against the nodal interpolant).** A single-mesh knee is
ambiguous — c₁×2 already looks healed at (12,12,3). Re-testing at *fixed* c₁ across the Kuhn ladder against
the nodal-interpolant error (the best the FE space can do) separates a genuine coercivity fix (ratio-to-
interpolant *pinned* at ~1) from a mere error-constant shrink (ratio *drifts*):

| c₁ mult | ratio→interp across (12,12,3)→(16,16,4)→(20,20,5) | L²u rate | H¹u | verdict |
|---|---|---|---|---|
| **×2** | 1.34 → 1.51 → **1.94** (drifts up) | 2.19 → **1.13** (falls off) | **stalls** (≈0) | **MASKING** |
| **×4** | 1.14 → 1.09 → 1.07 → **1.06** (→1) | tracks interp to 2 dp | converges | **FIXING** |

So c₁×2 heals the coarse mesh then *re-diverges* under refinement; only c₁×4 pins. The h-robust coercivity
threshold is therefore **c₁\* ∈ ×(2,4)** — higher than the clean-room report's single-mesh ×(1.5,2)
estimate, which its own §5.3 discriminator (ratio-pinning across a family) now corrects.

**OSGS at c₁×4 is method-independent.** The full k2 sweep also ran OSGS at c₁×4 (3D recipe: boot-skip +
matrix-free JFNK). It converges (`ok=true`, `eps_used=1`) with ratio-to-interpolant 1.08 → 1.04 and rates
matching ASGS. So the c₁ *discretisation* fix is method-independent — and OSGS-P2-3D's separate
non-contractive-π solver issue manifests here as **cost** (~1–2 h/mesh of JFNK grinding), **not failure**,
a materially more positive result than "OSGS-P2 unsolvable."

**Clean-room corroboration (now committed).** An independent NumPy/SciPy reimplementation reproduces the
same 40× collapse and the same knee and attributes it to the element-family inverse-inequality constant
`C_inv` (paper `4k⁴` calibrated for quads/hexes, sub-critical on right-angle Kuhn tets). It is committed at
[`docs/convergence_problems_audit/`](docs/convergence_problems_audit/) (`b20cb78`, report + NumPy
reproducer) and cross-linked to the canonical
[`docs/mms/3d-p2-instability-investigation.md`](docs/mms/3d-p2-instability-investigation.md) as
**contested-pending-in-stack-confirmation** — which the §5.1 Gridap sweep above now provides, refuting that
doc's TL;DR #2 "not c₁."

**Verdict — what is right and wrong in the existing 3D-P2 narrative.**

1. ✅ **The c₁/coercivity hypothesis is correct.** Scaling c₁ up is the single lever that turns P2-3D from
   diverging-and-failing to converging-and-optimal, exactly as `docs/mms/3d-p2-convergence-investigation.md`
   claims, and it matches the paper's own remark that the effective c₁ is element-type dependent.
2. ❌ **It is NOT "mesh quality" / a bad-element tail.** The structured Kuhn mesh has uniform,
   refinement-invariant quality, yet P2 still fails / grows at paper c₁ and is fixed by c₁×4. So the
   deficiency is that **c₁ = 4k⁴ is structurally too small for P2 tetrahedra in general** (the geometric
   part of `C_inv` for tets — even uniform ones — exceeds the `4k⁴` budget at k=2 in 3D), *not* that the
   gmsh generator produces an occasional sliver. The "Frontal-mesh fix" helps P1-OSGS, but it cannot be
   the explanation for P2, which fails on a flawless mesh.
3. ⚠️ **The committed "divergence" tables conflate two things** (B.2): a genuine under-stabilization
   signal *and* failed-solve reporting contamination. The control separates them: the genuine signal is
   the growing error on *converged* fine pairs; the contamination is the `success=False` spikes.
4. ⚠️ **The P2 solve failures at paper c₁ are a symptom, not a separate solver bug.** Under-stabilization
   (too-small c₁) leaves the discrete operator poorly conditioned/near-non-coercive, so Newton's line
   search depletes. Fixing c₁ fixes the solver failures too (every c₁×4 solve converges).

**Recommended action.** (1) Adopt an **element-type-aware c₁** for P2 tetrahedra — a config-driven
`c1_multiplier` (or a per-(element, order) table), explicitly labelled `[code-actual]` as a deviation
from the paper's uniform `4k⁴`, with 2D/quad staying paper-faithful. This is the principled fix the
investigation doc already floats as option (3); the control shows it is *necessary*, not a band-aid.
(2) **Reconcile the canonical doc with its own README pointer:** `docs/README.md` (line 24) already
states the structured-Kuhn control *falsifies* the mesh-quality hypothesis, but
`docs/mms/3d-p2-convergence-investigation.md` still leads with "mesh quality / `C_inv` tail" in its TL;DR
(its 2026-06-21 body predates the control). Update that doc's TL;DR to the **c₁-budget-vs-tet-`C_inv`**
verdict (reproduced on a uniform mesh, *not* a bad-element tail), with this run's clean P2 c₁×1-vs-c₁×4
numbers as the evidence. (3) Measure the actual `C_inv` for P2 Kuhn vs P2 gmsh tets to calibrate the
multiplier.

> **Run log:** `test/extended/ManufacturedSolutions3D/results/debug_results/audit_3d_structured.log`
> (full P1/P2, ASGS/OSGS, c₁×1 and c₁×4; ~50 min wall). Driver: `/tmp/audit_3d_structured.jl`.

### B.6 [Med] A failed OSGS solve silently reports the ASGS Stage-I state under the OSGS label

The control reproduced the byte-identical anomaly (B.3) **and revealed its mechanism.** Every P2-OSGS
solve at paper c₁ failed (`success=False`) and reported an error **bit-identical to the standalone
P2-ASGS run at the same level** — including at (20,20,5) where ASGS *succeeded* (1.3101e-2) and OSGS
*failed* but still reported 1.3101e-2. Because the OSGS path runs the **ASGS Stage-I boot first**
([solver_core.jl:511](src/solvers/solver_core.jl#L511)) and the coupled solve, on failure, leaves the
iterate at that ASGS state, **a non-converging OSGS solve degenerates to ASGS and reports ASGS's error
under the OSGS label.** [osgs_solver.jl:163-168](src/solvers/osgs_solver.jl#L163) already flags this exact
failure mode as a `[known-fragility]` for the stall sensor; the control shows it also happens via
`linesearch_failed`. When OSGS *does* converge (all P1 cases), it produces genuinely different, more
accurate results — so the degeneration is silent and only bites on the hard (P2) cases that most need
OSGS.

**Recommended action.** When the OSGS coupled solve does not reach its own convergence verdict, mark the
result distinctly (e.g. `method="OSGS(degenerated→ASGS)"` or a separate flag) so a sweep cannot record an
ASGS error in an OSGS column. Combined with B.2/B.3, this stops the failed-solve numbers from masquerading
as method comparisons.


---

## Appendix 1 — Control experiment: full 3D structured-Kuhn data

Paper §5.2 case (Deviatoric, Constant-σ, Re=Da=1, α₀=0.5), structured Kuhn simplex mesh, domain
(0,1)×(0,1)×(0,0.3). Per-segment rates between consecutive levels (optimal: L²u=k+1, H¹u=k, L²p=k).
`†` = a segment that includes a `success=False` solve (rate is meaningless).

| case | h-pairs | L²u rates | H¹u rates | L²p rates | verdict |
|---|---|---|---|---|---|
| P1 ASGS c₁×1 | 0.149/0.099/0.075/0.060 | 1.24, 1.06, 1.27 | 0.92, 0.67, 0.88 | 0.84, 0.83, 0.95 | **sub-optimal** (L²u≈1.2) |
| P1 OSGS c₁×1 | same | 2.20, 1.78, 2.08 | 0.99, 0.86, 0.98 | 1.67, 1.93, 1.60 | **optimal** |
| P2 ASGS c₁×1 | same | 1.64†, 5.24, **−0.82** | 0.63†, 4.08, **−1.96** | 0.45†, 5.14, **−1.14** | **defect** (errors grow on converged fine pair; coarse solves fail) |
| P2 OSGS c₁×1 | same | all †† | all †† | all †† | all solves failed → reports ASGS state (B.6) |
| P2 ASGS c₁×4 | same | 1.67, 2.67, 2.30 | 0.83, 1.76, 1.26 | 0.94, 1.73, 1.48 | **converges, climbing to optimal**; ≈40× smaller errors |

Headline reads: P1 ASGS sub-optimal vs P1 OSGS optimal on the *same* mesh (B.1); P2 at paper c₁ has a
genuine coercivity defect on a *uniform* mesh, fixed by c₁×4 (B.5); a failed OSGS solve reports the ASGS
state (B.6).

## Appendix 2 — Index of the still-open code findings (by domain)

Each was confirmed by an independent code-grounding skeptic and re-checked against the theory. The
**resolved** findings (FORM-01/02/03/05, TAU-01/05, VISC-01/02/03, MODE-01/02/03, DRIV-02/07/08,
NONL-01/02/03/05, PROJ-01, SOLV-04) have moved to the §0 Resolved ledger; only the open ones remain below.

**convergence-criterion**
- `CONV-05` (fragility, medium): MMS verifier numerical parameters (tau_err, eps_*, max_extra_cycles, require_consecutive_passes, rate_check_factor) are hard-coded — `test/extended/CocquetFormMMS/run_test.jl:401-412`
- `CONV-02` (inconsistency, low): Inline literal `1e-2` √d self-check margin in evaluate_convergence violates the repo no-magic-numbers / config-strictness rule — `src/solvers/convergence_criterion.jl:267` (moved by the Route-B rewrite; now on the `eps_C_strong`/`div_ratio` DIAGNOSTIC path, not the gate)
- `CONV-04` (fragility, low): Sub-optimal-rate budget uses a bare power of h with NO reference-error / leading-constant normalization, so the rate-check is scale-dependent — `src/solvers/mms_verification.jl:130-131`

**driver-mms-io**
- `DRIV-03` (inconsistency, low): JSON schema declares no `required` fields for physical_properties/solver/etc., so it does not enforce the no-implicit-defaults rule (= A.4) — `config/porous_ns.schema.json`
- `DRIV-05` (inconsistency, low): Production export_results writes no provenance (Re/Da/α₀/params) into or alongside the VTK output — `src/io.jl:28-51`

**formulation-core**
- `FORM-04` (fragility, low): τ rational/Forchheimer |u| nonlinearity is under-integrated by the polynomial quadrature rule (inexact) — `src/formulations/continuous_problem.jl` (quadrature-degree helpers)

**models**
- `MODE-04` (fragility, low): Porosity logistic saturation guard uses hard-coded ±100 thresholds (magic numbers) and α is silently 2D-only — `src/models/porosity.jl:70-74`

**nonlinear-safeguards**
- `NONL-04` (fragility, low): Anderson update! has no zero/near-zero residual guard before solving the least-squares system (now live behind the opt-in `osgs_anderson_enabled`) — `src/solvers/accelerators.jl:53-127`
- `NONL-06` (fragility, low): Picard line-search reuses Armijo c1 as a multiplicative residual-reduction factor (1 - c1·α), a different mathematical role — `src/solvers/nonlinear.jl:434-438`

**projection**
- `PROJ-02` (simplification, low): ProjectResidualWithoutPressurePenalty is never instantiated in production code — dead policy reachable only from tests — `src/stabilization/projection.jl:34,121-143`

**solver-orchestration**
- `SOLV-03` (inconsistency, low): One-way ASGS Picard success uses scalar ℓ∞ ftol, not the scale-free cascade_step_outcome the ping-pong path and Newton stage use — `src/solvers/asgs_solver.jl:144`
- `SOLV-05` (fragility, low): discrete_l2_projection re-solves a fresh allocate_in_domain RHS each residual eval; b_vec/x_solve scratch is re-allocated — `src/solvers/osgs_solver.jl:59-64`

**tau**
- `TAU-02` (fragility, low): σ inside τ₁ is evaluated at effective_speed (mesh-dependent diffusive floor), not at the physical reaction_speed used in the residual — `src/stabilization/tau.jl` (Tau1Op). NOTE: re-check vs `docs/solver/paper-code-divergences.md` — the σ-in-τ₁ choice may be intentional.

**Refuted / trivial (19, not carried):** the second-pass verifiers rejected, among others: "all-Dirichlet
pressure block is singular" (refuted — the τ₂/εp stabilisation regularises it), "homotopy dilution changes
the converged solution" (refuted — `mult_mom/mult_mass` are always 1.0 in the core), "Da=0 mis-sizes the
Forchheimer MMS pressure scale" (refuted — the MMS sweeps use Constant-σ, which *does* expose Da),
"`base_config.json` missing `eps_val`" (already documented), and several `[trivial]` cosmetics
(unused imports, the 2D `0.0*grad_div` multiply, the ρ_{Λ⁻¹}(σ)→scalar simplification).

## Appendix 3 — Method & reproducibility

- Multi-agent audit run record (53 findings, per-finding two-lens verdicts): workflow `wf_9dc5a56d-800`.
- 2D rate recomputation: `previous_results/validated_k1_quad_N640/phase1_quad_k1.h5` (48 configs).
- 3D committed-data reanalysis: `previous_results/convergence3d/convergence3d_results_{frontal_c1x1,pre_frontal}_*.json`.
- 3D control experiment: driver `/tmp/audit_3d_structured.jl`, log
  `test/extended/ManufacturedSolutions3D/results/debug_results/audit_3d_structured.log`.
- Every code `file:line` in this document was opened and read; every results claim was recomputed from
  raw error arrays, per the "do not trust the reports" directive.
