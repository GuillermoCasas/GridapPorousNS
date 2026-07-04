# `docs/` — meta-documentation (observations, status, references)

All non-LaTeX documentation lives here, organized by topic. The LaTeX sources are in
[`../theory/`](../theory/) (see [`../theory/README.md`](../theory/README.md)). Each topic has **one
canonical doc**; the rest are kept for traceability and carry a status header pointing to it.

## Top level — repo-wide ledgers

- [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) — **deep theory↔code + results
  audit (2026-06-24/25).** Verdict: the continuous VMS formulation is faithfully transcribed (no
  weak-form correctness bug). The substantive findings are (A) two doc↔code contract drifts — the dynamic
  Re/Da budget knobs have no `src/` consumer, and the scale-free ε_M/ε_C gate is the *authoritative*
  production gate despite "not yet wired in / diagnostic only" comments; (B) the results story — 3D
  P1-ASGS is method-intrinsically sub-optimal and P2 is under-stabilised at paper c₁ *on a perfect
  uniform mesh* (c₁×4 fixes P2; ≈40× smaller errors), the committed "P2 divergence" tables mix in
  `success=False` solves, and a failed OSGS solve silently reports the ASGS state; (C/D) ILU silent
  identity fallback, missing config validation, a copy-pasted `1e-12` floor, and several dead-code /
  duplication cleanups. Corroborates the structured-Kuhn finding below and completes the P2 case.
- [`lessons_learned.md`](lessons_learned.md) — append-only ledger of past regressions and their
  canonical fixes. The authority on anything marked "fixed". Read it before touching
  `src/formulations/`, `src/stabilization/`, or `src/solvers/`.
- [`known_issues.md`](known_issues.md) — open code-correctness issues (report-only). Still open: the
  schema/loader **method-enum drift** and the **CocquetFormMMS hardcoded-ASGS** dispatch. Now **resolved**:
  the `cfg.phys.f_x` crash, the `base_config.json` missing-field (renamed `eps_val`→`physical_epsilon`,
  2026-07-04), and the high-Da OSGS rate (pre-asymptotic, recovers at N=640) — each kept with its banner.

## `mms/` — Manufactured-Solution convergence

| Role | Doc | Notes |
|---|---|---|
| **Canonical** | [convergence-status.md](mms/convergence-status.md) | Grid-wide status & knowledge of the `(Re, Da, α₀, h)` sweep. **2026-06-10: the k=1 QUAD sweep is complete (N=10→640) and a success** — velocity optimal across the whole grid (L² O(h²), H¹ O(h)), pressure optimal everywhere too (≥ its nominal O(h^{kp})=O(h) equal-order order; super-optimal at 1.5–2.4×), the high-Da OSGS rate confirmed pre-asymptotic (recovers to ≥1.0 at N=640), OSGS ≈2× more accurate than ASGS. See its success box. |
| **Reference** | [convergence-baseline.md](mms/convergence-baseline.md) | Per-cell baseline (H¹ rates + inner-iteration costs). The N≤320 table is the pre-asymptotic reference; the 2026-06-10 note records the N=640 recovery. Remaining tracked target: JFNK = cut OSGS iters (the high-Da rate target is **met** — recovers). |
| Sub-topic | [fold-recovery.md](mms/fold-recovery.md) | Stiff-corner fold diagnosis + continuation rescue (status §3, region B). |
| **2D gate + sweep — CURRENT** | [route-b-2d-sweep-status.md](mms/route-b-2d-sweep-status.md) | **Current status of the 2D mass gate + sweep.** Route B: the mass convergence gate is now the Philosophy-A **algebraic** ε_C=‖r_C‖/D_C (symmetric with ε_M, gated ~1e-9); the old loose `eps_tol_mass=0.8` is demoted to the diagnostic `eps_C_strong`. The full **k1 & k2 QUAD sweeps completed 2026-07-03**, behavior-preserving, via the companion `residual_floor_reached` accept. |
| Gate + JFNK lesson | [high-order-convergence-gate-and-jfnk.md](mms/high-order-convergence-gate-and-jfnk.md) | The k=2 momentum-gate lesson (`eps_tol_momentum=1e-9`) + **JFNK for OSGS is landed & verified** (canonical: [solver/jfnk-phase0-preconditioner-gate.md](solver/jfnk-phase0-preconditioner-gate.md)). |
| **3D-P2 — CANONICAL (RESOLVED)** | [3d-p2-instability-investigation.md](mms/3d-p2-instability-investigation.md) | **2026-07-03: root cause = element-family c₁ coercivity deficit.** P2-3D is "converged-but-wrong" at paper c₁ because `4k⁴` under-budgets `2ξ·C_inv²` for P2 **Kuhn tets**; the mesh-family ratio-to-interpolant test shows **c₁×4 fixes** (ratio pins ~1, ASGS & OSGS) while **c₁×2 masks**. NOT mesh quality / grad-div / inf-sup / Jacobian (all refuted with numbers, §3). The c₁ multiplier is confirmed effective but **not adopted** (root-cause fix preferred). NumPy clean-room: [convergence_problems_audit/](convergence_problems_audit/). OSGS-P2 has a separate ∂π/∂u solver issue (solution accurate but `ok=false`). |
| 3D penalty + OSGS coupling | [3d-iterative-penalty-fix-and-osgs-coupling.md](mms/3d-iterative-penalty-fix-and-osgs-coupling.md) | Canonical for the **iterative-penalty (well-posedness) fix** — 3D all-Dirichlet is ill-posed at ε=0; adds `ε_num·(pⁿ−pⁿ⁻¹)` to the mass residual — plus the eps_pert homotopy and the OSGS **∂π/∂u** coupling problem. ⚠️ Its earlier "P2 root cause = penalty, NOT c₁" claim is **reversed**: the penalty fixes *well-posedness*, not the P2 *accuracy* defect (which is c₁ — see canonical above). |
| 3D ASGS (P₁ & P₂) — superseded | [3d-p2-convergence-investigation.md](mms/3d-p2-convergence-investigation.md) | Superseded as canonical, but its **original c₁ verdict was RE-CONFIRMED (2026-07-03)** — the 2026-06-30 "not c₁" reversal was itself reversed. Still-valid unique content: §4 (iteration/Jacobian/orchestration exonerated) + §6 (OSGS-vs-high-Da reconciliation); also documents the genuine **3D P1-ASGS** L²-order sub-optimality (OSGS recovers it). |
| Clean-room diagnosis | [convergence_problems_audit/files/p2_3d_diagnosis_report.md](convergence_problems_audit/files/p2_3d_diagnosis_report.md) | Independent NumPy/SciPy reimplementation confirming the P2-3D c₁ element-family diagnosis + its reproducer. Evidence bundle for the CANONICAL doc above. |

(The complete, validated k=1 QUAD sweep is the current authority; the old May-2026 partial-sweep
artifacts were removed in the 2026-06-10 cleanup — see
`../test/extended/ManufacturedSolutions/previous_results/README.md`.)

## `cocquet/` — Cocquet benchmark reproduction

| Role | Doc | Notes |
|---|---|---|
| **Canonical** | [investigation-synthesis.md](cocquet/investigation-synthesis.md) | Audited synthesis. **Settled: the convergence cap is mesh-topology** (structured mesh locks the outlet-corner singularity; unstructured Delaunay recovers O(h²)); the residual **magnitude gap is ~30–300×** (coarse→fine) — correcting the earlier "~2×" (an order-of-magnitude misread) and "845×" figures. |
| **MMS sibling** | [cocquet-form-mms-status.md](cocquet/cocquet-form-mms-status.md) | Canonical doc for `CocquetFormMMS` (equal-order VMS vs Taylor–Hood under MMS, Forchheimer reaction). 2026-06-16: moderate-α complete & clean; high-Re×low-α corner **folds for VMS, Taylor–Hood converges**; exact low-α fold cause **open** (σ̃_α hypothesis unconfirmed; ASGS≠OSGS reaction handling). |
| Historical | [convergence-analysis.md](cocquet/convergence-analysis.md) | 12-phase slope investigation (some early phases withdrawn). |
| Artifact | [email-questions.md](cocquet/email-questions.md) | Draft email to the authors (unsent). |
| Transcripts (archived) | [archive/replicating-cocquet-transcript.md](cocquet/archive/replicating-cocquet-transcript.md), [archive/corner-singularity-transcript.md](cocquet/archive/corner-singularity-transcript.md) | Raw session logs, archived 2026-07-04 — verdicts merged into the synthesis; the corner-untag recommendation was falsified (H1). |

## `solver/` — algorithm references & process

| Role | Doc | Notes |
|---|---|---|
| **Canonical ref** | [paper-code-divergences.md](solver/paper-code-divergences.md) | Ledger of every code/paper divergence, classified and justified. Update when implementation diverges from the paper. |
| **Canonical ref** | [algorithm-code-mapping.md](solver/algorithm-code-mapping.md) | 1:1 map from `osgs_algorithm/osgs_algorithm.tex` algorithm boxes to the solver files (`solver_core.jl` shared core + orchestrator, `asgs_solver.jl` ASGS Stage-I, `osgs_solver.jl` coupled solve, `mms_verification.jl` verifier, `nonlinear.jl` kernel). |
| **Canonical** | [osgs-reaction-dominated-rate.md](solver/osgs-reaction-dominated-rate.md) | The high-Da OSGS velocity-rate loss — **RESOLVED 2026-06-10: pre-asymptotic, recovers to ≥1.0 at N=640** (see its banner). Two corrections noted there: the empirical recovery, and the 2026-06-09 theory fix (OSGS retains σ; ASGS drains to σ_a) — so the dip is *not* a coercivity loss. Body TL;DR/mechanism predate both (historical). |
| **Canonical (JFNK)** | [jfnk-phase0-preconditioner-gate.md](solver/jfnk-phase0-preconditioner-gate.md) | JFNK for the OSGS coupled solve: Phase-0 preconditioner gate PASSED and Phase-1 **LANDED** (matrix-free `JFNKLinearSolver`, `osgs_jfnk_enabled`; recovers the dropped ∂π/∂U). Verified Blitz/Quick/Extended. |
| **Canonical (Anderson)** | [osgs-anderson-acceleration.md](solver/osgs-anderson-acceleration.md) | Anderson-accelerated OSGS staggered fixed-point, **landed** behind `osgs_anderson_enabled` (default off). |
| Leaning (history) | [coupled-only-leaning-and-jfnk-plan.md](solver/coupled-only-leaning-and-jfnk-plan.md) | 2026-06-07 leaning: OSGS collapsed to a single `coupled` route (an equivalence oracle proved `freeze_after_k` diverges in the reaction corner). The JFNK plan it floated is **now landed** — see the JFNK row above. |
| Archived plans | `solver/archive/` — [simplification-proposal.md](solver/archive/simplification-proposal.md), [plan-review-and-mms-decoupling.md](solver/archive/plan-review-and-mms-decoupling.md) | Implemented proposals (the ASGS/OSGS file split + MMS-decoupling seam shipped), archived 2026-07-04. Realized design: [algorithm-code-mapping.md](solver/algorithm-code-mapping.md). |
| **Canonical** | [nonlinear-convergence-criterion-prompt.md](solver/nonlinear-convergence-criterion-prompt.md) | Spec for the scale-free outer-loop stopping criterion (`convergence_criterion.jl`): converged ⇔ ε_M ≤ tol_M and ε_C ≤ tol_C, ε_M = ‖r_M‖/D_M. ⚠️ **Mass measure updated (Route B):** ε_C is now the Philosophy-A **algebraic** ‖r_C‖/D_C (symmetric with ε_M); the strong-form/flux-gradient measure in the spec is the diagnostic `eps_C_strong` — see [mms/route-b-2d-sweep-status.md](mms/route-b-2d-sweep-status.md). No a-priori `U`/`L`/`P`/`Re`/`Da` scales. |
| Backlog | [efficiency-ideas.md](solver/efficiency-ideas.md) | Newton/Picard scheduling & convergence-gate proposals (2026-06-04; the ping-pong cascade has since landed). |
| Audit | [normalization-audit.md](solver/normalization-audit.md) | Dimensional/scaling audit of every solver convergence/drift/divergence gate (encoding-invariance, P5 2026-06-04). |

## `paper/` — about the SIAM article

- [errata.md](paper/errata.md) — open editorial items for `../theory/paper/article.tex` needing author judgment.
