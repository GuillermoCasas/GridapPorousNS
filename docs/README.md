# `docs/` — meta-documentation (observations, status, references)

All non-LaTeX documentation lives here. The LaTeX sources are in [`../theory/`](../theory/) (see
[`../theory/README.md`](../theory/README.md)).

## ⭐ Living docs — START HERE (the current state, kept up to date)

Consolidated 2026-07-08 into a small purpose-organized set. These four answer *what do we know? what's
unresolved? what's next? how does the code map to the paper?* — read these first; the by-topic
investigation records below are the **detailed evidence** behind them.

- **[findings.md](findings.md)** — *Results & debugging conclusions*: the settled state per area (2D MMS,
  the low-α fold, 3D-P2, OSGS reaction rate, JFNK/Anderson, CocquetFormMMS) with the canonical numbers.
- **[open-questions.md](open-questions.md)** — *Open theoretical/numerical questions*: the α=0.1 fold
  mechanism (σ̃_α), the Gridap↔Kratos magnitude offset, the 3D-P2 caveats, OSGS-P2 coupling, paper errata.
- **[pending-tasks.md](pending-tasks.md)** — *Backlog*: sweeps to run, tests to promote, solver
  efficiency ideas, open code-correctness issues.
- **[theory-code-map.md](theory-code-map.md)** — *Theory↔code reference*: the paper↔code divergence
  ledger, the algorithm→file map, the stabilization/τ design, the convergence-gate spec.

Plus the two standing living ledgers: **[lessons_learned.md](lessons_learned.md)** (regression history,
the authority on anything "fixed") and **[known_issues.md](known_issues.md)** (open code-correctness).

> The by-topic docs below (`mms/`, `cocquet/`, `solver/`) remain as the **detailed investigation
> records / evidence** the living docs consolidate and link into. (A follow-up pass may move them under
> `docs/evidence/` or prune those fully subsumed — see the doc-consolidation note in the git log.)

## Top level — repo-wide ledgers

- [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) — **deep theory↔code + results
  audit (2026-06-24/25).** Verdict: the continuous VMS formulation is faithfully transcribed (no
  weak-form correctness bug). The substantive findings are (A) two doc↔code contract drifts — the dynamic
  Re/Da budget knobs have no `src/` consumer, and the scale-free ε_M/ε_C gate is the *authoritative*
  production gate despite "not yet wired in / diagnostic only" comments; (B) the results story — 3D
  P1-ASGS is method-intrinsically sub-optimal and Gridap **under-stabilises P2-3D relative to the paper**
  *on a perfect uniform mesh* (c₁×4 *masks* it ~40×, but c₁ is a SYMPTOM — paper c₁ is CORRECT per the first
  author / Kratos full-terms; root cause OPEN, a Gridap↔paper discrepancy), the committed "P2 divergence"
  tables mix in `success=False` solves, and a failed OSGS solve silently reports the ASGS state; (C/D) ILU silent
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
| **2D — canonical** | [convergence-2d.md](mms/convergence-2d.md) | The settled 2D case (consolidated 2026-07-09). Sweep status (k1 & k2 QUAD complete N=10→640, fully optimal — velocity L² `O(h^{k+1})`/H¹ `O(h^k)`, pressure ≥ nominal, OSGS ≈2× more accurate than ASGS); the scale-free **convergence gate** (Route-B algebraic `ε_C=‖r_C‖/D_C`, the k≥2 `eps_tol_momentum=1e-9` lesson, the `residual_floor_reached` valve); the **stiff-corner fold** (discrete turning-point, recedes with mesh, α-continuation vs the production direct-solve); key learnings + the reusable gate-vs-solver diagnostic recipe; and JFNK/Anderson (OSGS iteration cost). |
| **2D — provenance** | [convergence-baseline.md](mms/convergence-baseline.md) | Frozen per-cell **N≤320 reference snapshot** of the k=1 QUAD sweep (H¹ rates + inner-iteration costs) + the 2026-06-10 N=640-recovery note. Kept as the provenance record (per the reproducibility rule); live headline numbers are in [findings.md](findings.md). |
| **3D-P2 — canonical** | [p2-3d.md](mms/p2-3d.md) | The 3D-P2 cluster (consolidated 2026-07-09). **(A)** the converged-but-wrong *accuracy* defect — leading verdict is a coercivity margin (`4k⁴` under-margined for high-`C_inv` structured Kuhn tets: `C_inv²` 214 vs quad 60, mesh-indep; Céa 50×-error ⇒ coercivity≈0; the `c1_mult=4` monotone ladder; remedy = element-aware c₁ per line 910) — **with the honest caveats preserved** (Reading A/B undistinguished, Kratos solves optimally at paper c₁, ξ unpinned → a term-level Gridap↔paper discrepancy is *not* 100% excluded); plus the refuted "do-not-rechase" list. **(B)** the iterative-penalty *well-posedness* fix (ε=0 all-Dirichlet is ill-posed). **(C)** the OSGS **∂π/∂u** coupling fix (ρ_prec 1249→3.8 via `osgs_jfnk_precond_c1_mult`, 2026-07-09, solution-preserving). |
| Clean-room diagnosis | [convergence_problems_audit/files/p2_3d_diagnosis_report.md](convergence_problems_audit/files/p2_3d_diagnosis_report.md) | Independent NumPy/SciPy reimplementation — its **c₁ element-family verdict is REFUTED** (it was transcribed term-by-term from `continuous_problem.jl`, so it inherited the same Gridap↔paper discrepancy; the "C_inv exceeds 4k⁴" argument is circular). Kept for provenance + the code-transcription it encodes; the reproducer is still a live artifact. |

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
| Archived plans (removed) | — | The `solver/archive/` implemented-proposal drafts (ASGS/OSGS file split + MMS-decoupling seam, shipped 2026-07-04) were **removed 2026-07-08** as fully implemented. The realized design is [algorithm-code-mapping.md](solver/algorithm-code-mapping.md). |
| **Canonical** | [nonlinear-convergence-criterion-prompt.md](solver/nonlinear-convergence-criterion-prompt.md) | Spec for the scale-free outer-loop stopping criterion (`convergence_criterion.jl`): converged ⇔ ε_M ≤ tol_M and ε_C ≤ tol_C, ε_M = ‖r_M‖/D_M. ⚠️ **Mass measure updated (Route B):** ε_C is now the Philosophy-A **algebraic** ‖r_C‖/D_C (symmetric with ε_M); the strong-form/flux-gradient measure in the spec is the diagnostic `eps_C_strong` — see [mms/convergence-2d.md](mms/convergence-2d.md). No a-priori `U`/`L`/`P`/`Re`/`Da` scales. |
| Backlog | [efficiency-ideas.md](solver/efficiency-ideas.md) | Newton/Picard scheduling & convergence-gate proposals (2026-06-04; the ping-pong cascade has since landed). |
| Audit | [normalization-audit.md](solver/normalization-audit.md) | Dimensional/scaling audit of every solver convergence/drift/divergence gate (encoding-invariance, P5 2026-06-04). |

## `paper/` — about the SIAM article

- [errata.md](paper/errata.md) — open editorial items for `../theory/paper/article.tex` needing author judgment.
