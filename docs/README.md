# `docs/` — meta-documentation (observations, status, references)

All non-LaTeX documentation lives here, organized by topic. The LaTeX sources are in
[`../theory/`](../theory/) (see [`../theory/README.md`](../theory/README.md)). Each topic has **one
canonical doc**; the rest are kept for traceability and carry a status header pointing to it.

## Top level — repo-wide ledgers

- [`lessons_learned.md`](lessons_learned.md) — append-only ledger of past regressions and their
  canonical fixes. The authority on anything marked "fixed". Read it before touching
  `src/formulations/`, `src/stabilization/`, or `src/solvers/`.
- [`known_issues.md`](known_issues.md) — open code-correctness issues found in the 2026-06-04 audit
  (report-only): the `cfg.phys.f_x` crash, `base_config.json` missing `eps_val`, schema/loader enum
  drift, and the open OSGS high-Da rate stagnation.

## `mms/` — Manufactured-Solution convergence

| Role | Doc | Notes |
|---|---|---|
| **Canonical** | [convergence-status.md](mms/convergence-status.md) | Grid-wide status & knowledge of the `(Re, Da, α₀, h)` sweep. |
| Sub-topic | [fold-recovery.md](mms/fold-recovery.md) | Stiff-corner fold diagnosis + continuation rescue (status §3, region B). |
| Snapshot | [next-actions.md](mms/next-actions.md) | Dated action list (2026-06-02); its gate/cap "cure" is **superseded** (see `lessons_learned.md` 2026-06-02). |
| Snapshot | [sweep-difficult-cases.md](mms/sweep-difficult-cases.md) | Convergence-difficulty taxonomy from a partial sweep (commit `15e466b`); historical. |

(The raw artifacts of that partial sweep — `.partial-stdout.log`, `.partial.h5` — live under
`../test/extended/ManufacturedSolutions/previous_results/`.)

## `cocquet/` — Cocquet benchmark reproduction

| Role | Doc | Notes |
|---|---|---|
| **Canonical** | [investigation-synthesis.md](cocquet/investigation-synthesis.md) | Audited synthesis; the former magnitude hypothesis-ledger is now its Appendix. Latest mesh-recipe result narrows the gap to ~2×. |
| Historical | [convergence-analysis.md](cocquet/convergence-analysis.md) | 12-phase slope investigation (some early phases withdrawn). |
| Artifact | [email-questions.md](cocquet/email-questions.md) | Draft email to the authors (unsent). |
| Transcripts | [replicating-cocquet-transcript.md](cocquet/replicating-cocquet-transcript.md), [corner-singularity-transcript.md](cocquet/corner-singularity-transcript.md) | Raw investigation conversations. |

## `solver/` — algorithm references & process

| Role | Doc | Notes |
|---|---|---|
| **Canonical ref** | [paper-code-divergences.md](solver/paper-code-divergences.md) | Ledger of every code/paper divergence, classified and justified. Update when implementation diverges from the paper. |
| **Canonical ref** | [algorithm-code-mapping.md](solver/algorithm-code-mapping.md) | 1:1 map from `osgs_algorithm.tex` algorithm boxes to `src/solvers/porous_solver.jl`. |
| Backlog | [efficiency-ideas.md](solver/efficiency-ideas.md) | Newton/Picard scheduling & convergence-gate proposals (2026-06-04). |
| Historical | [algorithm-improvement.md](solver/algorithm-improvement.md) | Consolidated critique → plan → progress (formerly three files). |
| Historical | [refactor-brief.md](solver/refactor-brief.md) | The `solve_system` refactor brief. |
| Transcripts | [code-audit-findings.md](solver/code-audit-findings.md), [reply.md](solver/reply.md) | Raw audit/review conversations. |

## `paper/` — about the SIAM article

- [errata.md](paper/errata.md) — open editorial items for `../theory/paper/article.tex` needing author judgment.
