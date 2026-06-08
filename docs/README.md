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
  drift, and the (now-characterised, not-a-bug) high-Da OSGS rate item.

## `mms/` — Manufactured-Solution convergence

| Role | Doc | Notes |
|---|---|---|
| **Canonical** | [convergence-status.md](mms/convergence-status.md) | Grid-wide status & knowledge of the `(Re, Da, α₀, h)` sweep. (Grid-wide physics is current; the caveat block predates the 2026-06-08 coupled-only leaning — see convergence-baseline.md for current numbers.) |
| **Reference** | [convergence-baseline.md](mms/convergence-baseline.md) | Per-cell baseline (H¹ rates + inner-iteration costs) from the corrected coupled-only solver — the authoritative post-leaning numbers; the reference future changes are measured against (JFNK target = OSGS iters; formulation target = high-Da rate). |
| Sub-topic | [fold-recovery.md](mms/fold-recovery.md) | Stiff-corner fold diagnosis + continuation rescue (status §3, region B). |

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
| **Canonical** | [osgs-reaction-dominated-rate.md](solver/osgs-reaction-dominated-rate.md) | The (now-characterised, not-a-bug) high-Da OSGS velocity-rate loss: the coercivity-gap mechanism, with historical `staggered`-vs-`coupled` evidence that it is the fixed point (`coupled` is now the sole route). (The `freeze_after_k` recommendation here is SUPERSEDED — see the coupled-only-leaning doc below.) |
| **Canonical** | [coupled-only-leaning-and-jfnk-plan.md](solver/coupled-only-leaning-and-jfnk-plan.md) | 2026-06-07 leaning: OSGS collapsed to a single `coupled` route (an equivalence oracle proved `freeze_after_k` diverges in the reaction corner), plus the deferred JFNK plan to recover near-quadratic speed (the dropped ∂π/∂U). |
| Backlog | [efficiency-ideas.md](solver/efficiency-ideas.md) | Newton/Picard scheduling & convergence-gate proposals (2026-06-04; the ping-pong cascade has since landed). |
| Audit | [normalization-audit.md](solver/normalization-audit.md) | Dimensional/scaling audit of every solver convergence/drift/divergence gate (encoding-invariance, P5 2026-06-04). |

## `paper/` — about the SIAM article

- [errata.md](paper/errata.md) — open editorial items for `../theory/paper/article.tex` needing author judgment.
