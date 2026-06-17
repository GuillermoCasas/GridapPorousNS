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
  drift. The high-Da OSGS rate item is now **RESOLVED (2026-06-10)** — pre-asymptotic, recovers to
  ≥1.0 at N=640 (kept there with its resolution banner).

## `mms/` — Manufactured-Solution convergence

| Role | Doc | Notes |
|---|---|---|
| **Canonical** | [convergence-status.md](mms/convergence-status.md) | Grid-wide status & knowledge of the `(Re, Da, α₀, h)` sweep. **2026-06-10: the k=1 QUAD sweep is complete (N=10→640) and a success** — velocity optimal across the whole grid (L² O(h²), H¹ O(h)), pressure optimal everywhere too (≥ its nominal O(h^{kp})=O(h) equal-order order; super-optimal at 1.5–2.4×), the high-Da OSGS rate confirmed pre-asymptotic (recovers to ≥1.0 at N=640), OSGS ≈2× more accurate than ASGS. See its success box. |
| **Reference** | [convergence-baseline.md](mms/convergence-baseline.md) | Per-cell baseline (H¹ rates + inner-iteration costs). The N≤320 table is the pre-asymptotic reference; the 2026-06-10 note records the N=640 recovery. Remaining tracked target: JFNK = cut OSGS iters (the high-Da rate target is **met** — recovers). |
| Sub-topic | [fold-recovery.md](mms/fold-recovery.md) | Stiff-corner fold diagnosis + continuation rescue (status §3, region B). |

(The complete, validated k=1 QUAD sweep is the current authority; the old May-2026 partial-sweep
artifacts were removed in the 2026-06-10 cleanup — see
`../test/extended/ManufacturedSolutions/previous_results/README.md`.)

## `cocquet/` — Cocquet benchmark reproduction

| Role | Doc | Notes |
|---|---|---|
| **Canonical** | [investigation-synthesis.md](cocquet/investigation-synthesis.md) | Audited synthesis; the former magnitude hypothesis-ledger is now its Appendix. Latest mesh-recipe result narrows the gap to ~2×. |
| **MMS sibling** | [cocquet-form-mms-status.md](cocquet/cocquet-form-mms-status.md) | Canonical doc for `CocquetFormMMS` (equal-order VMS vs Taylor–Hood under MMS, Forchheimer reaction). 2026-06-16: moderate-α complete & clean; high-Re×low-α corner **folds for VMS, Taylor–Hood converges**; exact low-α fold cause **open** (σ̃_α hypothesis unconfirmed; ASGS≠OSGS reaction handling). |
| Historical | [convergence-analysis.md](cocquet/convergence-analysis.md) | 12-phase slope investigation (some early phases withdrawn). |
| Artifact | [email-questions.md](cocquet/email-questions.md) | Draft email to the authors (unsent). |
| Transcripts | [replicating-cocquet-transcript.md](cocquet/replicating-cocquet-transcript.md), [corner-singularity-transcript.md](cocquet/corner-singularity-transcript.md) | Raw investigation conversations. |

## `solver/` — algorithm references & process

| Role | Doc | Notes |
|---|---|---|
| **Canonical ref** | [paper-code-divergences.md](solver/paper-code-divergences.md) | Ledger of every code/paper divergence, classified and justified. Update when implementation diverges from the paper. |
| **Canonical ref** | [algorithm-code-mapping.md](solver/algorithm-code-mapping.md) | 1:1 map from `osgs_algorithm/osgs_algorithm.tex` algorithm boxes to the solver files (`solver_core.jl` shared core + orchestrator, `asgs_solver.jl` ASGS Stage-I, `osgs_solver.jl` coupled solve, `mms_verification.jl` verifier, `nonlinear.jl` kernel). |
| **Canonical** | [osgs-reaction-dominated-rate.md](solver/osgs-reaction-dominated-rate.md) | The high-Da OSGS velocity-rate loss — **RESOLVED 2026-06-10: pre-asymptotic, recovers to ≥1.0 at N=640** (see its banner). Two corrections noted there: the empirical recovery, and the 2026-06-09 theory fix (OSGS retains σ; ASGS drains to σ_a) — so the dip is *not* a coercivity loss. Body TL;DR/mechanism predate both (historical). |
| **Canonical** | [coupled-only-leaning-and-jfnk-plan.md](solver/coupled-only-leaning-and-jfnk-plan.md) | 2026-06-07 leaning: OSGS collapsed to a single `coupled` route (an equivalence oracle proved `freeze_after_k` diverges in the reaction corner), plus the deferred JFNK plan to recover near-quadratic speed (the dropped ∂π/∂U). |
| **Canonical** | [nonlinear-convergence-criterion-prompt.md](solver/nonlinear-convergence-criterion-prompt.md) | Spec for the scale-free outer-loop stopping criterion (`convergence_criterion.jl`): converged ⇔ ε_M ≤ tol_M and ε_C ≤ tol_C, with ε_M = ‖r_M‖/D_M (momentum residual / dynamic force-magnitude envelope) and ε_C the source-subtracted mass residual over a flux-gradient+source envelope. No a-priori `U`/`L`/`P`/`Re`/`Da` scales. |
| Backlog | [efficiency-ideas.md](solver/efficiency-ideas.md) | Newton/Picard scheduling & convergence-gate proposals (2026-06-04; the ping-pong cascade has since landed). |
| Audit | [normalization-audit.md](solver/normalization-audit.md) | Dimensional/scaling audit of every solver convergence/drift/divergence gate (encoding-invariance, P5 2026-06-04). |

## `paper/` — about the SIAM article

- [errata.md](paper/errata.md) — open editorial items for `../theory/paper/article.tex` needing author judgment.
