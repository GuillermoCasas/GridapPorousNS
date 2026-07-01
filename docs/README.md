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
| **3D MMS (§5.2) — CANONICAL** | [3d-iterative-penalty-fix-and-osgs-coupling.md](mms/3d-iterative-penalty-fix-and-osgs-coupling.md) | **2026-06-30:** root cause of the 3D P2 failure = the **missing Codina iterative penalty** in the mass residual (NOT c₁ — Kratos runs this 3D case at paper c₁). The code had `ε_num` only in the Jacobian; adding `ε_num·(pⁿ−pⁿ⁻¹)` to the residual (article.tex line 1383) makes **ASGS-3D converge at paper c₁ with optimal rate** (L²u 5.25). Fix gated (Blitz 240/240). eps_pert homotopy ported to 3D; the ASGS boot is harmful for OSGS (boot-skip flag). **OSGS-3D OPEN:** robust from a near guess, but the dropped **∂π/∂u** coupling (stronger in 3D) blocks robust far-guess convergence (staggered π-update diverges; JFNK no far-guess traction) — needs a real saddle-point/MG preconditioner or stabilized staggering. |
| **3D P2 instability — OPEN** | [3d-p2-instability-investigation.md](mms/3d-p2-instability-investigation.md) | **2026-07-01, UNRESOLVED.** Why 3D §5.2 MMS fails at k=2 while k=1 and 2D work. The P2-3D discrete solution is **converged-but-wrong** (20–95× the interpolant, erratic), **independent of** viscous operator, mesh (structured *and* Frontal), pressure space (equal-order *and* Taylor-Hood), and method. Refutes c₁, mesh-quality, grad-div, inf-sup, tolerance, and bad-Jacobian — each with numbers. Root cause unidentified; top lead = 2D-P2-on-triangles (simplex vs 3D). Every experiment + numbers recorded in the doc (§3 table); ad-hoc probe scripts cleaned post-investigation. |
| 3D ASGS (P₁ & P₂) — SUPERSEDED | [3d-p2-convergence-investigation.md](mms/3d-p2-convergence-investigation.md) | **Superseded 2026-06-30** by the iterative-penalty doc above (its c₁/coercivity verdict was masking the missing penalty). Still-valid: §4 (iteration/Jacobian/orchestration exonerated via residual-identity + Jacobian Taylor + bare-Newton), the structured-Kuhn falsification of the mesh-quality hypothesis, and the OSGS-vs-high-Da reconciliation. |

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
