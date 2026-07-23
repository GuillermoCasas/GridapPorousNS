# `docs/` — meta-documentation (process, status, findings, backlog)

All non-LaTeX documentation lives here. The **theory itself** (derivations, the formulation, the algorithm
boxes) lives in LaTeX under [`../theory/`](../theory/) (see [`../theory/README.md`](../theory/README.md));
`docs/` records the **process** of building the codebase and the theory — what we tried, what we found, what
is left to do.

The folder is organized in **three tiers**. You navigate the six **living** docs day-to-day; the **evidence**
dossiers are the full experiment logs behind them; the **archive** holds dead ends and raw transcripts, kept
for provenance but out of the way.

> **Keeping this lean is a rule, not an aspiration.** Before every commit, reconcile these docs with the work
> done — close finished to-dos, promote settled results into `findings.md`, append regressions to
> `lessons_learned.md`, and prefer *net deletion*. See [`.agents/rules/docs-hygiene.md`](../.agents/rules/docs-hygiene.md).

---

## ⭐ Tier 1 — Living docs (START HERE — the current state, kept up to date)

These six answer *what do we know? what's unresolved? what's next? how does the code map to the paper? what
must we never break again?* Read these first.

| Doc | Purpose |
|---|---|
| **[findings.md](findings.md)** | **Settled results + the argument that makes each true.** The non-obvious conclusions, with the decisive measurement/proof, so they help future investigations. Also holds resolved code-correctness issues (§7). |
| **[open-questions.md](open-questions.md)** | **Open theoretical/numerical questions** — the ones that survive investigation: the α=0.1 fold mechanism (σ̃_α), the 3D-P2 caveats, paper editorial items. Each states the leading hypothesis, what's ruled out, and what would settle it. |
| **[pending-tasks.md](pending-tasks.md)** | **The backlog, grouped by block** (theory · code–theory consistency · formulation · solver/numerics · post-processing · input/output · tests · cleanups). The *only* home for open code-correctness issues. |
| **[theory-code-map.md](theory-code-map.md)** | **Paper ↔ code reference.** The algorithm-box→file map, the divergence ledger (`[paper-faithful]`/`[code-actual]`/…), and a scale-free convergence-gate summary (full proof now in `theory/scale_free_gate_note/`). |
| **[lessons_learned.md](lessons_learned.md)** | **Append-only regression ledger** — the authority on anything "fixed"; read before touching `src/formulations/`, `src/stabilization/`, or `src/solvers/`. |
| this **README.md** | The index (this file). |

---

## Tier 2 — Evidence dossiers (the full experiment logs behind the living docs)

Detailed, mostly-read-only records. Open one when `findings.md` / `open-questions.md` points you here for the
blow-by-blow. Each topic has **one canonical dossier**.

**Repo-wide audit**
- [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) — the deep 2026-06-24/25 theory↔code +
  results audit. Verdict: the continuous VMS formulation is faithfully transcribed. Its resolved-ledger and
  open findings feed `findings.md` / `pending-tasks.md`; kept as the dated forensic record.

**`mms/` — Manufactured-Solution convergence**

| Role | Doc | Notes |
|---|---|---|
| **2D — canonical** | [mms/convergence-2d.md](mms/convergence-2d.md) | The settled 2D case: sweep status, the convergence gate, the stiff-corner fold, and the reusable gate-vs-solver diagnostic recipe. |
| 2D — provenance | [mms/convergence-baseline.md](mms/convergence-baseline.md) | Frozen per-cell N≤320 reference table (H¹ rates + inner-iteration costs). Provenance record; live numbers are in `findings.md`. |
| **3D-P2 — canonical** | [mms/p2-3d.md](mms/p2-3d.md) | The 3D-P2 cluster: the coercivity-margin accuracy verdict (with honest caveats), the iterative-penalty well-posedness fix, and the OSGS ∂π/∂u coupling fix. |
| Clean-room diagnosis | [convergence_problems_audit/files/p2_3d_diagnosis_report.md](convergence_problems_audit/files/p2_3d_diagnosis_report.md) | Independent NumPy reimplementation (+ its `.py` reproducer bundle). Its c₁ element-family verdict is **REFUTED**; kept for provenance + the code-transcription it encodes. |

**`cocquet/` — Cocquet benchmark reproduction**

| Role | Doc | Notes |
|---|---|---|
| **Canonical** | [cocquet/investigation-synthesis.md](cocquet/investigation-synthesis.md) | Settled: the slope cap is a **structured-mesh corner artifact**; the **Frontal-Delaunay** mesh recovers O(h²) for all three methods (§1c). Boundary/formulation **audited exactly faithful** to Cocquet Eq 1/2/4 (§1d). Open, in-repo & testable: the **bulk** magnitude gap + pre-asymptotic climb (our best-approx interpolation floor is itself 5–56× above the paper's numbers). |
| **MMS sibling** | [cocquet/cocquet-form-mms-status.md](cocquet/cocquet-form-mms-status.md) | Canonical `CocquetFormMMS` doc: moderate-α clean; the high-Re×low-α corner folds then converges optimally above the fold (k=1); exact fold mechanism open. |

The test-harness layout/variant record (formerly `cocquet-tube-test-unification-status.md`, deleted 2026-07-20)
now lives in the test's own [`README.md`](../test/extended/CocquetTubeTest/README.md).

**`solver/` — algorithm decision record (process, not paper theory)**

| Role | Doc | Notes |
|---|---|---|
| **JFNK** | [solver/jfnk-phase0-preconditioner-gate.md](solver/jfnk-phase0-preconditioner-gate.md) | The JFNK preconditioner gate (Phase-0 PASSED, Phase-1 LANDED) + its four unexpected findings (ρ_prec null-mode contamination; GMRES beats the spectral bound; ε_num-hurts-as-preconditioner; dropped-∂π/∂u-is-large). Its "3D watch item" **predicted** the 3D-P2 fix — whose c₁-inflated-preconditioner detail lives in [mms/p2-3d.md](mms/p2-3d.md) §C. |

The **Anderson acceleration** record was merged into [findings.md](findings.md) §5 (2026-07-11). Two resolved
solver dossiers were archived to `archive/` (Tier 3): the OSGS reaction-dominated-rate investigation and the
coupled-only-leaning decision record.

**Paper preparation & external audit** (the manuscript work — active)

| Role | Doc | Notes |
|---|---|---|
| **Submission checklist** | [pre-submission-checklist.md](pre-submission-checklist.md) | The final read-through checklist against the article. Most items ✅ done (2026-07-19 audit-response pass). Its line anchors are `article.tex` (v1); the paper's harmonized form is now `article_v2.tex`. |
| Revision plan | [paper-revision-plan.md](paper-revision-plan.md) | Point-by-point response to `archive/review_numerics_vs_theory.md`; largely applied — its own banner defers current status to the checklist. |
| Part-I erratum | [part_i_erratum.md](part_i_erratum.md) | The elemental-matrix assembly-display fix (RESOLVED 2026-07-12) + the durable `assembly_consistency_verification.py` guard. |
| **External audit dossier** | [`ChatGPT audit/`](ChatGPT%20audit/) | Slimmed (2026-07-23): consumed audit dumps pruned; keeps the forward-looking records — `latest_audit_response.md` (live "what remains / do-not-re-fix" reference), `validity_verdicts.csv`, and the round-2 plans that produced `article_v2.tex`. The 2026-07-23 commented-appendix disposition is [`audit_commented_article.md`](audit_commented_article.md). |

> **Paper version note.** The docs were written against [`../theory/paper/article.tex`](../theory/paper/article.tex)
> (the original) and cite its line numbers. A harmonized [`../theory/paper/article_v2.tex`](../theory/paper/article_v2.tex)
> now integrates the OSGS theory as Appendix D with symmetric ASGS/OSGS appendices; `coq_coverage.tex` already
> targets it. Both compile and share `continuity_appendix.tex`. **Prefer citing labels/section names over raw
> `article.tex` line numbers** — the numbers drift on every `\amend` pass and are ambiguous across the two files.

---

## Tier 3 — Archive (`archive/`) — dead ends & raw transcripts (provenance only)

Kept for traceability; conclusions long since folded into the living docs / evidence dossiers. Do not treat as
current.

- [archive/cocquet-convergence-analysis.md](archive/cocquet-convergence-analysis.md) — the 12-phase Cocquet slope
  diary (later phases self-withdrew; superseded by `investigation-synthesis.md`). Unique provenance: the h-floor
  bug + √2 h-consistency fixes and the per-phase evidence tables.
- [archive/cocquet-replicating-transcript.md](archive/cocquet-replicating-transcript.md) — raw session transcript;
  uniquely holds the five lateral hypotheses H-A…H-E and the "paper is below the interpolation floor" argument
  (since **confirmed + quantified** by the always-on interpolation floor — synthesis §1c/§1d).
  (The `corner-singularity-transcript` and `email-questions` drafts were deleted 2026-07-20 — falsified /
  superseded; the email's 7 open author-questions were salvaged into `investigation-synthesis.md` §7.)
- [archive/osgs-reaction-dominated-rate.md](archive/osgs-reaction-dominated-rate.md) — the high-Da OSGS
  velocity-rate-loss investigation (**RESOLVED**: pre-asymptotic, recovers at N=640; settled verdict in
  `findings.md` §4, mechanism in `theory/osgs_reaction_note/`).
- [archive/coupled-only-leaning-and-jfnk-plan.md](archive/coupled-only-leaning-and-jfnk-plan.md) — the decision
  record for collapsing OSGS to the single `coupled` route (lessons canonical in `lessons_learned.md`; its §3
  dead-config list is stale).
- [archive/review_numerics_vs_theory.md](archive/review_numerics_vs_theory.md) — the archived external numerics-vs-theory
  review that `paper-revision-plan.md` responds to point-by-point (carries an "ARCHIVED INPUT" header; current status
  lives in the checklist).
- [archive/final_AI_revision.md](archive/final_AI_revision.md) — a raw v1-era external AI review of `article.tex`,
  folded into `pre-submission-checklist.md` §10 (kept as provenance for a few reviewer items — quadrature-excess, √6 nodal factor).

---

## Where the theory lives (not here)

Permanent mathematics is in [`../theory/`](../theory/): [`article.tex`](../theory/paper/article.tex) (the
authoritative formulation), plus the dedicated notes (`osgs_algorithm`, `osgs_reaction_note`,
`tau_saturation_note`, `velocity_floor_regularization`, `pressure_recentering_note`, `centered_encoding`,
`scale_free_gate_note`). When
a *permanent* derivation is still stranded in `docs/`, that is a task under `pending-tasks.md` §1 (Theory) to
move it to LaTeX — not something to grow here.

## Where the machine-checked proofs live (not here)

The paper's a priori chain (stability, continuity, interpolation, convergence) is machine-checked in Coq
under [`../proof_verification/`](../proof_verification/): the **ASGS** abstract theorems (four, proved from a
~50-hypothesis trusted base, plus non-vacuity witnesses) **and** the **OSGS** abstract theorems (inf–sup
stability, interpolation, consistency, convergence — `abstract_osgs_*`). The tree is now **24 `.v` files**,
0 `Admitted` / 0 `Axiom`. The paper↔Coq map (ASGS = App. C, OSGS = App. D of `article_v2.tex`) and the
trusted-base inventory are in
[`../proof_verification/coq_coverage.tex`](../proof_verification/coq_coverage.tex); that tree is
self-documenting (its own `AUDIT.md` / `README.md`), so `docs/` only points to it. See `CLAUDE.md` for the
verification gate.

Beyond the Coq chain, a **SymPy suite** machine-checks the paper's *displayed algebra* — every matrix identity,
τ closed form, robustness estimate, and elemental matrix — at **242 checks across 17 scripts**
([`../proof_verification/sympy/`](../proof_verification/sympy/)). A 2026-07-21 every-equation coverage audit
classified all 369 displayed equations and found zero further algebra errors; the per-equation map is
[`../proof_verification/EQUATION_COVERAGE_LEDGER.md`](../proof_verification/EQUATION_COVERAGE_LEDGER.md).
