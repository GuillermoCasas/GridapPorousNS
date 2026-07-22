# `ChatGPT audit/` — external manuscript-audit dossier

External audits of the SIAM paper and the response/revision plans they drove. **Three rounds**, oldest → newest.
Round 1 is done (folded into the living paper docs); Round 2 drove the `article_v2.tex` harmonization; Round 3 is
the newest deep audit, being processed. Read this index before assuming any single file is "current."

The living response docs are outside this folder: [`../paper-revision-plan.md`](../paper-revision-plan.md) and
[`../pre-submission-checklist.md`](../pre-submission-checklist.md).

## Round 1 — audit of `article.tex` (v1) · **DONE / superseded**

Its 82 issues were folded into `paper-revision-plan.md` + `pre-submission-checklist.md`; Round 2 confirms most
were addressed. Kept as provenance.

- `Paper-audit-request preamble.md` — the v1 top-level response ("major revision").
- `full_paper_audit.md` — the 23-section v1 audit.
- `audit_issue_register.csv` — the v1 82-issue register (`M01…`); superseded by `validity_verdicts.csv`.
- `ai_revision_brief.md` — the v1 implementation brief; superseded by `ai_revision_instructions_v2.md`.

## Round 2 — audit of the revised manuscript + proposed OSGS theory · **ACTIVE**

Drives the harmonized `theory/paper/article_v2.tex` (OSGS integrated as Appendix D). `coq_coverage.tex` targets v2.

- `Paper-audit-request_v2.md` — the Round-2 top-level response (audit input).
- `revised_paper_osgs_audit.md` — the detailed Round-2 audit (audit input).
- `theory_integration_blueprint.md` — the adopted architecture ("results in main text, proofs in appendices").
- `ai_revision_instructions_v2.md` — imperative Round-2 instructions (being applied).
- `revision_plan_v2.md` — the Round-2 plan + record of Part 1/2/3 implementation (**in execution**).
- `harmonization_plan_v2.md` — the reorganization plan (**in execution**; most current execution log).
- `validity_verdicts.csv` — the Round-2 per-issue disposition (cited by `pre-submission-checklist.md`).

## Round 3 — `latest_audit/` — deep audit of `article_v2.tex` (2026-07-22) · **being processed**

The newest, most rigorous audit (74 items, groups A–I), of the harmonized manuscript.

- `latest_audit/latest_full_paper_audit.md` — the prose audit (sections 1–15).
- `latest_audit/latest_issue_register.csv` — 82 rows (B/M/A/N/I/S).
- `latest_audit/latest_revision_brief.md` — 74 numbered action items.
- `latest_audit/Paper-latest-audit-request.md` — the request/context.

The verified response + enriched revision plan lives in `latest_audit_response.md` (this session's integration:
each item classified CONFIRMED / MISREAD / ALREADY-FIXED, with the definite algebraic fixes and the
proof-verification coverage follow-ups).
