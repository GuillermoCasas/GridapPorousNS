# `ChatGPT audit/` — external manuscript-audit dossier (slimmed)

External audits of the SIAM paper and the response/plans they drove. **Consumed inputs have been pruned**
(2026-07-23) — the raw Round-1/Round-2/Round-3 audit dumps and imperative instruction files were fully folded
into the paper and the living docs, so only the forward-looking records remain here. What each round changed
is now visible in git history and in `docs/pre-submission-checklist.md`.

## What's kept, and why

- **`latest_audit_response.md`** — the verified, per-item response to the deepest audit (Round 3, of
  `article_v2.tex`): each item classified CONFIRMED / MISREAD / ALREADY-DONE, the definite algebraic fixes,
  the false-positives-do-not-touch list, and the proof-verification coverage analysis. **This is the live
  reference** for what remains and what must not be "re-fixed."
- **`validity_verdicts.csv`** — the Round-2 per-issue disposition table (cited by
  `../pre-submission-checklist.md`).
- **`revision_plan_v2.md`**, **`harmonization_plan_v2.md`** — the Round-2 plan + execution log that produced
  `article_v2.tex` (OSGS integrated as Appendix D). Kept for the IMPL status tables; the work itself is done.

## The 2026-07-23 round (commented-appendix audit)

Its verified disposition — what was applied, the false positives left untouched, and the residual
author-judgment items — lives in **[`../audit_commented_article.md`](../audit_commented_article.md)** (kept
compact). The two verification gates it motivated live under `proof_verification/`:
`sympy/theorem_statement_verification.py` (statement-decoration lint, in the `run_all.py` gate) and
`hypothesis_transcription_audit.md` (the trusted-base checklist).
