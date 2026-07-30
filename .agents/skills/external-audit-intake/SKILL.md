---
name: external-audit-intake
description: Use when an external audit, review, critique or AI-generated report on the paper, the proofs or the codebase arrives (a pasted register, a `docs/*audit*.md` dump, a reviewer letter, a Repomix-based review). Verifies every item against primary sources, routes the survivors into the Tier-1 living docs, and deletes the audit dump. Never leave a per-audit document behind.
---

# External-audit intake

An external audit is **input**, not documentation. Its value is extracted by verification and
routing; the dump itself is a liability — it duplicates the living docs, freezes line numbers that
drift, and keeps asserting defects the repo has already closed. This repository has been through
five such rounds and each one left a file behind (`audit_article_v2.md`,
`audit_commented_article.md`, `ChatGPT audit/`, `convergence_problems_audit/`,
`porous_ns_vms_audit_report.md`). **The intake ends with the dump deleted.**

## The contract

1. **Verify before believing.** Every item is checked against primary sources by reading them.
2. **Verify before disbelieving.** "The repo already handles that" needs the same quoted evidence.
3. **Route, don't accumulate.** Each surviving item lands in exactly one Tier-1 living doc.
4. **Delete the dump — but only once its provenance is safe.** The routed items carry the audit's ID,
   so the *content* lives in the destination. Git holds the original **only if it was committed**: a
   pasted audit arrives untracked, and `rm` on an untracked file is unrecoverable. That happened on
   2026-07-30 — `docs/porous_ns_vms_audit_report.md` was never committed on any ref
   (`git log --all --diff-filter=A -- <path>` is empty), so it is gone. Commit the dump first (or
   record its `sha256` + where it came from in the destination), *then* delete it.

## Step 1 — Establish what the auditor actually read

Do this first; it decides the verdict of a whole class of items.

- If the audit was produced from a **Repomix / packed export** (`repomix*.md`, `repomix/`), read
  the pack's exclusion rules. Claims of the form *"file X is missing"*, *"the archive is not
  self-contained"*, *"no compiled objects"* are usually **ARTIFACT**: true of the pack, false of
  the repo. Check both: does X exist live, and was X really absent from the pack?
- Check the audit's date against `git log`. Quoted paper text is often **stale** — a sentence the
  repo softened weeks ago. Grep the quoted phrase in the live source before accepting it.
- Note which of the two mains the auditor saw. `article.tex` (v1) and `article_v2.tex`
  (submission) share App. A/B/C; App. D exists only in v2, and **v2 `\input`s the *commented*
  twin** `osgs_appendix_commented.tex`, not the clean `osgs_appendix.tex` that the lints read.

## Step 2 — Verify each item against primary sources

Fan out (one agent per cluster of related items) and require, per item:

- **verdict**: `CONFIRMED_OPEN` · `CONFIRMED_DONE` · `PARTIAL` · `FALSE` · `ARTIFACT`
- **evidence**: `file:line` **with the quoted text**. If a line is cited, that line was opened.
- **action**: `EDIT_PAPER` · `EDIT_COQ` · `EDIT_SYMPY` · `EDIT_CODE` · `EDIT_DOC` · `NONE`
- **patch**: exact old → new text.
- **contradictions**: any standing decision it would reverse, quoted.

Then **adversarially re-check** the verdicts with a second pass whose job is to refute them. That
pass has repeatedly been the one that caught: line numbers shifted by a concurrent edit, a fourth
occurrence of a defect the first pass called "exactly three sites", a proposed patch that breaks a
gate, and `FALSE` calls that were real defects.

### Failure modes this repo has actually hit

| Failure mode | Guard |
|---|---|
| Audit quotes a sentence the repo already softened | grep the phrase in the live file; check `git log -S` |
| "Missing file" from a packed export | read the pack's exclusion list |
| Patch weakens a claim the repo *proved more* about | search `theory/*_note/` — a companion note may prove a sharper result than the audit's remedy |
| Patch edits one appendix twin | `sympy/appendix_twins_verification.py` gates them line-for-line |
| Patch fixes v2 and forgets v1 | **the most frequent miss** (6 of them in one pass). Most body text is byte-identical in `article.tex`; mirror by *string match*, and drop any clause citing a v2-only label (`sec:CommonSetting`, `H:patch`, `H:advectionsmooth`, `th:StabilityOSGS`, any `oa:*`) or v1 gains undefined references |
| A note claims one main is *exempt* | verify by grepping **that file**, never by which main the batch happened to edit. A 2026-07-30 checklist entry asserted "v1 is unaffected — it has no `P₂/P₁ ASGS` rows"; `article.tex` had all four rows, so v1 sat with a refuted headline number and two undisclosed captions |
| A claim about the runs is checked against the *config*, not the harness | a swept factor the harness never reads is invisible to a config grep. Both MMS harnesses hardcode `physical_epsilon = 1e-8` over a declared `[0.0]` (see `findings.md` §9.5) — trace the value to the object actually assembled |
| Patch cites `file:line` in a doc | line numbers drift the moment the same file is edited again; cite **labels**, and if a number is given, say it is a hint |
| Patch edits the shared App. C (`asgs_convergence.tex`) with a v2-only label | App. C is `\input` by **both** mains; only labels present in both are allowed |
| Patch edits a `.tex` and stops | `document_hygiene_verification.py` H7 fails on a log older than its sources — **rebuild before gating** |
| An item is "recorded as open" and closed in the same batch | sequence: land the fix, then write the doc in post-fix wording |
| Two agents propose mutually contradictory doc rows | reconcile before writing; never write both |

## Step 3 — Fix in dependency order

1. **Coq / code / SymPy** first (the artefacts other files describe).
2. **Paper** next.
3. **Doc/ledger sync** last, in post-fix wording.
4. **Rebuild, then gate**, in this order:
   ```
   cd theory/paper && latexmk -pdf article.tex article_v2.tex     # H7 needs fresh logs
   cd proof_verification && latexmk -pdf coq_coverage.tex          # if it was edited
   python3 proof_verification/sympy/run_all.py                     # read the per-rule counters
   cd proof_verification/coq-formal && ./run_all.sh                # compiles + coqchk only
   # run_all.sh does NOT do Print Assumptions -- check it separately:
   #   Print Assumptions <headline theorem>  =>  exactly the 3 stdlib axioms
   julia -O0 -t 1 test/run_blitz_tests.jl                          # if src/ was touched
   ```
   Record the fresh build invariants (pages / newlabels / undefined / overfull) — they drift with
   prose, so a stale count is what makes a "verified clean" claim unverifiable.

Never weaken a lemma, relax a threshold, or downgrade a gate to make something pass. If a Coq
statement is weaker than the printed one, **strengthen the Coq** — in this tree the mechanical
mirror has twice been nearly free, because the sibling file already carried the sharper form.

## Step 4 — Route the survivors (the whole point)

| Item kind | Destination |
|---|---|
| Paper edit applied, or a submission gate still open | `docs/pre-submission-checklist.md` — one addendum block per audit, with severity + status |
| Settled result worth keeping | `docs/findings.md` — **with the argument that makes it true**, not just the verdict |
| Actionable work not yet done | `docs/pending-tasks.md`, in the right numbered block |
| Genuine open question needing author judgment | `docs/open-questions.md` |
| A defect class that could recur | `docs/lessons_learned.md` (append-only) |
| Paper↔code mapping drift | `docs/theory-code-map.md` §2 divergence ledger |
| Coverage/trust claim wording | `proof_verification/coq_coverage.tex`, `hypothesis_transcription_audit.md`, `EQUATION_COVERAGE_LEDGER.md` |

Rules for the routed text:

- **Carry the audit's item ID** (`C1`, `T3`, …) and its date, so a later reviewer can see the
  item was adjudicated rather than missed.
- **Record declines with their reason and their standing ruling.** A declined item that is not
  written down is re-raised by the next audit — that has already happened three times here
  (per-theorem trust boxes, the `c₁` "admissible because" clause, the regime-map table).
- **Record false positives explicitly** in a *do NOT re-fix* list. A verified `FALSE` is a result.
- **Prefer net deletion** (`.agents/rules/docs-hygiene.md`): if an item closes a checklist entry
  or a pending task, close it in the same pass.

## Step 5 — Delete the dump, and check the neighbours

```
# If it is UNTRACKED, make it recoverable first -- rm on an untracked file is final:
git log --all --diff-filter=A -- docs/<the-audit-file>.md   # empty => never committed
git add docs/<the-audit-file>.md && git commit -m "audit: record the <date> intake dump verbatim"
git rm docs/<the-audit-file>.md        # now the deletion is a revertible commit
# (Or, if committing the dump is unwanted: record its sha256 and origin in the destination doc,
#  then rm. Never delete an untracked dump with neither.)
```

Then look for what the audit's routing made stale:
- entries in `docs/README.md` that indexed the deleted file;
- a companion note quoting the *old* paper sentence you just changed
  (`theory/*_note/*.tex` — this bites every time);
- stale build counts, module counts and file names in `theory/README.md`,
  `docs/README.md`, `proof_verification/*/README.md`.

If the audit's input export (`repomix*.md`, a downloaded PDF) is still lying around, say so in the
summary and let the author remove it — do not delete their inputs unasked.

## What "done" looks like

- Every audit item has a verdict, evidence, and a destination — including the false ones.
- The fixes are landed and every gate is green **on fresh logs**.
- The living docs answer the audit's questions; the audit file is gone.
- Someone reading only `docs/` learns what was true, what was declined and why, and what remains.
