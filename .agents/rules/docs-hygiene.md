---
name: porousns-docs-hygiene
description: Before every commit, reconcile docs/ with the work just done — close finished to-dos, promote settled results into findings.md (with their argument), append regressions to lessons_learned.md, retire superseded prose. docs/ should shrink as problems are solved, not only grow.
glob: "**/*.{md,jl,json}"
---
# Mindset: docs/ is a living ledger — reconcile it before every commit

`docs/` records the **process** of building this solver and its theory: what is known, what is open,
what is next. That value decays the moment it drifts from reality — a closed question left flagged
OPEN, a done task still in the backlog, a settled finding buried in a dossier. So `docs/` is not
write-once: **finishing a piece of work includes updating the docs it touched.**

The health metric is counter-intuitive: **a good docs change often makes `docs/` smaller.** Solving a
problem should *remove* a to-do, *retire* an investigation dossier into one distilled finding, and turn
an open question into a settled one. If your docs edits only ever add prose, the folder is trending
toward the exact hard-to-navigate state this rule exists to prevent.

Treat "commit without reconciling the docs" the same as "commit without running the verification gate."

---

## The three tiers (know where each kind of content belongs)

- **Living** (navigate daily): `findings.md`, `open-questions.md`, `pending-tasks.md`,
  `theory-code-map.md`, `lessons_learned.md`, `README.md`.
- **Evidence** (`mms/`, `cocquet/`, `solver/`, `formulation-audit-*`, the clean-room bundle): the full
  experiment logs *behind* the living docs. One canonical dossier per topic.
- **Archive** (`archive/`): dead ends and raw transcripts, provenance only.

Permanent **theory** (derivations, the formulation, algorithm boxes) does **not** belong in `docs/` at
all — it lives in LaTeX under `theory/`. `docs/` may carry the *argument* for a finding (enough to trust
and reproduce the reasoning), but a permanent derivation stranded in `docs/` is a task to move it to
`theory/`, not something to grow here.

## Before-commit checklist (run for any change that touched `src/`, tests, sweeps, or theory)

1. **Backlog — `pending-tasks.md`.** Did this work *complete* a task? Move it to the "Superseded / done"
   section (don't silently delete — a one-line done-entry with the date/commit is the record). Did it
   *reveal* new work? Add it under the right block (theory · code–theory consistency · formulation ·
   solver/numerics · post-processing · input/output · tests · cleanups). Did it change a task's
   premise? Edit the task, don't leave the stale version.
2. **Findings — `findings.md`.** Did you *settle* something non-obvious? Record the claim **with the
   decisive argument/measurement** (the numbers that make it true), so a future investigation can trust
   and reproduce it — not just the conclusion. Fold any now-resolved open question in from
   `open-questions.md`.
3. **Open questions — `open-questions.md`.** Did you close one? Remove it (its verdict goes to
   `findings.md`). Did you narrow one? Update "what's ruled out" / "what would settle it". Only genuinely
   open items stay.
4. **Regressions — `lessons_learned.md`.** Did something break, get mis-fixed, or reveal a trap? Append
   a new row (this file is **append-only** — never edit or delete an existing row; a reversal is a new
   row referencing the one it supersedes). Include *why it was wrong* (mathematical/numerical, not
   emotional) and the concrete fix.
5. **Paper↔code — `theory-code-map.md`.** Did the implementation diverge from the paper, or did a
   documented divergence change? Update the divergence ledger (§2) / algorithm map (§1) / gate spec (§3).
6. **Evidence & archive.** Did a dossier become fully subsumed by a distilled finding? Retire it (delete
   if worthless, move to `archive/` if it carries provenance or a falsified-but-instructive argument).
   Rewire any links that pointed at what you moved/deleted (grep for the old path).
7. **README.** If you added/removed/re-tiered a doc, update `docs/README.md`'s index.

## Discipline

- **Prefer net deletion.** When you can replace three overlapping investigation docs with one finding +
  one archived dossier, do it. Duplicated facts across `findings.md` / `known-issues` / a dossier are a
  smell — pick one canonical home and link to it.
- **Separate historical from current.** Never leave a doc mixing "we thought X (2026-06)" with "actually
  Y (2026-07)" as equal-weight prose. Mark the historical, or archive it. A later reversal must not read
  as a live claim.
- **No orphaned links.** After any move/delete, `grep -rn` the old path across `docs/`, `CLAUDE.md`,
  `README.md`, and `.agents/` and rewire every reference in the same change.
- **Provenance still governs** ([`reproducible-results.md`](reproducible-results.md)): consolidating docs
  must never sever a result↔config link. A dossier that is the provenance record of a result set is not
  clutter.
