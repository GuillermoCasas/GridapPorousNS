---
name: post-correction-docs-sync
description: Use IMMEDIATELY after applying a batch of corrections that address tracked issues — paper/`.tex` edits, bug fixes, config or schema changes, resolved findings. Scans every living doc that referenced those issues (checklists, `findings.md`, `open-questions.md`, `pending-tasks.md`, `lessons_learned.md`, `theory-code-map.md`, status banners, MEMORY) and reconciles it with the ACTUAL post-fix state: marks applied items done, corrects now-stale claims and summary banners, closes to-dos, fixes drifted line numbers and cross-references, retires superseded prose. Every "done"/"applied" claim is verified against the real diff or file — never asserted from memory. Complements the before-commit docs-hygiene rule; this fires at the moment of the fix, which is the step most often skipped and most often done inaccurately.
---

# Reconcile the docs the instant a fix lands — accurately

**The problem this skill exists for.** When you fix a tracked issue, the docs that *tracked* it go
stale in the same instant: a checklist item still reads "open," a status banner still says "all
applied" when two weren't, a finding still cites a line number the edit moved, an open question is
now settled. Reconciling this is *supposed* to happen — but it is routinely skipped ("I'll do it at
commit time"), and when it is done, it is done **from memory** and over-claims. The canonical failure:
writing "✅ all items applied" without checking each one, when in fact two were missed. That banner is
now worse than no banner — it actively misleads the next reader.

So: **finishing a correction is not "the code/`.tex` compiles." It is "the code compiles AND every doc
that referred to the issue now tells the truth about it."** Do the second half immediately, and do it
by *checking*, not remembering.

## When to run

Right after you apply a batch of corrections that resolve issues something was tracking:
- paper / `theory/**/*.tex` edits that address review findings or a checklist;
- bug fixes, config/schema changes, or a resolved `open-questions.md` / `pending-tasks.md` entry;
- any change that makes an existing doc claim (status, line number, "OPEN", a numeric value) stale.

Run it *before* moving on to the next task and *before* committing — not deferred to commit time,
because by commit time the context of "what exactly did I just change" has decayed.

## The reconciliation pass

1. **Enumerate what the fix touched.** From the actual diff (`git diff --stat`, `git diff`), list the
   files and the specific claims/items each edit resolves. Work from the diff, not from your plan —
   the plan is what you *intended*; the diff is what you *did*. They differ (that is how items get
   missed).

2. **Find every doc that references those issues.** Grep the living docs for the issue IDs, symbols,
   equation labels, file/line references, and status words involved. Cover the Tier-1 living docs
   (`docs/README.md` lists them), any pre-submission / audit checklist, `MEMORY.md` + memory files,
   and in-doc **status banners** ("✅ APPLIED", "OPEN", "DONE", counts like "N/N", page counts).

3. **Reconcile each hit against the ACTUAL state — the non-negotiable step:**
   - Mark resolved items done; move finished to-dos to "Superseded/done"; flip settled open questions.
   - **Verify every "applied/done" claim by re-deriving it from the diff or re-reading the file.**
     For each item a banner or checklist calls done, confirm the corresponding edit is actually
     present. If you cannot point to the concrete change, the item is **not** done — say so and either
     apply it or mark it deferred with the reason. Never let a summary claim outrun the item-level truth.
   - Correct drifted line numbers, `\cref`/`\label`, file paths, and numeric values the edit changed.
   - Update counts and build facts to the re-measured values (page count, label count, "N/N checks"),
     not the remembered ones.
   - Retire prose the fix made obsolete. Prefer **net deletion** — a solved problem should shrink the
     docs, per `docs-hygiene`.

4. **Record genuinely new, durable outcomes** where they belong: a settled result → `findings.md`
   *with the argument that makes it true*; a real regression → append to `lessons_learned.md`; a
   code↔paper mapping change → `theory-code-map.md`. Not everything is a new doc; most reconciliation
   is *editing existing docs down to the truth*.

5. **State deferrals honestly.** Anything the fix batch did **not** complete (needs data you must not
   fabricate, is fragile/risky, or is out of scope) gets listed explicitly as deferred, with the
   reason — in the same banner that lists what was done. A truthful "applied X, deferred Y because Z"
   beats a tidy "all done."

## Accuracy discipline (the heart of it)

- **Check, don't remember.** Before writing "done" against an item, produce the evidence (the diff
  hunk, the grep hit, the rebuilt count). No evidence → not done.
- **Reconcile the summary to the items, never the reverse.** If the item-by-item pass shows 2 of 15
  undone, the banner says "13/15 applied, 2 deferred" — you do **not** round up to "all applied."
- **Re-measure build/derived facts.** Rebuild and read the real page/label/warning counts; do not
  carry forward a stale "66 pp / 722 labels" if you added a page.
- **A wrong "done" is worse than an open "todo."** The open item invites work; the false "done"
  suppresses it. When unsure, mark it open/deferred, not done.

## Anti-patterns (do not do these)

- Writing "✅ all applied" from the plan without diffing what actually changed. *(This is the exact
  miss this skill was created to prevent.)*
- Deferring all doc reconciliation to commit time, then reconstructing "what I changed" from memory.
- Only ever *adding* doc prose after a fix; a reconciliation that never deletes/retires anything is a
  sign you updated status theater instead of the underlying claims.
- Leaving stale line numbers / labels / counts because "the prose is still roughly right."
- Fabricating a value (a DOI, an email, a tolerance) to close a doc item — leave it flagged for the
  human instead.

## Definition of done for this pass

Every doc that referenced the fixed issues now (a) reflects the real post-fix state item-by-item,
(b) carries no summary claim the items don't support, (c) has correct line numbers/labels/counts, and
(d) lists any deferrals with reasons. Then — and only then — proceed to the verification gate and commit.
