---
name: porousns-official-results-path
description: Produce every result through the OFFICIAL test/sweep path (the committed sweep config + the harness, writing the official results DB, read by the standard analysis/plot scripts). Never fork a parallel side-config / side-DB and merge it back; never alter analysis or plotting tools to read non-official files or to stitch results across databases. To change what a sweep covers, edit the official config or use run_test.jl --filter / --shard. To keep prior results for comparison, archive them under previous_results/ and retrieve them later.
glob: "**/*.{json,jl,py,h5,md}"
---
# Results come from the OFFICIAL path — do not fork it

Sibling of `reproducible-results.md`. That rule says *never sever* the result↔config link; this one
says *never fork the results pipeline*. Both exist because side-channels make the tree untidy and the
science hard to trust — this specific mistake **happens repeatedly** and generates a lot of avoidable
complexity.

**Single source of truth for any result = the official run:** the committed sweep config +
`run_test.jl` (the harness), writing to the official results DB, read by the standard analysis/plot
scripts. Every number that ends up in a doc, a plot, or a decision must come from that path.

## Do
- **Change what a sweep covers by changing the OFFICIAL config** — its mesh ladder
  (`convergence_partitions`), its `Re`/`alpha_0`/`k_velocity` lists, etc. — or by targeting a subset of
  the official config with `run_test.jl <official>.json --filter … / --shard …`. The results land in
  the official DB and are read by the standard plotter with **no special wiring**.
- **Need the prior results for comparison?** ARCHIVE them first (copy the official DB + report into a
  tracked `previous_results/<name>/` together with the config that made them), then re-run the official
  path. Retrieve the archived snapshot later when you actually do the comparison. Do **not** keep a
  live parallel DB around to diff against.
- **Overwrite the standard result files on a re-run** (the standard DB, the standard plot outputs).
  Genuinely ad-hoc / debug output goes to `results/debug_results/`, never loose in `results/`.

## Don't
- **Do not spawn a parallel "side" config + side-DB** (`*_corner.json` → `*_corner.h5`, `*_ext.h5`,
  `*_patch.h5`, …) for a result you then have to MERGE back into the main series. This is the exact
  anti-pattern this rule exists to stop: it fragments provenance, needs bespoke merge logic, and
  litters the tree.
- **Do not alter analysis / plotting scripts to read non-official files or to stitch multiple DBs.**
  `plot_combined_all.py`, `analyze_results.py`, `plot_convergence*.py`, etc. read the OFFICIAL result
  files only. If a plot "doesn't show the new data," the fix is to **produce that data in the official
  DB** — never to teach the plotter to merge a side-DB or to point it at ad-hoc files.
- **Do not run results outside the official tests at all**, unless the deliverable *is* a change to the
  test or tool itself (a genuine new test behaviour, a real plotting feature). In that case, change the
  official file directly and run through it. "Trying something out in a sweep" is **not** such a case —
  do it through the official config (extend it or `--filter`/`--shard`).

## The tell — STOP if you catch yourself doing any of these
- writing merge / stitch / de-dup logic to recombine result series;
- adding a second DB to a plotter's or analyzer's input list;
- naming a file `*_corner` / `*_ext` / `*_patch` / `*_test` to hold results you will later recombine;
- pointing an analysis script at a file under `results/debug_results/` for a headline number.

When you hit the tell: route it through the official config instead (extend it, or `--filter`/`--shard`
it), archiving the prior official results into `previous_results/` first if you need them for
comparison. If the official harness genuinely cannot express what you need (e.g. a per-cell mesh
ladder), the fix is to add that capability to the harness — not to fork a side-pipeline around it.
