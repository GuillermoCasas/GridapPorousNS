# Archived CocquetFormMMS results

Deliberately-kept snapshots, archived before a superseding official re-run (per
`.agents/rules/reproducible-results.md`: never sever the parameters→results link).

## `cocquet_form_mms_taylorhood_stabilized_N160_2026-07-30.{h5,json}`

The stabilized-Taylor–Hood (`P2/P1` ASGS) control DB **as published in
`tab:CocquetMMSL2/H1` of `article_v2.tex` up to 2026-07-30**, with the config that produced it.

Why archived: the run had not completed its own declared ladder. The config asks for
`convergence_partitions = [10,20,40,80,160,320]`, but the DB contains only up to **N=160**
(`h=[0.1 … 0.00625]`), and only up to **N=80** at `(Re,α₀)=(10⁵,0.1)` — the earlier attempt
"crashed (OOM, concurrent jobs) at N≥160" (see the corner config's `_comment`). The published
`(10⁵,0.1)` row was therefore taken from the side DB
`results/debug_results/cocquet_stabth_corner.h5` (N=160,320), which
`.agents/rules/official-results-path.md` forbids for published numbers.

Because every other row of those tables is at N=320, the finest-mesh-error column mixed
resolutions, which produced a false headline (a "factor ~10" accuracy gap that was in fact one
extra refinement level of a cubically convergent method; the like-for-like factor at N=160 is
1.16–1.47). See `docs/lessons_learned.md` (2026-07-30 (b)) and `docs/pending-tasks.md` §7h.

The superseding run re-executes the **same, unmodified official config** to completion.
