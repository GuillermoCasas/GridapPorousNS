# Previous results — archived MMS sweep artifacts

Raw output from earlier Manufactured-Solution sweeps, kept for reference. These are **not live tests**
and are not run by the tier runners; live results go to the gitignored `../results/`.

The old May-2026 partial sweep (`mms-sweep-difficult-cases.partial.{h5,-stdout.log}`) and its retired
difficulty-taxonomy doc were **removed in the 2026-06-10 cleanup** — fully superseded by the complete,
validated k=1 QUAD sweep (N=10→640). Current status + numbers:
[`../../../../docs/mms/convergence-status.md`](../../../../docs/mms/convergence-status.md) (success box),
[`../../../../docs/mms/fold-recovery.md`](../../../../docs/mms/fold-recovery.md), and
[`../../../../docs/mms/convergence-baseline.md`](../../../../docs/mms/convergence-baseline.md).

## Authoritative validated snapshot — `validated_k1_quad_N640/` (2026-06-10)

The **complete, validated k=1 P1/P1 QUAD sweep** (N=10→640, both ASGS and OSGS, scale-free ε_M/ε_C
gate; 24 physics cells × 2 methods, the 3 Re=1e6/α₀=0.05 fold cells `skip_cells`-excluded). This is
the dataset behind the 2026-06-10 success in [`../../../../docs/mms/convergence-status.md`](../../../../docs/mms/convergence-status.md).
Contents:

- `phase1_quad_k1.h5` — the convergence database (per-mesh `err_{u,p}_{l2,h1}`, `eval_iters/times`,
  `eval_eps`). Reproduce with `julia --project=../../.. run_test.jl data/phase1_quad_k1.json` (resumable).
- `summary_tables.txt`, `merged_convergence_report.md`, `convergence_report.md`, `flagged_cells.json`
  — the `analyze_results.py` outputs on this DB (the per-cell rates/FME).

Result: velocity optimal everywhere (L² O(h²), H¹ O(h)); pressure optimal everywhere (≥ its nominal
O(h^{kp})=O(h) order, super-optimal at 1.5–2.4×); high-Da OSGS H¹ recovers to ≥1.0 by N=640 (pre-asymptotic).
Convergence plots regenerate from the `.h5`: `python3 analyze_results.py --h5
previous_results/validated_k1_quad_N640/phase1_quad_k1.h5 --config data/phase1_quad_k1.json`. The 57 MB
of per-iteration `traces/` and the 55 MB of trajectory PNGs are intentionally **not** snapshotted (live,
regenerable, gitignored under `../results/`).

## Other kept snapshot — `pre_route_b_2026-07-01/`

The committed pre-Route-B baseline that [`../../../../docs/mms/route-b-2d-sweep-status.md`](../../../../docs/mms/route-b-2d-sweep-status.md)
cites as the reference the current algebraic mass gate was measured against.

## Pruned 2026-07-08 (config-lifecycle cleanup)

The superseded intermediate covariance-investigation snapshots (`_archive_{preNorm,preFix,postFix_covariant_complete}/`),
the `pre_jfnk_relayout_2026-06-27/` pre-relayout backup, and the `_archive_pingpong_N80_AB/` iterator A/B were
**removed** as fully subsumed by `validated_k1_quad_N640/`. Per
[`../../../../.agents/rules/official-results-path.md`](../../../../.agents/rules/official-results-path.md), results
embed their config, so any is reconstructable from an archived result if ever needed; the findings they backed
remain in the docs cited above.
