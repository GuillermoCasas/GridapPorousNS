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
regenerable, gitignored under `../results/`). **Supersedes** the N≤320 `_archive_postFix_covariant_complete/` below.

## Snapshot archives (`_archive_*/`)

Each holds a `phase1_quad_k1.h5` from a labelled point in the covariance investigation:

- `_archive_preNorm/` (2026-06-02) — before the per-field residual-normalisation gate work.
- `_archive_preFix/` (2026-06-02→03) — before the scale-covariance fixes landed.
- `_archive_postFix_covariant_complete/` (**2026-06-04**) — **the definitive complete k=1 sweep**
  (288/288, N=10→320) on the fully-covariant code: covariant inner gate (`‖R₀‖`) + covariant
  relative warmup + covariant `eps_val`, `minmax` encoding. This is the dataset behind the
  high-Da OSGS rate-stagnation conclusion — see [`../../../../docs/mms/convergence-status.md`](../../../../docs/mms/convergence-status.md)
  caveat (2026-06-04) and [`../../../../docs/known_issues.md`](../../../../docs/known_issues.md).
