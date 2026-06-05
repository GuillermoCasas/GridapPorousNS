# Previous results — archived MMS sweep artifacts

Raw output from earlier Manufactured-Solution sweeps, kept for reference. These are **not live tests**
and are not run by the tier runners; live results go to the gitignored `../results/`.

- `mms-sweep-difficult-cases.partial-stdout.log` — raw stdout of a partial sweep (commit `15e466b`).
- `mms-sweep-difficult-cases.partial.h5` — partial results database for that run.

The analysis/writeup of this run lives in [`../../../../docs/mms/sweep-difficult-cases.md`](../../../../docs/mms/sweep-difficult-cases.md)
(it's a document about a result, so it sits with the other MMS docs).

## Snapshot archives (`_archive_*/`)

Each holds a `phase1_quad_k1.h5` from a labelled point in the covariance investigation:

- `_archive_preNorm/` (2026-06-02) — before the per-field residual-normalisation gate work.
- `_archive_preFix/` (2026-06-02→03) — before the scale-covariance fixes landed.
- `_archive_postFix_covariant_complete/` (**2026-06-04**) — **the definitive complete k=1 sweep**
  (288/288, N=10→320) on the fully-covariant code: covariant inner gate (`‖R₀‖`) + covariant
  relative warmup + covariant `eps_val`, `minmax` encoding. This is the dataset behind the
  high-Da OSGS rate-stagnation conclusion — see [`../../../../docs/mms/convergence-status.md`](../../../../docs/mms/convergence-status.md)
  caveat (2026-06-04) and [`../../../../docs/known_issues.md`](../../../../docs/known_issues.md).
