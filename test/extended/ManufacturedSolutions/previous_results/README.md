# Previous results — archived MMS sweep artifacts

Raw output from earlier Manufactured-Solution sweeps, kept for reference. These are **not live tests**
and are not run by the tier runners; live results go to the gitignored `../results/`.

The old May-2026 partial sweep (`mms-sweep-difficult-cases.partial.{h5,-stdout.log}`) and its retired
difficulty-taxonomy doc were **removed in the 2026-06-10 cleanup** — fully superseded by the complete,
validated k=1 QUAD sweep (N=10→640). Current status + numbers:
[`../../../../docs/mms/convergence-status.md`](../../../../docs/mms/convergence-status.md) (success box),
[`../../../../docs/mms/fold-recovery.md`](../../../../docs/mms/fold-recovery.md), and
[`../../../../docs/mms/convergence-baseline.md`](../../../../docs/mms/convergence-baseline.md).

## Snapshot archives (`_archive_*/`)

Each holds a `phase1_quad_k1.h5` from a labelled point in the covariance investigation:

- `_archive_preNorm/` (2026-06-02) — before the per-field residual-normalisation gate work.
- `_archive_preFix/` (2026-06-02→03) — before the scale-covariance fixes landed.
- `_archive_postFix_covariant_complete/` (**2026-06-04**) — **the definitive complete k=1 sweep**
  (288/288, N=10→320) on the fully-covariant code: covariant inner gate (`‖R₀‖`) + covariant
  relative warmup + covariant `eps_val`, `minmax` encoding. This is the dataset behind the
  high-Da OSGS rate-stagnation conclusion — see [`../../../../docs/mms/convergence-status.md`](../../../../docs/mms/convergence-status.md)
  caveat (2026-06-04) and [`../../../../docs/known_issues.md`](../../../../docs/known_issues.md).
