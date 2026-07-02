# Route B — 2D MMS official sweep: status & resume note

**Date:** 2026-07-02. **State:** ⏸ **PAUSED** (k2 QUAD partial) — stopped mid-k2 to pack the machine for
travel. Everything needed to resume is below. Branch: `feat/route-b-algebraic-mass-gate`.

## What Route B is (one line)

The mass convergence gate is now the Philosophy-A **algebraic** measure `ε_C = ‖r_C‖/D_C` (pressure block
of the assembled residual, symmetric with the momentum gate `ε_M`), replacing the strong-form measure that
floored at O(h^{kv}) and forced the loose `eps_tol_mass = 0.8` rubber-stamp. The strong-form value is kept
as the diagnostic `eps_C_strong`. Code committed in `e455f36`; full write-up in
[docs/formulation-audit-2026-06-24.md](../formulation-audit-2026-06-24.md) §C.3 / F4.

## Sweep state

Run via the **canonical harness → canonical locations** (`run_test.jl` auto-routes each single-family
config to `results/k<kv>/<etype>/results.h5`; `write_vtk=false`, `trace_convergence_norms=true`, MMS
plateau verification on; BLAS pinned to 1 thread/shard).

| family | config | official DB | status |
|---|---|---|---|
| k1 QUAD | `phase1_quad_k1.json` | `results/k1/QUAD/results.h5` | ✅ 48/48 (done ~03:24) |
| k1 TRI  | `phase1_tri_k1.json`  | `results/k1/TRI/results.h5`  | ✅ 48/48 (done ~10:04) |
| k2 QUAD | `phase1_quad_k2.json` | `results/k2/QUAD/results.h5` | ⏸ **39/48 — PAUSED** |

The k2 DB was verified **readable** after the pause (no corruption, no stale `.lock`), so resume continues
cleanly from 39.

## Behavior-preservation so far (k1 — verified)

- **0 NaN**; every finest-pair L²u rate ≈ 2.0 (optimal). The tight symmetric mass gate did **not** break
  any k1 cell.
- vs the gold reference `previous_results/validated_k1_quad_N640` at N=320 (k1 QUAD): **median relative
  Δe_u = 6.5e-7**, **40/48 cells within 1%**.
- The 8 outliers (~6%) are all **Da=1e6 OSGS** cells — the known reaction-dominated OSGS regime (the
  ∂π/∂u linear-rate sensitivity), **not** a Route B defect.
- **⚠ Comparison caveat:** the archived pre-Route-B DBs are **merges** of `phase1_*` + `phase1_*_preJFNK`
  (24 + 24 cells). A merge-blind per-cell diff spuriously shows ~4× (it reads the worse preJFNK cells).
  Always filter by the `config_file` attr, or compare against `validated_k1_quad_N640` (the clean gold).

## Backup (pre-Route-B baseline)

`previous_results/pre_route_b_2026-07-01/` — the pre-Route-B official DBs (k1/QUAD, k1/TRI, k2/QUAD) +
their reports, **committed** (traces gitignored). Each DB embeds the config that produced it under its
`configs/` group. See that folder's `README.md`.

## HOW TO RESUME k2 (on return)

Resume is **automatic** — the harness skips completed `(cell, N)` entries, so re-running only fills the ~9
remaining k2 cells. **Do NOT `rm` the k2 DB** (it is intact at 39/48). `erase_past_results=false` in the
config, so it accumulates.

```bash
cd test/extended/ManufacturedSolutions
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1     # avoid BLAS oversubscription across shards
for k in 1 2 3 4; do
  julia --project=../../.. run_test.jl phase1_quad_k2.json --shard $k/4 > /tmp/k2_shard_$k.log 2>&1 &
done; wait
```

(The overnight launcher `run_official_2d_sweep.sh` was a **scratch** file — not committed, not in the repo.
The command above is the self-contained resume; it needs nothing from that launcher.)

## Regenerate the latest traces & plots (standard pythons — latest by default)

```bash
python analyze_results.py                     # convergence PNGs + reports; auto-discovers results/k*/*/results.h5
python plot_trajectory.py --sweep k2/QUAD     # per-cell solver-path plots from traces/  (also k1/QUAD, k1/TRI)
```

The k1 convergence PNGs + reports are **already generated** (`results/k1/{QUAD,TRI}/convergence_*.png`,
`results/convergence_report.md`, `results/summary_tables.txt`). k2 output is partial until resumed; re-run
`analyze_results.py` after resume to refresh it.

## Remaining verification once k2 completes (TODO)

1. **Full per-cell Route-B-vs-baseline comparison** for all three families — use `validated_k1_quad_N640`
   as the clean gold for k1; for k2, compare against the archived pre-Route-B k2 (noting it was the
   failing-OSGS-P2 baseline: OSGS-P2 was `success=False`/`eps_used=0` there).
2. **k2 high-Re cells** — confirm they reach optimal O(h³)/O(h²) under the tight symmetric mass gate; this
   is the sensitive regime (the k=2-gate lesson) and the main thing the overnight run was meant to prove.
3. The 3D **OSGS-P2 F7 preconditioner** frontier is a *separate* track (audit §F7) — not part of this 2D
   gate verification.
