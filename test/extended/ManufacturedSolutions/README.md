# Manufactured Solutions (MMS) convergence harness

The Method-of-Manufactured-Solutions sweep that is the driving correctness criterion for the solver:
it forces an exact `(u_ex, p_ex)` through the full nonlinear solve and checks the error converges at
`O(h^{k+1})` across `(Re, Da, α₀, h, k, element-type, stabilization)`. This directory is a **manual
research harness** — it is *not* part of the automated `runtests.jl` tiers.

## Layout

| File | Role |
|---|---|
| `run_test.jl` | **Canonical sweep driver.** Reads a `data/<config>.json`, sweeps the factor grid, writes one HDF5 results DB under `results/`. |
| `analyze_results.py` | **Single analysis entry point.** Detects flagged cells, builds the merged + detailed convergence reports and summary tables, optional plots. |
| `run_continuation.jl` | Coarse-N fold-**reach** driver (α / mesh-ladder continuation to *find* a root when none exists at coarse N; batch `phase2` mode). At N≥512 a root exists, so prefer the direct solve below. See [`docs/mms/fold-recovery.md`](../../../docs/mms/fold-recovery.md). |
| `run_corner_article.jl` | **Direct exact-guess corner solve (ASGS).** Solves the Re=1e6/α₀=0.05 fold corner cells at base N=512 + mesh-step to N=768 — the fold clears by ≈N=512, so plain Newton from the exact guess converges (~3 iters); no α-continuation needed. Writes `results/debug_results/corner_tri_k1_a005.json`. |
| `run_corner_osgs.jl`, `osgs_corner_lib.jl` | **Direct corner solve (OSGS).** OSGS coupled solve (mirrors `solve_osgs_stage!`), warm-started from the ASGS corner root. Writes `corner_tri_k1_a005_osgs*.json`. |
| `make_results_tables.py` | Generates `results/paper_tables.tex` — the four `article.tex` tables filled with the latest Gridap results (FME at a common N=320, corner cells extrapolated + daggered; an `ε_pert (N_NS+N_Pic)` solver-effort column). |
| `merge_corner_results.py` | **Merges the direct-solve corner JSONs into the sweep HDF5** as proper content-addressed groups, so `analyze_results.py` plots/reports the corner cells alongside the swept ones (the table generator already reads both). Run it, then **wipe `results/k<kv>/<etype>/*.png` and re-run `analyze_results.py`** — the per-cell `config_<idx>` numbering shifts when groups are added, so stale PNGs must be cleared. |
| `plot_mesh.py` | Mesh visualizations (thin wrapper over `tools/mesh_viz/`). |
| `probe_stiff_diagnose.jl`, `run_diagnostics.jl` | `[diagnostic-tool]` — manually-run, single-cell investigations (not the sweep). `probe_stiff_diagnose.jl` supplies the `build_cell`/`probe_a2_heavy_solve` primitives that `run_continuation.jl` + the corner drivers reuse (it was restored 2026-06-17 — `run_continuation.jl`'s `include` of it had been left broken by its deletion in `3c66edd`). |
| `diagnostics/{jacobian_equilibration_osgs,velocity_centering}_probe.jl` | Retained negative-result probes (see `docs/lessons_learned.md`). |
| `data/*.json` | Sweep configs (see below). | 
| `results/` | All output (**gitignored**). HDF5 DB + merged reports at the root; per-cell artifacts under `results/k<kv>/<etype>/` (convergence `.png` from `analyze_results.py`, plus `vtk/` and `traces/`). Ad-hoc/debug runs mirror under `results/debug_results/`. |
| `previous_results/` | Deliberately-kept archived sweep snapshots (tracked; `traces/` within them are gitignored via `**/traces/`). |

## Running a sweep

There is **one** sweep config — [`data/test_config.json`](data/test_config.json) — the full reference
grid (Re/Da/α₀ ∈ {1e-6, 1, 1e6}; QUAD+TRI; k=1,2; ASGS+OSGS; N=10–320; strict honest-exit gate
`k_nf=10`; MMS plateau verification; the Re=1e6/α₀=0.05 fold corner deferred via `skip_cells`). Every
*study* (a particular k / element-type / sub-grid) is a CLI selection on this one config, routed to its
own DB — no per-study config files.

```bash
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl test_config.json
python analyze_results.py          # reads the per-(kv,etype) DBs + their embedded configs automatically
```

**Results layout (2026-06-27).** An *official* run writes one generically-named DB per `(kv, etype)`,
co-located with that family's plots/traces/vtk:

```
results/k<kv>/<etype>/results.h5      # the DB (no study-specific filename)
results/k<kv>/<etype>/convergence_*.png, traces/, vtk/
```

The **full config JSON(s)** that produced the data are embedded *inside* each DB under the group
`configs/<config-file>`, and every result group points to its config via the `config_file` attribute
(usually one config per DB; a merged DB — e.g. a surgical re-run of a few cells under a second config —
holds several). So a DB is self-describing: `analyze_results.py` reads the embedded config and needs no
`--config`. The output HDF5 group key is **content-addressed**: `config_<idx>_<tag>_<method>`, where
`<tag>` hashes the physics cell `(Re, Da, α₀, kv, kp, etype)` identically across runs and `<idx>` is a
deterministic, shard-independent label.

**Per-cell outputs (ParaView + traces).** VTK field snapshots are written **by default** so each cell
can be inspected visually, under `results/k<kv>/<etype>/vtk/mms_<method>_Re…_Da…_a…_N….vtu`, alongside
the `traces/` JSON sidecars and the convergence `.png` plots. **Ad-hoc/debug or A/B study runs** keep the
*old* explicit-DB-name behavior for per-study isolation: set `h5_filename` (or `--h5`) under
`debug_results/` (e.g. `"debug_results/_ab_off.h5"`) and the whole tree — including that single
explicitly-named DB — mirrors under `results/debug_results/…`, so two studies of the same `(kv,etype)`
never collide. A full sweep can write many GBs of `.vtu`; set `"write_vtk": false` for large sweeps.

### CLI overrides — run any sub-combination without authoring a config

| Flag | Effect |
|---|---|
| `--filter Re=…,Da=…,alpha0=…,kv=…,kp=…,etype=…,method=…` | Select a sub-grid. AND across keys, OR within a key's repeated values. `1e6` matches `1000000.0`. |
| `--h5 debug_results/<name>.h5` | Route this run to an explicitly-named single DB under `debug_results/` — keeps A/B / scratch studies isolated (two studies of the same `(kv,etype)` won't collide). Omit for official runs, which auto-route to `results/k<kv>/<etype>/results.h5`. |
| `--max-N <int>` | Cap the N-ladder (quick gates, without editing `convergence_partitions`). |
| `--shard k/N` | Run shard k of N; concurrent launches share one DB (below). |

```bash
# k=2 QUAD study (official) -> auto-routes to results/k2/QUAD/results.h5:
julia --project=../../.. run_test.jl test_config.json --filter etype=QUAD,kv=2
# quick smoke (1 cell, short ladder), isolated under debug_results/ so it can't touch official DBs:
julia --project=../../.. run_test.jl test_config.json --filter Re=1.0,Da=1.0,etype=QUAD,kv=1 --max-N 40 --h5 debug_results/smoke.h5
```
The **fold corner** (Re=1e6, α₀=0.05) is excluded from the sweep by `skip_cells` (it folds — no root —
at N≤320). It is **reproduced** (P1/TRI, both methods) by a direct exact-guess solve at N≥512 via
`run_corner_article.jl` (ASGS) / `run_corner_osgs.jl` (OSGS); `run_continuation.jl` is the fallback for
reaching a root at coarse N. The Q2/QUAD-k2 corner is **also** reproduced (it does not fold — k=2
converges directly at N=160→320). See [`docs/mms/fold-recovery.md`](../../../docs/mms/fold-recovery.md).

### Concurrent launches into ONE shared database

`--shard k/N` deterministically partitions the selected cell list into N disjoint shards. All shards
write the **same** HDF5 file; a `.lock` sidecar (`FileWatching.mkpidlock`) serializes the brief
per-cell writes, and the content-addressed keys keep shards from clobbering each other. No per-worker
files, no post-hoc merge step.

```bash
for k in 1 2 3 4; do
  julia --project=../../.. run_test.jl test_config.json --filter etype=QUAD,kv=2 \
        --shard $k/4 > logs/shard_${k}.log 2>&1 &
done
wait
python analyze_results.py --h5 results/k2/QUAD/results.h5   # one shared DB; embedded config read automatically
```

Notes / limits:
- `erase_past_results=true` is rejected with `--shard` (a shard would wipe another's work). The config
  ships `erase_past_results=false`; to start fresh, `rm` the target DB (or pick a new `--h5` name).
- Sharding is round-robin by count (not cost-balanced); if load balance ever matters, sort the
  selected cells by a cost proxy before sharding — there is no auxiliary-file machinery to add.
- Resume is automatic: completed `(cell, N)` entries are skipped, so re-running or adding shards only
  fills gaps. Don't mix a legacy-format DB (`config_<idx>_<method>`) with new content-addressed writes —
  start fresh, or keep legacy DBs read-only.

## Configs (`data/`)

Consolidated to the minimum — *which* cells to run is a CLI concern (`--filter`/`--h5`/`--max-N`), not a
config-file one, so the per-(k, element-type) variants are gone:

| Config | Harness | Purpose |
|---|---|---|
| `test_config.json` | `run_test.jl` | **The** full reference sweep. Every k / element-type / sub-grid study is a `--filter` selection on this one config, routed to its own DB via `--h5`. |
| `continuation_c24.json` | `run_continuation.jl` | α-continuation fold recovery at the Re=1e6/α₀=0.05 corner. |
| `continuation_c24_rate.json` | `run_continuation.jl` | `mesh_ladder` regime — convergence rate at the corner via interpolate-up. |

To recreate a retired sibling (e.g. the old `continuation_c21` = `c24` with `Da=1e-6`, or any
`phase1_*` = a `--filter` slice), copy the nearest config and change the one field, or use the CLI.

## Quick robustness assessment

A fast multi-regime gate to confirm the harness runs cleanly and to spot convergence regressions —
e.g. before merging a change. No throwaway config needed: `--filter` a representative spread, `--max-N`
a short ladder, into a scratch `--h5` DB. (The α₀=0.05 fold corner is already excluded by the config's
`skip_cells`; it is reproduced separately by the direct-solve corner drivers — see the Layout table.)

```bash
# Da=1 column × all Re × all α, QUAD k=1, ASGS+OSGS, N≤40, 2 shards into one scratch DB:
for k in 1 2; do
  julia --project=../../.. run_test.jl test_config.json \
        --filter Da=1.0,etype=QUAD,kv=1 --max-N 40 --shard $k/2 --h5 debug_results/_robustness.h5 \
        > logs/rob_$k.log 2>&1 &
done; wait
python analyze_results.py --h5 results/debug_results/_robustness.h5 --no-plots --outdir /tmp/rob
```

### How to read it — two independent signals

1. **Harness robustness** (what an I/O / concurrency / CLI change can affect): every shard exits 0,
   all solves complete, **no** `unable to lock` / truncation / corruption in the logs, and the DB has
   exactly `#cells × #methods` content-addressed groups (`config_<idx>_<tag>_<method>`). Cross-check
   that a sharded run is content-identical to a single-process run of the same cells. This is the
   signal that validates the cleanup/generalization itself.

2. **Solver convergence** (affected only by `src/` changes, *not* by harness I/O): read
   `analyze_results.py`'s summary. Expected, non-regression behavior on these intentionally **coarse**
   meshes (N≤40):
   - **ASGS velocity L² rate ≈ 2** (optimal) at `α₀=1.0` and at `Re=1e6`; **≈ 1.6 at `α₀=0.5`** for
     low/unit Re — the known coarse-mesh *pre-asymptotic porosity-layer* effect (the layer is resolved
     by only 2–8 cells at N≤40; it recovers to optimal at fine N). See
     [`docs/mms/convergence-status.md`](../../../docs/mms/convergence-status.md).
   - **OSGS is flagged on coarse meshes, and this is pre-existing** (independent of harness I/O; it was
     present in pre-rework baselines too — see [`docs/mms/convergence-status.md`](../../../docs/mms/convergence-status.md)).
     Low/unit-Re OSGS converges to ‖R‖~1e-12 (true roots) but trips the strict honest-root gate; high-Re
     OSGS genuinely diverges on coarse N.
   - **Solver-success vs analyzer "no-root" disagreements** are surfaced by the honest-exit gate:
     `is_true_root` accepts a finest-mesh stop only if `‖R‖ ≤ k_nf · dynamic_ftol`, where `k_nf =
     solver.noise_floor_success_max_ftol_multiple`. `test_config.json` ships
     `noise_floor_success_max_ftol_multiple: 10.0`, so the gate is active by default; finer meshes
     reduce the coarse-mesh disagreements.

Rule of thumb: a *harness/IO/cleanup* change is robust if signal 1 is clean and signal 2 is **unchanged
from prior runs** — convergence quality is a `src/`-level concern, not a harness one.
