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
| `run_continuation.jl` | Fold-recovery driver for the stiff high-Re/low-α corner (α / mesh-ladder regimes; batch `phase2` mode). See [`docs/mms_fold_recovery.md`](../../../docs/mms_fold_recovery.md). |
| `plot_mesh.py` | Mesh visualizations (thin wrapper over `tools/mesh_viz/`). |
| `probe_stiff_diagnose.jl`, `run_diagnostics.jl` | `[diagnostic-tool]` — manually-run, single-cell investigations (not the sweep). `probe_stiff_diagnose.jl` also supplies the cell primitives `run_continuation.jl` reuses. |
| `diagnostics/{jacobian_equilibration_osgs,velocity_centering}_probe.jl` | Retained negative-result probes (see `docs/lessons_learned.md`). |
| `data/*.json` | Sweep configs (see below). | 
| `results/` | Output DB + reports (**gitignored**). |

## Running a sweep

```bash
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl test_config.json
python analyze_results.py --h5 results/<db>.h5 --config data/test_config.json
```

The output HDF5 group key is **content-addressed**: `config_<idx>_<tag>_<method>`, where `<tag>` is a
hash of the physics cell `(Re, Da, α₀, kv, kp, etype)` that is identical across runs. `<idx>` is a
human-readable label only. VTK field export is **opt-in** — set `"write_vtk": true` in the config
(default off, so a re-run does not regenerate GBs of `.vtu`).

### Selecting a sub-combination of factors (no new config needed)

```bash
# AND across keys, OR within repeated keys. Keys: Re, Da, alpha0, kv, kp, etype, method.
julia --project=../../.. run_test.jl test_config.json --filter Re=1e6,etype=QUAD,kv=1
```
`--filter` narrows the config's grid; the full N-ladder always runs so convergence slopes stay
computable. `1e6` matches a config value of `1000000.0` (both canonicalised identically).

### Concurrent launches into ONE shared database

`--shard k/N` deterministically partitions the selected cell list into N disjoint shards. All shards
write the **same** HDF5 file; a `.lock` sidecar (`FileWatching.mkpidlock`) serializes the brief
per-cell writes, and the content-addressed keys keep shards from clobbering each other. No per-worker
files, no post-hoc merge step.

```bash
for k in 1 2 3 4; do
  julia --project=../../.. run_test.jl test_config.json --shard $k/4 > logs/shard_${k}.log 2>&1 &
done
wait
python analyze_results.py --h5 results/<db>.h5 --config data/test_config.json
```

Notes / limits:
- `erase_past_results=true` is rejected with `--shard` (a shard would wipe another's work).
- Sharding is round-robin by count (not cost-balanced); if load balance ever matters, sort the
  selected cells by a cost proxy before sharding — there is no auxiliary-file machinery to add.
- Resume is automatic: completed `(cell, N)` entries are skipped, so re-running or adding shards only
  fills gaps. Mixing a legacy-format DB (`config_<idx>_<method>`) with new content-addressed writes is
  not recommended — start fresh, or keep legacy DBs read-only.

## Configs (`data/`)

| Config | Purpose |
|---|---|
| `test_config.json` | Canonical full reference sweep (Re/Da/α₀ ∈ {1e-6, 1, 1e6}; QUAD+TRI; k=1,2; N=10–320; ASGS). |
| `phase1_{quad,tri}_{k1,k2}.json` | The 2 element-type × 2 order convergence matrix (back the documented results). |
| `phase1_hard_corner.json` | Systematic Da sweep at the extreme Re=1e6, α₀=0.05 corner. |
| `scout_centered.json` | Minimal quick-validation gate (easy corner). |
| `continuation_{c24,c24_rate,c21}.json` | Fold-recovery evidence for `run_continuation.jl` (see `docs/mms_fold_recovery.md`). |

For a quick smoke run, `--filter` down to a single cell rather than authoring a throwaway config:
`run_test.jl test_config.json --filter Re=1.0,Da=1.0,etype=QUAD,kv=1`.

## Quick robustness assessment

A fast multi-regime gate to confirm the harness runs cleanly and to spot convergence regressions —
e.g. before merging a change. Use an **ephemeral** subset config (do **not** commit it; it is trivially
recreated). A representative spread that finishes in a few minutes:

- `Re ∈ {1e-6, 1.0, 1e6}` (Darcy/Stokes → unit → convection-dominated) × `Da=1` × `α₀ ∈ {1.0, 0.5}`
- `ASGS + OSGS`, `k=1`, `QUAD`, `convergence_partitions: [10, 20, 40]`, `max_n_pert: 2`,
  `erase_past_results: false` (so it can be sharded), a unique `h5_filename`.
- Deliberately **omit** the `α₀=0.05` fold corner — it is *expected* to fold and is slow; it is
  covered by `run_continuation.jl` (see [`docs/mms_fold_recovery.md`](../../../docs/mms_fold_recovery.md)).

Copy the solver block from `data/test_config.json`. Run it sharded (this also exercises the shared-DB
concurrency), then analyze:
```bash
for k in 1 2; do julia --project=../../.. run_test.jl <subset>.json --shard $k/2 > logs/rob_$k.log 2>&1 & done; wait
python analyze_results.py --h5 results/<subset>.h5 --config data/<subset>.json --no-plots --outdir /tmp/rob
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
     [`docs/mms_convergence_status.md`](../../../docs/mms_convergence_status.md).
   - **OSGS is flagged on coarse meshes, and this is pre-existing** — the frozen baseline
     `results/phase1_quad_k1.h5` (generated before the harness rework) shows the same class of
     total-failures / disagreements / folds. Low/unit-Re OSGS actually converges to ‖R‖~1e-12 (true
     roots) but trips the strict honest-root gate; high-Re OSGS genuinely diverges on coarse N.
   - **Solver-success vs analyzer "no-root" disagreements** are surfaced by the honest-exit gate:
     `is_true_root` accepts a finest-mesh stop only if `‖R‖ ≤ k_nf · dynamic_ftol`, where `k_nf =
     solver.noise_floor_success_max_ftol_multiple`. A throwaway subset that omits this key leaves the
     solver's gate disabled (base default), so it reports cells as converged that the analyzer flags.
     The real `phase1_*` configs set `noise_floor_success_max_ftol_multiple: 10.0`; **carry the same
     key + finer meshes for a clean pass**, or read the flags as "coarse-mesh, gate-disabled" noise.

Rule of thumb: a *harness/IO/cleanup* change is robust if signal 1 is clean and signal 2 is **unchanged
from the frozen baseline** — convergence quality is a `src/`-level concern, not a harness one.
