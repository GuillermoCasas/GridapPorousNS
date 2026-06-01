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
