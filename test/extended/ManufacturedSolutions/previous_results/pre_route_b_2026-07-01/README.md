# pre_route_b_2026-07-01 — archived pre-Route-B 2D MMS official sweep

**Archived:** 2026-07-02, when the official 2D sweep was rebuilt under **Route B** (the Philosophy-A
algebraic mass gate; code commit `e455f36`, branch `feat/route-b-algebraic-mass-gate`).

## What this is

The **pre-Route-B** official 2D MMS results — i.e. the sweep as it stood on `HEAD = 05d2ad4` (mass gate =
the old strong-form `ε_C` with the loose `eps_tol_mass = 0.8`). Moved here verbatim from
`results/k{1,2}/…` so the Route-B rerun could write fresh official DBs without resume-skipping the old
cells. Kept as the **backup baseline** for the Route-B behavior-preservation check.

## Contents

| path | what |
|---|---|
| `k1/QUAD/results.h5` | pre-Route-B k1 QUAD DB |
| `k1/TRI/results.h5`  | pre-Route-B k1 TRI DB |
| `k2/QUAD/results.h5` | pre-Route-B k2 QUAD DB |
| `merged_convergence_report.md`, `flagged_cells.json` | reports from that run |

Each DB is self-describing: the config JSON(s) that produced it are embedded under the `configs/` group,
and every result group carries a `config_file` attr. `traces/` are gitignored (`**/traces/`).

## ⚠ Comparison caveat (important)

These DBs are **merges** of `phase1_*.json` (24 cells) + `phase1_*_preJFNK.json` (24 cells) — the JFNK
re-run only re-ran part of the grid under the current config, leaving the rest under the older pre-JFNK
config. A per-cell diff that ignores the `config_file` attr will read the **worse preJFNK cells** for half
the grid and spuriously report a ~4× regression. For a clean Route-B-vs-baseline comparison:

- **k1 QUAD:** compare against `../validated_k1_quad_N640/phase1_quad_k1.h5` (the clean gold, N→640), at
  the common N=320. (Route B matched it to median Δe_u ≈ 6.5e-7, 40/48 within 1%.)
- **k1 TRI / k2 QUAD:** filter these DBs by `config_file == 'phase1_tri_k1.json'` / `'phase1_quad_k2.json'`
  to isolate same-config cells, or note the preJFNK provenance per cell.

## Status / provenance pointer

The live Route-B sweep (partial: k1 done, k2 paused at 39/48) and the resume instructions are documented
in [docs/mms/convergence-2d.md](../../../../../docs/mms/convergence-2d.md).
