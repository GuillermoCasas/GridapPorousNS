# convergence3d archived snapshots

- `convergence3d_results_frontal_c1x1_20260623.json` — Frontal (gmsh alg 4) P1+P2 sweep at the PAPER c1 (c1_mult=1), archived 2026-06-23 before the c1x4 study. P1 ASGS L2u~1.4 (sub-optimal), OSGS optimal; P2 diverges at lc=0.070. The baseline the c1x4 study is recomputed against — the kept authority for this dir.

- `convergence3d_results_k2_TET_structured_c1x4.json` — archived 2026-07-19. A k=2 TET structured-Kuhn c1x4 (=16k⁴) run in the OLD single-DB format (has `interp_l2u`/`ratio`, no per-mesh `levels`). It duplicated the canonical `results/k2/TET/structured/convergence3d_results.json` (same experiment: kv=2, structured_kuhn, c1_mult=4) and sat as a stray VARIANT LEAF next to it, violating the single-channel rule — so it was pulled out of the live results tree and archived here. The canonical DB (per-mesh `levels` + `success` flags) is the authority; this is kept only as provenance of the earlier exploratory run. (Its regenerable `_P2.png` plot is gitignored per repo convention, like all plots.)

Pruned 2026-07-08 (config-lifecycle cleanup): the superseded intermediate snapshots
(`*_pre_cells_*`, `*_pre_frontal_*`, `*_k{1,2}_structured_pre_homotopy_*`) were removed as subsumed by
the frontal_c1x1 baseline above and the current 3D docs (docs/mms/3d-*). Results embed their config, so
any is reconstructable from an archived result if ever needed.
