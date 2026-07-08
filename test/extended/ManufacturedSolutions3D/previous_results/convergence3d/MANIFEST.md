# convergence3d archived snapshots

- `convergence3d_results_frontal_c1x1_20260623.json` — Frontal (gmsh alg 4) P1+P2 sweep at the PAPER c1 (c1_mult=1), archived 2026-06-23 before the c1x4 study. P1 ASGS L2u~1.4 (sub-optimal), OSGS optimal; P2 diverges at lc=0.070. The baseline the c1x4 study is recomputed against — the kept authority for this dir.

Pruned 2026-07-08 (config-lifecycle cleanup): the superseded intermediate snapshots
(`*_pre_cells_*`, `*_pre_frontal_*`, `*_k{1,2}_structured_pre_homotopy_*`) were removed as subsumed by
the frontal_c1x1 baseline above and the current 3D docs (docs/mms/3d-*). Results embed their config, so
any is reconstructable from an archived result if ever needed.
