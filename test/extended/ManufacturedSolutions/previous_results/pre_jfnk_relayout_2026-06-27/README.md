# Backup: official MMS results immediately BEFORE the 2026-06-27 layout change + JFNK re-sweep

Safety copy of the pre-relayout official results (the old `results/phase1_*.h5` flat-named DBs and the
`k1/`,`k2/` convergence PNGs + reports), taken before:
  1. the H5 layout change (DBs -> per-(kv,etype) `results/k<kv>/<etype>/results.h5` with the full config
     JSON embedded in each DB under group `configs/`, and every result group pointing to its config via
     the `config_file` attribute), and
  2. replacing the official results with a fresh full sweep run under the current (JFNK-enabled) solver.

Contents: the H5 DBs (full data + provenance) and the convergence_*.png plots + reports.
Excluded: traces/ and vtk/ (regenerable from the DBs). The DBs are the complete record.

Pruned 2026-06-28 (cleanup): the two k2-gate A/B snapshots (`_phase1_quad_k2_{lowmidRe,tight}_provenance.h5`)
and the `phase1_quad_k2.h5.bak{2,3}` duplicates — diagnostic cruft, not part of the official pre-relayout
results; the A/B conclusion lives in `docs/mms/high-order-convergence-gate-and-jfnk.md`, and git history
retains the files if ever needed.
