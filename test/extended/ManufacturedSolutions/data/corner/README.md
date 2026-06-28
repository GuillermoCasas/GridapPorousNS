# data/corner/ — direct-solve CORNER provenance (TRACKED)

These JSONs are **provenance for the official sweep plots/tables**, so they live here (tracked), not in the
gitignored `results/debug_results/` scratch. They hold the direct-solve / warm-start convergence results for
the `Re=1e6, α₀=0.05` corner cells that the standard sweep lists in `skip_cells` (the convergence "fold"
only clears at very fine meshes, so the monolithic sweep cannot reach a root there).

Pipeline:
- **Produced by** `run_corner_article.jl` (ASGS) and `run_corner_osgs.jl` (OSGS) — see `osgs_corner_lib.jl`.
- **Consumed by** `merge_corner_results.py`, which injects them as content-addressed groups into the official
  per-(kv,etype) DBs (`results/k1/TRI/results.h5`, `results/k2/QUAD/results.h5`) with
  `config_file = "corner_direct_solve"`. `analyze_results.py` / `make_results_tables.py` then plot and
  tabulate all 27 grid cells per family. The merge is idempotent (re-running rewrites the corner groups).

Files:
- `corner_quad_k2_a005.json`      — QUAD k=2 corner, ASGS (N=160→320)
- `corner_quad_k2_a005_osgs.json` — QUAD k=2 corner, OSGS (N=160→320)
- `corner_tri_k1_a005.json`       — TRI  k=1 corner, ASGS (N=512→768)
- `corner_tri_k1_a005_osgs.json` / `corner_tri_k1_a005_osgs_da1e6.json` — TRI k=1 corner, OSGS

To refresh after re-running the corner solves: `python merge_corner_results.py && python analyze_results.py`.
