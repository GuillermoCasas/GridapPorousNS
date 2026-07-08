# CocquetTubeTest unification — status & pending verification

**Status header:** `[code-actual]` living status doc for the 2026-07-08 refactor that unified the nine
sibling Cocquet convergence tests into one config-driven test. Canonical usage guide:
[`test/extended/CocquetTubeTest/README.md`](../../test/extended/CocquetTubeTest/README.md).

## What was achieved

The nine near-duplicate Cocquet convergence directories were collapsed into a single, config-driven
test at [`test/extended/CocquetTubeTest/`](../../test/extended/CocquetTubeTest/):

- **One driver** — `run_convergence.jl` is the union of the former structured (S3/corner-multimask)
  and irregular (gmsh/FreeFem + KDTree) driver families. All variant axes are now explicit,
  config-driven test-harness keys (read with `get(dict, key, default)`, stripped before strict-schema
  parsing): `mesh_generator`, `wall_divisions`, `freefem_mesh_dir`, `mesh_algorithm`, plus the newly
  **lifted** `boundary_policy` (`standard`/`all_dirichlet`/`modified_corner`), `alpha_profile`
  (`exponential`/`constant`) and `quadrature_degree_bonus`. Method labels (`ASGS`/`OSGS`/`Galerkin`/
  `Galerkin_LiteralPicard`) select the execution path. Previously these flips were hardcoded per-dir.
- **Shared helpers** — `galerkin_driver.jl`, `literal_picard_driver.jl` at the test root (not per
  variant); consolidated `plot_convergence.py` + `plot_mesh.py`.
- **10 variant configs** under `data/<variant>/` (filenames unchanged to preserve the
  parameters→results provenance link): `structured`, `all_dirichlet`, `alpha_one`, `deviatoric`,
  `linear_reaction`, `modified_corner`, `unstructured_gmsh`, `freefem_meshes` (+ the FreeFem `.msh`
  sequence under `meshes/`), `freefem_divisions`, `literal_picard`.
- **Uniform results/logic** — every variant runs the same code path and writes the same HDF5 schema
  into the parallel folder `results/<variant>/` (`convergence.h5` + copied `config.json` + `vtk/`).
  The reference cross-mesh `Interpolable` is always built with a KDTree tolerant search
  (`num_nearest_vertices=10`) — required on non-nested meshes, rate-neutral on structured ones — so
  the S3 mode-decomposition, corner-excluded multimask, and trial-projection diagnostics are computed
  identically for all variants (no structured/unstructured branch).
- **Cleanup** — 7 debugging leftovers from closed investigations were removed (git history retains
  them): `plot_results.py`, `print_h5.py`, `read_h5.jl`, `freefem_recipe_diag.jl`, `localize_err.jl`,
  `xfamily_diag.jl`, `plot_h_a_squared_overlay.py`.
- **Docs** — all references in `CLAUDE.md`, `docs/cocquet/*`, `docs/lessons_learned.md`,
  `theory/cocquet/cocquet_formulation.tex` repointed to the new layout; the removed diagnostic scripts'
  "kept as evidence" notes updated (findings stay recorded; scripts recoverable from git history).

## Verification done

- **Blitz** — 272/272 pass, no failures/tier warnings (`julia -O0 -t 1 test/run_blitz_tests.jl`).
- **`structured` variant — full end-to-end run passes and reproduces the documented baseline
  exactly**: Cocquet Galerkin P2/P1 L²(u) = **3.01e-4 @N=10** and **2.31e-5 @N=80** (matches the
  baselines recorded in [`convergence-analysis.md`](convergence-analysis.md)). Confirms the
  uniform-KDTree error suite is behavior-preserving on structured meshes. Uniform HDF5 schema,
  `config.json`, and `vtk/` all produced; both plotters render (`convergence.png`, mesh PNGs).

## Pending (deferred 2026-07-08 — verification only, not a code gap)

The remaining **9 variants have NOT yet been run end-to-end** through the new driver. `structured`
already exercises the hardest shared paths (VMS + Galerkin execution, the full uniform diagnostic
suite, the `results/<name>/` layout), so risk is low, but these still need a confirming run:

- **Config-lifted switches** (structured mesh, low risk): `all_dirichlet` (outlet-Dirichlet +
  `quadrature_degree_bonus=12`), `alpha_one` (`alpha_profile=constant`), `deviatoric`, `linear_reaction`,
  `modified_corner` (corner-untag mesh policy).
- **Alternate mesh sources** (exercise gmsh / on-disk FreeFem load / capped-Picard dispatch):
  `unstructured_gmsh`, `freefem_meshes`, `freefem_divisions`, `literal_picard`.

Resume the full matrix (multi-minute per config; run when it won't compete with more urgent jobs):

```bash
cd test/extended/CocquetTubeTest
for c in all_dirichlet/all_dirichlet.json alpha_one/alpha1.json deviatoric/deviatoric.json \
         linear_reaction/linear_reaction.json modified_corner/paper_comparison_modified_corner.json \
         unstructured_gmsh/paper_comparison_irregular.json freefem_meshes/paper_comparison_freefem.json \
         freefem_divisions/paper_comparison_irregular_freefem_divs.json \
         literal_picard/paper_comparison_literal_picard.json ; do
    julia --project=../../.. run_convergence.jl "data/$c"
done
```

Expected per-variant checks: run exits 0; `results/<variant>/convergence.h5` has the uniform schema;
the numbers reproduce the corresponding rows recorded in `convergence-analysis.md` /
`investigation-synthesis.md` (structured variants may differ from the pre-refactor consistent-norm
values only at the last digit near the boundary, from the switch to KDTree point-location — this is
rate-neutral).
