# CocquetTubeTest — unified Cocquet tube-flow convergence study

One config-driven convergence test for the Cocquet et al. (2020) porous tube-flow benchmark
(Re=500, `c_in`=0.5, transverse exponential porosity `ε(y)=0.45+0.55·e^{y-1}` on `[0,2]×[0,1]`).
Every experimental variant is a JSON config under [`data/`](data/); the single
driver [`run_convergence.jl`](run_convergence.jl) reads one config and writes to the parallel folder
`results/<variant>/` with a **uniform** structure and error-diagnostic suite, so all variants are
directly comparable.

This replaces the former sibling directories `CocquetExperiment`, `CocquetExperimentIrregularMesh`,
`CocquetExperimentIrregularMeshFreefemDivs`, `CocquetExperimentLiteralPicard`,
`CocquetExperimentModifiedCorner`, `CocquetAlpha1`, `CocquetDeviatoric`, `CocquetLinearReaction`,
`CocquetAllDirichlet`. (The manufactured-solution sibling `CocquetFormMMS` is separate and unchanged.)

## Run

```bash
cd test/extended/CocquetTubeTest
julia --project=../../.. run_convergence.jl data/structured/paper_comparison.json
python plot_convergence.py structured          # -> results/structured/convergence.png
python plot_mesh.py structured                 # -> results/structured/meshes/*.png
```

The first CLI argument is the path to the variant config; the parent folder name (`structured`, …)
names the parallel results folder `results/<name>/`.

## Variants (`data/<name>/`)

| Variant | Config | What it changes vs the structured benchmark |
|---|---|---|
| `structured` | `paper_comparison.json` | the baseline: structured Cartesian-simplexified TRI, 3-way VMS P1/P1 + P2/P2 + Cocquet Galerkin P2/P1 |
| `all_dirichlet` | `all_dirichlet.json` | `boundary_policy=all_dirichlet` (outlet pinned to the inlet parabola) + `quadrature_degree_bonus=12` |
| `alpha_one` | `alpha1.json` | `alpha_profile=constant` (α≡1 ⇒ plain NS, no porosity weighting) |
| `deviatoric` | `deviatoric.json` | `viscous_operator_type=DeviatoricSymmetric` (the canonical/MMS operator) |
| `linear_reaction` | `linear_reaction.json` | `sigma_nonlinear=0` (Forchheimer \|u\| term off, linear Darcy drag only) |
| `modified_corner` | `paper_comparison_modified_corner.json` | `boundary_policy=modified_corner` (outlet-corner wall tags released) |
| `unstructured_gmsh` | `paper_comparison_irregular.json` | `mesh_generator=UNSTRUCTURED` (gmsh Delaunay `mesh_algorithm=5`, uniform 1/N edges) |
| `unstructured_frontal` | `paper_comparison_frontal.json` | `mesh_algorithm=6` (gmsh **Frontal-Delaunay**, near-equilateral elements) — best-quality unstructured; recovers the Taylor-Hood O(h²) slope. Galerkin ordered first. |
| `freefem_meshes` | `paper_comparison_freefem.json` | Cocquet's literal FreeFem `.msh` files from `meshes/` (`freefem_mesh_dir=meshes`) |
| `freefem_divisions` | `paper_comparison_irregular_freefem_divs.json` | gmsh with `wall_divisions=freefem` (N-per-border, walls 2× coarser) |
| `literal_picard` | `paper_comparison_literal_picard.json` | capped pure-Picard Cocquet protocol (`method=Galerkin_LiteralPicard`, `picard_iterations=10`) |

## Config keys (test-harness keys, stripped before strict-schema parsing)

Beyond the strict solver schema, each config carries variant-selecting keys with sensible defaults:

- `mesh_generator` — `"STRUCTURED"` (default) | `"UNSTRUCTURED"`
- `wall_divisions` — `"uniform"` (default) | `"freefem"`
- `freefem_mesh_dir` — relative (to the config folder) dir of FreeFem `.msh` files; `""` = generate
- `mesh_algorithm` — gmsh algorithm id (default `5`)
- `boundary_policy` — `"standard"` (default) | `"all_dirichlet"` | `"modified_corner"`
- `alpha_profile` — `"exponential"` (default) | `"constant"`
- `quadrature_degree_bonus` — extra quadrature degree (default `0`)
- `comparison_runs` — explicit list of `{k_velocity, k_pressure, method, porosity_order?}` runs;
  `method` ∈ `ASGS` / `OSGS` (VMS) / `Galerkin` (Taylor-Hood) / `Galerkin_LiteralPicard`.

## Results layout (`results/<variant>/`, gitignored)

- `convergence.h5` — one uniform schema for all variants: per `<method>/P<kv>P<kp>` group,
  `errors_{l2,h1}_{u,p}` (consistent norms), `errors_*_trial` (trial-projection),
  `interp_{l2,h1}_{u,p}` (**interpolation-error floor** — see below), S3 probes
  (`chi_Omega_*`, `cellavg_frac_*`, `l2_cellavg_*`, `l2_domainmean_*`) and corner-excluded matrices
  (`{l2,h1}_e{u,p}_corner_excl`), plus timing/iteration counts.

The **interpolation-error floor** `interp_{l2,h1}_{u,p}` is computed on every run (the MMS-harness
practice — cf. `test/extended/ManufacturedSolutions/run_interpolation_reference.jl`): it is the error
of the nodal interpolant of the N=200 reference onto each coarse space, `‖u_ref − I_h u_ref‖`, measured
with the SAME consistent (fine-mesh) metric as the FE-solution error. Its slope is the OPTIMAL slope
achievable on the mesh sequence and its magnitude the optimal constant, so each FE row becomes an
efficiency ratio FE/interp (printed live as `eff L2(u)`; drawn as a thin dotted floor by
`plot_convergence.py`). This turns a bare rate check into an efficiency check and separates a
pre-asymptotic *solution-roughness* slope loss from a genuine *method* deficiency.
- `config.json` — a copy of the exact config that produced the run (self-describing provenance).
- `vtk/` — per-solve VTU exports.

The error suite is uniform across variants: the reference solution's cross-mesh `Interpolable` is
always built with a KDTree tolerant search (`num_nearest_vertices=10`), which is required on
non-nested (unstructured/FreeFem) meshes and rate-neutral on structured ones.

See [`../../../docs/cocquet/`](../../../docs/cocquet/) for the investigation history and
[`theory/cocquet/cocquet_formulation.tex`](../../../theory/cocquet/cocquet_formulation.tex) for the
exact unstabilized-Galerkin (Cocquet) formulation.
