# mesh_viz — visualise the meshes a test uses

`mesh_plot.py` draws the finite-element mesh behind any test and annotates the
figure with the data you actually want when inspecting a mesh: element type,
element / vertex counts, domain extent, mean cell area, and the **average
element characteristic length `h_K`** — computed with the *same* formula the
solver uses for the stabilization parameter τ (see `src/run_simulation.jl`):

```
h_K = sqrt(area_K)        for QUAD cells
h_K = sqrt(2 * area_K)    for TRI cells (simplexified Cartesian or unstructured)
```

For FreeFem meshes the paper's reported `h = sqrt(2)/N` is also shown when `N`
is recoverable from the filename.

## Two mesh families

1. **Structured Cartesian** — built procedurally from a JSON config's
   `domain.bounding_box` + `numerical_method.mesh.{partition, element_type}`.
   `element_type` is `"QUAD"` or `"TRI"` (the Cartesian mesh `simplexify`-d).
2. **Explicit FreeFem `.msh`** — the unstructured Delaunay meshes shipped as
   data with the Cocquet irregular-mesh experiments.

## Direct CLI use

```bash
# Cartesian mesh from a config (uses that config's own partition):
python tools/mesh_viz/mesh_plot.py --config path/to/config.json

# ... overriding the partition for one convergence level:
python tools/mesh_viz/mesh_plot.py --config cfg.json --partition 80 40 --element-type TRI

# Explicit FreeFem mesh file:
python tools/mesh_viz/mesh_plot.py --msh path/to/mesh_fig2_N40.msh

# Save instead of showing interactively:
python tools/mesh_viz/mesh_plot.py --msh foo.msh --out foo.png
```

## Per-test wrappers

Each mesh-bearing test directory has a tiny `plot_mesh.py` that already knows its
config path and its mesh-density convention, so you don't have to. They write one
PNG per convergence level into that test's (gitignored) `results/meshes/`.

| Test directory | Convention |
| --- | --- |
| `CocquetExperiment`, `CocquetAlpha1`, `CocquetDeviatoric`, `CocquetLinearReaction`, `CocquetAllDirichlet`, `CocquetExperimentModifiedCorner`, `CocquetExperimentLiteralPicard` | TRI, `N -> [2N, N]` over `[0,2]x[0,1]` |
| `CocquetFormMMS` | TRI, `N -> [N, N]` over `[-0.5,0.5]^2` |
| `ManufacturedSolutions` | QUAD + TRI, `N -> [N, N]` over `[-0.5,0.5]^2` |
| `CocquetExperimentIrregularMesh` | explicit FreeFem `.msh` files |

```bash
python test/extended/CocquetExperiment/plot_mesh.py            # all convergence levels
python test/extended/CocquetExperiment/plot_mesh.py 20 40      # only N=20, N=40
python test/extended/CocquetExperiment/plot_mesh.py --show     # display first interactively
python test/extended/CocquetExperiment/plot_mesh.py --ref      # also the huge reference mesh
```

Adding a wrapper to a new test is ~10 lines: point `CONFIG` at the config and
pass the right `partition_rule` (`mp.square_partition` or `mp.cocquet_partition`)
to `wc.run_cartesian`, or call `wc.run_freefem` for a `.msh` directory. See any
existing `plot_mesh.py` for the template.

Requires `numpy` + `matplotlib` (already used by the other `plot_*.py` scripts).
