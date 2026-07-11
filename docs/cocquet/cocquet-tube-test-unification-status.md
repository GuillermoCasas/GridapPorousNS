# CocquetTubeTest unification — status & pending verification

**Status header:** `[code-actual]` living status for the 2026-07-08 refactor that unified the nine sibling
Cocquet convergence tests into one config-driven test at
[`test/extended/CocquetTubeTest/`](../../test/extended/CocquetTubeTest/). Canonical usage guide — the durable
record of the layout, driver, and all variant axes — is
[`test/extended/CocquetTubeTest/README.md`](../../test/extended/CocquetTubeTest/README.md). The 10 variant
configs live under `data/<variant>/` (filenames unchanged to preserve the parameters→results provenance link).
Remaining-variants backlog: [`../pending-tasks.md`](../pending-tasks.md) §7f.

## Design decision worth keeping — always build the cross-mesh reference with a KDTree search

Every variant builds the reference `Interpolable` with a KDTree tolerant search (`num_nearest_vertices=10`),
**not** a structured/unstructured branch. It is **required** on non-nested meshes and **rate-neutral** on
structured ones, so the S3 mode-decomposition, corner-excluded multimask, and trial-projection diagnostics are
computed identically for all variants (no branch). Consequence: a structured re-run may differ from the
pre-refactor consistent-norm values only at the last digit near the boundary (the KDTree point-location switch —
rate-neutral).

## Verification status

- **Blitz** — 272/272 pass.
- **`structured` variant — full end-to-end run reproduces the documented baseline exactly**: Cocquet Galerkin
  P2/P1 L²(u) = **3.01e-4 @N=10** and **2.31e-5 @N=80** (matches
  [`../archive/cocquet-convergence-analysis.md`](../archive/cocquet-convergence-analysis.md)). Confirms the
  uniform-KDTree error suite is behavior-preserving on structured meshes; the uniform HDF5 schema, `config.json`,
  and `vtk/` are all produced.
- **The other 9 variants have NOT yet been run end-to-end** (verification only, not a code gap — `structured`
  already exercises the hardest shared paths: VMS + Galerkin execution, the full diagnostic suite, the
  `results/<name>/` layout). Resume the full matrix (multi-minute per config):

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

Per-variant checks: run exits 0; `results/<variant>/convergence.h5` has the uniform schema; numbers reproduce
the recorded rows (structured variants may differ only at the last digit near the boundary — rate-neutral).
