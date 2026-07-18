# Committed base meshes for the 3D MMS irregular (nested-red) family

## Why these files are tracked

The irregular 3D family of `sec:NumericalExamplesStationaryCase3D` is **one** unstructured gmsh base
mesh plus **deterministic** uniform red (1→8) refinement, so the base mesh alone fixes every level
(verified: 425 → 3400 → 27200 cells, exactly ×8 per level, mean h halving exactly).

Until 2026-07-17 that base was generated into an `mktempdir()` and deleted (`mesh3d.jl`
`build_box_tet_model`), `save_msh` existed but no caller passed it, and the gmsh version was not
pinned. The published irregular-family numbers therefore could not be guaranteed to correspond to any
regenerable mesh — a violation of the reproducible-results rule (`.agents/rules/reproducible-results.md`):
the parameters→results link was severed by construction.

This matters more here than it would elsewhere. gmsh's unstructured tet generator is **not** contractually
stable across versions — a different gmsh may return a different, equally valid mesh — and this family's
results are known to be sensitive to precisely its **element-quality tail** (`docs/mms/p2-3d.md`: the red
tail degrades `qmin` 0.47→0.25→0.15 under refinement). A regenerated base is a *new* family, not a
reproduction of the published one.

`mesh3d.jl:load_or_build_base_mesh` therefore **prefers loading** the committed file and only generates
(and persists, loudly) when it is absent.

## Provenance

| file | lc | gmsh algorithm | cells | mean h | generated with |
|---|---|---|---|---|---|
| `nested_red_base_lc0.200_alg1.msh` | 0.2 | 1 (MeshAdapt) | 425 | 0.174593 | gmsh 4.9.3 via GridapGmsh 0.7.4, 2026-07-17 |

Nested family derived from it (`build_nested_family`): L0 425 cells (h̄ 0.174593), L1 3400 (0.0872963),
L2 27200 (0.0436482).

## Regenerating deliberately

Delete the file and re-run; `load_or_build_base_mesh` will regenerate and persist, and warn that the
result is a new family. Do **not** do this and then compare against previously published numbers as if
they were the same mesh sequence — archive the old file first.
