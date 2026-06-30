# 3D Manufactured-Solutions harness (paper §5.2)

Self-contained convergence study for the 3D MMS case of
[theory/paper/article.tex](../../../theory/paper/article.tex) §5.2 (tables `tab:3DL2` / `tab:3DH1`):
a z-extruded 2D field on the parallelepiped `(0,1)×(0,1)×(0,0.3)`, fixed
`(α₀, Re, Da) = (0.5, 1, 1)`, solved with `P1` (4 meshes) and `P2` (3 meshes), both **ASGS** and **OSGS**.
The meshes are a **nested red-refined tetrahedral family**: one coarse gmsh tet mesh subdivided `1→8`
per level, so `h` halves exactly at each level.

## Files

| file | role |
|------|------|
| `mesh3d.jl` | `build_nested_family(nlevels; lc, domain)` — base gmsh tet mesh + recursive red refinement. |
| `mms3d.jl` | the z-extruded exact-solution oracle (`Paper3DMMS`, exactness diagnostics). |
| `smoke3d.jl` | the solve driver + sweep/cells runners (see below). |
| `plot_convergence3d.py` | log–log convergence plots (`results/convergence3d_P{1,2}.png`), 2D-harness format. |
| `plot_mesh3d.py` | the nested-family mesh mosaic (`results/mesh/mesh_levels.png`). |

Results live under `results/` (gitignored); deliberately-archived snapshots under `previous_results/`
(tracked).

## Linear solver — "choose whatever fits"

Each solve auto-picks the inner linear backend by problem size (`solve_one`, `linsolver="auto"`):

- **`ndof ≤ LU_DOF_LIMIT`** (`80 000`) → exact sparse **`LUSolver`** (the small meshes).
- **`ndof > LU_DOF_LIMIT`** → low-memory **`ILU_GMRES`** (`ILUGMRESSolver`), for the fine 3D OSGS
  systems whose LU fill-in would exhaust a ~32 GB machine.

The backend choice does **not** change the converged solution. It is exposed in production as a config
knob, `numerical_method.solver.linear_solver` (`method = "LU" | "ILU_GMRES"` plus the ILU/GMRES
parameters); `base_config.json` defaults to `LU` (zero behaviour change). Force a backend for validation
with `solve_one(...; linsolver="ILU_GMRES")`.

## Running

### Official sweep — regular (structured) mesh, run like the 2D harness

`sweep_structured` is the **official** §5.2 sweep on the **regular structured Kuhn-tet mesh** (the
"regular mesh"), driven exactly like the 2D harness: each cell uses the **eps_pert homotopy**
(perturbed start, hard→easy `1 → 0.1 → 0.01 → 0`, first success wins) on top of the **Codina
iterative penalty** (`ε_num·(pⁿ−pⁿ⁻¹)` in the mass residual — required for the 3D all-Dirichlet
problem, ill-posed at `ε=0`; see [docs/mms/3d-iterative-penalty-fix-and-osgs-coupling.md](../../../docs/mms/3d-iterative-penalty-fix-and-osgs-coupling.md)).
The headline robustness metric per cell is `eps_used` = the **largest** perturbation it still converged
from (`eps_used=1` ⇒ converged from the hardest start, exactly the 2D behaviour).

```bash
cd test/extended/ManufacturedSolutions3D

# Official sweep: ALL ASGS first (P1 then P2), THEN all OSGS, both at paper c₁. Writes per-(kv,etype)
# to results/k{1,2}/TET/structured/convergence3d_results.json (each record self-describes its solver
# recipe). The trailing 2 is max_n_pert (homotopy depth: eps_pert = 1, 0.1, 0.01, 0).
julia --project=../../.. smoke3d.jl sweep_structured 2
python plot_convergence3d.py structured     # -> results/k{1,2}/TET/structured/convergence3d_P{1,2}.png
```

Per-method **solver recipe** (recorded in each JSON record's `solver` block for reproducibility):
| method | recipe | rationale |
|---|---|---|
| **ASGS** | default coupled solve + ASGS Stage-I boot | the honest production path; converges robustly at paper c₁ |
| **OSGS** | boot-skip + matrix-free **JFNK** | recovers the dropped `∂π/∂u`; also fast-fails doomed perturbations so the homotopy descent stays practical (the default coupled+boot OSGS grinds ~15 min per failing attempt) |

Both paths use the iterative penalty (`ε_num = 1e-4·α₀/(ν·(1+Re+Da)) ≈ 1.667e-5` here) and the
`Deviatoric` viscous operator with the `Constant_Sigma` reaction. Prior structured runs are archived to
`previous_results/convergence3d/` before each launch.

### Legacy modes — nested red-refined family (aggregate JSON)

```bash
cd test/extended/ManufacturedSolutions3D

# Full sweep (P1: 4 meshes, P2: 3; ASGS+OSGS). Heavy: the finest OSGS meshes are large.
julia --project=../../.. smoke3d.jl sweep

# Memory-capped, RESUMABLE sweep: ASGS keeps full mesh counts (P1=4, P2=3); OSGS is capped to the
# meshes that fit RAM with a direct solver (P1=3, P2=2). Builds only to level 2 (never the 223744-tet
# mesh). Seeds from any existing JSON and skips already-present (kv,method) blocks.
julia --project=../../.. smoke3d.jl sweep_capped

# Plots + tables
python plot_convergence3d.py
python plot_mesh3d.py
python ../ManufacturedSolutions/make_results_tables.py   # 3D tables tab:3DL2 / tab:3DH1
```

## Running the larger OSGS meshes later

The capped sweep deliberately skips the finest OSGS meshes (P1 level-3 = 223 744 tets; P2 level-2 ≈
150 K DOF) because a direct LU factorization OOMs on a 32 GB machine — a **hardware** limit, not a method
one (the 3D OSGS solve is from the exact-solution guess, ~2 Newton iters; only peak RAM is the blocker).
To run them later and merge **provenance-safely**, use the on-demand `cells` mode. It runs the requested
`(kv, method, n_meshes)` blocks at full mesh count (auto-picking `ILU_GMRES` for the large meshes), and
**before overwriting any capped block it snapshots the current JSON into `previous_results/convergence3d/`
plus a manifest line** — so the capped (paper-committed) numbers stay reconstructable to the exact config
that produced them.

```bash
# Upgrade OSGS to the full ladders (P1→4 meshes, P2→3). ILU-GMRES kicks in automatically on the big cells.
julia --project=../../.. smoke3d.jl cells \
  results/convergence3d_results.json "kv=1,method=OSGS,n=4;kv=2,method=OSGS,n=3"

# Regenerate artifacts — the consumers prefer the now-full ladder and the memory-cap footnote auto-drops.
python plot_convergence3d.py
python ../ManufacturedSolutions/make_results_tables.py
```

Validate the iterative solver before trusting the finest cell: run a coarser full block first
(e.g. `"kv=2,method=OSGS,n=3"`) and confirm the rates (`L²u~h³`, `H¹u~h²`, `L²p~h²` for P2). If RAM is
ample (64 GB+), the same command works with the default direct `LU` backend — just keep meshes below
`LU_DOF_LIMIT` or raise it.
