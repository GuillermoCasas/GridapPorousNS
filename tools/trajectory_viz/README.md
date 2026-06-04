# trajectory_viz — nonlinear-solver trajectory diagrams

Shared, **generic** tool for turning a solver run's stored trajectory into a PNG "trajectory
diagram", so a failure mode (where the initial guess went, which sub-algorithm diverged/stalled,
how the residual evolved) is obvious at a glance.

It renders **one trajectory = an ordered list of algorithm stages** into a single-column vertical
diagram. It knows nothing about any test's notion of "runs", "attempts", eps_pert homotopy, sweep
cells, etc. — those are test-specific and belong in each test's small wrapper.

## Pipeline

1. The Julia solver records every iteration of the Newton/Picard kernel (Algorithm A) into
   `SafeSolverResult.iteration_history`; the orchestrator (`src/solvers/porous_solver.jl`,
   Algorithm O → B/C) tags each sub-algorithm invocation and assembles `diag_cache["trajectory"]`.
2. The driver (e.g. `test/extended/ManufacturedSolutions/run_test.jl`) writes per-run JSON sidecars
   under `results/traces/traj_*.json`.
3. A per-test wrapper reads those, splits them however that test wants, and calls `plot_stages`.

## API

```python
import trajectory_plot as tp
tp.plot_stages(stages, out_path, title="...", subtitle="...", subtitle_color="#c62828")
# helpers for building mathtext titles consistently:
tp.fmt_pow(1e6)  # -> "10^{6}"
tp.sci(1.8e6)    # -> "1.8\\times10^{6}"
```

`stages` is a list of stage dicts (`stage`, `state`, `stop_reason`, `res_in`, `res_out`,
`res_in_norm`, `res_out_norm`, `history=[{i,f_inf,f_norm,merit,step_inf,alpha,accepted}]`). `title`
is drawn black, `subtitle` in `subtitle_color`. The `*_norm` fields hold the **normalized residual**
`f_norm = maxₖ(‖R[block_k]‖_∞ / effective_ftol_per_field[k])` — the scalar the per-field
convergence gate compares to 1; `res_in`/`res_out`/`f_inf` are the raw inf-norm `‖R‖_∞`. The plotter
prefers the normalized residual and falls back to the inf-norm for traces that predate it.

## Nomenclature (matches `theory/osgs_algorithm.tex`)

| code | meaning |
|------|---------|
| Alg. O | Orchestrator (`solve_system`): Stage I then Stage II |
| Alg. B | `RobustNonlinearCascade` — Newton-1 → Picard → Newton-2; once in Stage I (ASGS init) and once per OSGS outer iteration |
| Alg. C | OSGS outer staggered loop (Stage II) |
| Alg. A | `ExactNewtonPipeline` — the Newton/Picard kernel that produces the per-iteration residual dots |

Stage codes: `B:StageI:{N1,Picard,N2}`, `C:OSGS[k]:{N1,Picard,N2}`.

## Diagram

- Stage **boxes** stack **top-to-bottom** in execution order; each has a label **on top** giving
  the algorithm part + step + `stop_reason`, and a downward arrow into the next box.
- Inside each box: **residual evolution** dots (one per iteration, log-y) from the entry residual to
  the stage's final residual. When the trace carries the **normalized residual** the y-axis is
  `‖R‖/tol` with a dashed reference line at **y=1** (dots below it are converged); otherwise it falls
  back to the raw inf-norm `‖R‖_∞`. Filled dot = accepted step; hollow red = rejected (line-search)
  step; ■ = entry, ★ = exit. (Threshold-line style lives in the `threshold` block of `plot_params.json`.)
- START node (initial guess) above the first box; END node (final residual + outcome) below — both
  labelled `‖R‖/tol` when normalized, else `‖R‖`.
- Boxes colour-coded: green = converged, red = diverged/linesearch-failed/stalled, amber = cap.
- All mathematical symbols use matplotlib mathtext (`$...$`); no system LaTeX needed.

## Per-test wrapper

The wrapper owns the test-specific structure. The MMS sweep
(`test/extended/ManufacturedSolutions/plot_trajectory.py`) loops the homotopy `eps_pert` attempts of
each run and writes **one independent PNG per attempt**:

```bash
python plot_trajectory.py                                      # every attempt of every trace
python plot_trajectory.py --cell Re=1e6,Da=1e-6,a0=0.5 --N 160 --method ASGS
python plot_trajectory.py --file results/traces/traj_....json
```
