# Algorithm-to-code mapping

This file is the durable paper-code correspondence for the VMS solver. After the
Phase-1 refactor of [src/solvers/porous_solver.jl](../src/solvers/porous_solver.jl)
each named algorithm box in [osgs_algorithm.tex](osgs_algorithm.tex) maps 1:1 to
a single Julia function. Helpers are file-local (leading-underscore convention)
and not exported. The OSGS-MMS hook (Algorithm D for the OSGS branch) is the
only piece that remains inlined — extracting it would have required returning a
`break` signal through a function boundary.

Reviewers should re-check this table whenever `porous_solver.jl` is touched.

| Algorithm / section | TeX location | Code helper | Code location |
|---|---|---|---|
| Algorithm O — `SimulationOrchestration` | [osgs_algorithm.tex:460](osgs_algorithm.tex#L460) | `solve_system` (orchestration body) | [porous_solver.jl:929-1085](../src/solvers/porous_solver.jl#L929-L1085) |
| Algorithm B — `RobustNonlinearCascade` (Stage I) | [osgs_algorithm.tex:565](osgs_algorithm.tex#L565) | `_initialize_asgs_state!` (H1) | [porous_solver.jl:450-553](../src/solvers/porous_solver.jl#L450-L553) |
| Algorithm B — `RobustNonlinearCascade` (OSGS inner) | [osgs_algorithm.tex:565](osgs_algorithm.tex#L565) | `_run_osgs_inner_cascade!` (H3) | [porous_solver.jl:318-427](../src/solvers/porous_solver.jl#L318-L427) |
| Algorithm C — `OSGSFractionalRelaxation` | [osgs_algorithm.tex:907](osgs_algorithm.tex#L907) | `_run_osgs_relaxation!` (H7) | [porous_solver.jl:708-916](../src/solvers/porous_solver.jl#L708-L916) |
| Algorithm D — `VerifyMMSPlateau` (ASGS branch) | [osgs_algorithm.tex:1334](osgs_algorithm.tex#L1334) | `_run_asgs_mms_extension!` (H2) | [porous_solver.jl:571-664](../src/solvers/porous_solver.jl#L571-L664) |
| Algorithm D — `VerifyMMSPlateau` (OSGS branch) | [osgs_algorithm.tex:1334](osgs_algorithm.tex#L1334) | inlined inside H7 (OSGS-MMS hook) | [porous_solver.jl:~828-895](../src/solvers/porous_solver.jl#L828-L895) |
| §3.6.1 `StateDrift` (`ℓ∞` vs continuous `L²`) | [osgs_algorithm.tex:959](osgs_algorithm.tex#L959) | `_compute_state_drift` (H4) | [porous_solver.jl:187-204](../src/solvers/porous_solver.jl#L187-L204) |
| §3.6.2 `Mode_stop` (stopping-mode decision + ℓ∞ short-circuit) | [osgs_algorithm.tex:1005](osgs_algorithm.tex#L1005) | `_decide_osgs_convergence` (H6) | [porous_solver.jl:270-295](../src/solvers/porous_solver.jl#L270-L295) |
| (no paper label) projection + Anderson mixing per outer iter | — | `_update_and_project!` (H5) | [porous_solver.jl:222-252](../src/solvers/porous_solver.jl#L222-L252) |

## Why H5 has no algorithm-box anchor

H5 (`_update_and_project!`) executes the residual evaluation, `L²` projection, optional
Anderson mixing, and projection-drift computation between H3 (inner cascade) and H6
(convergence decision) of each OSGS outer iteration. The paper folds this work into the
"compute `π_h^{m+1}`" step of Algorithm C; there is no separate box for it. The helper
exists for code-side modularity (it is a pure function — no `diag_cache` writes, no
`println`) rather than to mirror a named paper construct.

## Reciprocal: code helpers seen from the paper

- Stage I (Algorithm B) and OSGS inner (Algorithm B) share the *same* algorithm box
  but have *two* code helpers (H1, H3). This matches the paper, which explicitly notes
  the dual instantiation (see [osgs_algorithm.tex line 308](osgs_algorithm.tex#L308)
  in the algorithm-to-code correspondence table). The two helpers differ in their
  success policy: H1 rejects Newton-1 noise-floor finishes to force Picard homotopy
  (Stage I quadratic-basin guarantee); H3 accepts any non-structural finish (the
  outer OSGS loop re-evaluates anyway).
- The OSGS-MMS hook is inlined inside H7 rather than extracted as a separate `_*!`
  helper. The hook needs to influence the OSGS outer loop's `break` decision and seed
  the MMS error history on the first `S_conv` event; extracting it would require
  threading either a `break_requested` return flag or a callback through the H7
  signature, both of which are uglier than the ~70-line inline block.

## What is *not* in this mapping

- `safe_fe_solve!` (the try/catch + tuple-unwrap wrapper at
  [porous_solver.jl:139](../src/solvers/porous_solver.jl#L139)) is a Julia-side
  utility, not a paper algorithm. The paper's Algorithm A
  (`ExactNewtonPipeline`) corresponds to `_safe_solve_inner!` in
  [src/solvers/nonlinear.jl](../src/solvers/nonlinear.jl), not to anything in
  `porous_solver.jl`. See the paper-side correspondence table at
  [osgs_algorithm.tex line 310](osgs_algorithm.tex#L310).
- The four `SafeNewtonSolver` constructor sites (currently verbose, planned for
  consolidation under a `_with_overrides` helper in Phase 2b of the refactor work)
  are pure plumbing and have no paper anchor.
