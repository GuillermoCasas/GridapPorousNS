# Algorithm-to-code mapping

This file is the durable paper-code correspondence for the VMS solver. After the
Phase-1 refactor of [src/solvers/porous_solver.jl](../../src/solvers/porous_solver.jl)
each named algorithm box in [osgs_algorithm.tex](../../theory/osgs_algorithm.tex) maps 1:1 to
a single Julia function. Helpers are file-local (leading-underscore convention)
and not exported. The OSGS-MMS hook (Algorithm D for the OSGS branch) is the
only piece that remains inlined — extracting it would have required returning a
`break` signal through a function boundary.

Reviewers should re-check this table whenever `porous_solver.jl` is touched.

| Algorithm / section | TeX location | Code helper | Code location |
|---|---|---|---|
| Algorithm O — `SimulationOrchestration` | [osgs_algorithm.tex:460](../../theory/osgs_algorithm.tex#L460) | `solve_system` (orchestration body) | [porous_solver.jl:974-1133](../../src/solvers/porous_solver.jl#L974-L1133) |
| Algorithm B — `RobustNonlinearCascade` (Stage I) | [osgs_algorithm.tex:565](../../theory/osgs_algorithm.tex#L565) | `_initialize_asgs_state!` (H1) | [porous_solver.jl:482-596](../../src/solvers/porous_solver.jl#L482-L596) |
| Algorithm C — `CoupledOSGSSolve` (single Newton; per-eval re-projection of `π`; frozen-`π` Jacobian; Picard fallback gated on `pingpong_enabled`; `stall_window=0`) | [osgs_algorithm.tex:1220](../../theory/osgs_algorithm.tex#L1220) | `_run_osgs_relaxation!` (H7) | [porous_solver.jl:749-961](../../src/solvers/porous_solver.jl#L749-L961) |
| Algorithm D — `VerifyMMSPlateau` (ASGS branch) | [osgs_algorithm.tex:1334](../../theory/osgs_algorithm.tex#L1334) | `_run_asgs_mms_extension!` (H2) | [porous_solver.jl:614-707](../../src/solvers/porous_solver.jl#L614-L707) |
| Algorithm D — `VerifyMMSPlateau` (OSGS branch) | [osgs_algorithm.tex:1334](../../theory/osgs_algorithm.tex#L1334) | inlined inside H7 (OSGS-MMS hook) | [porous_solver.jl:869-951](../../src/solvers/porous_solver.jl#L869-L951) |

## Reciprocal: code helpers seen from the paper

- The OSGS-MMS hook is inlined inside H7 rather than extracted as a separate `_*!`
  helper. The hook needs to influence the OSGS outer loop's `break` decision and seed
  the MMS error history on the first `S_conv` event; extracting it would require
  threading either a `break_requested` return flag or a callback through the H7
  signature, both of which are uglier than the ~70-line inline block.

## What is *not* in this mapping

- `safe_fe_solve!` (the try/catch + tuple-unwrap wrapper at
  [porous_solver.jl:147](../../src/solvers/porous_solver.jl#L147)) is a Julia-side
  utility, not a paper algorithm. The paper's Algorithm A
  (`ExactNewtonPipeline`) corresponds to `_safe_solve_inner!` in
  [src/solvers/nonlinear.jl](../../src/solvers/nonlinear.jl), not to anything in
  `porous_solver.jl`. See the paper-side correspondence table at
  [osgs_algorithm.tex line 310](../../theory/osgs_algorithm.tex#L310).
- The four `SafeNewtonSolver` constructor sites (now consolidated via the
  `_with_overrides` helper in [src/solvers/nonlinear.jl:93](../../src/solvers/nonlinear.jl#L93))
  are pure plumbing and have no paper anchor.

## Iterator-scheduling helpers (no paper anchor — efficiency, default-off)

These were added by the 2026-06 iterator optimization (`docs/solver/efficiency-ideas.md`). They are
scheduling/plumbing, not paper algorithm boxes, and every behavioural switch defaults OFF so the shipped
config reproduces prior results bit-identically.

- `build_iter_solvers` ([src/solvers/nonlinear.jl](../../src/solvers/nonlinear.jl)) — single construction
  point for the `(picard, newton)` `FESolver` pair, used by both `run_simulation.jl` (production) and
  `run_test.jl` (MMS harness). Removes the structural divergence between the two solver-construction sites.
- `cascade_step_outcome` / `CascadePolicy` (`porous_solver.jl`) — the Stage-I (H1) Algorithm-B cascade's
  accept/reject verdict, parameterized by the `STAGE_I_POLICY` / `STAGE_I_N2_POLICY` constants encoding the
  B5/B6 success asymmetry as `(accept_noise_floor, accept_soft_stall, max_iters_caught_is_failure)`. The truth
  tables are pinned by `test/blitz/cascade_policy_symmetry_blitz_test.jl`.
- `_pingpong_cascade!` (`porous_solver.jl`) — **opt-in** adaptive Newton↔Picard ping-pong that
  replaces the one-way Newton→Picard→Newton cascade in H1 (the Stage-I ASGS boot) when `pingpong_enabled`.
  Runs Newton until it stalls, a Picard segment that stops the moment it has driven ‖R‖∞ down `pingpong_picard_gain_orders`
  orders (the Picard solver's `picard_gain_target`, stop_reason `picard_gain_reached`, checked in
  `_safe_solve_inner!`), then back to Newton, bounded by `pingpong_max_swaps`. Each segment is tagged honestly
  via `_record_stage!` (`…:PP[swap]:N` / `:P`). Default off ⇒ H1 runs its existing one-way body.
