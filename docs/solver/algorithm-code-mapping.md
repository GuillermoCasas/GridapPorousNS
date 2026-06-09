# Algorithm-to-code mapping

This file is the durable paper-code correspondence for the VMS solver. Each named algorithm box in
[osgs_algorithm.tex](../../theory/osgs_algorithm/osgs_algorithm.tex) maps 1:1 to a single Julia
function. After the shared-core extraction (the commonalities vs per-method differences split), the
solver lives in four files:

- [src/solvers/solver_core.jl](../../src/solvers/solver_core.jl) — the **shared core + orchestrator**:
  the FE containers (`FETopology`/`VMSFormulation`/`StageSolvers`), the `SolutionVerifier` seam, the
  shared cascade machinery (`CascadePolicy` type + `cascade_step_outcome` + `_pingpong_cascade!`), the
  FE-solve plumbing (`safe_fe_solve!`/`_solve_one_step!`/`_record_stage!`), and the orchestrator
  `solve_system`. This is the only solver file that names both methods (it dispatches between them).
- [src/solvers/asgs_solver.jl](../../src/solvers/asgs_solver.jl) — **ASGS only**: the Stage-I boot
  `_initialize_asgs_state!` and the `STAGE_I_POLICY`/`STAGE_I_N2_POLICY` cascade-policy values.
- [src/solvers/osgs_solver.jl](../../src/solvers/osgs_solver.jl) — **OSGS only**: the L²-projection
  helpers, the coupled OSGS solve (`solve_osgs_stage!`), and the `OSGS_INNER_POLICY` value.
- [src/solvers/mms_verification.jl](../../src/solvers/mms_verification.jl) — the optional Algorithm-D
  MMS plateau verification (`MMSPlateauVerifier`), decoupled behind the `SolutionVerifier` seam.

The shared Newton kernel (Algorithm A) is [src/solvers/nonlinear.jl](../../src/solvers/nonlinear.jl).
Helpers are file-local (leading-underscore convention) and not exported.

Reviewers should re-check this table whenever the solver files are touched.

| Algorithm / section | Code symbol | Code file |
|---|---|---|
| Algorithm O — `SimulationOrchestration` | `solve_system` | `solver_core.jl` |
| Algorithm A — `ExactNewtonPipeline` | `_safe_solve_inner!` (via `SafeNewtonSolver`) | `nonlinear.jl` |
| Algorithm B — `RobustNonlinearCascade` (Stage I) | `_initialize_asgs_state!` | `asgs_solver.jl` |
| Algorithm C — `CoupledOSGSSolve` (single Newton; per-eval re-projection of `π`; frozen-`π` Jacobian; Picard fallback gated on `pingpong_enabled`; `stall_window=0`) | `solve_osgs_stage!` | `osgs_solver.jl` |
| Algorithm D — `VerifyMMSPlateau` (ASGS branch) | `on_asgs_converged!(::MMSPlateauVerifier, …)` | `mms_verification.jl` |
| Algorithm D — `VerifyMMSPlateau` (OSGS branch) | `on_osgs_converged!(::MMSPlateauVerifier, …)` | `mms_verification.jl` |
| Scale-free convergence criterion — the **authoritative** outer-iteration success gate (`ε_M ≤ tol_M ∧ ε_C ≤ tol_C`, momentum/mass dimensionless residual measures) | `evaluate_convergence` (momentum envelope `momentum_force_envelope`, mass measure `mass_criterion`); `solve_system` injects it via `build_convergence_probe` into each solver's `conv_probe` | `convergence_criterion.jl` (probe in `solver_core.jl`) |

## The verification seam (Algorithm D)

Algorithm D is no longer inlined in the solver. The core (`solve_system` / `solve_osgs_stage!`) is
verification-blind: at each convergence point it invokes a hook on a `SolutionVerifier`
(`on_asgs_converged!` / `on_osgs_converged!`). Production passes `NoVerification` (multiple dispatch
resolves both hooks to no-ops), so it costs nothing and the core never names MMS, reads an oracle, or
writes an `mms_*` key. The MMS harnesses pass an `MMSPlateauVerifier`, which owns the manufactured-
solution oracle and the plateau loop. The asymmetry in the two hook signatures (`on_asgs_converged!`
takes a `step_once!` closure; `on_osgs_converged!` does not) reflects the real asymmetry in the
algorithm: the ASGS path plateaus by extra single-Newton cycles, while the OSGS coupled solve is a
single solve evaluated once.

## What is *not* in this mapping

- `safe_fe_solve!` (the try/catch + tuple-unwrap wrapper in `solver_core.jl`) and `_solve_one_step!`
  (the raw single-step variant the ASGS verifier drives) are Julia-side utilities, not paper boxes.
- The `SafeNewtonSolver` constructor / `_with_overrides` (`nonlinear.jl`) is pure plumbing.

## Iterator-scheduling helpers (no paper anchor — efficiency, default-off)

Scheduling/plumbing, not paper algorithm boxes; every behavioural switch defaults OFF so the shipped
config reproduces prior results bit-identically.

- `build_iter_solvers` (`nonlinear.jl`) — single construction point for the `(picard, newton)`
  `FESolver` pair, used by both `run_simulation.jl` (production) and `run_test.jl` (MMS harness).
- `cascade_step_outcome` / `CascadePolicy` (`solver_core.jl`) — the shared Algorithm-B accept/reject
  verdict (type + interpreter). It is parameterized by one `CascadePolicy` value per method, each
  declared in its own method file: `STAGE_I_POLICY` / `STAGE_I_N2_POLICY` (`asgs_solver.jl`) and
  `OSGS_INNER_POLICY` (`osgs_solver.jl`), as `(accept_noise_floor, accept_soft_stall,
  max_iters_caught_is_failure)`. Truth tables pinned by `test/blitz/cascade_policy_symmetry_blitz_test.jl`.
- `_pingpong_cascade!` (`solver_core.jl`) — **opt-in** adaptive Newton↔Picard ping-pong (shared by
  the Stage-I boot and the OSGS coupled solve) that replaces the one-way Newton→Picard→Newton cascade
  when `pingpong_enabled`. Runs Newton until it stalls, a Picard segment that stops the moment ‖R‖∞ has
  dropped `pingpong_picard_gain_orders` orders (stop_reason `picard_gain_reached`), then back to
  Newton, bounded by `pingpong_max_swaps`. Each segment is tagged honestly via `_record_stage!`
  (`…:PP[swap]:N` / `:P`).
