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
| Algorithm B — `RobustNonlinearCascade` (OSGS inner) | [osgs_algorithm.tex:565](../../theory/osgs_algorithm.tex#L565) | `_run_osgs_inner_cascade!` (H3) | [porous_solver.jl:342-459](../../src/solvers/porous_solver.jl#L342-L459) |
| Algorithm C — `OSGSFractionalRelaxation` | [osgs_algorithm.tex:970](../../theory/osgs_algorithm.tex#L970) | `_run_osgs_relaxation!` (H7) | [porous_solver.jl:749-961](../../src/solvers/porous_solver.jl#L749-L961) |
| Algorithm D — `VerifyMMSPlateau` (ASGS branch) | [osgs_algorithm.tex:1334](../../theory/osgs_algorithm.tex#L1334) | `_run_asgs_mms_extension!` (H2) | [porous_solver.jl:614-707](../../src/solvers/porous_solver.jl#L614-L707) |
| Algorithm D — `VerifyMMSPlateau` (OSGS branch) | [osgs_algorithm.tex:1334](../../theory/osgs_algorithm.tex#L1334) | inlined inside H7 (OSGS-MMS hook) | [porous_solver.jl:869-951](../../src/solvers/porous_solver.jl#L869-L951) |
| §3.6.1 `StateDrift` (`ℓ∞` vs continuous `L²`) | [osgs_algorithm.tex:959](../../theory/osgs_algorithm.tex#L959) | `_compute_state_drift` (H4) | [porous_solver.jl:211-228](../../src/solvers/porous_solver.jl#L211-L228) |
| §3.6.2 `Mode_stop` (stopping-mode decision + ℓ∞ short-circuit) | [osgs_algorithm.tex:1005](../../theory/osgs_algorithm.tex#L1005) | `_decide_osgs_convergence` (H6) | [porous_solver.jl:293-318](../../src/solvers/porous_solver.jl#L293-L318) |
| (no paper label) projection + Anderson mixing per outer iter | — | `_update_and_project!` (H5) | [porous_solver.jl:246-276](../../src/solvers/porous_solver.jl#L246-L276) |

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
  the dual instantiation (see [osgs_algorithm.tex line 308](../../theory/osgs_algorithm.tex#L308)
  in the algorithm-to-code correspondence table). The two helpers differ in their
  success policy: H1 rejects Newton-1 noise-floor finishes to force Picard homotopy
  (Stage I quadratic-basin guarantee); H3 accepts any non-structural finish (the
  outer OSGS loop re-evaluates anyway).
- **[P3a] That success-policy difference is now a single explicit decision, not divergent
  control flow.** Both helpers route their accept/reject verdict through the pure
  `cascade_step_outcome(res, policy::CascadePolicy)` (in `porous_solver.jl`, just above H3),
  parameterized by three policy constants — `STAGE_I_POLICY`, `STAGE_I_N2_POLICY`,
  `OSGS_INNER_POLICY` — encoding the documented B5/B6 asymmetry as
  `(accept_noise_floor, accept_soft_stall, max_iters_caught_is_failure)`. Stage-I Newton-2
  needs its own constant because it accepts a noise-floor finish though Newton-1 rejects it,
  and rejects soft stalls though OSGS-inner accepts them (three independent flags). The truth
  tables are pinned by `test/blitz/cascade_policy_symmetry_blitz_test.jl`. **B6 note:** the
  asymmetry only triggers when a Newton-1 finishes exactly at `stagnation_noise_floor_reached`
  (the narrow high-Re fold regime); on ordinary stiff cells N1 hits `ftol_reached` or a soft
  stall, so a quantitative accept-vs-reject A/B is vacuous there. Current behavior retained.
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
- `cascade_step_outcome` / `CascadePolicy` (`porous_solver.jl`, above H3) — the shared Algorithm-B success
  verdict (see the reciprocal note above).
- `_pingpong_cascade!` (`porous_solver.jl`, above H3) — **opt-in** adaptive Newton↔Picard ping-pong that
  replaces the one-way Newton→Picard→Newton cascade in both H1 and H3 when `pingpong_enabled`. Runs Newton
  until it stalls, a Picard segment that stops the moment it has driven ‖R‖∞ down `pingpong_picard_gain_orders`
  orders (the Picard solver's `picard_gain_target`, stop_reason `picard_gain_reached`, checked in
  `_safe_solve_inner!`), then back to Newton, bounded by `pingpong_max_swaps`. Each segment is tagged honestly
  via `_record_stage!` (`…:PP[swap]:N` / `:P`). Default off ⇒ H1/H3 run their existing one-way bodies.

## OSGS projection-coupling modes (inside H7 `_run_osgs_relaxation!`)

`_run_osgs_relaxation!` dispatches on `stab_cfg.osgs_projection_coupling` *before* the staggered loop:

- `"staggered"` (default; paper `alg:StationarySystem`) — the Algorithm-C outer fixed-point loop above.
- `"coupled"` — a single Newton solve whose residual recomputes `π = Π(R(u))` each evaluation with a
  frozen-π (local, sparse) Jacobian; a Picard-type coupling, linearly convergent to the same fixed point.
- `"freeze_after_k"` (2026-06-05; `efficiency-ideas.md` Idea 5) — `osgs_freeze_after_k` (k) single-step
  warm-up iterations (each a 1-step `_run_osgs_inner_cascade!` with `max_iters=1`, re-projecting π between)
  that refresh π at every nonlinear iteration, then a full frozen-π Newton finish (`_run_osgs_inner_cascade!`
  at full budget) — quadratic, since π is then constant. The projection is ALWAYS applied (no ASGS fallback;
  an interim H¹-drift gate was removed at the user's direction). No paper-box anchor; an efficiency mode.
  Optimal rate for any fixed k (orthogonality buys the error *constant*, not the rate — `B_S` is
  π-independent); in the reaction-dominated corner it tracks OSGS's own sub-optimal fixed point honestly.
  See `osgs_algorithm.tex` and `efficiency-ideas.md` Idea 5.
