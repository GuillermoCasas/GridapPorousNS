# Solver-efficiency ideas — Newton/Picard scheduling & convergence-gate diagnostics

**Status:** proposal / backlog. Authored 2026-06-04 from a trace-reading session. Nothing here is a
correctness bug — these are efficiency and observability refinements to the nonlinear orchestration.
Implement in a dedicated session; each item below names the exact code site and the validation it
needs before being adopted.

**Scope:** `src/solvers/nonlinear.jl` (Algorithm A — the safeguarded Newton inner solve) and
`src/solvers/porous_solver.jl` (the VMS orchestrator: Exact-Newton → Picard → Newton cascade and the
OSGS staggered loop). See [`lessons_learned.md`](../lessons_learned.md) before editing either — the
solver safeguards are intentional design (CLAUDE.md: *"do not weaken them in pursuit of speed"*).

---

## Background: what the convergence gate actually compares

The inner solve does **not** stop at `‖R‖∞ ≤ ftol`. It stops on a per-field *normalized* residual
([nonlinear.jl:680](../../src/solvers/nonlinear.jl#L680), checked at
[nonlinear.jl:706](../../src/solvers/nonlinear.jl#L706)):

```
f_norm = maxₖ( ‖R[block_k]‖∞ / effective_ftol_per_field[k] ) ≤ 1     ⟺  converged
effective_ftol_per_field[k] = max( ftol , relative_ftol_per_field[k] · ‖R₀_k‖ )
relative_ftol_per_field      = c_sf · h^(kv+1)        # run_test.jl:823, 849
```

i.e. the algebraic target is deliberately pinned to **~1% of the FE discretization error**
`O(h^{kv+1})` — the `dynamic_ftol_ceiling` policy (CLAUDE.md): *"couples the residual tolerance to
mesh resolution so the linear solve doesn't oversolve relative to discretization error."* The scalar
`ftol` is only an absolute machine floor (`≈ 10·eps`), not the working target. The trajectory plots now
draw `f_norm` against a `y = 1` threshold line, so "converged" reads directly as "dots dropped below 1".

This framing is the key to reading the three observations below.

---

## Idea 1 — "Converges in 2 iterations" is correct-by-design, not a lax tolerance

**Observation.** Many cells reach convergence in ~2 Newton iterations; is the tolerance too loose?

**Why it is sound (and why it is *not* a defect).** For a mild / nearly-linear cell, Newton converges
quadratically, so it reaches the `c_sf·h^(kv+1)` target in 1–2 steps. Pushing further would be
**oversolving**: driving the algebraic residual orders of magnitude below the discretization error
cannot reduce the *actual* solution error, which is floored at `O(h^{kv+1})`. So 2 iterations is the
right answer there. Two structural consequences:

- The threshold **tightens with mesh refinement** (`h^{kv+1}` shrinks), so coarse meshes legitimately
  converge in fewer iterations than fine ones. "Many 2-iteration cases" is expected to cluster on the
  coarse-N / mild-(Re,Da) corner of the sweep.
- The honest test of "too lax" is **orthogonal to iteration count**: a tolerance is only too loose if a
  cell converges fast *and* its observed MMS slope underperforms `h^{kv+1}`. Fast convergence + clean
  optimal slope = correct; do not tighten.

**Validation before acting.** Cross-tabulate, per (cell, method, N): iterations-to-converge vs. the
observed convergence rate (`err_u_l2`, `err_u_h1`, `err_p_l2` slopes). Flag only cells that are
*both* fast *and* rate-deficient. (Note: the known OSGS high-Da rate plateau — `Re=1`, `Da=1e6` — is a
*rate* issue already tracked in [`mms_convergence_status.md`](../mms/convergence-status.md); it is not the
tolerance.) **Likely conclusion: no change needed** — this item is mostly a verification, documented so
the question is not re-litigated.

---

## Idea 2 — Adaptive Newton↔Picard ping-pong on stagnation

**Observation.** When the initial Newton stagnates it burns iterations. Proposed policy: the moment
Newton fails to decrease the residual in **two consecutive iterations**, switch immediately to Picard;
once Picard has driven the residual down **1–2 orders of magnitude**, switch back to Newton; repeat.

**Why it is sound.** This is the classic adaptive Newton–Picard globalization: Picard (a contraction
under mild conditions) is robust but linearly convergent and drags the iterate into Newton's quadratic
basin; Newton is fast but only locally convergent and wastes work when far from the basin. Ping-ponging
on a cheap stagnation signal captures the best of both — Picard for globalization, Newton for the final
fast descent — and is well-established practice for stabilized incompressible-flow solvers.

**What already exists (the infra is half-there).**

- **Stall guard — present but switched off in production.** [nonlinear.jl:698](../../src/solvers/nonlinear.jl#L698):
  the no-progress bail (`"no_progress_stall"`) fires only when `stall_window > 0`, and the comment notes
  it is *"Disabled when stall_window == 0 (production single-run default)."* So today a stagnating Newton
  grinds its **entire** `max_iters` budget — exactly the wasted iterations observed. The bail is keyed on
  `best_b_inf` failing to improve by `≥ stall_min_rel_improvement` over `stall_window` iterations
  ([nonlinear.jl:686-704](../../src/solvers/nonlinear.jl#L686)).
- **Cascade — one-way, not a ping-pong.** When Newton *does* fail, the orchestrator runs
  Newton → Picard → **one** Newton retry, both in the ASGS initializer
  ([porous_solver.jl:487-580](../../src/solvers/porous_solver.jl#L487)) and the OSGS inner cascade
  ([porous_solver.jl:343-460](../../src/solvers/porous_solver.jl#L343)). There is no loop that keeps
  swapping back keyed on per-iteration stagnation.

**Implementation, in two tiers.**

1. **Cheap win (low risk): enable the stall guard for ASGS.** Set `stall_window ≈ 2` with a sensible
   `stall_min_rel_improvement`, so Newton bails fast and the existing cascade hands off to Picard. This
   alone recovers most of the wasted iterations and needs no new control flow — it is a documented
   config control, not a safeguard weakening.
2. **Full ping-pong (real change): a new orchestrator mode.** Returning to Newton after Picard gains
   1–2 orders, and repeating, is a genuine new loop, because Exact-Newton and Picard are *different*
   `FEOperator`s (different Jacobian closures — `jac_newton_*` vs `jac_picard_*`). The swap therefore
   lives at the `porous_solver.jl` orchestrator level, not inside Algorithm A. New numerical controls
   (stagnation window, Picard-gain-to-return-to-Newton threshold, max swap count) must go through the
   schema → config struct → JSON → consumer chain per the no-hard-coded-parameters rule.

**Caveats / validation.**

- Watch the interaction with Idea 1: a Newton that stalls for *one* step then drops quadratically must
  not be bailed prematurely. Tune `stall_window`/`stall_min_rel_improvement` so genuine quadratic
  descent survives.
- This is a **performance** optimization (fewer iterations / less wall-time), not a correctness fix.
  Adopt only behind a measured A/B: iteration counts **and** final MMS rates must be unchanged-or-better.
  Do not relabel a Picard step as Exact-Newton (CLAUDE.md invariant).

---

## Idea 3 — Picard is effectively never used inside OSGS (correct reading)

**Observation.** Picard does not appear to run for OSGS in the trace plots.

**Why this is correct — by design.** In the OSGS inner cascade Picard is **only a structural-failure
fallback**: it triggers solely on `:nonfinite`, `:exception`, or
`stop_reason ∈ {linesearch_failed, merit_divergence_escaped, linear_solve_nan}`
([porous_solver.jl:349-379](../../src/solvers/porous_solver.jl#L349)). And OSGS runs with
`osgs_inner_newton_iters = 1`: that single inner step exits via `:max_iters_caught`, which is
**explicitly accepted as outer-loop progress, not a failure** ([porous_solver.jl:352-354](../../src/solvers/porous_solver.jl#L352);
see the P-003 note at [porous_solver.jl:370-378](../../src/solvers/porous_solver.jl#L370) — do not add
`IterationCap` to the failure set without first revising `theory/osgs_algorithm.tex`). Consequently a
well-behaved OSGS cell emits only `C:OSGS[k]:N1` stages — never `:Picard`. Picard surfaces only on
catastrophic cells. So Picard being invisible for OSGS in the traces is **expected**, not a plotting gap.

**Consequence for Idea 2.** With a *single-step* inner solve there is no room inside it for a Picard
smoother — the OSGS "iteration" *is* the outer staggered projection loop
(`_run_osgs_relaxation!`, [porous_solver.jl:765](../../src/solvers/porous_solver.jl#L765)). The Idea-2
ping-pong therefore applies cleanly to **ASGS**, and to OSGS only if `osgs_inner_newton_iters > 1`. Any
OSGS ping-pong experiment must first raise the inner-step budget, which itself changes the
encoding-covariance story (the covariant inner-gate/warmup fixes were tuned for `inner = 1`; see
[`lessons_learned.md`](../lessons_learned.md) 2026-06-02) — so re-run `encoding_invariance_test.jl` if the
inner budget changes.

---

## Idea 4 — Short-circuit OSGS plateau-verification cycles when the state is already at the machine floor

**Observation.** The OSGS outer loop often runs ~10 iterations with little or nothing evolving past
the first ~3. The substantive work is front-loaded; the rest are static.

**Why this happens (it is mostly *not* the fixed-point solving).** The outer loop is a staggered
fixed-point on `(U_h, π_h)` (`_run_osgs_relaxation!`, [porous_solver.jl:802-943](../../src/solvers/porous_solver.jl#L802)):

- **Iter 1**: `π_h^0 = 0`, so this *is* the ASGS problem ([porous_solver.jl:789-797](../../src/solvers/porous_solver.jl#L789)) → `U_h` = ASGS solution.
- **Iter 2**: first non-trivial `π_h` from `R(U_ASGS)`; `U_h` re-solved → the OSGS correction.
- **Iter ≥3**: for mild cells, `U_h`/`π_h` are already at the fixed point.

There **already is a stop criterion** — the loop `break`s the moment it fires:

- state-drift gate `x_diff ≤ max(osgs_tol, stagnation_tol)` ([porous_solver.jl:297](../../src/solvers/porous_solver.jl#L297), mode `state_drift`);
- MMS plateau gate: once base-converged, requires `max_r < tau_err` for `require_consecutive_passes`
  consecutive cycles, then breaks ([porous_solver.jl:904-925](../../src/solvers/porous_solver.jl#L904)).

So the count cap is **not** the intended iteration number. It is
`eff_osgs_iters = osgs_iterations + mms_max_extra_cycles` ([porous_solver.jl:770-772](../../src/solvers/porous_solver.jl#L770)) —
for the k=1 sweep, `5 + 5 = 10` — a **budget ceiling** that fully runs only when convergence is never
cleanly declared. The "static" tail iterations are therefore one of:

- **(a) benign** — base convergence happened early and iters 4-10 are **MMS plateau-verification
  re-solves** (re-solve → recompute error → confirm `< tau_err` over `require_consecutive_passes`
  cycles, [porous_solver.jl:880-927](../../src/solvers/porous_solver.jl#L880)). They look static because
  *confirming staticity is their job*;
- **(b) the open rate issue** — `Re=1, Da=1e6` OSGS never cleanly base-converges/plateaus, so the loop
  runs the full budget and exits `"mms_budget_exhausted"` ([porous_solver.jl:949-955](../../src/solvers/porous_solver.jl#L949)).
  Tracked in [`mms_convergence_status.md`](../mms/convergence-status.md) — a criterion never *satisfied*,
  not a missing criterion.

**The refinement (targets case (a)).** When the state drift has already reached the machine floor
(`x_diff ≤ osgs_tol`, with `osgs_tol = 1e-10` in the sweep), `U_h` is at a fixed point. The MMS error is
a **deterministic function of `U_h`** (the oracle `mms_cfg.oracle(final_x0...)`), so it provably **cannot
change** on the next cycle. Requiring `require_consecutive_passes` additional full inner-solve +
error-eval cycles to "verify" a plateau that is already frozen is redundant work for the easy cells.

Concretely: when `x_diff ≤ osgs_tol` (true machine-floor convergence, distinct from the looser
`stagnation_tol` short-circuit), the plateau is trivially satisfied — accept it in **one** confirming
cycle instead of `require_consecutive_passes`. This is **recognizing** the criterion holds, not
**weakening** it (CLAUDE.md: *"Weakening plateau criteria to pass a flaky sweep is forbidden"* — this
does not weaken; it skips re-confirming a provably-static quantity).

**Implementation sketch.**

- In the MMS verification branch ([porous_solver.jl:880-927](../../src/solvers/porous_solver.jl#L880)), gate
  the "needs `require_consecutive_passes`" requirement: if the *state drift that produced this iterate*
  was `≤ osgs_tol` (not just `≤ dynamic_osgs_tol = max(osgs_tol, stagnation_tol)`), treat one passing
  plateau check as sufficient. This needs the raw `x_diff` (already computed at
  [porous_solver.jl:839](../../src/solvers/porous_solver.jl#L839)) threaded into the decision.
- Keep the `rate_check_factor` sub-optimal-rate flag intact ([porous_solver.jl:918-921](../../src/solvers/porous_solver.jl#L918)) —
  it is a *quality* flag, not a stop criterion, and must still fire.
- Do **not** apply the short-circuit when `x_diff` only met the looser `stagnation_tol` (the iterate may
  be stalled, not converged) or in `projection_drift`/`both` modes where `π_h` may still be moving.

**Caveats / validation.**

- Purely an efficiency change for *already-converged* cells; the hard cells (case b) are untouched
  because they never reach `x_diff ≤ osgs_tol`. So this must **not** alter any final error or rate.
- Validate with a before/after MMS run: identical `err_u_l2`/`err_u_h1`/`err_p_l2` and identical
  `mms_stop_reason` per cell, with a reduced outer-iteration count only on the benign cells. Any change
  in a *final error* means the short-circuit is firing where the state was not truly frozen — back it out.
- New numerical control (if any threshold is introduced beyond reusing `osgs_tol`) goes through the
  schema → config struct → JSON → consumer chain per the no-hard-coded-parameters rule.

---

## Suggested order of work

1. **(c)** Idea 1's empirical cross-check (iterations vs. rate) once the k=1 sweep lands — change with
   data in hand, and likely confirm no tolerance change is warranted.
2. **(d)** Idea 4 — OSGS plateau short-circuit at the machine floor (low risk, bounded by an
   exact-error A/B; the most direct answer to the "10 mostly-static outer iterations" observation).
3. **(a)** Idea 2 tier 1 — measured `stall_window` experiment for ASGS (cheap, reversible, config-only).
4. **(b)** Idea 2 tier 2 — prototype the full Newton↔Picard ping-pong as a new orchestrator mode (real
   change; new schema controls; A/B on iterations + rates).

---

## Idea 5 — `freeze_after_k`: coupled warm-up, then freeze π (IMPLEMENTED, 2026-06-05)

**Status: implemented.** New `osgs_projection_coupling = "freeze_after_k"` mode in `_run_osgs_relaxation!`
(`porous_solver.jl`). It is the user's "project a few nonlinear iterations, then freeze" scheme:

1. **Warm-up** — `osgs_freeze_after_k` (=k) *single-step* staggered iterations: each takes ONE Newton step
   against the current frozen π, then re-projects `π = Π(R(u))`. So the projection is refreshed at EVERY
   nonlinear iteration for the first k iterations (the lagged map; ~linearly contracting). Because π is
   frozen *within* each single step, every warm-up step is a valid Newton step (the `D=−2Φ` line-search
   certificate holds) — π only changes *between* steps, so no two-phase line-search hack is needed.
2. **Freeze + finish** — freeze `π = π_k` and run a full Newton solve to convergence. With π constant the
   Jacobian is the *exact* tangent of `R(·;π_k)` (no `∂π/∂u` term), so the finish converges **quadratically**.

**Why it is correct (rate vs constant).** `U*(π_k)` is optimal-**rate** for *any* fixed k: the paper's
stability/convergence theory rests on the π-independent bilinear form `B_S` (`article.tex` 553-557; the
theorem is proved for π=0), so **ASGS is already optimal-order and orthogonality only shrinks the error
*constant*.** k slides the constant from ASGS (k=0) toward OSGS (k→∞) without leaving the optimal-rate band.

**The projection is ALWAYS applied — there is no ASGS fallback.** (An earlier draft added a velocity-H¹
relative-drift gate that fell back to ASGS in the reaction-dominated corner; it was removed at the user's
direction — "always do the projection in all cases.") Consequence: in that corner the OSGS *fixed point
itself* is sub-optimal (the staggered-map defect, caveat #5 in
[`../mms/convergence-status.md`](../mms/convergence-status.md)), so `freeze_after_k` there converges toward
that sub-optimal point and those cells **honestly show OSGS's poor rate** — the mode does NOT mask the
defect behind ASGS. Where the map is well-behaved (most of the parameter space), k=2-3 gives optimal rate +
~all of the OSGS error-constant advantage in ~ASGS iteration counts.

**A/B:** the full k=1 QUAD sweep (`data/k1_quad_freeze.json`, N=10→320) is the characterization run; numbers
to be filled from `results/k1_quad_freeze.h5` + `merged_convergence_report.md`. Earlier staggered-warm-up
probes (since superseded by the single-step warm-up) showed, on the *mild* cell, freeze k=3 ≈ optimal rate
(2.1/1.0) at ~8-9 iters/mesh capturing ~99% of the OSGS L² gain vs full-OSGS's ~40; on the *defect* cell
(Da=1e6) the partially-frozen result tracks OSGS's own sub-optimal/negative H¹ rate (error grows with
refinement) — expected, and now shown rather than gated away.

Knob: `osgs_freeze_after_k` (k≥1; k=2-3 default-good). Config-strictness pinned by
`test/blitz/freeze_after_k_config_blitz_test.jl`; encoding-covariance of the mode pinned by the
`freeze_after_k` case in `test/quick/encoding_invariance_quick_test.jl`.
