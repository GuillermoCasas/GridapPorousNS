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
- **Cascade — now landed as a ping-pong (`_pingpong_cascade!`).** The full Newton↔Picard ping-pong
  described below landed: it is gated on `pingpong_enabled` and returns to Newton once Picard gains
  `pingpong_picard_gain_orders`, keyed on the `newton_stall_window` stall sensor. It drives the ASGS
  Stage-I boot and (when `pingpong_enabled`) the coupled Stage-II OSGS solve. (Historically, before the
  2026-06-08 coupled-only leaning, this was a one-way Newton → Picard → **one** Newton retry, and a
  separate OSGS inner cascade — both since superseded.)

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

**Why this is correct — by design.** For a well-behaved OSGS cell Newton converges on its own, so the
Picard fallback never has to fire — Picard surfaces only on catastrophic cells. So Picard being
invisible for OSGS in the traces is **expected**, not a plotting gap.

**Consequence for Idea 2.** Post the 2026-06-08 coupled-only leaning, OSGS is **one coupled Newton
solve** whose residual re-projects `π = Π(R(u))` at every evaluation (no staggered outer loop, no
single-step inner solve). The Stage-II coupled solve already carries the Idea-2 ping-pong (gated on
`pingpong_enabled`), so the smoother applies directly there as well as in the ASGS Stage-I boot — there
is no longer an inner-step budget to raise first.

---

## Idea 4 — Short-circuit OSGS plateau-verification cycles when the state is already at the machine floor — **SUPERSEDED / REVERTED**

**Status: superseded by the 2026-06-08 coupled-only leaning** — see
[`coupled-only-leaning-and-jfnk-plan.md`](coupled-only-leaning-and-jfnk-plan.md). This idea optimized the
*staggered outer loop* (`_run_osgs_relaxation!`) and its plateau-verification cycles, all of which were
**deleted** in that leaning: the staggered loop, the `_decide_osgs_convergence` gate, the
`projection_drift`/`both` stopping modes, and the multi-cycle plateau re-confirmation are all gone. The
coupled solve now performs **one** Newton solve and sets `mms_plateau_reached` in a single shot, so there
are no static tail iterations to short-circuit. Retained here only as a record of the pre-leaning loop
shape; do not act on it.

---

## Suggested order of work

1. **(c)** Idea 1's empirical cross-check (iterations vs. rate) once the k=1 sweep lands — change with
   data in hand, and likely confirm no tolerance change is warranted.
2. **(d)** ~~Idea 4 — OSGS plateau short-circuit at the machine floor.~~ **Superseded** by the
   2026-06-08 coupled-only leaning (the staggered outer loop it optimized was deleted) — see
   [`coupled-only-leaning-and-jfnk-plan.md`](coupled-only-leaning-and-jfnk-plan.md).
3. **(a)** Idea 2 tier 1 — measured `stall_window` experiment for ASGS (cheap, reversible, config-only).
4. **(b)** Idea 2 tier 2 — the full Newton↔Picard ping-pong landed as `_pingpong_cascade!`
   (`pingpong_enabled`); remaining work is the A/B on iterations + rates.

---

## Idea 5 — `freeze_after_k`: coupled warm-up, then freeze π — **REVERTED**

**Status: reverted** — see [`coupled-only-leaning-and-jfnk-plan.md`](coupled-only-leaning-and-jfnk-plan.md)
section 2. The `osgs_projection_coupling = "freeze_after_k"` mode (and the `osgs_freeze_after_k` knob,
its `_run_osgs_relaxation!` warm-up-then-freeze machinery, and its blitz/encoding tests) was removed in
the 2026-06-08 coupled-only leaning. OSGS is now a single coupled Newton solve that re-projects
`π = Π(R(u))` at every residual evaluation, so there is no warm-up phase to freeze after. Retained as a
record of the explored design only; the mode no longer exists.

---

## Idea 6 — Enrichment: make future OSGS diagnosis self-contained from the saved data

**Status: proposal / backlog.** (Migrated from the now-deleted `docs/mms/next-actions.md` section 5 — the
only salvage from that doc.)

Store, per mesh, in the HDF5 group attrs + the JSON trace sidecar: `tau1`/`tau2` (min / max /
representative over Ω), `sigma`, `|u|_max`, the `encoding_strategy`, and the `L`/`U` scale factors.
Optionally surface `tau1` and the σ-share-of-`(1/tau1)` on the trajectory plot.

The solver already has all of these at each mesh — it is **only plumbing** into the `run_test.jl` HDF5
write and the trace sidecar. The payoff is that the encoding / τ-balance story becomes readable straight
from the saved data, so an OSGS rate diagnosis (e.g. the high-Da coercivity-gap cases) does not require
re-instrumenting or re-running the solver.
