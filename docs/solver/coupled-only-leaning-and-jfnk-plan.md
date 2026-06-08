# OSGS solver leaning → single `coupled` route, and the JFNK speed plan

**Status:** CANONICAL. Records (1) the 2026-06-07 decision to collapse the OSGS nonlinear
solver to a single route (`coupled`) and the evidence that forced it, and (2) the deferred
**JFNK** enhancement that recovers near-quadratic speed for that route (to be implemented in a
separate session, per user). Supersedes the "keep `freeze_after_k`" recommendation in the
clean-slate redesign synthesis — see "Why not freeze_after_k" below.

---

## 1. Decision: OSGS = one nonlinear route, `coupled`

The three OSGS coupling modes (`staggered`, `coupled`, `freeze_after_k`) were three implementations
of **one** inexact-Newton method — drop the dense `∂π/∂U` tangent term, keep the cheap sparse
frozen-π Jacobian, and update `π = Π(R(U))` in the *residual* at some cadence. They differed only in
that cadence (and whether they end with a frozen-π polish). We keep exactly one:

> **`coupled`** — a single Newton solve whose residual recomputes `π = Π(R(u))` at **every**
> evaluation (no staggering lag); the Jacobian stays the local frozen-π form (sparse, non-monolithic).
> The per-eval projection is the Cholesky-cached mass solve. Implemented at
> [`porous_solver.jl` `_run_osgs_relaxation!`](../../src/solvers/porous_solver.jl) (the function is now
> coupled-only; the dispatch and the other two branches were deleted).

**Why `coupled`:** it is the **fewest-parts** route (no warm-up loop, no freeze, no `k`, no freeze
criterion, no drift/stopping satellite) and it is **robust on the whole grid** — it re-projects `π`
every iteration, so it always tracks the true OSGS discrete fixed point. The price is **speed at high
Re** (see §3), which the JFNK plan (§4) addresses without re-growing the machinery.

### 1a. Two refinements to the coupled solve (2026-06-08)

The "single Newton solve" above gained two safeguards after hands-on review of the production traces.
Both live in the coupled branch of `_run_osgs_relaxation!`:

1. **Picard fallback (gated on `pingpong_enabled`).** The coupled solve is now wrapped in the same
   `_pingpong_cascade!` as the Stage-I boot: if the coupled Newton's line search fails or it diverges,
   it hands off to a frozen-advection (Oseen) Picard segment (wider basin), then back to Newton once
   Picard has cleared `pingpong_picard_gain_orders`. Converging cells run Newton straight to `ftol`
   (Picard inert). This gives the genuinely-hard `linesearch_failed` cells a real fallback instead of
   quitting after one step. (It does **not** rescue the α₀=0.05 / N=10 cells — those are below the
   resolution floor for the porosity layer and fail even under Picard; they converge at N≥20.)

2. **The stall sensor is DISABLED on the coupled solve (`stall_window=0`) — load-bearing.** The Stage-I
   boot's Newton stall sensor (`newton_stall_window`) must **not** ride onto the coupled solve. The
   coupled inexact-Newton converges *slowly-monotone* (linear rate, the dropped `∂π/∂U`); the stall
   sensor would mistake a slow step for a stall and bail after ~2 steps, leaving `U` at the ASGS state —
   **OSGS silently degenerates into ASGS** and reports ASGS's optimal rate under the OSGS label. This
   was a real regression (2026-06-08); see `lessons_learned.md`. The coupled solve builds its Newton and
   Picard solvers via `_with_overrides(base_nls; stall_window=0)`; its genuine failures (line-search,
   divergence, max-iters) are still caught, and the boot keeps the stall sensor (cold-start oscillation).

## 2. Why NOT `freeze_after_k` (the oracle that forced the decision)

The redesign synthesis recommended keeping `freeze_after_k` (k slaved-π warm-ups + a frozen-π
quadratic finish) because its finish is fast. Before deleting anything, we gated on a **coupling
equivalence oracle** (`coupling_equivalence_extended_test.jl` — a one-time pre-deletion check, since
retired; see the note after the table): does `freeze_after_k` reach the same fixed point as `coupled`?
It does **not**:

| cell | metric | coupled | **freeze_after_k=2** |
|------|--------|---------|------|
| Re=10⁶, Da=10⁶, α=1 (healthy) | H¹ rate | ≈1.0 | ≈1.0 (just a ~27% larger L²-constant) |
| **Re=1, Da=10⁶, α=1 (reaction-dominated)** | H¹(u), N=10/20/40 | 0.902 / 0.609 / 0.419 (converging) | **1.168 / 1.283 / 1.615 (DIVERGING)** |
| | H¹ local rate | 0.57, 0.54 | **−0.14, −0.33** |

In the reaction-dominated corner, `π` needs ~17 warm-up steps to settle (the π_p relative-drift tail);
freezing at `k=2` captures a wildly-wrong `π`, and the frozen-π finish then converges `U` to a wrong
solution that gets **worse with mesh refinement** (negative convergence rate). So a *fixed* small `k`
is non-viable as the sole path. (An *adaptive* freeze — warm up until the relative-π-drift gate fires,
then finish — would also be robust and faster than `coupled`, but it re-introduces the warm-up loop +
criterion + finish; rejected on the "fewest parts" criterion. `coupled` + JFNK is preferred.)

This is the value of the gate: it caught that the "obvious" fast path silently breaks the one corner
that matters most, **before** any deletion. The oracle test was retired in the 2026-06-08 code-debt
cleanup — with `freeze_after_k` removed there is no second route left to compare against, so it had
become a coupled-vs-coupled tautology; its evidence lives on in this section and in `lessons_learned.md`.

## 3. What was deleted (the leaning)

[`src/solvers/porous_solver.jl`](../../src/solvers/porous_solver.jl): **1,590 → 954 LOC (−636)**.
- The `staggered` outer-relaxation branch and the `freeze_after_k` branch of `_run_osgs_relaxation!`
  (the dispatch collapsed to coupled-only).
- The now-orphaned helpers `_compute_state_drift` (H4), `_update_and_project!` (H5),
  `_decide_osgs_convergence` (H6), `_run_osgs_inner_cascade!` (H3).

**Kept (each earns it):** the ASGS Stage-I boot (`_initialize_asgs_state!`); the shared cascade
policy (`CascadePolicy` / `cascade_step_outcome` / `OSGS_INNER_POLICY`, used by the coupled solve);
`_pingpong_cascade!` (the boot's Newton↔Picard cascade); `safe_fe_solve!`; the per-field relative gate
on frozen ‖R₀‖; the `k_nf` honest-exit; `dynamic_ftol_ceiling`; the Cholesky-cached mass; and the
reaction-projection trim. Verified: Blitz 189/189 after each stage.

**Remaining cleanup (dead config, not yet removed):** `osgs_projection_coupling`, `osgs_freeze_after_k`,
`osgs_stopping_mode`, `osgs_state_drift_scale`, `osgs_projection_tolerance`, `osgs_warmup_*`, ping-pong
knobs, `ablation_mode`, the inert off-switches — all now ignored by the coupled path. They remain in the
schema / `StabilizationConfig` / configs (so nothing breaks) and can be retired in a follow-up.

## 4. JFNK plan (deferred to a separate session)

**Diagnosis — `coupled`'s only weakness is exactly the dropped `∂π/∂U`.** The exact tangent of
`F(U) = R(U; π(U))` with `π(U)=Π(R(U))` is `J = J_frozen + (∂R/∂π)(∂π/∂U)`, and `∂π/∂U = M⁻¹(∂B/∂U)` is
**dense** (the `M⁻¹`). `coupled` drops that second term → inexact Newton → **linear** convergence.
Proof it is the bottleneck: `freeze`'s finish uses the **same** frozen-π Jacobian but with `π` *frozen*,
so its tangent is exact → **quadratic** (12–27 iters incl. warm-up), while `coupled` re-projects `π`
→ inexact → **44–69 iters at high Re**. The *only* difference is whether `π` moves. Recover the action
of `∂π/∂U` and `coupled` becomes as fast as the frozen finish — simple **and** fast **and** robust.

**Note:** the `ExactNewtonMode` "∂Π/∂u" terms already in the code (`_get_dsigma_du`, `_get_dtau_1_du`,
…) are the τ/σ/convection derivatives inside the *frozen-π* stabilization — **not** the projection
coupling `∂π/∂U`. There is no existing term for it.

**Options, best first:**
1. **JFNK (Jacobian-free Newton–Krylov) — recommended.** Solve the Newton system with GMRES where the
   matvec is `J·v ≈ [F(U+εv) − F(U)]/ε`. Because `F` **re-projects π** at `U+εv`, the finite difference
   captures the *full* tangent — including `∂π/∂U` — **exactly** (to FD accuracy), for one extra
   residual eval per Krylov vector (each eval = one cheap Cholesky-cached projection). **Precondition
   GMRES with the frozen-π Jacobian we already assemble and factor** (`jac_fn_coupled`); since it is
   `J` minus the compact coupling term, GMRES converges in a handful of vectors. Net: near-quadratic
   Newton, a few extra residual evals/step, **one clean mechanism** consistent with the lean. (Knoll &
   Keyes 2004.) Julia: Krylov.jl / IterativeSolvers GMRES, matrix-free operator wrapping `res_fn_coupled`.
2. **Lumped-mass sparse approximation.** Approximate `M⁻¹ ≈ diag(M)⁻¹` (M is diagonally dominant) so
   `∂π/∂U ≈ diag(M)⁻¹(∂B/∂U)` becomes **sparse and addable** to the Jacobian — stays in the direct-solve
   world, no Krylov. Cost: assembling `∂B/∂U` (the strong-residual-projection linearization).
3. **Anderson acceleration.** `AcceleratorConfig` infra still exists; depth-m Anderson on the coupled
   iterates → low-rank history-based curvature approximation → superlinear-ish. Cheap; instability risk.
4. **Broyden** rank-1 secant updates from `(ΔF, ΔU)` → superlinear; middle-ground effort.

**Before implementing:** run the leaned `coupled` k=1 sweep first and record the **actual** high-Re
iteration counts as the baseline — that quantifies how much of the 44–69 iters is the π-coupling
(JFNK-addressable) vs the convection nonlinearity, and gives a clean before/after for JFNK.

**Acceptance test for JFNK:** identical converged MMS errors to plain `coupled` (same fixed point — JFNK
only changes the path), with materially fewer nonlinear iterations at high Re, on the reaction-dominated
and healthy Da=10⁶ cells (Re∈{1, 10⁶}, α=1, N=10/20/40) and the high-Re fold cells.
