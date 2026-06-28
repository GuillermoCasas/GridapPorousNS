# High-order MMS: the convergence-gate lesson + JFNK verification (and what it means for 3D)

**Date:** 2026-06-27. **Why this matters for 3D:** the 3D manufactured-solution case is P2 and stiff —
exactly the regime where the issue below first bit. Read this before debugging "3D OSGS loses optimal
convergence": the most likely cause is the *convergence gate*, not the discretization or the solver.

## TL;DR

1. **k ≥ 2 (high-order) MMS needs a TIGHTER convergence gate than k = 1.** The scale-free convergence
   criterion stops on `ε_M = ‖momentum residual‖/D_M ≤ tol_M` (`tol_M = eps_tol_momentum`). At the k=1
   default `eps_tol_momentum = 1e-6`, high-order / high-Re cells **stop early at a 5–10× worse solution**,
   collapsing the last-segment rate (e.g. k=2 L²u 160→320 rate 3.3 → 1.7). Setting
   **`eps_tol_momentum = 1e-9`** (sweep-wide for k=2) recovers optimal O(h³). Leave `eps_tol_mass = 0.8`;
   do **not** over-tighten (1e-12 → NaN). This is the `k2-needs-tighter-convergence-gate` lesson, now
   reproduced and pinned.
2. **JFNK is correct.** The opt-in `osgs_jfnk_enabled` matrix-free full-tangent OSGS solve reproduces the
   frozen-π root *exactly* — it is **not** the cause of any convergence-rate change (proven below). Use it.
3. **A reusable diagnostic recipe** (gate vs solver vs inner-tolerance) is at the end — apply it verbatim
   to 3D.

## What was verified about JFNK (it works)

JFNK recovers the dense `∂π/∂u` coupling the frozen-π OSGS tangent drops (see
[../solver/jfnk-phase0-preconditioner-gate.md](../solver/jfnk-phase0-preconditioner-gate.md) and
`theory/osgs_algorithm.tex` §sec:jfnk). Verified end-to-end:
- **Phase-0 gate:** the factored frozen-π Jacobian is a viable inner-GMRES preconditioner (1–21 Krylov
  vectors); `N_j`=2–3 quadratic Newton steps vs `N_c`=60 (non-converging) for the frozen-π inexact Newton.
- **Phase-1:** `JFNKLinearSolver` (matrix-free FD action + frozen-π preconditioner, C.1 honesty + frozen-π
  fallback) plugged into the existing `SafeNewtonSolver` via `_osgs_jfnk_solve!` — "change exactly one
  thing: the inner linear solve." Blitz 240/240, Quick 76/76, extended equivalence test pass.
- **Full 2D MMS sweep:** k=1 TRI/QUAD optimal everywhere; k=2 QUAD optimal once the gate is fixed (below).

### JFNK fallback behaviour (observed in the full sweep — benign, expected)

Across the full 2D OSGS sweep **~5% of cells (14/288) fell back** from JFNK to the frozen-π coupled solve.
These are **outer-Newton safeguards** firing (line-search depletion / residual-divergence guard) — **not**
inner-GMRES or preconditioner failures (Phase-0 gated the inner solve; it converges in 1–4 outer steps on
the other ~95%). Two regimes, both benign:
- **Convective corner** (Re=1e6 × coarse `h` = max cell-Péclet): the full-tangent step is aggressive and
  the re-projecting merit is too nonlinear, so Armijo depletes; frozen-π's gentler steps win. This is the
  "OSGS Newton-vs-Picard is a wash" lesson at the extreme — the exact tangent is *not* universally better.
- **Boot-already-converged cells** (low-Re): the ASGS boot already drove `‖R‖∞` to ~1e-9, so the matrix-free
  mat-vec finite-differences a residual at the noise floor and the divergence guard correctly bails.
Every fallback cell still produced the **validated optimal error** (the two N=320 fallbacks gave the same
5.45e-9 L²u as their non-fallback neighbours). **Zero correctness cost** — C.1 honesty + frozen-π fallback.
**For 3D:** expect the same on the convective corner; do not mistake it for a bug or chase 100% JFNK
coverage — falling back there is the *correct* behaviour.

## The k=2 high-order gate regression (discovered by the full sweep)

The 2D k=2 QUAD OSGS sweep showed the **high-Re (Re=1e6) cells** converging to a **5–10× worse** error at
the finest meshes than the prior official results, with the **last-segment L²u rate collapsing** (≈3.3 →
≈1.7) and only **~⅕ the iterations** (≈47 vs ≈309). Symptom signature, worth memorising:

> early stop (few iterations) + error several × above optimal at the *fine* meshes + last-interval rate
> well below O(h^{k+1}), while coarse meshes look fine.

### Root cause: the gate, not JFNK, not the inner tolerance

A clean isolation on the worst cell (Re=1e6, Da=1, α=1.0, k2 OSGS) — **all three current-code variants
produced byte-identical errors**:

| variant | e@N=80 | e@N=160 |
|---|---|---|
| OLD code, frozen-π (optimal reference) | 1.23e-6 | 1.38e-7 |
| NEW code, JFNK (η=1e-2) | 1.46e-6 | 2.34e-7 |
| NEW code, frozen-π (jfnk off) | **1.46e-6** | **2.34e-7** |
| NEW code, JFNK with tight inner tol (η=1e-6) | **1.46e-6** | **2.34e-7** |

So it is **not JFNK** (frozen-π gives the same), and **not the inner-GMRES forcing tolerance** (η=1e-6
identical). The config is identical old-vs-new (`eps_tol_momentum=1e-6`, `trace_convergence_norms=true`).
The difference is a **code change**: the recent "Fix A" made the scale-free convergence probe the
**authoritative** success gate (it was diagnostic-only before). With `tol_M = eps_tol_momentum = 1e-6`,
that gate is too loose for k=2 high-Re and stops the solve before it reaches the true root.

### The fix (verified)

`eps_tol_momentum = 1e-9` on the worst cell recovers the old-optimal error **exactly**:

| N | tight gate (1e-9) | old-optimal | loose gate (1e-6) |
|---|---|---|---|
| 80 | 1.23e-6 | 1.232e-6 ✓ | 1.46e-6 |
| 160 | 1.38e-7 | 1.377e-7 ✓ | 2.34e-7 |

Applied **sweep-wide to the k=2 config** (`phase1_quad_k2.json`), not per-corner — every k=2 cell wants
the deeper convergence; the high-Re corner just makes the under-convergence visible first. k=1 (TRI, QUAD)
is left at the default — it already reaches optimal at the looser gate.

## Implications for 3D MMS (act on these)

- **Expect to set `eps_tol_momentum = 1e-9` (or tighter) for the 3D P2 sweep.** 3D is P2 and stiffer than
  2D k=2; the gate that under-converges 2D k=2 high-Re will under-converge 3D P2 at least as much. This is
  the FIRST knob to check if "3D P2 loses optimal convergence" — before blaming stabilization (c₁), mesh
  quality, or the solver. (Cross-ref the `numerical-epsilon` / `3d-p1-eps` / `k2-gate` lessons.)
- **Watch for the signature:** few iterations + fine-mesh error several × above optimal + last-segment
  rate collapse. If you see it, suspect the gate first.
- **Keep `eps_tol_mass = 0.8`** (the mass gate floors at O(h^{kv}); tightening it risks the
  finest-segment loss / NaN documented in the k2-gate lesson). The momentum gate is the operative lever.
- **JFNK is safe to use in 3D** for OSGS — it reproduces the frozen-π root and cuts Newton steps where the
  dropped ∂π/∂u sets the rate. The C.1 fallback + frozen-π fallback mean it never does worse. (Watch the
  3D inner-GMRES count: the frozen-π preconditioner may need a real saddle-point/MG preconditioner if `G`
  blows up — see the phase-0 doc's 3D watch-item.)

## Reusable diagnostic recipe (gate vs solver vs inner tolerance)

When a sweep shows degraded convergence after a code change:

1. **Compare by RATE and per-segment, old vs new** — a full-range fit hides where it diverges; the
   per-interval rate shows the exact mesh where it breaks.
2. **Isolate the solver:** re-run the degraded cell with the *old* solver path under the *new* code (here:
   `osgs_jfnk_enabled=false`). If it reproduces the degradation, the solver is exonerated → it's the gate
   or the code.
3. **Isolate the inner tolerance** (for JFNK): tighten the inner forcing tol. No change ⇒ not the inner
   solve.
4. **Test the outer gate:** tighten `eps_tol_momentum` and re-run. Recovery ⇒ the gate was the cause.
5. **Confirm by exact-error match** to the known-good reference at the coarse meshes (they should already
   match) and at the fine meshes after the fix.

This is how the above was pinned in four targeted single-cell runs; reuse it for 3D.

## Pointers
- Gate config: `eps_tol_momentum` / `eps_tol_mass` in `SolverConfig` (src/config.jl); the MMS harness
  emits them as the trace `tol_M`/`tol_C` and the scale-free probe consults them (convergence_criterion.jl,
  `_safe_solve_inner!` "scale_free" path in src/solvers/nonlinear.jl).
- The k=2 official config now carries `eps_tol_momentum=1e-9`: `test/extended/ManufacturedSolutions/data/phase1_quad_k2.json`.
- JFNK: `osgs_jfnk_enabled` (+ `osgs_jfnk_gmres_*`, `osgs_jfnk_fd_epsilon`) in the schema/config;
  `src/solvers/linear_solvers.jl` (`JFNKLinearSolver`), `src/solvers/osgs_solver.jl` (`_osgs_jfnk_solve!`).
