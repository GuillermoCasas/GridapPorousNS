# Route B — 2D MMS official sweep: verification record

**Completed:** 2026-07-03. **Verdict: Route B — with the `residual_floor_reached` fix (`3b76864`) — is
behavior-preserving across the full 2D sweep.** Branch: `fix/scale-free-residual-floor-accept` (built on
`feat/route-b-algebraic-mass-gate`). (This file began as a pause/resume note on 2026-07-02; it is now the
completion record. An earlier version of it mis-attributed the k2 failures to an OOM alone — corrected below.)

## What Route B is (one line)

The mass convergence gate is the Philosophy-A **algebraic** measure `ε_C = ‖r_C‖/D_C` (pressure block of the
assembled residual, symmetric with the momentum gate), replacing the strong-form measure that floored at
O(h^{kv}) and forced the loose `eps_tol_mass = 0.8` rubber-stamp. Strong-form kept as the diagnostic
`eps_C_strong`. Core: `e455f36`; the essential companion fix: `3b76864` (below). Write-up:
[docs/formulation-audit-2026-06-24.md](../formulation-audit-2026-06-24.md) §C.3 / F4.

## The k2 root cause (a GENUINE Route B gate limitation) and its fix — `3b76864`

Route B as first committed (`e455f36`) had a real flaw the sweep exposed at k2 fine meshes:

- For a **near-divergence-free** converged flow, the mass envelope `D_C = ‖∫q ∇·(αu)‖ + ‖∫q εp‖ + ‖∫q g‖`
  **collapses** (its dominant `∇·(αu)` term → 0). So `ε_C = ‖r_C‖/D_C` **floors ~1e-8**, a decade above
  `tol_C = 1e-9`, *even though the solution is fully converged* (`ε_M ~ 1e-14`, residual at the ~3e-12
  machine floor). The pure `ε_C` gate then **rejects a converged solution** and burns the entire
  homotopy-perturbation fallback (`eps_pert 1.0 → 0.1 → 0.0`, each a full ~1.2M-DOF solve) on the largest,
  most expensive cells — slow, and prone to fold-stalls/divergence at the extremes.
- **Fix `3b76864` (`residual_floor_reached`):** honor the existing honest-exit valve
  (`noise_floor_success_max_ftol_multiple`) under the scale-free branch — accept iff (1) not degenerate,
  (2) `ε_M ≤ tol_M` (momentum genuinely converged — rejects high-Re fold stalls), and (3) the per-field
  residual is at the machine/noise floor. No new magic number, no gate loosening; it fires only when
  momentum has converged and the residual cannot be reduced further. In the clean k2 sweep this valve
  **fired 263×** (the most common stop reason; every high-Re k2 cell ends on it) — it is what makes the
  symmetric tight `eps_tol_mass = 1e-9` viable at k2.

**Correction to the earlier note:** the k2 NaN was **not** "just a zombie-shard OOM." The primary cause was
this `ε_C` envelope collapse (a genuine Route B design issue), which caused the homotopy-burning; a
**secondary** aggravator was that leftover **zombie shards** from earlier pause/resume cycles (~13GB) piled
onto the already-expensive fine-N P2 solves and exhausted 32GB → OOM (a shard was `Terminated: 15`). The
clean re-run fixed *both*: it ran on the branch that includes `3b76864`, and with 2 shards / no zombies.

## Final verdict — all three families (canonical `results/k<kv>/<etype>/results.h5`)

| family | cells | NaN | median L2u rate | behavior-preservation |
|---|---|---|---|---|
| k1 QUAD | 48 | 0 | 2.00 (optimal) | median rel Δe_u **6.5e-7** vs gold `validated_k1_quad_N640` @N=320; 40/48 within 1% |
| k1 TRI  | 48 | **2** (pre-existing) | 2.00 (optimal) | the 2 NaN (Re=1,Da=1,α=0.05) were **also NaN in the baseline**; Route B **fixed 2** other α=0.05 cells the baseline had NaN (4 → 2) |
| k2 QUAD | 48 | 0 | 3.00 (optimal), min 2.79 | vs pre-Route-B baseline @N=160: median rel Δe_u **1.97e-11** (byte-identical), 48/48 within 1% |

The α=0.05 NaN cells are the known curved-interface-on-structured-mesh difficulty
([convergence-status.md](convergence-status.md)), not a Route B effect.

## Guardrails for future k2 (P2) sweeps on this 32GB machine

- **k2 QUAD: use ≤ 2 shards** (P2 N=320 LU is multi-GB; 4+ concurrent OOMs). k1 (P1) tolerates 6.
- **Before launching, `pgrep -f run_test.jl` and kill stragglers** — pause/resume can orphan shards.
- **Reproduce a suspected regression single-process before alarming** — a sweep-execution failure (OOM,
  orphaned process) can masquerade as, or pile onto, a numerical one. (I raised a false "tight-gate
  regression" alarm here by reading the corrupted sweep at face value; the isolation run + `3b76864`'s
  own commit message were the corrections.)

## Backup, and regenerating plots

- Backup: `previous_results/pre_route_b_2026-07-01/` (committed; see its README for the phase1/preJFNK merge
  caveat — compare same-config, or use the gold `validated_k1_quad_N640` for k1).
- `python analyze_results.py` → convergence PNGs + reports (auto-discovers `results/k*/*/results.h5`).
- `python plot_trajectory.py --sweep k2/QUAD` (also `k1/QUAD`, `k1/TRI`) → per-cell trajectory plots.

All three families' PNGs + reports + trajectory plots are generated; results live in the canonical
(gitignored, on-disk) `results/k*/*/`, so the standard pythons show the latest by default.
