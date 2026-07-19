# 2D MMS convergence — settled state, the convergence gate, and the stiff-corner fold

> The 2D manufactured-solution case is **SETTLED and optimal**; this doc holds the load-bearing learnings, the
> gate spec, the fold mechanism + production method, and the reusable diagnostic recipes. Headline numbers →
> [findings.md](../findings.md); per-cell provenance → [convergence-baseline.md](convergence-baseline.md).

## Status (one line)

The 2D **k=1 and k=2 QUAD** sweeps are complete (N=10→640) and **fully optimal** — velocity L² is `O(h^{k+1})`,
H¹ is `O(h^k)`; pressure meets or beats its nominal equal-order order (often super-optimal, 1.5–2.4×). **OSGS is
~2× more accurate than ASGS** at the same rate. (k=1 TRI has 2 pre-existing NaN cells at the α=0.05 corner — a
curved-interface-on-structured-mesh difficulty, not a regression.)

## 1. The convergence gate — how the solver decides "converged"

- **Scale-free / dimensionless.** Converged ⇔ `ε_M = ‖r_M‖/D_M ≤ tol_M` **and** `ε_C = ‖r_C‖/D_C ≤ tol_C`. No
  a-priori `U/L/P/Re/Da` scales. Spec: [theory-code-map.md](../theory-code-map.md) §3.
- **Route B (the mass gate).** `ε_C` is the *algebraic* Philosophy-A `‖r_C‖/D_C` (pressure block of the assembled
  residual, symmetric with `ε_M`), **replacing** the old strong-form measure that floored at `O(h^{kv})` and
  forced the loose `eps_tol_mass = 0.8` rubber-stamp. The strong-form measure survives only as the diagnostic
  `eps_C_strong`. Route B is behavior-preserving (k2 QUAD byte-identical to the pre-Route-B baseline).
- **k≥2 needs a TIGHTER gate: `eps_tol_momentum = 1e-9`** (not the k=1 default `1e-6`). At `1e-6`,
  high-order / high-Re cells **stop early at a 5–10× worse solution** and the last-segment rate collapses (k=2
  L²u 160→320: 3.3 → 1.7). **If a high-order run loses its optimal rate, the gate is the FIRST suspect** —
  signature: *few iterations + fine-mesh error a multiple above optimal + last-segment rate ≪ `O(h^{k+1})` while
  coarse meshes look fine*. This is orthogonal to `c₁`: the gate controls *when the solver stops*, `c₁` controls
  *which fixed point it reaches*.
- **`residual_floor_reached` valve (`3b76864`).** For nearly divergence-free flows the mass denominator `D_C`
  collapses (~1e-8), so `ε_C` floors a decade above tol and *falsely rejects a converged solution*. The valve
  accepts machine-floor convergence when (1) not degenerate, (2) `ε_M ≤ tol_M`, (3) residual at the noise floor
  (`‖R‖∞ ≤ k_nf·effective_ftol`, `k_nf = noise_floor_success_max_ftol_multiple`). It fired **263×** in the clean
  k2 sweep — the intended high-Re exit, not a band-aid.
- Config: `eps_tol_momentum` / `eps_tol_mass` in `SolverConfig` (`src/config.jl`); the k=2 official config uses
  `1e-9` (`test/extended/ManufacturedSolutions/data/phase1_quad_k2.json`).

## 2. The stiff-corner fold (high-Re / low-porosity)

- The **Re=1e6, α₀=0.05** corner "fails" on coarse meshes because the **discrete VMS solution branch has a
  turning point (saddle-node)** before the corner — there is *no discrete root at coarse N*. It is **not** a
  solver/Jacobian bug: Exact-Newton `J·v` vs FD of the residual ≈ **4.8e-12**.
- **It recedes with refinement.** α-fold ≈ 0.24 (N=10) → 0.16 (N=40) → 0.106 (N=80). Once the target α is above
  the fold, the solve converges directly — clears at **N≥512 for P1/TRI, N≥160 for Q2/QUAD** (Q2 resolves the
  porosity layer ~2× better per direction). Past the fold: optimal **H¹ ≈ 1.0–1.04**, super-convergent **L² ≈
  2.99–3.03**.
- **Trouble axis is `Re×α₀`, not `Re×Da`** — constant `σ` is a benign, coercive knob (bit-identical roots at
  Da=1e6 vs 1e-6 for the same `(Re, α₀)`).
- **α-continuation is the only viable continuation axis** (Da-/Re-continuation stall — they hold α=0.05 fixed, so
  the layer persists all the way). α-continuation from α=1 relieves the layer and tracks the fold as it recedes.
- **Production method (since 2026-06-17): direct exact-guess Newton** at N≥512 (P1) / N≥160 (Q2) — ~3 iters, ~6
  LU solves (~25 min/cell) vs ~70 solves / ~4 h for α-continuation. Drivers: `run_corner_article.jl` /
  `run_corner_osgs.jl` (direct); `run_continuation.jl` (α-ramp, legacy).
- **OSGS warm-start.** The OSGS root is `O(h²)` from the ASGS root, so error settles in a few iters; use a
  production residual stop (`ftol=1e-6`), not a tight true-root (OSGS is frozen-π linear-rate).

## 3. Key learnings / gotchas (2D)

- **Pre-asymptotic ≠ order reduction.** Layer-dominated error depresses the slope until the layer is resolved,
  then it recovers. The high-Da OSGS H¹ rate climbs `0.57→0.54→0.58→0.73→1.11→1.85` (N=10→640) — recovers to
  ≥1.0 (the "high-Da coercivity gap" is pre-asymptotic; gap closes like `Da_h ∝ 1/N²`). Mechanism:
  [findings.md](../findings.md) §4 (full dossier archived under `archive/`).
- **Aubin–Nitsche L² extra order** (`O(h^{k+1})`) needs elliptic-duality regularity — holds with Dirichlet BCs
  (the manufactured case), **fails with Neumann** (the Cocquet benchmark's traction-free outlet).
- **k2 P2 memory guardrails on 32 GB.** P2 N=320 LU is multi-GB — use **≤2 concurrent shards** (4+ → OOM); kill
  orphaned processes before launch; **suspect an execution/OOM failure before a numerical one** (a killed shard
  masquerades as a numerical defect).
- **Gridap-vs-Kratos magnitude offset — REMOVED 2026-07-19.** No longer tracked: Kratos is not part of the paper
  (all experiments are Gridap), so a Gridap-vs-Kratos magnitude comparison is moot. (`open-questions.md §2`.)
- **Diagnostic recipe — is it the gate, the solver, or the inner tolerance?** (reusable, incl. for 3D): (1)
  per-segment rate old-vs-new; (2) re-run the degraded cell with the *old* solver on new code; (3) tighten the
  inner tol → no change ⇒ not the inner solve; (4) tighten the *outer gate* → recovery ⇒ the gate is the cause;
  (5) confirm exact-error match to a reference at both coarse and fine meshes.

## JFNK / Anderson (OSGS iteration cost)

Both landed and verified — they cut OSGS iteration count **without changing the root** (`osgs_jfnk_enabled`,
`osgs_anderson_enabled`; JFNK ~16–100× fewer Newton steps, ~5% benign safeguard fallback). Canonical:
[findings.md](../findings.md) §5 and
[solver/jfnk-phase0-preconditioner-gate.md](../solver/jfnk-phase0-preconditioner-gate.md).
