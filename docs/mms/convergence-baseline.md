# MMS convergence baseline — frozen N≤320 provenance table (k=1 QUAD)

> **FROZEN PROVENANCE SNAPSHOT (N≤320, k=1 QUAD, 2026-06-08).** Kept as the reference the completed Route-B
> k1/k2 sweeps are measured against; the per-cell numbers are provenance and are intentionally left unrewritten.
> Current headline numbers + the settled recovery/gate story: [`convergence-2d.md`](convergence-2d.md) and
> [`findings.md`](../findings.md) §1/§4. The two "targets" this table was tracking are both **met**: the high-Da
> OSGS H¹ rate **recovers to ≥1.0 at N=640** (pre-asymptotic, `Da_h ∝ 1/N²`, not an order ceiling), and the OSGS
> iteration cost (30–104 inner Newton steps, the dropped `∂π/∂U`) is cut by the landed **JFNK**.

**Snapshot:** 2026-06-08 (N≤320 table).
**Solver:** coupled-only OSGS (a single coupled Newton, `π = Π(R(u))` re-projected every evaluation) + Stage-I
Newton↔Picard sensors + coupled stall-exemption; the authoritative gate is the scale-free ε_M/ε_C criterion.
**Source (provenance):** `data/phase1_quad_k1.json` (k=1, P1/P1, QUAD), `results/phase1_quad_k1.h5`.

## Per-cell baseline (k=1, QUAD, N = 10→640)

Rate = H¹(u) slope; OSGS rate is the **final-segment** value (the pre-asymptotic creep). Iters = inner Newton
steps per mesh, range across N.

| Re | Da | α₀ | ASGS H¹-rate | ASGS iters | OSGS H¹-rate (final) | OSGS iters | note |
|------|------|------|:----:|:----:|:----:|:----:|------|
| 1e-6 | 1e-6 | 0.05 | 1.04 | 2 | 1.00 | 45–58 | OSGS fails N≤20 (layer) |
| 1e-6 | 1e-6 | 0.5  | 1.03 | 2 | 1.00 | 43–80 | |
| 1e-6 | 1e-6 | 1    | 1.00 | 2 | 1.00 | 61–91 | |
| 1e-6 | 1    | 0.05 | 1.02 | 2 | 1.00 | 41–62 | OSGS fails N=10 (layer) |
| 1e-6 | 1    | 0.5  | 1.02 | 2 | 1.00 | 43–81 | |
| 1e-6 | 1    | 1    | 1.00 | 2 | 1.00 | 61–91 | |
| 1e-6 | 1e6  | 0.05 | 1.08 | 3 | 0.95 | 9–52  | layer-resolution recovery |
| 1e-6 | 1e6  | 0.5  | 1.05 | 3 | **0.74** | 10–50 | reaction-dominated (pre-asymptotic) |
| 1e-6 | 1e6  | 1    | 1.04 | 3 | **0.71** | 30–53 | reaction-dominated (pre-asymptotic) |
| 1    | 1e-6 | 0.05 | 1.02 | 3–4 | 1.00 | 42–72 | OSGS fails N=10 (layer) |
| 1    | 1e-6 | 0.5  | 1.02 | 4 | 1.00 | 43–93 | |
| 1    | 1e-6 | 1    | 1.00 | 4 | 1.00 | 61–104 | |
| 1    | 1    | 0.05 | 1.02 | 3–4 | 1.00 | 42–72 | OSGS fails N=10 (layer) |
| 1    | 1    | 0.5  | 1.02 | 4 | 1.00 | 43–93 | |
| 1    | 1    | 1    | 1.01 | 4 | 1.00 | 61–104 | |
| 1    | 1e6  | 0.05 | 1.07 | 4–6 | 0.95 | 14–53 | layer-resolution recovery |
| 1    | 1e6  | 0.5  | 1.05 | 4–5 | **0.74** | 13–51 | reaction-dominated (pre-asymptotic) |
| 1    | 1e6  | 1    | 1.03 | 4–5 | **0.71** | 31–54 | reaction-dominated (pre-asymptotic) |
| 1e6  | 1e-6 | 0.5  | 1.01 | 6–17 | 1.01 | 83–93 | high-Re: many OSGS iters |
| 1e6  | 1e-6 | 1    | 1.01 | 6–24 | 1.02 | 65–98 | high-Re |
| 1e6  | 1    | 0.5  | 1.01 | 6–18 | 1.01 | 78–93 | high-Re |
| 1e6  | 1    | 1    | 1.00 | 6–21 | 1.01 | 64–95 | high-Re |
| 1e6  | 1e6  | 0.5  | 1.01 | 8–15 | 1.01 | 40–77 | |
| 1e6  | 1e6  | 1    | 1.00 | 10–20 | 1.00 | 44–86 | |

*(The three `Re=1e6, α₀=0.05` cells are deliberately skipped — the high-Re/low-porosity fold has no coarse-mesh
root; it clears by ≈N=512, after which a direct exact-guess Newton solve converges in ~3 iters. See
[convergence-2d.md](convergence-2d.md).)*

## Reading it — the two durable findings

- **OSGS is pre-asymptotically suboptimal in the reaction corner (Da=10⁶), then recovers.** The H¹ rate creeps
  upward with refinement — α₀=1 goes `0.57 → 0.54 → 0.58 → 0.73 → 1.11 → 1.85` (N=10→640), reading ≈0.71 through
  N≤320 (the value tabulated above) but climbing to ≥1.0 once the N=320→640 pair is in (1.60 at α₀=0.5). A
  **formulation property** (the coercivity gap closing like `Da_h ∝ 1/N²`), not a solver bug — mechanism in
  [`findings.md`](../findings.md) §4 (full dossier archived under `archive/`).
- **α₀=0.05 at N=10 fails** — the steep porosity layer is below a 10×10 mesh's resolution floor (the old code
  "converged" these to garbage, H¹≈24); converges at N≥20. Mesh/α-continuation is the lever, not the inner solver.
- ASGS is optimal (H¹ ≈ 1.0) and cheap (2–6 inner Newton steps, up to ~24 at Re=10⁶) everywhere — the reference
  for a correct, well-conditioned solve.

## Other configs in `data/`

- **`phase1_quad_k2.json`** — the k=2 (P2/P2) sweep, **run and complete** (`results/k2/QUAD/results.h5`); its
  optimal O(h³) result is in [`findings.md`](../findings.md) §1.
- **`continuation_c24.json` / `continuation_c24_rate.json`** — α+mesh continuation for the high-Re/low-α fold
  (the three skipped `Re=10⁶, α₀=0.05` cells). Superseded 2026-06-17 by the direct exact-guess solve
  (`run_corner_article.jl` / `run_corner_osgs.jl`, N≥512) — see [convergence-2d.md](convergence-2d.md).
- **`test_config.json`** — the minimal MMS-*sweep* example config (CLAUDE.md's sweep section runs it via
  `run_test.jl test_config.json`); the true "single simulation" example is `config/base_config.json`.
