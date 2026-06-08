# MMS convergence baseline — reference for tracking progress

**Snapshot:** 2026-06-08.
**Solver:** corrected coupled-only OSGS (a single coupled Newton, `π = Π(R(u))` re-projected every
evaluation) + Stage-I Newton↔Picard sensors (`stall_window=2`, `pingpong_picard_gain_orders=1.5`) +
common Newton budget (150) + **coupled stall-exemption** (`stall_window=0` on the coupled solve) +
coupled Picard fallback.
**Source:** `data/phase1_quad_k1.json` (k=1, P1/P1, QUAD), `results/phase1_quad_k1.h5`.

This is the reference future changes are measured against. The two columns that should move:
- **OSGS iteration cost** is the **JFNK target** — the coupled inexact-Newton's *linear* rate (the dropped
  dense `∂π/∂U`) costs 30–104 Newton steps; JFNK should cut these to ~5–15 without changing the errors.
- **OSGS high-Da H¹ rate** is the **formulation target** — the reaction-dominated suboptimality (a
  pre-asymptotic coercivity gap) is what split / term-by-term OSGS would lift.

## Per-cell baseline (k=1, QUAD, N = 10→640)

Rate = H¹(u) slope; OSGS rate is the **final-segment** value (the pre-asymptotic creep — see below).
Iters = inner Newton steps per mesh, shown as range across N.

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

*(The three `Re=1e6, α₀=0.05` cells are deliberately skipped — the high-Re/low-porosity fold has no
coarse-mesh root; they are rescued by α/mesh continuation, see `continuation_c24.json` below.)*

## Reading it

- **ASGS** is optimal (H¹ ≈ 1.0) and cheap (2–6 inner Newton steps; up to ~24 at Re=10⁶) everywhere —
  the reference for a "correct, well-conditioned" solve.
- **OSGS matches ASGS's optimal rate at Da ≤ 1** (negligible reaction). At **Da=10⁶** it is
  **pre-asymptotically suboptimal in the reaction corner**: the H¹ rate *creeps upward* with refinement —
  e.g. α₀=1 goes `0.57 → 0.54 → 0.57 → 0.71` (coarse→fine), reported here as the final-segment value 0.71
  (0.74 at α₀=0.5; recovers to 0.95 at α₀=0.05 where layer-resolution dominates). This is a **formulation
  property** (the coercivity gap closing like `Da_h ∝ 1/N²`), not a solver bug — see
  [osgs-reaction-dominated-rate.md](../solver/osgs-reaction-dominated-rate.md). Whether it asymptotes *to*
  1.0 or just below is the open question the N=640/1280 ladder settles.
- **OSGS iteration cost is high (30–104)** — the coupled inexact-Newton converges *linearly* (the dropped
  `∂π/∂U`). This is the **JFNK target**: restoring the full tangent should drop these to ~5–15 with
  identical errors (the regression invariant for any JFNK implementation).
- **α₀=0.05 at N=10** (and N≤20 for the mildest cell) **fails** — the steep porosity layer is below the
  resolution floor of a 10×10 mesh (the old code "converged" these to garbage, H¹≈24); converges at N≥20.
  Not an inner-solver issue; mesh/α-continuation is the lever.

## Other configs in `data/`

- **`phase1_quad_k2.json`** — the k=2 (P2/P2) sweep; **not yet run** on the corrected solver. Run it to
  extend this baseline to higher order.
- **`continuation_c24.json` / `continuation_c24_rate.json`** — α + mesh continuation for the high-Re/low-α
  **fold** (the three skipped `Re=10⁶, α₀=0.05` cells), via `run_continuation.jl`; reaches H¹≈1.0 at the
  target corner (see [fold-recovery.md](fold-recovery.md)).
- **`test_config.json`** — the small CLAUDE.md "single simulation" example; not a convergence study.

## Regenerate

Re-run the sweep (`julia --project=../../.. run_test.jl phase1_quad_k1.json` from
`test/extended/ManufacturedSolutions/`) and re-extract per-cell `err_u_h1` / `eval_iters` rates from
`results/phase1_quad_k1.h5`. Compare OSGS iteration counts (JFNK) and high-Da OSGS rates (formulation)
against this table to track progress.
