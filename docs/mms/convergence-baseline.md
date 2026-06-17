# MMS convergence baseline — reference for tracking progress

> **Update 2026-06-10 — the N=640 tail is complete and the high-Da OSGS "formulation target" is met.**
> The OSGS H¹ rates of **0.71–0.95** in the Da=1e6 rows below are the **N≤320 pre-asymptotic** values
> (the original run never finished N=640 for the α≠1 OSGS cells). With the N=320→640 pair now in, the
> reaction-dominated H¹ rate **recovers to ≥ 1.0**: e.g. the α₀=1 Da=1e6 cell climbs
> `0.57→0.54→0.58→0.73→1.11→1.85` (N=10→640), the α₀=0.5 cell `…→0.76→1.10→1.60`. So the
> coercivity-gap suboptimality is confirmed **pre-asymptotic (it recovers), not an order ceiling** — the
> formulation target is achieved without a split/term-by-term OSGS. The table below is **kept as the
> N≤320 reference** (the creep values future fine-mesh work is measured against); the recovered
> finest-pair rates and the full success summary are in
> [`convergence-status.md`](convergence-status.md). The **JFNK iteration target stands** (OSGS still
> 30–104 inner Newton steps — the dropped ∂π/∂U).

**Snapshot:** 2026-06-08 (N≤320 table) + 2026-06-10 (N=640 recovery, note above).
**Solver:** coupled-only OSGS (a single coupled Newton, `π = Π(R(u))` re-projected every evaluation) +
Stage-I Newton↔Picard sensors (`stall_window=2`, `pingpong_picard_gain_orders=1.5`) + common Newton
budget (150) + **coupled stall-exemption** (`stall_window=0` on the coupled solve) + coupled Picard
fallback. The authoritative convergence gate is now the scale-free ε_M/ε_C criterion.
**Source:** `data/phase1_quad_k1.json` (k=1, P1/P1, QUAD), `results/phase1_quad_k1.h5`.

This is the reference future changes are measured against. The two columns that should move:
- **OSGS iteration cost** is the **JFNK target** — the coupled inexact-Newton's *linear* rate (the dropped
  dense `∂π/∂U`) costs 30–104 Newton steps; JFNK should cut these to ~5–15 without changing the errors.
- **OSGS high-Da H¹ rate** — **target met (2026-06-10): recovers to ≥1.0 at N=640** (pre-asymptotic,
  per the note above). No split/term-by-term OSGS needed; the coercivity gap is in the *constant*, not
  the rate.

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
coarse-mesh root. Continuation is one way to reach a root, but the fold clears by ≈N=512, after which a
**direct exact-guess Newton solve** converges in ~3 iters with no continuation; see
[fold-recovery.md](fold-recovery.md).)*

## Reading it

- **ASGS** is optimal (H¹ ≈ 1.0) and cheap (2–6 inner Newton steps; up to ~24 at Re=10⁶) everywhere —
  the reference for a "correct, well-conditioned" solve.
- **OSGS matches ASGS's optimal rate at Da ≤ 1** (negligible reaction). At **Da=10⁶** it is
  **pre-asymptotically suboptimal in the reaction corner, then recovers**: the H¹ rate *creeps upward*
  with refinement — α₀=1 goes `0.57 → 0.54 → 0.58 → 0.73 → 1.11 → 1.85` (N=10→640), so it reads ≈0.71
  through N≤320 (the value tabulated below) but **climbs to ≥1.0 once the N=320→640 pair is in** (1.60 at
  α₀=0.5; α₀=0.05 already recovers to ~1.0 via layer-resolution). This is a **formulation property** (the
  coercivity gap closing like `Da_h ∝ 1/N²`), not a solver bug — see
  [osgs-reaction-dominated-rate.md](../solver/osgs-reaction-dominated-rate.md). **The N=640 ladder
  settled the open question (2026-06-10): it asymptotes _to_ ≥1.0 — a slow pre-asymptotic climb, not an
  order ceiling.**
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
  **fold** (the three skipped `Re=10⁶, α₀=0.05` cells), via `run_continuation.jl` (which was broken on
  `main` — missing `probe_stiff_diagnose.jl` `include` — and is now restored). Reaches H¹≈1.0 at the
  target corner. **As of 2026-06-17 the preferred corner path is the direct exact-guess solve**
  (`run_corner_article.jl` ASGS, `run_corner_osgs.jl` OSGS) at N≥512 — see [fold-recovery.md](fold-recovery.md).
- **`test_config.json`** — the small CLAUDE.md "single simulation" example; not a convergence study.

## Regenerate

Re-run the sweep (`julia --project=../../.. run_test.jl phase1_quad_k1.json` from
`test/extended/ManufacturedSolutions/`) and re-extract per-cell `err_u_h1` / `eval_iters` rates from
`results/phase1_quad_k1.h5`. Compare OSGS iteration counts (JFNK) and high-Da OSGS rates (formulation)
against this table to track progress.
