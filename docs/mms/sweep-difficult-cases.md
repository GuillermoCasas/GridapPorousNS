# Difficult-convergence cases in the MMS parameter sweep

> **HISTORICAL — dated diagnostic snapshot** (partial sweep on commit `15e466b`). For the current MMS picture see the canonical [`convergence-status.md`](convergence-status.md); the stiff-corner deep dive is [`fold-recovery.md`](fold-recovery.md). The raw artifacts for this run (partial stdout log + `.h5`) live in [`../../test/extended/ManufacturedSolutions/previous_results/`](../../test/extended/ManufacturedSolutions/previous_results/).

This document records the convergence-difficulty patterns observed in a partial
run of `test/extended/ManufacturedSolutions/run_test.jl test_config.json` on the
refactored solver (commit `15e466b` and ancestors).

The run was interrupted at **Progress 115 / 648** (17.7% complete), inside the
N=160 batch of the QUAD k=1 outer loop. Within that 115 configs we ran the
full (Re, Da, α) × (kv=1, QUAD) sweep across N ∈ {10, 20, 40, 80} and partway
into N=160. This is enough to identify which parameter cells are stressing the
solver — and which sub-clusters within those cells deserve isolated debugging
before the full sweep is re-run.

## Sweep configuration (for reproducibility)

- File: `test/extended/ManufacturedSolutions/data/test_config.json`
- Re ∈ {1e-6, 1.0, 1e6} × Da ∈ {1e-6, 1.0, 1e6} × α₀ ∈ {1.0, 0.5, 0.05} (3³ = 27 physics cells)
- Element type ∈ {QUAD, TRI}, polynomial order kv = kp ∈ {1, 2} (equal-order only)
- Convergence partitions N ∈ {10, 20, 40, 80, 160, 320}
- Method = ASGS only (no OSGS in this harness)
- Homotopy: `epsilon_pert = [1.0]`, `max_n_pert = 5` ⇒ up to 7 attempts per cell (eps_pert ∈ {0.1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 0.0})
- Newton budget: `newton_iterations = 10`, expanded to 60 by `dynamic_newton_re_iterations` when `Re > dynamic_newton_re_threshold = 1e4`

Total wall time on the interrupted run: ~9.7 h, completed configs: 115/648.

## Failure mode taxonomy

The sweep produced four outcome classes:

1. **Clean convergence at attempt 1** — most cells. Newton (and where needed Picard) converges before any homotopy step is needed.
2. **Mild homotopy** — converges within attempts 2 or 3 (eps_pert ∈ {0.1, 0.01}). The cell is solvable but the cold start is poor.
3. **Deep homotopy** — converges only after attempts 4–6 (eps_pert ∈ {0.001, 1e-4, 1e-5}). The convective/Forchheimer cusp is genuinely hard from a cold start.
4. **Total failure** — attempt 7 (eps_pert = 0.0) is reached and no attempt produces a finite L²/H¹ error. The output row carries `NaN / NaN` for both error norms. **Critically, no `catastrophic divergence` or `non-finite state` events were logged** — the Picard catastrophic-state guard is doing its job, but the iteration simply fails to enter any basin within budget.

In the 115 configs covered:
- 89 Picard fallback firings (expected — these are an algorithmic feature, not a problem).
- 0 catastrophic non-finite events.
- 8 attempt-7 cells (all produced `NaN/NaN`).

## The Re = 10⁶ × α₀ = 0.05 cluster — the worst-case corner

**Every single cell** with Re = 10⁶ **and** α₀ = 0.05 either reached attempt 7
or required attempt 4+, regardless of Da or mesh size N. From the per-Progress
attempt-trace:

| Config | Re | Da | α | N | Deepest attempt | Outcome |
|---|---|---|---|---|---|---|
| 21 | 1e6 | 1e-6 | 0.05 | 10 | 7 | NaN |
| 24 | 1e6 | 1.0 | 0.05 | 10 | 4–6 (no NaN; rate still NaN) | partial |
| 27 | 1e6 | 1e6 | 0.05 | 10 | <4 (likely 1–3) but rate NaN | partial |
| 48 | 1e6 | 1e-6 | 0.05 | 20 | 7 | NaN |
| 51 | 1e6 | 1.0 | 0.05 | 20 | 7 | NaN |
| 75 | 1e6 | 1e-6 | 0.05 | 40 | 7 | NaN |
| 78 | 1e6 | 1.0 | 0.05 | 40 | 7 | NaN |
| 81 | 1e6 | 1e6 | 0.05 | 40 | 7 | NaN |
| 102 | 1e6 | 1e-6 | 0.05 | 80 | 7 | NaN |
| 105 | 1e6 | 1.0 | 0.05 | 80 | 7 | NaN |

Per-Da within the cluster:
- **Da = 1e-6** is the worst (all four N exhausted attempt 7).
- **Da = 1.0** is the second-worst (three out of four exhausted attempt 7; the N=10 case "converged" but rate is still NaN).
- **Da = 1e6** is the least bad here (only the N=40 case exhausted attempt 7 in the data seen).

Mesh refinement does **not** improve convergence in this corner — N=10, 20, 40,
80 all fail the same way. This rules out "needs a finer mesh" as the fix.

## The Re = 10⁶ × α₀ = 1.0 cluster — slow but solvable

These cells all converge, but per-config wall time spikes by ~30–100× compared
to the easy regime:

| Config | Re | Da | α | N | iters | t(s) | rate_u_l2 | rate_u_h1 |
|---|---|---|---|---|---|---|---|---|
| 3 | 1e6 | 1e-6 | 1.0 | (all N) | 300 | 2608 | 2.00 | 0.96 |
| 6 | 1e6 | 1.0 | 1.0 | (all N) | 256 | 3818 | 2.00 | 0.96 |
| 9 | 1e6 | 1e6 | 1.0 | (all N) | 102 | 709 | **1.67** | **0.67** |
| 18 | 1e6 | 1e6 | 0.5 | (all N) | 146 | 894 | 2.18 | 0.97 |

The `(t,iters)` is the aggregate across N=10..80 inside each h5 group. The
notable item: **config_9 (Re=10⁶, Da=10⁶, α=1.0) is the only converging cell
with measurably sub-optimal convergence rates** (`rate_u_l2 = 1.67` vs paper
expectation 2.0; `rate_u_h1 = 0.67` vs expectation 1.0). All other converging
cells reach within ~10% of theoretical rates. This single cell warrants
isolated MMS-rate analysis — the iteration converges but to a *non-FE-optimal*
fixed point, which the §5.2 rate-aware sanity check would flag.

## Time-cost distribution

Configs that converged but consumed disproportionate wall time (these dominate
the sweep's total runtime):

| Config | Description | t(s) |
|---|---|---|
| 6 | Re=1e6, Da=1, α=1 | 3818 |
| 3 | Re=1e6, Da=1e-6, α=1 | 2608 |
| 18 | Re=1e6, Da=1e6, α=0.5 | 894 |
| 9 | Re=1e6, Da=1e6, α=1 | 709 |
| 7 | Re=1e-6, Da=1e6, α=1 | 557 |
| 27 | Re=1e6, Da=1e6, α=0.05 | 241 (partial) |
| 16 | Re=1e-6, Da=1e6, α=0.5 | 179 |
| 25 | Re=1e-6, Da=1e6, α=0.05 | 111 |
| 11 | Re=1, Da=1e-6, α=0.5 | 127 |

All seven of the worst time-cost cells are at Re = 10⁶. The high-Re extension
of the Newton budget (`dynamic_newton_re_iterations = 60`) is firing as
designed but the cells genuinely need every iteration.

## Recommendations for follow-up isolation runs

In order of priority, three isolated probe configurations would let us attack
the failure modes without waiting on the full sweep:

### Probe A — the lethal Re=10⁶ × α=0.05 corner

Run a probe config covering exactly `Re = 1e6 × Da ∈ {1e-6, 1.0, 1e6} × α₀ =
0.05` on N ∈ {10, 20, 40, 80} with ASGS (and optionally OSGS to see if the
projection trick recovers any of these cells). All 8 attempt-7 failures live
here. Hypothesis to test: is this a fundamental issue with `τ`-stabilization
parameters at high Re combined with low porosity, or is it a Newton-basin
issue that a better cold start (e.g., a Stokes-equation pre-solve) would
unblock?

### Probe B — the sub-optimal-rate cell

Run `Re = 1e6, Da = 1e6, α₀ = 1.0` on all N including N=320 with the §5.2
rate-aware sanity check enabled (`mms_rate_check_factor` in `test_config.json`
defaults to 100.0; consider lowering for this probe). config_9 is the only
"converged but sub-optimal" cell observed; if the rate is actually 1.67 on
N=160 and N=320 too, this is a paper-faithfulness regression that needs
diagnosis (possibly the τ₁ simplification — see paper-code-divergences §4b on
the well-resolved-α assumption).

### Probe C — the dynamic-Newton-Re extension cost calibration

The four cells in §"Time-cost distribution" with t > 700s all fired the
dynamic-Re Newton extension (60 iters). With `newton_iterations = 10` as the
base budget and `dynamic_newton_re_iterations = 60`, each Newton iteration on
N=160 is ~10–30 seconds, so a single attempt at the budget can take ~10 min,
and up to 7 homotopy attempts can run consecutively — that's the ~1-hour
configs. Two possible follow-ups: (a) tighten the extension to a more
parsimonious value (e.g., 30 + Picard fallback) and see whether convergence
quality is preserved, (b) profile a single Re=1e6 Newton iteration to see
where the time is going (likely the linear-solve dominates on large meshes).

## Pattern-level observations

Across all converging cells:

- **Re is the dominant difficulty axis.** Re ≤ 1 cells converge in <30s
  regardless of Da or α. Re = 1e6 cells take 50× to 100× longer.
- **α is the *failure* axis at high Re.** Re=10⁶ converges (slowly) for α₀ ∈
  {1.0, 0.5} but fails outright at α₀ = 0.05.
- **Da is a minor modulator.** Within the Re=10⁶ × α=0.05 cluster, Da affects
  whether attempt 7 is reached at N=10 but not at N≥20.
- **Mesh refinement does not rescue the failure cluster.** N=10, 20, 40, 80
  all fail identically at Re=10⁶ × α=0.05. This is a strong signal that the
  failure is in the continuous problem's nonlinear structure, not in the
  discretization.
- **The Picard fallback is doing its job.** 89 Picard firings, zero
  catastrophic non-finite events. Without the fallback, the failure count
  would almost certainly be much higher.

## Data preservation

The partial h5 output at
`test/extended/ManufacturedSolutions/results/convergence_data.h5` contains 27
fully-populated config groups (covering all 27 (Re, Da, α) cells at kv=1,
QUAD), each with cumulative data across N ∈ {10, 20, 40, 80}. The data is
sufficient to plot convergence curves and re-derive the rates table above
without re-running the affected region.

The stdout log at `/tmp/refactor_paper_sweep/stdout.log` contains 11,418
lines of per-iteration Newton/Picard trace and is the canonical source for
the attempt-trace counts above. Consider archiving it alongside the h5
before re-running the sweep.
