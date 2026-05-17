# Algorithm-improvement Plan — Progress and Resumption Guide

**Companion to** [algorithm-improvement-plan.md](algorithm-improvement-plan.md).

This file tracks which plan items have landed, what verification was done,
what state is pending, and the recommended order for resuming work in a
future session. Update it after each phase commit.

Last updated: **2026-05-17**.

---

## What has landed (this session, 2026-05-17)

| Commit | Plan item | Files | Verification |
|---|---|---|---|
| `dc83e28` | (none — pure plumbing) | `src/solvers/porous_solver.jl` | Blitz 32/32, probe_k1 shows real residuals |
| `dfb8470` | Phase 1 — §1.2 + §1.5 | `src/solvers/porous_solver.jl`, `predictions_small_test.md` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `1caa633` | Phase 3 — §1.1 | `src/solvers/nonlinear.jl`, `src/run_simulation.jl`, 3 test files | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `9c4e1db` | Phase 4 part 1 — §1.4 | `src/config.jl`, `porous_ns.schema.json`, `base_config.json`, `src/run_simulation.jl` | Blitz 32/32, Quick 10/10 |
| `110e0d7` | Phase 4 part 2 — §1.3 | `src/solvers/porous_solver.jl` | Blitz 32/32, Quick 10/10 |
| `e702c16` | Phase 2 — §3.3 Anderson hardening | `src/solvers/accelerators.jl`, `src/solvers/porous_solver.jl`, `src/config.jl`, `porous_ns.schema.json`, `base_config.json` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `41c55ec` | Phase 6 part 1 — §2.4 `!isfinite` guards | `src/solvers/porous_solver.jl` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |

### Headline empirical result

Re-running [small_test_config.json](../test/extended/ManufacturedSolutions/data/small_test_config.json)
after P1 + P3 + §1.4 dropped total wall-clock from **~9.7 hours → ~18 minutes**
(~30× overall). Per-config highlights:

| Config | Pre-P1 OSGS time | Post-P1+P3+§1.4 OSGS time | Speedup |
|---|---|---|---|
| C3 k=3 QUAD | 5776 s | 119 s | **48×** |
| C4 k=1 TRI | 13126 s (3.6 hr) | 265 s | **49×** |
| C5 k=2 TRI | 3650 s (1.0 hr) | 53 s | **69×** |
| C3 k=3 QUAD ASGS | 9559 s (2.7 hr) | 118 s | **81×** |

Errors **bit-identical** across all 12 cases between pre-P1 and post-P4. This
empirically validates §1.5 (`π_h^0 = 0`) in the wall-clock dimension that
matters: the OSGS-on-TRI bootstrap pathology is eliminated.

---

## Verification baselines for regression detection

If a future change is supposed to be rate- and error-preserving, it should
not move these reference values. If it does, investigate before continuing.

### probe_k1.json (Re=1e-6, Da=1e6, k=1, QUAD, n=[10, 80], OSGS+ASGS)

```
config_1_ASGS / OSGS:
  iters     = [2, 1]
  residuals = [6.87e-5, 1.85e-4]
  u_L2      = [1.90179591, 1.25406420e-03]
  p_L2      = [2.62977746e-03, 4.06508762e-05]
```

ASGS and OSGS must produce identical numbers to ≥ 8 decimal places (constant-σ
trim makes the projection ≈ 0).

### small_test_config.json finest-mesh u_L2 (post-Phase-4)

| Config | k | Elem | u_L2 (n=160) |
|---|---|---|---|
| C1 | 1 | QUAD | 3.3040e-04 |
| C2 | 2 | QUAD | 5.9384e-06 |
| C3 | 3 | QUAD | 3.8680e-08 |
| C4 | 1 | TRI  | 3.3561e-04 |
| C5 | 2 | TRI  | 1.5443e-05 |
| C6 | 3 | TRI  | 6.6947e-08 |

### Quick tier orthogonality test

```
ASGS: L2 Velocity Error = 3.12525176e-02
OSGS: L2 Velocity Error = 3.39545818e-02 ; Proj(Subgrid_R) ≈ 1.5e-14
```

---

## Pending work — what's open, in recommended order

### Phase 2 — Numerical infrastructure (deferred earlier as higher-effort)

| Sub-item | Effort | Risk | Notes |
|---|---|---|---|
| ~~§3.3 Anderson hardening~~ | ~~Small~~ | ~~Low~~ | **Done** in the row above. Three fixes landed: configurable `safety_factor` (was hard-coded `10.0`), history clear on safety trigger, Tikhonov shift on the LS solve. probe_k1 bit-identical confirms Anderson code path (probe_k1 has `type: "Anderson"`, `m: 10`, `β: 0.8`) is unchanged in well-conditioned regime. |
| §3.2 Cholesky factorization | Medium (1 day) | Medium | Requires Gridap LinearSolver interface adaptation; may be a rabbit hole in Gridap 0.18.6. |

§3.3 landed: configurable `safety_factor` field added to `AcceleratorConfig`
following the §1.4 pattern (schema + base_config + validate! assertion
`safety_factor >= 1.0`), history-clearing safety trigger in
[src/solvers/accelerators.jl:88-93](../src/solvers/accelerators.jl#L88-L93),
and Tikhonov shift $\lambda = \varepsilon_{\mathrm{mach}}\,\mathrm{tr}(A)/n$
on both LS branches in
[src/solvers/accelerators.jl:65-77](../src/solvers/accelerators.jl#L65-L77).
Default `safety_factor = 10.0` preserves prior behaviour bit-identical.
Theory write-up updated in
[theory/osgs_algorithm.tex §"Block-wise Anderson acceleration"](osgs_algorithm.tex)
(Tikhonov paragraph + safety-fallback paragraph rewritten; parameter
table extended with $C_{\mathrm{sf}}^{\mathrm{And}}$).

Picard-side payoff is unverified in the current Re=1e-6 regime (Anderson
extrapolates cleanly without ever tripping the safety check). The hardened
paths will be exercised properly by Phase 6 continuation runs and the
stiff-regime probe config that the progress doc flags as needed.

### Phase 5 — Rate-affecting batch (bundle; re-baseline once)

These items each shift MMS error magnitudes slightly. The plan groups them
to amortize re-baselining:

- §3.1 block-equilibrated merit function
- §3.5 adaptive quadrature for Forchheimer (per-reaction-law trait dispatch)
- ~~§3.6 pressure mean removal in projection~~ — **deferred**, see
  [paper-code-divergences.md §6](paper-code-divergences.md) and
  [algorithm-improvement-plan.md §3.6](algorithm-improvement-plan.md) for
  the full rationale (regime-dependence on Dirichlet/mixed BCs, Option A
  vs Option B forms, and re-evaluation triggers).
- §5.1 h-scaling MMS floors
- §5.2 rate-aware plateau verification

**§5.1 + §5.2 are the items that will let us revert the bridge floors**
in small_test_config.json (see "Bridge state" section below). Until P5
lands, those floors stay as regime-specific overrides.

After P5: re-run small_test_config.json end-to-end and re-baseline the
finest-mesh u_L2 table above. Expect pressure rates to improve slightly
from §3.1 (block-equilibrated merit removing the velocity-bias in line
search); velocity rates should be at least as good. Any regression is a
bug, not a feature. The §3.6 improvement (constant-mode removal from
projection-drift metric) is **not** part of this batch, so the
post-Phase-5 baseline is the reference against which a future §3.6
commit will be judged.

### Phase 6 — Continuation driver (§2.1, §2.3, ~~§2.4~~)

Biggest remaining phase. Stage-I cascade currently bails on hard divergence;
Phase 6 §2.1 adds a `(Re, Da)` ramp continuation wrapper, §2.3 adds best-state
retention, and §2.4 (the `!isfinite` guards) has now landed — see the
"What has landed" table. §2.4 was promoted from Phase 6 because it's
independent of the continuation driver and "essentially free" per this doc's
"Strongly recommended" section; landing it now means high-Re/high-Da
continuation runs (when §2.1 lands) will fail-cleanly instead of NaN-leaking.

§2.4 covers six guard sites in [src/solvers/porous_solver.jl](../src/solvers/porous_solver.jl):
three in the Stage I Newton-Picard-Newton cascade (initial Newton, Picard
fallback, second Newton) and three in the OSGS inner cascade (added by §1.3
in `110e0d7`: inner Newton, Picard fallback, inner second Newton). Each
guard checks `any(!isfinite, get_free_dof_values(x))` immediately after
the corresponding `solve!`, restores from the appropriate backup
(`x0_backup` in Stage I, `x_prev` in OSGS inner), and short-circuits the
local control flow to the natural failure branch (fall-through to Picard
for the first Newton sites, `success = false` + `break` elsewhere). One
pre-existing Stage I Picard guard was strengthened to also restore from
`x0_backup`; the §1.3 OSGS Picard guard was already correct and is
unchanged.

Verification in current Re=1e-6 regime is "no regression": Anderson,
Newton, Picard, and OSGS all stay well-conditioned, so the guards never
trip. The defensive value is realised only in the stiff regimes Phase 6
§2.1 will unlock — at which point a deliberately divergent case should
exit with a clean abort and a "produced non-finite state" diagnostic
rather than propagating NaN through `diag_cache` and the MMS pipeline.

### Phase 7 — Quality-of-life

Low-risk, low-effort polish:

- §3.4 element-wise `freeze_jacobian_cusp`
- §4.2 cross-parameter validation (some already added in §1.4)
- §4.3 partial export on failure
- §4.4 configurable Picard variants (research feature; defer)

---

## Bridge state — things a future session must be aware of

### Floor overrides in `small_test_config.json` are regime-specific

The file currently has:
```
ftol = xtol = 1e-3
dynamic_ftol_ceiling = 1e-3
dynamic_ftol_spatial_safety_factor = 1e-1
stagnation_noise_floor = 1e-3
condition_noise_floor_absolute_min = 1e-4
newton_iterations = 10
```

These are sized for the `Re=1e-6, Da=1e6, P_c ≈ 1e12` regime, where the
machine-residual floor is `eps_mach × P_c ≈ 1e-4`. **Do not copy these
into other configs.** In a milder regime they would make Newton declare
success at the initial guess.

The floors are temporary bridge values pending Phase 5 (§5.1 h-scaling MMS
floors). After P5 lands, the floors should be reverted to base_config
defaults; rates and errors should then be at least as good across all
regimes without per-regime tuning.

A note documenting this lives at
[predictions_small_test.md §0](../test/extended/ManufacturedSolutions/predictions_small_test.md).

### Picard-side payoff of §1.1, §1.3, §1.4 is unverified

The current regime keeps Picard from firing (Newton always succeeds), so
the only check on these three items is "doesn't break Newton" (verified).
The actual Picard correctness gains require a stiff config exercising the
Picard fallback paths:

- §1.1: case where Stage-I previously hit `linesearch_failed` on Picard.
- §1.3: case where OSGS inner Newton previously aborted with
  `linesearch_failed` / `merit_divergence_escaped` / `linear_solve_nan`.
- §1.4: case where Picard previously hit `picard_iterations` cap.

The plan suggests "high Re *and* high Da" as the test point. A
`probe_stiff.json` would need regime-specific floors (~`P_c × eps_mach`
for that case's `P_c`). Designing this is a small task; it would slot
naturally into a Phase 6 (continuation) test bench since the continuation
driver targets exactly these cases.

### Stage I cascade was *not* refactored

§1.3 inlined the Picard fallback at the OSGS inner site rather than
extracting a reusable function. Stage I's existing cascade (Newton →
Picard → Newton at [porous_solver.jl:168-244](../src/solvers/porous_solver.jl#L168-L244))
is **left untouched** to bound regression risk. A future refactor can
factor both into a shared `picard_fallback_cascade!` helper. Until then,
the cascade logic is duplicated — keep both copies in sync if you change
the cascade semantics.

### Pre-existing tier warnings (not introduced by this session)

- `formulation_consistency_blitz_test.jl` runs in 5.0–5.3 s (Blitz limit
  is 5.0 s). Should be promoted to Quick eventually.
- `osgs_orthogonality_quick_test.jl` runs in 150–160 s (Quick limit is
  120 s). Should be promoted to Extended eventually.

Neither blocks correctness; both pre-date this session's work.

---

## How to resume

### Picking the next phase

The plan's own ordering says: P1 → P2 → P3 → P4 → P5 → P6 → P7. We did
P1, P3, P4 (skipping P2 because P3 was the single highest-impact item).
**P2 §3.3 (Anderson hardening) is the natural next move** — small, low-risk,
high-value, and independent of P5/P6.

If you want to push directly to P5 instead, see the "Phase 5" section
above and budget for at least one full MMS sweep re-baseline (~18 min
post-P4, but P5 may shift it).

### Verification cadence per phase

Per [CLAUDE.md](../CLAUDE.md):
1. Edits in `src/formulations/`, `src/stabilization/tau.jl`,
   `src/models/reaction.jl`, or `src/solvers/nonlinear.jl` → run **Blitz**.
2. Edits to assembly / residual / Jacobian / solver orchestration → run
   **Quick** after Blitz.
3. MMS-touching or rate-affecting changes → run **Extended** after Quick;
   for Phase 5 specifically, run small_test_config.json and compare
   against the baseline tables above.

### Useful invocations

```bash
# Blitz
julia -O0 -t 1 test/run_blitz_tests.jl

# Quick
julia --project=. test/run_quick_tests.jl

# probe_k1 (fast regression check, ~1-2 min, exercises the OSGS path)
cd test/extended/ManufacturedSolutions && julia --project=../../.. run_test.jl probe_k1.json

# small_test_config full sweep (~18 min post-P4; expect changes after P5)
cd test/extended/ManufacturedSolutions && julia --project=../../.. run_test.jl small_test_config.json

# Inspect an h5
python3 test/extended/ManufacturedSolutions/print_h5.py test/extended/ManufacturedSolutions/results/<name>.h5
```

The four probe configs (`probe_k1.json`, `probe_k2.json`, `probe_k3.json`,
`probe_k2_tri.json`) in `test/extended/ManufacturedSolutions/data/` are
scaffolding for per-phase regression tests. `probe_k1.json` is the
cheapest (~1-2 min) and exercises both ASGS and OSGS paths; it's the
default fast smoke test.

### Per-commit conventions used this session

- One commit per plan sub-item (or pair, when items are tightly coupled).
- Commit message includes: what changed, why (referencing the plan item),
  verification done, and any caveats (e.g., "Picard-side payoff
  unverified in this regime").
- All commits sign with `Co-Authored-By: Claude Opus 4.7 (1M context)`.

---

## Necessity for the full-range MMS sweep

The full-range MMS test
([test_config.json](../test/extended/ManufacturedSolutions/data/test_config.json))
sweeps `Re ∈ {1e-6, 1, 1e6}` × `Da ∈ {1e-6, 1, 1e6}` × `α₀ ∈ {1, 0.5, 0.05}`,
`k ∈ {1, 2}`, `n ∈ {10, 20, 40, 80, 160, 320}`, both element types and both
methods — roughly 1300 individual cases spanning nine `(Re, Da)` corners
with `P_c` varying ~12 orders of magnitude. Honest assessment of which
pending items are essential vs optional:

### Necessary

- **Phase 5 §5.1 + §5.2 (h-scaling MMS floors and rate-aware plateau).**
  The current bridge floors only work because `P_c ≈ 1e12` is fixed. The
  full sweep has `P_c` from ~3 to ~1e12; no single fixed `ftol` covers it.
  Without P5 you'd need 9 sets of regime-specific floors — impractical.
  **Without P5, the full sweep is fundamentally not portable.**
- **Phase 6 §2.1 (continuation driver).** The `(Re=1e6, Da=1e6, α₀=0.05)`
  corner is strong convection + strong reaction + narrow channel. Newton
  and Picard from a generic initial guess will not converge there.
  Continuation ramping `(Re, Da)` along log steps with warm starts is the
  only way to enter those basins. **Likely the highest-impact remaining
  item; without it, corner cells of the sweep fail outright.**

### Strongly recommended (needed for some cells, not all)

- **Phase 5 §3.6 (pressure mean removal).** Constant-mode pollution at
  `α₀ = 0.05` corrupts pressure rate measurements.
- ~~**Phase 6 §2.4 (!isfinite guards).**~~ **Done** (commit pending); six
  symmetric guards across Stage I + OSGS-inner cascades.
- ~~**Phase 2 §3.3 (Anderson hardening).**~~ **Done** in `e702c16`.

### Nice to have

- **Phase 2 §3.2 (Cholesky).** Pure speed.
- **Phase 5 §3.1, §3.5.** Refines constants, not rates.
- **Phase 7 §3.4, §4.3.** Debugging aids.

### Not necessary

- **Phase 7 §4.4 (Picard variants), §4.2 (extra cross-param validation).**

### Caveat — Picard payoff is untested

P3 (§1.1) + P4 (§1.3 + §1.4) all assume the Picard branch fires correctly
in stiff cases. The current Re=1e-6 regime keeps Picard idle, so these
paths are verified only as "doesn't break Newton". **The high-Re corners
of the full sweep will be the first time they actually fire.** Budget
time after P5 + P6 to inspect diagnostics there before trusting them.

### Minimum order to attempt the full sweep

**P5 → P6 §2.1 + §2.4 → P2 §3.3.** Roughly 8–12 days per the plan's own
estimates. Skipping P5 means manual per-regime tuning. Skipping P6's
continuation means corner-cell failures (acceptable if you accept ~80%
coverage). Skipping P2 §3.3 means occasional Anderson stalls.

If "most cells converge with reasonable rates" is the goal, P5 alone
covers ~80%. If "every cell converges within ±0.15 of theoretical rate"
is the goal, P5 + P6 are both essential.

---

## Open questions worth a future session's attention

1. **Should we extract the Picard cascade into a reusable helper?**
   See "Stage I cascade was not refactored" above. Pro: removes
   duplication. Con: touches working Stage I code, regression risk.
   Defer until there's a third call site that would benefit.

2. **Do the bridge floors need to be reverted before P5, or after?**
   They're declared as bridge in `predictions_small_test.md`. Reverting
   before P5 would make small_test_config.json fail (Newton can't reach
   the tight `ftol`). After P5's h-scaling, the base defaults should work.
   So: revert as part of P5.

3. **The k=2 sub-optimal velocity rate (~2.5 instead of 3) seen
   pre-P5.** §5.2 is designed to *flag* this case (`mms_plateau_at_suboptimal_rate`).
   §5.1's h-scaling floors should *fix* it by pushing Newton further
   only when the FE error budget warrants it. Verify both behaviours
   when P5 lands.

4. **Pressure rates are super-optimal** (h^{k+1} instead of h^k) in the
   current data. The plan predicts §3.1 + §3.6 will improve them
   "slightly". If they get worse, that's a regression bug.
