# Algorithm-improvement Plan — Progress and Resumption Guide

**Companion to** [algorithm-improvement-plan.md](algorithm-improvement-plan.md).

This file tracks which plan items have landed, what verification was done,
what state is pending, and the recommended order for resuming work in a
future session. Update it after each phase commit.

Last updated: **2026-05-19**.

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
| `dc8f302` | Phase 5 — §3.6 deferral (docs only) | `theory/paper-code-divergences.md`, plan, progress | — |
| `a307bcd` | Phase 5 part 1 — §3.5 reaction-law-aware quadrature | `src/models/reaction.jl`, `src/formulations/continuous_problem.jl`, `src/run_simulation.jl`, 2 test runners | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `f72c819` | Phase 5 part 2 — §3.1 block-equilibrated merit | `src/solvers/nonlinear.jl` | Blitz 32/32, Quick 10/10, probe_k1 state bit-identical (Φ values differ as expected, line search never fires here) |
| `af0bf07` | Phase 5 part 3 — §5.1 h-scaled MMS plateau floors | `src/solvers/porous_solver.jl`, `test/extended/ManufacturedSolutions/run_test.jl` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |
| `af05173` | Phase 5 part 4 — §5.2 rate-aware plateau verification | `src/solvers/porous_solver.jl`, `test/extended/ManufacturedSolutions/run_test.jl` | Blitz 32/32, Quick 10/10, probe_k1 bit-identical |

## What has landed (next session, 2026-05-19) — Phase 6 §2.1 *resolution*

| Commit (pending) | Plan item | Files | Verification |
|---|---|---|---|
| (uncommitted) | Phase 6 §2.1 — diagnostic harness + Fix 1 (Picard safeguard) + Fix 2 (dynamic Newton budget) | `src/solvers/nonlinear.jl` (Picard divergence safeguard tracks `‖R‖_∞`), `src/config.jl` + `base_config.json` (new `dynamic_newton_re_threshold`/`dynamic_newton_re_iterations` fields), `test/extended/ManufacturedSolutions/run_test.jl` (wire dynamic Newton budget), `test/extended/ManufacturedSolutions/probe_stiff_diagnose.jl` (new diagnostic script), `test/extended/ManufacturedSolutions/data/probe_stiff_failing.json` (new probe config), `test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md` + `probe_stiff_raw.md` (write-up) | Blitz 32/32, Quick 10/10 bit-identical (ASGS L2 3.12525176e-02, OSGS L2 3.39545818e-02, proj norm ~1.5e-14), probe_k1 bit-identical (iters=[2,1], residuals=[6.87e-5, 1.85e-4], u_L2/p_L2 match), probe_stiff_diag.json 2×2×2 grid now **8/8** (was 7/8 — the historically failing `(Re=1e6,Da=1,α=0.05)` cell converges, `‖R‖_∞ = 5.47e-4` at iter 91), probe_stiff_failing.json now ✅ converges (was ❌ all 7 attempts fail) |

### Headline result — Phase 6 §2.1 deferral overturned

The cell the previous session (2026-05-18) ruled "irrecoverable by continuation"
— `(Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS, eps_pert=0)` — now **converges
cleanly** under defaults. ‖R‖_∞ drops 0.215 → 5.47e-4, exits via
`stagnation_noise_floor_reached`, L2 u/p errors 0.127 / 0.025. The H2/H3/H4
diagnostic settled the question:

- **H2 (Jacobian ill-conditioning) ruled out.** Probe D: `cond(J) = 2.72e7` at
  u_ex — well-conditioned. The easy cell at the same α₀=0.05 actually has
  `cond = 2.25e18` (P_c scaling dominates), so cond is not a useful sanity
  criterion in this codebase.
- **H3 (no discrete root) ruled out.** Probe C: Picard makes monotone ‖R‖
  progress; full cascade reaches the root once safeguards stop misfiring.
- **H4 (τ-stabilisation breakdown) partial / secondary.** Probe E shows isolated
  residual-landscape spikes along the Newton line, compatible with
  `SmoothVelocityFloor` / τ-regularisation thresholds — but they don't block
  convergence, just slow it (more line-search backtracking → more Newton iters
  needed).

**Root cause: two compounding solver-safeguard bugs**:

1. **Picard mode's `divergence_merit_factor` safeguard tracks Φ growth.** Φ is
   a Newton-merit metric (`½‖b/w‖²` with `w = diag(J_Newton)`). Across Picard
   iterations the weights `w` shift because the Picard Jacobian's diagonal
   differs from Newton's, making Φ non-stationary even when `‖R‖_∞` shrinks
   monotonically. Fix: in `:picard` mode, compare `‖R‖_∞` growth instead.
   [src/solvers/nonlinear.jl:166-180](../src/solvers/nonlinear.jl#L166).
2. **`newton_iterations = 20` is too tight at high Re.** With Fix 1 in place,
   the second Newton pass (after Picard fallback) trends down to ‖R‖_∞ ≈ 0.014
   but hits the 20-iter cap. Adding 40+ iterations covers both the transient
   smoothing-out phase (where line-search backtracks through spikes) and the
   quadratic tail. Fix: new `dynamic_newton_re_*` config pair mirroring the
   existing `dynamic_picard_re_*` pattern, default `threshold = 1e4`,
   `iterations = 60`. Low-Re regimes unchanged.

Full write-up with probe data, line-probe landscape, and verdict in
[test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md](../test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md).
Raw probe output auto-dumped to `probe_stiff_raw.md` each script run.

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

### small_test_config.json finest-mesh u_L2 (post-Phase-5, `af05173`)

| Config | k | Elem | u_L2 (n=160) |
|---|---|---|---|
| C1 | 1 | QUAD | 3.304e-04 |
| C2 | 2 | QUAD | 5.938e-06 |
| C3 | 3 | QUAD | 3.868e-08 |
| C4 | 1 | TRI  | 3.356e-04 |
| C5 | 2 | TRI  | 1.544e-05 |
| C6 | 3 | TRI  | 6.695e-08 |

Bit-identical to the post-Phase-4 baseline across all six configs (only
display precision differs — the underlying HDF5 numbers match to ≥ 4
displayed digits in every case). All 60 (config × mesh × method) entries
in the sweep hit `base_convergence_only` → the Phase 5 plateau verifier
(§5.1, §5.2) was dead code in this Re=1e-6 regime; the §3.1 line-search
merit never fires here (Newton converges in 1–2 iters at α=1 throughout);
and §3.5's Forchheimer quadrature bump is a no-op because the MMS runner
uses `ConstantSigmaLaw`. The Phase 5 changes are therefore *latent* — they
hardened the harness for stiff-regime tests Phase 6 will unlock, without
shifting any visible number in the current regression suite. Wall-clock
~19 minutes (matching the post-P4 baseline within noise).

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

### Phase 5 — Rate-affecting batch — **LANDED** (`a307bcd`–`af05173`)

Four of the five planned items landed; §3.6 was explicitly deferred
(documented in [paper-code-divergences.md §6](paper-code-divergences.md)
and the plan's §3.6 entry).

- ~~§3.1 block-equilibrated merit function~~ — landed in `f72c819`.
- ~~§3.5 adaptive quadrature for Forchheimer~~ — landed in `a307bcd`
  via a `min_quadrature_degree(::AbstractReactionLaw, k_v)` trait that
  each reaction law overrides (default 0; Forchheimer adds ⌊k_v/2⌋).
- ~~§3.6 pressure mean removal in projection~~ — **deferred**, see
  [paper-code-divergences.md §6](paper-code-divergences.md).
- ~~§5.1 h-scaling MMS floors~~ — landed in `af0bf07`.
- ~~§5.2 rate-aware plateau verification~~ — landed in `af05173` with
  configurable `mms_rate_check_factor` (default 100×).

Re-baseline observation. small_test_config.json re-ran in ~19 min
(matching pre-P5 wall-clock within noise). All six configs at n=160
produced **bit-identical** u_L2 values vs the post-P4 reference
(see the table above). All 60 (config × mesh × method) cases exited
via `base_convergence_only` — the plateau verifier was never engaged
in this regime, so §5.1 and §5.2 were dead code throughout. §3.1's
line-search merit never fires here (Newton always succeeds in 1–2
iters at α=1), and §3.5's Forchheimer bump is a no-op because the MMS
runner uses `ConstantSigmaLaw` exclusively. The Phase 5 changes are
therefore **latent harness improvements** — they preserve current
behaviour bit-identical, and their effect surfaces only when:

1. Newton-OSGS has to iterate enough times for the merit's velocity-vs-
   pressure bias to matter (stiff regimes Phase 6 unlocks).
2. The plateau verifier is engaged (also stiff-regime / corner cells).
3. A Forchheimer MMS variant is added (currently every MMS config
   builds `ConstantSigmaLaw` via `build_mms_formulation`).

The post-P5 baseline above is the reference against which a future §3.6
commit will be judged for actual numerical improvement. The bridge
floors in small_test_config.json are still in place — they're solver
tolerances, not MMS plateau floors; reverting them is a separate cleanup
that can happen once Phase 6 has built enough confidence to do so safely.

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

### Phase 6 §2.1 — Continuation driver — **DEFERRAL CONFIRMED; cell unblocked via solver-safeguard fixes (2026-05-19)**

**Status as of 2026-05-19:** §2.1 (Re/Da ramp continuation) remains not landed,
**and the cell that motivated it now converges without it.** Diagnostic probes
C/D/E (deliberately not run on 2026-05-18) settled the live H2/H3/H4 hypotheses
and pointed at two solver-safeguard bugs, not the basin-of-attraction issue
continuation would have addressed.

**Fix 1 — Picard divergence safeguard tracks `‖R‖_∞`** (was Φ growth, a
Newton-merit metric that fluctuates spuriously across Picard iterations as the
Jacobian-diagonal weights shift).
[src/solvers/nonlinear.jl:166-180](../src/solvers/nonlinear.jl#L166).
Newton-mode bit-identical.

**Fix 2 — Dynamic Newton iteration budget at high Re.** Added
`dynamic_newton_re_threshold` / `dynamic_newton_re_iterations` config pair
mirroring the existing `dynamic_picard_re_*` pattern. Default
`threshold = 10000`, `iterations = 60`. Low-Re regimes unchanged. Wired into
[run_test.jl:485-498](../test/extended/ManufacturedSolutions/run_test.jl#L485);
non-MMS `run_simulation.jl` not yet plumbed (its `cfg.phys.nu` doesn't expose
characteristic Re).

After both fixes, [probe_stiff_failing.json](../test/extended/ManufacturedSolutions/data/probe_stiff_failing.json)
(the exact failing cell, `Re=1e6, Da=1, α₀=0.05, k=1, n=10, QUAD, ASGS`)
converges on the first homotopy attempt with `‖R‖_∞ → 5.47e-4`, stop reason
`stagnation_noise_floor_reached`, L2 u/p errors `0.127 / 0.025`. Full
investigation lives in
[test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md](../test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md).

§2.1 remains a valid future item *if* a case appears where the **discrete root
exists** AND **no warm-start within Newton's basin can be constructed from the
existing initial-guess machinery** AND **the cell can't be unlocked by larger
solver budgets**. None of those conditions are currently demonstrated in this
codebase.

---

### Phase 6 §2.1 — DEFERRED (pre-2026-05-19 framing, preserved for context)

The remainder of this section preserves the 2026-05-18 investigation framing for historical reference. The verdict above supersedes it.

Status as of 2026-05-18: not landed; **not justified by current empirical
evidence**.

#### Pre-investigation framing

The plan and earlier progress-doc revisions described §2.1 as "the
highest-impact remaining item" on the grounds that the high-Re/high-Da
corner cells of [test_config.json](../test/extended/ManufacturedSolutions/data/test_config.json)
would *fail outright* without continuation. That framing was inherited
from the original critical-analysis document; **it was never
empirically tested**.

#### What was actually tested

A focused diagnostic sweep at the asymptotic corner of the
parameter space, executed via four scaffolding configs in
[test/extended/ManufacturedSolutions/data/](../test/extended/ManufacturedSolutions/data/):

- [probe_stiff.json](../test/extended/ManufacturedSolutions/data/probe_stiff.json)
  — single case `(Re=1e6, Da=1e6, α₀=0.05, k=1, n=10, ASGS)`.
- [probe_stiff_diag.json](../test/extended/ManufacturedSolutions/data/probe_stiff_diag.json)
  — 2×2×2 isolation grid `(Re, Da, α₀) ∈ {tame, stiff}³` at `k=1, n=10`.
- [probe_stiff_n80.json](../test/extended/ManufacturedSolutions/data/probe_stiff_n80.json)
  — h-refinement sweep `n ∈ {40, 80, 160}` on the failing case.
- [probe_stiff_psweep.json](../test/extended/ManufacturedSolutions/data/probe_stiff_psweep.json)
  — p-refinement sweep `k ∈ {1, 2, 3}` at `n=10` on the failing case.

#### Findings

**The 2×2×2 isolation revealed only ONE genuinely difficult cell:**

| α | Re | Da | Outcome | Required `eps_pert` |
|---|---|---|---|---|
| 0.5 | × | × | × | All 4 cells: ✅ clean, `eps_pert = 0.1` (first attempt) |
| 0.05 | 1 | 1 | ✅ clean | `eps_pert = 0.1` |
| 0.05 | 1e6 | **1** | ❌ ALL 7 attempts fail | nothing works |
| 0.05 | 1 | 1e6 | ✅ clean | `eps_pert = 0.1` |
| 0.05 | 1e6 | 1e6 | ⚠️ noise-floor saturation | `eps_pert = 1e-5` |

The "expected corner cell" `(α=0.05, Re=1e6, Da=1e6)` is **not** the hard
one — it converges, just slowly. The genuinely hard cell is the
*opposite of intuition*: **high Re + narrow channel + LOW Da**. With
`Da = 1` the reaction term `σ ≈ 1e-6` is effectively switched off,
removing the Darcy damping that otherwise stabilises high-Re flow in
the confined geometry.

**Neither h- nor p-refinement rescues this cell:**

| Refinement | k | n | $\|R(u_{\mathrm{ex}})\|_\infty$ at iter 0 | Outcome |
|---|---|---|---|---|
| (baseline) | 1 | 10  | 2.15e-1 | ❌ |
| h-refine | 1 | 40  | 1.13e-2 | ❌ |
| h-refine | 1 | 80  | 1.88e-3 | ❌ |
| h-refine | 1 | 160 | 1.95e-4 | ❌ |
| p-refine | 2 | 10  | 2.18e-3 | ❌ |
| p-refine | 3 | 10  | 1.27e-3 | ❌ |

The FE consistency residual at $u_{\mathrm{ex}}$ shrinks at the expected
$\mathcal{O}(h^{k+1})$ rate, but Newton/Picard cannot find any nearby
discrete root at any combination tried — including `eps_pert = 0`
(exact MMS solution as initial guess) at n=160 where the consistency
residual is only ~$10^{-4}$.

#### Implications for §2.1

The corner cell that motivated §2.1 turns out **not** to be a
basin-of-attraction failure. It's not rescued by:
- Better initial guess (`eps_pert = 0` at any tested resolution fails);
- More mesh resolution (h-sweep up to n=160 fails);
- Higher polynomial order (p-sweep up to k=3 fails).

§2.1 (Re/Da ramp continuation) produces *warm-starts*; the worst-case
warm-start it could possibly deliver is no better than $u_{\mathrm{ex}}$
itself, and $u_{\mathrm{ex}}$ as a warm-start has already been shown to
fail. So §2.1 cannot rescue this case. The general value proposition
of §2.1 (rescuing cases where the *discrete root exists and Newton
basin is just narrow*) is not falsified — but **no such case has been
identified in this codebase**. Every case in `small_test_config.json`
converges trivially; every case in the 2×2×2 isolation grid either
converges trivially or fails for the structural reason below.

§2.1 is therefore deferred until either (a) a case appears that
demonstrably has the basin-failure pathology continuation can fix, or
(b) the codebase is extended beyond MMS validation toward general
engineering use (where exact-solution warm-starts are unavailable and
continuation becomes the only realistic strategy).

#### What this case probably IS — open question

The failure pattern matches one of three live hypotheses, undiscriminated
by the experiments so far:

- **H2 — Jacobian ill-conditioning at the asymptotic corner.** With
  `σ ≈ 1e-6` and `ν ≈ 1e-6`, the discrete momentum block loses its
  dominant diagonal and the per-element Reynolds number is ~$10^5$.
- **H3 — discrete VMS has no fixed point with residual ≤ ftol nearby.**
  The stabilised system's coercivity proof in
  [article.tex](article.tex) degrades as `ν, σ → 0`; the discrete
  solution may not exist in the strong sense the test demands.
- **H4 — τ stabilisation breaks down asymptotically.** At
  `Da=1 + Re=1e6` both viscous and reactive contributions to `τ_1`
  vanish, leaving `τ_1 ≈ h/(c_2·|a|)` — pure convective stabilisation.
  Whether this provides sufficient crosswind dissipation as `Re → ∞`
  is exactly the regime the paper's analysis pushes against.

#### Where to look next (probes deliberately NOT run in this session)

Each probe targets one of H2/H3/H4 and would cost ~30 min of scripting
plus < 1 min of compute. A future session that wants to settle the
question should write a standalone Julia script in
[test/extended/ManufacturedSolutions/](../test/extended/ManufacturedSolutions/)
that mirrors the failing case's FE setup (mesh, spaces, formulation,
$u_{\mathrm{ex}}$ interpolation) and performs:

- **Probe D — Jacobian condition number.** Assemble the exact-Newton
  Jacobian at $u_{\mathrm{ex}}$ via `jacobian!(A, op, x0)` where
  `op = FEOperator(res_fn, jac_newton, X, Y)` and `x0` is the
  interpolated exact solution. Compute `cond(Matrix(A))` or
  `svdvals(Matrix(A))[end] / svdvals(Matrix(A))[1]`. A value $> 10^{14}$
  is a smoking gun for H2.
- **Probe C — Picard-only iteration from $u_{\mathrm{ex}}$.** Construct
  the cascade's Picard `FESolver` (`SafeNewtonSolver(...; mode=:picard)`)
  and run `solve!(x0, fe_solver_picard, op_picard)` from
  $u_{\mathrm{ex}}$. If Picard converges where Newton fails, the
  iteration map has a fixed point and Newton's Jacobian is the obstacle
  (H2 confirmed). If Picard also wanders, the discrete fixed point
  doesn't exist in the contraction-mapping sense (H3 / H4).
- **Probe E — residual landscape line probe.** Compute the Newton
  step `δ = -J\b` at $u_{\mathrm{ex}}$, then evaluate
  `‖R(u_ex + s·δ)‖_∞` for `s ∈ [-1, 1.5]` at ~40 sample points. Plot.
  A clear minimum near `s = 1` indicates a usable Newton direction
  Newton's step-size policy isn't taking; a featureless / oscillatory
  landscape supports H3 (no nearby root) or H4 (stabilisation pathology).

The four probe configs listed above stay committed as scaffolding so
the next session can reuse the exact same `(Re, Da, α₀, n, k)`
combinations.

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

Four additional stiffness-frontier probes were added in the 2026-05-18
investigation that deferred Phase 6 §2.1:

- `probe_stiff.json` — single stiff cell `(Re=1e6, Da=1e6, α=0.05)`.
- `probe_stiff_diag.json` — 2×2×2 isolation grid over `(Re, Da, α)`.
- `probe_stiff_n80.json` — h-refinement sweep `n ∈ {40, 80, 160}` on the
  failing cell.
- `probe_stiff_psweep.json` — p-refinement sweep `k ∈ {1, 2, 3}` on the
  failing cell.

These are not regression tests — they're scaffolding for future
diagnostic work. See the "Phase 6 §2.1 — DEFERRED" section above for
how to use them and what probes (C/D/E) remain unanswered.

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

- ~~**Phase 5 §5.1 + §5.2.**~~ **Done** in `af0bf07` (h-scaling) and
  `af05173` (rate-aware plateau). The plateau verifier is now
  discretization-budget-scaled; bridge floors in `small_test_config.json`
  can be revisited once Phase 6 has been validated.
- ~~**Phase 6 §2.1 (continuation driver).**~~ **Deferred** as of 2026-05-18.
  The "highest-impact remaining item" framing was overturned by direct
  empirical investigation: the corner cell that motivated §2.1
  (`Re=1e6, Da=1, α₀=0.05` — note: low Da, not high Da as originally
  framed) fails for a reason continuation cannot fix — it fails even
  with the exact MMS solution as initial guess, at every tested mesh up
  to n=160, and at every polynomial order up to k=3. See the "Phase 6
  §2.1 — DEFERRED" section above for the full investigation,
  alternative hypotheses, and the C/D/E probes that would settle which
  hypothesis is correct.

### Strongly recommended (needed for some cells, not all)

- **Phase 5 §3.6 (pressure mean removal).** Constant-mode pollution at
  `α₀ = 0.05` corrupts pressure rate measurements. **Deferred** with full
  rationale in [paper-code-divergences.md §6](paper-code-divergences.md);
  re-evaluation triggers documented there.
- ~~**Phase 6 §2.4 (!isfinite guards).**~~ **Done** in `41c55ec`.
- ~~**Phase 2 §3.3 (Anderson hardening).**~~ **Done** in `e702c16`.

### Nice to have

- **Phase 2 §3.2 (Cholesky).** Pure speed.
- ~~**Phase 5 §3.1, §3.5.**~~ **Done** in `f72c819` and `a307bcd`.
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
