# Theoretical predictions — `small_test_config.json` MMS sweep

This file records, **before looking at results**, what I expect from the
manufactured-solutions sweep specified by
[small_test_config.json](data/small_test_config.json). The goal is to compare
predictions against the actual run, flag surprises, and learn from
discrepancies.

## 0. Bridge parameter overrides — regime-specific, temporary

The numerical floors currently set in [small_test_config.json](data/small_test_config.json)
(`ftol = xtol = 1e-3`, `dynamic_ftol_ceiling = 1e-3`,
`dynamic_ftol_spatial_safety_factor = 1e-1`, `stagnation_noise_floor = 1e-3`,
`condition_noise_floor_absolute_min = 1e-4`, `newton_iterations = 10`) are
**sized for this specific regime only** (`Re = 1e-6`, `Da = 1e6`, constant
`σ ≈ 1e12`, `P_c ≈ 1e12`). They match the machine-residual floor
`eps_mach × P_c ≈ 1e-4` so Newton can declare convergence at saturation
rather than grind against an unreachable target.

Do **not** copy these values into other configs. In a milder regime
(say `Re = 1, Da = 1` ⇒ `P_c ≈ 3`), an `ftol` of `1e-3` would let
Newton declare success at essentially the initial guess.

These overrides are a bridge until Phase 5 of
[theory/algorithm-improvement-plan.md](../../../theory/algorithm-improvement-plan.md)
lands (items §5.1 *h-scaling MMS floors* and §5.2 *rate-aware plateau
verification*), which will make per-regime floor tuning unnecessary.
After Phase 5, these overrides should be reverted and the config
should sit on the base defaults.

## 1. The physical regime in this sweep

The config fixes a single physical point and sweeps the discretization:

| symbol | value | meaning |
|--------|-------|---------|
| `Re`   | `1e-6` | input Reynolds number |
| `Da`   | `1e6`  | input Darcy number (this code's convention — see below) |
| `α_0`  | `0.5`  | porosity at the bump core; `α_∞ = 1` outside |
| `f_x`,`f_y` | `0` | no extra forcing (MMS source is computed analytically) |
| `epsilon_pert` | `0.1` | initial homotopy perturbation around the exact root |

The code maps these inputs in [run_test.jl:73-76](run_test.jl#L73-L76) as

```
ν       = U_amp · L / Re                          = 1·1/1e-6 = 1e6
σ_const = Da · α_∞ · ν / L²                       = 1e6·1·1e6/1 = 1e12
```

and pressure is rescaled by the MMS module
([mms_paper_2d.jl:108-126](../../../src/problems/mms_paper_2d.jl#L108-L126)):

```
P_c = (1 + Re_internal + Da_internal) · U · ν / L = (1+1e-6+1e6)·1·1e6/1 ≈ 1e12
```

**Important:** Despite the name, this code's input `Da` measures the
**dimensionless reaction strength** (`σ L²/(α_∞ ν)`), i.e. the *inverse* of the
classical permeability-based Darcy number. So `Da = 1e6` is **strong porous
resistance**, not "free fluid". The current sweep therefore sits in a regime
that is simultaneously:

* **Stokes-like in inertia** (`Re = 1e-6` → convection negligible relative to
  viscous diffusion);
* **Strongly Darcy/reaction dominated** (`σ/(ν/L²) = 1e6` → momentum is
  effectively `σu + ∇p ≈ f`, with viscous diffusion six orders of magnitude
  smaller than reaction);
* **Constant σ** (uses `ConstantSigmaLaw`, so
  `ProjectResidualWithoutReactionWhenConstantSigma` trims the σu term out of
  the orthogonal projection — paper §4.4).

The MMS velocity is
`u_ex = U α_0 (1/α(x)) [sin(πx)sin(πy), cos(πx)cos(πy)]` of order 1, but
pressure is `p_ex = P_c cos(πx)sin(πy)` of magnitude `~1e12`. All errors in
the report are normalized by `U_c=1` and `P_c≈1e12`, so the pressure column
will *look* tiny in normalized units even when its absolute discrete error is
large.

## 2. Expected asymptotic convergence rates

The paper proves (and the MMS plateau test enforces) optimal rates for the
stabilized formulation with equal-order Lagrange `Pₖ–Pₖ` / `Qₖ–Qₖ`:

| norm | rate (theory) | comment |
|---|---|---|
| ‖u − uₕ‖_{L²} | `h^{k+1}` | Aubin–Nitsche-style velocity L² rate |
| ‖∇(u − uₕ)‖_{L²} | `h^{k}` | velocity H¹ semi-norm |
| ‖p − pₕ‖_{L²} | `h^{k}` | pressure (equal-order, stabilized) |
| ‖∇(p − pₕ)‖_{L²} | `h^{k-1}` | pressure H¹ semi-norm |

For this sweep `kv = kp ∈ {1, 2, 3}` with `equal_order_only = true`, so we
expect:

* `k=1`: u_L2 ~ h², u_H1 ~ h, p_L2 ~ h.
* `k=2`: u_L2 ~ h³, u_H1 ~ h², p_L2 ~ h².
* `k=3`: u_L2 ~ h⁴, u_H1 ~ h³, p_L2 ~ h³.

The smallest mesh is `h = 1/160 ≈ 6.25e-3`, the largest `h = 1/10 = 0.1`. The
absolute error budget at the finest mesh and `k=3` is roughly
`(6.25e-3)^4 ≈ 1.5e-9` — relevant for what follows.

## 3. Where I expect the *spatial* convergence rates to deviate

**A. `k=3`, fine meshes (`n=80, 160`) — possible plateau against the algebraic
floor.**
The dynamic algebraic tolerance is
[run_test.jl:441-446](run_test.jl#L441-L446):
`dyn_ftol = max(ftol, min(c_ceil, c_sf · h^{k+1}))` with `c_ceil=1e-4`,
`c_sf=1e-2`, `ftol=1e-11`. For `k=3, n=160` this gives
`min(1e-4, 1e-2 · (1/160)⁴) = 1.5e-11`, which is *below* `ftol=1e-11`'s floor,
so the effective target is `1.5e-11`. Because the residual norm includes the
huge pressure scale (`~1e12` magnitudes), achieving `ftol=1.5e-11` on the
*absolute* residual is well past anything physically meaningful; the
non-linear loop will declare convergence at the *noise floor* it actually
reaches (currently it's hitting the early-N output around `≈ 5e-6` inf-norm,
which is `Φ_merit ≈ 1e-11` — see §6 below). So I expect:
* `u_L2` and `u_H1` rates at `k=3` to look clean up to `n=80`, possibly a
  *flattening* at `n=160` because the iterate sits at the noise floor before
  the discrete root is fully resolved.
* If we see the slope drop at `n=160`, the symptom is not a stabilization bug
  but the algebraic floor catching up with the discretization error.

**B. `k=1` with such large σ — pressure rate possibly slightly sub-optimal.**
`τ₁ → 1/σ ≈ 1e-12` and `τ₂ = h²/(c₁ α τ_{1,NS} + ε)`. The pressure
stabilization weight in equal-order Galerkin is `τ₁ |∇q|²`, which becomes
*tiny* in this regime. Theoretically OSGS/ASGS still give optimal `h^k` for
pressure because the reaction term itself supplies coercivity via
`σ ∫|u_h|² ≥ ...`, but the absolute pressure error can carry a large
prefactor (`P_c ~ 1e12`), so noise / round-off contaminates the lowest-order
mesh more visibly. Prediction: `k=1` pressure rate is the most likely place to
see pre-asymptotic noise on the coarse meshes (`n=10, 20`).

**C. TRI vs QUAD.**
The simplexify path uses `h_T = √(2A)` and Lagrangian `Pₖ` triangles; QUAD
uses `h_K = √A` and tensor-product `Qₖ`. Rates should be the same, but the
*constants* differ. Two specific things I am watching:
* For `k=2,3` on TRI, the number of edge/face DOFs grows differently — I
  expect TRI to require slightly more outer OSGS state-drift iterations than
  QUAD at the same `n,k`, because the projection space is richer.
* I do **not** expect rate degradation on TRI per se; if I see the slope
  drop on TRI but not on QUAD at the same `(k,n)`, the suspect is the
  `h_T = √(2A)` calibration vs the `c₁ = 4k⁴, c₂ = 2k²` constants, not the
  formulation itself.

**D. OSGS vs ASGS.**
Because σ is constant, the residual projection trims the σu term by
`ProjectResidualWithoutReactionWhenConstantSigma`. The remaining projected
residual is essentially the viscous + pressure-gradient part, which is *small*
in this regime (viscous is `1e6/L² · u ≈ 1e6`, vs σu ≈ `1e12`). So OSGS
should look almost identical to ASGS here in terms of the final discrete
solution — the orthogonal projection is operating on a small piece of the
residual. Concretely I expect:
* OSGS and ASGS error tables to agree to 2–3 significant digits at every
  `(k, n)`.
* OSGS outer iterations to converge in *few* state-drift steps (≤ 3) at every
  `n` because the projection is already nearly zero.

## 4. Where I expect the *non-linear* solver to struggle

The non-linear solver in [porous_solver.jl] chains Exact Newton → Picard
homotopy → ε-perturbation re-tries. For *this* regime:

**E. Newton should solve the discrete system in 1–2 iterations.**
`Re = 1e-6` ⇒ convective Jacobian terms are negligible. σ is constant, so the
reaction Jacobian is just `σI`, contributing a well-conditioned mass-like
block. The only "non-linear" entries are τ derivatives and the OSGS state
update. From the exact root as initial guess (with perturbation `eps_pert =
0.1`), I expect Newton to drop from `‖r‖ ~ 1e5` (because P_c · π ≈ 3e12
appears in the source!) to `‖r‖ ~ 1e-5` in a single step, then stall at the
noise floor.

**F. Picard fallback should rarely fire.**
With `picard_iterations = 2` it would barely do anything if it did. The
config also leaves `dynamic_picard_re_threshold` / `dynamic_picard_da_threshold`
in the default state, which means at our extreme `Da=1e6` we *may* hit one of
them and inflate the Picard budget. Worth checking by grepping the run log
for "Picard Homotopy fallback".

**G. The merit-function noise floor.**
[run_test.jl:441-455](run_test.jl#L441-L455) builds `dynamic_noise_floor` from
`n² · max(1,Re)`. With `Re=1e-6` the `max(1,Re)=1`, so the noise floor scales
purely with `n²`. At `n=160` this gives `n_base · 25600`. The actual log
output already shows `‖f‖_∞ ≈ 5e-6` accepted as "noise floor saturated" at
`n=10, k=1`. **Prediction:** at `n=160, k=3`, the merit stagnation will trip
*before* Newton drives the residual to `ftol`, and the run will print
"saturated at numerical noise floor" rather than "converged to ftol". As long
as the discrete error is already below the spatial estimate, this is healthy;
but if the noise floor is too generous, we could mask a real stalling
problem.

**H. The "Exact Newton aborted → Picard homotopy fallback" pattern.**
Already visible in the log for `n=80, k=1, ASGS`. This is the safeguard
chain working as designed: Newton stagnates at the noise floor without
*formally* satisfying `ftol`, so the orchestrator falls back to Picard for
smoothing, which immediately re-stagnates at the same level, and the run
finishes at noise floor. Concern: this triggers extra solves we don't really
need. Tradeoff is between robustness and time.

**I. Possible non-convergence cases.**
Given the linearity of this regime I do **not** expect any case in this
sweep to terminate with `NaN` errors. If any does, the most likely places
are:
* `OSGS, k=3, n=160, TRI`: combination of tight algebraic tolerance, fine
  mesh, OSGS state-drift mode and the `√(2A)` h-calibration. If OSGS state
  drift fails to settle below `osgs_tolerance = 1e-10` within
  `osgs_iterations = 5`, the case will return NaN.
* `OSGS, k=1, n=10`: pre-asymptotic regime where the projection is the most
  *non-trivial* part of the residual; 5 outer iterations might be borderline.

## 5. Parameter sensitivities — what would change the picture

| parameter | current | effect of increasing | effect of decreasing |
|---|---|---|---|
| `osgs_iterations` | 5 | tighter state-drift convergence; helps at fine `n` and high `k`; longer runtime | risk of OSGS giving up before state drift settles → NaN |
| `osgs_inner_newton_iters` | 20 | rarely binding here (1–2 iters suffice) | could bind if σ is changed to non-constant |
| `osgs_tolerance` | 1e-10 | rarely binding (overridden by `dynamic_ftol`) | — |
| `picard_iterations` | 2 | smoothing if Newton stalls in convection-dominated; here unused | here would not matter |
| `xtol`, `ftol` | 1e-11 | too tight → noise floor trips first | looser → false positives in noise floor before discrete error is resolved |
| `dynamic_ftol_ceiling` | 1e-4 | looser algebraic target on coarse meshes; faster runs | tighter target; risk of plateau on `n=10–20` |
| `dynamic_ftol_spatial_safety_factor` | 1e-2 | residual scaling pinned more tightly to `h^{k+1}`; rate-honest but slow | residual allowed to be a bigger fraction of `h^{k+1}` — could pollute rate |
| `stagnation_noise_floor` | 1e-5 | accept more stagnation; faster termination; risk of accepting un-converged states | tighter; risk of declaring stagnation as divergence |
| `max_linesearch_iterations` | 15 | rarely binding here | — |
| `linesearch_contraction_factor` | 0.5 | standard; smaller (e.g. 0.3) gives finer α-grid | — |
| `max_increases` | 3 | tolerates more merit increase; risky for divergence | strict — Newton kicks back to Picard sooner |
| `Anderson m` | 10 | longer memory; helps OSGS state drift converge faster | shorter memory; OSGS may not accelerate |
| `Anderson relaxation_factor` | 0.8 | mixing; 1.0 = pure Picard, 0.5 = heavy mixing | — |
| `epsilon_pert` | 0.1 | larger perturbation; tests basin width but slower | smaller; closer to exact root, fewer retries |
| `max_n_pert` | 5 | more chances to fall back to smaller perturbation; resilience at extreme regimes | fewer retries; faster failure |

**Single most impactful knob in this regime:** `dynamic_ftol_ceiling`. The
current `1e-4` is the *effective* target at coarse meshes (`n=10`,
`k=1,2`). If pressure errors look "too clean" because they ride on
`P_c=1e12`, drop the ceiling to `1e-6` to expose real discrete error;
conversely, raising it to `1e-3` will mask small inaccuracies in the
projection.

## 6. Concrete check-list for the run

I will compare the actual `convergence_data.h5` against the predictions above
and flag:

1. **Rate sanity** per `(method, etype, k)`: slopes within ±0.15 of theory?
2. **OSGS vs ASGS parity**: errors agree to ≥ 2 digits?
3. **`n=160, k=3`**: does the slope flatten (algebraic floor) or hold
   (computer is patient enough)?
4. **`k=1` pressure on coarse meshes**: pre-asymptotic noise visible?
5. **Any NaN** in the output: which case, and why?
6. **Picard fallbacks**: were any triggered? If so, in what cases?
7. **TRI vs QUAD**: same rates, different constants, as expected?
8. **`successful_eps`**: did the homotopy perturbation ever need to back off
   from `0.1` to a smaller scale?

## 7. References I might want

I have everything I need from the in-repo material — specifically:

* [theory/article.tex](../../../theory/article.tex) — the paper this is a
  transcription of.
* [theory/paper-code-divergences.md](../../../theory/paper-code-divergences.md)
  — known places where code deviates from the paper, for failure-mode
  triage.
* [docs/lessons_learned.md](../../../docs/lessons_learned.md) — past
  regressions.

If a result is surprising in a way none of those explain, the additional
reference that would help is **Codina's original 2002 OSGS paper**
(*"Stabilized finite element approximation of transient incompressible
flows using orthogonal subscales"*, CMAME 191, 4295–4321) and the related
**Codina–Badia 2008** work on equal-order stabilization at extreme Re/Da.
Please upload those PDFs if you have them; otherwise I can work from
`article.tex` and the bibliography I can already see.
