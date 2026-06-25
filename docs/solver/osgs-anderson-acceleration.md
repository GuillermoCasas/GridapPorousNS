# OSGS Anderson acceleration — wiring + does it actually accelerate?

**Status:** landed 2026-06-25 (NONL-03). Opt-in, **OFF by default** (behavior-preserving). Verified to
accelerate.

## What it is

`src/solvers/accelerators.jl` (`AndersonAccelerator`) was fully implemented but **dead** (no consumer, no
config, no test — audit §D / NONL-03). It is now wired into the OSGS stage behind a config flag.

The default OSGS coupled solve (`solve_osgs_stage!`) is a *single* inexact-Newton solve: the residual
recomputes `π = Π(R(u))` from the current iterate every evaluation while the Jacobian holds `π` frozen
(the dense `∂π/∂u` is dropped). That mismatch makes it only **linearly** convergent.

When `osgs_anderson_enabled = true`, the OSGS stage instead runs the **staggered fixed point** of the OSGS
problem (`_osgs_anderson_outer!`):

```
for k = 1,2,…
    π_k  = Π(R(x_k))                       # freeze the projection at the current iterate
    g_k  = solve frozen-π nonlinear system  # G(x_k): exact-tangent Newton (π constant ⇒ no ∂π/∂u to drop)
    x_{k+1} = AndersonAccelerator.update!(acc, x_k, g_k)   # extrapolate the (u,p) DOF vector
    stop when ‖x_{k+1}−x_k‖∞ / ‖x_{k+1}‖∞ < xtol
```

The bare staggered π-iteration (no extrapolation) is linearly convergent; Anderson mixes a short history
of `(state, residual)` pairs via a small least-squares fit to extrapolate a faster step. The fit is
**L²-weighted** by the block `(u,p)` mass matrix so it respects the function-space norm rather than raw DOF
magnitudes. Both schemes converge to the same discrete OSGS fixed point `R̃ = R − Π(R) = 0`.

## Config (all in `numerical_method.solver`)

| key | default | meaning |
|---|---|---|
| `osgs_anderson_enabled` | `false` | opt in; OFF ⇒ the existing single-inexact-Newton path runs, **bit-identical** |
| `osgs_anderson_depth` | `5` | Anderson history depth `m` (past differences mixed) |
| `osgs_anderson_relaxation` | `1.0` | mixing factor `β` on the fixed-point residual |
| `osgs_anderson_safety_factor` | `10.0` | Powell-style restart threshold (reject a too-large extrapolation) |
| `osgs_anderson_max_outer` | `50` | cap on staggered outer iterations |

## Does it actually accelerate? — yes

Experiment (`scratchpad anderson_accel_experiment.jl`): the staggered loop on a P2/P1 2D MMS problem with
three accelerators — `m=0` (relaxed-Picard, **no** Anderson mixing — the baseline staggered iteration),
`m=5`, `m=10` — counting **outer iterations** to a fixed state-drift tolerance (`xtol = 1e-9`). Outer/inner
iteration counts are the fair metric (compile-independent); wall-time on a cold first call is JIT-dominated.

| case | accelerator | outer iters | inner Newton | L2u velocity error |
|---|---|---|---|---|
| **MILD** (8×8, σ=1) | Picard `m=0` (no accel) | 33 | 40 | 2.46593e-2 |
| | Anderson `m=5` | **18** | 23 | 2.46593e-2 |
| | Anderson `m=10` | **15** | 20 | 2.46593e-2 |
| **STIFF** (16×16, σ=1e3) | Picard `m=0` (no accel) | 19 | 21 | 9.69220e-4 |
| | Anderson `m=5` | **14** | 16 | 9.69216e-4 |
| | Anderson `m=10` | **12** | 14 | 9.69215e-4 |

**Findings:**
1. **It accelerates.** Anderson cuts the staggered outer-iteration count by **≈1.4–2.2×** vs the
   non-accelerated (`m=0`) baseline. The effect is larger where the bare staggered rate is slower (MILD:
   33→15; STIFF: 19→12).
2. **Deeper history helps more.** `m=10` beats `m=5` in both cases (15<18, 12<14).
3. **Same solution.** Every accelerator converges to the identical discrete solution (`L2u` matches to ~6
   significant figures), confirming Anderson changes only the *path*, not the fixed point.
4. **Wall-time.** In the warm STIFF comparison (everything already compiled) the wall-clock also drops with
   acceleration (≈22 s → ≈15 s). The MILD `m=0` time (≈135 s) is a cold-start JIT artifact of the first
   solve, not a real 6× slowdown — read the iteration counts, not that one wall-time.

## Caveats / scope

- This is the **opt-in** path; it is OFF everywhere by default, and OFF is bit-identical to the prior code
  (verified: 2D Re=1e6 ASGS+OSGS solution unchanged).
- The staggered scheme does a *full* frozen-π Newton solve per outer step (each converges in 1–2 Newton
  iterations via the exact tangent), so "inner" counts are small; the win is in the **outer** count.
- The default single-inexact-Newton OSGS path remains the production default. Anderson is infrastructure
  for the reaction-/convection-dominated regimes where the linear rate is the bottleneck; tuning
  (`depth`/`relaxation`/`safety`) and a broader sweep are future work.

## Enable / reproduce

Set `"osgs_anderson_enabled": true` (optionally bump `osgs_anderson_depth`) in the `solver` block. The
acceleration experiment above is reproducible from the scratchpad driver
`anderson_accel_experiment.jl` (staggered loop with swappable `AndersonAccelerator` depth).
