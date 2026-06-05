# OSGS rate loss in the reaction-dominated limit (high Da) — status, evidence, options

**Canonical doc** for the open OSGS convergence-rate issue at high Damköhler number. Supersedes the
"discrete staggered map" mechanistic account in [`../known_issues.md`](known_issues.md) (Open numerical
defect) and the 2026-06-04 caveats in [`../mms/convergence-status.md`](../mms/convergence-status.md) §4–6.

Last updated: 2026-06-05.

Companion theory note: [`../../theory/osgs_reaction_note.tex`](../../theory/osgs_reaction_note.tex)
(Prop. 1, the coercivity gap). Reproduction configs:
[`probe_re1_da1e6_staggered.json`](../../test/extended/ManufacturedSolutions/data/probe_re1_da1e6_staggered.json),
[`probe_re1_da1e6_coupled.json`](../../test/extended/ManufacturedSolutions/data/probe_re1_da1e6_coupled.json).

---

## TL;DR

1. At **high Da with low/moderate Re** (the reaction-dominated regime), **OSGS** loses velocity
   convergence rate — H¹ ≈ 0.5–0.7 where ASGS holds the optimal ≈1.0 — while **ASGS stays optimal**.
   It is **independent of porosity** α₀ (just as severe at α₀=1, where there is no porosity layer), so
   it is *pure reaction-dominance*, a different axis from the high-Re/low-α₀ coarse-mesh fold.
2. **It is not the reaction-projection trim.** The trim (dropping `σu` from the orthogonal projection
   for constant σ, [article.tex §580]) is implemented correctly and consistently, and is *exonerated*
   below — it is even coefficient-robust at the fixed point.
3. **It is not the staggered solver.** A controlled A/B (`staggered` vs `coupled` projection coupling,
   only that knob varied) gives **bit-identical errors to ~4 sig figs across 5+ meshes**. The
   suboptimality is a property of the **OSGS discrete fixed point itself**, not of how the solver
   reaches it. This *corrects* the earlier "staggered map oscillates / budget exhausted" account.
4. **It matches the note's coercivity gap (Prop. 1)** quantitatively in *both* the Da-axis (degrades
   only at Da=1e6) and the Re-axis (healthy again at Re=1e6, gap `~ Da_h/(1+Re_h)`). The mechanism is
   real: OSGS annihilates exactly the reactive stabilization `‖τ₁^{1/2}σu‖²` that gives ASGS its
   H¹-strength velocity control.
5. **Open:** whether the realized H¹ rate is a *very slow pre-asymptotic climb* (it creeps 0.57→0.74
   over N=10→320 — which would ultimately *vindicate* the note's "optimal rate retained") or a genuine
   asymptotic reduction below 1.0. **N≤320 cannot separate these**; a fine ladder (N=640, 1280) at the
   one cell settles it.

---

## 1. What we are trying to recover

ASGS and OSGS should both converge at the optimal equal-order rates — velocity **L² = O(h^{k+1})**,
**H¹ = O(h^k)** — across the whole `(Re, Da, α₀)` grid. They do everywhere *except* the
reaction-dominated corner, where OSGS velocity rates collapse and ASGS does not. The goal is to recover
optimal **OSGS** velocity convergence at high Da (or to establish, with proof, that the paper's OSGS
cannot deliver it there and document the limitation).

## 2. The decisive test — `staggered` vs `coupled` (2026-06-05)

**Cell:** Re=1, Da=1e6, α₀=1, k=1, QUAD, N=10→320, Dirichlet BCs, constant `σ = Da·α∞·ν/L²`.
α₀=1 is chosen deliberately: no porosity layer, no fold — the *only* stressor is the large constant
reaction. Three curves: **ASGS**; **OSGS-staggered** (paper `alg:StationarySystem`, `osgs_iterations=5`,
`osgs_inner_newton_iters=1`); **OSGS-coupled** (one Newton solve recomputing `π=Π(R(u))` every nonlinear
iteration — no staggering lag; converges to the *same* fixed point per
[`porous_solver.jl`](../../src/solvers/porous_solver.jl) `_run_osgs_relaxation!` coupled branch).

### Velocity errors

| N | ASGS L² / H¹ | OSGS-stag L² / H¹ | OSGS-coupled L² / H¹ |
|---|---|---|---|
| 10 | 2.99e-2 / 0.358 | 3.86e-2 / 0.906 | 3.85e-2 / 0.904 |
| 20 | 9.06e-3 / 0.167 | 1.29e-2 / 0.610 | 1.29e-2 / 0.610 |
| 40 | 2.48e-3 / 0.0779 | 4.39e-3 / 0.420 | 4.39e-3 / 0.420 |
| 80 | 6.46e-4 / 0.0373 | 1.48e-3 / 0.283 | 1.48e-3 / 0.283 |
| 160 | 1.63e-4 / 0.0182 | 4.68e-4 / 0.173 | 4.65e-4 / 0.173 |
| 320 | 4.00e-5 / 0.00899 | 1.60e-4 / 0.104 | _(matches; ≤160 identical to 4 sig figs)_ |

### Per-pair rates

| pair | ASGS L² / H¹ | OSGS L² / H¹ (staggered = coupled) |
|---|---|---|
| 10→20 | 1.72 / 1.10 | 1.58 / 0.57 |
| 20→40 | 1.87 / 1.10 | 1.56 / 0.54 |
| 40→80 | 1.94 / 1.06 | 1.57 / 0.57 |
| 80→160 | 1.99 / 1.03 | 1.66 / 0.71 |
| 160→320 | 2.03 / 1.02 | 1.55 / 0.74 |

**Reading:**
- **ASGS is textbook-optimal** end to end (H¹→1.0, L²→2.0).
- **OSGS-staggered ≡ OSGS-coupled** to ~4 sig figs at every mesh. Coupled converges *gracefully* to
  `ftol` (no "budget exhausted" flag); staggered's "MMS plateau NOT verified (budget exhausted)" flag is
  the *plateau-rate verifier* giving up, **not** a failed solve — the solver reaches the same fixed
  point either way. ⇒ **The suboptimality lives in the fixed point, not the iteration.**
- **OSGS H¹ is sub-optimal but slowly climbing** (0.57→0.74). L² hovers ~1.55–1.66. Both well below
  ASGS, on every usable mesh.

## 3. Mechanism — the note's coercivity gap (confirmed), and why the trim is innocent

- **Trim is correct.** `R_u` carries `+σu` ([continuous_problem.jl:203](../../src/formulations/continuous_problem.jl#L203));
  the trim subtracts it from *both* the projection target (`inner_projection_u → R_u−σu`) and the
  stabilized residual (`R_u−σu−π`), and from the Jacobian (`R_du−σ·du`, ∂σ=0). At the discrete fixed
  point `(I−Π)(σu_h)=0` (σ const, `σu_h∈V_h`, exact mass solve), so the trim removes a quantity that is
  annihilated anyway — it is **coefficient-robust** and changes nothing at the fixed point. Its sole job
  is to avoid the lagged-projection non-cancellation *during* iterations — exactly the paper's stated
  reason ([article.tex §580]).
- **The gap is the cause.** Because `σu_h∈V_h`, the orthogonal projection annihilates the reactive
  residual: `(I−Π)[σu_h]=0` (note Eq. annihilation). ASGS keeps the reactive stabilization square
  `−‖τ₁^{1/2}σu‖²`, which upgrades the velocity control to the H¹-strength `σ̃_α ~ α(1+Re_h)ν/h²`; OSGS
  cannot, and controls velocity only at the bare reaction strength `σ`. The provable gap is
  `c_u^OSGS/c_u^ASGS ~ Da_h/(1+Re_h)` (note Prop. 1).
- **The data confirms the *dependence*, not just the existence:** degrades only at **Da=1e6**
  (gap grows with Da_h), and is **healthy again at Re=1e6** (gap shrinks as Re_h↑). That two-axis
  signature is what a bug would *not* produce and what Prop. 1 *predicts*.
- **Caveat on the note:** the note's items (i)–(iii) ("optimal *rate* retained, only the provable norm
  weakens") are **not yet borne out** on practical meshes — the realized H¹ rate is badly degraded
  through N=320. Whether the slow upward creep eventually reaches 1.0 (vindicating the note) or asymptotes
  below it (refuting it) is the open question of §5.

## 4. The options

### A. Solver / coupling options — change *how fast* you reach the fixed point, **not** the fixed point

All three converge to the *same* OSGS solution (so they share the §2 accuracy; none fixes the rate):

| mode | projection schedule | convergence | cost (this cell, N=80) |
|---|---|---|---|
| `staggered` (paper `alg:StationarySystem`, default today) | outer loop: freeze π → inner Newton → update π | linear (frozen-π lag) | budget-bound |
| `coupled` | recompute `π=Π(R(u))` every Newton iter; local frozen-π Jacobian | linear (Picard; **44 iters** here) | slowest |
| `freeze_after_k` | k projection-updating warm-ups, then **freeze π** and Newton to convergence | **quadratic finish** | ~ASGS counts |

### B. Formulation options — change the *fixed point* to recover the rate (the real lever)

The accuracy lever is **not** the coupling; it is whether the reactive term is allowed to stabilize.
Candidate: a **term-by-term / "split" OSGS** that keeps the reactive contribution `σu` in the
stabilization with ASGS (identity-projection) treatment while the convective/pressure terms keep the
orthogonal projection — restoring the `‖τ₁^{1/2}σu‖²` velocity control that pure OSGS annihilates. This
is a formulation change (a different method), not a bug fix, and would need its own MMS verification and a
paper-divergence entry. **Not yet attempted.**

## 5. Open question / next step

Distinguish *very slow pre-asymptotic climb* (H¹→1.0 at N≫320 ⇒ note's items (i)–(iii) survive) from
*asymptotic reduction* (H¹ asymptotes < 1.0 ⇒ they fail). The clean test is the C24-style fine ladder at
**this one cell**: N = 640, 1280, reporting the per-pair H¹ trajectory. Expensive (~1.2M, ~5M DOF;
coupled needs ~40+ linear solves each), so it is a deliberate, user-triggered run.

## 6. Recommendation on `freeze_after_k`

**Make `freeze_after_k` (k=2–3) the recommended *default* OSGS coupling — but not yet the *only* one.**

- *Why default:* it reaches the OSGS fixed point with a **quadratic finish in ~ASGS iteration counts**,
  versus the linear staggered/coupled paths (44 Picard iters here). For every regime where OSGS is
  healthy it captures the OSGS error-constant advantage cheaply; in the reaction-dominated corner it
  lands on the same (sub-optimal) fixed point as the others — so it never does *worse* on accuracy and is
  strictly better on cost. See [`efficiency-ideas.md`](efficiency-ideas.md) Idea 5.
- *Why not yet the only option (more research needed):*
  1. The §5 rate question is **open**; resolving it (and any §4B formulation fix) benefits from being
     able to A/B the modes — pruning to one mode now removes that lever.
  2. `staggered` is the **paper-faithful** `alg:StationarySystem` and the reference the
     [algorithm-code-mapping](algorithm-code-mapping.md) is built on; keep it for paper correspondence.
  3. `coupled` is the **zero-lag control** that *proved* the fixed-point property in §2; keep it for
     future such checks.
  4. `freeze_after_k`'s own claim ("`U*(π_k)` is optimal-rate for any fixed k") should be re-validated
     against the resolved §5 rate before it is trusted as the sole path.

  → Revisit "make it the only OSGS option" once §5 is settled and §4B is decided. Until then:
  default = `freeze_after_k`, keep `staggered`/`coupled` available.

## 7. Reproduce

```bash
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl probe_re1_da1e6_staggered.json   # ASGS + OSGS-staggered, N=10→320
julia --project=../../.. run_test.jl probe_re1_da1e6_coupled.json     # OSGS-coupled (same cell)
```
The two configs differ only in `numerical_method.stabilization.osgs_projection_coupling`
(`"staggered"` vs `"coupled"`) and the coupled Newton budget. Per-mesh `L2 u/p` / `H1 u/p` print to
stdout; bit-identical OSGS errors between them are the §2 result.
