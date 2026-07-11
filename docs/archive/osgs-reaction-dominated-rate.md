# OSGS rate loss in the reaction-dominated limit (high Da) — status, evidence, options

> **ARCHIVED 2026-07-11 — RESOLVED.** The settled verdict + the four keeper findings (the annihilation-probe
> root cause `‖(I−Π)(σu_h)‖/‖σu_h‖=3e-16`, the innocent-but-load-bearing trim, the 22-agent methodological
> caution, and the OSGS-retains-full-σ theory-sign correction) are canonical in
> [`findings.md`](../findings.md) §4; the permanent mechanism derivation is in `theory/osgs_reaction_note/`.
> Kept here as the full record of the deleted staggered route and the audit process.

**Canonical doc** for the open OSGS convergence-rate issue at high Damköhler number. Supersedes the
"discrete staggered map" mechanistic account in [`../findings.md`](../findings.md) §4/§7 (Open numerical
defect) and the 2026-06-04 caveats in [`../mms/convergence-2d.md`](../mms/convergence-2d.md) §4–6.

Last updated: 2026-06-10.

> **RESOLVED 2026-06-10 — the rate is pre-asymptotic and recovers; read two corrections below.**
> 1. **Empirical (N=640).** The completed k=1 QUAD sweep shows the reaction-dominated OSGS velocity H¹
>    rate climbing `0.57→0.54→0.58→0.73→1.11→1.85` (α₀=1, Da=1e6, N=10→640): the "0.5–0.7" quoted
>    throughout this doc is the **N≤320 pre-asymptotic value**, and it **recovers to ≥1.0 once N=320→640
>    is in**. So OSGS reaches the **optimal** rate — this is a slow pre-asymptotic climb, **not** an order
>    ceiling. (Numbers: [`../mms/convergence-2d.md`](../mms/convergence-2d.md) success box.)
> 2. **Mechanism (theory correction).** The companion note's thesis was **inverted and corrected
>    2026-06-09**: the constant-σ annihilation means OSGS *retains the full Galerkin σ* on ‖u‖, whereas
>    ASGS's stabilization *drains* its reactive coercivity to σ_a = σ−τ₁σ² < σ — so OSGS's coercivity
>    constant is the **larger** of the two, not smaller. The high-Da OSGS rate dip is therefore **not a
>    coercivity loss** (OSGS coercivity is stronger); it is a pre-asymptotic *consistency/approximation*
>    transient in the under-resolved reaction regime that vanishes as h→0 (item 1). The TL;DR and
>    §-mechanism below predate this correction and read the gap backwards — treat them as historical;
>    the empirical conclusion (recovery) stands regardless.

Companion theory note: [`../../theory/osgs_reaction_note/osgs_reaction_note.tex`](../../theory/osgs_reaction_note/osgs_reaction_note.tex)
(corrected 2026-06-09 — OSGS retains σ; ASGS drains to σ_a). Reproduction recipe: see §7 (the coupled
probe is the only surviving route; the historical `staggered` A/B config was deleted in the 2026-06-08
leaning).

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

## 2. The decisive test — `staggered` vs `coupled` (2026-06-05) — HISTORICAL

> **Route note (2026-06-08):** the `staggered` coupling mode and its satellite (the outer relaxation
> loop, the state-/projection-drift stopping metrics, the warm-up schedule, Anderson acceleration) were
> subsequently **deleted**. The OSGS solver now runs the **coupled** scheme unconditionally, regardless
> of any projection-coupling knob. The A/B below is retained as **historical evidence** that
> `staggered == coupled` to ~4 sig figs — i.e. that the suboptimality is a property of the OSGS fixed
> point, not of the route that reaches it. It is not a current reproduce recipe; see §7.

**Cell:** Re=1, Da=1e6, α₀=1, k=1, QUAD, N=10→320, Dirichlet BCs, constant `σ = Da·α∞·ν/L²`.
α₀=1 is chosen deliberately: no porosity layer, no fold — the *only* stressor is the large constant
reaction. Three curves: **ASGS**; **OSGS-staggered** (paper `alg:StationarySystem`, `osgs_iterations=5`,
`osgs_inner_newton_iters=1`); **OSGS-coupled** (one Newton solve recomputing `π=Π(R(u))` every nonlinear
iteration — no staggering lag; converges to the *same* fixed point — this is the **sole surviving route**
today).

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

### A. Solver / coupling — changes *how fast* you reach the fixed point, **not** the fixed point

The coupling never changes the OSGS solution (it shares the §2 accuracy; it does not fix the rate).
`coupled` is the sole surviving route — the `staggered` and `freeze_after_k` modes, and the outer
relaxation satellite, were deleted in the 2026-06-08 leaning:

| mode | projection schedule | convergence | cost (this cell, N=80) |
|---|---|---|---|
| `coupled` | recompute `π=Π(R(u))` every Newton iter; local frozen-π Jacobian | linear (Picard; **44 iters** here) | slowest |

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

## 6. Coupling: superseded — coupled is the only route

The earlier recommendation here ("make `freeze_after_k` the default OSGS coupling") is **superseded and
falsified**: against the coupling-equivalence oracle, `freeze_after_k` diverged / gave a negative rate in
the reaction corner — it does *not* land on the coupled fixed point there — so it was rejected and, with
`staggered`, **deleted** in the 2026-06-08 leaning. The OSGS solver now has the single `coupled` route.
For the coupling rationale and the iteration-cost fix (JFNK — landed; see
[`jfnk-phase0-preconditioner-gate.md`](../solver/jfnk-phase0-preconditioner-gate.md)), see
[`coupled-only-leaning-and-jfnk-plan.md`](coupled-only-leaning-and-jfnk-plan.md) §2. The rate cure itself
is the deferred **split / term-by-term OSGS** of §4B / §8.5 — a formulation change, not a coupling choice.

## 7. Reproduce

Current (coupled is the only OSGS route):

```bash
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl probe_re1_da1e6_coupled.json     # ASGS + OSGS-coupled, N=10→320
```
Per-mesh `L2 u/p` / `H1 u/p` print to stdout.

The original §2 A/B used a second config, `probe_re1_da1e6_staggered.json`, differing only in
`numerical_method.stabilization.osgs_projection_coupling` (`"staggered"` vs `"coupled"`). That knob and
the `staggered` route **no longer exist** (2026-06-08 leaning), so the A/B is not re-runnable; the
bit-identical OSGS errors it produced are recorded as the §2 historical result.

---

# 8. Deep root-cause audit (2026-06-06) — confirmed genuine, bug-hypothesis refuted

A full audit was run to settle whether the rate loss is a **code bug** or a **genuine OSGS property**,
double-checking every assumption. Method: a 22-agent code/theory/literature audit (parallel audits of
trim, τ, stabilization operator/adjoint, coercivity-gap math, consistency, projection/boundary, MMS
forcing, literature, asymptotics → adversarial verification → synthesis), plus **three independent
empirical double-checks** I ran myself. **Conclusion: genuine, pre-asymptotic OSGS coercivity gap — not a
bug.** The fix for optimal rate at practical meshes is formulation-level split/term-by-term OSGS.

## 8.1 Components audited and cleared (no bug)
τ₁/τ₂ (faithful to `eq:Tau1/Tau2/TauNavierStokes`, `c₁=4k⁴`, `c₂=2k²`); the adjoint and convective-adjoint
signs; the MMS forcing and `σ_c = Da·α∞·ν/L²`; the L² projection operator; constant-σ derivative; the
solver staggering. The reaction-projection **trim is correct** (see 8.2). No code error was found in any
component.

## 8.2 The decisive double-checks

**(a) Annihilation probe — machine-exact.** Direct numerical test
(`/tmp/annihilation_probe.jl`-style): project `σ·u_h` onto the implemented unconstrained `V_free`
(`TestFESpace(model, refe_u)`, same degree as `u_h`, [run_test.jl:797](../../test/extended/ManufacturedSolutions/run_test.jl#L797))
with **inhomogeneous Dirichlet** boundary data and σ=1e6:

> `‖(I−Π)(σu_h)‖ / ‖σu_h‖ = 3e-16` at quadrature degree 2, 4, and 8.

So `(I−Π)(σu_h)=0` holds to machine precision (`σu_h ∈ V_free` exactly; `b = σM·u ⇒ π = σu`). Therefore
the **trimmed and full-residual OSGS stabilized residuals are bit-identical** — `(I−Π)(R_u−σu)=(I−Π)(R_u)` —
and they target the **same OSGS fixed point**. The trim is provably equivalent at the fixed point and is
**not** the cause.

**(b) Da-sweep — the mechanism is the mesh-Damköhler.** Re=1, α=1, OSGS, N=40/80/160, with the mesh
Damköhler `Da_h = σh²/(c₁ν) = Da·α∞/(c₁N²)`:

| Da | Da_h@N160 | OSGS velocity rate | reading |
|---|---|---|---|
| 1 | 1e-5 | H¹ 1.00 / L² 2.00 | optimal — gap absent |
| 10² | 1e-3 | H¹ 1.00 / L² 2.00 | optimal — gap absent |
| 10⁴ | 0.1 (1.56→0.10 over N=40→160) | H¹ **1.63** / L² **2.68** (40→80) | *super-convergent recovery* as `Da_h` crosses 1 |
| 10⁶ | 9.8 | H¹ 0.57→0.71 / L² 1.57 | degraded but **climbing** as h↓ |

OSGS is **bit-optimal wherever `Da_h ≪ 1`** and the loss switches on as `Da_h` crosses O(1) — the signature
of a coercivity gap `∝ Da_h`, **not** a bug (a bug would hit every Da). Because `Da_h ∝ 1/N²`, the gap
**closes as h→0** ⇒ the degradation is **pre-asymptotic** (very slow at Da=10⁶: `Da_h<1` needs N≳640). This
vindicates the note's "optimal rate retained asymptotically" (items i–iii) — the meshes were just never fine
enough at Da=10⁶.

**(c) Full-vs-trim A/B — full-residual OSGS is *unstable*, not "optimal".** Switching the harness to
`ProjectFullResidual` (config-driven, [run_test.jl:185](../../test/extended/ManufacturedSolutions/run_test.jl#L185),
`experimental_reaction_mode != "standard"`): full-residual OSGS at Da=10⁶ **stalls/diverges at every mesh**
(`[❌] Outer loop completely stalled`, → NaN at N=80) and reports the **ASGS Stage-I fallback** — N=40 H¹
`0.07789` and N=160 H¹ `0.01822`, *bit-identical to ASGS*. It never reaches the OSGS fixed point. The reason:
although the residual equals the trim's (by 8.2a), the **Jacobians differ** — the trim subtracts a `−σ·du`
term that is **load-bearing for solver stability**; the full Jacobian is unstable at high σ. **This is why
the paper trims**, and the trim is not removable.

## 8.3 Verified root cause
The OSGS orthogonal projection annihilates the reactive residual (`(I−Π)(σu_h)=0`, 8.2a), so OSGS loses the
reactive stabilization square `−‖τ₁^{1/2}σu‖²` that gives ASGS its H¹-strength `σ̃_α ~ α(1+Re_h)ν/h²` velocity
control; OSGS controls velocity only at strength σ, weaker by the **mesh Damköhler** `Da_h = σh²/(c₁ν)`. The
realized velocity rate is therefore degraded **pre-asymptotically**, recovering as `Da_h ∝ 1/N² → 0`. This is
a genuine property of the paper's OSGS, faithfully implemented — **not a bug.**

## 8.4 Methodological caution (recorded deliberately)
The 22-agent audit's *synthesis* reached the **wrong** verdict ("code-bug; full-residual recovers optimal
rate; trim is the cause", conf 88) by reading the **ASGS-fallback number** of an unstable full-OSGS solve as
"full-OSGS optimal." Its own component agents disagreed (the trim dimension rated it 8/100, "no bug"). The
error was caught by the two independent checks above (annihilation probe = 3e-16; the full-OSGS stall). **A
confident multi-agent synthesis is not a substitute for a direct numerical probe of the load-bearing
assumption.**

## 8.5 The fix (in full detail)
**Primary — split / term-by-term OSGS.** Keep the *reactive* zeroth-order term `σu` in the stabilization
with **ASGS (identity-projection)** treatment, while the *convective* and *pressure-gradient* terms keep the
**orthogonal** projection. This restores the `−‖τ₁^{1/2}σu‖²` reactive square (hence the `σ̃_α` velocity
coercivity) **at all h**, recovering the optimal rate without the mesh-Damköhler pre-asymptotic penalty, and
**preserves solver stability** (unlike full-residual). Standard in the OSGS reaction-dominated literature.
Implementation: a new projection policy (e.g. `ProjectResidualSplitReaction`) whose `apply_projection_u`
returns `(σu) + (I−Π)(R_u − σu)` for the OSGS branch — i.e. the convection/pressure part orthogonally
projected, the reaction part kept whole (ASGS-style) — with the matching Jacobian; wired through
`continuous_problem.jl` and selectable per config. Verify on the Da=10⁶ ladder: expect H¹→1.0 **and** a
stable solve.

**Alternatives.** (i) *Do nothing* — the rate is correct asymptotically; acceptable if meshes reach N≳640 at
Da=10⁶ (confirm with the N=640/1280 ladder). (ii) *Full-residual OSGS* — **rejected**: solver-unstable
(8.2c). (iii) *Constrained-space projection* — rejected (breaks the O(h^{k+1}) boundary property).

## 8.6 Reproduce the audit checks
```bash
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl da_sweep_audit.json   # Da-gated degradation (rate vs Da)
julia --project=../../.. run_test.jl da1e6_full.json       # full-residual OSGS (experimental_reaction_mode=full_residual) -> unstable/ASGS-fallback
# annihilation probe: project σ·u_h onto V_free, measure ‖(I-Π)(σu_h)‖  -> ~3e-16
```
