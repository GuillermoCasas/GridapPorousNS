# Manufactured-Solution (MMS) convergence — status & knowledge

**Scope.** This document is about the *standard* manufactured-solution test in
`test/extended/ManufacturedSolutions` — the `(Re, Da, α₀, h)` convergence sweep that is the
codebase's primary correctness criterion. It is written to stand **on its own**: the Cocquet
benchmark (`docs/cocquet/convergence-analysis.md`) is a separate, more complicated case (mixed
inlet/wall/**traction-free Neumann** boundary conditions, and a now-fixed mesh-dependent reaction
floor) and is **not** extreme; it will be re-interpreted in light of the manufactured case, not the
other way around. Where a Cocquet/CocquetFormMMS result is used below it is clearly labelled as
**external supporting evidence**, never as a premise.

Last updated: **2026-06-10** — the k=1 QUAD sweep is **complete (N=10→640) and a success**; see the
status box below. This is the **canonical** MMS reference; `fold-recovery.md` is its fold sub-topic.
The earlier dated caveat block (2026-05-26 → 06-05, dispatch-bug / encoding-covariance / "Da benign" /
the high-Da coercivity-gap "open defect") is now **resolved or superseded** — preserved in git history
and [`lessons_learned.md`](../lessons_learned.md); the live conclusions are folded into the box below.

> **Mass-gate update (2026-07-04).** The convergence mass gate is now the Route-B **Philosophy-A
> algebraic ε_C = ‖r_C‖/D_C**, gated ~1e-9 and symmetric with the momentum gate ε_M; the old loose
> `eps_tol_mass = 0.8` gate is demoted to a diagnostic (`eps_C_strong`). See
> [`route-b-2d-sweep-status.md`](route-b-2d-sweep-status.md) for the current gate and the completed
> k1/k2 QUAD sweeps. The success box and §7 below describe the older ε_M/ε_C-with-0.8 regime and are
> kept for the narrative; the Route-B doc is authoritative on the gate.

> ## ✅ 2026-06-10 — k=1 QUAD sweep COMPLETE (N=10→640): success
>
> The full `(Re, Da, α₀)` × {ASGS, OSGS} sweep at k=1 P1/P1 QUAD is finished on the scale-free-gate
> solver (the authoritative ε_M/ε_C criterion; DB `results/phase1_quad_k1.h5`). Verified numbers:
>
> - **Velocity is optimal across the entire grid.** L² rate (finest pair N=320→640) ≥ **1.93** on every
>   one of the 48 cells (median 2.00 ASGS / 2.07 OSGS); H¹ rate ≥ **1.00** everywhere.
> - **Pressure is optimal across the entire grid too.** Its nominal equal-order order is
>   O(h^{kp}) = O(h) (the "(1)" on the convergence plots, and the analyzer's `opt_p = kp` target); every
>   cell's finest-segment L² rate is **1.5–2.4**, i.e. **at or above** that order on all 48 cells — in
>   fact consistently **super-optimal** (1.5–2.4× the nominal order), reaching toward O(h²) on most.
>   **Zero sub-optimal cells.** The low-Re/low-Da/α₀=1 (Stokes-like) cells sit at the low end (~1.5–1.7),
>   still comfortably above the O(h) order; the rest run ~2.0+. (Pitfall: do **not** score pressure
>   against the velocity O(h²) target — pressure's optimal order is O(h) for equal-order P1/P1.)
> - **The high-Da OSGS "coercivity gap" open question is SETTLED — pre-asymptotic, and it recovers.**
>   The Da=1e6 reaction-dominated OSGS cells show the H¹ rate climbing
>   `0.57 → 0.54 → 0.58 → 0.73 → 1.11 → 1.85` (N=10→640): flat ≈ 0.7 through N≤320 (the value the old
>   baseline reported as the "defect"), then climbing to ≥ 1.0 once the N=320→640 pair is in. This is
>   exactly the **pre-asymptotic erosion, not asymptotic order reduction** hypothesis of §4/§7 — now
>   demonstrated, not conjectured. Mechanism: the coercivity gap of
>   [`../../theory/osgs_reaction_note/osgs_reaction_note.tex`](../../theory/osgs_reaction_note/osgs_reaction_note.tex)
>   degrades the coercivity *constant* (σ_a), not the convergence *rate*, and closes like Da_h ∝ 1/N².
> - **OSGS is ≈ 2× more accurate than ASGS** at the same rate (finest-mesh-error ratio 0.50 velocity,
>   0.41 pressure) — the orthogonal projection buys accuracy, at higher iteration cost (now cut by the
>   landed JFNK solve; see [`high-order-convergence-gate-and-jfnk.md`](high-order-convergence-gate-and-jfnk.md)).
> - **Behaviour-preserving:** errors are byte-identical to the pre-scale-free-gate archive on every
>   overlapping mesh — the ε_M/ε_C gate changed *when* the solver stops, not *where*.
> - **Expected caveats (not defects):** Re=1e6 @ N=10 is `NaN` (boundary layers ∼ Re^{-1/2}=1e-3 ≪ h=0.1,
>   hopeless on a 10×10 grid); the three Re=1e6/α₀=0.05 cells are `skip_cells` (the coarse-mesh fold, §3).
>   **(Update 2026-06-17:** these three TRI/P1 corner cells are now reproduced — ASGS+OSGS — by a *direct
>   exact-guess solve* at N≥512, no continuation needed; the Q2/QUAD-k2 corner is also done (it does
>   not fold — k=2 converges directly at N=160→320). See
>   [`fold-recovery.md`](fold-recovery.md).)
>   `analyze_results.py`'s detector still flags ≈ 29/48 as fold/no-root — a **conservative per-pair
>   artifact** (pre-asymptotic coarse meshes, the N=10 NaN pulling global fits, super-convergent tails);
>   its own rate-check reports **0 sub-optimal and 0 super-convergent**. Read the per-pair ratios, not the
>   one-word verdict.
>
> Per-cell numbers: [`convergence-baseline.md`](convergence-baseline.md). The former "open numerical
> defect" (high-Da OSGS) is closed — [`../known_issues.md`](../known_issues.md),
> [`../solver/osgs-reaction-dominated-rate.md`](../solver/osgs-reaction-dominated-rate.md).

---

## TL;DR

1. **The manufactured test is well-behaved.** Across the bulk of the `(Re, Da, α₀)` grid the
   stabilized equal-order method converges at **full optimal order** — velocity `L²` at `O(h^{k+1})`,
   velocity `H¹` at `O(h^k)` — straight from coarse meshes. No mystery there.
2. **All the difficulty is concentrated in one extreme corner: high `Re` combined with low porosity
   `α₀`** (worst when the reaction is also weak, i.e. low `Da`). The simpler manufactured formulation
   removes every confound, so this corner is the *pure* signal.
3. **In that corner the discrete solution branch _folds_ at coarse mesh** — there is literally no
   FE root to converge to, so the sweep "fails" there for a genuine mathematical reason, not a bug.
   The fold **recedes with refinement**; once a root exists (≈N=512) a *direct exact-guess solve*
   reaches it (continuation is only needed to reach a root at coarse N).
4. **Once past the fold, the corner converges optimally — even super-convergently.** The decisive
   measurement (cell C24, `Re=1e6, Da=1, α₀=0.05`, via continuation to `N=512→1024`): velocity
   **`H¹` rate ≈ 1.01–1.04** (textbook-optimal for `k=1`) and velocity **`L²` rate ≈ 2.99–3.03**
   (*above* the nominal 2), at machine-zero residuals. **So the apparent "sub-optimal h-convergence"
   is a pre-asymptotic transient near the fold, not a fixed order ceiling** — your hypothesis.
5. **Open question (now narrowly scoped):** confirm that this optimal recovery holds for *every*
   extreme cell across `k∈{1,2}`, `{QUAD,TRI}`, and the full `Da` range — not just the two corner
   cells (C24/C21) measured so far. The full sweep + a fine-mesh ladder answers it per cell.

---

## 1. The manufactured problem — why it is the clean case

`run_test.jl` builds, for each `(Re, Da, α₀, k, element)`:

| Ingredient | Choice | Consequence for convergence |
|---|---|---|
| Boundary conditions | **Dirichlet on all 8 tags** (`run_test.jl:381,391`) | Aubin–Nitsche elliptic-duality holds ⇒ the velocity-`L²` *extra* order `k+1` is **expected** here. (Contrast: the Cocquet benchmark's traction-free Neumann outlet breaks exactly this duality — see §6.) |
| Reaction | **Constant** `σ_c = Da·α∞·ν/L²` (`run_test.jl:76`; "always uses `ConstantSigmaLaw`", `:400`) | `σ` does **not** vary in space, so there is no steep `σ(α)` gradient. A *larger* `σ` (high `Da`) is a *coercive*, stabilizing term — not a trouble source. `Da` is therefore a benign linear knob, **not** a difficulty axis. |
| Porosity | `SmoothRadialPorosity`: `C∞` logistic `α₀→1` across `r∈[r₁,r₂]=[0.2,0.4]` (`src/models/porosity.jl`) | The varying `α(x)` still enters the **porosity-weighted convection / mass** operators. It is infinitely smooth (no regularity limit), but **steep** when `α₀` is small — this is the only spatial-gradient stressor in the problem. |
| Exact solution | smooth closed-form `(u_ex, p_ex)`; forcing computed analytically (incl. `∇α, ∇²α`) | The error we measure is purely discretization error; the metric is verified exact (see §5). |
| Element | equal-order `Pₖ/Pₖ`, VMS-stabilized (ASGS / OSGS) | Optimal targets: velocity `L² = O(h^{k+1})`, velocity `H¹ = O(h^k)`. |

**Grid:** `Re, Da ∈ {1e-6, 1, 1e6}`, `α₀ ∈ {1.0, 0.5, 0.05}` → 27 physics cells, × {ASGS,OSGS}
× `k∈{1,2}` × {QUAD,TRI}. Meshes `N ∈ {10,20,40,80,160,320}`.

Because the BCs are Dirichlet and `σ` is constant, the manufactured test isolates **convection
(`Re`) interacting with a steep but smooth porosity layer (`α₀`)** as the *only* mechanism that can
degrade convergence. That is what makes it the clean case.

---

## 2. The parameter map — where it is optimal, where it complicates

| Region | Where | Behaviour | Mechanism |
|---|---|---|---|
| **A — optimal** | Most of the grid: low/moderate `Re`, **or** `α₀ ∈ {1, 0.5}` at any `Re`; any `Da` | Root exists at every mesh; **full optimal rates from coarse meshes**. | Nothing stresses the discretization beyond standard FE approximation. (`α₀=1` is pure NS — the porosity coupling vanishes.) |
| **B — fold (coarse-mesh)** | The **extreme corner**: `Re=1e6` **and** `α₀=0.05` (esp. low `Da`) | At coarse `N` the discrete branch **folds**: *no* root with `‖R‖≤ftol` exists. Sweep correctly reports non-convergence there. Root appears once `N` is fine enough; continuation reaches it. | Cell-Péclet `≈ 4.7e4` in the `α=0.05` core at `N=10`: convection (`c₂|u|/h`) overwhelms diffusion (`c₁ν/h²`) by `~10⁴–10⁵`. The steep porosity layer + dominant convection create a coarse-mesh coercivity turning point. |
| **C — pre-asymptotic erosion** | Same extreme cells, **at intermediate meshes just past the fold** | A root *exists*, but the per-pair slope is temporarily **below** optimal, then climbs to optimal/superconvergent as `h→0`. | The steep `α`-layer and high-`Re` convective layers are under-resolved on coarse meshes; the error is layer-dominated until the layer is resolved. **This is region B's recovery seen as a rate, not a separate set of cells.** |

**Refinements to the three-bucket mental model:**

- The trouble axis is **`Re × α₀`**, not `Re × Da`. `Da` only scales the *constant* reaction
  `σ_c`; a large `σ_c` (high `Da`) *helps* (coercive). The fold is driven by **convection + the
  steep porosity layer**, confirmed independent of `Da` (C24 at `Da=1` and C21 at `Da=1e-6` give
  **bit-identical** corner roots — see [`docs/mms/fold-recovery.md`](fold-recovery.md)).
- **Buckets 2 (fold) and 3 (sub-optimal rate) are the same physics at two severities**, not two
  disjoint sets. At the extreme corner the branch folds (no coarse root); short of/just past the
  fold the root exists but the rate is pre-asymptotically eroded. Both resolve by **refining past the
  fold**, after which the rate is optimal.

### Cells across the porosity layer (why coarse meshes struggle there)

Domain width 1, `h = 1/N`; the layer `r∈[0.2,0.4]` has radial width `0.2`, so cells spanning it ≈ `0.2·N` (the *steep core* is roughly half that):

| N | 10 | 20 | 40 | 80 | 160 | 320 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| cells across layer | 2 | 4 | 8 | 16 | 32 | 64 | ~102 | ~205 |

At `N≤40` the steep part of the `α₀=0.05` transition is resolved by only **1–4 cells** — exactly
where slopes are depressed. By `N=512–1024` it is resolved by ~100–200 cells — exactly where C24
recovers optimal/superconvergent rates. The numbers and the observed recovery agree.

> **Direct confirmation for the *standard* (non-folding) `α₀=0.05` cells (2026-06-18).** The same
> layer-under-resolution depresses the velocity-L² rate of the **low/unit-Re** `α₀=0.05` ASGS cells —
> these read ≈1.88–1.95 at the sweep's `N=160→320` finest pair (the values in
> [`paper_tables.tex`](../../test/extended/ManufacturedSolutions/results/paper_tables.tex) / the
> article's `P1` table), below the optimal 2. They are **pre-asymptotic, not a rate loss**: a direct
> exact-guess solve of the *worst* cell (`Re=1, Da=1e-6, α₀=0.05`, ASGS, TRI k=1) extended one mesh
> shows the velocity-L² slope **climbing 1.888 (160→320) → 1.960 (320→512)** — within ~2 % of optimal.
> This matches the QUAD k=1 sweep, where every `α₀=0.05` cell recovers to ≥1.93 by `N=640`. So the
> TRI table's ~1.9 values are honest pre-asymptotic readings at `N=320`; they reach ≈2.0 by `N≈640`.
> (The TRI k=1 sweep stops at `N=320`, so the table cannot show the recovered value without re-running
> the sweep finer — a deliberate scope choice, not a defect.)

---

## 3. The fold (region B) — established

Full diagnosis: [`docs/mms/fold-recovery.md`](fold-recovery.md). Key points:

- **It is a true turning point, not a solver/Jacobian bug.** The Exact-Newton Jacobian was verified
  against finite differences to `4.8e-12`; heavy Newton *and* Picard from the exact solution both
  stall at `‖R‖≈5e-2` (no root near `u_ex`); continuation folds in every parameter direction with
  adaptive step-halving confirming a genuine fold.
- **`α` is the only viable continuation axis.** It starts at `α=1` (easy) and *relieves* the layer;
  `Re`/`Da` continuation hold `α=0.05` fixed and fold almost immediately.
- **The fold recedes with mesh:** α-fold `≈0.24 (N=10) → 0.16 (N=40) → 0.106 (N=80) → …`; solutions
  above the fold are clean (machine-zero residual).
- **Honest reporting (now fixed in core, commit `5ae4c25`).** The production solver used to report a
  noise-floor *stall* (`‖R‖≈1e-5`, ~10²–10³× above `ftol`, at a ~25× wrong solution) as
  *converged*. The new `noise_floor_success_max_ftol_multiple` gate (default `Inf`=off; sweep uses
  `10`) makes the sweep tell the truth about the corner. **This is why the convergence flag can now
  be trusted** — the foundation for the whole detect/rescue workflow.

---

## 4. The sub-optimal-rate question — your hypothesis, assessed in detail

> *"Some cases do converge but with sub-optimal h-convergence — I suspect under-resolved gradients,
> so perhaps for even finer mesh the optimality is recovered."*

**Assessment: this is correct for the manufactured case, and we already have a direct proof at the
worst cell.** The reasoning:

**Three candidate mechanisms for a sub-optimal velocity-`L²` slope, and which apply here:**

1. **Pre-asymptotic erosion (your hypothesis) — APPLIES.** The steep `α`-layer (and high-`Re`
   convective layers) are under-resolved on coarse meshes; the consistency error is layer-dominated
   until the layer is resolved, depressing the apparent slope; the asymptotic `O(h^{k+1})` is
   recovered as `h→0`. The cells-across-layer table (§2) is the quantitative basis; the C24
   measurement is the confirmation.
2. **Boundary-condition `L²` ceiling (Aubin–Nitsche failure) — DOES NOT APPLY here.** The extra `L²`
   order needs elliptic-duality regularity, which a **Neumann** outlet breaks. The manufactured test
   is **Dirichlet everywhere**, so duality holds and the ceiling does not bite. *(This mechanism is
   real — it is exactly what caps the Cocquet benchmark's velocity-`L²` at `O(h²)`, §6 — but it is a
   property of the benchmark's BCs, not of this method.)*
3. **Genuine stabilization order reduction — NOT observed; this is the thing to rule out.** A `τ`
   mis-balance in a reaction/convection-dominated limit would show as a slope that **plateaus** below
   `k+1` across ≥2 fine pairs *after* the layer is resolved. We have not seen this; the test below
   would detect it if present.

**The decisive test — the rate _trajectory_, not a single slope:** measure consecutive-pair slopes
up a fine-mesh ladder. **Climbing toward `k+1`** ⇒ pre-asymptotic (mechanism 1, your hypothesis).
**Plateau below `k+1` after the layer is resolved** ⇒ genuine reduction (mechanism 3) — a real
finding worth a footnote in the paper.

**Direct evidence we already have (manufactured, Dirichlet, the worst corner):** C24 continued to
the target `α₀=0.05`:

| pair | rate `L²` u | rate `H¹` u |
|---|---|---|
| 512 → 768 | **3.03** | **1.04** |
| 768 → 1024 | **2.99** | **1.01** |

`H¹` is dead-on the optimal `k=1` rate; `L²` is *super-convergent* (≈3, above the nominal 2),
holding steady — the opposite of a sub-optimal plateau. The earlier coarse-mesh slopes for this same
cell were depressed; they climbed to this once the layer was resolved. **That is your hypothesis,
demonstrated.**

**External supporting evidence (label: not a premise).** A Dirichlet exact-solution MMS with the
related Forchheimer reaction (CocquetFormMMS, §6 of the Cocquet doc) shows the same signature —
velocity-`L²` slopes `2.39→2.77→2.86` climbing toward 3, `H¹→1.92` optimal. Consistent with
mechanism 1; it is *additional* evidence, and the standard constant-`σ` grid is what makes it
conclusive for *this* test.

**My expectation for the full run:** essentially every cell that admits a root converges optimally
(or super-convergently in `L²`) once it is past the fold and the `α`-layer is resolved. I expect the
`‡` "genuine sub-optimal rate" mark (see §7 workflow) to be **rare or empty** for the manufactured
case. If any cell *does* plateau, that is a clean, defensible scientific result — and the workflow
surfaces it distinctly rather than hiding it.

---

## 5. What we KNOW (established)

- **The measurement is exact.** Cross-mesh / exact-field `L²` error returns slope `3.00–3.01` on
  smooth fields (verified in `quick/interpolation_projection_quick_test.jl` and the Phase-1 coherency
  check). Sub-optimal slopes are never a metric artifact.
- **Non-extreme grid → fully optimal**, from coarse meshes. (User-confirmed; consistent with the
  smoke: `Re=1e6,α=0.5` and `Re=1,α=0.05` both verify optimal.)
- **The extreme corner folds at coarse mesh** — a genuine discrete turning point, recedes with
  refinement, independent of `Da`. Not a solver bug.
- **Past the fold, the corner converges optimally / super-convergently** (C24: `H¹≈1.0`, `L²≈3.0`).
- **The solver now reports the corner honestly** (noise-floor false-success closed, commit
  `5ae4c25`); the sweep's convergence flag is trustworthy.
- **`Da` is benign** (linear knob on a coercive constant reaction); the difficulty axis is `Re×α₀`.
- **Energy-norm (`H¹`) optimality is robust** — it does not rely on the duality that the `L²` extra
  order needs, so it holds even where `L²` is pre-asymptotically eroded.

## 6. The Cocquet case — why it is *separate* (and not a premise here)

Kept only to prevent conflation. The Cocquet benchmark is **not extreme** (`Re=500`), but it is
**complicated** by two things the manufactured test deliberately avoids: (i) a **traction-free
Neumann outlet** + wall↔outlet corner, which genuinely caps velocity-`L²` at `O(h²)` by weakening
Aubin–Nitsche duality (matching Cocquet et al.'s own reported `O(h²)`); and (ii) a historical
**mesh-dependent velocity floor** in the reaction (`u_floor ∝ ν/h`, grew under refinement, corrupted
the reference) — since fixed by separating `reaction_speed` from `effective_speed`. Neither applies
to the manufactured test (Dirichlet BCs; `h_floor_weight=0`; constant `σ`). The manufactured case
will *inform* the Cocquet interpretation, per the project decision — not the reverse.

## 7. What we DON'T know yet (open)

1. **~~Does optimal recovery hold for every extreme cell?~~ RESOLVED (k=1 QUAD).** The full N=10→640
   k=1 QUAD sweep (2026-06-10, success box above) confirms recovery is the rule across the *whole*
   grid: every cell's velocity reaches optimal L²/H¹ at the finest pair, including the once-scoped
   **high-Da reaction-dominated corner** (`Da=1e6`, `Re∈{1e-6,1}`) — its H¹ rate climbs
   `0.57→…→0.73→1.11→1.85` and recovers by N=640 (pre-asymptotic, as hypothesised; not order
   reduction). Still merely *unverified* (not problematic): `k=2` and `TRI`.
2. **The exact A/B/C boundary** in `(Re,Da,α₀)` — which cells fold, which merely erode, which are
   clean — is not yet mapped. The sweep maps it.
3. **OSGS-specific pressure rate.** A documented OSGS sub-optimality from **constant-mode noise in
   the projection increment `d_π^m`** (`docs/solver/paper-code-divergences.md`) can depress the *pressure*
   rate independently of the velocity story above — watch OSGS pressure slopes specifically.
4. **~~Whether any genuine `τ`-induced order reduction exists at all~~ RESOLVED: no.** The high-Da OSGS
   reaction corner (`Da=1e6`, `Re∈{1e-6,1}`) showed a **pre-asymptotic coercivity gap**, not a `τ`
   artifact — and the N=640 pair (above) settles the asymptotic-vs-pre-asymptotic question decisively:
   the H¹ rate **climbs to ≥1.0**, i.e. a slow pre-asymptotic climb that recovers, *not* a fixed order
   reduction. The mechanism is the OSGS coercivity *constant* (σ_a) of
   [`../solver/osgs-reaction-dominated-rate.md`](../solver/osgs-reaction-dominated-rate.md), which
   degrades the bound but not the rate.

---

## 8. Sweep status — DONE (see route-b-2d-sweep-status.md)

**The k1 and k2 QUAD sweeps completed 2026-07-03** under the Route-B algebraic mass gate; the current
per-cell status, rates, and fold/rescue outcomes live in
[`route-b-2d-sweep-status.md`](route-b-2d-sweep-status.md). The earlier "not yet launched / user-
triggered / multi-day" framing is retired.

**Analysis workflow (still current).** All Python analysis is the single tool **`analyze_results.py`**
(it replaced `plot_results.py` + `merged_report.py` + `detect_flagged_cells.py` +
`mms_convergence_lib.py`): one run does true-root detection (→ `flagged_cells.json`), the annotated
merged paper table, the per-config plots, and a detailed per-config table with an honest true-root
`Converged` column. Cell categories (`fold` / `suboptimal_rate` / `total_failure`) route Phase-2
rescue: **fold** → α-continuation past the fold + mesh ladder `[320,512,768,1024(→1536)]` (the finest
2 consecutive true-root slope, how C24 was measured); **suboptimal_rate** → direct base solve + fine
ladder (`subopt_base_candidates` / `subopt_fine_ladder`), reporting the per-pair trajectory (the §4
decisive test). Report marks: verified / `*` recovered-at-fine / `‡` genuine sub-optimal / `ˢ`
super-convergent / `**` fold-best-root / `N/A`. One-sided slope acceptance (`slope ≥ target − tol`)
keeps super-convergence from being mislabelled sub-optimal.
