# Manufactured-Solution (MMS) convergence — status & knowledge

**Scope.** This document is about the *standard* manufactured-solution test in
`test/extended/ManufacturedSolutions` — the `(Re, Da, α₀, h)` convergence sweep that is the
codebase's primary correctness criterion. It is written to stand **on its own**: the Cocquet
benchmark (`docs/cocquet_convergence_analysis.md`) is a separate, more complicated case (mixed
inlet/wall/**traction-free Neumann** boundary conditions, and a now-fixed mesh-dependent reaction
floor) and is **not** extreme; it will be re-interpreted in light of the manufactured case, not the
other way around. Where a Cocquet/CocquetFormMMS result is used below it is clearly labelled as
**external supporting evidence**, never as a premise.

Last updated: 2026-05-22.

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
   The fold **recedes with refinement**; continuation reaches the root at fine mesh.
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
  **bit-identical** corner roots — `src/.../diagnostics/c24_resolution_and_continuation.md`).
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

---

## 3. The fold (region B) — established

Full diagnosis: [`diagnostics/c24_resolution_and_continuation.md`](../test/extended/ManufacturedSolutions/diagnostics/c24_resolution_and_continuation.md). Key points:

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

1. **Does optimal recovery hold for every extreme cell?** We have proven it for the `Re=1e6,α=0.05`
   corner at `k=1,QUAD` (C24/C21). **Unverified:** `k=2` (steeper finest-mesh cost, higher target
   rates); `TRI` (simplex layer resolution + cross-mesh interpolation robustness); the full `Da`
   range at the corner. → the full sweep + fine ladder per cell.
2. **The exact A/B/C boundary** in `(Re,Da,α₀)` — which cells fold, which merely erode, which are
   clean — is not yet mapped. The sweep maps it.
3. **OSGS-specific pressure rate.** A documented OSGS sub-optimality from **constant-mode noise in
   the projection increment `d_π^m`** (`theory/paper-code-divergences.md`) can depress the *pressure*
   rate independently of the velocity story above — watch OSGS pressure slopes specifically.
4. **Whether any genuine `τ`-induced order reduction exists at all** for this method — expected *no*
   based on C24, but only the multi-pair trajectory across the grid can assert it.

---

## 8. Launch plan — per-region handling + documentation workflow

The two-phase pipeline (committed on `feat/mms-honest-exit-two-phase`) already routes each region to
its proper treatment; the steps below note the **small refinements** the C24 evidence shows are
needed before launch.

All Python analysis is the single tool **`analyze_results.py`** (it replaced `plot_results.py` +
`merged_report.py` + `detect_flagged_cells.py` + `mms_convergence_lib.py`): one run does true-root
detection (→ `flagged_cells.json`), the annotated merged paper table, the per-config plots, and a
detailed per-config table with an honest true-root `Converged` column.

**Per batch** (`phase1_{quad,tri}_k{1,2}.json`; background, resumable, separate `h5`):

1. **Phase-1 blanket sweep to `N=320`** (`run_test.jl`), fail-fast (`max_n_pert=1`,
   `linesearch_alpha_min=1e-4`, `osgs_iterations=5`), honest-exit on (`k_nf=10`). → Region A
   verifies here; region B fails honestly; region C converges with an eroded slope. True per-mesh `‖R‖`.
2. **Analyze → detect & categorize** (`analyze_results.py`): writes `flagged_cells.json`
   (`fold` / `suboptimal_rate` / `total_failure`) for Phase-2, plus a first table + plots.
3. **Phase-2 rescue, by category** (`run_continuation.jl phase2`):
   - **fold** → α-continuation past the fold + mesh ladder `[320,512,768,1024(→1536)]`; report the
     slope over the finest 2 consecutive true roots (this is how C24 was measured).
   - **suboptimal_rate** → the root already exists, so Phase-2 **solves directly at the base mesh
     (no α-ramp) and mesh-continues up the fine ladder** (`subopt_base_candidates`,
     `subopt_fine_ladder`), reporting the **per-pair trajectory** (climb vs plateau = the §4
     decisive test). *(Routing is by detected `category`, hardest across the two methods.)*
4. **Re-analyze with the rescue** (`analyze_results.py --phase2 …`): the same tool joins the Phase-2
   results into the paper table — error at `N=320` for verified cells, finest **true-root** error
   (asterisked, mesh noted) otherwise. Marks: verified / `*` recovered-at-fine / `‡` genuine
   sub-optimal / `ˢ` super-convergent (confirm asymptotics) / `**` fold-best-root / `N/A`. Refreshes
   plots + the detailed honest-`Converged` table too.
5. **Document:** append a dated "Run findings" section *to this file* — which cells fell in A/B/C,
   the region-C rate trajectories (the answer to §4/§7-1), the fold boundary, anything surprising.

**Pre-launch refinements (DONE — small, justified by the C24 data):**

- **One-sided slope acceptance + super-convergence caveat.** C24's `L²` rate is `≈3.0`
  (super-convergent, `k=1`); the old two-sided check would have mislabelled it sub-optimal.
  `analyze_results.py` accepts one-sided (`slope ≥ target − tol`),
  so super-convergence is **not** a failure. *But* a rate above nominal is marked `ˢ` in the report
  and counted separately — a high rate can itself signal the **asymptotic regime is not yet
  established**, so it is surfaced for confirmation (one further refinement) rather than silently
  passed. (The true-root gate already guards the machine-floor case.)
- **Category-aware Phase-2 routing** — done (§8.3): `suboptimal_rate` cells take the cheap direct
  base solve; folds keep the α-continuation root search.

*Validation status:* refinement 1 unit-tested + re-run on the smoke (clean); refinement 2
syntax-checked and defaults wired. End-to-end Phase-2 dry-run re-validation of the new direct-base
path is **deferred to launch time** so it does not compete with the Cocquet runs in flight.

**Order & cost.** `k=1 QUAD` first (cheapest finest mesh, contains the corner — doubles as timing
calibration), then `k=1 TRI`, then the two `k=2` batches. The corner cells (region B) are the
expensive tail; everything else is fast. The full grid is plausibly multi-day — launch is
**user-triggered** (run on AC power; resumable so a power blip loses nothing).
