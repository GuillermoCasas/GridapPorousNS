# CocquetFormMMS — status & findings

**Scope.** This is the **canonical** doc for `test/extended/CocquetFormMMS`, the
manufactured-solution (MMS) sibling of the Cocquet benchmark. It compares, under one MMS
convergence sweep, the **equal-order stabilized VMS** method (P1/P1 & P2/P2, ASGS + OSGS) against the
unstabilized **Galerkin Taylor–Hood P2/P1** ("Cocquet element"), using the **full Forchheimer–Ergun
reaction** `σ = a(α)+b(α)|u|`. The driving question: *how does the equal-order stabilized method hold
up against the inf-sup-stable Cocquet element across the porosity/Reynolds space?*

Related: the standard MMS sweep is `docs/mms/convergence-2d.md` (a *different*, constant-σ
harness); the physical Cocquet benchmark is `docs/cocquet/investigation-synthesis.md`. The
τ-saturation mechanism referenced below is written up in
`theory/tau_saturation_note/tau_saturation_note.tex`.

**Last updated: 2026-07-07.** Moderate-porosity results are complete and clean. The low-porosity
high-Re corner (α=0.1 × Re=1e5) is **RESOLVED for k=1 (2026-07-07): it is a genuine coarse-mesh
solution-branch fold — no discrete root exists for N≤80, but a TRUE root appears at N≥160 and is
FE-OPTIMAL there.** Extending the mesh ladder past the main sweep's N=160 cap to N=320 (config
`data/cocquet_form_mms_vms_corner.json`, its own DB `cocquet_form_mms_vms_corner.h5`) gives two
converged corner meshes with **optimal rate — H¹u ≈ 1.07 (ASGS) / 1.10 (OSGS), L²u ≈ 3.0 both**
(see §4.1). This is the same fold-recedes-with-mesh phenomenon the sister `ManufacturedSolutions`
harness diagnosed decisively for its α=0.05 corner (`docs/mms/convergence-2d.md`), and it matches that
harness's corner rates (H¹≈1.0, L²≈3.0) — so the fold is not a stabilization defect but the paper's
intrinsic 1/α₀ degradation pushing the coarse-mesh branch past a turning point. The **k=2 corner**
already has clean roots at N=40 & N=80 (it clears the fold ~2× earlier); extending it to N=160 to
firm the rate is a cheap remaining follow-up.

---

## 1. Setup

| Factor | Values |
|---|---|
| Min porosity `α₀` | **0.5, 0.1** (the spatially-varying `SmoothRadialPorosity` core value; `α∞=1`) |
| Reynolds `Re` | **1, 1e5** |
| Damköhler `Da` | held at **1** — under the fixed Forchheimer coefficients `Da` is a no-op (only the `Constant_Sigma` branch reads it). **`α₀` is the effective-Da knob**: the collective Darcy+Forchheimer drag gives `Da_eff(α) = σ_lin·((1−α)/α)² + σ_nl·((1−α)/α)` (≈40 at α=0.1, ≈2 at α=0.5). |
| Element / method | **k=1 & k=2 equal-order** (ASGS + OSGS) + **Taylor–Hood P2/P1** (Galerkin) |
| Meshes (TRI) | 10, 20, 40, 80, 160 |
| Reaction | `Forchheimer_Ergun`, `σ_lin=0.3`, `σ_nl=1.75` (the Cocquet-et-al. coefficients) |
| Gate | `eps_tol_momentum = 1e-9` (tight; required to recover the k=2 finest-segment rate — a fixed 1e-6 stops k=2 solves early; don't over-tighten, 1e-12 → NaN) |

Configs: `data/cocquet_form_mms_vms.json` (VMS k=1,2 ASGS+OSGS) and
`data/cocquet_form_mms_taylorhood.json` (the Cocquet element). Plot:
`plot_combined_all.py` (k=1 ○, k=2 □, TH ✕; ASGS blue/solid, OSGS red/dashed, TH black/dotted;
velocity filled, pressure hollow; L² and H¹ split) → `results/combined/`.

### Harness design (all fixes are harness-level — `src/`, `config/`, schema untouched)
- **(L,U) minmax encoding** (`compute_L_and_U`, strategy `forchheimer_minmax`): rescales the cell to
  keep the `{U,ν,σ}` dynamic range tame at high Re / low α (mesh, porosity field, perturbation and
  error norms all L-scaled; `physical_epsilon` made encoding-covariant). `"unit"` reproduces the legacy L=U=1.
- **Encoding-covariant reaction scaling** (`build_mms_formulation`): `a(α)=σ_lin·((1−α)/α)²·(ν/L²)`,
  `b(α)=σ_nl·((1−α)/α)·(ν/(U·L²))`, so `Da_eff` is **Re-independent** (matches `article.tex`
  `eq:DimensionlessParameters`). The old code passed `a,b` fixed ⇒ `Da_eff ∝ Re`.
- **Graceful fold recording**: on tight-gate failure, retry once with a relaxed gate
  (`eps_tol_momentum_fold`, default 1e-2) and record the achievable-floor solution flagged `fold`
  instead of discarding it as NaN. Converging cells never reach the retry.
- **ASGS≠OSGS fix**: the per-cell config previously hard-coded `method="ASGS"`, so OSGS silently ran
  ASGS (byte-identical). Now each method gets its own config; OSGS genuinely differs.

---

## 2. Results (2026-06-16)

| Cell | k=1 (ASGS/OSGS) | k=2 (ASGS/OSGS) | Taylor–Hood |
|---|---|---|---|
| **α=0.5, Re=1** | ✅ 5/5 both | ✅ 5/5 both | ✅ 5/5 |
| **α=0.5, Re=1e5** | ✅ 5/5 both | ✅ 5/5 both | ✅ 5/5 |
| **α=0.1, Re=1** | ✅ 5/5 both | ASGS 4/5, OSGS 3/4 | ✅ 5/5 |
| **α=0.1, Re=1e5** | fold at N≤80, ✅ **optimal at N≥160** (§4.1) | ASGS root at N=40,80; OSGS at N=80 | ✅ 5/5 |

- **Moderate porosity (α=0.5): the definitive result.** Clean, complete convergence for *every*
  method (k=1, k=2, ASGS, OSGS, Taylor–Hood) at both Re. Velocity optimal (L² rate ≈2.2 for k=1),
  pressure converges. ASGS and OSGS are genuinely distinct (e.g. α=0.5/Re=1e5/N=80: ASGS L²u≈3.2e-4
  vs OSGS≈4.2e-4).
- **α=0.1, Re=1:** stabilized methods converge.
- **α=0.1, Re=1e5 — the hard corner:** on the main sweep's mesh range (N=10…160) the equal-order
  **stabilized methods fold on the coarse meshes** (gate unreachable; solution recorded at the
  achievable floor), while the **unstabilized Taylor–Hood converges everywhere**. This VMS-folds /
  TH-converges contrast at the high-Re × low-α corner is the headline phenomenon on *coarse* meshes.
  (This is also the corner the *paper itself skips* in its own sweeps via `skip_cells`.) **But the
  fold is a coarse-mesh turning point, not a stabilization defect: a true, FE-optimal discrete root
  exists at N≥160 — see §4.1 (RESOLVED 2026-07-07).**

---

## 3. Why the regular MMS harness doesn't hit this (it's the corner, not the solver)

The standard `ManufacturedSolutions` harness reaches `Re=1e6` cleanly, yet CocquetFormMMS folds at
`Re=1e5`. After the encoding port both harnesses share the same `(L,U)` conditioning fix and the same
gate, so the difference is the *problem*, not the solver:

1. **The corner is excluded there.** The regular sweep's `skip_cells` removes `(Re=1e6 × α=0.05)`;
   its high-Re cells run only at `α∈{0.5,1.0}`. It never certifies high-Re × low-α — that's deferred
   to its Phase-2 continuation harness. CocquetFormMMS probes the corner head-on.
2. **Nonlinear, heterogeneous Forchheimer reaction** vs the regular harness's **constant σ** (which
   OSGS even trims out of the projection). At α=0.1 the drag coefficients are 81×/9× larger.
3. **Tighter gate** (1e-9 vs the regular `test_config` default 1e-6), which exposes residual floors.

---

## 4. Why low-α folds — investigation status

### 4.1 RESOLVED (2026-07-07, k=1): a coarse-mesh solution-branch fold; the corner is FE-optimal above it

> **Provenance note (2026-07-08 cleanup) — applies to ALL of §4.** Every diagnostic *side-run* named
> throughout §4 — the α-sweep / linear-reaction control (`isolation_alphasweep.json`,
> `isolation_linctrl.json`), the N=320 corner extension (`*_corner`), the reaction-out-of-stabilization
> and c₁ probes (`*_strip_*`, `*_c1x64*`) — was a **debug config/side-DB authored during the
> investigation and has been REMOVED** per the `.agents/rules/official-results-path.md` rule (a test
> keeps only its designed-mode configs; finished-debug scrap is deleted once documented). The harness's
> designed modes remain: `cocquet_form_mms_{vms,vms_k2,taylorhood}.json`. **The findings below stand**
> as documented — the numbers were read off those runs, and results embed their config, so any removed
> run is reconstructable from its (archived) result. Config/DB paths named in §4 are a *record of what
> was run*, not live files. To reproduce the corner rates as an *official* result, extend the designed
> `data/cocquet_form_mms_vms.json` mesh ladder (`convergence_partitions` → N=320) and re-run through the
> harness, archiving the prior official DB into `previous_results/` first.

The α=0.1 × Re=1e5 "failure" is a **genuine coarse-mesh turning-point fold of the discrete solution
branch** — on coarse meshes there is *no* root with ‖R‖≤tol to converge to — that **recedes with mesh
refinement**. It is not a solver bug and not a stabilization defect. Evidence, all from the committed
DBs:

- **The fold recedes as N↑ and deepens as α↓** (α-sweep at Re=1e5, `results/isolation_alphasweep.h5`,
  and the main sweep `cocquet_form_mms_vms.h5`): α=0.9/0.5 converge at every mesh; **α=0.2 folds at
  N≤20 but converges at N=40**; **α=0.1 folds at N≤80 but has a TRUE root at N=160** (main sweep,
  both methods: ASGS ‖R‖=1.3e-7 L²u=1.84e-3 in 7 it; OSGS ‖R‖=1.0e-6 L²u=2.09e-3 in 48 it).
- **It is not a basin/initial-guess problem.** The harness *already* initializes each cell from the
  exact-solution interpolant (`run_test.jl` `x0_exact`) plus a perturbation-homotopy, and still folds
  at N≤80 — so a better globalizer cannot manufacture a root that does not exist. This mirrors the
  sister harness's decisive A1/A2 tests (`docs/mms/convergence-2d.md`): exact Jacobian (4.8e-12),
  heavy Newton **and** Picard from `u_ex` both stall at ‖R‖≈5e-2 ⇒ no root near the exact solution.

**Recovery (the fix): extend the mesh ladder above the fold.** Config
`data/cocquet_form_mms_vms_corner.json` (α=0.1, Re=1e5, k=1, ASGS+OSGS, N=[160,320], separate DB
`cocquet_form_mms_vms_corner.h5`, reusing the existing exact-guess init — no `src/` change) gives two
converged corner meshes. Both N=320 cells reach genuine roots (‖R‖~1e-7, 6/30 iters — a clean solve,
not a struggling fold):

| method | N=160 (‖R‖) | N=320 (‖R‖) | rate L²u | rate H¹u | rate L²p |
|---|---|---|---|---|---|
| ASGS | L²u=1.84e-3 (1.3e-7) | L²u=2.28e-4 (1.7e-7) | **3.01** | **1.07** | 3.03 |
| OSGS | L²u=2.09e-3 (1.5e-6) | L²u=2.44e-4 (5.6e-7) | **3.10** | **1.10** | 3.12 |

H¹u ≈ 1.07/1.10 is **textbook-optimal O(h) for k=1**; L²u ≈ 3.0 is super-optimal for this smooth MMS —
**identical to the sister harness's α=0.05 corner** (H¹≈1.0, L²≈3.0, `docs/mms/convergence-2d.md`),
which cross-validates the 2-point slope. So the equal-order stabilized method converges optimally at
α=0.1 × Re=1e5 once the mesh clears the fold — matching the clean α=0.5 deliverable. The
VMS-folds-at-coarse-mesh / TH-converges contrast stays a true finding; it is a coarse-mesh basin of
the *nonlinear discrete map*, not a loss of the FE convergence order.

**Reconciles with the paper:** the paper proves the *solution* stays optimal in the triple norm and
handles this exact corner by skipping it in the coarse sweep + using continuation — precisely what the
extended ladder does here. The reaction-magnitude driver below (Layer 2, the paper's 1/α₀ degradation)
is *why* the coarse-mesh branch folds; it does not degrade the rate once a root exists.

**Remaining follow-ups (cheap):** (i) k=2 corner to N=160 to firm its rate (it already has clean roots
at N=40 & N=80); (ii) optional N=640 for a 3-point k=1 slope; (iii) a mesh-continuation warm-start
(interpolate the N=160 root onto N=320) is *not* needed here — the cold exact-guess reached N=320
directly — but would add robustness if the ladder is pushed to where the exact-guess basin narrows.

### 4.2 Why the coarse-mesh branch folds (the mechanism — still the reaction magnitude)

**Established empirically.**
- The fold is **reaction-magnitude driven**, not nonlinearity-driven. An α-sweep at fixed Re=1e5
  (`isolation_alphasweep.json`) converges at α=0.9 & 0.5 and folds at α=0.2 & 0.1; a **linear-reaction
  control** at α=0.1 with matched `Da_eff` (`σ_nl=0`, `isolation_linctrl.json`) folds *identically* —
  so the `b(α)|u|` Forchheimer nonlinearity (`∂σ/∂u` basin-shrink) is **not** the cause.
- The fold cells are genuinely **pre-asymptotic** on coarse meshes (err_u ~0.12–0.20, ~50–100× the
  converged cells), not "optimal-but-gate-floored".

**Two candidate layers (only Layer 1 is fully written up; the rest is a hypothesis).**
- **Layer 1 — high-Re τ₁-saturation residual floor** (`theory/tau_saturation_note`): generic to
  high-Re stabilized Navier–Stokes (survives σ=0); the assembled stabilized residual is the
  near-cancellation of two saturated terms, so a tight *absolute* gate becomes unreachable. **Dormant
  at Re=1e5** (the α=0.9 near-NS cell converges there); it's a `Re≳1e6` effect. So it does **not**
  explain the Re=1e5 fold.
- **Layer 2 — low-porosity reaction** is the operative one at Re=1e5, but the *exact* mechanism is
  **NOT yet confirmed**. The leading candidate is the paper's **`σ̃_α` coercivity weakening**: in the
  *ASGS* stability estimate (`article.tex` §`sec:StabilityASGS`, `eq:SigmaAlpha`) the reaction's
  presence in the stabilization replaces the full velocity control `σ` by the weaker
  `σ̃_α = τ_{1,NS}⁻¹σ/(τ_{1,NS}⁻¹+σ/α_K)`, which collapses as `α→0`; Taylor–Hood, having no
  stabilization, keeps the full `σ`. **Caveat:** the paper deems `σ̃_α` *benign for the convergence
  rate* — the link "weaker coercivity ⇒ nonlinear-solver fold" is an extension we have **not** proven.

### ⚠️ ASGS vs OSGS — do not conflate (corrected 2026-06-16)
- `σ̃_α` is the **ASGS** estimate (the paper's stability section is explicitly "the ASGS method").
- **OSGS** handles the reaction differently: it removes the reaction from the orthogonal projection
  (for constant σ its orthogonal subscale is exactly zero — `article.tex` ~line 619), and the paper
  states this removal *"facilitates the convergence of the nonlinear iterations."* So OSGS has a
  **documented** reaction/convergence concern that ASGS does not phrase the same way. It is **not**
  "only OSGS" and **not** "all VMS uniformly" — the two methods carry the reaction differently.
- In code the reaction trim `ProjectResidualWithoutReactionWhenConstantSigma` is **OSGS-only**
  ([`src/stabilization/projection.jl:76-80`](../../src/stabilization/projection.jl#L76):
  `if is_osgs: R_u−σu−π; else return R_u`), and only fires for `Constant_Sigma`. With the
  **Forchheimer** law (what the sweep uses) the reaction stays in the stabilization for **both**
  methods — consistent with both folding at low α.

### Experiments attempted (and their verdicts)
- **ASGS trim-vs-full A/B — INVALID.** Tried to A/B "reaction in/out of the ASGS stabilization", but
  the trim is OSGS-only, so the two ASGS runs were byte-identical (both fold). It isolated nothing.
- **OSGS trim-vs-full A/B — inconclusive.** The valid version (OSGS, where the toggle is real) was
  started but **killed before completing** (OSGS low-α fold cells thrash for a long time).
- **c₁×4 (raised coercivity constant) — PARTIAL help, NOT a fix (2026-07-05).** Motivated by the 3D-P2
  result, where **c₁×4 gives a fully optimal convergent sweep in Gridap** (note: c₁ there is now understood to
  *mask* a Gridap↔paper discrepancy, not fix a coercivity deficit — paper `c₁=4k⁴` is correct per the first
  author / Kratos full-terms; see [`../mms/p2-3d.md`](../mms/p2-3d.md)).
  Ran `isolation_alphasweep.json` (Re=1e5, k=1, ASGS) at **c₁×1 vs c₁×4** (via a new
  `C1_MULT` env-var hook at the `get_c1_c2` site in `run_test.jl` — default `1.0` = byte-identical):
  - **α=0.1** (folds at *all* N at paper c₁ — N=10/20/40 → NaN): c₁×4 converges **only N=10** (L²u≈0.428,
    large) and **still folds at N=20 and N=40** — erratic (coarse converges, fine folds), i.e. **not a
    convergent sweep**.
  - **α=0.2** (solves only at N=40 at paper c₁): c₁×4 keeps the same convergence pattern but the N=40 error
    is **~5× smaller** (L²u 0.0736 → 0.0151).
  So raising c₁ helps *at the margin* (one extra cell converges; better accuracy where it already did) but
  **does not rescue the fold**. This is UNLIKE 3D-P2, and expected: this probe is **k=1**, where the
  viscous 2nd-derivative subscale (the term c₁ fixed for P2 tets) is *identically zero*, so c₁ acts only
  through `τ_NS`. Still, the partial help is mild evidence that the fold has a **coercivity/stabilization
  component** (consistent with the `σ̃_α` Layer-2 hypothesis), not purely reaction-magnitude.
  **⏭ Open next step (deferred to a future session): the k=2 analog** (`cocquet_form_mms_vms_k2.json`),
  where the viscous 2nd-derivative subscale exists — the faithful test of whether the 3D-P2 c₁ mechanism
  transfers. The `C1_MULT` hook is committed (default-off, byte-identical) so that run is ready to go.

**Bottom line (superseded by §4.3):** the *exact* cause of the equal-order stabilized low-α fold is
**open**. The `σ̃_α`/reaction-in-stabilization mechanism is the leading hypothesis but is unconfirmed;
`theory/tau_saturation_note` deliberately does not assert it. The 2026-07-08 direct tests below
sharpen this (and correct one premise): the premise "Taylor–Hood avoids the fold" is itself **false**
at the corner — see §4.3.

### 4.3 Mechanism investigation (2026-07-08) — what was ruled out, and what stays open

A focused round of direct tests. Net result: several candidates **eliminated**, the σ̃_α hypothesis
**not confirmed**, and the exact mechanism **still open** — but the *practical* result (§4.1) is
unaffected and secure.

- **Newton is EXACT for the Cocquet formulation — the fold is NOT a linearization bug.** A dedicated
  probe (now the permanent test `test/extended/cocquet_jacobian_consistency_extended_test.jl`) assembles
  the Exact-Newton Jacobian for **SymmetricGradient + Forchheimer** and compares it to a centered
  finite-difference of the residual: **‖J−J_fd‖/‖J_fd‖ ≈ 1e-8..1e-11** for k=1 *and* k=2, at
  corner-like (small ν, large drag) parameters, and Newton converges **quadratically** (‖R‖:
  2.9e-4→1.3e-5→2.7e-8→2.3e-13). This closed a real coverage gap: `picard_jacobian_equivalence` only
  checks *Picard*-mode equivalence, and `osgs_frozen_pi_jacobian` uses ConstantSigma — the
  velocity-dependent `∂σ/∂u` Exact-Newton tangent for the Cocquet combo was previously unguarded. ⇒ the
  fold is a genuine property of the (correctly-linearized) nonlinear problem, not a solver/Jacobian defect.

- **The "Taylor–Hood converges everywhere" claim is FALSE at the corner (correction).** The committed
  `cocquet_form_mms_taylorhood.h5` shows TH (Galerkin P2/P1) at α=0.1×Re=1e5 does **not** reach a root:
  its nonlinear residual **stalls at O(1)** (‖R‖ = 650→190→50→13→**3.1** at N=10→160, vs ~1e-9 when it
  works) and its velocity error is **flat at 0.40** (rate 0). TH is *convectively unstable* here (P2/P1
  has inf-sup for pressure but no SUPG for convection) — its **pressure** converges optimally (L²p rate
  2.0, via LBB) while its **velocity** is garbage. So the real contrast is not "TH solves / VMS folds":
  it is **VMS folds *hard* at coarse mesh (NaN) but converges to an accurate root at N≥160, where TH
  still cannot** (res 3.1, L²u 0.40). VMS is the *better* method at the corner. (At Re=1, TH converges
  cleanly — rate 2.94 — confirming the corner failure is the high-Re convective instability.)

- **Reaction-out-of-stabilization A/B — INCONCLUSIVE (σ̃_α not confirmed).** Via a temporary gated
  diagnostic — `STRIP_REACTION_FROM_STAB`, since **REVERTED** (not paper-faithful, so it does not stay
  in the core), but fully described here for reproducibility: an env-var in the coefficient/residual/
  Jacobian builders of `src/formulations/continuous_problem.jl`, default off = byte-identical, Blitz
  243/243 incl. exact `picard_jacobian_equivalence`; the *stripped* formulation was itself
  self-consistent (J-vs-FD ~1e-11, quadratic) — that removes σ from the
  stabilization (τ₁, 𝓛U, 𝓛*V, and their derivatives) while the coercive Galerkin term `(v,σu)` keeps
  the real σ — i.e. TH-like full-σ velocity control but with VMS's convective stabilization. Ran the
  corner A/B (`data/cocquet_form_mms_strip_reaction_test.json`, N=[20,40,80,160], ASGS): stripping
  cleared **only 1 of 3** folding meshes (N=40, and to an *inaccurate* root L²u=0.092 ~3× the accurate
  value), while **N=20 and N=80 still fail even from the exact guess**. Not the clean "strip ⇒ no fold"
  a confirmation needs. **Confound (flagged honestly):** the strip also *enlarges* τ₁ (removes σ from
  its denominator, `1/(α/τ_NS+σ)→τ_NS/α`), and σ is genuinely *entangled* in the stabilization scale
  (σ̃_α itself contains τ₁), so "still folds" cannot be cleanly attributed to σ̃_α-persistence vs the
  τ₁ over-strip. So this **neither confirms nor cleanly refutes** σ̃_α — but it does show the fold is
  **not reducible to a single removable term**; the reaction-in-stab is at most a partial contributor.

- **c₁ (coercivity constant) — not the lever** (from §4.2's c₁×4 probe + theory). At high Re, τ_NS⁻¹ is
  convection-dominated (`c₂|u|/h ≫ c₁ν/h²`), so raising c₁ barely moves the stabilization; c₁×4 gave
  only marginal help. A c₁×64 confirmation was launched but killed under CPU contention before finishing
  (superseded by the strip test, which targets the reaction directly).

**Updated bottom line:** the exact fold mechanism is **OPEN**. Ruled out: a Jacobian/linearization bug
(Newton is exact) and c₁. The leading σ̃_α / reaction-in-stabilization hypothesis is *paper-grounded and
consistent with every observation* (reaction-magnitude driven, low-α specific, fold-recedes-with-mesh)
but was **not confirmed** by the direct strip test, which was confounded by the τ₁ entanglement. A
cleaner isolation (strip σ from 𝓛U/𝓛*V only, holding τ₁ physical) is the natural next probe but is
**deferred** — the practical deliverable (§4.1, convergence above the fold) does not depend on it. The
`STRIP_REACTION_FROM_STAB` gate was **reverted** (a stabilization-reaction strip is not paper-faithful
and does not belong in the core); the method above is complete enough to re-derive it if that clean
follow-up is pursued.

---

## 5. Open items / next steps
- **✅ DONE (2026-07-07) — recovering the α=0.1 × Re=1e5 corner (k=1):** extend the mesh ladder above
  the fold (N=[160,320], `data/cocquet_form_mms_vms_corner.json`). Both ASGS & OSGS reach FE-optimal
  roots (H¹u ≈ 1.07/1.10, L²u ≈ 3.0); see §4.1. This is the "recovering the fold cells" item below,
  executed via the direct exact-guess route (mesh-continuation not needed — the cold exact-guess
  reached N=320 directly). Cheap remaining: k=2 corner → N=160; optional k=1 → N=640 for a 3-point slope.
- **Layer-2 mechanism (why the coarse branch folds) is characterized, not a blocker.** §4.1 shows the
  fold is a coarse-mesh turning point driven by the reaction magnitude (the paper's 1/α₀ degradation),
  and that the rate is optimal once a root exists — so confirming the exact σ̃_α coupling is now a
  *theory-completeness* question, not a prerequisite for the deliverable. If still wanted: (a) finish
  the **OSGS** trim-vs-full A/B at low α; (b) for **ASGS**, a code change to strip σu from the
  stabilization residual (the projection trim won't do it for ASGS).
- **Recovering the fold cells (alternative route, not needed here):** **continuation into the corner**
  (start at α=0.5 or Re=1, walk to α=0.1 / Re=1e5), the device the regular harness's Phase-2
  `run_continuation.jl` uses. The direct exact-guess mesh-ladder (§4.1) superseded it for k=1; keep
  continuation in reserve if a future push hits a mesh where the exact-guess basin narrows.
- **Throwaway probes removed (2026-06-16):** the `diagnose_*.jl` probes, `data/_validate_*.json`, the
  superseded `plot_combined.py`, the stale `cocquet_form_mms.json.orig`, and the dead-end A/B configs
  (`isolation_{trim,full,osgs_trim,osgs_full}.json`) were deleted; `results/` h5 outputs are gitignored.
- **Config prune to designed modes (2026-07-08)** — per [`.agents/rules/official-results-path.md`](../../.agents/rules/official-results-path.md)
  (a test keeps only the configs its designed modes run; finished-debug scrap is removed once its finding
  is documented). `data/` is now the **3 designed modes** `cocquet_form_mms_{vms,vms_k2,taylorhood}.json`
  (the VMS k1/k2 vs Taylor–Hood comparison the harness is built to run). Removed as finished-debug scrap:
  the pre-redesign `cocquet_form_mms.json` (`plot_mesh.py` repointed to `_vms`); the
  `cocquet_form_mms_comparison_*` configs (`galerkin`/`stab` + the 6 `Cconv`/`Cmild`/`Creact` files);
  `cocquet_form_mms_equalorder.json`; and the `isolation_{alphasweep,linctrl}.json` diagnostics (their
  §4.2 reaction-magnitude finding stays fully documented above). Also removed 5 orphaned result DBs whose
  configs were already gone (`isolation_{osgs_full,trim,osgs_trim,full}.h5`, `_validate_osgs.h5`).
- **Diagnostic harness change left in place:** `run_test.jl` now gates the `Constant_Sigma` reaction
  trim on `experimental_reaction_mode` (mirroring `src/run_simulation.jl:57`); it does not affect the
  Forchheimer path the real sweep uses.
