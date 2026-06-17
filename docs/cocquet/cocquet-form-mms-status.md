# CocquetFormMMS — status & findings

**Scope.** This is the **canonical** doc for `test/extended/CocquetFormMMS`, the
manufactured-solution (MMS) sibling of the Cocquet benchmark. It compares, under one MMS
convergence sweep, the **equal-order stabilized VMS** method (P1/P1 & P2/P2, ASGS + OSGS) against the
unstabilized **Galerkin Taylor–Hood P2/P1** ("Cocquet element"), using the **full Forchheimer–Ergun
reaction** `σ = a(α)+b(α)|u|`. The driving question: *how does the equal-order stabilized method hold
up against the inf-sup-stable Cocquet element across the porosity/Reynolds space?*

Related: the standard MMS sweep is `docs/mms/convergence-status.md` (a *different*, constant-σ
harness); the physical Cocquet benchmark is `docs/cocquet/investigation-synthesis.md`. The
τ-saturation mechanism referenced below is written up in
`theory/tau_saturation_note/tau_saturation_note.tex`.

**Last updated: 2026-06-16.** Moderate-porosity results are complete and clean; the low-porosity
high-Re corner folds for the stabilized method (Taylor–Hood converges there); the *exact* cause of
that fold is **investigated but not yet confirmed** (see §4). Runs were stopped before the full
k=2/N=160 sweep finished — moderate α is the kept deliverable.

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
  error norms all L-scaled; `eps_val` made encoding-covariant). `"unit"` reproduces the legacy L=U=1.
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
| **α=0.1, Re=1e5** | **FOLD** (1/5) both | ASGS 3/4, OSGS 1/4 | ✅ 5/5 |

- **Moderate porosity (α=0.5): the definitive result.** Clean, complete convergence for *every*
  method (k=1, k=2, ASGS, OSGS, Taylor–Hood) at both Re. Velocity optimal (L² rate ≈2.2 for k=1),
  pressure converges. ASGS and OSGS are genuinely distinct (e.g. α=0.5/Re=1e5/N=80: ASGS L²u≈3.2e-4
  vs OSGS≈4.2e-4).
- **α=0.1, Re=1:** stabilized methods converge.
- **α=0.1, Re=1e5 — the hard corner:** the equal-order **stabilized methods fold** (gate
  unreachable; solution recorded at the achievable floor), while the **unstabilized Taylor–Hood
  converges everywhere** (all 4 cells × 5 meshes). This VMS-folds / TH-converges contrast at the
  high-Re × low-α corner is the headline open phenomenon. (This is also the corner the *paper itself
  skips* in its own sweeps via `skip_cells`.)

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

## 4. Why low-α folds — investigation status (the open question)

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

**Bottom line:** the *exact* cause of the equal-order stabilized low-α fold (that Taylor–Hood avoids)
is **open**. The `σ̃_α`/reaction-in-stabilization mechanism is the leading hypothesis but is
unconfirmed; `theory/tau_saturation_note` deliberately does not assert it.

---

## 5. Open items / next steps
- **Confirm or refute Layer 2.** Valid tests: (a) finish the **OSGS** trim-vs-full A/B at low α; (b)
  for **ASGS**, a code change to strip σu from the stabilization residual (the projection trim won't
  do it for ASGS). If trimming the reaction recovers convergence, the mechanism is confirmed.
- **Recovering the fold cells** (if wanted as converged, not folded): **continuation into the corner**
  (start at α=0.5 or Re=1, walk to α=0.1 / Re=1e5), the same device the regular harness's Phase-2
  `run_continuation.jl` uses for its excluded corner.
- **Throwaway probes to clean** before/after commit: `data/isolation_*.json`, `results/isolation_*.h5`,
  `diagnose_*.jl`, `data/_validate_*.json`.
- **Diagnostic harness change left in place:** `run_test.jl` now gates the `Constant_Sigma` reaction
  trim on `experimental_reaction_mode` (mirroring `src/run_simulation.jl:57`); it does not affect the
  Forchheimer path the real sweep uses.
