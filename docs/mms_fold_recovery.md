# Fold recovery at the stiff MMS corner (C24/C21): diagnosis + continuation method

Permanent record of why the high-Re / low-porosity MMS corner appears to "fail" on coarse meshes,
and the harness-level continuation method that recovers the true FE solution. Consolidated from the
investigation diagnostics (`probe_stiff_diagnose.jl` → `probe_diag_v2.md`, and the
`c24_resolution_and_continuation.md` scratch note, both since removed). The reusable driver is
[run_continuation.jl](../test/extended/ManufacturedSolutions/run_continuation.jl); the evidence
configs are `data/continuation_{c24,c24_rate,c21}.json`.

Cells under study — **C24**: `Re=1e6, Da=1, α₀=0.05, k=1, QUAD` (and its sibling **C21** `Da=1e-6`).
These are the only cells in the MMS sweep that genuinely fail from the exact-solution initial guess.

---

## TL;DR

The failure is **not a solver or Jacobian bug**. At coarse mesh the **discrete VMS solution branch
folds (turning point) before reaching the corner** — there is *no* root with ‖R‖≤ftol to converge to.
The fold **recedes with mesh refinement**; at fine enough N the corner has a root and converges at the
expected MMS rate. A harness-level **α-continuation + mesh-continuation** driver reaches it, with
**zero changes to the paper-faithful core** (`src/solvers`, `src/formulations`, `src/stabilization`,
`src/config`, schema).

---

## Decisive diagnosis

A benign-cell gate (`Re=1, Da=1, α=0.5`) validates the diagnostic harness before any stiff-cell
number is trusted.

- **A1 — Jacobian consistency (the decisive test).** Assembled Exact-Newton `J·v` vs centered
  finite-difference of the residual at `u_ex`: **best relative error 4.8e-12**, clean ε-convergence
  in both velocity and pressure blocks. ⇒ the Exact-Newton Jacobian is exact; the "inconsistent
  τ-derivative" hypothesis is dead.
- **A2 — root existence.** Heavy Newton *and* Picard from `u_ex` (budget 500, `noise_floor=1e-12` so
  no false success) both **stall at ‖R‖≈5e-2**, L2 u ≈ 0.12–0.17. ⇒ no root near the exact solution.
- **Continuation folds in every parameter direction** (warm-started from converged neighbours): Da
  folds at ≈5e5 (target 1), Re at ≈5.6 (target 1e6), α at ≈0.16 (target 0.05, N=40). Adaptive
  step-halving to a tiny step **confirms a TRUE fold**, not a step-size artifact.
- **The fold recedes with mesh:** α-fold ≈ 0.24 (N=10) → 0.16 (N=40) → 0.106 (N=80) → … Solutions
  *above the fold* are clean (machine-zero residual) and converge at rate ≈2.
- **A3 — landscape.** cell-Péclet ≈ 4.7e4 in the α=0.05 core at N=10 (convective `c₂|u|/h` dominates
  viscous `c₁ν/h²` by ~10⁴–10⁵). The velocity floor never activates, so the fold is a coarse-mesh
  coercivity limit, not a regularization artifact.

**Why α is the only viable continuation axis:** Da- and Re-continuation hold α=0.05 fixed, so the
stiff low-porosity layer is present the whole way and they fold almost immediately, astronomically
far from their targets. α-continuation starts at α=1 (smooth, easy) and *relieves* the layer, so it
tracks closest to the target — and the fold it hits recedes with refinement.

---

## The method: `run_continuation.jl` (harness-level, zero core-src changes)

Reuses the package solve path (`build_cell` mirrors `run_test.jl`'s FE setup; per-step solves use a
tight-tolerance Newton-with-Picard-fallback — a continuation driver needs a *true* root at each step,
which the production dynamic noise floor does not guarantee). Continuation knobs come from a harness
JSON, as `run_test.jl` reads `epsilon_pert` / `mms_*`. Regimes are selected by the config `regime`
field (see the file header):

- **`alpha`** (per-mesh α-continuation): from α=1 down to the target α, geometric ramp with adaptive
  step-halving, warm-starting each step. Reports how far each mesh tracks before folding.
- **`mesh_ladder`** (mesh-continuation rate study): α-continuation ONCE at a base mesh fine enough
  that the target α is above its fold (so no expensive fold-probing), then **interpolate the converged
  solution onto finer meshes** (Gridap `Interpolable`) and solve once each. Yields ≥2 converged meshes
  **at the target α** → a defensible MMS rate at the corner.

### Usage
```bash
cd test/extended/ManufacturedSolutions
# α-continuation, one or more meshes (regime auto-selected as "alpha"):
julia --project=../../.. run_continuation.jl continuation_c24.json
# mesh-continuation rate study at the target α (regime "mesh_ladder"):
julia --project=../../.. run_continuation.jl continuation_c24_rate.json
# batch-rescue every flagged cell from a sweep:
julia --project=../../.. run_continuation.jl phase2 results/flagged_cells.json results/phase2_results.json
```

---

## Results — C24 convergence at the target α=0.05

Base α-continuation at N=512 (which clears the fold — 0.05 is well above it) reached α=0.05 with a
**true root** (‖R‖=6.3e-10) in 2 Newton iters/step; mesh-continuation interpolated up to N=768 and
N=1024 (one solve each, 2 iters). Normalized errors **at the target corner**:

| N | h | L2 u | H1 u | ‖R‖∞ |
|---|---|---|---|---|
| 512  | 1.953e-3 | 2.648e-4 | 8.558e-2 | 6.3e-10 |
| 768  | 1.302e-3 | 7.746e-5 | 5.623e-2 | 3.6e-9  |
| 1024 | 9.766e-4 | 3.273e-5 | 4.206e-2 | 4.1e-12 |

| pair | rate L2 u | rate H1 u |
|---|---|---|
| 512→768  | 3.03 | 1.04 |
| 768→1024 | 2.99 | 1.01 |

**H1 u rate ≈ 1.0** is textbook for k=1; **L2 u rate ≈ 3.0** is consistently above the nominal 2 (the
α=0.20 validation showed L2 2.80 too) — a superconvergent / pre-asymptotic L2 behaviour of this smooth
MMS in the stabilized equal-order setting. Decisive point: **C24 converges to a true root at the exact
target α=0.05, at FE-optimal H1 rate, on ≥2 meshes** — the coarse-mesh "failure" was the absence of a
discrete root (the fold), not a solver defect.

## Generalization — C21 (Da=1e-6, same corner)

α-continuation at N=512 reaches α=0.05 with a true root, **bit-identical to C24**:
`‖R‖=6.26e-10, L2u=2.648e-04, H1u=8.558e-02` (2 iters/step throughout). Every per-α step matches C24
to ~4 sig figs, because at this corner σ ∝ Da is negligible against convection (σ=1e-6 for C24, 1e-12
for C21, vs convective scale ≈19). **The fold is driven by the α-layer + high Re, not Da** — so the
α-continuation method rescues the whole high-Re/low-α corner (C21, C24, and the Da=1e6 sibling C27,
which already converged) regardless of Da.

---

## Resolved follow-up — honest-exit gate

The original diagnosis flagged a latent defect: the production solver reported past-the-fold corners
as **"converged"** when the cascade stopped at the loose dynamic noise floor while the iterate was
~25× from any true root. This has since been addressed by the **honest-exit gate**: a noise-floor stop
counts as success only when `‖R‖∞ ≤ k_nf · effective_ftol`, controlled by
`solver.noise_floor_success_max_ftol_multiple` (`k_nf`) and surfaced in the sweep's analysis via the
relative-residual `is_true_root` test. The Phase-1 sweep configs set `k_nf = 10.0` to reject these
fold stalls so they are reported as flagged cells (then rescued via the continuation method above)
rather than as wrong "successes."
