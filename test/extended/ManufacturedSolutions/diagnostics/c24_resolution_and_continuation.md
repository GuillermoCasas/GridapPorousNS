# The stiff MMS corner (C24): diagnosis + continuation fix

**Date:** 2026-05-21. **Supersedes** the self-flagged-unreliable `probe_stiff_raw.md` /
`probe_stiff_findings.md` (whose harness mis-classified its own easy-cell sanity check).

Cell under study — **C24**: `Re=1e6, Da=1, α₀=0.05, k=1, QUAD, ASGS` (and its sibling **C21**
`Da=1e-6`). These are the only cells in the MMS sweep that genuinely fail.

---

## TL;DR

C24's failure is **not a solver or Jacobian bug**. At coarse mesh the **discrete VMS solution
branch folds (turning point) before reaching the corner** — there is *no* root with ‖R‖≤ftol to
converge to. The fold **recedes with mesh refinement**; at fine enough N the corner has a root
and converges at the expected MMS rate. A harness-level **α-continuation + mesh-continuation**
driver (`run_continuation.jl`) reaches it. **No `src/` core files were modified.**

The prior session's "fix" only stopped the NaN; the production solver then reported the cell as
*converged* at the loose dynamic noise floor (‖R‖≈5e-4 at N=10) with L2 u ≈ 0.13 — a ~25× wrong
solution. That false-success reporting is the real latent defect (see "Open item" below).

---

## Decisive diagnosis (trustworthy harness: `probe_stiff_diagnose.jl`, report `probe_diag_v2.md`)

A benign-cell gate (`Re=1,Da=1,α=0.5`) validates the harness before any stiff-cell number is read.

- **A1 — Jacobian consistency (the decisive test, never cleanly run before).** Assembled
  Exact-Newton `J·v` vs centered finite-difference of the residual at `u_ex`:
  **best relative error 4.8e-12**, clean ε-convergence, both velocity and pressure blocks.
  ⇒ the Exact-Newton Jacobian is exact. The "inconsistent τ-derivative" hypothesis (H4) is **dead**.
- **A2 — root existence.** Heavy Newton *and* Picard from `u_ex` (budget 500, `noise_floor=1e-12`
  so no false success) both **stall at ‖R‖≈5e-2**, L2 u ≈ 0.12–0.17. ⇒ no root near the exact
  solution.
- **Continuation folds in every parameter direction** (warm-started from converged neighbors):
  Da folds at ≈5e5 (target 1), Re at ≈5.6 (target 1e6), α at ≈0.16 (target 0.05, N=40). Adaptive
  step-halving down to a tiny step **confirms a TRUE fold** (not a step-size artifact).
- **The fold recedes with mesh:** α-fold ≈ 0.24 (N=10) → 0.16 (N=40) → 0.106 (N=80) → … The
  branch *solutions above the fold* are clean (machine-zero residual) and converge at rate ≈2.
- **A3 — landscape.** cell-Péclet ≈ 4.7e4 in the α=0.05 core at N=10 (convective `c₂|u|/h`
  dominates viscous `c₁ν/h²` by ~10⁴–10⁵). The velocity floor never activates (eff_speed=|u|),
  so the fold is a coarse-mesh coercivity limit, not a regularization artifact.

**Why α is the only viable continuation axis:** Da- and Re-continuation hold α=0.05 fixed, so the
stiff low-porosity layer is present the whole way and they fold almost immediately, astronomically
far from their targets. α-continuation starts at α=1 (smooth, easy) and *relieves* the layer, so
it tracks closest to the target — and the fold it hits recedes with refinement.

---

## The fix: `run_continuation.jl` (harness-level, zero core-src changes)

Reuses the package's solve path (`build_cell` mirrors `run_test.jl`'s FE setup; per-step solves use
the tight-tolerance Newton-with-Picard-fallback from the diagnostic — a continuation driver needs a
*true* root at each step, which the production dynamic noise floor does not guarantee). Continuation
knobs come from a harness JSON (`data/continuation_*.json`), as `run_test.jl` already does for
`epsilon_pert` / `mms_*`.

Two modes:
- **α-continuation** (`mesh.convergence_partitions`): from α=1 down to the target α, geometric ramp
  with adaptive step-halving; warm-starts each step. Reports how far each mesh tracks before folding.
- **mesh-continuation rate study** (`mesh_continuation` block): α-continuation ONCE at a base mesh
  fine enough that the target α is above its fold (so no expensive fold-probing), then **interpolate
  the converged solution onto finer meshes** (Gridap `Interpolable`) and solve once each. Yields ≥2
  converged meshes **at the target α** → a defensible MMS rate at the corner.

**Validation:** mesh-continuation at α=0.20 (N=80→160) gave L2 u rate **2.80**, H1 u rate **1.09**
(true roots, 4 iters for the interpolated fine solve). The rate machinery is sound.

### Usage
```bash
cd test/extended/ManufacturedSolutions
# α-continuation, one or more meshes, report fold/convergence:
julia --project=../../.. run_continuation.jl continuation_c24.json
# mesh-continuation rate study at the target α (≥2 meshes at α):
julia --project=../../.. run_continuation.jl continuation_c24_rate.json
```

---

## Results — C24 convergence at the target α=0.05

Base α-continuation at N=512 (which clears the fold — 0.05 is well above it) reached α=0.05 with a
**true root** (‖R‖=6.3e-10) in 2 Newton iters/step; mesh-continuation then interpolated up to N=768
and N=1024 (one solve each, 2 iters). Normalized errors **at the target corner**:

| N | h | L2 u | H1 u | ‖R‖∞ |
|---|---|---|---|---|
| 512  | 1.953e-3 | 2.648e-4 | 8.558e-2 | 6.3e-10 |
| 768  | 1.302e-3 | 7.746e-5 | 5.623e-2 | 3.6e-9  |
| 1024 | 9.766e-4 | 3.273e-5 | 4.206e-2 | 4.1e-12 |

| pair | rate L2 u | rate H1 u |
|---|---|---|
| 512→768  | 3.03 | 1.04 |
| 768→1024 | 2.99 | 1.01 |

**H1 u rate ≈ 1.0** is textbook for k=1; **L2 u rate ≈ 3.0** is consistently above the nominal 2
(the α=0.20 validation showed L2 2.80 too) — a superconvergent / pre-asymptotic L2 behaviour of
this smooth MMS in the stabilized equal-order setting. The point for this investigation is decisive:
**C24 converges to a true root at the exact target α=0.05, at FE-optimal H1 rate, on ≥2 meshes** —
the coarse-mesh "failure" was the absence of a discrete root (the fold), not a solver defect.

## Generalization — C21 (Da=1e-6, same corner)

α-continuation at N=512 reaches α=0.05 with a true root, **bit-identical to C24**:
`‖R‖=6.26e-10, L2u=2.648e-04, H1u=8.558e-02` (2 iters/step throughout). In fact every per-α step
matches C24 to ~4 sig figs, because at this corner σ ∝ Da is negligible against convection
(σ=1e-6 for C24, 1e-12 for C21, vs convective scale ≈19). **Conclusion: the fold is driven by the
α-layer + high Re, not Da** — so the α-continuation fix rescues the whole high-Re/low-α corner
(C21, C24, and the Da=1e6 sibling C27 which already converged) regardless of Da. Da- and
Re-continuation, by contrast, hold α=0.05 fixed and fold almost immediately.

---

## Open item (not addressed here, per the no-core-change constraint)

The production solver reports past-the-fold corners as **"converged"** when the cascade stops at the
dynamic noise floor (`run_test.jl` forces `dynamic_noise_floor ≈ dynamic_ftol·10` ≈ 1e-3 at N=10,
Re=1e6) while the iterate is ~25× from any true root. An honest-exit fix — count
`stagnation_noise_floor_reached` as success only when ‖R‖ ≤ k·dynamic_ftol — would make the sweep
report these corners as failures instead of wrong "successes." It touches `run_test.jl`
(harness) and the success classification in `src/solvers/porous_solver.jl` (core), so it is left
for a follow-up that is allowed to touch the core.
