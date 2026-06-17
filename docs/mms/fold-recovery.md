# Fold recovery at the stiff MMS corner (C24/C21): diagnosis + continuation method

> Canonical fold sub-topic — parent: [`mms_convergence_status.md`](convergence-status.md) §3 (region B). This file carries the deep diagnosis and the continuation rescue; the parent has the grid-wide picture.

Permanent record of why the high-Re / low-porosity MMS corner appears to "fail" on coarse meshes,
and the two methods that recover the true FE solution: the original harness-level **continuation**
driver, and the simpler **direct exact-guess solve** (the production path since 2026-06-17). The
diagnostics were consolidated from `probe_stiff_diagnose.jl` (the minimal version was **restored**
2026-06-17 — see below) and the `c24_resolution_and_continuation.md` scratch note (removed). Drivers:
[run_continuation.jl](../../test/extended/ManufacturedSolutions/run_continuation.jl) (continuation;
evidence configs `data/continuation_{c24,c24_rate}.json`) and the newer
[run_corner_article.jl](../../test/extended/ManufacturedSolutions/run_corner_article.jl) /
[run_corner_osgs.jl](../../test/extended/ManufacturedSolutions/run_corner_osgs.jl) (direct solve).

Cells under study — `Re=1e6, α₀=0.05, Da∈{1e-6,1,1e6}`, both ASGS and OSGS, for each element family.
These are the only cells in the MMS sweep that fail from the exact-solution initial guess at coarse N.

> ## ✅ 2026-06-17 — TRI k=1 (P1) corner REPRODUCED; direct solve supersedes continuation
>
> The three P1/TRI corner cells (and both stabilizations) were solved in Gridap. **Key finding: once
> the discrete fold has cleared (≈N=512), no α-continuation is needed** — a plain Newton solve from
> the interpolated exact solution reaches the true root in ~3 iters. Continuation is only required to
> *reach* a root when **none exists** at coarse N (the fold). Evidence: N=160 still folds
> (`err_u=NaN`, ‖R‖≈0.85); N=512 converges in 3 Newton iters to ‖R‖≈7e-11. So the production recipe is
> now **direct-solve-at-N≥512 + mesh-step up**, not the ~70-LU α-ramp.
>
> Results match the QUAD continuation numbers below (vel L² slope ≈3.0, H¹ ≈1.0). OSGS≈ASGS at this
> corner (Pe≈5e4, projection negligible). `Da=1e-6 ≡ Da=1` (Forchheimer Da no-op). Full per-cell
> numbers and the LaTeX reproduction of the paper tables: `make_results_tables.py` →
> `results/paper_tables.tex`. **Remaining work and the unresolved Gridap-vs-Kratos magnitude offset:
> see "Current status & remaining work" at the bottom of this file.**

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

---

## The production method (2026-06-17): direct exact-guess solve

Continuation is the **conservative** route — it can reach the corner even before knowing a root
exists. But the diagnosis above ("the fold recedes with mesh") implies a cheaper path: at any mesh
where a root *does* exist, the interpolated exact solution is an excellent initial guess for it, so a
plain Newton solve converges directly — no α-ramp. Empirically the fold clears α=0.05 by **≈N=512**, so:

- **`run_corner_article.jl` (ASGS).** For each `Da`, direct Newton from the exact guess at base N=512
  (≈3 iters to ‖R‖~1e-11), then mesh-step (Gridap `Interpolable`) up to N=768 (~2 iters). All four
  normalized norms (u/p × L²/H¹) + the two-finest-mesh slopes are written to
  `results/debug_results/corner_tri_k1_a005.json`. This is ~6 LU solves total vs ~70 for an α-ramp
  (each N=512 LU is ~200 s, so the α-ramp was ~4 h/cell; direct solve is ~25 min/cell).
- **`run_corner_osgs.jl` + `osgs_corner_lib.jl` (OSGS).** The OSGS coupled solve (mirrors
  `solve_osgs_stage!`) is **warm-started from the ASGS root** at each mesh — the ASGS root is O(h²)
  from the OSGS root, so the error is reliable in a few iters. ⚠️ **Caveat:** the OSGS coupled solve
  converges *slowly-linearly* (frozen-π) and is stopped at the production-level residual
  (`ftol=1e-6`/noise-floor `1e-5`), not a tight true root. The **FME is reliable** (warm-from-ASGS,
  OSGS≈ASGS), but the OSGS *slope* can be mildly inflated when the coarse (N=512) point is not fully
  settled — e.g. Da=1e6 reads slope 2.63 vs ASGS 2.11; tighten the N=512 OSGS solve for a cleaner slope.

**`run_continuation.jl` was broken on `main`** and is fixed: it `include`s `probe_stiff_diagnose.jl`,
which had been deleted in commit `3c66edd`. A minimal `probe_stiff_diagnose.jl` (just `build_cell` +
`probe_a2_heavy_solve` + `CellArtifacts` + `calc_normalized_errors`, verbatim from `3c66edd~1`) was
restored; all referenced package symbols/config fields survived the refactor. Both the continuation
driver and the direct-solve drivers reuse these primitives.

---

## Current status & remaining work (for future sessions)

| Corner family | Cells | Status | Path used |
|---|---|---|---|
| **P1 / TRI k=1** | Re=1e6, α₀=0.05, Da∈{1e-6,1,1e6}, ASGS+OSGS | ✅ **DONE** | direct solve N=512→768 |
| **Q2 / QUAD k=2** | same 6 cells | ❌ **NOT COMPUTED** | — (future work) |

The P1 results populate the `n.c.` corner rows of the **Linear** tables in `results/paper_tables.tex`;
the **Quadratic** tables still show `n.c.` for the corner.

**Open item — Gridap-vs-Kratos magnitude offset (unresolved).** Our Gridap corner FME are **~3–12×
larger** than the article's (Kratos) values, in a norm-dependent way (e.g. vel L²: Gridap 7.9e-5 @N=768
vs the paper's 1.1e-5 @N=640; pressure L² ~5×; H¹ ~2.4×). The **rates agree** (optimal/super-optimal,
≈2–3) and the Gridap TRI numbers match the Gridap QUAD continuation to ~2%, so the discretization is
internally consistent — the offset is a code-vs-code calibration question (candidates: characteristic
scale `U_c`/`P_c` normalization, porosity-field definition, MMS amplitude). Worth reconciling before
the table is taken as a literal reproduction of the paper.

**To finish the Q2 corner (next session).** Expected to be the same recipe — direct exact-guess solve
at N≥512 — but k=2 was not yet run, so a few things to verify/watch:
- The fold-clearing mesh may differ for Q2 (k=2 resolves the α-layer better per-DOF; the fold may
  clear at a coarser N, or the larger per-mesh DOF count may make N=512+ LU heavier — watch memory).
- Q2 has **no corner trace/JSON** yet, so `make_results_tables.py` shows those iteration cells as
  `n.c.`/`--`; the generator already reads a `corner_quad_k2_*.json` if one is added (mirror the
  `corner_tri_k1_a005*.json` plumbing in `load_corner`).
- Reuse `run_corner_article.jl` (it takes `etype`/`kv` — generalize `main()` from TRI/k=1 to QUAD/k=2)
  and `run_corner_osgs.jl`. Both already worked for TRI; the only TRI-specific bit is the hardcoded
  output filename and the `[1e6]/[0.05]` grid in `main()`.
- **What was tried that did NOT work / was wasteful (avoid repeating):** (i) the standard sweep
  (`run_test.jl`) cannot produce these — they fold at N≤320 (`skip_cells` exists for this reason);
  (ii) the α-continuation `mesh_ladder` at base N=512 is correct but **~4 h/cell** (each failing step
  near the fold burns up to `max_iters_per_step` LU factorizations) — the direct solve is ~10× faster
  and was validated to give identical roots; (iii) a transient **swap episode** (process RSS climbing
  across cells) made one N=512 solve 9× slower — run corner cells in **fresh processes** to keep
  memory clean; (iv) tight OSGS tolerance (ftol=1e-8) grinds for ~hours (frozen-π linear rate) — use
  the production noise-floor stop and warm-start from the ASGS root.
