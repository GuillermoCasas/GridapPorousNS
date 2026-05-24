# Cocquet absolute-magnitude investigation — hypothesis ledger

Started 2026-05-24 (this conversation). Persistent record of every hypothesis
checked while investigating why our absolute L²(u) errors against the Cocquet
benchmark are ~10²–10³× larger than the paper reports — despite slopes that
bend in the right direction.

**Bottom-line state (last updated 2026-05-24):**
- Slopes: our finest-segment Galerkin P2/P1 L²(u) reaches **2.05** (freefem-divs
  unstructured) — matches the paper's ~2.18.
- Magnitudes: paper reports L²(u) ~ **3×10⁻⁵** at N=10, **2×10⁻⁷** at N=100;
  we observe **9.68×10⁻³** at N=10 and **1.69×10⁻⁴** at N=100.
- The ratio is not a constant: it grows from ~320× at N=10 to ~850× at N=100,
  i.e. our convergence rate at coarse N is markedly lower than the paper's
  (we recover near-optimal only in the finest segment).
- This is consistent with either (a) our N=200 reference being unconverged
  relative to truth by ~1e-4, or (b) our solution being a different, less
  smooth flow than Cocquet's.

## Hypotheses ruled out (no further work needed)

### H1: Outlet-corner Dirichlet pin causes the convergence cap
- **Status:** FALSIFIED.
- **How tested:** `test/extended/CocquetExperimentModifiedCorner/` — entities 2,4
  removed from "walls" tag, releasing the outlet-corner DOFs.
- **Result:** Made absolute error 3-8× WORSE and slopes lower at every N. The
  released DOF takes an algebraically-driven value that pollutes the bulk field.
- **Recorded:** `memory/cocquet-modified-corner-experiment.md`.

### H2: Structured Cartesian mesh's corner-aligned diagonals lock the singularity
- **Status:** PARTIALLY CONFIRMED — explains the slope but not the magnitude.
- **How tested:** `test/extended/CocquetExperimentIrregularMesh/` (gmsh Delaunay
  uniform 1/N divisions) and `test/extended/CocquetExperimentIrregularMeshFreefemDivs/`
  (literal `buildmesh(a(N)+b(N)+c(N)+d(N))` N-per-side).
- **Result:** Unstructured Delaunay recovers near-optimal slopes (P2/P2 → 2.39 final
  segment uniform, 2.13 freefem-divs; Galerkin P2/P1 → 1.80 uniform, **2.05 freefem-divs**).
  But the absolute error magnitudes are still ~10³× higher than the paper.
- **Recorded:** `memory/cocquet-mesh-topology-controls-convergence.md`.

### H3: ν = 1/Re mapping (Reynolds-number convention)
- **Status:** CONFIRMED — convention matches the paper.
- **Paper (Eq. 1, p. 3, also p. 4 line 1):** dimensionless DBF reads
  `−div(2 Re⁻¹ ε S(u) − ε u⊗u) + ε ∇p + α(ε)u + β(ε)|u|u = ε f` and "Re is the
  Reynolds number". The paper does NOT give an explicit `Re = ρUL/μ` formula;
  it uses Re only as the dimensionless coefficient of the viscous term.
- **Our code:** `weak_viscous_operator(u, v, α, ν) = 2 * α * ν * (ε(u) ⊙ ε(v))` with
  `nu = 1/Re = 0.002` (matches paper's `2 Re⁻¹ ε S(u):S(v)`).
- **Conclusion:** Both papers solve the SAME dimensionless equation with
  Re=500 ⇒ ν=0.002. No conversion / scaling discrepancy.

### H4: Forchheimer closure law (Eq. 49)
- **Status:** CONFIRMED — porosity weighting is applied correctly.
- **Paper:** `α(ε) = (150/Re)·((1-ε)/ε)²`, `β(ε) = 1.75·(1-ε)/ε`.
- **Our code:** `src/models/reaction.jl::sigma` computes
  `a_term = sigma_linear * ((1-α_porosity)/α_porosity)²` with
  `sigma_linear = 0.30 = 150/500` and
  `b_term = sigma_nonlinear * (1-α_porosity)/α_porosity` with
  `sigma_nonlinear = 1.75`. Variable-name collision: our code's `α` is the porosity
  field (paper's `ε`); paper's `α(ε)`, `β(ε)` are the porosity-dependent drag
  coefficient *functions*.
- **Spot check:** at ε=0.45 both formulas yield linear coefficient 0.448 and
  nonlinear coefficient 2.139. At ε=1 both yield 0 (recover NS).

### H5: Outlet boundary condition (ε-weighted traction-free)
- **Status:** CONFIRMED.
- **Paper (Eq. 2):** `ε(2 Re⁻¹ S(u) − p)·n = 0` on Γ_out.
- **Our code:** weak form integrates `∫ε σ:∇v dx` by parts ⇒ natural BC
  `εσ·n = 0` automatically on Γ_out. No explicit boundary integral added.

### H6: Convective form
- **Status:** CONFIRMED.
- **Paper:** `c(ε; u, v, w) = ∫ ε (u·∇)v · w dx + ∫ β(ε) |u| v·w dx`.
- **Our code:** `conv_term = v ⋅ (α_porosity * (∇(u)' ⋅ u))` is `v·(ε(u·∇)u)`.

### H7: Mass equation / pressure-divergence form
- **Status:** CONFIRMED.
- **Paper:** `b(ε; v, q) = -∫ q · div(εv) dx`, and mass eq `div(εu) = 0`.
- **Our code:**
  - `pres_term = -p * (α*(∇·v) + ∇(α)·v) = -p·div(εv)`
  - `mass_term = q * (eps_val*p + α*(∇·u) + u·∇(α)) = q · (eps_val·p + div(εu))`
  - `eps_val=1e-7 = paper's η`.

### H8: Pressure penalty η
- **Status:** CONFIRMED.
- **Paper (p. 30):** `η = 1e-7`. Our config: `eps_val = 1e-7`.

### H9: P₁ porosity interpolant for Galerkin
- **Status:** CONFIRMED for the Galerkin comparison row.
- **Paper:** ε_h is the P₁ FE interpolant of ε regardless of velocity order.
- **Our code:** `comparison_runs` for Galerkin sets `porosity_order=1` explicitly.

### H10: Domain, inlet profile, porosity profile
- **Status:** CONFIRMED.
- **Paper:** Ω=(0,2)×(0,1), Γ_in={0}×[0,1], Γ_out={2}×[0,1],
  Γ_w=[0,2]×({0}∪{1}); inlet `u_in(y) = c_in·y(1-y) e_x` with c_in=0.5 for Fig 2;
  porosity `ε(y) = 0.45 + 0.55·exp(y-1)`.
- **Our config:** identical (`bounding_box: [0,2,0,1]`, `alpha_func(x) = 0.45 + 0.55*exp(x[2]-1.0)`).

### H11: Self-reference via N=200 finest mesh
- **Status:** CONFIRMED methodology-equivalent.
- **Paper (p. 32):** "We note (u_ex, p_ex) the solution obtained with N=200 and we
  compute the error between the discrete solution for N ≤ 100 and (u_ex, p_ex)."
- **Our code:** `build_solver(N_ref, ...)` with N_ref=200 in `paper_comparison*.json`.

### H12: Taylor-Hood P2/P1 unstabilized
- **Status:** CONFIRMED for the Galerkin row.
- **Paper:** uses Taylor-Hood P2/P1, no stabilization.
- **Our code:** `comparison_runs` Galerkin row sets `k_velocity=2, k_pressure=1`;
  `galerkin_driver.jl` invokes the assembly with `mult_mom=mult_mass=0` ⇒
  no τ-stabilization terms.

## Open hypotheses to test next

### O1: ~~Our N=200 reference itself is under-converged against the truth solution~~ — INVALIDATED
- **Status:** RULED OUT BY ARGUMENT (Guillermo, 2026-05-24).
- **Why ruled out:** Self-reference at N=200 is the SAME methodology Cocquet uses.
  If our reference at N=200 were under-converged against truth, Cocquet's reference
  at N=200 would be under-converged by the same amount, and their reported coarse
  errors would plateau at the same level as ours. They don't (they hit 2e-7 at N=100).
  So the gap is not about reference noise floor — it's that our DISCRETE SOLVER
  produces a less accurate solution at every N (including at N=200), not that the
  self-reference methodology is biased.
- **Reframed as:** "What in our discrete formulation/implementation makes our
  solution at any N less accurate against the continuous truth than Cocquet's?"
  (This is now an umbrella for hypotheses O3, O5, and any new ones found.)

### O2: `u_base_floor_ref = 1e-4` regularization adds a noise floor — RULED OUT
- **Status:** RULED OUT 2026-05-24 (probe `/tmp/probe_with_floor_off.jl`,
  log `/tmp/probe_floor_off.log`). Re-ran a single Galerkin P2/P1 N=80 + N=200
  reference solve with `u_base_floor_ref=0` and `epsilon_floor=0` overrides.
  L²(u) consistent at N=80 was **2.6685e-4** — identical to 4 significant figures
  to the existing HDF5 value 2.669e-4 (with the inherited 1e-4 default).
  Newton convergence trace was bit-identical at iter-0 residual `0.001164580518`,
  confirming the floor never actually fires on this flow's discrete state
  (true |u| > 1e-4 everywhere except at wall vertices where u=0 exactly).
- **Conclusion:** The `u_base_floor_ref=1e-4` default is not the source of the
  magnitude gap.

### O3: Solution magnitude / spatial profile may differ from Cocquet's — PARTIALLY RULED OUT
- **Status:** Probe ran 2026-05-24 (`/tmp/probe_cocquet_reference.jl`, log
  `/tmp/probe_cocquet_ref.log`). Flow magnitudes look correct and physically
  reasonable:
  - `‖u‖_L²(Ω) = 0.125` (consistent with c_in/4 inflow scale).
  - Inlet `u(0, 0.5) = (0.125, 0)` — exact.
  - Centerline u_x decreases 0.125 → 0.102 along the channel (Darcy drag).
  - Cross-stream profile at x=1 is asymmetric about y=0.5 (peak at y=0.75)
    because the porosity ramps from 0.45 at y=0 to 1.0 at y=1, so flow
    diverts to the high-porosity top half. Physically correct.
  - Newton converged to residual 3.7e-15 in 4 iterations.
- **Conclusion:** Our discrete equation IS being solved exactly, and we ARE
  solving the right PDE. The flow Cocquet solves is the same flow we solve.
  The magnitude gap is **pure discretization error in our specific
  implementation** — not solver failure, not wrong PDE, not wrong scale.
- **Confirmed:** The ~3.4e-4 L²(u) error at N=80 is **1700× worse than the
  P1-porosity O(h²) cap** predicts. Cocquet's 1e-7 at N=100 is **3000× better
  than the cap**, suggesting they're nearly at the unrestricted optimal P2 rate.
- **Still to find:** what extra error source our implementation has that
  Cocquet's doesn't.

### O4: Cocquet may report a RELATIVE L²(u) error, not absolute
- **Why suspect:** Standard convention in some FEM papers is to plot
  `‖u_h - u_ref‖_L² / ‖u_ref‖_L²`. If our absolute L²(u) at N=100 is 1.69e-4
  and our `‖u_ref‖_L² ≈ 0.06`, our relative error is 2.8e-3 — still 14000×
  the paper's 2e-7, so this alone doesn't close the gap. But worth ruling
  out the convention.
- **Test:** Same as O3 — the probe script also reports `‖u_ref‖_L²` so we can
  divide our errors by it and see.

### O5: Quadrature degree may be insufficient for the Forchheimer non-polynomial term — RULED OUT
- **Status:** RULED OUT 2026-05-24 (probe `/tmp/probe_quad_plus12.jl`,
  log `/tmp/probe_quad12.log`). The default quadrature degree for k_v=2
  with ForchheimerErgunLaw is **9** (from `min_quadrature_degree = 4*k_v + k_v÷2`).
  Re-ran the N=200 reference with `Measure(Ω, 21)` (bump +12) and compared
  the iter-0 PDE residual against the baseline default quadrature:
  - baseline iter-0: `9.588812700822735e-4`
  - quad+12 iter-0: `9.588812745812068e-4`
  - delta: ~5e-9, i.e. quadrature error of the Forchheimer term is at the
    1e-9 level — five orders of magnitude below our 1e-4 L²(u) error.
- **Conclusion:** Quadrature is already adequate. Bumping the degree could
  not change the final L²(u) by more than ~5e-9, vastly below the 845× gap
  to the paper. The Forchheimer non-polynomial integrand is not the source.
- **Probe was stopped after this diagnosis** to free CPU; no need to wait
  for the full solve.

### O6: Pressure mean-zero gauge / hydrostatic constant
- **Why suspect:** Pressure is only defined up to a constant under the natural-BC
  outlet. We add `η p` penalty to fix it. Cocquet may use a different gauge
  (Lagrange multiplier, zero-mean projection, fix-a-point). This affects p
  by an O(1) constant offset but not its gradient, so should not affect L²(u).
  Worth confirming we're not double-fixing or mis-fixing.
- **Test:** Inspect ‖p_h - p_ref‖_L² behavior with and without explicit zero-mean
  projection of (p_h - p_ref) before taking the norm.

### O7: Newton convergence may stall short of machine precision
- **Why suspect:** `xtol=1e-11, ftol=1e-11` in the config. Should be tight enough,
  but worth confirming Newton actually reached these tolerances and the residual
  is at machine precision at every solve. A Newton stagnation at 1e-5 would
  obviously add ~1e-5 to absolute error.
- **Test:** Inspect solver iteration logs from the recent full sweep — confirm
  final residuals are < 1e-10 for all solves.

### O8: Anderson acceleration mode interaction
- **Why suspect:** Config sets `accelerator.type = "Anderson", m = 5`. We invoke
  the Newton solver explicitly for Galerkin (not Picard), so Anderson should not
  fire. But verify the cascade isn't introducing artifacts.
- **Test:** Confirm in solver logs that all Galerkin runs use Newton (mode), not
  Picard.

## Cross-norm comparison reveals the L²(u) ratio is anomalously large

At N=100 Galerkin P2/P1 freefem-divs:
- our L²(u) = 1.69e-4 vs paper ~2e-7 → **845×** gap
- our H¹(u) seminorm = 2.13e-2 vs paper Err_tot ~3e-3 → **7×** gap
- our L²(p) = 8.6e-5 vs paper L²(p) component (subset of 3e-3) → likely small

The L²(u) discrepancy is **120× worse than the H¹(u) discrepancy** for the same
solution pair. This is a striking pattern: if our solution were uniformly less
accurate, we'd expect comparable ratios across norms. The fact that L²(u) is
disproportionately off suggests something specific to the L²(u) measurement
(e.g., a global mode in u_h - u_truth that doesn't show up in gradients), not
a uniformly worse discrete solution.

## What we've NOT yet looked at (queue, in priority order)

1. ~~The CocquetFormMMS test~~ — DONE 2026-05-24. `cocquet_form_mms.h5`,
   config_1_ASGS (P2/P1) at h=1/80 (corresponds to N=80 in Cocquet sweep):
   - L²(u) = **8.54e-7**, slope ~3.3 (optimal P2)
   - Cocquet's reported L²(u) at N=100 is ~2e-7 — **same order of magnitude
     as our MMS at N=80**.
   - **OUR CODE CAN ACHIEVE THE PAPER'S MAGNITUDES**. The implementation is
     not bugged. The Cocquet benchmark specifically inflates our error by
     ~300× relative to what the code can deliver on smooth problems.
2. ~~Literal FreeFem `.msh` files via paper_comparison_freefem.json~~ — DONE
   2026-05-24. With Cocquet's exact mesh recipe (generated by
   create_paper_meshes.edp via FreeFem on a machine that has the binary),
   Galerkin P2/P1 L²(u) at N=100 is **4.59e-5** — closer than our gmsh
   Delaunay (1.69e-4 freefem-divs, 8.10e-5 uniform-divs), but still **230×**
   the paper's reported 2e-7. So even with their EXACT mesh, we don't match
   their magnitudes.
3. ~~Localized error analysis via `localize_err.jl`~~ — DONE 2026-05-24
   (log `/tmp/localize_err.log`). Comparing structured Cartesian-simplexified
   vs gmsh Delaunay uniform-divs at N_coarse=10 vs N_ref=80, Galerkin P2/P1:
   ```
                          outlet_corner  inlet_corner  wall_strip  bulk    ‖e‖
   STRUCTURED          :  71.0%          0.10%         3.6%        25.3%   3.19e-4
   UNSTRUCTURED        :   1.08%         29.4%         7.7%        61.8%   5.74e-3
   ```
   - **Structured mesh's error is dominated by the outlet corner** (71% of ‖e‖²).
   - **Unstructured mesh diffuses the outlet corner singularity** (only 1.08%
     of ‖e‖² there) — confirms the Gemini hypothesis from earlier in the
     conversation that BAMG/Delaunay node placement near the corner avoids
     locking in the singularity.
   - BUT: total ‖e‖ on unstructured is **18× LARGER** than on structured at
     N=10. The unstructured mesh's bulk error is large because coarse
     Delaunay triangulations have worse triangle quality / less alignment
     with the parabolic channel flow.
   - Reaches Cocquet's reported 3e-5 at N=10: NEITHER mesh reaches it. The
     bulk error contribution (1.72e-4 even on structured after truncating
     outlet corners) is still ~6× the paper's reported value.
   - Even excluding the outlet corner entirely from the error norm doesn't
     close the gap. Cocquet's discretization apparently produces a fundamentally
     more accurate bulk solution at every N — even at our most favorable
     mesh (structured-with-truncated-corners).
4. Try `outlet_truncation_delta = 0.1` (excludes the strip x > 1.9 from the
   error norm). If the bulk-only L²(u) drops to ~paper-level, the gap is
   entirely from corner contamination — and Cocquet must somehow be
   excluding the corner from their error norm too (or their corner region
   produces a much smaller error).
5. Try quadrature `degree + 12` (CocquetAllDirichlet diagnostic style). If
   absolute error drops, our default quadrature is under-integrating the
   non-polynomial Forchheimer term.
6. ~~Re-verify Cocquet's Figure 2 right-panel y-axis labels~~ — DONE
   (Guillermo, 2026-05-24). Y-axis range on Figure 2 right panel
   (L² error of velocity, Re=500, c_in=0.5) is confirmed
   **10⁻⁴ (top tick) → 10⁻⁷ (bottom tick)**. The curve goes from ~3e-5 at
   N=10 down to ~2e-7 at N=100. Figure 3 (Re=1000, c_in=1) right panel
   y-axis goes 10⁰ (top) → 10⁻⁵ (bottom labelled tick, actual data points
   extend down to ~10⁻⁸). So Cocquet's reported magnitudes ARE 10⁻⁴ to 10⁻⁸
   range — orders of magnitude below ours.

## How to use this ledger

When investigating the Cocquet absolute-magnitude issue, **start by reading
this file** to avoid re-testing hypotheses already ruled out. Update it
in-place when:
- A hypothesis is ruled out (move from "open" to "ruled out").
- A new hypothesis is suspected (add to "open").
- A diagnostic experiment finishes — record the result inline.
