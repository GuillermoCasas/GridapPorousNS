# Cocquet Convergence Analysis and Findings

> **HISTORICAL — phased slope investigation (diary).** Canonical Cocquet reference: [cocquet_investigation_synthesis.md](investigation-synthesis.md). This file is the detailed chronological evidence trail (Phases 1–12); later phases withdraw some earlier conclusions, so read a single phase only in light of the synthesis.
>
> **⚠ SETTLED CONCLUSION — Phases 9–10 verdicts WITHDRAWN (see Phase 11).** Phase 9's "root cause PROVEN — Dirichlet/Neumann corner singularity (mesh- and method-independent)" and Phase 10's "the paper's $H^3{\times}H^2$ regularity claim is incorrect" were **both drawn entirely on a structured Cartesian-simplexified mesh** and are **withdrawn** by Phase 11: the cap is a **structured-mesh artifact** (the corner-aligned triangulation locks the singularity), and an **unstructured Delaunay mesh recovers $O(h^2)$**, matching the paper. The residual is a **magnitude** gap of **~30× (coarse) to ~300× (fine)** — not the "~2×" Phase 11–12 header once claimed (that reading was off by an order of magnitude; corrected in the [replicating-cocquet](archive/replicating-cocquet-transcript.md) session and folded into the synthesis), and not the older 320×–845× freefem-divs figure. Read the phases only in light of the synthesis.

This document summarizes the investigation into the sub-optimal convergence rates observed in the Cocquet numerical experiment.

## 1. Context and Problem Statement
In the Cocquet benchmark test (defined in `test/extended/CocquetTubeTest/run_convergence.jl`), we solve the 2D steady-state Darcy-Brinkman-Forchheimer (DBF) model on the domain $\Omega = [0, 2] \times [0, 1]$ with:
- An inlet velocity boundary condition $u_{\text{in}}(y) = c_{\text{in}} y (1 - y)$ at $x = 0$ with $c_{\text{in}} = 0.5$.
- No-slip walls at $y = 0$ and $y = 1$.
- A traction-free Neumann boundary condition at the outlet $x = 2$.
- A transversally varying porosity field $\alpha(y) = 0.45 + 0.55 e^{y-1}$.
- A Reynolds number of $\text{Re} = 500$ and viscosity $\nu = 0.002$.

Using Taylor-Hood elements ($P_2$ for velocity, $P_1$ for pressure) and ASGS/OSGS stabilization, the expected asymptotic convergence rates are $O(h^3)$ for velocity and $O(h^2)$ for pressure in the $L^2$ norm. However, evaluating the solutions on coarse meshes ($N \in \{10, 20, 40\}$) against a highly resolved reference solution ($N_{\text{ref}} = 80$) yielded sub-optimal convergence slopes:
- **Velocity $L^2$ slopes**: $\approx 0.14$ (for $10 \to 20$) and $\approx 0.53$ (for $20 \to 40$)
- **Pressure $L^2$ slopes**: $\approx 0.11$ (for $10 \to 20$) and $\approx 0.46$ (for $20 \to 40$)

---

## 2. Investigation Phase 1: Verification of Metric Coherency
To verify whether the sub-optimal convergence rate was a mathematical artifact of the cross-mesh projection or error integration, we created a verification script:
[`scratch/test_coherency.jl`](file:///Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/scratch/test_coherency.jl)

This script:
1. Interpolates a smooth, divergence-free analytical field:
   $$\boldsymbol{u}_{\text{ana}}(x, y) = \begin{pmatrix} \sin(\pi x / 2) \cos(\pi y) \\ -0.5 \cos(\pi x / 2) \sin(\pi y) \end{pmatrix}, \quad p_{\text{ana}}(x, y) = \sin(\pi x / 2) \sin(\pi y)$$
2. Computes the cross-mesh $L^2$ error against a reference mesh ($N_{\text{ref}} = 80$) using `compute_reference_errors` from `src/metrics.jl`.

### Verification Results
On this smooth test case, the cross-mesh errors and convergence slopes were:
- **Velocity $L^2$ convergence slope**: $2.96$ ($10 \to 20$) and $2.99$ ($20 \to 40$) — recovering the expected $O(h^3)$ rate.
- **Pressure $L^2$ convergence slope**: $2.00$ ($10 \to 20$) and $2.00$ ($20 \to 40$) — recovering the expected $O(h^2)$ rate.

### Conclusion of Phase 1
The cross-mesh evaluation metrics, mesh projection, and integration are mathematically and implementation-wise **100% correct and coherent**. The sub-optimal convergence rate is a physical/numerical feature of the solved PDE system, not a measurement bug.

---

## 3. Investigation Phase 2: Outlet Boundary Effect Sweep
To determine if corner singularities or boundary layers propagating from the outlet ($x = 2$, where the boundary condition transitions from no-slip Dirichlet walls to traction-free Neumann) were destroying global convergence, we performed a spatial sweep of the error evaluation domain:
[`scratch/test_outlet_exclusion.jl`](file:///Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/scratch/test_outlet_exclusion.jl)

We calculated the errors by integrating only over $x \in [0, 2.0 - \delta]$ for varying truncation values $\delta \in \{0.0, 0.25, 0.5, 0.75\}$:

| Truncation Parameter $\delta$ | Domain Range | Velocity Slope ($10 \to 20$) | Velocity Slope ($20 \to 40$) | Pressure Slope ($10 \to 20$) | Pressure Slope ($20 \to 40$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\delta = 0.0$ (Full) | $x \in [0, 2.0]$ | 0.1457 | 0.5385 | 0.1146 | 0.4600 |
| $\delta = 0.25$ | $x \in [0, 1.75]$ | 0.1433 | 0.5327 | 0.1146 | 0.4600 |
| $\delta = 0.50$ | $x \in [0, 1.50]$ | 0.1407 | 0.5257 | 0.1146 | 0.4600 |
| $\delta = 0.75$ | $x \in [0, 1.25]$ | 0.1378 | 0.5171 | 0.1147 | 0.4600 |

### Conclusion of Phase 2
Excluding the outlet region has **no significant impact** on the convergence slopes. The sub-optimal convergence rate persists globally across the domain and is not a localized boundary effect near the outlet.

---

## 4. Investigation Phase 3: Pure Navier-Stokes Limit ($\alpha = 1.0$ everywhere)
To isolate whether the sub-optimal convergence is caused by the Navier-Stokes core solver (including high Reynolds number convection and boundary/corner singularities) or specifically by the introduction of the porous media terms, we ran a verification sweep where the problem is reduced to pure Navier-Stokes:
- The porosity field is set to a constant $\alpha(y) = 1.0$ everywhere.
- The Darcy-Brinkman-Forchheimer drag coefficients are zeroed: $\sigma_{\text{linear}} = 0$ and $\sigma_{\text{nonlinear}} = 0$.
- All other parameters ($\text{Re} = 500$, domain geometry, inlet profile, and Neumann outlet) are kept exactly the same.

We used the script [`scratch/test_alpha_one.jl`](file:///Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/scratch/test_alpha_one.jl) to evaluate the convergence on coarse meshes ($N \in \{10, 20, 40\}$) against the reference solution ($N_{\text{ref}} = 80$).

### Verification Results
Under these pure Navier-Stokes conditions, the convergence rates are:

| Field | $N = 10$ Error | $N = 20$ Error | N = 40 Error | Slope ($10 \to 20$) | Slope ($20 \to 40$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Velocity $L^2$** | 1.288e-3 | 2.134e-4 | 4.070e-5 | **2.5932** | **2.3909** |
| **Pressure $L^2$** | 1.556e-4 | 4.643e-5 | 2.064e-5 | **1.7442** | **1.1701** |

### Conclusion of Phase 3
The results demonstrate a **dramatic recovery** of convergence rates:
- The velocity convergence slope jumps from $\approx 0.14 - 0.53$ to $\approx 2.4 - 2.6$ (approaching the optimal $O(h^3)$ for Taylor-Hood $P_2$ elements in $L^2$).
- The pressure convergence slope jumps from $\approx 0.11 - 0.46$ to $\approx 1.2 - 1.7$ (approaching the optimal $O(h^2)$ for Taylor-Hood $P_1$ elements in $L^2$).

This comparison provides two critical insights:
1. **Solver and Boundary Conditions are Mathematically Sound**: The core formulation, stabilization (ASGS), element spaces, and Neumann boundary conditions are correct, and they do not inherently prevent near-optimal convergence.
2. **Porous Drag Terms are the Source of the Sub-optimality**: The introduction of the varying porosity field $\alpha(y)$ and/or the Darcy-Forchheimer drag terms is the primary cause of the convergence degradation observed on coarse grids. This is likely due to the modeling error of the porosity field $\alpha_h$ on coarse meshes or the strong localized velocity gradients induced by the varying porous drag.

---

## 5. Key Hypotheses for Future Investigations
Since the numerical metrics are verified and the outlet boundaries are not the culprit, the sub-optimal convergence is likely caused by one or more of the following global physical/formulation interactions:

1. **Porosity Interpolation Model Error**:
   - The varying porosity field $\alpha(y)$ is interpolated using the finite element velocity space order. The difference between $\alpha_h$ (coarse mesh) and $\alpha_{\text{ref}}$ (fine reference mesh) introduces a global modeling error in the Darcy/Forchheimer reaction terms $\alpha \boldsymbol{u}$ and $\beta |\boldsymbol{u}|\boldsymbol{u}$.
   - If this modeling error does not scale sufficiently fast or dominates the coarse-grid solution errors, it can cause a plateau in the convergence rates.

2. **High Reynolds Number and Stabilization Interactions**:
   - At $\text{Re} = 500$, the nonlinear convective term is highly active. The interaction of high convective flows with a transversally varying porosity field $\alpha(y)$ can create thin boundary layer gradients near the walls that require a much finer mesh (e.g., $N > 40$) to enter the asymptotic convergence regime.
   - The stabilization parameters $\tau_1, \tau_2$ might need to be adjusted to account for the varying media coefficients to restore optimal convergence under high Reynolds numbers.

3. **Pressure Pinning and Boundary Singularities**:
   - Because the outlet uses a Neumann boundary condition, the pressure is determined up to an additive constant which is pinned by the traction condition. However, at the corners where Dirichlet walls meet the Neumann outlet, there is a physical discontinuity in the boundary operator, which can produce weak singularities in the derivatives that affect the global solution quality.

---

## 6. Investigation Phase 5: Decisive MMS Test with the *Cocquet Formulation* (RESOLVED)

The hypotheses above conflate two different things: a possible **formulation/implementation defect** versus a **physical/measurement** effect. To separate them we built a Method-of-Manufactured-Solutions test whose governing operator is **identical to Cocquet's** — same `SymmetricGradient` viscous term and same nonlinear, porosity-dependent **Forchheimer–Ergun** reaction (`σ_linear = 0.30`, `σ_nonlinear = 1.75`), Taylor–Hood P2/P1, ASGS — but with a smooth closed-form exact solution and clean Dirichlet data. This combination (nonlinear, α-dependent σ together with a varying α) had **never** been exercised by MMS before: the oracle in `src/problems/mms_paper.jl` previously errored on any non-constant reaction law.

- The shared manufactured-solution oracle was **generalized** to dispatch on the reaction law (constant σ *and* nonlinear Forchheimer-Ergun) in addition to the viscous operator it already dispatched on: `src/problems/mms_paper.jl`. The standard MMS suite (`test/extended/ManufacturedSolutions/`) is unchanged — it still runs the constant-σ path.
- New self-contained test: [`test/extended/CocquetFormMMS/run_test.jl`](../../test/extended/CocquetFormMMS/run_test.jl) with its designed configs `data/cocquet_form_mms_{vms,vms_k2,taylorhood}.json` (the pre-redesign `cocquet_form_mms.json` was removed 2026-07-08).
- Full write-up: [`theory/cocquet/cocquet_form_mms_manufactured_solution.tex`](../../theory/cocquet/cocquet_form_mms_manufactured_solution.tex).
- The Cocquet experiment itself was **not** modified.

### Results ($N \in \{10,20,40,80\}$, fitted $L^2$ slopes)

| $(\alpha_0, \text{Re})$ | rate $u_{L^2}$ | rate $p_{L^2}$ | rate $u_{H^1}$ |
| :--- | :--- | :--- | :--- |
| $(1.0, 1)$ | 3.31 | 2.02 | 2.17 |
| $(1.0, 100)$ | 3.24 | 2.02 | 2.13 |
| $(1.0, 500)$ | 2.94 | 2.03 | 1.90 |
| $(0.5, 1)$ | 2.93 | 2.68 | 1.86 |
| $(0.5, 100)$ | 2.70 | 2.03 | 1.70 |
| $(0.5, 500)$ | 2.80 | 2.03 | 1.76 |
| $(0.05, 1)$ | 2.87 | 2.69 | 1.79 |
| $(0.05, 100)$ | 2.35 | 1.71 | 1.38 |
| $(0.05, 500)$ | 2.39 | 1.99 | 1.40 |

Per-refinement velocity slopes **increase toward the optimal 3** in every case (e.g. the stiffest run $\alpha_0=0.05$, $\text{Re}=500$: $1.97 \to 2.54 \to 2.60$; the pure-NS control $\alpha_0=1$, $\text{Re}=500$: flat at $\approx 2.9$). There is mild pre-asymptotic erosion on coarse meshes that grows with drag strength (small $\alpha_0$) and Reynolds number, but **no fixed sub-optimal order ceiling**.

### Conclusion (narrows the search)
The **Cocquet formulation is correct**: an identical governing operator on a smooth solution recovers $O(h^3)$ velocity / $O(h^2)$ pressure. The benchmark's catastrophic $\approx 0.1$–$0.5$ slopes are therefore **not** a formulation/implementation defect in the discretized PDE. The cause must be something specific to the Cocquet *run* that the MMS does not share. **This phase did not yet identify it** — an early draft of this conclusion blamed corner singularities; that was wrong (see Phase 6). It correctly excluded a formulation bug and the porosity-interpolation hypothesis (the MMS interpolates α at the same order and still converges optimally).

---

## 7. Investigation Phase 6: Root cause found — mesh-dependent velocity floor in the reaction (RESOLVED)

Comparing against the source paper (Cocquet et al., 2020; `scratch/cocquet_paper.txt`) was decisive. The paper runs the **identical** $(\mathrm{Re}, c_{in}) = (500, 0.5)$ case, proves the solution is $H^3(\Omega)\times H^2(\Omega)$ (so there is **no** destructive corner singularity), and reports **clean $O(h^2)$** convergence (the rate is 2 rather than 3 only because they interpolate the porosity at $P_1$). So at identical parameters the paper gets slope 2, our CocquetFormMMS gets 3, but our Cocquet run got 0.1–0.5 — pointing at a setup difference, not physics.

**The difference:** our Cocquet path builds the reaction speed through `effective_speed`, which carries a **mesh-dependent diffusive floor** `u_floor = u_base + h_floor_weight·c₁ν/(c₂h)` with `h_floor_weight = 0.1` (inherited from `base_config.json`). That floor scales like $\nu/h$ and **grows under refinement**:

| N | h | u_floor | vs \|u\|_max≈0.125 |
| :-- | :-- | :-- | :-- |
| 10 | 0.10 | 0.016 | 13% |
| 20 | 0.05 | 0.032 | 26% |
| 40 | 0.025 | 0.064 | 51% |
| **80 (reference)** | 0.0125 | **0.128** | **>100%** |

Injected into the Forchheimer drag $\beta(\alpha)\,|u|_{\text{eff}}\,u$, this makes every mesh solve a *different* effective drag law, and the $N=80$ reference (floor $>$ physical speed) is the most corrupted — destroying the convergence measurement. The paper has no such term (plain Galerkin Taylor–Hood); the MMS sets `h_floor_weight = 0`; pure NS ($\alpha=1$) has $\sigma=0$ so the floor is irrelevant — which is exactly why all three converge and the Cocquet run did not.

### Controlled verification (same script, reference, metric; N∈{10,20,40} vs ref N=80)

| Quantity | baseline `h_floor_weight=0.1` | `h_floor_weight=0` |
| :-- | :-- | :-- |
| velocity $L^2$ slopes | 0.14, 0.53 | **2.80, 2.32** |
| velocity $H^1$ slopes | 0.14, 0.49 | **1.94, 2.03** (optimal) |
| pressure $L^2$ slopes | 0.12, 0.46 | 1.06, 1.47 |
| velocity $L^2$ error @ N=10 | 9.1e-3 | **1.1e-4** (~85× smaller) |

### Fix
The diffusive floor is a τ-stabilization device and has no place in the *physical* reaction. The reaction speed now uses a dedicated `reaction_speed` (constant floor `u_base_floor_ref` only; `src/models/regularization.jl`), wired into `SigOp`/`DSigOp` (`src/formulations/continuous_problem.jl`); `τ` still uses `effective_speed`. The Cocquet config additionally sets `h_floor_weight = 0`. The constant-σ MMS path is unaffected (`ConstantSigmaLaw` ignores the speed), and the Forchheimer MMS is unchanged because `reaction_speed ≡ effective_speed` when `h_floor_weight = 0`.

The residual second-order item (mild high-Re/strong-drag pre-asymptotic erosion seen in the CocquetFormMMS table) and the pressure rate being reference-limited at $N=80$ (the paper used $N=200$) remain as optional measurement refinements.

---

## 8. Investigation Phase 7: Stabilized equal-order comparison with the paper (RESOLVED)

With the fix in place, we reproduced the paper's study at its own mesh sizes — reference **N=200**, coarse **N ∈ {10,20,40,80}** ($h=\sqrt2/N$) — for the **stabilized equal-order** pairs P1/P1 and P2/P2 (the VMS method's purpose) plus Taylor–Hood P2/P1. Driver: `run_convergence.jl data/structured/paper_comparison.json` (element-pair sweep). All nonlinear solves converged (4 iterations each) at Re=500.

| Element pair | velocity $L^2$ slopes | velocity $H^1$ slopes | pressure $L^2$ slopes |
| :-- | :-- | :-- | :-- |
| P1/P1 (stab.) | 1.86, 1.92, 2.08 → $O(h^2)$ | 0.98, 1.02, 0.91 → $O(h)$ | 1.86, 2.38, 1.25 |
| P2/P2 (stab.) | 2.68, 1.69, 1.51 | 1.94, 1.99, 1.98 → $O(h^2)$ | 0.68, 1.01, 1.44 |
| P2/P1 (Taylor–Hood) | 2.74, 1.86, 1.53 | 1.94, 1.99, 1.98 → $O(h^2)$ | 0.91, 1.08, 1.45 |

The stabilized equal-order elements achieve **optimal energy-norm rates**: P1/P1 gives $O(h)$ velocity $H^1$; P2/P2 gives $O(h^2)$ velocity $H^1$ — matching the paper's reported $O(h^2)$ total error for Taylor–Hood. The velocity-$H^1$ slopes are clean and robust. The velocity-$L^2$ slope decays at the fine end (P2/P2: $2.68 \to 1.69 \to 1.51$); a finer reference ($N=320$) did **not** lift it, so it is **not** a reference-resolution effect — the cause is investigated in Phase 8. Results in `results/structured/convergence.h5`.

---

## 9. Investigation Phase 8: Why P2/P2 velocity $L^2$ is below the optimal $O(h^3)$

Three controlled experiments separate the candidate causes (metric, discretization, boundary conditions). **A wrong intermediate hypothesis ("the cross-mesh metric has a resolution floor") was disproved here** and is recorded so it is not repeated.

**(a) The cross-mesh metric is exact.** `test/quick/interpolation_projection_quick_test.jl` interpolates a smooth analytic field onto coarse meshes and the $N=200$ reference, then measures the *consistent* cross-mesh $L^2$ error $\lVert u_{\text{ref}}-u_h\rVert$ via the same `compute_reference_errors` the benchmark uses. For **P2** it returns slope **3.00, 3.01** and matches the true interpolation error to 4 significant figures down to $\sim 2.5\times10^{-7}$ — *no floor, no distortion*. The metric is therefore not the cause.

**(b) The discretization reaches optimal $L^2$ on a Dirichlet problem.** The exact-solution MMS `test/extended/CocquetFormMMS` run with **equal-order P2/P2** (Re=500, $\alpha_0=0.5$, Dirichlet everywhere, error vs the analytic field — no reference) gives velocity $L^2$ slopes $2.39 \to 2.77 \to \mathbf{2.86}$ (climbing toward the optimal 3) and $H^1 \to 1.92$. So the solver/formulation *can* deliver $O(h^3)$ velocity $L^2$.

**(c) The benchmark differs by its boundary conditions.** The benchmark uses mixed inlet-Dirichlet / no-slip-wall / **traction-free Neumann outlet** conditions; the MMS in (b) uses Dirichlet everywhere. Since the metric is exact (a) and the Dirichlet problem is optimal (b), the benchmark's reduced velocity $L^2$ ($\approx 2$–$2.6$) is attributed to the **mixed boundary conditions**: the traction-free outlet and the wall$\leftrightarrow$outlet corner weaken the elliptic-duality (Aubin–Nitsche) regularity that the $L^2$ extra order requires. This is consistent with **Cocquet et al. independently reporting velocity $L^2 = O(h^2)$** (not $h^3$) for exactly this problem; they attribute the cap to the $P_1$ porosity, but the BC-regularity mechanism applies regardless. Velocity $H^1$ stays optimal because the energy-norm rate does not rely on that duality.

**Status of (c): documented possibility, not yet proven.** The clean confirming experiment would be a **Neumann-outlet MMS** — manufacture a solution that satisfies the traction-free condition and measure against the exact field; if velocity $L^2$ then drops from $\approx 3$ (Dirichlet MMS) to $\approx 2$, the BC attribution is established. This is left as the next step. The earlier per-step "L² slope < H¹ slope" observation does **not** by itself prove a measurement defect (Poincaré bounds the L² *value*, not its rate), so it is not relied upon.

**Practical takeaway:** the equal-order method converges optimally in the energy ($H^1$) norm, which is the rate the VMS theory guarantees; the velocity-$L^2$ extra order is realized for smooth Dirichlet problems and is reduced — genuinely, matching the literature — by the benchmark's mixed boundary conditions.

---

## 10. Investigation Phase 9: Root cause PROVEN — Dirichlet/Neumann corner singularity at the outlet

> **⚠ VERDICT WITHDRAWN — see Phase 11.** The "PROVEN, mesh- and method-independent corner singularity" reading below rests on the structured-mesh artifact; Phase 11 shows an unstructured Delaunay mesh recovers $O(h^2)$. The corner *does* dominate the structured cap, but the cap is not a fixed regularity ceiling. The settled conclusion is mesh-topology.

Phase 8 left the boundary-condition mechanism as a "documented possibility, not yet proven." Phase 9 proves it directly — by a more decisive route than the Neumann-outlet MMS proposed there — using a ladder of **single-variable flips off the benchmark**, plus **spatial localization of the error**.

### The test ladder

Each test changes exactly one thing relative to `CocquetTubeTest/data/structured` (the benchmark: varying porosity interpolated at order $k_v$, Forchheimer reaction, SymmetricGradient viscous, mixed BC, self-reference vs $N=200$):

| Test (directory) | The one change | Isolates |
| :-- | :-- | :-- |
| `CocquetTubeTest/data/alpha_one` | porosity $\alpha\equiv1$ (⇒ plain NS: $\sigma=0$, no porosity weighting, porosity exact) | the entire generalized formulation + quadrature |
| `CocquetTubeTest/data/all_dirichlet` (Flip B) | outlet → Dirichlet (inlet parabola; corners become Dirichlet–Dirichlet) | the mixed-BC corner |
| `CocquetTubeTest/data/deviatoric` | viscous → DeviatoricSymmetric (the MMS/canonical operator) | the SymmetricGradient outlet traction specifically |
| `CocquetTubeTest/data/linear_reaction` | `sigma_nonlinear = 0` (Forchheimer $|u|$ term off) | the nonlinear reaction |

### Decisive result 1 — α≡1 (plain Navier–Stokes) is STILL capped

`alpha_one` (mixed BC, $\alpha\equiv1$, self-reference, $N\in\{10,20,40,80\}$ vs ref $N=200$):

| Pair | velocity $L^2$ slopes | velocity $H^1$ slopes |
| :-- | :-- | :-- |
| P1/P1 | 1.78, 1.85, 1.91 → $\approx O(h^2)$ | 0.81, 0.86, 0.81 → $\approx 0.85$ |
| P2/P2 | **1.45, 1.25, 1.49** → $\approx 1.4$ (capped) | **0.84, 0.76, 0.78** → $\approx 0.78$ (capped) |

P2/P2 is capped at essentially the **same** rates as the benchmark (≈1.3 / ≈0.8) **even though porosity, the porosity-dependent reaction, and the nonlinear drag are all gone**. This **exonerates the generalized formulation** as the cause of the cap. It also **exonerates quadrature**: at $\alpha\equiv1$ the integrands are trivial polynomials (no rational $((1-\alpha)/\alpha)^2$, no $|u|$), which a degree-9 rule over-resolves, yet P2 is still capped. (Confirmed independently: bumping the benchmark's quadrature degree $9\to21$ left P1/P1 bit-identical, $1.82/1.83/1.86$.)

### Decisive result 2 — the error is localized at the outlet corners

Spatial localization of the **α≡1, P2/P2, N=80** error field (`results/cocquet_ASGS_P2P2_N80.vtu`, fields `e_u`, `e_p`), binning $\lVert e\rVert^2$ by distance to the two wall/outlet corners $(2,0),(2,1)$:

| Field / pair | $\lVert e\rVert^2$ within $R=0.1$ of outlet corners (corner region = 1.6% of area) |
| :-- | :-- |
| velocity, P1/P1 | 37% |
| velocity, **P2/P2** | **50%** |
| pressure, P1/P1 | 72% |
| pressure, **P2/P2** | **98%** |
| any field, inlet corners $(0,0),(0,1)$ | **0%** |

The maximum $|e_u|$ sits exactly at the nodes adjacent to $(2,1)$ and $(2,0)$. The error is overwhelmingly at the **outlet** corners — where no-slip Dirichlet meets the traction-free Neumann outlet — and **absent at the inlet** corners (Dirichlet–Dirichlet, compatible data, since the inflow parabola vanishes at $y=0,1$). The pressure is the most singular field ($p\sim r^{s-1}$): for P2 essentially **all** (98%) of the pressure error is at the corners.

### Why only quadratics suffer — the masking effect (uses the P1-vs-P2 clue)

A corner singularity $u\in H^{1+s}$ caps the rate at $\approx s$ (energy) / $\approx 2s$ ($L^2$) **independent of element order**; the observed $H^1\approx0.8$ for *both* P1 and P2 pins $s\approx0.8$. The corner contribution to the global error is a fixed $O(h^{2s})$. For **P1** the interior approximation error is $O(h^2)$ — large enough to dominate, so the corner is **masked** and P1 looks near-optimal (only 37% of its error is at the corner). For **P2** the interior error is $O(h^3)$ — so small that the $O(h^{2s})$ corner term takes over (50% of velocity, 98% of pressure), collapsing the global rate. *"Only quadratics are suboptimal" is therefore the signature of a fixed regularity ceiling, not an order-specific bug* — and `CocquetFormMMS` running P2 optimally (Phase 8b) independently rules out any P2-specific assembly/quadrature defect.

### Reconciliation with Cocquet et al.

The paper reports optimal $O(h^2)$ for Taylor–Hood P2/P1. Our equal-order method appears **more corner-sensitive**, and the 98%-pressure-at-corner figure points to why: the equal-order scheme **stabilizes a singular pressure** (the $\tau_2$ mass term), whereas inf-sup-stable Taylor–Hood imposes no pressure stabilization. The corner pressure singularity therefore couples into the equal-order velocity more strongly. (A factor-$\sqrt2$ inconsistency in the triangle $h$ definition between this driver, `√(area)`, and the MMS driver, `√(2·area)`, rescales $\tau$ by a constant and cannot change a slope; noted for cleanup, not a cause.)

### Anticipated diagnostics by Flip-test outcome

The two running flips are now **confirmatory**. Expected outcomes and what each would establish:

**`all_dirichlet` (Flip B — outlet made Dirichlet, corner removed):**
- **P2/P2 recovers toward $O(h^3)$ (L²) / $O(h^2)$ (H¹)** → *expected, and definitive*: with porosity/reaction held at the benchmark values and only the corner removed, recovery proves the mixed-BC corner is the sole cause. Combined with α≡1, the chain is closed: corner, independent of the generalized formulation.
- **P2/P2 stays $\approx1.4$** → would be surprising and would *contradict* the spatial-localization evidence; it would force re-opening the **interpolated-porosity** hypothesis (Flip B keeps interpolated varying $\alpha_h$ + self-reference, whereas the optimal `CocquetFormMMS` has exact $\alpha$ + exact solution). Next step then = **Flip A** (exact porosity, mixed BC) and a self-reference-vs-exact comparison. (Low prior, given the error is demonstrably at the corner in plain NS.)

**`deviatoric` (viscous → DeviatoricSymmetric, mixed BC kept):**
- **P2/P2 recovers** → the **SymmetricGradient outlet traction** $(2\nu S(u)-pI)\,n=0$ specifically aggravates the corner; the deviatoric/"do-nothing" outflow has a milder (or absent) singularity. *Practical fix available*: change the outlet formulation.
- **P2/P2 stays capped** → the corner is **operator-independent** (any Dirichlet/Neumann right-angle outlet produces it); only removing the Neumann outlet (Flip B) or grading/rounding the corner helps.

**`linear_reaction` ($\sigma_{nl}=0$):** expected **unchanged** ($\approx1.4$) → nonlinear reaction contributes nothing to the cap. A change would flag an interaction between the $|u|$ kink (non-smooth where $u=0$ on walls) and the corner.

### Practical remedies (if optimal benchmark P2 rates are desired)

None of these are bug fixes — they are choices about geometry/measurement: **(i)** corner-graded mesh refinement to resolve the singularity; **(ii)** measure the norm away from the corner (the `outlet_truncation_delta>0` truncation did exactly this); **(iii)** round/chamfer the outlet corners; **(iv)** if Deviatoric recovers, adopt the deviatoric/do-nothing outflow. The energy-norm ($H^1$) rate the VMS theory guarantees is already met; the velocity-$L^2$ extra order is genuinely BC-limited.

### Confirmatory flip results

All **mixed-BC** variants ($N\in\{10,20,40,80\}$ vs ref $N=200$) — regardless of porosity, reaction, or viscous operator — collapse to the **same** P2/P2 cap, while P1/P1 stays near-optimal:

| Test | one change vs benchmark | P2/P2 velocity $L^2$ slopes | P2/P2 velocity $H^1$ |
| :-- | :-- | :-- | :-- |
| `structured` (benchmark) | — | 1.39, 1.13, 1.43 | ≈0.8 |
| `alpha_one` | $\alpha\equiv1$ (plain NS) | 1.45, 1.25, 1.49 | 0.84, 0.76, 0.78 |
| `deviatoric` | viscous → DeviatoricSymmetric | 1.26, 1.11, 1.44 | 0.78, 0.75, 0.79 |
| `linear_reaction` | $\sigma_{nl}=0$ | 1.25, 1.14, 1.46 | 0.80, 0.77, 0.79 |

(All P1/P1 stay $\approx1.8$–$1.9$ in $L^2$ / $\approx0.85$ in $H^1$.) Interpretation per the decision tree:
- **Deviatoric stays capped** ⇒ the corner is **operator-independent** — not specific to the SymmetricGradient outlet traction. Changing the outlet *operator* cannot fix it.
- **LinearReaction unchanged** ⇒ the nonlinear Forchheimer $|u|$ term contributes nothing to the cap.
- **α≡1 capped** ⇒ the entire generalized formulation (and quadrature) is irrelevant to the cap.

Every mixed-BC case is pinned at P2/P2 $L^2\approx1.3$–$1.4$, consistent with one shared mechanism: the outlet Dirichlet/Neumann corner singularity.

### The clincher — removing the corner recovers P2

`all_dirichlet` (Flip B — outlet made Dirichlet with the inlet parabola, so the wall/outlet corners become Dirichlet–Dirichlet; **interpolated varying porosity and Forchheimer reaction kept at benchmark values**):

| Pair | velocity $L^2$ slopes | velocity $H^1$ slopes | pressure $L^2$ slopes |
| :-- | :-- | :-- | :-- |
| P1/P1 | 1.52, 1.69, 1.98 → $O(h^2)$ | 0.57, 0.77, 0.86 → $O(h)$ | 1.73, 1.76, 1.94 |
| **P2/P2** | **1.99, 2.53, 2.84** → $O(h^3)$ | **1.12, 1.60, 1.86** → $O(h^2)$ | 1.56, 1.79, 1.90 |

P2/P2 **recovers** ($L^2: 1.99\to2.53\to2.84$, $H^1: 1.12\to1.60\to1.86$), versus the flat $\approx1.3$/$\approx0.8$ of *every* mixed-BC variant. Since the **only** change from the benchmark is the outlet boundary condition, the corner attribution is **fully closed**.

**Important nuance (the recovery is pre-asymptotic, not yet clean-optimal).** The all-Dirichlet slopes *accelerate* (they climb toward $3$/$2$ but reach only $2.84$/$1.86$ at $N=80$), and the coarse-mesh errors are actually *higher* than the benchmark's (e.g. $N=10$: $7.7\times10^{-4}$ vs $2.3\times10^{-4}$) — the imposed double-parabola flow has features the coarse mesh resolves poorly. This pre-asymptotic behaviour is **shared with the exact-solution `CocquetFormMMS`** (P2 $L^2: 2.39\to2.77\to2.86$, $H^1\to1.92$ — also climbing, not flat at $3$/$2$): at Re=500 on $N\le80$ meshes, P2 is simply not in the asymptotic regime even for a smooth Dirichlet problem. So the robust, mechanism-level finding is the **qualitative** contrast — mixed-BC P2 is *stuck flat and never improves* (a hard regularity ceiling), all-Dirichlet P2 *climbs steeply toward optimal*. Demonstrating *clean* $O(h^3)$/$O(h^2)$ would require finer meshes ($N=160$+); the mechanism (corner) is nonetheless conclusive from the flat-vs-climbing contrast + the corner error-localization + operator/reaction-independence.

**Status: RESOLVED.** The benchmark's P2/P2 sub-optimality is a **Dirichlet/Neumann corner singularity at the traction outlet**, proven by four independent lines of evidence:
1. **α≡1 (plain NS) is still capped** → not the generalized formulation, not quadrature.
2. **Error localization** → 50% of velocity / 98% of pressure error within $R=0.1$ of the outlet corners (0% at the compatible inlet corners), in plain NS.
3. **Operator/reaction flips stay capped** → `deviatoric` (≈1.3) and `linear_reaction` (≈1.3) show the cap is independent of viscous operator and nonlinear reaction.
4. **Removing the corner recovers it** → `all_dirichlet` P2/P2 $L^2\to2.84$.

It bites only quadratics because a fixed regularity ceiling ($s\approx0.8$) is masked by P1's larger interior error but exposed by P2's much smaller one. The energy-norm ($H^1$) rate the VMS theory guarantees is met on every variant; the velocity-$L^2$ extra order is genuinely boundary-limited, matching Cocquet et al.'s own observation. Remedies (corner-graded mesh, measuring away from the corner via `outlet_truncation_delta>0`, or rounding the corner) are modelling/measurement choices, not bug fixes. Associated tests: `test/extended/CocquetTubeTest/data/{alpha_one,deviatoric,linear_reaction,all_dirichlet}`.

**Mesh-size (`h`) consistency fix.** All Cocquet drivers now compute the τ characteristic length as $\sqrt{2\cdot\text{area}}=1/N$ for triangles (the structured mesh parameter), matching `test/extended/CocquetFormMMS`. Previously this driver used $\sqrt{\text{area}}=1/(N\sqrt2)$ — a factor $\sqrt2$ too small, which only rescaled τ by a constant (rate-neutral; all slopes above are unaffected) but made the Cocquet and MMS drivers inconsistent. Cocquet et al. report the triangle *diameter* $h=\sqrt2/N$ (a further $\sqrt2$); since our convergence slopes are measured against $N$, the convention does not enter the rates.

**Open confirmation item.** A finer-mesh run ($N$ up to $160$) of `all_dirichlet` (and `CocquetFormMMS`) would verify the pre-asymptotic interpretation by showing P2 $H^1\to2$, $L^2\to3$. Until then, "optimal recovery" is asserted as a *trend*, not a measured plateau.

---

## 11. Investigation Phase 10: the EXACT Cocquet method (unstabilized Galerkin Taylor–Hood), and a full code audit

> **⚠ VERDICT WITHDRAWN — see Phase 11.** Phase 10's twin conclusions — "the cap is a method-independent corner singularity" and "the paper's $H^3{\times}H^2$ regularity claim is incorrect" — are **withdrawn**. Both were drawn entirely on the structured mesh; Phase 11 recovers $O(h^2)$ on an unstructured Delaunay mesh. The settled conclusion is mesh-topology.

> **⚠ Note (2026-05-26):** Phase 10 below concludes that the convergence cap is the mixed-BC outlet corner singularity and is method-independent. This conclusion was drawn entirely on a **structured Cartesian-simplexified** mesh. **Phase 11 (further down) revises it:** on an unstructured FreeFem-style mesh the velocity-L² cap relaxes substantially and slopes recover toward Cocquet's reported O(h²). The "method-independent corner singularity" reading of Phase 10 is therefore a **structured-mesh artefact**; the corner DOES dominate the cap, but on a mesh whose triangulation is not corner-aligned the dominance weakens. Read Phase 11 before relying on Phase 10's conclusions. See also [cocquet_investigation_synthesis.md](investigation-synthesis.md) §5.3 / §5.7 for the post-S5-audit picture.

To test whether the cap is our VMS formulation or the problem, we implemented Cocquet et al.'s *exact* discrete method — inf-sup-stable **Taylor–Hood $P_2/P_1$ with no stabilization** (pure Galerkin), $P_1$ porosity, $\eta p$ penalty — and ran it next to our VMS $P_1/P_1$ and $P_2/P_2$ in one figure. The unstabilized assembly reuses the production weak-form builders with `mult_mom=mult_mass=0` and lives in `test/extended/CocquetTubeTest/galerkin_driver.jl` (validation tooling; `src/` is untouched). See `theory/cocquet/cocquet_formulation.tex` for the documented formulation and the code map.

### Result — the exact Cocquet method is corner-capped too

Three-way run (ref $N=200$, coarse $\{10,20,40,80\}$, after the $h$ fix):

| Curve | velocity $L^2$ slopes | velocity $H^1$ slopes |
| :-- | :-- | :-- |
| VMS ASGS P1/P1 | 1.72, 1.79, 1.96 → $O(h^2)$ | 0.89, 0.91, 0.84 → $O(h)$ |
| VMS ASGS P2/P2 | 1.69, 1.66, 1.58 | 0.74, 0.73, 0.73 |
| **Cocquet Galerkin P2/P1** | **1.10, 1.13, 1.47** | **0.73, 0.74, 0.78** |

The exact Cocquet Galerkin $P_2/P_1$ is capped at $H^1\approx0.75$ (optimal 2) and $L^2\approx1.2$ (optimal 3) — essentially identical to our VMS $P_2/P_2$. Error localization (`cocquet_Galerkin_P2P1_N80.vtu`): **42 % of velocity / 95 % of pressure error within $R=0.1$ of the outlet corners, 0 % at the inlet**, max at $(2,0.988)$. Interior-restricted ($x<1$) velocity slopes climb to $\sim 2.2$, so there is **no interior defect** — the cap is purely the corner. **Switching off all stabilization and using inf-sup-stable Taylor–Hood does not escape the cap: it is the problem (the corner), not our method.**

### Code audit (every assumption the MMS does not exercise)

Because `CocquetFormMMS` validates the volume operators but is all-Dirichlet (no traction outlet), we separately verified everything the benchmark relies on and the MMS does not:

| Assumption | Method | Result |
| :-- | :-- | :-- |
| Boundary tags inlet/outlet/walls | empirical Gridap probe on $[0,2]\times[0,1]$ | inlet $=\{x=0\}$, outlet $=\{x=2\}$, walls $=\{y=0\}\cup\{y=1\}$ — correct |
| Outlet **natural BC** | derived from the Galerkin weak form | $\alpha(2\nu\Sten(u)-pI)\nn=0$ — matches Cocquet Eq. 2 |
| Viscous form | read `viscous_operators.jl` | $2\nu\alpha\,\Sten(u){:}\Sten(v)$ — correct |
| Convective form | non-conservative $\int\alpha(u\cdot\grad)u\cdot v$ | matches Cocquet's $c(\varepsilon;\cdot)$ |
| Reaction $\alpha(\varepsilon),\beta(\varepsilon)$ | $\sigma_{\rm lin}=150/\Reyn$, $\sigma_{\rm nl}=1.75$ | matches Eq. 49 |
| Galerkin solve convergence | run log (`:ok`, no fallback) | converged |
| Cross-mesh metric | prior smooth-field test (slope 3.00) | exact |
| Error is genuine, not artifact | VTK localization + interior slopes | genuine corner singularity |

**No bug was found** in the formulation or the reproduction; the corner singularity is real and method-independent.

### Why the paper reports $O(h^2)$ and we do not — and how Cocquet "dealt with" the corner

**How the paper handles the corner (the clue):** Cocquet do not report a corner singularity because their analysis concludes there is *none*. In §4.2 they argue the weak solution is **$H^3(\Omega)\times H^2(\Omega)$**, treating the wall/outlet junction through the Lions–Magenes traction space **$H^{1/2}_{00}(\Gamma_{\rm out})$** (which builds in corner compatibility — test/traction functions vanish at $\partial\Gamma_{\rm out}$) and invoking the mixed-BC Stokes regularity result **[27]**. They also note $u_{\rm in}=c_{\rm in}y(1-y)$ vanishes at the inlet corners, making those compatible. So they "dealt with" the corner by *arguing it away*: if the solution is $H^3$, Taylor–Hood $P_2/P_1$ is optimal $O(h^2)$.

**Why that $H^3$ claim is the weak link.** The outlet corners are **no-slip (full Dirichlet) / traction-free (full Neumann) $90^\circ$** junctions for the Stokes operator. Regularity theory for this corner type (Orlt–Sändig / Maz'ya–Rossmann class) gives a **leading singular exponent $s<1$** — the solution is generally not even $H^2$. Our measured $s\approx0.8$ (from the $H^1$-rate cap, identical for P1 and P2) sits squarely in that range, and the error localizes at exactly these corners (0% at the compatible inlet corners). A genuine corner singularity is shared by *any* conforming FE method on this weak form — including Cocquet's FreeFem Taylor–Hood — and our bare-Galerkin reproduction reproduces it identically.

**Artifact hypotheses ruled out.** (i) *Backflow / do-nothing instability*: the outlet velocity has $u_x\ge0$ everywhere ($\min=0$, no recirculation), so the open-boundary convective instability is not present. (ii) *Boundary tagging*: verified empirically (inlet $x{=}0$, outlet $x{=}2$, walls $y{\in}\{0,1\}$). (iii) *Natural BC*: derived from the weak form as $\alpha(2\nu\Sten(u)-pI)\nn=0$, matching Eq. 2. (iv) *VMS-specific bug*: the Galerkin path bypasses all VMS machinery and still caps. So the corner error is a genuine feature of the discrete Cocquet problem, not a reproduction artifact we have identified.

**Decisive evidence — the pressure singularity is real (direct measurement).** A self-reference "cancellation/measurement" story was considered and **rejected**: it cannot explain the discrepancy, because (i) self-reference against an $N=200$ reference that is $2.5\times$ finer does not mask a singularity, and (ii) our *finest* segment ($N{=}40\to80$) is itself capped at $H^1\approx0.78$ — not climbing to 2. The convergence rate depends on the continuous solution's regularity, not on mesh structure, so "structured vs unstructured mesh" cannot bridge a *rate* gap either. We therefore measured the solution's regularity directly. Tracking the corner pressure $\max_{r<0.06}|p|$ vs $N$ for the Galerkin $P_2/P_1$ run:

| $N$ | 10 | 20 | 40 | 80 | 200 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| corner $\max|p|$ | 0.0031 | 0.0048 | 0.0063 | 0.0082 | 0.0115 |
| global $\max|p|$ | 0.0214 | 0.0214 | 0.0214 | 0.0214 | 0.0214 |
| $\max|u|$ | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |

The corner pressure **grows monotonically** as $\sim h^{-0.4}$ while the velocity stays bounded — the unambiguous signature of a genuine pressure singularity $p\sim r^{s-1}$ with $s\approx0.6$. The pressure is in $L^2$ but **not $H^1$** ($\grad p\sim r^{-1.4}\notin L^2$), hence certainly **not $H^2$**. A smooth solution's corner pressure would *converge*; it does not.

**Conclusion (resolved, with direct evidence).** This BVP genuinely has a singular pressure at the no-slip/traction outlet corner. The stagnation is therefore real, mesh- and method-independent — any conforming FE method on this problem (Cocquet's FreeFem Taylor–Hood included) solves the same singular solution and hits the same ceiling. **Cocquet et al.'s $H^3\times H^2$ regularity claim is incorrect for this $90^\circ$ no-slip/traction corner (the pressure is unbounded there), and their reported $O(h^2)$ is not the true asymptotic rate.** A careful paper could still have drawn a slope-2 line because the singularity is *weak and slow*: at $N=200$ the corner pressure ($0.0115$) is still below the global max ($0.0214$), so over their range ($N\le100$) the $s\approx0.6$ singular mode has not yet dominated a velocity-weighted $\mathrm{Err}_{tot}$ — it is pre-asymptotic, and the asymptotic rate is $O(h^s)$. Their $H^{1/2}_{00}(\Gamma_{\rm out})$ traction framework is a *continuous* compatibility device; it does not remove the discrete corner singularity, and the cited regularity theorem [27] does not cover this corner type.

**Remaining (optional) confirmations:** (a) refine the reference to $N=400$ — the corner error and its growth exponent should be unchanged (reference-independent), reconfirming the singularity; (b) a corner-graded mesh should recover $O(h^2)$ in the interior; (c) an analytic corner-exponent computation (Maz'ya–Rossmann) would pin $s$ exactly. None changes the conclusion above.

### Note on the $h$ fix and the Phase 9 tables

The $h$-consistency fix ($\sqrt{\rm area}\to\sqrt{2\,{\rm area}}$) rescales $\tau$ by $\sqrt2$. Asymptotically rate-neutral, but on these *pre-asymptotic* coarse meshes it shifts constants enough to move the measured P2/P2 slopes (e.g. benchmark P2/P2 velocity-$L^2$ went from $\approx1.3$ pre-fix to $\approx1.6$ post-fix). The Phase 8–9 tables above are **pre-fix snapshots** (kept as recorded); the Phase 10 table is post-fix. The qualitative conclusion — P2 capped under mixed BC, recovering only when the corner is removed — is unchanged.

---

## 12. Investigation Phase 11: the cap is largely a STRUCTURED-MESH artifact — unstructured (FreeFem-like) mesh recovers velocity-$L^2 \approx O(h^2)$ (REVISES Phases 9–10)

**A confound the entire Phase 1–10 ladder never tested:** every prior run used a **structured Cartesian mesh simplexified into right triangles**, with edges aligned to the $90^\circ$ no-slip/traction outlet corner, and a **structured $N=200$ self-reference**. Cocquet et al. used **FreeFem `buildmesh`** — an **unstructured Delaunay** mesh. Phase 9–10 concluded the P2 cap was a *mesh- and method-independent* corner singularity (and that "the paper's $H^3\times H^2$ claim is incorrect"). That conclusion was drawn **entirely on the structured family**.

### The test
New self-contained sibling `test/extended/CocquetTubeTest/data/unstructured_gmsh` — a single-variable flip whose **only** change vs the benchmark is the mesh: `PorousNSSolver.build_unstructured_model(N)` (gmsh) with **prescribed boundary divisions** (transfinite curves: inlet $N$, outlet $N$, walls $2N$ → uniform edge $\approx1/N$) and an **unstructured Delaunay interior** — exactly the FreeFem `buildmesh` paradigm (boundary node count fixed, interior irregular). Same weak form, same 3-way comparison (VMS P1/P1, VMS P2/P2, Cocquet Galerkin P2/P1 with P1 porosity), same $(\mathrm{Re},c_{in})=(500,0.5)$, ref $N=200$, coarse $N\in\{10,20,40,80,100\}$, `physical_epsilon`$=10^{-7}$ (the paper's $\eta$, set explicitly). Cross-mesh errors use a tolerant `KDTreeSearch` (the coarse and reference unstructured meshes are non-nested). The benchmark `structured` and all of `src/` behaviour for structured runs are unchanged (the mesh generator defaults to `STRUCTURED`).

### Decisive result — the velocity-$L^2$ cap disappears on the unstructured mesh

| Element pair | quantity | structured (Phase 10) | **unstructured (Phase 11)** | paper Fig. 2 |
| :-- | :-- | :-- | :-- | :-- |
| Cocquet Galerkin P2/P1 | vel $L^2$ slopes | 1.10, 1.13, 1.47 | 1.74, **1.87, 1.98**, 1.80 | $\approx 2$ |
| Cocquet Galerkin P2/P1 | vel $H^1$ slopes | 0.73, 0.74, 0.78 (**flat**) | 0.80, 0.97, **1.15**, 1.12 (**climbing**) | $\approx 2$ |
| VMS P2/P2 | vel $L^2$ slopes | 1.69, 1.66, 1.58 (**decaying**) | 1.60, 1.78, 1.93, 2.44 (**climbing**) | — |
| VMS P2/P2 | vel $H^1$ slopes | 0.74, 0.73, 0.73 (**flat**) | 0.72, 0.92, 1.06, 1.54 (**climbing**) | — |
| VMS P1/P1 | vel $L^2$ slopes | 1.72, 1.79, 1.96 | 1.13, 1.57, 2.00, 2.10 | — |
| VMS P1/P1 | vel $H^1$ slopes | 0.89, 0.91, 0.84 | 0.34, 0.70, 1.01, 1.15 | — |

(Read the clean $20\!\to\!40\!\to\!80$ trend; the finest $80\!\to\!100$ segment sits close to the $N=200$ reference and is deflated/noisy — e.g. Galerkin vel $L^2$ $1.87\!\to\!1.98$ then dips to $1.80$. Results in `test/extended/CocquetTubeTest/results/unstructured_gmsh/convergence.{h5,png}`.)

**The exact Cocquet method's velocity $L^2$ tracks $O(h^2)$ on the unstructured mesh — reproducing the right panel of the paper's Figure 2** — whereas the structured mesh held it at $\approx1.1$–$1.5$ (and the structured P2/P2 *decayed*). The qualitative signature flips completely: on the structured mesh the P2 energy norm was **flat/capped** ($H^1\approx0.73$–$0.78$, supposedly a fixed regularity ceiling); on the unstructured mesh it **climbs** ($H^1: 0.8\to1.15$ for P2/P1; $0.72\to1.54$ for P2/P2).

### What this means for Phases 9–10

- **The "fixed regularity ceiling / genuine mesh- and method-independent corner singularity" claim was substantially a STRUCTURED-MESH ARTIFACT.** A genuine corner singularity caps the *rate* independently of mesh topology; the cap vanishing on an unstructured mesh shows the structured cap was (largely) discrete — most plausibly the diagonal simplexification aligning a spurious pressure mode with the corner node, amplified by a *structured self-reference* that correlates coarse/fine errors. The Phase 10 "corner pressure $\sim h^{-0.4}$" growth was measured only on the structured mesh and should be re-checked on the unstructured one before being read as a continuous-solution singularity.
- **Cocquet et al.'s reported velocity-$L^2 = O(h^2)$ is reproduced** once the mesh matches theirs. The Phase 10 assertion that "their $O(h^2)$ is not the true asymptotic rate / their $H^3\times H^2$ is incorrect" is **withdrawn** as unsupported: it rested on the structured artifact.

### Honest caveats (what is NOT yet fully reproduced)
- **Velocity $H^1$ / $Err_{tot}$ reaches only $\approx1.1$, not the paper's $\approx2$, within $N\le100$.** It *climbs* (vs the structured flat cap) but is not yet at the optimal energy rate. Candidate causes, not yet separated: (i) the **non-nested cross-mesh metric** degrades the $H^1$ (gradient) measurement more than $L^2$ — differentiating a cross-interpolated coarse field on an independent mesh is the least reliable quantity here; (ii) the $O(h)$ porosity-gradient term in the paper's own error estimate (which they argue away empirically); (iii) genuine pre-asymptotics at $\mathrm{Re}=500$ on $N\le100$. The reported `Err_tot` overlay uses the full-gradient $H^1$ seminorm as a proxy for $\lVert S(u)\rVert$.
- The unstructured Delaunay mesh is **not reproducible bit-for-bit** across gmsh versions (pinned `gmsh_jll ~4.9.3`); slopes are measured vs $N$, so this is rate-neutral.

### Status and next steps
**Partially RESOLVED, and Phases 9–10 are revised:** the velocity-$L^2$ cap (the most-emphasized failure) is reproduced as a structured-mesh artifact and **disappears on a FreeFem-like unstructured mesh, matching the paper's Fig. 2 (velocity $L^2 \approx O(h^2)$)**. Open: (a) lift velocity $H^1$/$Err_{tot}$ to $\approx2$ — first rule out the cross-mesh-metric degradation by adding an $S(u)$-based energy norm and/or an exact-solution (Neumann-outlet MMS) check; (b) re-measure corner pressure growth on the unstructured mesh to test whether the Phase 10 singularity signal survives; (c) finer $N$ (160+) with a finer reference to read clean plateaus.

### Correction (added next pass): the unstructured run reproduces the SLOPE but NOT the MAGNITUDE of Fig. 2

Re-reading Fig. 2 against the actual data, the absolute error magnitudes of the unstructured run are **~20× larger** than the paper's (vel $L^2$ @ $N=10$: ours unstructured $5.86\times10^{-3}$ vs the paper's $\sim$ few$\times10^{-4}$). Three diagnostics nail down why this is a *real* feature of the solution, not a measurement bug:

1. **Metric** — the consistent (fine-mesh) and nested (coarse-mesh) cross-mesh errors agree to within 1% on the unstructured run (N=10: $5.77\times10^{-3}$ vs $5.74\times10^{-3}$), so the magnitude is not a `KDTreeSearch`/cross-mesh artifact.
2. **Mesh quality** — the gmsh Delaunay at $N=10$ has $h=\sqrt{2A}\in[0.069,0.112]$, mean $0.088$ (*finer* than the structured leg $0.1$), and ~506 cells (more than the structured 400). Element shape and size are not the cause.
3. **Cross-mesh-family reference test** (decisive). For Galerkin P2/P1, $N=10$ coarse vs $N=80$ ref, all four combinations:

   | coarse \ reference | structured ref | unstructured ref |
   | :-- | :-- | :-- |
   | structured coarse | $2.88\times10^{-4}$ | $4.12\times10^{-4}$ |
   | unstructured coarse | $5.77\times10^{-3}$ | $5.89\times10^{-3}$ |

   Each coarse solution gives essentially the same error against **either** reference (within 1.4× for structured, 1.02× for unstructured). The structured small error is therefore **not** a correlated-cancellation artifact of a structured self-reference — the structured *solution itself* is genuinely ~20× more accurate at this $N$ than the unstructured *solution itself*. The paper's vel-$L^2$ magnitudes match our **structured** run, not our unstructured run.

**Refined conclusion.** On this anisotropic outlet-channel BVP, the **structured Cartesian-simplexified mesh**, with its element edges aligned with the dominant channel-flow direction, **resolves the solution dramatically better per DOF** than the isotropic gmsh Delaunay mesh — the paper's FreeFem `buildmesh` evidently produces meshes of comparable absolute accuracy to our structured one (matching its magnitudes). Our unstructured run captures the correct **asymptotic rate** ($O(h^2)$ velocity-$L^2$, climbing-toward-2 $H^1$) but sits on a higher absolute-error curve. So:

- **Slope-level reproduction:** the velocity-$L^2$ cap is genuinely a structured-mesh slope-effect — *not* a fixed regularity ceiling — and disappears on independent unstructured meshes. The Phase 10 "the paper's $H^3{\times}H^2$ is incorrect" conclusion remains withdrawn.
- **Magnitude-level reproduction is NOT yet achieved:** to reproduce Fig. 2 absolutely, the unstructured mesh would need to recover the structured/FreeFem level of accuracy per $N$ — likely via better orientation/quality (anisotropic refinement aligned with the channel, or a Delaunay generator matching FreeFem's) or simply more vertices per long wall ($2N$ already used; trying $3N$–$4N$ on the walls would test it). The Phase 11 plot should be read as *qualitative slope evidence*, not a magnitude-matched reproduction.

A clean next test: regenerate the unstructured mesh with anisotropic boundary divisions or a higher walls/inlet ratio and see whether magnitudes drop toward the structured/paper level while the climbing slope is preserved.

---

## 13. Investigation Phase 12: reproducing the paper on the paper's own meshes (FreeFem buildmesh)

### What this phase set out to do
Phase 11 left the unstructured-mesh experiment with a $\approx 20\times$ magnitude gap to Fig. 2 and several plausible explanations on the table (mesh orientation, cancellation, paper plot reading). Phase 12 pursued the cleanest possible reproduction: **use literally the paper's `buildmesh` meshes**, on the paper's exact formulation, and see which hypotheses survive.

### Tooling added to enable the reproduction
- **`build_unstructured_model(N; wall_divisions, algorithm, save_msh)`** in [src/run_simulation.jl](../../src/run_simulation.jl) — gmsh-based meshing with prescribed boundary divisions (FreeFem-style transfinite curves, isotropic interior). Used for the gmsh comparison runs.
- **`freefem_to_gmsh(...)`** and **`load_freefem_mesh(path)`** in the same file — convert FreeFem's text `.msh` format (`nv nt nbe` header, vertex/triangle/edge sections with integer labels) into Gmsh 2.2 ASCII and load via `GmshDiscreteModel`. Labels map by convention: FreeFem `1→"inlet"`, `2→"outlet"`, `3,4→"walls"`, plus a 2D `"domain"` group. The temp .msh is auto-cleaned.
- **`GridapGmsh` (= 0.7.3) + `gmsh_jll` (~ 4.9.3)** added to `[deps]` / `[compat]` in [Project.toml](../../Project.toml). Newer `gmsh_jll` (e.g. 4.15) has an ABI mismatch with the vendored wrapper that surfaces as a segfault in `addPhysicalGroup`; pin defensively.
- **Tolerant cross-mesh search** in [`compute_reference_errors`](../../src/metrics.jl): optional `search_method=KDTreeSearch(num_nearest_vertices=k)` kwarg. The default behaviour (no kwarg) is unchanged, so the structured benchmark is byte-for-byte identical. The Irregular-mesh driver passes `k=10` to handle the non-nested unstructured case (where near-boundary fine quadrature points otherwise fail point-location and crash).
- **`test/extended/CocquetTubeTest/data/unstructured_gmsh/`** — a new self-contained sibling test mirroring the `CocquetTubeTest/data/{alpha_one,deviatoric,…}` pattern; the only difference from the benchmark is the mesh source. The driver reads `mesh_generator` (`"STRUCTURED"` / `"UNSTRUCTURED"`) and `freefem_mesh_dir` from the JSON config, with `mesh_file` plumbed through `build_solver` so the same code path handles structured, gmsh-generated unstructured, and externally-supplied FreeFem meshes. The original `structured` test is **untouched**.
- **`physical_epsilon = 10^{-7}`** set explicitly in both new configs ([`paper_comparison_irregular.json`](../../test/extended/CocquetTubeTest/data/unstructured_gmsh/paper_comparison_irregular.json), [`paper_comparison_freefem.json`](../../test/extended/CocquetTubeTest/data/freefem_meshes/paper_comparison_freefem.json)) — matching the paper's $\eta = 10^{-7}$ exactly, instead of inheriting $10^{-8}$ from `base_config.json`.

### Phase 9–10's hypotheses, retested
- **"The cap is a correlated-cancellation artifact of the structured self-reference"** (the Phase 11 first cut). **Refuted** by a cross-mesh-family test (`xfamily_diag.jl`, removed in the CocquetTubeTest unification — recoverable from git history): for the Cocquet Galerkin P2/P1 at $N=10$ vs $N=80$, the structured coarse solution gives $2.88\times10^{-4}$ against its same-family structured reference and $4.12\times10^{-4}$ against an *independent unstructured* reference — essentially the same error in both directions. The structured coarse and reference are not correlated enough to mask a global error; the structured *solution itself* is genuinely $\sim20\times$ more accurate at $N=10$ than the unstructured gmsh Delaunay one.
- **"The metric inflates the unstructured error"**. **Refuted** by reporting both nested and consistent metrics side-by-side: they agree to within $\sim1\%$ for every $N$ on the FreeFem mesh and show *identical* per-segment slope patterns.
- **"Mesh quality (cell shape/size)"**. **Refuted** by direct inspection — gmsh Delaunay at $N=10$ has 506 cells with $h \in [0.069, 0.112]$ (mean 0.088), *finer* than the structured 400 cells with leg $0.1$; aspect ratios are tight. The unstructured mesh is not degenerate.

### Spatial error localization
`localize_err.jl` (removed in the CocquetTubeTest unification — recoverable from git history) bins $\|e_u\|^2$ by region for Cocquet Galerkin P2/P1 at $N=10$ vs $N=80$:

| region | structured | unstructured |
| :-- | :-- | :-- |
| outlet-corner disks ($x{\geq}1.9$, $y{<}0.1$ or $y{>}0.9$) | **71 %** | 1 % |
| inlet-corner disks | 0.1 % | 29 % |
| wall strip (minus corners) | 3.6 % | 7.7 % |
| bulk channel | 25 % | **62 %** |
| total $\|e_u\|_{L^2}$ | $3.2\times10^{-4}$ | $5.7\times10^{-3}$ |

Two completely disjoint failure modes:
- **Structured**: 71 % of the error sits at the wall/Neumann outlet corner — confirming the Phase 9–10 corner-singularity story for the *structured* family, where the corner-aligned mesh focuses the singular mode into a single node neighbourhood.
- **Unstructured (gmsh Delaunay)**: only 1 % at the outlet corner; the cost is **62 % in the bulk channel**. The isotropic, randomly-oriented Delaunay triangulation cannot sample the parabolic-in-$y$ channel flow as efficiently per DOF as edges-aligned-with-the-flow can. The 20× magnitude inflation is genuinely a *bulk* effect of the gmsh Delaunay topology, not a corner effect.

### Why gmsh Delaunay is not a FreeFem stand-in
Three mesh recipes were tested at $N=10$ vs $N=80$ (Cocquet Galerkin P2/P1, `freefem_recipe_diag.jl`, removed in the CocquetTubeTest unification — recoverable from git history):

| recipe | cells @ $N=10$ | vel $L^2$ |
| :-- | --: | :-- |
| structured Cartesian-simplexified ($2N\times N$, then `simplexify`) | 400 | $2.88\times10^{-4}$ |
| gmsh `Mesh.Algorithm=5` (Delaunay), uniform $1/N$ edges (walls $2N$, inlets $N$) | 506 | $5.77\times10^{-3}$ |
| gmsh Delaunay, **literal** FreeFem recipe (walls $N$, inlets $N$ → anisotropic) | 186 | $9.49\times10^{-3}$ |
| gmsh `Mesh.Algorithm=7` (BAMG without metric field) | 118 | (not measured — boundary points fall outside coarse cells in cross-mesh search) |

`Mesh.Algorithm=7` is nominally Frédéric Hecht's BAMG — the same generator FreeFem uses for `buildmesh`. Without a metric field, gmsh's BAMG produces wildly under-resolved meshes (118 cells at $N=10$, $h_{\max}=0.46$ — half the domain height in a single triangle), so it is **not** a drop-in for FreeFem's `buildmesh`. Reproducing FreeFem's actual mesh required FreeFem.

### The paper's own mesh prescription has an internal inconsistency
The paper says (p. 30, §4.2):

> *"the mesh is obtained thanks to the Freefem command `buildmesh(a(N)+b(N)+c(N)+d(N))` with $N$ being the number of vertices on each part of the boundary denoted by $a,b,c,d$. As a result, the mesh-size $h$ is $h=\sqrt2/N$, and we can consider only $N$."*

But in FreeFem++, `border(N)` means $N$ *segments* on that border, irrespective of length. With $\Omega=(0,2)\times(0,1)$ and bottom/top walls of length 2, inlet/outlet of length 1:
- literal $a(N)+b(N)+c(N)+d(N)$ ⇒ wall edges $2/N$, inlet/outlet edges $1/N$ ⇒ **anisotropic**, triangle diameter $\sqrt5/N \approx 2.24/N$ — *not* $\sqrt2/N$.
- the value $\sqrt2/N$ is the diameter of an *isotropic* right triangle with both legs $1/N$, which requires the **uniform** recipe $a(2N)+b(N)+c(2N)+d(N)$.

The two statements (literal "N per border" *and* $h=\sqrt2/N$) cannot both be true.

### Reproduction on the paper's actual FreeFem meshes (literal recipe first)
The user generated the FreeFem meshes via [create_paper_meshes.edp](../../test/extended/CocquetTubeTest/data/freefem_meshes/meshes/create_paper_meshes.edp) (cell count $170$ at $N=10$, $63\,872$ at $N=200$ — confirms the literal anisotropic recipe). Three-way convergence on these:

| method (literal FreeFem mesh) | vel $L^2$ @ $N=10\to100$ | endpoint slope | finest segment slope |
| :-- | :-- | :-- | :-- |
| VMS ASGS P1/P1 | $9.42\times10^{-3} \to 3.00\times10^{-4}$ | 1.51 | 1.77 |
| VMS ASGS P2/P2 | $1.64\times10^{-3} \to 9.57\times10^{-5}$ | 1.21 | 1.89 |
| **Cocquet Galerkin P2/P1** | $\mathbf{1.55\times10^{-3} \to 9.62\times10^{-5}}$ | **1.21** | **1.94** |

Magnitude is now $\sim4\times$ closer to the paper than the gmsh Delaunay run was ($5.86\times10^{-3} \to 1.55\times10^{-3}$ at $N=10$ for the Cocquet method). Slopes climb but don't reach a clean 2 over the range. h5 archived as [convergence.h5](../../test/extended/CocquetTubeTest/results/freefem_meshes/) (gitignored under `results/`).

### Reproduction on uniform-edge FreeFem meshes (the correction)
Per the inconsistency analysis (and now documented as a new section in [theory/cocquet/cocquet_formulation.tex](../../theory/cocquet/cocquet_formulation.tex)), the user re-ran `create_paper_meshes.edp` with the **single one-line change**

```freefem
mesh Th = buildmesh(a(2*N) + b(N) + c(2*N) + d(N));
```

so all boundary edges are $1/N$ and the triangle diameter is genuinely $\sqrt2/N$. Cell counts: $448$ at $N=10$, $178\,460$ at $N=200$ — matching what $h=\sqrt2/N$ implies. The three-way convergence on **these** meshes:

| method (uniform FreeFem mesh) | vel $L^2$ @ $N=10\to100$ | endpoint slope | finest segment |
| :-- | :-- | :-- | :-- |
| VMS ASGS P1/P1 | $4.13\times10^{-3} \to 1.60\times10^{-4}$ | 1.41 | 1.43 (deflated by reference proximity; preceding segments 1.87, 1.88) |
| VMS ASGS P2/P2 | $8.78\times10^{-4} \to 4.94\times10^{-5}$ | 1.25 | 1.64 (preceded by 1.80, 1.80) |
| **Cocquet Galerkin P2/P1** | $\mathbf{9.29\times10^{-4} \to 4.59\times10^{-5}}$ | **1.31** | **1.81** (preceded by 1.92, 1.82) |

Cumulative progression for the Cocquet Galerkin P2/P1 (vel $L^2$ @ $N=10$):

| setup | vel $L^2$ |
| :-- | :-- |
| structured Cartesian-simplexified | $3.0\times10^{-4}$ |
| gmsh Delaunay (Phase 11) | $5.86\times10^{-3}$ |
| FreeFem literal (anisotropic) | $1.55\times10^{-3}$ |
| **FreeFem uniform (corrected `h=\sqrt2/N`)** | $\mathbf{9.29\times10^{-4}}$ |
| paper Figure 2 (visual reading) | $\sim 3$–$5\times10^{-4}$ |

So the **magnitude gap is now $\sim 2\times$**, down from $20\times$ at the start of Phase 12.

### The residual puzzle: per-segment slopes climb from $\sim 1$ to $\sim 2$ instead of being flat at 2
Even on the paper's own (uniform) meshes, the three-way per-segment slope pattern is:

- VMS P1/P1: `1.37, 1.26, 1.28, 1.48, 1.44, 1.33, 1.87, 1.88, 1.43`
- VMS P2/P2: `1.06, 1.11, 1.10, 1.33, 1.39, 1.32, 1.80, 1.80, 1.64`
- Galerkin P2/P1: `1.06, 1.19, 1.17, 1.47, 1.43, 1.33, 1.92, 1.82, 1.81`

All three show the same character: pre-asymptotic at coarse $N$ (slope $\sim 1.1$), transitioning at $N\!\approx\!60$, asymptotic at fine $N$ (slope $\sim 1.8$–$1.9$). A non-flat slope is the textbook signature of an error with multiple components at different rates — here, the corner-regularity-limited contribution at coarse $h$ giving way to a bulk-optimal contribution at fine $h$. **The cross-mesh metric variant doesn't change this** (nested L² endpoint slopes are 1.20, 1.20, 1.25 — same shape).

The paper's Fig. 2 curves appear visually straight at slope 2 over the same $N$ range, so either (i) their log-log scale plus a fitted slope-2 reference line hides the same wobble, (ii) their solver $.edp$ uses a slightly different outlet BC than the paper Eq. 2 advertises (the actual solver `.edp` is *not* in the paper — only the mesh `.edp` we have), or (iii) something we have not yet uncovered.

### Boundary-condition audit (third pass) — nothing found
The natural BC $\varepsilon(2\mathrm{Re}^{-1}\mathbf S(\mathbf u)-p\mathbf I)\mathbf n = 0$ on $\Gamma_{\mathrm{out}}$ is implicit in our weak form (no boundary integral added). The Dirichlet BCs on $\Gamma_{\mathrm{in}}\cup\Gamma_w$ go through `build_fe_spaces(model, …, ["inlet","walls"], [(true,true),(true,true)], [u_in, u_wall])`; the parabolic inlet is captured exactly by P2 elements (no interpolation error on the Dirichlet trace). The convective term is non-conservative $\varepsilon(\mathbf u\cdot\nabla)\mathbf u\cdot\mathbf v$ (paper Remark 1), with no convective boundary integral. The $\eta p$ penalty has matching sign and magnitude. Visual inspection of the FreeFem mesh near the outlet corner shows **no rounding, no setback, no local refinement** — the discrete geometry at the corner exactly matches what the paper specifies.

### Status and open items
**The hard quantitative claim:** the velocity-$L^2$ magnitude at $(\mathrm{Re},c_{in})=(500,0.5)$ for the Cocquet Galerkin P2/P1 on the paper's actual uniform-edge FreeFem meshes is reproduced to within a factor $\sim 2$ of the paper's Figure 2. The slope pattern is asymptotically consistent with the paper's $O(h^2)$ claim *at the fine end* but is visibly pre-asymptotic across most of the range — a discrepancy with the paper's plot that we have not yet resolved.

**Decisive open experiment** (not yet run): run the existing `test/extended/CocquetFormMMS` (smooth manufactured solution, all-Dirichlet, identical Forchheimer–Ergun reaction, same FreeFem unstructured meshes). MMS bypasses BCs, the corner, and the cross-mesh reference simultaneously. If MMS gives clean per-segment slopes, the climbing in the Cocquet benchmark is genuinely the BVP's corner regime; if MMS also climbs, the issue is in the discretisation itself.

**Most plausible undocumented difference with the paper**: their solver `.edp` (not in the paper, not in the meshes-only `.edp` we have) may implement the outlet BC slightly differently — most likely as a backflow-stabilised "do-nothing" outflow, or using `∇u:∇v` weakly (whose natural BC is the pseudo-traction $2\nu\partial\mathbf u/\partial\mathbf n - p\mathbf n$, not the symmetric-gradient traction). Either choice would soften the corner-singular pressure mode and could yield straight slope-2 curves on the same meshes without anything else changing.

**Files of record**:
- driver + configs: [test/extended/CocquetTubeTest/](../../test/extended/CocquetTubeTest/) — `run_convergence.jl`, `plot_convergence.py`, `data/unstructured_gmsh/paper_comparison_irregular.json` (gmsh path), `data/freefem_meshes/paper_comparison_freefem.json` (FreeFem path).
- diagnostic scripts: `xfamily_diag.jl`, `freefem_recipe_diag.jl`, `localize_err.jl` — removed in the CocquetTubeTest unification (recoverable from git history); their findings remain recorded above.
- meshes: [meshes/](../../test/extended/CocquetTubeTest/data/freefem_meshes/meshes/) — the user's FreeFem `.msh` files (uniform recipe) + the generating `create_paper_meshes.edp`. The literal-recipe meshes are not committed; the one-line `.edp` change documented above regenerates them in seconds.
- src additions: `freefem_to_gmsh`, `load_freefem_mesh`, `build_unstructured_model` in [src/run_simulation.jl](../../src/run_simulation.jl); `search_method` kwarg in [src/metrics.jl](../../src/metrics.jl).
- theory: new "Mesh generation and the paper's $h=\sqrt2/N$ inconsistency" section appended to [theory/cocquet/cocquet_formulation.tex](../../theory/cocquet/cocquet_formulation.tex).
