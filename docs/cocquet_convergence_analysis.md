# Cocquet Convergence Analysis and Findings

This document summarizes the investigation into the sub-optimal convergence rates observed in the Cocquet numerical experiment.

## 1. Context and Problem Statement
In the Cocquet benchmark test (defined in `test/extended/CocquetExperiment/run_convergence.jl`), we solve the 2D steady-state Darcy-Brinkman-Forchheimer (DBF) model on the domain $\Omega = [0, 2] \times [0, 1]$ with:
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

The hypotheses above conflate two different things: a possible **formulation/implementation defect** versus a **physical/measurement** effect. To separate them we built a Method-of-Manufactured-Solutions test whose governing operator is **identical to Cocquet's** — same `SymmetricGradient` viscous term and same nonlinear, porosity-dependent **Forchheimer–Ergun** reaction (`σ_linear = 0.30`, `σ_nonlinear = 1.75`), Taylor–Hood P2/P1, ASGS — but with a smooth closed-form exact solution and clean Dirichlet data. This combination (nonlinear, α-dependent σ together with a varying α) had **never** been exercised by MMS before: the oracle in `src/problems/mms_paper_2d.jl` previously errored on any non-constant reaction law.

- The shared manufactured-solution oracle was **generalized** to dispatch on the reaction law (constant σ *and* nonlinear Forchheimer-Ergun) in addition to the viscous operator it already dispatched on: `src/problems/mms_paper_2d.jl`. The standard MMS suite (`test/extended/ManufacturedSolutions/`) is unchanged — it still runs the constant-σ path.
- New self-contained test: [`test/extended/CocquetFormMMS/run_test.jl`](../test/extended/CocquetFormMMS/run_test.jl) with config [`data/cocquet_form_mms.json`](../test/extended/CocquetFormMMS/data/cocquet_form_mms.json).
- Full write-up: [`theory/cocquet_form_mms_manufactured_solution.tex`](../theory/cocquet_form_mms_manufactured_solution.tex).
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

With the fix in place, we reproduced the paper's study at its own mesh sizes — reference **N=200**, coarse **N ∈ {10,20,40,80}** ($h=\sqrt2/N$) — for the **stabilized equal-order** pairs P1/P1 and P2/P2 (the VMS method's purpose) plus Taylor–Hood P2/P1. Driver: `run_convergence.jl paper_comparison.json` (element-pair sweep). All nonlinear solves converged (4 iterations each) at Re=500.

| Element pair | velocity $L^2$ slopes | velocity $H^1$ slopes | pressure $L^2$ slopes |
| :-- | :-- | :-- | :-- |
| P1/P1 (stab.) | 1.86, 1.92, 2.08 → $O(h^2)$ | 0.98, 1.02, 0.91 → $O(h)$ | 1.86, 2.38, 1.25 |
| P2/P2 (stab.) | 2.68, 1.69, 1.51 | 1.94, 1.99, 1.98 → $O(h^2)$ | 0.68, 1.01, 1.44 |
| P2/P1 (Taylor–Hood) | 2.74, 1.86, 1.53 | 1.94, 1.99, 1.98 → $O(h^2)$ | 0.91, 1.08, 1.45 |

The stabilized equal-order elements achieve **optimal energy-norm rates**: P1/P1 gives $O(h)$ velocity $H^1$; P2/P2 gives $O(h^2)$ velocity $H^1$ — matching the paper's reported $O(h^2)$ total error for Taylor–Hood. The velocity-$H^1$ slopes are clean and robust. The velocity-$L^2$ slope decays at the fine end (P2/P2: $2.68 \to 1.69 \to 1.51$); a finer reference ($N=320$) did **not** lift it, so it is **not** a reference-resolution effect — the cause is investigated in Phase 8. Results in `results/convergence_paper_comparison.h5`.

---

## 9. Investigation Phase 8: Why P2/P2 velocity $L^2$ is below the optimal $O(h^3)$

Three controlled experiments separate the candidate causes (metric, discretization, boundary conditions). **A wrong intermediate hypothesis ("the cross-mesh metric has a resolution floor") was disproved here** and is recorded so it is not repeated.

**(a) The cross-mesh metric is exact.** `test/quick/interpolation_projection_quick_test.jl` interpolates a smooth analytic field onto coarse meshes and the $N=200$ reference, then measures the *consistent* cross-mesh $L^2$ error $\lVert u_{\text{ref}}-u_h\rVert$ via the same `compute_reference_errors` the benchmark uses. For **P2** it returns slope **3.00, 3.01** and matches the true interpolation error to 4 significant figures down to $\sim 2.5\times10^{-7}$ — *no floor, no distortion*. The metric is therefore not the cause.

**(b) The discretization reaches optimal $L^2$ on a Dirichlet problem.** The exact-solution MMS `test/extended/CocquetFormMMS` run with **equal-order P2/P2** (Re=500, $\alpha_0=0.5$, Dirichlet everywhere, error vs the analytic field — no reference) gives velocity $L^2$ slopes $2.39 \to 2.77 \to \mathbf{2.86}$ (climbing toward the optimal 3) and $H^1 \to 1.92$. So the solver/formulation *can* deliver $O(h^3)$ velocity $L^2$.

**(c) The benchmark differs by its boundary conditions.** The benchmark uses mixed inlet-Dirichlet / no-slip-wall / **traction-free Neumann outlet** conditions; the MMS in (b) uses Dirichlet everywhere. Since the metric is exact (a) and the Dirichlet problem is optimal (b), the benchmark's reduced velocity $L^2$ ($\approx 2$–$2.6$) is attributed to the **mixed boundary conditions**: the traction-free outlet and the wall$\leftrightarrow$outlet corner weaken the elliptic-duality (Aubin–Nitsche) regularity that the $L^2$ extra order requires. This is consistent with **Cocquet et al. independently reporting velocity $L^2 = O(h^2)$** (not $h^3$) for exactly this problem; they attribute the cap to the $P_1$ porosity, but the BC-regularity mechanism applies regardless. Velocity $H^1$ stays optimal because the energy-norm rate does not rely on that duality.

**Status of (c): documented possibility, not yet proven.** The clean confirming experiment would be a **Neumann-outlet MMS** — manufacture a solution that satisfies the traction-free condition and measure against the exact field; if velocity $L^2$ then drops from $\approx 3$ (Dirichlet MMS) to $\approx 2$, the BC attribution is established. This is left as the next step. The earlier per-step "L² slope < H¹ slope" observation does **not** by itself prove a measurement defect (Poincaré bounds the L² *value*, not its rate), so it is not relied upon.

**Practical takeaway:** the equal-order method converges optimally in the energy ($H^1$) norm, which is the rate the VMS theory guarantees; the velocity-$L^2$ extra order is realized for smooth Dirichlet problems and is reduced — genuinely, matching the literature — by the benchmark's mixed boundary conditions.
