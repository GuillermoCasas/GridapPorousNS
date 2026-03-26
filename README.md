# PorousNSSolver

A modular Finite Element Method (FEM) solver for the stabilized Darcy-Brinkman-Forchheimer (DBF) and Porous Navier-Stokes equations using `Gridap.jl`. This repository implements the **Algebraic Sub-Grid Scale (ASGS)** formulation presented in the theoretical work by Casas et al.

## Mathematical Formulation

The solver discretizes the time-independent porous Navier-Stokes continuous equations:

1. **Momentum:**
   $$ \alpha (u \cdot \nabla) u - 2\nabla \cdot (\alpha \nu \nabla^S u) + \alpha \nabla p + \sigma(\alpha, u)u = f $$
2. **Mass:**
   $$ \epsilon p + \nabla \cdot (\alpha u) = 0 $$

To stabilize the convective and pressure terms optimally across standard equal-order elements (e.g., $P_1/P_1$ or $P_2/P_2$), we employ the ASGS stabilization method where the projection operator is exactly chosen as $\mathbb{\Pi} \equiv \mathbb{I}$. The stabilized weak form is given by:

$$ B_{ASGS}(u, U_h, V_h) = B_{Gal}(u, U_h, V_h) + \sum_K \langle \mathcal{L}_u^* V_h, \tau_1 \mathcal{R}_u(U_h) \rangle_K + \sum_K \langle \mathcal{L}_p^* Q_h, \tau_2 \mathcal{R}_p(U_h) \rangle_K $$

### Resistance and Stabilization Constants
The nonlinear Darcy-Forchheimer resistance is governed by the structural porosity $\alpha$:
$$ \sigma(\alpha, u) = \frac{1}{Da}\frac{150}{Re}\left(\frac{1-\alpha}{\alpha}\right)^2 + 1.75\left(\frac{1-\alpha}{\alpha}\right)\|u\| $$

Where the ASGS temporal scaling operators are calculated strictly cell-wise as:
$$ \tau_{1,NS} = \left( c_1 \frac{\nu}{h^2} + c_2 \frac{\|u\|}{h} \right)^{-1} $$
$$ \tau_1 = \left( \alpha \tau_{1,NS}^{-1} + \sigma(\alpha, u) \right)^{-1} $$
$$ \tau_2 = \frac{h^2}{c_1 \alpha \tau_{1,NS} + \epsilon h^2} $$

These formulations dynamically infer the molecular viscosity directly from the Reynolds number ($\nu = 1/Re$) when treated in a strictly dimensionless format.

## Algorithmic Architecture

The codebase leverages a rigid separation of concerns:
*   `src/config.jl`: Implements a JSON parsing engine establishing base default configs, nested parsing, deep-merging capabilities, and strict typing routines.
*   `src/geometry_mesh.jl`: Constructs and labels `Gridap` discrete Cartesian box models.
*   `src/formulation.jl`: Directly translates the abstract mathematical ASGS theory into the code. The non-linear `FEOperator` bypasses automatic differentiation (AD) limitations by implementing an exact analytical Picard/Newton Jacobian, ensuring optimal non-linear convergence and robust evaluation of momentum, mass, and stabilization residuals. 
*   `src/run_simulation.jl`: Handles high-level pipeline orchestration. Links configuration, FEM spaces, boundaries, and executes the Newton-Raphson non-linear solver. 
*   `src/io.jl`: Automates the serialization to high-fidelity VTU datasets compatible with ParaView visualizations.

## Experimental Suites

Both major validation scenarios from the literature are explicitly implemented:

### 1. Manufactured Solutions (`tests/ManufacturedSolutions/`)
Executes an artificially constructed vector/pressure profile perfectly satisfying local governing continuity restrictions. Operates across successively refined mesh levels ($10\dots 40$) and strictly validates convergence behaviors. 

**Recent Validation**: The stabilizing ASGS formulation achieves theoretically optimal scaling ($O(h^{k+1})$ for velocity $L^2$ and $O(h^k)$ for pressure $L^2$ on equal-order spaces) even under a highly non-linear spatially-varying porosity field $\alpha(x)$. This is achieved by strictly balancing the convective operators and preserving the complete viscous stress divergence $\nu (\nabla u + \nabla u^T) \cdot \nabla \alpha$ within the strong discrete residual $\mathcal{R}_u(U_h)$.

Python Matplotlib endpoints visualize the $L^2$/$H^1$ convergence slopes cleanly:

<p align="center">
  <img src="tests/ManufacturedSolutions/results/convergence.png" alt="MMS Convergence Plot" width="600"/>
</p>

### 2. Cocquet Experiment (`tests/CocquetExperiment/`)
Configured to validate Section 4.2 of the benchmark error analysis established by Cocquet et al. Simulates a heterogeneous structural transition driving flow through a "free space" ($\alpha = 1$) into an active DBF internal porous matrix ($\alpha \ll 1$), properly capturing rapid interfacial non-linear shear deterioration and macroscopic flow diversion without analytical boundary instabilities.

## Usage

You can run individual test cases directly by loading the local Julia environment:

```bash
# Run the Method of Manufactured Solutions Tests
julia --project=. tests/ManufacturedSolutions/run_test.jl
python3 tests/ManufacturedSolutions/plot_results.py

# Run the Cocquet DBF Verification
julia --project=. tests/CocquetExperiment/run_test.jl
python3 tests/CocquetExperiment/plot_results.py
```

## 🤖 Context Base for AI Assistants

This section is explicitly designed to bootstrap AI agents reading this repository, outlining critical architectural nuances and Gridap.jl implementation constraints that define this solver.

### 1. Convective Operator Orientation in Gridap
The solver builds upon Gridap's tensor operators. In Gridap, the gradient of a vector field `u` returns a Jacobian tensor structured such that the expression for the convective directional derivative $(u \cdot \nabla) u$ maps strictly to:
```julia
conv_u = transpose(∇(u)) ⋅ u  # OR  ∇(u)' ⋅ u
```
Using `∇(u) ⋅ u` computes $\nabla u \cdot u$ and will break formulation consistency. All manufactured solution forcing derivations (`f_ex`) explicitly respect this mapping.

### 2. Strong Residuals and Viscous Divergence
In the ASGS stabilization framework, the strong residual $\mathcal{R}_u(U_h)$ drives the stabilization operators $\tau_1$. Although the standard Laplacian $\Delta u_h$ evaluates natively to $0.0$ inside piecewise-linear ($P_1$) elements, the porous Navier-Stokes equations couple velocity gradients with the spatially varying porosity field $\alpha(x)$. 
Therefore, the **full viscous divergence** must be manually constructed and retained within `R_u` evaluated inside `src/formulation.jl`:
```julia
div_visc_u = α * ν * Δ(u) + ν * (∇(u) + transpose(∇(u))) ⋅ ∇(α)
```
Omitting the right-hand term $\nu (\nabla u + \nabla u^T) \cdot \nabla \alpha$ breaks consistency and immediately caps pressure $L^2$ convergence below optimal $O(h^k)$ rates.

### 3. Analytical Picard/Newton Jacobian
Gridap's Automatic Differentiation (AD) struggles with nested `Operation` closures involving highly non-linear scalar coefficients ($\sigma(\alpha, u)$, $\tau_1$, $\tau_2$). To bypass JIT compilation hangs and `MethodError` exceptions during `FEOperator` assembly, this repository manually defines the `weak_form_jacobian` (found in `src/formulation.jl`). 
- When editing the residual, agents **must exactly mirror** the changes within the Jacobian's $R_{du}$ and $R_{u\_old}$ closures.
- The scalar fields ($\tau_1$, $\tau_2$, $\sigma$) are effectively "frozen" (Picard-style) with respect to $du$ derivations to ensure unconditional matrix stability, but the vector derivatives $\mathcal{L}^* \cdot \tau_1 \mathcal{R}_u$ are fully expanded via the product rule into $dL_{du}^* \cdot \tau R_{u\_old} + L_u^* \cdot \tau R_{du}$.
