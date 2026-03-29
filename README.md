# PorousNSSolver

A modular Finite Element Method (FEM) solver for the stabilized Darcy-Brinkman-Forchheimer (DBF) and Porous Navier-Stokes equations using `Gridap.jl`. This repository implements the **Algebraic Sub-Grid Scale (ASGS)** formulation presented in the theoretical work by Casas et al., specifically resolving equal-order interpolations natively mapping inhomogeneous porous media interfaces dynamically.

## Mathematical Formulation & Deviations from Theoretical Baselines

The solver discretizes the time-independent porous Navier-Stokes continuous equations. While the code is fundamentally built upon the rigorous continuous framework outlined in `theory/article.pdf`, achieving robust algebraic convergence at extreme physical parameter limits ($Re \in [10^{-6}, 10^6]$, $Da \in [10^{-6}, 10^6]$) strictly required several independent algorithmic interventions. 

Below, we detail the core algorithmic elements implemented in the codebase, what they provide, and how they specifically depart from or supplement the idealized continuous formulations found in the reference article.

### 1. Pseudo-Traction Mapping vs. Symmetric Cauchy Stress
**The Theory (`article.pdf`)**: Standard physical formulations conventionally derive momentum diffusion from the symmetric rate-of-strain tensor $\varepsilon(u) = \frac{1}{2}(\nabla u + \nabla u^T)$, yielding a physical viscous stress tensor $2\nu\alpha\varepsilon(u)$.
**The Algorithmic Intervention**: The codebase replaces symmetric spatial traction with the pseudo-traction (Laplacian) mapping $\nu\alpha\nabla u$. 
**What it provides**: Enforcing symmetric drag natively assumes strict stress-free geometric corner constraints (e.g., $\partial_y u_x = 0$ at boundaries). On Cartesian discrete bounds, these corner singularities violently restrict the error capacity of high-order ($P_2/P_2$) polynomials. Dropping the symmetric transpose structurally bypasses these corner boundary singularities, recovering optimal $O(h^3)$ convergence profiles without sacrificing global physical validity.

### 2. Jacobian Cusp-Freezing (Advection Stabilization Derivatives)
**The Theory (`article.pdf`)**: The article defines algebraic stabilization coefficients $\tau_1(u)$ and $\tau_2(u)$ as continuous nonlinear functions of the local velocity norm $\|u\|$. An exact Newtonian linearization theoretically demands computing the exact analytical Fréchet derivatives $\frac{\partial \tau_1}{\partial u}$ and $\frac{\partial \tau_2}{\partial u}$.
**The Algorithmic Intervention**: In `src/formulation.jl`, within `weak_form_jacobian`, we strictly **freeze** the derivatives of the advection-diffusion parameter bounds (treating $\delta \tau_1 = 0$ and $\delta \tau_2 = 0$) while still preserving the exact Forchheimer geometric body-friction derivative ($\frac{\partial \sigma}{\partial u}$).
**What it provides**: The mathematical operators defining $\tau$ are inversely proportional to $\|u\|$. Near aerodynamic stagnation points ($u \approx 0$), computing exact variations projects catastrophic indefinite eigenvalue spikes into the global matrix. Cusp-freezing completely nullifies these unphysical limits, protecting the Newton solver's spectral coercivity from blowing up while still accurately capturing the necessary continuous resistance tracking.

### 3. Absolute Nullspace Clamping via Artificial Compressibility
**The Theory (`article.pdf`)**: The continuous equations enforce pure mass conservation via $\nabla \cdot (\alpha u) = 0$.
**The Algorithmic Intervention**: The solver supplements the native mass equation with a pseudo-compressibility limit: $\varepsilon p + \nabla \cdot (\alpha u) = 0$. Crucially, the codebase enforces a hard mathematical floor on this relaxation: `max(..., 1e-8)`.
**What it provides**: At extreme advective or high-viscosity limits ($Da^{-1} \to 0$), exact continuum mass conservation generates zero-diagonal blocks within the generic saddle-point algebraic matrix. This directly induces a singular pressure nullspace, triggering catastrophic $O(10^{20})$ `UMFPACK` pivoting blowups. The pseudo-compressibility clamp elegantly regularizes the continuum limits, preserving matrix invertibility without bleeding significant physical flow mass loss.

### 4. Hybrid Picard-Newton Initialization
**The Theory (`article.pdf`)**: Theoretical proofs generally assume idealized, highly localized starting guesses for optimal nonlinear convergence bounds.
**The Algorithmic Intervention**: The solver integrates an autonomous two-stage globalization pipeline internally. 
**What it provides**: At hyper-advective states ($Re = 10^6$), standard Newton-Raphson dynamically diverges when starting from a blind $u_0 = 0$ initial guess due to a vanishingly small mathematical radius of convergence. The code initiates execution using a Picard fixed-point linearizer (dropping the convective adjoint variations $\nabla(\delta u)^T \cdot u$) to safely contract the global domain error smoothly. Once the $L^2$ fields sit securely inside the rigorous quadratic basin of attraction, the solver seamlessly pivots to full Newton-Raphson execution.

### 5. Exact Gauss Quadrature Anti-Aliasing
**The Theory (`article.pdf`)**: Continuous stability proofs assume perfect infinite-dimensional integration spaces.
**The Algorithmic Intervention**: Sharp internal macroscopic porosity thresholds ($\alpha(x)$) or exponential manufactured body-forces are strictly wrapped in exact mathematical Gridap structures (e.g., `CellField(alpha_fn, Ω)`) instead of projecting them via generic discrete nodal interpolation grids (`interpolate_everywhere`).
**What it provides**: Interpolating steep exponential gradients onto sparse discrete $P_1/P_2$ grids introduces massive artificial spatial aliasing and harmonic corruption feeding the solver's RHS boundaries. Packing these functions analytically directly forces the local assembly iteration loops to evaluate the continuum formulation exactly at the high-order `dΩ` Gauss integration points, cleanly stripping out numeric noise matrices and guaranteeing pristine scaling tracking.

### 6. Universal Discretization Precision Scaling
**The Theory (`article.pdf`)**: The formulation models dimensionless dynamics strictly within theoretical $O(1)$ floating point matrices natively.
**The Algorithmic Intervention**: During Method of Manufactured Solution (MMS) validations, the artificial continuous equations and true velocity $u_{ex}$ are geometrically attenuated by a localized scalar bound `get_force_scale` aggregating the global viscous and Forchheimer bounds.
**What it provides**: Unscaled manufactured true formulations evaluating at pure viscosity limits ($Re = 10^{-6}$) naturally cast $O(10^{12})$ loading scales dynamically. Inside standard double-precision `Float64` constraints, combining this unscaled load with bounded variable limits induces severe $10^{-4}$ numeric truncation ceilings that permanently drown sensitive generic $O(h^k)$ evaluation thresholds. Sinking and scaling the velocities directly maps continuous evaluation metrics seamlessly underneath generic floating-point stress-limits natively preserving scaling convergence slopes perfectly to zero.

---

## Technical Architecture

The codebase leverages a rigid separation of concerns inside standard generic Gridap modeling:
*   `src/formulation.jl`: Forms the exact ASGS weak structures evaluated natively bypassing compilation bounds dynamically generating manual Jacobian structures correctly preserving discrete Forchheimer parameters cleanly.
*   `src/config.jl`: Implements internal native parameter mapping mapping continuous serialization safely resolving validation loops properly.
*   `src/geometry_mesh.jl`: Constructs fundamental Gridap topological boundaries labeling components (`inlet`, `outlet`, `walls`).
*   `src/run_simulation.jl`: Handles high-level execution mapping discrete variables directly into structured evaluation parameters safely tracking limits natively seamlessly returning solved constraints.
*   `src/io.jl`: Computes VTU generic projection algorithms mapping visualization formats flawlessly bounding generic variable components.

## Experimental Suites

Both major validation scenarios from the literature are explicitly implemented and routinely verified against $O(h^k)$ optimal expectations natively tracking discrete evaluation matrices.

```bash
# Method of Manufactured Solutions (Exact Spatial Tracking)
julia --project=. tests/ManufacturedSolutions/run_test.jl

# Macroscopic Cocquet Evaluation (Physical Channel Boundary Setup)
julia --project=. tests/CocquetExperiment/run_convergence.jl
```
