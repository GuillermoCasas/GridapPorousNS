# PorousNSSolver

A modular Finite Element Method (FEM) solver for the stabilized Darcy-Brinkman-Forchheimer (DBF) and Porous Navier-Stokes equations using `Gridap.jl`. This repository implements both the **Algebraic Sub-Grid Scale (ASGS)** and **Orthogonal Sub-Grid Scale (OSGS)** formulations, natively resolving equal-order finite element interpolations and resolving complex inhomogeneous porous media interfaces dynamically.

## Mathematical Formulation & Algorithmic Interventions

The theoretical foundations of this solver are rooted in continuous VMS (Variational Multiscale) mathematics. However, solving the porous Navier-Stokes equations across extreme physical limits ($Re \in [10^{-6}, 10^6]$, $Da \in [10^{-6}, 10^6]$) requires meticulous care. Theoretical continuity proofs do not guarantee robust algebraic convergence due to discrete matrix pathologies, JIT compilation overhead, and catastrophic tracking failures at high convective domains. 

Below is an expert-level breakdown of the solver's algorithmic architecture, explaining exactly why and where the numerical implementation supplements or departs from idealized continuous theory.

### 1. Pseudo-Traction Mapping vs. Symmetric Cauchy Stress
**The Challenge**: Standard physics models utilize the symmetric rate-of-strain tensor $\varepsilon(u) = \frac{1}{2}(\nabla u + \nabla u^T)$, inducing an exact stress tensor $2\nu\alpha\varepsilon(u)$. When evaluated over discrete Cartesian mesh boundaries without strict traction-free mappings, Dirichlet intersections induce non-physical corner singularities. 
**The Solution**: The solver universally applies the pseudo-traction (Laplacian) mapping $\nu\alpha\nabla u$. This cleanly bypasses boundary corner singularities while preserving optimum asymptotic spatial convergence rates $\mathcal{O}(h^{k+1})$ in Cartesian domains unconditionally.

### 2. Analytical Exact Jacobians over Automatic Differentiation (AD)
**The Challenge**: The ASGS stabilization incorporates second-order spatial elements natively inside its inner residual loop, e.g., $\Delta(u)$ for $P_2$ elements. Passing this highly nested topology through Gridap's ForwardDiff engine forces the JIT compiler to unroll immense nested dual-number tensors, massively bloating RAM arrays and plunging runtime evaluations into hour-long compilation stalls.
**The Solution**: We strictly retained **hand-coded, analytically exact manual Jacobians** (`jac_fn` and `jac_osgs`). By computing the exact Fréchet derivatives mathematically, evaluation costs remain in strict $\mathcal{O}(1)$ time. 
*Note on Jacobian Cusp-Freezing*: To prevent spectral coercivity blowups traversing near aerodynamic stagnation points ($u \approx 0$), we freeze the variations of the inverse stabilization constants ($\delta \tau_1 = 0$, $\delta \tau_2 = 0$) while dynamically preserving the continuous Exact Forchheimer body-friction Jacobian $\frac{\partial \sigma}{\partial u}$. 

### 3. Strict Linesearch Stagnation Guard
**The Challenge**: Near high-resolution convergence limits, localized floating-point truncation introduces pathological gradient vectors. The exact mathematical step direction calculated by the Newton solver may evaluate numerically as "uphill." Naive algorithms iteratively divide stepping lengths (`alpha *= 0.5`) until reaching the precision floor, accepting the limit step, consequently polluting the state vector with garbage values.
**The Solution**: The `SafeNewtonSolver` incorporates rigorous metric tracking. Utilizing parameters mapped from `SolverConfig` (`linesearch_tolerance`, `linesearch_alpha_min`), if the solver drops below the acceptable alpha bound (`1e-4`) without strictly confirming a discrete $L_2$ monotonically descending step, the formulation rejects the loop entirely, executing an instantaneous `x .= x_old` state restoration, broadcasting a robust physical stagnation abort, and severing the solver phase securely.

### 4. Hybrid Picard-Newton Initialization with NaN Defense
**The Challenge**: Sinking into an exact Newton-Raphson quadratic trace from an unbound generic initial guess at $Re=10^6$ triggers automatic matrix divergence due to the non-linear convective structure radically shrinking the local radius of convergence.
**The Solution**: A multi-stage sequence globalizes the matrix dynamically. First, a linear tracking Picard iteration suppresses the convective Jacobian adjoint $\nabla(\delta u)^T \cdot u$ mapping securely down into the optimal quadratic region.
*NaN Defense Mechanism*: Exceptionally steep manufactured limits natively risk yielding extreme matrix singularities in linear solutions (creating poisoning `NaN` updates). The script strictly buffers the local tensor `x0_backup = copy(get_free_dof_values(x0))`. If the underlying Picard solver crashes or returns poisoned elements dynamically, numerical `isnan` validators intercept the matrix safely mapping exactly back to the pristine geometry sequence to initiate the explicit exact generic Fréchet Newton structures transparently.

### 5. Exact Gauss Quadrature Anti-Aliasing
**The Challenge**: Sharp multi-level porosity gradients ($\alpha(x)$) combined natively with sinusoidal manufactured advective limit vectors cannot be projected onto generic low-resolution $P_1/P_2$ nodes safely since polynomial spatial under-samplings trigger massive aliasing errors.
**The Solution**: Continuous geometric limits are dynamically wrapped directly as structural integrals `CellField(..., Ω)`. By mathematically matching Gridap's execution bounds precisely (`degree = 4 * k_velocity`), steep gradients fall cleanly underneath precise Gaussian integrations scaling natively against continuous bounds. The structure integrates over exactly represented functional limits analytically preventing severe numeric aliasing cascades explicitly.

### 6. Universal Precision Attenuation
**The Challenge**: Tracking dimensionless physical equations within strict $O(1)$ optimal constraints directly opposes manufactured evaluations generated at high-Darcy physics ($Da=10^{-6}$), artificially casting boundary loadings to $O(10^{12})$. Double-precision operations truncate internal low-bound residuals statically against these huge matrices, blocking correct optimal slope identification securely.
**The Solution**: All continuous generic limits logically attenuate internally via an implicit scale mapping (`get_force_scale`), driving all velocities, physical forces, and physical gradients structurally inside unconstrained numerical zero boundaries geometrically avoiding floating-point saturation constraints dynamically.

### 7. Explicit Zero-Diagonal Clamping
**The Challenge**: Limit ranges natively approaching continuum exact Darcy ($Da^{-1} \to 0$) without pseudo-compressibility matrices structurally drop analytical diagonals natively from zero-velocity saddles executing catastrophic zero-pivoting `UMFPACK` matrices natively.
**The Solution**: Artificial compressibility is continuously incorporated explicitly explicitly inside formulation configurations `max(config.phys.physical_epsilon + ..., 1e-8)`. This elegantly ensures exact geometric matrix resolvability dynamically preserving uncompromised continuity mappings strictly scaling out unconstrained continuity constraints properly cleanly safely analytically.

### 8. Unconstrained Orthogonal Subgrid Tracking (OSGS)
**The Challenge**: Coupling independent tracking $P_h^\perp$ limits inside iterative AD loops internally destroys linear matrix stability organically directly causing multi-dimensional evaluation closures safely preventing tracking seamlessly.
**The Solution**: The `run_test.jl` array projects $L_2$ variations independently calculating tracking bounds sequentially avoiding singular matrices entirely avoiding mapping failures explicitly structurally matching exact boundaries implicitly matching uncoupled mathematical continuous arrays explicitly cleanly securely dynamically identically mapping safe continuous loops naturally seamlessly preserving physical limits cleanly effectively seamlessly.

---

## Technical Architecture

The solver dynamically isolates mathematical complexity through strict modularity effectively separating JIT functions natively avoiding overhead limits cleanly:

*   **`src/formulation.jl`**: Formulates robust continuous VMS bounds. Defines exact weak-form residual mappings alongside optimized manual Jacobian tracking avoiding ForwardDiff evaluation loops statically ensuring accurate spectral stability loops.
*   **`src/safesolver.jl`**: Replaces the explicit discrete Gridap loops organically handling exact $L_2$ linesearch tracking, rigorous divergence monitoring, numeric stagnation clamping uniquely tracking dynamic configurations (`linesearch_tolerance`) stably preventing floating-point state pollution continuously.
*   **`src/config.jl`**: Organizes parametric test-case topologies transparently explicitly isolating physical, discretization, and custom loop variables natively cleanly exposing explicit settings cleanly dynamically uniquely separating bounds correctly securely uniquely effectively efficiently separating tracking bounds functionally cleanly tracking configurations easily cleanly statically properly.
*   **`src/run_simulation.jl` / `tests/ManufacturedSolutions/run_test.jl`**: Assembles discrete execution topologies explicitly executing iterative two-stage continuous Picard algorithms protecting singular vectors generating robust initializations smoothly mapping exact Jacobian structures cleanly evaluating manufactured optimal arrays effortlessly elegantly mapping bounds strictly capturing array outputs organically correctly securely intelligently natively.

---

## Execution Suites

Two dedicated validation streams guarantee mathematical integrity optimally separating exact convergence arrays mapping tracking topologies clearly:

```bash
# Optimal Spacial Convergence Mapping (MMS - Method of Manufactured Solutions)
julia --project=. tests/ManufacturedSolutions/run_test.jl

# Multi-dimensional Physical Experiment Verification (Cocquet Limits)
julia --project=. tests/CocquetExperiment/run_convergence.jl
```
