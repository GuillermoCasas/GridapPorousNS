# Porous Navier-Stokes Solver (Gridap.jl)

A modular, mathematically rigorous Finite Element Method (FEM) solver for the stabilized Darcy-Brinkman-Forchheimer (DBF) and Porous Navier-Stokes equations using `Gridap.jl`. This repository is strictly structured around continuous Variational Multiscale (VMS) Mathematics with explicit typed tracking of stabilization branches.

## Project Purpose
To provide a highly reliable, testable, and analytically exact nonlinear solver pipeline that reproduces specific literature benchmarks (the "paper") while offering fallback legacy branches. The architecture is explicitly designed to maintain mathematical transparency and separate the continuous weak forms, stabilization parameters, and discrete solvers into non-overlapping domains. 

## Architecture Map

- **`src/formulations/continuous_problem.jl`**: The overarching stabilized weak-form generator. Exposes canonical strong residuals, exact pointwise tracking of derivatives, and explicit `TypedLinearizationMode` (e.g. `ExactNewtonMode()`, `PicardMode()`) injections.
- **`src/formulations/viscous_operators.jl`**: Explicit integration-by-parts derivations of Strong Viscous Operators (`DeviatoricSymmetricViscosity`, `SymmetricGradientViscosity`, `LaplacianPseudoTractionViscosity`), Weak Pairings, and true adjoint mappings.
- **`src/models/reaction.jl`**: Contains standard `ConstantSigmaLaw` (used for exact MMS validation) and non-linear `ForchheimerErgunLaw`.
- **`src/models/regularization.jl`**: Implements differentiable scalar tracking limits (e.g. `SmoothVelocityFloor`) for regularizing inverse norms near $\mathbf{u}=0$.
- **`src/stabilization/tau.jl`**: Exactly parsed chain-rule evaluations for calculating true $d\tau_1/d\mathbf{u}$ components to drive continuous exact Jacobians.
- **`src/stabilization/projection.jl`**: Controls the Sub-Grid Scale operator paths (ASGS/OSGS) mapping exact validation constraints conditionally based on physical models.
- **`src/solvers/nonlinear.jl`**: An exact custom `SafeNewtonSolver` orchestrating non-linear Newton bounds tracked by a mathematically sound Armijo merit linesearch $\Phi(x) = \frac{1}{2}\| F(x) \|^2$ and robust divergence/stagnation guards.

---

## Canonical Formulation Branches

The overarching API dictates exact structural formulations explicitly.

### Paper-Faithful Branch (`PaperGeneralFormulation`)
**Validation Status**: Authoritative baseline for exact spatial error scaling.
**Design Rationale**: Deploys explicit Deviatoric Symmetric viscous conditions (`DeviatoricSymmetricViscosity`) and applies strong operators consistently across all stabilization limits. The adjoint mapping $\mathcal{L}^*(\mathbf{v}, q)$ includes the reaction term (`- \sigma \mathbf{v}`). 
**Risks When Refactoring**: Any LLM modifying this branch must ensure that exact Manufactured Solution bounds remain intact. Do not change the `DeviatoricSymmetricViscosity` divergence rule unless paper validation is updated.

### Legacy Reference Branch (`Legacy90d5749Mode`)
**Validation Status**: Exists only for regression against legacy mathematical derivations.
**Design Rationale**: Replaces deviatoric formulations with a simplfied `LaplacianPseudoTractionViscosity` to bypass complex boundary locking mappings ($\nu\alpha\Delta \mathbf{u}$). Its adjoint does not inherit the full reaction load.
**What changed from legacy**: Identified and quarantined as a non-physical simplification from commit `90d5749`.
**Risks When Refactoring**: MUST NOT be used to validate continuous formulation benchmarks against the reference paper. Future agents should treat this as a controlled legacy endpoint and avoid "syncing" it to `PaperGeneralFormulation`.

---

## Developer Invariants & Numerical Philosophy

### 1. The Nonlinear Solution Pipeline
- **Exact Newton Homotopy**: The code has transitioned structurally from a monolithic Picard-Newton hybrid to an exact Newton Homotopy tracked by `SafeNewtonSolver`.
- **Initialization**: A `PicardMode` evaluation is still occasionally used to provide a damped initial condition during harsh perturbations, but it explicitly avoids calculating $\partial\sigma/\partial\mathbf{u}$ or full $\partial\tau/\partial\mathbf{u}$.
- **Linesearch Merit**: The Armijo merit is derived strictly from the pointwise $L^2$ mapping of the strong residual: $\Phi(x) = \frac{1}{2}\| F(x) \|^2$. Divergence guards will quietly abort execution if $\Phi(x)$ balloons uncontrollably.

### 2. Jacobian Consistency
- The Fréchet analytical derivative in `build_stabilized_weak_form_jacobian` MUST match the exact finite-difference perturbations of the continuous problem.
- **Known Pitfalls**: Freezing the temporal $\tau$ derivative (`freeze_cusp`) compromises quadratic convergence. Do not re-enable it blindly without assessing the loss of Newton iteration speed.

### 3. Verification Workflow (Method of Manufactured Solutions)
**Philosophy**: Every algorithmic change must pass a zero-root Exactness Diagnostic check.
- The continuous formulations expect an analytical input (sines, cosines) yielding an exact $O(h^{k+1})$ residual mapping. 
- If `L2_diag` exceeds $1e-2$ or `taylor_error` surpasses $1e-4$ initially, atmospheric mathematical continuity has been severed.

---

## Execution Suites

Run specific integration tracking pipelines dynamically:

```bash
# Validating Spatial Error Loops (Manufactured Solutions)
# This includes strict Exact Root Tests and Taylor Error tracking
julia --project=. test/long/ManufacturedSolutions/run_test.jl

# Multi-dimensional Parameter Triggers
julia --project=. test/long/CocquetExperiment/run_convergence.jl
```

## How to Safely Modify the Code
1. **Never alter Viscous Operators without consulting `PaperGeneralFormulation`**: The specific cancellation of terms in `strong_viscous_operator` relies on precise spatial derivatives (e.g. $\nabla\cdot\mathbf{D} = 0.5 \Delta \mathbf{u}$ in 2D).
2. **Beware gridap AST growth**: Re-using the same closures multiple times blows up compilation time. If you create a new $\tau$ or strong residual operation, define it as an explicit callable struct (e.g., `SigOp`, `Tau1Op`).
3. **Respect Parameter Structs**: Do not hardcode arbitrary $Da$ or $1.0/Re$ variables inline; pipe them cleanly through the hierarchical `PhysicalParameters` in `src/config.jl`.
