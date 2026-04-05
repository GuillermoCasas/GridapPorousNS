# Porous Navier-Stokes Solver (Gridap.jl)

A modular, mathematically rigorous Finite Element Method (FEM) solver for the stabilized Darcy-Brinkman-Forchheimer (DBF) and Porous Navier-Stokes equations using `Gridap.jl`. This repository is strictly structured around continuous Variational Multiscale (VMS) Mathematics with explicit typed tracking of stabilization branches.

## Architecture

The codebase has undergone a rigid mathematical architecture overhaul to separate the continuous equations, tracking bounds, and discrete nonlinear problem into fully isolated explicit modules:

- **`src/formulations/continuous_problem.jl`**: The overarching stabilized weak-form generator. Exposes canonical strong residuals, exact pointwise tracking of derivatives, and explicit `TypedLinearizationMode` (e.g. `ExactNewtonMode()`, `PicardMode()`) injections.
- **`src/formulations/viscous_operators.jl`**: Explicit integration-by-parts derivations of Strong Viscous Operators, Weak Pairings, and true adjoint mappings.
- **`src/models/reaction.jl`**: Contains standard `ConstantSigmaLaw` and non-linear `ForchheimerErgunLaw`.
- **`src/models/regularization.jl`**: Implements differentiable scalar tracking limits (e.g. `SmoothVelocityFloor`) for regularizing inverse norms near $\mathbf{u}=0$.
- **`src/stabilization/tau.jl`**: Exactly parsed chain-rule evaluations for calculating true $d\tau_1/d\mathbf{u}$ components to drive continuous exact Jacobians.
- **`src/stabilization/projection.jl`**: Controls the Sub-Grid Scale operator paths (ASGS/OSGS) mapping exact validation constraints conditionally based on physical models.
- **`src/solvers/nonlinear.jl`**: An exact custom `SafeNewtonSolver` orchestrating non-linear Newton bounds tracked by a mathematically sound Armijo merit linesearch $\Phi(x) = \frac{1}{2}\| F(x) \|^2$.

## Formulation Branches

The overarching API dictates exact structural formulations explicitly.

### Paper-Faithful Branch (`PaperGeneralFormulation`)
This is the only formulation to be trusted for mathematically verifying boundary exactness with 2D literature solutions. It deploys explicit Deviatoric Symmetric viscous conditions and applies strong operators consistently across all stabilization limits.

### Experimental Variant (`PseudoTractionFormulation`)
A purely experimental variant designed to bypass complex boundary locking via Laplacian mappings ($\nu\alpha\Delta \mathbf{u}$). Marked strictly experimental and must not be used to validate continuous formulation benchmarks against the reference paper.

## Execution Suites

Run specific integration tracking pipelines dynamically:

```bash
# Validating Spatial Error Loops (Manufactured Solutions)
julia --project=. tests/ManufacturedSolutions/run_test.jl

# Multi-dimensional Parameter Triggers
julia --project=. tests/CocquetExperiment/run_convergence.jl
```
