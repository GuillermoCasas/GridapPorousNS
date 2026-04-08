# Porous Navier-Stokes Solver (Gridap.jl)

A modular, mathematically rigorous Finite Element Method (FEM) solver for the stabilized Darcy-Brinkman-Forchheimer (DBF) and Porous Navier-Stokes equations using `Gridap.jl`. This repository is strictly structured around continuous Variational Multiscale (VMS) Mathematics with explicit typed tracking of stabilization branches.

## Project Purpose
To provide a highly reliable, testable, and analytically exact nonlinear solver pipeline that reproduces specific literature benchmarks (the "paper") while offering fallback legacy branches. The architecture is explicitly designed to maintain mathematical transparency and separate the continuous weak forms, stabilization parameters, configuration schemas, and discrete solvers into non-overlapping domains. 

## Core Mathematical Theory (VMS Stabilization)

This solver rigorously implements the **Variational Multiscale (VMS)** formulation, offering both **Algebraic Subgrid Scale (ASGS)** and **Orthogonal Subgrid Scale (OSGS)** flavors. In the VMS framework, the unresolved fine scales $\mathbf{u}'$ are algebraically modeled via the element-wise momentum residual: $\mathbf{u}' \approx -\tau \mathcal{R}_{mom}$.

When substituting this closure into the standard continuous Galerkin weak form $(\mathcal{L}(\bar{\mathbf{u}}), \mathbf{v})$, integration by parts analytically forces the subgrid stress to pair exactly against the **formal adjoint** of the momentum operator $\mathcal{L}^*(\mathbf{v}, q)$. This mathematically dictates that the stabilization weight augmenting the continuous weak solver must be precisely $-\mathcal{L}^*(\mathbf{v}, q)$.

> [!WARNING]
> **Adjoint Streamline Upwinding Sign Invariant**
> The formal adjoint of the continuous convective derivative $\mathcal{L}_{conv}(\mathbf{u}) = \mathbf{a} \cdot \nabla \mathbf{u}$ is derived via integration by parts: $\int \mathbf{v} \cdot (\mathbf{a} \cdot \nabla \mathbf{u}) = -\int \mathbf{u} \cdot (\mathbf{a} \cdot \nabla \mathbf{v})$. 
> Thus, $\mathcal{L}^*_{conv}(\mathbf{v}) = -\mathbf{a} \cdot \nabla \mathbf{v}$. 
> Because the stabilization term mathematically evaluates to $\langle -\mathcal{L}^*, \tau \mathcal{R} \rangle$, the weighting applied to the convective residual **MUST** strictly be positive ($+\mathbf{a} \cdot \nabla \mathbf{v}$). Reversing this scalar enforces *Anti-SUPG* negative streamline diffusion, which destroys diagonal Jacobian coercivity and triggers catastrophic Newton divergence at high $Re$.

By utilizing the full continuous $-\mathcal{L}^*$, the VMS formulation ensures robust and mathematically faithful consistency compared to ad-hoc truncated SUPG heuristics.

## Architecture Map

- **`src/formulations/continuous_problem.jl`**: The overarching stabilized weak-form generator. Exposes canonical strong residuals, exact pointwise tracking of analytical jacobian derivatives, and explicit `TypedLinearizationMode` (e.g. `ExactNewtonMode()`, `PicardMode()`) injections to completely avoid recursive AST compiler blowouts.
- **`src/formulations/viscous_operators.jl`**: Explicit integration-by-parts derivations of Strong Viscous Operators (`DeviatoricSymmetricViscosity`, `SymmetricGradientViscosity`, `LaplacianPseudoTractionViscosity`), Weak Pairings, and true adjoint mappings.
- **`src/models/reaction.jl`**: Contains standard `ConstantSigmaLaw` (used for exact MMS validation) and non-linear `ForchheimerErgunLaw`.
- **`src/stabilization/tau.jl`**: Exactly parsed chain-rule evaluations for calculating true $\partial\tau_1/\partial\mathbf{u}$ components to drive continuous exact Jacobians.
- **`src/solvers/nonlinear.jl`**: An exact custom `SafeNewtonSolver` orchestrating non-linear Newton bounds tracked by a mathematically sound Armijo merit linesearch $\Phi(x) = \frac{1}{2}\| F(x) \|^2$ and robust divergence/stagnation guards.

---

## Canonical Formulation Branches

### Paper-Faithful Branch (`PaperGeneralFormulation`)
**Validation Status**: Authoritative baseline for exact spatial error scaling.
**Design Rationale**: Deploys explicit Deviatoric Symmetric viscous conditions (`DeviatoricSymmetricViscosity`). The adjoint mapping $\mathcal{L}^*(\mathbf{v}, q)$ utilizes the complete analytical reaction adjoint (`- \sigma \mathbf{v}`). 

### Legacy Reference Branch (`Legacy90d5749Mode`)
**Validation Status**: Exists only for regression against legacy mathematical derivations.
**Design Rationale**: A simplified placeholder that utilizes `LaplacianPseudoTractionViscosity` to bypass complex boundary locking mappings. Its associated adjoints explicitly disregard the full reaction load.
**Risks When Refactoring**: MUST NOT be used to validate continuous formulation benchmarks against the reference paper.

---

## Developer Invariants & Numerical Philosophy

### 1. Hierarchical Configuration Schema
**Never hardcode physical, numerical, or geometrical properties inline.** The repository relies exclusively on a unified JSON-driven structural schema validated via JSON Schema logic.
Configuration properties must strictly map to their categorical hierarchy:
- `physical_properties` (e.g. `Re`, `Da`, `reaction_model`)
- `domain` (e.g. `alpha_0`, `bounding_box`)
- `numerical_method` (divided into `element_spaces`, `mesh`, `stabilization`, and `solver` configurations)
- `output`

### 2. Full Decoupling of Jacobian Dispatch
- **Exact Newton Homotopy**: The Fréchet analytical derivative in `build_stabilized_weak_form_jacobian` MUST match the exact finite-difference perturbations of the continuous problem. It is conditionally evaluated under `ExactNewtonMode()`.
- **Picard Independence**: A lightweight `PicardMode()` evaluates purely zero-gradient conditions for algebraic stabilization derivatives (e.g., setting $\partial\tau/\partial\mathbf{u} = 0$), providing a highly efficient damped initial condition strategy completely abstracted from Deep AST compiler stacking.

### 3. Verification Workflow (Method of Manufactured Solutions)
**Philosophy**: Every algorithmic change must pass a zero-root Exactness Diagnostic check natively integrated into `test_stacktrace.jl`.
- If tracking diagnostic roots (`L2_diag` exceeds $1e-2$ or `taylor_error` surpasses $1e-4$), exact finite element continuity has been broken and must be reverted.

---

## Testing & Execution Suites

Run specific mathematical test sweeps natively from the root repository.

### Running Fast Unit Validations
Small modular operator logic, parameter continuity, and numerical regularization validations:
```bash
# Bypass LLVM JIT Compiler overhead natively (-O0) and restrict to main thread (-t 1)
# to decrease 1x1 cell compilation tracking from several minutes to under 15 seconds.
julia -O0 -t 1 test/fast/runtests_fast.jl
```

### Running Top-Level MMS Benchmarks
The repository utilizes a modular root-level `run_test.jl` script. It automatically translates config bounds deeply into evaluating directories:
```bash
# Triggers the Manufactured Solutions suite evaluating spatial error tracking automatically
julia run_test.jl small_test_config.json
```

### Reproducing Targeted Convergence Experiments
Multi-dimensional benchmark targeting topological refinement spaces sequentially:
```bash
# From within the target experiment sub-folder:
cd test/long/CocquetExperiment
julia --project=../../.. run_convergence.jl
```
