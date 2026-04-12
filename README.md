# Porous Navier-Stokes Solver (Gridap.jl)

A modular, mathematically rigorous Finite Element Method (FEM) solver for the stabilized Darcy-Brinkman-Forchheimer (DBF) and Porous Navier-Stokes equations using `Gridap.jl`. This repository is strictly structured around continuous Variational Multiscale (VMS) Mathematics with explicit typed tracking of stabilization branches.

## Project Purpose
To provide a highly reliable, testable, and analytically exact nonlinear solver pipeline that reproduces specific literature benchmarks (the "paper theory") while explicitly safeguarding complex compilation limits intrinsic to algebraic AST trees. The architecture separates the continuous weak forms, stabilization parameters, configuration schemas, and discrete solvers into non-overlapping module spaces.

## Core Mathematical Theory (VMS Stabilization)

This solver rigorously implements the **Variational Multiscale (VMS)** formulation, offering both **Algebraic Subgrid Scale (ASGS)** and **Orthogonal Subgrid Scale (OSGS)** flavors. In the VMS framework, the unresolved fine scales $\mathbf{u}'$ are algebraically modeled via the element-wise momentum residual: $\mathbf{u}' \approx -\tau \mathcal{R}_{mom}$.

When substituting this closure into the standard continuous Galerkin weak form $(\mathcal{L}(\bar{\mathbf{u}}), \mathbf{v})$, integration by parts analytically forces the subgrid stress to pair exactly against the **formal adjoint** of the momentum operator $\mathcal{L}^*(\mathbf{v}, q)$. This mathematically dictates that the stabilization weight augmenting the continuous weak solver must be precisely $-\mathcal{L}^*(\mathbf{v}, q)$.

> [!WARNING]
> **[known-fragility] Adjoint Streamline Upwinding Sign Invariant**
> The formal adjoint of the continuous convective derivative $\mathcal{L}_{conv}(\mathbf{u}) = \mathbf{a} \cdot \nabla \mathbf{u}$ is derived via integration by parts: $\int \mathbf{v} \cdot (\mathbf{a} \cdot \nabla \mathbf{u}) = -\int \mathbf{u} \cdot (\mathbf{a} \cdot \nabla \mathbf{v})$. 
> Thus, $\mathcal{L}^*_{conv}(\mathbf{v}) = -\mathbf{a} \cdot \nabla \mathbf{v}$. 
> Because the stabilization term mathematically evaluates to $\langle -\mathcal{L}^*, \tau \mathcal{R} \rangle$, the weighting applied to the convective residual **MUST** strictly be positive ($+\mathbf{a} \cdot \nabla \mathbf{v}$). Reversing this scalar enforces *Anti-SUPG* negative streamline diffusion, which destroys diagonal Jacobian coercivity and triggers catastrophic Newton divergence at high $Re$.

By utilizing the full continuous $-\mathcal{L}^*$, the VMS formulation ensures robust and mathematically faithful consistency compared to ad-hoc truncated SUPG heuristics.

## Architecture Map

- **`src/formulations/continuous_problem.jl`**: `[paper-faithful]` The overarching stabilized weak-form generator. Exposes canonical strong residuals, exact pointwise tracking of analytical jacobian derivatives, and explicit `TypedLinearizationMode` (e.g. `ExactNewtonMode()`, `PicardMode()`) injections to completely avoid recursive AST compiler sub-tree stack overflows.
- **`src/formulations/viscous_operators.jl`**: Explicit integration-by-parts derivations of Strong Viscous Operators (`DeviatoricSymmetricViscosity`, `SymmetricGradientViscosity`, `LaplacianPseudoTractionViscosity`). `[must-test]` The primal formulation enforces **exact continuous deviatoric-symmetric evaluations** mapping natively to `∇⋅(SPi∇u)`, perfectly mirroring the MMS exact oracle. `[debugging-lore]` Gridap effectively handles exact full Hessians `∇∇(u)` on standard trial elements for this evaluation.
- **`src/models/reaction.jl`**: Contains standard `ConstantSigmaLaw` (used for `[must-test]` exact MMS validations) and non-linear `ForchheimerErgunLaw`. Enforces robust numerical flooring (`SmoothVelocityFloor`) strictly avoiding singular local parameter bounds.
- **`src/stabilization/tau.jl`**: Exactly parsed chain-rule evaluations calculating true $\partial\tau_1/\partial\mathbf{u}$ components ensuring exact analytical Newton bounds.
- **`src/solvers/nonlinear.jl`**: `[code-actual]` A robust custom `SafeNewtonSolver` orchestrating non-linear Newton bounds safely tracked mathematically by an Armijo-condition merit bounded linesearch loop with exact physical stagnation limits guarding bounds intrinsically.

---

## Canonical Formulation Branch

### Canonical Continuum Implementation (`PaperGeneralFormulation`)
**Validation Status**: Authoritative baseline evaluating exact continuous operators.
**Design Rationale**: Deploys the analytically exact `DeviatoricSymmetricViscosity` in the primal equations, preserving the formal properties of the continuous equations down to the finite limit. The test-side adjoint mapping $\mathcal{L}^*(\mathbf{v}, q)$ utilizes the analytically justified simplification `0.5*Δ(v)` because the remaining trace trace natively vanishes for divergence-free test spaces limits, avoiding potentially unstable test-element Hessians while maintaining exact theoretical coherence.

>`[legacy]` *Note that historically the solver utilized reduced structures (`LaplacianPseudoTractionViscosity`) prior to achieving exact operator mapping architecture. These branches are now formally deprecated.*

---

## Developer Invariants & Numerical Philosophy

### 1. Unified Scientific Configuration 
**Never hardcode physical, numerical, or geometrical properties inline.** The repository executes solely over a hierarchical logical parser validating JSON schema bounds accurately:
- `physical_properties`
- `domain`
- `numerical_method` (mapped accurately to execution parameters e.g. `element_spaces`, `mesh`, `stabilization`, `solver`)

### 2. Analytical Jacobian Dispatch Mapping
`[debugging-lore]` - The continuous weak form forces exact continuous spatial dependencies decoupled actively by discrete matrix states:
- **Exact Newton Homotopy**: `ExactNewtonMode()` matches finite local evaluations precisely, securing strict quadratic limits without gridap interface errors constraints through customized operator formulations.
- **Picard Initializations**: Algebraic limits abstracted via structurally defining $\partial\tau/\partial\mathbf{u} = 0$, explicitly shielding Gridap evaluations from mathematical singularity cascades inside extreme $Re$ limits natively via `PicardMode()`.

### 3. Verification Workflow (Method of Manufactured Solutions)
**Philosophy**: The core Exactness baseline validating mathematically verified equations (Eq 206 mapping). Any modification MUST cleanly pass specific category verification logic without error traces.

---

## Directory Reference Extensions
For in-depth explanations verifying paper tracking, observe the `references/paper-code-divergences.md` tracking detailed scalar limits truncations strictly enforced inside discrete Gridap code execution environments vs the theoretically continuous domain limits representations.

---

## Testing & Execution Boundaries

Testing is categorized securely bounding total execution speeds recursively validating structural operations globally natively:

### Running Verification Topologies

```bash
# BLITZ (< 5s bounds): Critical continuous structural math checks (e.g., AD mappings, analytical exactness). 
# Uses -O0 optimizations explicitly avoiding intense JIT latency
julia -O0 -t 1 test/run_blitz_tests.jl

# QUICK (6s - 2m bounds): Verifies global sparse continuous execution tests across structured limits constraints. 
julia --project=. test/run_quick_tests.jl

# EXTENDED (> 2m limits): Comprehensive convergence and parametric limit execution validations evaluating bounds matrices.
julia --project=. test/run_extended_tests.jl
```

### Reproducing Manufactured Solutions

The rigorous parameterized bounds convergence solver evaluations sweeping $O(h^{k+1})$ limits maps natively inside the extended limits directories:
```bash
# Run natively mapping configurations to parameterized test spaces inside local folder bounds
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl small_test_config.json
```
