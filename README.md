# Porous Navier-Stokes Solver (Gridap.jl)

A modular, mathematically rigorous Finite Element Method (FEM) solver for the stabilized Darcy-Brinkman-Forchheimer (DBF) and Porous Navier-Stokes equations using `Gridap.jl`. This repository is strictly structured around continuous Variational Multiscale (VMS) Mathematics spanning high Damköhler and high Reynolds numbers stably.

---

## Documentation Reading Guide & Universal Ontology

To maintain rigorous bounds separating finite compiler engineering from pure continuous theory, this code and documentation explicitly employ the following ontology for human maintainers and AI reasoning agents:

- `[paper-faithful]`: Strictly maps to evaluated continuous analytical mathematics.
- `[code-actual]`: The implemented numerical reality enforcing safeguarding, bounding, or noise-floors.
- `[debugging-lore]`: Fixes or structural splits mandated by AST compilation or performance latency.
- `[legacy]`: Outdated formulations preserved strictly for historical regressions.
- `[known-fragility]`: Sensitive mathematical thresholds breaking if algebraically adjusted.
- `[must-test]`: Assertions fundamentally bound to the validated exact Method of Manufactured Solutions (MMS) bounds.

---

## 1. Core Mathematical Theory (VMS Stabilization)

This solver rigorously implements the **Variational Multiscale (VMS)** formulation, structurally separating **Algebraic Subgrid Scale (ASGS)** and **Orthogonal Subgrid Scale (OSGS)** tracks cleanly. Unresolved fine scales $\mathbf{u}'$ are modeled structurally via continuous element-wise residual evaluations $\mathbf{u}' \approx -\tau \mathcal{R}$.

> [!WARNING]
> **`[known-fragility]` Adjoint Streamline Upwinding Sign Invariant**
>
> Integration by parts mathematically enforces subgrid stress to pair exactly against the formal adjoint $-\mathcal{L}^*(\mathbf{v}, q)$. Because the convective adjoint evaluates intrinsically to $\mathcal{L}^*_{conv}(\mathbf{v}) = -\mathbf{a} \cdot \nabla \mathbf{v}$, the corresponding stabilization weight **MUST** be mapped symmetrically positive ($+\mathbf{a} \cdot \nabla \mathbf{v}$). 
> Reversing this forces *Anti-SUPG* negative streamline diffusion, destroying coercivity violently at parameter extrema!

> [!NOTE]
> **`[code-actual]` Adjoint Exactness Traces**
>
> The test-side adjoint mapping $\mathcal{L}^*(\mathbf{v}, q)$ utilizes the analytically justified simplification `0.5*Δ(v)`. This trace trace natively vanishes for exact divergence-free test spaces limits computationally efficiently preventing unstable continuous test-element Hessians.

---

## 2. Solver Orchestration & Safeguarding Numerical Realities

To seamlessly span macroscopic ($Re \sim 1$) to violent singular limits ($Re \sim 10^6, Da \sim 10^6$), the nonlinear continuous geometry evaluates sequences through a robust numerical fallback structure preventing arbitrary Julia exceptions:

### `[code-actual]` The Double-Wall Fallback Pipeline
1. **Aggressive Exact Newton**: `ExactNewtonMode()` matches finite local evaluations precisely, securing strict quadratic limits executing the pure exact analytical derivative.
2. **Guarded Picard Globalization**: If exact geometric sequences structurally abort (\texttt{linesearch\_failed} or merit function expansion), the sequence strategically isolates chaotic boundary parameters via `PicardMode()`. By explicitly bounding $\partial\tau/\partial\mathbf{u} = 0$, it strictly anchors boundary limits logically before restoring pure Exact Newton sequences globally.
3. **External Homotopy Dilution**: If terminal noise floors physically restrict roots natively, the orchestration automatically attenuates the boundaries ($eps\_pert = 1.0 \to 0.1 \dots$) dynamically forcing roots down mathematically independently.

### `[code-actual]` Adaptive Discretization Tolerances
To aggressively bypass redundant linear over-solve computational bounds on primitive geometry, the fundamental residual limit `ftol` evaluates dynamically locally bounded explicitly mapping continuous spatial mesh resolutions natively $\mathcal{O}(h^{k_v+1})$. This anchors algebraic roots perfectly two numerical continuous orders of magnitude below spatial discretization arrays exclusively.

### `[paper-faithful]` Fractional Validation (OSGS)
Unlike heuristic block matrices, the Orthogonal sequence dynamically bounds terminal verification separating explicit mathematical fractions independently. Success mandates the $L^\infty(\Omega)$ coordinate drift ($d_{\text{state}}$) and block orthogonal vectors ($d_\pi$) natively fall independently continuously below continuous configuration boundaries without inner linesearch aborts falsely triggering stability.

---

## 3. Architecture Map

- **`src/formulations/continuous_problem.jl`**: `[paper-faithful]` Generates stabilized weak forms exposing explicit analytical jacobians tracked cleanly over `TypedLinearizationMode`.
- **`src/formulations/viscous_operators.jl`**: `[must-test]` Deploy the exact tensor `DeviatoricSymmetricViscosity` in primal equations mapping $2\mu\nabla^s\mathbf{u}$. Gridap tracks explicit global `∇∇(u)` traces for accurate structural tensor integrations natively.
- **`src/models/reaction.jl`**: Non-linear `ForchheimerErgunLaw`. Enforces robust numerical flooring strictly avoiding singular algebraic bounds dynamically globally.
- **`src/solvers/nonlinear.jl`**: `[code-actual]` `SafeNewtonSolver` orchestrates nested sequences tracking Armijo linesearch merit conditions physically intercepted seamlessly avoiding numerical noise-floors boundaries explicitly mathematically.

---

## 4. Execution Directories & Benchmarks

The repository evaluates parametric configurations safely relying natively over global schema properties decoupling theoretical source states implicitly:

### Rapid Bounds Verifications `[debugging-lore]`
Uses `-O0 -t 1` dynamically suppressing deep AST Gridap continuous JIT latencies explicitly for structurally resolving evaluation trees locally.
```bash
# BLITZ (< 5s bounds): AD trace validations structurally exact tests.
julia -O0 -t 1 test/run_blitz_tests.jl

# QUICK (6s - 2m bounds): Sparse pipeline execution integrations.
julia --project=. test/run_quick_tests.jl
```

### Parameter Sweep Generation (`[benchmark-specific]`)
```bash
# Robust parameterized $O(h^{k+1})$ mapping executions inside configuration bounds limits.
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl test_config.json
```
