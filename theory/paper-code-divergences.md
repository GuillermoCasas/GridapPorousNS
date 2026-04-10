# Code vs Paper Divergences Log

This document serves as the canonical map for all divergences between the literal mathematical theory detailed in the referenced `theory/article.tex` and the concrete `Gridap.jl` numerical codebase execution. It strictly adheres to the definitions imposed by the `porousns-doc-architect` framework.

Any discrepancies introduced due to numerical stability limits, algebraic bounds, discrete compilation behaviors, or Julia/LLVM restrictions MUST be securely recorded here.

---

## 1. Sub-Grid Mass Balancing Approximation
**Location**: `src/formulations/continuous_problem.jl`

**Apparent Divergence**: The strict analytical integration by parts of the subgrid convective velocities inside the Galerkin test derivations forces the theoretically exact test-side adjoint mapping to include the scalar compressibility term: $(1/\alpha)\nabla \cdot (\alpha \mathbf{u}) v$. In the code, `convective_adjoint` omits this term explicitly.

**Paper Alignment**: `[paper-faithful]` This is **not a divergence**. In `theory/article.tex` (Line 800), the authors explicitly justify the removal of this term across the entire mathematical theory to preserve stability:  
> *"Note that, strictly speaking, one has the term $\frac{1}{\alpha}\nabla \cdot (\alpha \boldsymbol{a}) \boldsymbol{v}_h$ in the expansion of $\mathcal{L}^* V_h$. The inclusion of such term generates a number of crossed terms in (\ref{eq:StabilityEstimate}) that actually harm stability. [...] Here, we have opted for simplifying the formulation by removing the aforementioned term from $\mathcal{L}^* V_h$, leading to a simpler formulation with similar stability properties."*  
Thus, the code omitting this term is a literal, faithful transcription of the exact simplified VMS operator specified in the theoretical methodology.

## 2. 2D Dilatancy Gradient Subgrid Cancellation
**Location**: `src/formulations/viscous_operators.jl`

**Paper Theory**: The Exact Viscous Operator definitions explicitly utilize $\nabla(\nabla \cdot u)$ (strong dilatancy) for resolving symmetric physical boundaries (Eq A.5 and Eq 206 limits). The baseline derivation allows the dropping of the dilatancy gradient entirely from the momentum stabilization residual acting on $\tau \mathcal{R}_{mom}$, based on theoretical limits assuming local mass balance natively. 

**Code Reality**: `[paper-faithful]` For `SymmetricGradientViscosity`, the codebase has structurally preserved the full native $\nabla(\nabla \cdot u)$ by utilizing exact internal Hessian map injections (`EvalStrongViscSymOp`), averting AST interface blowouts. For `DeviatoricSymmetricViscosity` in 2D limits, the structural term mathematically evaluates directly to $0.5 - 1/d \equiv 0$, identically canceling. Therefore, the codebase maintains robust zero-approximation theoretical fidelity scaling up through precision bounds, exactly paralleling the continuous equations.

## 3. Adjoint Streamline Mapping Positivity
**Location**: `src/formulations/continuous_problem.jl`

**Apparent Divergence**: Naively, one expects the formal convective adjoint operator $\mathcal{L}^*_{conv}$ to evaluate to $-\alpha \mathbf{a} \cdot \nabla \mathbf{v}$. However, the codebase explicitly evaluates `convective_adjoint` with a positive sign (evaluating as $+ \alpha \mathbf{a} \cdot \nabla \mathbf{v}$). Reversing the code sign triggers catastrophic numeric divergence at high Reynolds numbers.

**Paper Alignment**: `[paper-faithful]` This is **not a divergence**, but a mandatory requirement of the stability proof's coercivity. In `theory/article.tex` (Eq 39 / Line 554), the VMS stabilization bilinear form is explicitly constructed by **subtracting** the adjoint: $- \sum_{K}{\langle\mathcal{L}^* U_h, \boldsymbol{\tau} \mathcal{L} U_h\rangle}$. 
When evaluated for the velocity test function, $-\mathcal{L}_{conv}^* \mathbf{u}_h$ correctly evaluates to mathematically positive $A = + \alpha \mathbf{a} \cdot \nabla \mathbf{u}_h$. When multiplied by the strong residual which contains the same positive $A$, this forms the critical $(A - B) \cdot (A + B) = A^2 - B^2$ symmetry (Eq 50 / Line 797 in `article.tex`), creating the positive-definite stability bound $+ \Big\| \tau_1^{1/2} \alpha X(U_h)\Big\|_h^2$. 
Therefore, returning the positive evaluation in the code is structurally identical to evaluating $-\mathcal{L}^*$ in the paper's VMS inner product definition.

## 4. Jacobian Scalar Singularities Regularizations
**Location**: `src/models/reaction.jl`, `src/solvers/nonlinear.jl`

**Paper Theory**: The Jacobian bounds over limits of non-linear parameter expansions are considered mathematically continuously differentiable over the local phase transitions.

**Code Reality**: `[code-actual]` A strictly applied numerical flooring coefficient natively injects a safe non-zero structural element bounded securely by $O(1e-15)$ (via `SmoothVelocityFloor`) within geometric evaluations. It mathematically governs exact continuous Jacobians limits when structural magnitudes approach algebraic zero limits precisely where analytical derivatives of absolute norms structurally fracture.
