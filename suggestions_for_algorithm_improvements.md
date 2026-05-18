# Suggestions for Algorithm Improvements

## 1. Mathematical and Implementation Discrepancies

After cross-referencing `theory/osgs_algorithm.tex` with `src/solvers/nonlinear.jl` and `src/solvers/porous_solver.jl`, I have identified three critical mismatches where the code is actually more robust and sophisticated than the documentation suggests:

### A. Block-Equilibrated Merit Function (Section 3.1)
- **The Document:** Equation (2) defines the equilibration weights as $w_k = \max(|J_{kk}|, \varepsilon_{\text{mach}} \|\text{diag}(\mathcal{J})\|_\infty, \varepsilon_{\text{mach}})$. This implies a *global* infinity norm over the entire Jacobian diagonal.
- **The Code:** `_update_merit_weights!` in `nonlinear.jl` explicitly detects field blocks (velocity vs. pressure) and computes the infinity norm *per block* before calculating the scaling for each row.
- **Why it matters:** The code's approach is mathematically far more elegant for saddle-point/mixed-physics problems. A global max would allow the momentum block (scaled by large $Re$ or $Da$) to completely drown out the mass block's diagonal. The document should be updated to reflect this block-wise equilibration, especially since the code comments explicitly reference "§3.1" as if the document already explained it.

### B. Picard Mode in the Armijo Line Search (Algorithm A.2)
- **The Document:** Algorithm A.2 asserts that the line search acceptance criterion is always $\hat\Phi \le \Phi + c_1 \alpha D$. Section 3.2 correctly notes that the identity $D = -2\Phi$ is violated for Picard steps, but fails to explain what the solver actually does about it.
- **The Code:** The `eval_armijo_linesearch_pass!` function explicitly branches on `solver.mode`. For `:picard`, it abandons the merit function entirely and uses a monotone residual reduction check based on the infinity norm: $\|\hat{\boldsymbol{b}}\|_\infty \le (1.0 - c_1 \alpha) \|\boldsymbol{b}\|_\infty$.
- **Why it matters:** This is a major omission in the algorithm box. The document needs to show this branching logic to be mathematically rigorous, otherwise the Picard line search appears theoretically flawed or contradictory to Section 3.2.

### C. The OSGS Inner Solver Cascade (Algorithm C)
- **The Document:** Algorithm C shows the OSGS staggered loop invoking `ExactNewtonPipeline` (Algorithm A) exactly once per outer iteration.
- **The Code:** In `porous_solver.jl` (lines 453-571), the OSGS inner loop actually executes the exact same Newton $\to$ Picard $\to$ Newton cascade that is used in Stage I.
- **Why it matters:** The document underrepresents the robustness of the OSGS inner loop, omitting the crucial Picard fallback that secures convergence when the subgrid projection is stiff.

## 2. Suggestions for Elegance, Redundancy Removal, and Structural Reordering

The document is already excellently structured, particularly the use of interface cards and the strict separation of core vs. extensions. To make it the most elegant, readable, and rigorous representation of the codebase, I suggest the following architectural refactoring of the LaTeX document:

### A. Refactor Algorithm B into a Generic `RobustNonlinearCascade`
Currently, Algorithm B is named `PicardFallback` and is strictly tied to Stage I (ASGS initialization). Because the exact same cascade is used in Stage II (OSGS inner loop), you can significantly elevate the elegance of the document by refactoring Algorithm B:
1. Rename Algorithm B to `RobustNonlinearCascade(x0, op_newton, op_picard)`.
2. Update **Algorithm O (Orchestrator)** Stage I to simply say: `(S, U_h^{ASGS}) \gets \mathrm{RobustNonlinearCascade}(\boldsymbol{x}^0, \text{ASGS Operators})`.
3. Update **Algorithm C (OSGS)** inner loop to say: `(U_h^m, S_m^{inner}, \dots) \gets \mathrm{RobustNonlinearCascade}(U_h^{m-1}, \text{OSGS Operators frozen at } \boldsymbol{\pi}_h^{m-1})`.

This eliminates redundancy, correctly aligns the document with the code, and highlights the modular reuse of the stabilization cascade.

### B. Structural Reordering: Un-Sandwiching Stage I and Stage II
**Current Order:** Orchestrator (Sec 5) $\to$ Stage I Cascade (Sec 6) $\to$ Newton Kernel (Sec 7) $\to$ Stage II OSGS (Sec 8).
**The Problem:** Stage II (OSGS) is the logical peer of Stage I (ASGS), but they are awkwardly separated by the massive Newton Kernel section. 
**The Solution:** Group the modules by abstraction level.
- **Level 1 (Top):** The Orchestrator (Algorithm O) and The Stabilized Solvers (Stage I ASGS \& Stage II OSGS).
- **Level 2 (Middle):** The `RobustNonlinearCascade` (refactored Algorithm B) which drives both stages.
- **Level 3 (Bottom):** The `ExactNewtonPipeline` (Algorithm A) and its leaves (A.1, A.2, A.3).
This logical grouping flows much better, separating the formulation topology logic (Stages I \& II) from the pure algebraic engine logic (Algorithms A \& B).

### C. Remove Algorithm Box Redundancy (O$'$ and C$'$)
**Current State:** The extensions section defines completely new algorithm boxes (Algorithm O$'$ and C$'$) just to insert the MMS plateau verifier lines.
**The Solution:** You already established an excellent visual convention: `\extline{}` (blue text) for extension logic. Instead of creating redundant boxes, present Algorithm O and Algorithm C exactly **once** in the core text, with the MMS hooks included but styled in `\extline{blue}`. Readers will immediately grasp that the blue lines only activate during MMS configurations, saving space and preventing the reader from having to "diff" two nearly identical algorithm boxes.

### D. Consolidate "Code-Equivalence" Dictionaries
**Current State:** Section 3 has tables mapping mathematical parameters to code fields. Appendix A has a table mapping exit reasons to string literals.
**The Solution:** Consolidate Appendix A into Section 3 to create a single, unified "Dictionary of Code Equivalences." This provides a single point of reference for anyone trying to map the mathematics to the Julia struct fields and string outputs.

### E. Minor Rigor Clarifications
1. **Clarify the Drift Metric Modes (Section 5.1):** The document describes `L2_mass` as computing two independent continuous $L^2$ norms via quadrature. It would be clearer and more rigorous to explicitly state the formulas using the continuous integral notation (e.g., $\sqrt{\int_\Omega (e_u \cdot e_u) d\Omega}$) to contrast it perfectly against the `Linf` discrete array norm.
2. **Explicit Mention of Anderson Safety Fallback:** Section 5.5 mentions an Anderson safety fallback, but it's not present in the algorithm box for Algorithm C. Since `update!` in `AndersonAccelerator` handles this natively in the code, adding a simple comment `\tcp*{handles safety fallback natively}` to the Anderson lines in Algorithm C would bridge the gap beautifully without cluttering the algorithm.
