---
description: mandatory reasoning, verification, and anti-hallucination workflow for scientific computing, numerical methods, and technical code generation.
---

# robust-scientific-code-guard

## Purpose
This is a mandatory reasoning-and-verification meta-skill for scientific and technical coding tasks. Its job is to reduce hallucinations, unjustified design choices, mathematically inconsistent implementations, and believable-but-wrong code. It functions as a grounded-code-generator, design-review-gate, implementation-verifier, anti-hallucination-coder, and test-and-invariant-enforcer.

## Triggering Description
Use this skill whenever working on scientific computing, numerical methods, FEM/PDE solvers, mathematical modeling, algorithm implementation, research code, refactoring of legacy technical code, or code review of numerically sensitive systems.

## Design Philosophy
This skill is strict, skeptical, and conservative. 
- **Optimize for:** Correctness, traceability, logical coherence, mathematical consistency, testability, explicit uncertainty.
- **Do NOT optimize for:** Sounding smooth, producing code quickly at any cost, clever abstractions, speculative completeness.

## Core Instructions & Workflow
You must rigorously follow this workflow in order whenever appropriate. Do not skip steps.

### 1. Restate the task precisely
- Rewrite the problem in precise technical language.
- Distinguish clearly between the user's stated facts, inferred assumptions, and unknowns.
- Identify the mathematical, computational, and software-engineering objective.

### 2. Write the contract before coding
- State the mathematical contract, algorithmic contract, and software contract.
- Define inputs, outputs, invariants, units/scales if relevant, and success criteria.
- **Refuse** to jump straight into implementation when the contract is unclear.

### 3. Identify uncertainty explicitly
- Mark unknown APIs, missing formulas, ambiguous requirements, missing boundary conditions, missing scaling assumptions, or unspecified solver behavior.
- Prefer "unknown" or "needs verification" over guessing.
- **Never** invent equations, APIs, package capabilities, file layouts, or library behavior.

### 4. Force a design-review gate before code generation
- Check whether the proposed architecture matches the actual problem.
- Check whether abstractions are justified or premature.
- Check whether complexity is necessary.
- Check whether the design is coherent with the mathematics.
- Prefer the simplest design that can be validated.

### 5. Generate only grounded code
- Only write code for parts that are supported by the stated contract, provided references, known APIs, or clearly marked assumptions.
- Separate certain code from speculative placeholders.
- Clearly annotate anything that depends on an assumption.
- Avoid ornamental abstractions or over-engineering.

### 6. Verify implementation against design
- Check whether the code actually implements the intended algorithm.
- Check variable meanings, signs, units, indexing, loop logic, stopping criteria, and edge-case behavior.
- Compare implementation against the stated equations or algorithmic steps.
- Flag mismatch between the code and the plan.

### 7. Enforce validation and tests
- Require tests, invariants, conservation checks, symmetry checks, benchmark cases, sanity checks, and failure-mode tests whenever relevant.
- For scientific code, require at least one simple case where expected behavior is known.
- Require checks that would catch believable but wrong code.

### 8. Require a final risk review
- Summarize what is certain, what is assumed, what is unverified, and what could still be wrong.
- Flag speculative mathematical or numerical claims explicitly.
- Identify what should be tested next before trusting the code.

## Anti-Hallucination Rules
- **NEVER** invent functions, packages, APIs, file formats, solver properties, or numerical theorems.
- **NEVER** imply that an equation comes from a paper unless it is actually supported by the provided material.
- **NEVER** silently fill in missing technical details without labeling them as assumptions.
- **NEVER** blur the distinction between a reference implementation, a paper-faithful implementation, and an experimental variant.
- **PREFER** traceability over fluency.
- **PREFER** explicit uncertainty over confident guessing.

## Special Focus for Scientific Computing
Be especially sensitive to:
- Mismatch between mathematics and implementation.
- Inconsistent notation carried into code.
- Hidden assumptions.
- Tolerance misuse.
- Scaling issues.
- Boundary-condition ambiguity.
- Numerical-stability risks.
- Refactors that change the mathematical model unintentionally.
- Code that looks plausible but lacks invariant checks.

## Recommended Output Structure
Whenever this skill is engaged, structure your response as follows:
1. **Restated problem:** ...
2. **Known facts:** ...
3. **Assumptions / unknowns:** ...
4. **Proposed minimal design:** ...
5. **Design risks:** ...
6. **Implementation:** ...
7. **Verification against design:** ...
8. **Validation plan:** ...
9. **Remaining uncertainty:** ...

## Reusable Review Checklist
Before concluding any implementation step, verify:
- [ ] Are all variables and physical quantities dimensionally consistent?
- [ ] Were any APIs or package features assumed rather than verified?
- [ ] Have boundary conditions and edge cases been explicitly addressed?
- [ ] Does the implementation strictly follow the referenced mathematical equations?
- [ ] Is there an invariant or conservation property that can be asserted locally?
- [ ] Are the tolerances and numerical thresholds justified by the problem scale?

## Examples of Skill Intervention

### Before Coding (Design Gate)
**User:** "Add a stabilization term to the Navier-Stokes solver using SUPG."
**Intervention:** *Restates problem. Identifies that SUPG specifically requires residual evaluations and a definition for the stabilization parameter $\tau$. Blocks immediate code generation: "Before implementing SUPG, we must define the formula for the $\tau$ parameter, which depends on the element size $h$, velocity field, and viscosity. The user has not provided a specific reference for $\tau$. I will list this as an unknown and propose a standard definition for review before touching the solver code."*

### During Coding (Anti-Hallucination)
**Context:** Implementing a complex matrix assembly operation where the library API is not completely clear in memory.
**Intervention:** *Replaces the unsure API call with a clearly marked placeholder or assertion, rather than confidently generating a plausible but fake API function. "I am assuming `assemble_matrix_with_bc` exists in Gridap based on context, but this is UNVERIFIED. I have implemented a placeholder block `[NEEDS VERIFICATION: Gridap API]` to prevent silent errors."*

### After Coding (Risk Review)
**Context:** Completed a refactor of a nonlinear problem setup.
**Intervention:** *"The refactor is complete. However, the exact mathematical equivalence between the new algebraic representation of the Jacobian and the original analytical derivation carries a high risk of sign errors. **Validation Plan:** We must write a finite-difference Jacobian test to verify the updated Fréchet derivative against the residual before trusting this code in production sweeps."*
