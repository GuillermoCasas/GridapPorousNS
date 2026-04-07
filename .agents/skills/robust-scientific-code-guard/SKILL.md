---
name: robust-scientific-code-guard
description: mandatory reasoning, verification, and anti-hallucination workflow for scientific computing, numerical methods, mathematical modeling, fem/pde solvers, research software, numerically sensitive refactors, and technical code review. use when translating equations into code, designing or refactoring algorithms, implementing solver logic, interpreting papers or technical references, reviewing scientific code, or working where hidden assumptions, api uncertainty, scaling issues, sign errors, tolerance misuse, or mathematical drift could produce believable-but-wrong implementations.
---

# Objective

Act as a mandatory reasoning-and-verification guardrail for scientific and technical coding tasks.

Use this skill to reduce:
- hallucinated APIs, equations, solver properties, and package behavior
- unjustified abstractions and premature architecture
- mathematically inconsistent implementations
- silent drift between the intended model and the written code
- believable-but-wrong code that appears polished but is weakly grounded
- hidden assumptions about scaling, boundary conditions, indexing, or tolerances
- refactors that accidentally change the mathematical problem

Treat this skill as a **process guard**, not a domain replacement.  
If a more specific skill applies, use this skill to force disciplined reasoning **before** and **after** that specialized work.

# Core Operating Mode

Be strict, skeptical, and conservative.

Optimize for:
- correctness
- traceability
- logical coherence
- mathematical consistency
- minimal valid design
- explicit uncertainty
- testability
- verifiability

Do not optimize for:
- sounding smooth
- producing code quickly at any cost
- clever abstractions
- speculative completeness
- filling gaps with plausible guesses

When evidence is weak, prefer:
- explicit uncertainty
- constrained partial implementation
- placeholders marked for verification
- a validation plan

Do not let fluency outrun evidence.

# Non-Negotiable Rules

1. Do **not** invent equations, APIs, package capabilities, file layouts, solver properties, or numerical theorems.
2. Do **not** imply that a paper or reference states something unless the provided material actually supports it.
3. Do **not** silently fill missing technical details; label them as assumptions, unknowns, or items needing verification.
4. Do **not** blur the distinction between:
   - a reference implementation
   - a paper-faithful implementation
   - an experimental or convenience variant
5. Do **not** jump straight into implementation when the mathematical or software contract is unclear.
6. Do **not** over-design before establishing the simplest design that matches the problem.
7. Do **not** trust code that lacks a validation path.
8. Do **not** weaken verification merely because the implementation is inconvenient to test.
9. Do **not** present speculative mathematical or numerical claims as settled facts.
10. If uncertainty remains, say exactly where it is and how it affects the code.

# Evidence Hierarchy

Prefer evidence in this order:

1. user-provided code, equations, tests, and documents
2. explicit project conventions already present in the repository
3. official library or package documentation actually available in context
4. standard domain knowledge clearly identified as such
5. assumptions, clearly labeled as assumptions
6. placeholders marked `[NEEDS VERIFICATION]`

Do not reverse this order.

# Mandatory Workflow

Follow this workflow in order whenever appropriate.

## 1. Restate the task precisely

Rewrite the task in precise technical language.

Identify:
- the mathematical objective
- the computational objective
- the software-engineering objective

Separate:
- stated facts
- inferred assumptions
- unknowns
- requested deliverables

If the task is a refactor, state whether the goal is:
- strict behavioral preservation
- mathematical correction
- paper-faithful alignment
- numerical-stability improvement
- performance improvement
- architectural cleanup

## 2. Write the contract before coding

Write three contracts explicitly.

### Mathematical contract
State:
- governing equations or algorithmic equations
- variables and their meaning
- boundary or initial conditions if relevant
- invariants, conserved quantities, monotonicity, symmetry, positivity, or consistency expectations
- units or nondimensional assumptions if relevant
- what is considered mathematically unchanged versus intentionally changed

### Algorithmic contract
State:
- what algorithm is being implemented
- what approximations or linearizations are used
- iteration or stopping criteria
- what is frozen, updated, projected, regularized, or stabilized
- which branch is authoritative if multiple formulation branches exist

### Software contract
State:
- inputs
- outputs
- expected types and shapes
- mutability expectations
- side effects
- configuration dependencies
- failure modes or invalid inputs

If any of these contracts are unclear, do not proceed as if they were settled.

## 3. Identify uncertainty explicitly

List all important uncertainties.

Typical uncertainty sources:
- missing formulas
- ambiguous notation
- unclear sign conventions
- unspecified units or nondimensional scaling
- missing boundary conditions
- unknown API names or call patterns
- missing data layout details
- unclear solver semantics
- unclear reference-vs-experimental intent
- unclear tolerance meaning
- missing expected behavior for edge cases

Use explicit language such as:
- `known`
- `assumed`
- `unknown`
- `needs verification`
- `paper claim not yet confirmed`
- `api unverified`

If a missing fact materially affects the implementation, say so.

## 4. Perform a design-review gate before coding

Review the design before writing code.

Check:
- whether the proposed architecture matches the actual problem
- whether abstractions are justified
- whether complexity is premature
- whether the design preserves the mathematical contract
- whether there is a simpler design that is easier to validate
- whether the code structure cleanly separates paper-faithful, legacy, and experimental branches when those distinctions matter

Prefer the simplest design that can be validated.

Reject or revise designs that:
- introduce abstraction without reducing verified complexity
- mix multiple mathematical branches implicitly
- obscure key invariants
- make later verification harder
- move uncertainty deeper into the code

## 5. Generate only grounded code

Write code only for parts supported by:
- the stated contracts
- provided references
- verified project conventions
- clearly labeled assumptions

When uncertainty remains:
- keep speculative parts isolated
- mark them clearly
- avoid pretending the implementation is complete
- prefer a constrained placeholder over a fabricated API call

If an API is not verified, do not write confident pseudo-real code that may be false.

Use explicit markers when helpful:
- `[ASSUMPTION]`
- `[UNKNOWN]`
- `[NEEDS VERIFICATION]`
- `[EXPERIMENTAL]`

Avoid ornamental abstractions and over-engineering.

## 6. Verify implementation against design

After writing code, compare the implementation against the stated contracts.

Check:
- variable meanings
- signs
- indexing
- array shapes
- loop logic
- boundary-condition handling
- stopping criteria
- scaling and units
- branch behavior
- config usage
- error handling
- edge cases

For mathematical code, explicitly ask:
- does each code term correspond to the intended equation term?
- is any term missing, duplicated, transposed, sign-flipped, or applied in the wrong branch?
- did the refactor preserve the mathematical role of each operator?
- are tolerances and thresholds consistent with the problem scale?
- is a “small cleanup” actually a model change?

If the implementation and design diverge, say so explicitly.

## 7. Enforce validation and tests

Require validation appropriate to the task.

For scientific and numerical code, require at least one test that would catch believable-but-wrong implementations.

Possible checks include:
- unit tests
- invariant checks
- conservation checks
- symmetry checks
- monotonicity checks
- consistency checks between strong and weak forms
- finite-difference derivative checks
- known-solution checks
- benchmark-case checks
- branch-separation checks
- failure-mode tests
- shape and dispatch smoke tests
- tolerance-sensitivity checks

Choose tests that target the actual risk introduced by the change.

Do not settle for a smoke test when the change threatens a mathematical identity.

## 8. Require a final risk review

Before concluding, summarize:

- what is certain
- what is assumed
- what remains unverified
- what could still be wrong
- what must be tested next before trusting the code broadly

If the result is only partially grounded, say so.

# Behavior by Task Type

## When designing new scientific code
- define the contracts first
- select the simplest valid design
- identify missing information before implementation
- expose assumptions early
- avoid speculative generalization

## When refactoring legacy numerical code
- distinguish behavior-preserving refactors from mathematical changes
- identify hidden assumptions already embedded in the code
- preserve or explicitly replace invariants
- compare old and new semantics, not just old and new syntax
- require targeted equivalence or invariant checks

## When translating mathematics into code
- map each code term to a mathematical term
- state sign conventions explicitly
- state what is discretized, approximated, regularized, or linearized
- separate exact theory from numerical implementation choices
- do not silently introduce approximations

## When reviewing or debugging code
- reconstruct the intended contract from the available evidence
- identify where the implementation departs from it
- prefer precise diagnosis over broad rewrites
- isolate whether the issue is mathematical, algorithmic, numerical, or software-structural

## When working from a paper or external reference
- distinguish what the reference explicitly states from what the code must still choose
- do not claim paper-faithfulness unless the mapping is actually checked
- call out any implementation-level gap not resolved by the reference

# Anti-Hallucination Rules

Always enforce the following:

- never invent functions, packages, APIs, file formats, or solver properties
- never imply evidence you do not have
- never treat an assumption as a cited fact
- never claim a theorem, stability result, or equivalence without support
- never silently choose boundary conditions, units, norms, or stopping criteria when they are underspecified
- never collapse distinct formulation branches into one story
- never present speculative numerical claims as if they were validated
- prefer traceability over fluency
- prefer explicit uncertainty over confident guessing

# Scientific-Computing Risk Factors

Be especially sensitive to:

- mismatch between mathematics and implementation
- inconsistent notation carried into code
- hidden assumptions
- boundary-condition ambiguity
- tolerance misuse
- scaling and normalization issues
- dimension or units mismatch
- numerical-stability risks
- silent regularization or stabilization changes
- refactors that unintentionally change the mathematical model
- code that looks plausible but lacks invariant checks
- API memory illusions, especially in specialized scientific libraries

# Validation Guidance by Change Type

Use these patterns when relevant.

## New operator or residual term
Require:
- term-by-term mapping to the contract
- at least one local or tiny-case consistency check
- sign and branch verification

## Jacobian or derivative change
Require:
- directional finite-difference comparison when feasible
- branch-specific checks for frozen versus active terms
- explicit confirmation of what is exact versus approximate

## Solver or globalization change
Require:
- logic checks for acceptance/rejection
- monotonicity or descent checks where relevant
- explicit verification of stopping criteria meaning

## Refactor-only change
Require:
- equivalence test or invariant preservation test
- confirmation that the code structure changed without changing the model, unless a model change is intentional

## Parameter or tolerance change
Require:
- scale justification
- explanation of expected effect of increase/decrease
- confirmation that the name, config location, and meaning remain coherent

# Minimal Output Structure

When the task is substantive, structure the response in this order unless there is a strong reason not to:

1. **Restated problem**
2. **Known facts**
3. **Assumptions / unknowns**
4. **Mathematical / algorithmic / software contract**
5. **Proposed minimal design**
6. **Design risks**
7. **Implementation**
8. **Verification against design**
9. **Validation plan**
10. **Remaining uncertainty**

For smaller tasks, compress the structure, but do not skip the logic behind it.

# Reusable Review Checklist

Before concluding any implementation step, verify:

- [ ] Is the task restated precisely?
- [ ] Are facts, assumptions, and unknowns separated?
- [ ] Is the mathematical contract explicit?
- [ ] Is the algorithmic contract explicit?
- [ ] Is the software contract explicit?
- [ ] Are any APIs or package features assumed rather than verified?
- [ ] Are boundary conditions and edge cases explicitly addressed?
- [ ] Does the implementation map cleanly to the intended equations or algorithmic steps?
- [ ] Are variables, dimensions, and units or scales coherent?
- [ ] Are tolerances and numerical thresholds justified by the problem scale?
- [ ] Is the proposed design the simplest one that can be validated?
- [ ] Is there at least one validation step that would catch believable-but-wrong code?
- [ ] Are remaining risks and unverified points stated explicitly?

# Compose with More Specific Skills

Use this skill as the guardrail layer.

If a more specific skill is also relevant, apply it **after** this skill has established:
- the contract
- the uncertainties
- the design gate
- the validation expectations

Examples:
- use a floating-point or numerical-stability skill after identifying the mathematical and numerical contract
- use a parameter/config discipline skill after identifying which values are true parameters
- use a fast verification skill after identifying the local invariants at risk

Do not let a specialized skill bypass the reasoning discipline required here.

# Examples of Skill Intervention

## Before coding: design gate
**Task:** add a stabilization term to a Navier–Stokes solver using SUPG.

Required intervention:
- restate the task precisely
- identify that SUPG requires a residual form and a definition of the stabilization parameter `τ`
- block immediate implementation if `τ` is unspecified
- state the unknown explicitly
- propose a candidate definition only as a reviewed assumption, not as a fact

Good behavior:
“Before implementing SUPG, define the residual form, the `τ` formula, and the branch conditions under which it applies. The user has not specified the reference definition of `τ`, so that is currently an explicit unknown. I can propose a standard choice for review, but I should not silently encode one as if it were mandated.”

## During coding: anti-hallucination
**Task:** implement a specialized matrix assembly path when the exact library API is unclear.

Required intervention:
- do not invent a likely-looking API call
- isolate the unverified part
- mark it for verification
- keep the grounded part of the implementation separate

Good behavior:
“I can implement the algebraic structure of the assembly, but the exact Gridap API call is not verified from the provided material. I will keep the mathematical part explicit and label the assembly hook as `[NEEDS VERIFICATION: API]` rather than fabricate a function name.”

## After coding: risk review
**Task:** refactor a nonlinear problem setup and Jacobian assembly.

Required intervention:
- summarize what is definitely preserved
- state what remains risky
- propose the next validation step
- do not declare the code fully trustworthy without derivative validation

Good behavior:
“The refactor preserves the intended data flow and config interface, but there is still a high risk of sign or missing-term errors in the Jacobian. Before trusting this in production sweeps, compare the implemented Fréchet derivative against directional finite differences of the residual on a tiny case.”

# Final Rule

Do not write scientific code as if uncertainty were harmless.

Think:

**restate -> separate facts from assumptions -> write the contracts -> review the design -> generate only grounded code -> verify against the contracts -> require validation -> state remaining risk**