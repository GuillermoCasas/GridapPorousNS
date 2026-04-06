---
name: porousns-fast-verification
description: enforce ultra-fast invariant-based verification for porousnssolver refactors. use when editing continuous_problem.jl, viscous_operators.jl, tau.jl, projection.jl, reaction.jl, regularization.jl, nonlinear.jl, fast test files, fe-space assembly logic, or formulation flags. require tiny-mesh or local-state tests that validate operator compatibility, adjoint identities, tau derivatives, projection legality, linearization behavior, and globalization logic without running full pde solves, nonlinear benchmark solves, or convergence studies.
---

# Objective

Enforce an ultra-fast continuous verification loop for mathematically sensitive parts of the PorousNSSolver refactor.

Use this skill to catch **local mathematical regressions immediately**, especially regressions involving:
- strong vs weak operator mismatch
- broken adjoint formulas
- inconsistent stabilization terms
- incorrect `tau` or `dtau` derivatives
- invalid projection-policy / reaction-law combinations
- regularization-induced cusps or singularities
- accidental drift between `ExactNewtonMode` and `PicardMode`
- broken Armijo / merit-function logic
- Gridap assembly or dispatch breakage after refactors

This skill exists because small local changes in a stabilized FEM codebase can silently change the mathematical meaning of the method long before a slow MMS or benchmark run exposes the problem.

# Core Principle

Test the **invariant directly**, not indirectly through an expensive solve.

When a mathematical component changes, do not rely on:
- full PDE solves
- mesh-refinement studies
- MMS convergence loops
- long nonlinear runs
- broad integration tests that hide the source of failure

Instead:
1. identify the invariant the change can break
2. create or update a tiny targeted test for that invariant
3. run the fast suite immediately
4. treat failures as mathematical debugging tasks, not test-maintenance annoyances

# Non-Negotiable Rules

1. Do **not** add MMS solves, convergence loops, or full nonlinear benchmark runs to this fast suite.
2. Keep the fast suite **tiny-mesh and local-state based** wherever possible.
3. Target **under about 2 seconds after Julia compilation is warm** for the whole fast suite.
4. Every mathematical change must add or update at least one new fast invariant test.
5. Treat `PaperGeneralFormulation` as mathematically strict.
6. Do not let experimental paths silently weaken or redefine the invariants of `PaperGeneralFormulation`.
7. Do not weaken or delete an invariant test merely because the implementation changed. First determine whether the mathematics changed intentionally and whether the test must be rewritten to express the new invariant.
8. If a baseline fast test is already failing before the change, do not paper over it. Isolate the failure, document it, and avoid broad rewrites that hide the original issue.

# What This Fast Suite Is For

This suite is for:
- operator consistency
- Jacobian correctness
- adjoint correctness
- derivative checks against finite differences
- projection legality
- regularization safety
- tiny-mesh assembly integrity
- linearization-mode separation
- globalization logic correctness

This suite is **not** for:
- proving convergence rates
- validating final physical accuracy
- replacing MMS or Cocquet-style studies
- performance benchmarking of full simulations

# Activation Conditions

Use this skill whenever you modify any of the following, or any helper they depend on:

- `src/formulations/continuous_problem.jl`
- `src/formulations/viscous_operators.jl`
- `src/stabilization/tau.jl`
- `src/stabilization/projection.jl`
- `src/models/reaction.jl`
- `src/models/regularization.jl`
- `src/solvers/nonlinear.jl`

Also use it when modifying:
- formulation flags or branch-selection logic
- FE-space construction or tiny-mesh assembly logic
- test helpers used by the fast verification layer
- shared helper functions that can affect both paper-faithful and experimental branches

# Required Workflow

## 1. Run pre-flight baseline
Run the existing fast suite before modifying tests.

Do not start by rewriting tests blindly.

Establish whether:
- the baseline is green
- there are pre-existing unrelated failures
- the touched component already has an invariant test that should be extended rather than replaced

If the baseline is already red for unrelated reasons, note it explicitly and isolate your change.

## 2. Classify the change
Before coding, classify the change into one or more of these categories:

- strong/weak operator change
- adjoint operator change
- `tau` or `dtau` change
- projection policy change
- reaction-law change
- regularization change
- nonlinear globalization change
- linearization-mode change
- dimension / FE-space / assembly change
- experimental-branch gating change
- paper-vs-experimental branch separation change

## 3. Write down the invariant at risk
Before implementing the test, state the invariant that must remain true.

Examples:
- strong residual and weak form are compatible
- the adjoint corresponds to the implemented strong operator under the stated boundary assumptions
- `dtau` matches finite differences in the requested direction
- `freeze_cusp` zeros only the intended derivative contributions
- constant-sigma-only projection is rejected for Forchheimer
- `PicardMode` freezes only the intended terms
- Armijo uses the correct directional derivative of `Φ(x)=1/2‖F(x)‖²`
- paper-faithful and experimental branches remain distinct

If you cannot state the invariant precisely, do not write the test yet.

## 4. Add or update at least one micro-test
Every mathematical change must add or update at least one ultra-fast test.

Prefer:
- local-state tests with hand-computable values
- `(1,1)` or `(2,2)` `CartesianDiscreteModel` meshes
- direct field evaluation or direct assembly
- finite-difference directional checks
- positive-case and negative/perturbed-case checks when feasible

Avoid solves unless the solve is tiny, local, and directly necessary to verify the invariant.

## 5. Run the fast suite immediately
After the change and the micro-test are in place:
1. run the fast suite
2. inspect any failure mathematically
3. fix the implementation or the test
4. rerun until green

Do not defer failures to slow suites.

## 6. Perform final audit
Before calling the change complete, verify:
- the relevant fast test exists
- the test is actually targeted to the invariant
- the test would fail if the invariant were broken in the intended way
- no experimental behavior silently alters `PaperGeneralFormulation`
- no test was weakened without mathematical justification

# Test Authoring Rules

## General rules
- Use tiny meshes only.
- Prefer exact or semi-exact local states over full model runs.
- Use relative-plus-absolute tolerances.
- Choose quadrature/order carefully enough to avoid false negatives from underintegration.
- Seed any randomness explicitly, but prefer deterministic constructions.
- Name `@testset`s after the mathematical invariant, not the source file.
- Keep one test focused on one invariant when possible.
- Prefer one positive case and one deliberately perturbed case when practical.
- Keep failure messages mathematically legible.

## Tolerance rules
Use tolerances appropriate to the quantity being checked:
- machine-level for purely local algebraic identities
- looser for quadrature-based or assembled identities
- finite-difference tolerances that account for truncation and roundoff

Do not use a single blanket tolerance everywhere.

When checking finite differences:
- choose perturbation size deliberately
- avoid values that are too large to reflect the local derivative
- avoid values that are too small and dominated by roundoff

## Negative-case rule
When practical, add a nearby wrong or perturbed construction to verify the test would actually fail under the targeted regression.

This is especially valuable for:
- strong residual exactness
- weak-vs-strong compatibility
- adjoint relations
- projection legality
- branch-separation tests

# Project-Specific Fast Test Patterns

## A. Canonical strong residual exactness test

Use a tiny mesh and analytic fields `u`, `p`, `α`, `f`, `g`. Assemble or evaluate the strong residual from `eval_strong_residual_u` and `eval_strong_residual_p` and verify that:

- if `f` and `g` are built from the same operator, the residual is near machine precision
- if one operator term is deliberately perturbed, the residual departs clearly from zero

Purpose:
- catch inconsistencies between the canonical residual and forcing construction
- protect against mismatches where one branch uses a different strong operator inside stabilization than the canonical formulation uses outside it

Use whenever changing:
- strong residual structure
- canonical operator definitions
- branch-specific operator selection

## B. Weak-vs-strong consistency test

For each viscous operator branch being modified, verify on a tiny mesh that the assembled weak form matches the dual action of the strong operator on a smooth enough trial/test pair, up to quadrature tolerance and the stated boundary assumptions.

Purpose:
- protect `viscous_operators.jl`
- ensure the weak form actually corresponds to the chosen strong operator

This is mandatory whenever modifying:
- `strong_viscous_operator`
- `weak_viscous_operator`
- `weak_viscous_jacobian`
- boundary-assumption-sensitive helper logic

## C. Adjoint directional identity test

For a chosen viscous operator and smooth fields, verify the discrete adjoint identity corresponding to:
\[
\langle L u, v \rangle \approx \langle u, L^* v \rangle
\]
under the boundary assumptions used in the derivation.

Purpose:
- catch incorrect adjoint formulas in stabilization
- protect `PaperGeneralFormulation` in particular

This is mandatory whenever modifying:
- `adjoint_viscous_operator`
- strong/weak operator pairings used inside stabilized terms
- adjoint-related helpers in `continuous_problem.jl`

## D. `tau` finite-difference directional derivative test

For representative local states, check:
- `compute_dtau_1_du`
- `compute_dtau_2_du`

against finite differences in a chosen direction.

Test at least:
- moderate velocity
- near-zero velocity with regularization active
- high-Re-like state
- high-Da-like state

Also test:
- `freeze_cusp=true` returns the intended zero derivative behavior and only that behavior

Purpose:
- protect against chain-rule mistakes
- catch cusp issues and wrong dependence on regularized speed
- catch accidental changes in derivative logic after refactors

This is mandatory whenever changing:
- `tau.jl`
- reaction laws that enter `sigma`
- regularization laws used in effective speed
- flags that alter derivative freezing

## E. Projection legality test

Verify that:
- `ProjectResidualWithoutReactionWhenConstantSigma` is accepted for `ConstantSigmaLaw`
- the same policy errors cleanly for `ForchheimerErgunLaw`
- `ProjectFullResidual` remains valid for all reaction laws

If an autocorrect helper exists, verify that:
- invalid combinations map explicitly to `ProjectFullResidual()`
- the mapping is not silent if a warning or note is part of the intended behavior

Purpose:
- prevent mathematically invalid policy/law combinations from silently entering OSGS runs

## F. Reaction-law derivative sanity test

For each reaction law:
- verify `sigma` on a hand-computable state
- verify `dsigma_du` against finite differences in one direction

This is especially important for reaction laws that depend on regularized speed.

Purpose:
- protect `reaction.jl`
- ensure derivative logic remains consistent with the value logic

## G. Regularization smoothness and safety test

For `NoRegularization` and `SmoothVelocityFloor`, verify:
- value of `effective_speed`
- derivative behavior near zero velocity
- absence of `NaN`/`Inf` in states used by tau and reaction evaluations
- monotonic or at least directionally sensible behavior near the floor when such behavior is intended

Purpose:
- catch Newton-destructive cusps
- catch divide-by-zero pathways
- protect any local state used by `tau` and `sigma`

## H. Tiny-mesh assembly smoke test

On a `(1,1)` or `(2,2)` mesh:
- build FE spaces
- assemble stabilized residual and Jacobian for both `PaperGeneralFormulation` and `PseudoTractionFormulation`
- verify assembly succeeds
- verify no dispatch/type errors
- verify matrix/vector sizes are consistent

Do not solve.

Purpose:
- fast protection against Gridap dispatch breakage after refactors
- catch signature drift, field-type mismatches, or assembly path breakage

When relevant, exercise both:
- `ExactNewtonMode`
- `PicardMode`

## I. Linearization-mode separation test

Verify on a tiny setup that:
- `ExactNewtonMode` includes the intended Jacobian contributions
- `PicardMode` freezes only the intended terms
- the two matrices differ where expected
- the two matrices coincide where they should

If the implementation uses flags like `is_picard` or equivalent, test their effect directly.

Purpose:
- prevent accidental blanket zeroing
- prevent unrelated Jacobian pieces from being frozen
- protect exact-vs-Picard linearization semantics

## J. Merit-function globalization test

For the nonlinear solver:
- define a tiny synthetic nonlinear operator or a tiny assembled operator
- verify the Armijo condition uses the actual directional derivative of
  \[
  \Phi(x)=\tfrac12\|F(x)\|^2
  \]
  along the proposed Newton direction
- verify a descent step is accepted
- verify an ascent step is rejected
- verify termination or failure logic around minimum step size behaves as intended

Purpose:
- protect `nonlinear.jl` from regressions in line-search logic
- ensure globalization behavior is mathematically coherent

## K. Paper-vs-experimental branch separation test

Whenever a helper or shared path affects both `PaperGeneralFormulation` and `PseudoTractionFormulation`, verify that:
- the paper-faithful branch still uses the intended strict operators
- the experimental branch can differ where intended
- no shared refactor silently forces the paper branch to inherit an experimental shortcut

Purpose:
- prevent accidental branch collapse
- protect the mathematical strictness of `PaperGeneralFormulation`

This is especially important when modifying:
- shared helpers in `continuous_problem.jl`
- branch selection logic
- viscous operator dispatch
- stabilization or adjoint helper reuse

# Required Test Mapping by File

When you change a file, add or update at least these fast tests:

- If changing `continuous_problem.jl`: add or update **A**, **H**, and if stabilization or linearization changed also **C**, **I**, or **K**
- If changing `viscous_operators.jl`: add or update **B**, **C**, and often **K**
- If changing `tau.jl`: add or update **D**
- If changing `projection.jl`: add or update **E**
- If changing `reaction.jl`: add or update **F** and usually **D**
- If changing `regularization.jl`: add or update **G** and usually **D** and/or **F**
- If changing `nonlinear.jl`: add or update **J** and a tiny solver-setup smoke test
- If changing shared branch-selection helpers or formulation flags: add or update **H**, **I**, and **K**

# Implementation Guidance

Organize the fast suite in dedicated files, for example:
- `test/fast/test_fast_formulation.jl`
- `test/fast/test_fast_viscous_operators.jl`
- `test/fast/test_fast_tau.jl`
- `test/fast/test_fast_projection.jl`
- `test/fast/test_fast_reaction_regularization.jl`
- `test/fast/test_fast_nonlinear.jl`

Within those files:
- group by invariant, not by source file diff
- keep tests short and surgical
- isolate tiny helper constructors for local states and tiny models
- reuse deterministic tiny-mesh setup helpers when useful
- prefer explicit field choices over opaque random fields

# Anti-Patterns to Avoid

Do not:
- add a slow solve because it is easier than isolating the invariant
- use a large mesh to “be safe”
- test too many invariants in one testset
- replace a mathematical test with a smoke test
- loosen tolerances until the test passes without understanding why
- hide branch differences by over-sharing helpers
- assume that a green slow solve means the local operator identity is correct
- delete a failing invariant test without replacing it with a sharper one

# Execution and Self-Correction

After implementing the code change and the new fast test:

1. run the fast suite
2. inspect any failure mathematically
3. decide whether the failure means:
   - the implementation is wrong
   - the test expresses the old invariant and the invariant intentionally changed
   - the test setup or tolerance is invalid
4. fix the implementation or the test accordingly
5. rerun until green

If an invariant truly changed intentionally, update the test so it expresses the **new** invariant precisely. Do not simply weaken the old assertion.

# Completion Rule

A change is complete only when:

- the code builds
- all existing fast tests pass
- at least one new or updated invariant test exists for the modified mathematical component
- the new test directly targets the invariant at risk
- no experimental behavior has silently altered the invariants of `PaperGeneralFormulation`
- the fast suite remains fast enough to serve as a real continuous verification loop

# Final Mental Checklist

Before finishing, ask:

- what invariant did this change threaten?
- where is that invariant now tested?
- would the test fail if the bug reappeared?
- did I verify the paper-faithful branch separately from experimental behavior when needed?
- did I add a direct mathematical test rather than relying on a future slow run?

Think:

**change -> classify invariant -> add tiny direct test -> run fast suite -> inspect mathematically -> keep paper branch strict**