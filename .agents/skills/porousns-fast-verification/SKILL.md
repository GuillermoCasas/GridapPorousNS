---
name: porousns-fast-verification
description: Enforces an ultra-fast continuous verification loop for the PorousNSSolver refactor. Use whenever modifying formulations, viscous operators, stabilization parameters, projection policies, reaction laws, regularization, FE spaces, or nonlinear solver code. Focuses on operator consistency, Jacobian correctness, and formulation invariants in Gridap.jl without running full PDE solves or convergence studies.
---

# Continuous Verification Protocol: PorousNSSolver Fast Tests

Whenever you modify any of the following parts of the codebase, you must add or update ultra-fast verification tests and run them before considering the change complete:

* `src/formulations/continuous_problem.jl`
* `src/formulations/viscous_operators.jl`
* `src/stabilization/tau.jl`
* `src/stabilization/projection.jl`
* `src/models/reaction.jl`
* `src/models/regularization.jl`
* `src/solvers/nonlinear.jl`

The current codebase has already shown that small local inconsistencies can break the mathematical meaning of the stabilized method, especially in the relation between strong residuals, weak forms, adjoints, tau derivatives, projection policies, and nonlinear globalization. The fast suite must therefore check those invariants directly, not indirectly through expensive solves.

## Critical rules

* Do **not** add MMS solves, mesh-refinement loops, or full nonlinear benchmark runs to this fast suite.
* The suite must run in under about 2 seconds after Julia compilation is warm.
* Every mathematical change must be accompanied by at least one new fast invariant test.
* The `PaperGeneralFormulation` must be treated as mathematically strict. Experimental approximations must never silently alter its invariants.

## Required workflow

### 1. Pre-flight

Run the existing fast suite first. Do not start modifying tests blindly. Confirm the baseline is green.

### 2. Identify the invariant affected by the change

Before coding, classify the change into one or more of these categories:

* **Strong/weak operator change**
* **Adjoint operator change**
* **Tau or dtau change**
* **Projection policy change**
* **Reaction/regularization change**
* **Nonlinear globalization or linearization-mode change**
* **Dimension/FE-space/assembly change**

Then write down the invariant that must remain true.

Examples:

* strong residual and weak form are compatible,
* adjoint corresponds to the implemented strong operator,
* `dtau` matches finite differences,
* constant-sigma-only projection is rejected for Forchheimer,
* Picard mode freezes the intended terms and only those terms,
* Armijo decrease uses the correct merit directional derivative.

### 3. Add at least one targeted micro-test

Every change must add or update at least one ultra-fast test. Prefer tests on a tiny mesh such as `CartesianDiscreteModel` with `(1,1)` or `(2,2)` partitioning, and avoid solves unless it is a single tiny linear algebra check.

Use one or more of the following project-specific test patterns.

---

## Fast test patterns for this project

### A. Canonical strong residual exactness test

Use a tiny mesh and analytic fields `u`, `p`, `α`, `f`, `g`. Assemble the strong residual from `eval_strong_residual_u` and `eval_strong_residual_p` and verify that:

* if `f` and `g` are built from the same operator, the residual is near machine precision,
* if one operator term is deliberately perturbed, the test fails clearly.

Purpose:

* catches inconsistencies between the canonical residual and the forcing construction,
* protects against reintroducing the old mismatch where one branch used a different strong operator inside stabilization.

### B. Weak-vs-strong consistency test

For each viscous operator branch being modified, verify on a tiny mesh that the assembled weak form equals the dual action of the strong operator on a smooth enough trial/test pair up to quadrature tolerance.

Purpose:

* protects `viscous_operators.jl`,
* ensures the implemented weak form really comes from the chosen strong operator.

This is mandatory whenever modifying:

* `strong_viscous_operator`
* `weak_viscous_operator`
* `adjoint_viscous_operator` 

### C. Adjoint directional identity test

For a chosen viscous operator and smooth fields, verify the discrete identity corresponding to the adjoint relation:
[
\langle L u, v \rangle \approx \langle u, L^* v \rangle
]
under the boundary assumptions used in the derivation.

Purpose:

* catches incorrect adjoint formulas in stabilization,
* especially important for `PaperGeneralFormulation`.

### D. `tau` finite-difference directional derivative test

For representative local states, check:

* `compute_dtau_1_du`
* `compute_dtau_2_du`

against finite differences using a scalar directional perturbation.

Test at least:

* moderate velocity,
* near-zero velocity with regularization active,
* high-Re-like state,
* high-Da-like state.

Purpose:

* protects against chain-rule mistakes and cusp issues,
* mandatory whenever changing `tau.jl`, reaction laws, or regularization.

### E. Projection legality test

Verify that:

* `ProjectResidualWithoutReactionWhenConstantSigma` is accepted for `ConstantSigmaLaw`,
* the same policy errors cleanly for `ForchheimerErgunLaw`,
* `ProjectFullResidual` remains valid for all reaction laws.

If an autocorrect helper exists, verify that invalid combinations map explicitly to `ProjectFullResidual()` and emit a warning.

Purpose:

* prevents mathematically invalid policy/law combinations from silently entering OSGS runs.

### F. Reaction-law derivative sanity test

For each reaction law:

* verify `sigma` value on a hand-computable state,
* verify `dsigma_du` against finite differences in one direction.

This is especially important for `ForchheimerErgunLaw` when regularized speed is involved.

### G. Regularization smoothness test

For `NoRegularization` and `SmoothVelocityFloor`, verify:

* value of `effective_speed`,
* derivative behavior near zero velocity,
* absence of NaN/Inf in states used by tau and reaction evaluations.

Purpose:

* catches Newton-destructive cusps and division-by-zero pathways.

### H. Tiny-mesh assembly smoke test

On a `(1,1)` or `(2,2)` mesh:

* build FE spaces,
* assemble the stabilized residual and Jacobian for both `PaperGeneralFormulation` and `PseudoTractionFormulation`,
* verify that assembly succeeds with no dispatch/type errors,
* verify matrix/vector sizes are consistent.

Do not solve.

Purpose:

* fast protection against Gridap dispatch breakage after refactors.

### I. Linearization-mode test

If linearization modes exist, verify on a tiny setup that:

* `ExactNewtonMode` includes the expected Jacobian contributions,
* `PicardMode` freezes only the intended terms,
* the two matrices differ where expected and coincide where they should.

Purpose:

* prevents accidental blanket zeroing of unrelated Jacobian pieces. This is especially relevant given the current use of `is_picard` logic in the Jacobian assembly. 

### J. Merit-function globalization test

For the nonlinear solver:

* define a tiny synthetic nonlinear operator or a tiny assembled operator,
* verify the Armijo condition uses the actual directional derivative of
  [
  \Phi(x)=\tfrac12|F(x)|^2
  ]
  along the Newton direction,
* verify that a descent step is accepted and an ascent step is rejected.

Purpose:

* protects `nonlinear.jl` from regressions in line search logic. 

---

## Project-specific required test mapping

When you change a file, add at least these fast tests:

* If changing `continuous_problem.jl`: add/update A, H, and if stabilization changed also C or I.
* If changing `viscous_operators.jl`: add/update B and C.
* If changing `tau.jl`: add/update D.
* If changing `projection.jl`: add/update E.
* If changing `reaction.jl`: add/update F and usually D.
* If changing `regularization.jl`: add/update G and usually D/F.
* If changing `nonlinear.jl`: add/update J and a tiny smoke test of solver setup.

## Implementation guidance

* Put the fast suite in a dedicated test file group such as:

  * `test/fast/test_fast_formulation.jl`
  * `test/fast/test_fast_viscous_operators.jl`
  * `test/fast/test_fast_tau.jl`
  * `test/fast/test_fast_projection.jl`
  * `test/fast/test_fast_reaction_regularization.jl`
  * `test/fast/test_fast_nonlinear.jl`
* Use tiny meshes only.
* Prefer exact or semi-exact local states over full model runs.
* Use relative-plus-absolute tolerances.
* Name tests after mathematical invariants, not source files.

## Execution and self-correction

After implementing the code change and the new fast test:

1. run the fast suite,
2. inspect any failure mathematically,
3. fix the implementation or the test,
4. rerun until all fast tests pass.

Do not defer failures to the slow MMS suite. The whole point of this protocol is to catch local mathematical regressions immediately.

## Completion rule

A change is complete only when:

* the code builds,
* all existing fast tests pass,
* at least one new invariant test has been added or updated for the modified mathematical component,
* no experimental behavior has silently altered the invariants of `PaperGeneralFormulation`.
