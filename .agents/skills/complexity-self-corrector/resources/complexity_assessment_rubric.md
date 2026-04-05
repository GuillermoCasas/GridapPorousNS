# Complexity Assessment Rubric

Evaluate code changes explicitly against the following dimensions before proposing self-remediation suggestions.

## Positive Indicators (Good Complexity)
- Abstractions eliminating rigid duplicate code while mapping correctly to physical boundaries.
- Explicit `struct`/`trait` based dispatch mapping (e.g. `AbstractReactionLaw`).
- Decoupling solver iteration logic away from Continuous Formulation assembly matrices.
- Fast-test scaffolding capturing and mocking analytical bounds independently.
- Isolated, strictly tagged experimental code blocks (`is_experimental=true`).

## Negative Indicators (Harmful Complexity)
- **Mathematical Opacity**: Spreading a single analytical derivation (like `dtau/du`) across several disjoint unstructured closures.
- **Spaghetti Parameters**: Monolithic physical structs (`PhysicalParameters`) injected universally but only partially utilized by specific kernels.
- **Boolean Blindness**: Booleans functioning as abstract algorithm routing switches instead of formal typed logic structures (`is_picard`, `freeze_cusp`).
- **Unrolled Formulations**: Explicit tensor math unrolled blindly outside `continuous_problem.jl` making generic parameter validation impossible.
- **Divergent Test Boundaries**: Integration tests matching exact monolithic workflows scaling poorly with configuration variance logic limits.

## Suggestion Scope Limitations
- Always suggest the *minimum bounding intervention* needed to fix a complexity leak.
- **Never** propose eliminating Gridap memory-allocation structures correctly masking `CellField` evaluation caches.
- **Never** suggest flattening an experimental branch destructively into a "paper-faithful" structure.
- Treat reading side-by-side with the reference formulation PDF as a prime engineering benchmark.

## Risk Labels
Adhere to the following exact label taxonomy:
- `LOW RISK TO REFACTOR`
- `MODERATE RISK TO REFACTOR`
- `HIGH RISK TO REFACTOR`
- `HIGH RISK TO LEAVE AS-IS`
- `ACCEPTABLE COMPLEXITY`
- `JUSTIFIED SCIENTIFIC COMPLEXITY`
- `TEMPORARY COMPLEXITY, MONITOR CLOSELY`
