# Traceability Rubric

## Purpose
The paper-to-code traceability matrix securely locks mathematical expectations to algorithmic execution footprints. It acts as the canonical source of truth for solver integrity guarantees.

## File Location
The table should be maintained in `docs/paper_traceability.md`. If this file does not exist, the skill MUST instruct the user to create it or create it securely during Remediation mode.

## ID Generation
Assign stable, consecutive IDs mapping to `P-XXX`. 
- Check the `docs/paper_traceability.md` file dynamically to identify the last used `P-XXX`.
- Assign `P-001`, `P-002`, incrementing smoothly.
- Never drop an ID once registered; update its `Remediation Status` to `Resolved` or `Deprecated`.

## Matrix Columns

| Column | Description |
|---|---|
| **Issue ID** | `P-XXX` strict consecutive assignment. |
| **Paper Ref** | Section 4.1, Eq 17b, Algorithm A. |
| **Code Location** | `src/stabilization/tau.jl:145` or specific struct definitions. |
| **Status** | Checked & Passing, Checked & Failing, Unchecked, Needs Derivation. |
| **Severity** | CRITICAL INCONSISTENCY, LIKELY BUG, etc. |
| **Confidence** | Low, Medium, High, Absolute. |
| **Impact** | Physical outcome of the deviation (Stagnation, Oscillation, Divergence). |
| **Invariant** | The mathematical truth that should hold. |
| **Test Category** | Operator Consistency, Adjoint Exactness, Assembly Smoke, Limits. |
| **Recommended Test** | Minimum functional test script to seal the boundary (`test_fast_*.jl`). |
| **Remediation Status** | Resolved, Unresolved, Investigating. |

## Repository Rule
**"No paper-related change is complete unless the traceability matrix and at least one matching fast test are updated."**

The auditor must append this rule to reports when detecting unprotected code structures.
