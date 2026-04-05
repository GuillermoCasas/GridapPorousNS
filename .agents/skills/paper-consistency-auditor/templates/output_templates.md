# Output Templates

When auditing code and presenting your findings to the user, you MUST rigorously structure your response using the exact sections below.

## 1. ALERT SUMMARY
Start your response with a high-visibility summary of all issues found.

```text
# 🚨 ALERT SUMMARY 🚨

**Audit Scope:** [Files or concepts analyzed]
**Traceability/Test Mandate:** The codebase has been marked for inconsistencies. Traceability entries and tests are explicitly REQUIRED for resolution.

### Issue Counts by Severity
- [CRITICAL INCONSISTENCY]: <count>
- [DOCUMENTED APPROXIMATION]: <count>
- [UNDOCUMENTED DEVIATION]: <count>
- [LIKELY BUG]: <count>
- [NEEDS DERIVATION]: <count>

### Highlighted Findings
- **P-00X**: [Short description of critical issue]
- ...
```

## 2. TRACEABLE EXPLANATION
For every issue found, output a detailed explanation block.

```text
### Issue ID: [P-XXX]
**Paper Reference:** [e.g., Equation 34, Algorithm B, Section 5]
**Code Location:** [File path, function, or line number]

- **Severity:** [From classification framework]
- **Consistency Status:** [e.g., Checked & Failing]
- **Confidence:** [Low/Medium/High/Absolute]

**Logic Followed:**
[Step-by-step reasoning. Which mathematical statement did you use? What code did you compare? What assumptions did you make to reach this judgment?]

**Why this is/is not consistent:**
[Detailed rigorous mathematical divergence explanation.]

**Minimal Correction (if in Remediation Mode):**
[Code snippet showing the exact, narrowly scoped fix required.]
```

## 3. TRACEABILITY MATRIX ENTRY TEMPLATE
Instruct the user to add this row to `docs/paper_traceability.md`, or modify it directly if you have access.

```text
**Traceability Entry [P-XXX]:**
| Issue ID | Paper Ref | Code Location | Status | Severity | Confidence | Impact | Invariant | Test Category | Recommended Test | Remediation Status |
|---|---|---|---|---|---|---|---|---|---|---|
| P-XXX | Eq 12 | `tau.jl:105` | Failing | LIKELY BUG | High | Newton stagnation | dτ/du exact algebraic map | Derivative consistency | `test_fast_tau.jl:test_dtau_du` | Unresolved |
```

## 4. FAST-TEST RECOMMENDATION BLOCK
For every issue found, firmly establish a verification anchor.

```text
### Verification Anchor for [P-XXX]
- **Invariant to test:** [The mathematical law that must hold true (e.g. adjoint operation equivalent to integration by parts on smooth states)]
- **Test Category:** [e.g., Adjoint consistency, Assembly smoke test, Operator Exactness]
- **Suggested Test File:** [e.g., `test/fast/test_fast_viscous_operators.jl`]
- **Suggested Test Name:** `@testset "Paper Adjoint Invariant [P-XXX]"`
- **Expected Pass Condition:** [e.g., `norm(taylor_error) < 1e-10`]
- **Expected Failure Mode / Unrolling limitations:** [e.g., Fails dynamically via Gridap invJt unrolling limits if test-functions are high-order FE projections rather than affine setups]
```
