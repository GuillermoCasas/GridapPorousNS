# Complexity Output Templates

When invoked, you MUST strictly output your analysis using these two sections: `QUICK COMPLEXITY SUMMARY` and `SELF-CORRECTION SUGGESTIONS`.

## 1. QUICK COMPLEXITY SUMMARY
Provide a short overview of the structural trajectory.

```text
# 🧭 QUICK COMPLEXITY SUMMARY

**Did complexity increase materially?:** [Yes / Minor / No]
**Main Complexity Hotspots:** [e.g., `tau.jl` functional closures, `run_test.jl` monolithic setup]

- **Structural Health:** [Healthy / Deteriorating / Fragmented]
- **Mathematical Rigor Impact:** [Improved / Maintained / Weakened]
- **Human-Readable Transparency Impact:** [Improved / Maintained / Weakened]
- **Recommendation:** [Clean up now / Accept for now / Log and monitor]
```

## 2. SELF-CORRECTION SUGGESTIONS
Provide a structured list of actionable improvements. Do not list cosmetic tweaks; only list suggestions meeting the decision principles.

```text
### Suggestion ID: [CSC-001]
**Title:** [e.g., Replace `freeze_cusp` boolean with `CuspRegularizationPolicy` type]
**Affected Files:** [e.g., `src/solvers/nonlinear.jl`, `src/stabilization/tau.jl`]

- **Complexity Type:** [e.g., Boilerplate, Flag Propagation, Abstraction Leak]
- **Rationale:** [Why does this specific complexity matter?]
- **Math Rigor Impact:** [How does it affect the mathematical formulation boundary?]
- **Readability Impact:** [How does it affect a human expert tracing the logic?]

**Proposed Simplification:**
[Explain the concrete technical refactor. E.g., "Extract linesearch evaluation to a standalone `evaluate_descent_criterion` function."]

**Expected Benefit:** [e.g., Removes 4 if/else branches dynamically inside Newton assembly bounds]
**Downside / Tradeoff:** [e.g., Requires passing an extra function parameter to the solver kernel]

**Assessment Ratings:**
- **Risk Label:** [Use exact labels: LOW RISK TO REFACTOR, JUSTIFIED SCIENTIFIC COMPLEXITY, HIGH RISK TO LEAVE AS-IS, etc.]
- **Urgency:** [High / Medium / Low]
- **Action:** [Do Now / Defer / Record for Monitoring / Dismiss]
```
