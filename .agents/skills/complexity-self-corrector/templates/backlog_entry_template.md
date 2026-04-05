# Complexity Backlog Template

If the complexity is accepted temporarily or mapped as "Harmful Debt" that should be resolved in the future, instruct the user to append it to `docs/design_self_correction.md`. 

```text
## [CSC-00X] [Architecture Component]: [Short Title]
**Date:** [YYYY-MM-DD]
**Status:** [Open / Monitoring / Resolved]
**Complexity Profile:** [Why was this deferred? What triggers its future refactor?]
**Mathematical Risk:** [What traceability is obscured until this is fixed?]

**Condition to Refactor:**
[e.g., "Wait until the experimental 1D stabilization branch is merged, then unify both branches under a `StabilizationPolicy` trait."]
```
