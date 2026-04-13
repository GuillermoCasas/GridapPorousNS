---
name: porousns-no-hard-coded-parameters
description: Enforces strict parametrization for numerical properties. Prevents any magic inline numbers (tolerances, scaling coefficients, epsilon values, fallback iteration limits) across formulations and solvers from existing outside of structured schematization.
glob: "**/*.jl"
---
# Mindset: Explicit Parameterization Enforcement

You are operating in a numerically sensitive computational mathematics codebase. The safety, robustness, reproducibility, and mathematical interpretability of this Porous Navier-Stokes VMS solver depend on **explicit parameter schemas**, **transparent numerical policy**, and **full user control over all algorithmically relevant quantities**.

Treat hidden fallback values, implicit defaults, and silent parameter injection as violations of the same principle that forbids hard-coded numerical constants.

The repository follows a strict **no magic numbers / no implicit defaults** policy.

---

## 1. No hard-coded parameters, tolerances, or thresholds

Never introduce a hard-coded numerical parameter into solver, formulation, stabilization, convergence, verification, or diagnostics code unless it is a mathematically universal constant that is explicitly part of the governing equations or discretization definition.

Treat all of the following as configuration-bearing quantities that must never appear as unexplained inline literals in implementation code:

- tolerances
- thresholds
- damping factors
- relaxation factors
- line-search constants
- floors, limits, and epsilons
- safety margins
- iteration limits
- switching criteria
- plateau criteria
- regularization limits
- scaling factors
- heuristic cutoffs
- fallback values
- recovery constants
- auto-filled missing parameters

If a value changes solver behavior, stopping behavior, verification interpretation, or numerical robustness, it must be explicit, named, documented, and validated.

---

## 2. No implicit defaults or fallback parameter injection

Do **not** add default parameter values merely to avoid errors when the user omits a field.

The default behavior of the code must be:

- require the user to specify all numerically relevant parameters explicitly
- fail early and clearly when a required parameter is missing
- never silently invent, inject, infer, backfill, or substitute a numerical value
- never use “safe” fallback constants unless those exceptions were explicitly designed, requested, and documented by the user

Treat all of the following as violations unless explicitly user-approved and documented:

- constructor defaults such as `tol=1e-6`, `max_iters=10`, `alpha_min=0.5`, `eps=1e-12`
- schema loaders that silently populate missing numerical fields
- “temporary” fallback values added to prevent crashes
- convenience defaults for stabilization, Newton, OSGS, MMS, or verification parameters
- hidden auto-corrections that clamp missing inputs to preselected values
- internal helper functions that replace `nothing` / missing values with hard-coded constants

A missing numerical input is not a reason to invent a value.  
It is a reason to raise a clear configuration error.

### Required policy

If a parameter is required for mathematical meaning, numerical safety, or interpretation of results, then the code must require explicit user specification of that parameter.

### Limited exception policy

An exception is allowed only if **all** of the following are true:

1. the user explicitly wants that default behavior
2. the exception is intentionally designed as part of the interface
3. the value is documented as an explicit design choice
4. the rationale is stated clearly
5. the parameter remains visible to the user
6. the exception is validated and not silently injected

If any of the above is missing, do not add the default.

---

## 3. Parameter adoption workflow

If a new numerical control is needed, you must do all of the following:

### 3.1 Promote to configuration
Create a dedicated named input parameter in the appropriate configuration or schema layer rather than embedding the value inline.

### 3.2 Use explicit semantic naming
Choose a name that reflects the mathematical or numerical role precisely.

Good examples:
- `osgs_projection_tolerance`
- `stagnation_noise_floor`
- `mms_plateau_relative_change_tolerance`
- `linesearch_alpha_min`

Bad examples:
- `eps2`
- `magic_tol`
- `tmp_limit`
- `default_factor`

### 3.3 Document the parameter rigorously
Document all of the following:

- what it controls
- why it exists
- where it enters the algorithm
- admissible range or sign constraints
- what happens if it is increased
- what happens if it is decreased
- whether it is mathematically required, numerically protective, empirical, legacy, or diagnostics-related

### 3.4 Wire it explicitly
Thread the parameter through the relevant construction path transparently.

Do not hide it inside:
- local closures
- helper defaults
- fallback branches
- internal “temporary” constants
- recovery-only logic that changes mathematics silently

### 3.5 Validate at runtime
Add validation that rejects missing, invalid, or physically meaningless values with explicit errors.

Use validation to enforce:
- presence
- type
- sign
- range
- compatibility with the algorithmic contract

### 3.6 Record provenance
State clearly whether the parameter is:

- paper-defined
- numerically protective
- empirically tuned
- legacy-compatible
- diagnostics-only
- user-requested exception

### 3.7 Justify necessity
If the parameter is newly introduced, explain why existing parameters are insufficient and why this is a real numerical control rather than a hidden heuristic.

---

## 4. Mandatory review behaviors

Actively reject any implementation that introduces unexplained inline numerical values such as:

- `1e-6`
- `1e-8`
- `0.5`
- `0.1`
- `10`
- `50`
- `100`
- `1.05`

unless the value is a mathematically universal constant with obvious meaning in context.

Also actively reject any implementation that introduces hidden defaults for numerical controls.

When reviewing code, flag all unexplained literals or default values affecting:

- convergence criteria
- stopping rules
- nonlinear solver behavior
- outer / inner iteration coupling
- OSGS projection updates
- stabilization strength
- regularization
- line search
- damping
- verification logic
- MMS plateau detection
- diagnostics interpretation
- scaling of mixed velocity-pressure states

---

## 5. Required missing-parameter behavior

If a required numerical parameter is absent, the code should:

1. stop immediately
2. emit a clear, explicit error
3. identify the missing field by name
4. explain why it is required if appropriate
5. never continue with a guessed or fallback value

Preferred behavior:

- configuration owns numerical policy
- implementation consumes explicit named parameters
- validation enforces completeness
- missing required inputs produce loud, early failures

Disallowed behavior:

- silently substituting defaults
- partially filling schemas with fallback numbers
- continuing execution under guessed tolerances
- hiding omitted inputs behind “safe” constants
- introducing convenience defaults that change solver meaning

---

## 6. Default warning formatting pattern

If you detect a hard-coded parameter or implicit default, immediately report it using this structure:

**Parameterization violation**
- **Location:** [file / function / field]
- **Problem:** [inline literal or implicit default]
- **Why this is not acceptable:** [why this is a hidden numerical policy decision]
- **Required fix:** [promote to explicit named input parameter or require explicit user specification]
- **Documentation required:** [meaning, admissible range, and expected effect]

If the issue is specifically a fallback default, use:

**Implicit default violation**
- **Location:** [file / function / schema]
- **Hidden default:** [value being inserted automatically]
- **Why this is not acceptable:** [why missing input must not be converted into silent policy]
- **Required fix:** [remove the default and require explicit user input]
- **Allowed only if:** [explicit user-designed and documented exception]

---

## 7. Preferred analytical policy

The preferred policy in this repository is:

- master configuration defines all numerical policy explicitly
- code modules consume named parameters only
- comments explain purpose, not hidden values
- validation enforces admissible and complete input
- no unexplained numerical behavior is allowed
- no numerically relevant default is acceptable unless explicitly user-designed

Never accept:
- “we can tune it later”
- “it worked in one test”
- “this default is harmless”
- “this fallback is only to avoid crashes”
- “the loader can fill it automatically”

If a numerical value matters, it must be:
- explicit
- named
- documented
- validated
- user-visible
- intentionally chosen

Anything else is a violation.