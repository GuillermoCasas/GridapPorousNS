---
name: no-hard-code-parameters
description: enforce disciplined handling of tunable and semantically important parameters in technical code. use when adding, refactoring, reviewing, renaming, moving, splitting, merging, deprecating, or documenting configuration values, thresholds, tolerances, heuristic coefficients, numerical safeguards, enum options, structured arrays or objects, or subsystem-specific settings. requires synchronized updates to json config, config loaders/defaults, example configs, and matching json schema, with descriptions grounded in code usage, mathematical role, physical meaning, units or nondimensional interpretation, valid options, expected effect of change, and provenance.
---

# Objective

Enforce disciplined treatment of nontrivial parameters in technical code.

This skill exists to prevent:
- hard-coded thresholds, tolerances, heuristic coefficients, and numerical safeguards scattered through implementation code
- configuration values being added without updating the JSON schema
- stale, vague, misleading, or underspecified schema descriptions
- duplicated defaults across code, config, examples, and schema
- parameter drift, where implementation behavior changes but config/schema semantics do not
- “temporary” literals silently becoming long-term policy
- ambiguous ownership, where one concept appears in multiple sections with slightly different names or defaults

This skill treats **parameter surfacing**, **schema maintenance**, and **config/code/schema synchronization** as part of implementation correctness, not optional documentation.

# Non-Negotiable Rules

1. Do not introduce a nontrivial parameter as an unnamed literal in implementation code.
2. Do not add a new JSON config parameter without updating the matching JSON schema in the same change.
3. Do not rename, move, split, merge, or deprecate a parameter without updating config paths, schema entries, loaders/defaults, examples, and references accordingly.
4. Do not write shallow schema descriptions that merely restate the field name.
5. Do not invent mathematical or physical meaning not supported by the code or context.
6. If the value is not a true fixed constant, classify it explicitly and decide where it belongs.
7. If a value remains in code, give it a deliberate semantic name if it is referenced more than once or carries non-obvious meaning.
8. Prefer one authoritative source for each parameter and one coherent owner for each parameter family.

# Core Principle

For any value that is not a true fixed constant, do not leave it buried in arbitrary code.

Instead, follow this coupled workflow:

1. detect the value
2. decide whether it is a true constant or a parameter
3. classify it conceptually
4. choose the correct owner and JSON config location
5. update the JSON schema immediately
6. write a semantically meaningful schema entry
7. wire code to config
8. update related artifacts
9. run a final consistency audit across code, config, defaults, examples, and schema

Never add a parameter to config and “leave the schema for later.”

# What Counts as a Parameter

Treat a value as a parameter if changing it can reasonably alter:
- numerical behavior
- convergence behavior
- stability
- damping
- performance
- accuracy
- accepted inputs
- algorithm selection
- stopping conditions
- fallback behavior
- diagnostics, reporting, or experiment control
- the behavior of a specific sub-algorithm or subsystem

Typical parameters include:
- threshold
- tolerance
- iteration limit
- damping factor
- line-search control
- stabilization coefficient
- regularization floor
- clipping or saturation value
- filtering or smoothing value
- fallback or safeguard value
- retry count
- benchmark pass/fail threshold
- enum-like algorithm selector
- ablation or experimental switch
- subsystem policy choice
- model coefficient that may vary by run, study, or calibration
- reporting or diagnostic control that materially affects workflow or interpretation

Treat a value as a parameter even if it was originally introduced as a “temporary” literal.

# What Should Usually Stay in Code

Do not blindly externalize everything.

These may stay in code if they are genuinely fixed and not acting as tuning knobs:

- mathematical identities and universally fixed mathematical constants
- true physical constants when genuinely fixed by the model and not project-tunable
- unavoidable language or library literals
- trivial identity literals such as `0`, `1`, `2` when they are not serving as hidden policy
- values that are part of a fixed mathematical definition rather than a numerical or algorithmic choice
- compile-time or format constants that are not user-meaningful and not study-dependent

However, if a value is used as a practical approximation, regularization, safeguard, tolerance, heuristic, policy choice, or algorithm selector, treat it as a parameter, not a constant.

# Decision Procedure: Constant vs Parameter

Before moving anything, answer these questions explicitly:

1. Is this value an exact mathematical or physical constant, or is it a design choice?
2. Would a developer, researcher, or tester plausibly vary it across runs, studies, machines, or experiments?
3. Does changing it alter solver behavior, robustness, conditioning, accuracy, acceptance criteria, or algorithmic path selection?
4. Is it acting as a floor, cap, tolerance, cutoff, safeguard, smoothing strength, damping factor, or policy threshold?
5. Is it only relevant to one sub-algorithm or subsystem?
6. Is it user-facing, advanced, experimental, or internal-but-centralized?

If the answer to any of 2–4 is “yes,” it is usually a parameter.

If uncertain, prefer surfacing it into config unless there is a strong reason not to.

# Parameter Classes

Classify parameters as specifically as possible, without duplication.

Common categories include:
- physical_parameter
- material_parameter
- model_parameter
- discretization_parameter
- solver_parameter
- nonlinear_solver_parameter
- linear_solver_parameter
- linesearch_parameter
- time_integration_parameter
- stabilization_parameter
- regularization_parameter
- optimization_parameter
- filtering_parameter
- clipping_parameter
- safeguard_parameter
- fallback_parameter
- benchmark_threshold
- validation_threshold
- reporting_parameter
- visualization_parameter
- subalgorithm_parameter
- experimental_parameter

Also classify its audience and status:
- standard
- advanced
- experimental
- legacy
- internal_but_centralized

Use the most specific category that remains coherent.

# Ownership and JSON Placement

Place each parameter under the most specific subsystem subsection possible.

Prefer local semantic ownership over generic global buckets.

Examples:
- `solver.nonlinear.residual_absolute_tolerance`
- `solver.nonlinear.max_iterations`
- `solver.linesearch.alpha_min`
- `solver.linesearch.backtracking_reduction_factor`
- `stabilization.tau.c1`
- `stabilization.tau.regularization_floor`
- `regularization.velocity_magnitude_floor`
- `projection.mass_lumping_threshold`

Avoid:
- duplicated copies of the same parameter in different sections
- vague “misc”, “advanced”, or “settings” buckets when a real owner exists
- global solver buckets for sub-algorithm-specific controls
- multiple names for one concept
- cross-subsystem duplication when one subsystem clearly owns the value

If the parameter only matters to one sub-algorithm, it should live under that sub-algorithm unless there is a compelling project-wide reason not to.

# Naming Rules

Every surfaced parameter must have a self-descriptive, semantically meaningful name.

Prefer names that communicate:
- what the value controls
- where it applies
- what quantity it refers to
- whether it is absolute/relative/min/max/floor/cap

Good patterns:
- `residual_absolute_tolerance`
- `backtracking_reduction_factor`
- `velocity_magnitude_floor`
- `tau_regularization_floor`
- `maximum_backtracking_steps`

Avoid:
- `eps`
- `tol2`
- `c`
- `alpha0`
- `factor`
- `magic_value`
- `weight`
- `limit`
- `toggle`

Unless mathematically standard and still documented, cryptic names are not acceptable.

# Single Source of Truth

Prefer one authoritative source for each parameter and its default.

If project architecture permits, avoid duplicating defaults in multiple layers.

If defaults must exist in multiple places for practical reasons, keep them synchronized and audit them explicitly across:
- code
- config
- default/example JSON files
- schema

Do not silently allow divergence.

# Mandatory Workflow: Parameter-Config-Schema Lockstep

Follow these steps in order whenever a parameter is added, renamed, moved, split, merged, deprecated, reclassified, or semantically changed.

## Step 1: Detect
Identify:
- new hard-coded values
- renamed parameters
- moved parameters
- parameters whose meaning changed
- schema entries that no longer match code behavior
- defaults that no longer match runtime behavior
- config values with missing or weak schema support
- stale example configs
- deprecated paths still implied by old docs or schema

## Step 2: Decide Constant vs Parameter
Before moving anything, decide explicitly:
- is this a true constant?
- is it a tunable parameter?
- is it advanced?
- is it experimental?
- is it legacy?
- is it internal-but-centralized?
- is it subsystem-local?

Document the decision implicitly through placement and schema metadata.

## Step 3: Classify Precisely
Assign:
- conceptual category
- owner subsystem
- audience/status
- provenance if known

Keep the classification as specific as possible without creating duplication.

## Step 4: Choose Ownership and JSON Location
Add or move the parameter under the correct JSON path.

Also consider whether the same change requires:
- creating a new subsection
- consolidating duplicated parameters
- splitting one ambiguous parameter into multiple clearer ones
- deprecating an old path

## Step 5: Update the JSON Schema Immediately
Every parameter added to config must have a matching schema entry.

Whenever a parameter is renamed, moved, split, merged, deprecated, or repurposed, update the schema in lockstep.

Schema maintenance is mandatory.

## Step 6: Enrich the Schema Semantically
The schema entry must document as much as is known truthfully.

Do not write shallow labels such as:
- “cutoff”
- “weight”
- “format”
- “toggle”
- “advanced handling”
- “stabilization limit”
- “parameter for X”
- “used in solver”
- “controls algorithm”

Instead, explain the parameter using the structure below.

### Required semantic content where known

#### 1. Code Role
Explain:
- where it is used in the code
- which subsystem consumes it
- whether it affects assembly, solver behavior, stabilization, regularization, diagnostics, reporting, postprocessing, testing, or experiment orchestration

#### 2. Mathematical Role
Explain:
- what quantity or rule it modifies mathematically
- whether it is a tolerance, coefficient, floor, safeguard, selector, structural control, or constitutive/model parameter
- enough interpretation to understand its place in the algorithm

#### 3. Physical Role
If applicable, explain:
- what physical quantity it represents
- how it relates to the model
- whether it is physically meaningful or purely algorithmic

#### 4. Units or Scale Meaning
State one of:
- physical units
- nondimensional interpretation
- pure algorithmic control parameter
- scale-dependent numerical safeguard

If units are not known, say so rather than guessing.

#### 5. Expected Effect of Change
Describe what is generally expected if the value:
- increases
- decreases

Examples:
- stricter convergence requirement
- more conservative line search
- stronger regularization near zero
- greater damping
- greater risk of false stagnation
- possible effect on conditioning, robustness, or accuracy

If the effect is only partially known, say so conservatively.

#### 6. Valid Values and Options
Use:
- `enum` for categorical options
- `minimum`, `exclusiveMinimum`, `maximum` where known
- positivity/nonnegativity constraints where justified
- structural restrictions where appropriate

For arrays and structured objects, describe:
- item meaning
- item ordering
- per-entry units or interpretation if relevant

#### 7. Provenance
Record provenance when known, such as:
- theory_derived
- paper_derived
- physically_measured
- empirically_calibrated
- legacy_compatibility
- numerical_safeguard
- debugging_only
- experimental
- benchmark_control
- ui_default
- unknown

#### 8. Sensitivity and Intended Audience
Indicate whether the parameter is:
- numerically sensitive
- advanced
- experimental
- legacy
- recommended only for expert users
- mainly for ablation, debugging, or diagnostics

## Step 7: Update Related Artifacts
When a parameter is added or changed, update all relevant layers, not just code and schema.

Typical artifacts include:
- config loaders/parsers
- typed config structs or classes
- default JSON files
- example input files
- migration or backward-compatibility code
- tests that validate config/schema behavior
- docs that mention user-facing configuration

If a repo has a single schema but multiple config examples, update all affected examples.

## Step 8: Wire Code to Config
Refactor the implementation so the code reads the parameter from config instead of embedding a literal.

Prefer descriptive access paths such as:
- `config.solver.nonlinear.residual_absolute_tolerance`
- `config.solver.linesearch.alpha_min`
- `config.regularization.velocity_magnitude_floor`
- `config.stabilization.tau.regularization_floor`

Do not leave duplicate defaults in multiple code locations unless architecture requires it and synchronization is explicit.

## Step 9: Run a Final Consistency Audit
Before finishing, explicitly verify:

- every new parameter exists in config
- every config parameter has a schema entry
- names match across code/config/schema
- defaults match across code/config/schema/examples
- section ownership is coherent
- deprecated parameters are marked as such
- moved or split parameters do not leave stale schema remnants
- descriptions reflect current semantics, not historical ones
- enums and bounds reflect actual code behavior
- example configs are still valid
- nontrivial literals still left in code are truly justified

# JSON Schema Authoring Rules

Use standard JSON Schema fields whenever applicable:
- `title`
- `description`
- `type`
- `default`
- `enum`
- `minimum`
- `exclusiveMinimum`
- `maximum`
- `examples`
- `deprecated`

Prefer truthful precision over verbosity.

A good description usually answers:
- what the parameter controls
- where it applies
- what kind of parameter it is
- what increasing it tends to do
- what decreasing it tends to do
- whether it is physical, numerical, or algorithmic

## Description Template

When helpful, use a structure like:

`[What it controls]. Used by [subsystem/algorithm] to [purpose]. [Mathematical/physical interpretation]. [Units or nondimensional meaning]. Increasing it tends to [...]; decreasing it tends to [...]. [Constraints, caveats, or intended audience].`

Do not force this wording mechanically; adapt it to the parameter.

# Vendor Metadata Discipline

If the project supports vendor extensions, prefer richer semantic metadata such as:
- `x-category`
- `x-subsystem`
- `x-units`
- `x-dimensionless_meaning`
- `x-provenance`
- `x-used_by`
- `x-mathematical_role`
- `x-physical_role`
- `x-expected_effect_of_increase`
- `x-expected_effect_of_decrease`
- `x-sensitivity`
- `x-advanced`
- `x-tunable`
- `x-risk_if_misused`
- `x-notes`

Use them only when grounded in the code or context.

Do not introduce a chaotic metadata style. If vendor metadata is used, keep naming consistent across the schema.

# Behavior by Context

## When Generating Code
Think in this order:

1. detect likely parameters
2. decide which belong in config
3. decide which should remain true constants
4. choose ownership and JSON location
5. create or extend JSON config
6. create or extend JSON schema
7. write semantically meaningful schema descriptions and metadata
8. update loaders/defaults/examples
9. wire code to config
10. verify code/config/schema consistency

Do not generate implementation code that introduces new configuration values without schema updates.

## When Reviewing or Refactoring Code
Do the following:

1. scan for hidden magic numbers and hidden options
2. classify each nontrivial literal
3. decide whether it should remain in code or move to config
4. propose or create the config entry
5. propose or create the matching schema entry
6. improve weak schema descriptions using code context
7. identify stale, duplicated, misleading, or inconsistent entries
8. recommend refactors that eliminate duplication and semantic drift
9. check loaders/defaults/examples/tests for synchronization gaps

## When Handling Renames, Moves, Splits, Merges, and Deprecations
Treat these as critical maintenance events.

You must detect and fix cases where:
- a parameter name changed but the schema did not
- a parameter moved subsections but the schema did not
- a parameter’s meaning changed but the old description remained
- one parameter became several parameters but the schema still documents a single concept
- several parameters were consolidated but schema/examples still imply the old split
- a parameter is deprecated or unused but the schema does not say so
- compatibility aliases exist in code but are undocumented in schema or examples

# Preferred Output Pattern

When using this skill, structure the reasoning roughly as:

1. detected literals and candidate parameters
2. constant vs parameter decisions
3. classification of each parameter
4. recommended JSON config locations
5. proposed schema entries or schema changes
6. proposed names and metadata
7. updates required in loaders/defaults/examples/tests
8. code rewiring plan or refactored pattern
9. final consistency audit
10. justified exceptions left in code

# Examples

## Example 1: Solver Tolerance
Bad:
```julia
if residual < 1e-8
```

Better:
- classify as nonlinear solver tolerance
- move to config:
  `solver.nonlinear.residual_absolute_tolerance`
- add schema constraints such as:
  - `type: number`
  - `exclusiveMinimum: 0.0`
- description:
  “Absolute residual tolerance for nonlinear solver convergence. Used by the nonlinear solver stopping logic to decide when the monitored residual is small enough to accept convergence. Smaller values impose stricter convergence and may increase iteration count or expose conditioning and noise limits.”
- add metadata if supported:
  - `x-category: nonlinear_solver_parameter`
  - `x-provenance: numerical_safeguard`
  - `x-expected_effect_of_decrease: stricter convergence; may increase iterations and sensitivity to noise`

## Example 2: Line Search Reduction
Bad:
```julia
alpha *= 0.5
```

Better:
- classify as line-search policy control
- move to config:
  `solver.linesearch.backtracking_reduction_factor`
- schema description:
  “Factor applied to the step length after each failed backtracking line-search attempt. Used by the line-search sub-algorithm to reduce the trial step. Smaller values make the search more conservative but may increase the number of backtracking iterations.”

## Example 3: Iteration Cap
Bad:
```julia
for i in 1:15
```

Better:
- classify as subalgorithm iteration limit
- move to config:
  `solver.linesearch.max_backtracking_steps`
- schema description:
  “Maximum number of backtracking line-search reductions allowed before the line-search procedure is declared unsuccessful. Larger values permit more recovery attempts but may increase runtime.”

## Example 4: Regularization Floor
Bad:
```julia
mag_u = sqrt(u*u + 1e-12)
```

Better:
- determine whether `1e-12` is a regularization floor or numerical safeguard
- move to config:
  `regularization.velocity_magnitude_floor`
- schema description:
  “Floor value used to regularize expressions involving velocity magnitude near zero, preventing division by very small values and limiting derivative blow-up near stagnation. This is a numerical safeguard, not a physical parameter. Increasing it strengthens regularization and may reduce sensitivity near zero-velocity states.”

## Example 5: Moved Stabilization Parameter
Scenario:
`physical_parameters.tau_regularization_limit` is moved to `stabilization.tau.regularization_floor`

Required actions:
- move the config entry
- move and improve the schema entry
- update loaders and accessors
- mark the old path deprecated if backward compatibility is retained
- update example configs
- verify no stale references remain

## Example 6: Experimental Enum
Scenario:
a new option `solver.linear_solver_type` is introduced

Required schema behavior:
- define `type: string`
- define `enum`
- explain what each variant selects algorithmically
- mark as experimental or advanced if appropriate
- avoid descriptions that merely restate the name

## Example 7: Structured Array Parameter
Scenario:
a parameter `mesh.domain` is an array like `[xmin, xmax, ymin, ymax]`

Required schema behavior:
- describe the semantic ordering of entries
- state the units if known
- avoid generic text like “bounding box values”
- make clear how the array is interpreted by the mesh builder

# Style Guardrails

- Be strict, but not mechanical.
- Prefer semantic clarity over minimal edits.
- Prefer one authoritative source per parameter.
- Prefer local subsystem ownership when appropriate.
- Avoid vague names.
- Avoid generic descriptions.
- Avoid exposing values as user-facing knobs when they are better treated as advanced or internal-but-centralized.
- Do not invent mathematical or physical meaning not supported by code or context.
- If only partial meaning is known, document what is known clearly and say what is still uncertain.
- If a value remains in code, ensure the reason is explicit and defensible.
- Block any plan that adds a config parameter without a matching schema update.

# Final Rule

Every time the code gains a parameter, the config and schema must gain a coherent, synchronized, semantically meaningful representation of that parameter.

Think:

**new parameter -> classify -> choose owner -> place in config -> update schema -> enrich semantics -> update loaders/examples -> wire code -> audit consistency**