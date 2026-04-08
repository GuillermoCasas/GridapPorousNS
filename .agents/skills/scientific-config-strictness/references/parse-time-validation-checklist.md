# Parse-Time Validation Checklist

In scientific computing applications, input validation must trigger universally upon reading the configuration—not when a parameter is accessed deep within a simulation timestep loop. Use this checklist logically when designing, implementing, and auditing `raw input -> typed config` pipelines.

## 1. Presence & Exhaustiveness
- [ ] Are all conceptually indispensable physical parameters explicitly stated? (e.g., $Re$, $Nu$, Characteristic Lengths, Permeability, Base Porosity).
- [ ] Are all fundamental numerical switches present? (e.g., Stabilization formulations, element definitions).
- [ ] Is there an active trap for **unrecognized** schema keys? (If a user misspells `viscosity` as `vicosity`, the parser must crash violently rather than gracefully discarding it).

## 2. Type & Dimensional Constraints
- [ ] Are floating-point physics variables specifically and strictly checked/parsed as `Float64` (or other intended precisions)?
- [ ] Are iteration limits and discrete parameters strictly bound to integer representations?
- [ ] If spatial vectors/tensors are processed, does the configurator verify dimensions directly align with the domain model? (e.g., An assigned 2D gravity array `[0.0, -9.81]` must raise an exception when supplied to a strictly 3D mesh).

## 3. Allowed Domains & Enumerations
- [ ] Is categorical string masking mapping exactly into an Enum list? (e.g., `reaction_model`: `"None"`, `"Linear"`, `"Forchheimer"`).
- [ ] Will mapping crash specifically noting the explicit list of valid allowed choices upon ingesting a rogue string?
- [ ] Are strict mathematical bounds enforced natively? (e.g., `density > 0.0`, `porosity ∈ [0.0, 1.0]`).
- [ ] Are solver threshold boundaries uniformly positive scalar requirements? (e.g., `tol > 0.0`, rejecting `<= 0.0`).

## 4. Cross-Field State Consistency
- [ ] If Boolean flags conditionally mandate sub-parameters (e.g., `use_linesearch == true` demands `linesearch_tolerance`), does the parser fault on missing dependent contexts? 
- [ ] If the flag is deactivated (`use_linesearch == false`), is the erroneous active presence of `linesearch_tolerance` flagged as an incongruent "dead key" warning?
- [ ] Does the assembly of physical values or stabilization methods yield theoretical sensibility checks where viable? (e.g., Rebuffing a request to apply convective stabilization on an irrotational diffusion construct).
- [ ] Are provided characteristic length scales checked for geometric reality against domain constraints where computable?

## 5. Explicit Policy Object Allocation
- [ ] Instead of retaining raw boolean markers to trigger complex behavior mid-simulation, does initialization spawn explicit semantic trait/policy types? (e.g., Replace `if config.regularize_physics` with `typeof(config.viscosity_model) <: RegularizedViscosityProtocol`).

## Rule of Thumb 
**If the simulation logic can fail computationally deep inside an inner time loop due to user configuration artifacts, and said anomaly could have logically been trapped statically during dictionary parsing—your schema pipeline guarantees are too weak.**
