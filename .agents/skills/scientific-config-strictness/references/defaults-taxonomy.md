# Taxonomy of Configuration Defaults in Scientific Computing

This document categorizes how you should classify configuration defaults when auditing scientific code pipelines.

## 1. Acceptable Example Inputs (The "Good")
These do not exist in the source code logic itself, but rather in documentation, templates, or bootstrap utilities.
*   **Description**: A sample config file (e.g., `base_config.json` or `example_lid_driven_cavity.jl`) that a user explicitly copies and edits to instantiate their own problem.
*   **Why it is acceptable**: The user assumes total ownership of the values upon copying. The solver code itself does not fall back to these elements if the user subsequently deletes a line.
*   **Action**: Validate them but encourage their maintenance.

## 2. Unacceptable Hidden Defaults (The "Bad" / Blockers)
These exist inside the codebase runtimes and silently guess or alter the intended mathematical model.
*   **Description**: Code that fills in omitted user inputs, silently interprets missing configurations, or alters user inputs arbitrarily.
*   **Examples**:
    *   `get(dict, "porosity", 0.4)`
    *   `viscosity = coalesce(user_nu, 1e-3)`
    *   `safe_eps = max(config.eps, 1e-8)` (silently capping or regularizing a physical/mathematical parameter)
    *   `if config.model == "A" ... elseif config.model == "B" ... else ... # silent fallback to A`
    *   Defaulting to no-slip Dirichlet conditions simply because specific walls lacked an explicit BC mapping.
*   **Why it is dangerous**: 
    *   The user explicitly (or accidentally) omitted data, and the solver guessed incorrectly without notifying them.
    *   It breaks reproducibility across runs, machines, and API evolutions.
    *   It obscures physical or mathematical formulations within non-obvious utility code.
*   **Action**: Report as Blocker/High severity. Demand their immediate removal. Replace with explicit validation faults (`KeyError`s or domain errors) exclusively located during initialization.

## 3. Justified Expert-Only Defaults (The "Ugly but Accepted")
These are deeply technical numerical heuristics intentionally obfuscated to prevent casual users from unwittingly destabilizing the solver.
*   **Description**: Hard-coded parameters or backend defaults directly assigned to underlying linear algebra algorithms or preconditioner loops.
*   **Examples**:
    *   `max_gmres_restarts = 50` (When this parameter is not the focal bottleneck being analyzed).
    *   Strict numerical drop-tolerances within a preconditioning sparse iterative solver setup where exposition will just clutter the physics schema.
*   **Requirements to allow existence**: 
    To allow an expert-only default to remain in the codebase, you must be able to strictly declare:
    1.  **Why it exists**: e.g., "The underlying Krylov submodule necessitates a strict restart threshold memory cap."
    2.  **What safe regime it was validated in**: e.g., "Tested and verified unconditionally up to $Re=1000$ without performance drift."
    3.  **Why exposing it is dangerous**: e.g., "Casual users altering this often causes unpredictable stagnation that is falsely attributed to the physical models."
    4.  **What tests protect it**: e.g., "A strict MMS diagnostic script ensures quadratic convergence holds independently of this particular heuristic."
*   **Action**: Document its existence fiercely. Do not elevate it to the public schema arbitrarily, but ensure it is named and grouped transparently within the solver subsystem config (not buried implicitly as a magic number in a `.jl` loop).
