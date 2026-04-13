---
name: project-cleanup-guard
description: perform end-of-task or end-of-session cleanup for a technical code repository. use when the user asks to clean up the project, remove ai-created scratch or debug artifacts, reconcile documentation with code, restore canonical numerical states, prepare the repo for commit or review, or run a general repository hygiene pass after implementation work.
---

# Goal

Leave the repository in a clean, coherent, review-ready, and mathematically canonical state after development work.

Treat cleanup as a **consistency and hygiene pass**, not as a cosmetic sweep. The purpose is to remove temporary AI-generated artifacts, ensure that documentation and configuration remain synchronized with the code, and verify that project-specific skills, rules, and repository conventions still make sense after the latest changes.

Do not assume cleanup means only deleting files. Cleanup also includes checking that nothing important was left inconsistent, outdated, duplicated, or in an experimental "ablation" state.

# Core cleanup workflow

When the user requests project cleanup, follow this workflow in order.

## 1. Identify the cleanup scope

First determine which of these applies:

- full repository cleanup
- cleanup after a specific task or feature
- cleanup before commit / PR
- cleanup before release
- cleanup after debugging or experimentation
- cleanup of skills / rules / documentation only

If the user did not narrow the scope, default to a **full repository cleanup review**.

## 2. Remove temporary AI-created artifacts and Scientific Outputs

Find and remove files, directories, or outputs that were created only to inspect, test, debug, validate, compare, or stage work temporarily. Target the following explicitly:

- scratch scripts, debug scripts, or one-off notebooks
- temporary shell transcripts or AI-generated Markdown "thought" notes
- **Scientific Data:** Unversioned ParaView/VTK outputs (`*.vtu`, `*.pvd`) and local HDF5 caches (`*.h5`) left in root or source directories.
- **Visualizations:** Ad hoc `Plots.jl` or `matplotlib` images generated solely for an inline inspection.

Never preserve AI-created scratch artifacts “just in case” unless the user explicitly wants them kept. 
*Note: If a temporary file contains useful insight, convert its content into canonical documentation (e.g., `lessons_learned.md`) and delete the staging file.*

## 3. Restore Canonical Computational State

During development, solvers are often downgraded to bypass bottlenecks. You MUST ensure the code is restored to production limits:
- **Ablation Modes Disabled:** Verify that properties like `ablation_mode: full` are restored and experimental flags are disabled.
- **Diagnostics Silenced:** Ensure `run_diagnostics` or extreme verbose logging flags are toggled back to `false`.
- **Dependency Hygiene:** Audit `src/` for debugging dependencies (e.g., `using Plots`, `using Infiltrator`, `using BenchmarkTools`). Strip these out if they entered the core execution path, and verify the `Project.toml` remains clean.

## 4. Verify that documentation matches the code (Orchestration)

Review all relevant documentation affected by recent changes. (If profound changes occurred to theory or config, defer to `porousns-doc-architect` and `no-hard-code-parameters` for the heavy lifting). 

Audit for surface drift:
- `README` and developer docs
- command-line examples and expected file layout descriptions
- test-running instructions
- comments inside key source files when those comments define mathematical or numerical contracts

Do not leave “to be updated later” documentation drift behind.

## 5. Audit skills, rules, and project automation

Review the project’s `.agents/skills` and rules/workflows to ensure they reflect current architecture:
- outdated skill instructions or stale trigger descriptions
- rules that refer to renamed files or removed modules
- references to deprecated workflows or anti-pattern warnings that have been permanently resolved

Do not allow skills or rules to silently drift away from actual repository behavior.

## 6. Review repository hygiene 

### 6.1 Dead or stale code
Look for and remove commented-out PDE code, unused debugging branches, abandoned formulation closures, obsolete compatibility shims, and stale warnings.

### 6.2 Test and example consistency
Check that examples still run according to current interfaces, documented commands still match the repository layout, and no temporary test scaffolding (like forced convergence limits or artificially truncated meshes) remains inside the test suite.

### 6.3 Configuration and schema consistency
Check that required parameters are required everywhere they should be. Ensure no hidden defaults or fallback values were introduced during development (defer to `no-hard-code-parameters` for deep schema enforcement). 

### 6.4 Lessons learned / project memory
If code or policy changed in a way that future work could regress conceptually, verify that recent important mistakes were logged properly into the project memory files.

# Cleanup principles

- **Prefer explicit permanence:** Only keep files that have a clear, intentional long-term purpose.
- **Prefer one canonical source:** Move important truths into the correct canonical file instead of preserving scratch artifacts.
- **Prefer consistency over accumulation:** Do not keep extra files merely because they might someday be useful.
- **Prefer loud correction over quiet drift:** If documentation is stale, update it now rather than leaving subtle inconsistencies behind.

# Final cleanup report

At the end of the cleanup pass, explicitly report results using this structure:

## Cleanup summary
- removed temporary artifacts & scientific dumps
- restored canonical solver states & removed debug dependencies
- updated affected documentation & audited skills

## Files removed
- [list or summary]

## Code modifications (Hygiene & State Restoration)
- [list of un-commented code, removed debug flags, or restored configs]

## Remaining decisions for the user
- [only items that genuinely require user intent]
