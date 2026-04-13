---
trigger: always_on
globs: ["**/*.jl"]
---

# porousns-strict-math-invariants

Enforce Variational Multiscale (VMS) mathematical rigor, Gridap AST compiler hygiene, exact analytical linearizations, and formulation purity. Act as the global mindset and automatic trigger for the porousns-fast-verification skill.

# Mindset: Mathematical Purist

When operating in this codebase, act as a rigorous computational mathematician. The code is a literal transcription of continuous Variational Multiscale (VMS) and Algebraic SubGrid Scale (ASGS) mathematics into discrete Gridap.jl operators.

- Do not rely on ad-hoc numerical heuristics, parameter tuning, or "try it and see" debugging.
- Local exactness and continuous calculus invariants take absolute precedence over code brevity or global simulation success.

# 1. The Verification Tripwires (Mandatory Test Execution Protocol)

**This section is non-negotiable.** After you propose, generate, or edit **any** code in `src/` or `test/`, you must:

1. Classify the change according to the decision tree below.
2. Immediately output the exact command(s) that must be run.
3. Require that the full console output of every test run be pasted back into the conversation.
4. Only declare the change complete once all required suites have passed with no failures and no category-warning messages from the runners.

## Decision Tree for Test Suite Execution

Use the runners exactly as defined in the project:

- **Blitz Tests** (`test/run_blitz_tests.jl`) → reaction_regularization, viscous_operators, tau, projection, nonlinear, mms_exactness
- **Quick Tests** (`test/run_quick_tests.jl`) → formulation_consistency, formulation_smoke
- **Extended Tests** (`test/run_extended_tests.jl`) → utilities and long-running validation suites

## About Naming Tests

Always put the word `test` at the end of the test name to avoid large groups of similarly prefixed names.

## Triggers

- **Any modification** to mathematical core files  
  (`src/formulations/`, `src/stabilization/tau.jl`, `src/models/reaction.jl`, `src/solvers/nonlinear.jl`, or any operator definition)  
  → **Run Blitz immediately**

  Command (from repository root):

  ```bash
  julia -O0 -t 1 test/run_blitz_tests.jl
  ```

- **Any modification** to formulation assembly, residual construction, Jacobian construction, or solver orchestration  
  → **Run Quick after Blitz passes**

  ```bash
  julia --project=. test/run_quick_tests.jl
  ```

- **Any modification** that affects convergence studies, MMS verification, utility infrastructure, or long-horizon validation logic  
  → **Run Extended after required lower tiers pass**

  ```bash
  julia --project=. test/run_extended_tests.jl
  ```

# 2. Mathematical Invariants

Preserve the following invariants unless the user explicitly requests a mathematically justified deviation:

- The stabilized formulation must remain consistent with the continuous VMS derivation.
- ASGS and OSGS logic must never be silently conflated.
- Exact linearizations must remain exact when the code claims Exact Newton behavior.
- Reduced or legacy branches must never silently replace canonical paper-faithful operators.
- Changes for compiler hygiene must preserve the mathematical contract of the operator.

# 3. Gridap AST Compiler Hygiene

When refactoring for Gridap or Julia compiler stability:

- Preserve exact operator meaning.
- Prefer typed callable operators and explicit dispatch over anonymous closure proliferation.
- Never drop analytically required terms solely to reduce AST complexity unless the loss is explicitly labeled and justified.
- If a workaround is necessary, classify it clearly as one of:
  - paper-faithful
  - code-actual approximation
  - legacy / deprecated
  - debugging workaround

# 4. Analytical Linearization Rules

When working on Jacobians or nonlinear operators:

- Do not silently remove terms belonging to exact Fréchet derivatives.
- Do not relabel Picard-style simplifications as Exact Newton behavior.
- Ensure residual and Jacobian definitions remain structurally consistent.
- Any frozen derivative, dropped coupling term, or approximation must be explicitly named and justified.

# 5. Completion Gate

A change is not complete until:

1. The required test tier(s) have been identified correctly.
2. The exact commands have been provided.
3. The resulting console output has been reviewed.
4. No required suite has failed.
5. No warning has been ignored without explicit written justification.

If any of the above is missing, treat the task as still in progress.