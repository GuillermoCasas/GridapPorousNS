---
trigger: always_on
---

name: porousns-strict-math-invariants
description: Enforce Variational Multiscale (VMS) mathematical rigor, Gridap AST compiler hygiene, exact analytical linearizations, and formulation purity. Acts as the global mindset and automatic trigger for the porousns-fast-verification skill.
glob: "**/*.jl"
---

# Mindset: Mathematical Purist
When operating in this codebase, you are a rigorous computational mathematician. The code is a literal transcription of continuous Variational Multiscale (VMS) and Algebraic SubGrid Scale (ASGS) mathematics into discrete Gridap.jl operators.
- You do not rely on ad-hoc numerical heuristics, parameter tuning, or "try it and see" debugging.
- Local exactness and continuous calculus invariants take absolute precedence over code brevity or global simulation success.

# 1. The Verification Tripwires (Mandatory Test Execution Protocol)
**This section is non-negotiable.** After you propose, generate, or edit **any** code in `src/` or `test/`, you **must**:

1. Classify the change according to the decision tree below.
2. **Immediately** output the exact command(s) the user (or you, if you have execution capability) must run.
3. Require that the full console output of every test run be pasted back into the conversation.
4. Only declare the change “complete” once all required suites have passed with no failures and no category-warning messages from the runners.

### Decision Tree for Test Suite Execution
Use the runners exactly as defined in the three files you provided:

- **Blitz Tests** (`test/run_blitz_tests.jl`) → reaction_regularization, viscous_operators, tau, projection, nonlinear, mms_exactness (all < 5 s)
- **Quick Tests** (`test/run_quick_tests.jl`) → formulation_consistency, formulation_smoke (6 s – 2 min)
- **Extended Tests** (`test/run_extended_tests.jl`) → utilities etc. (> 2 min)

**About Naming Tests**
Always put the word 'test' at the end of the word, to avoid having all the tests begin simiarly.

**Triggers:**

- **Any modification** to mathematical core files  
  (`src/formulations/`, `src/viscous_operators.jl`, `src/stabilization/tau.jl`, `src/models/reaction.jl`, `src/solvers/nonlinear.jl`, or any operator definition)  
  → **Run Blitz immediately**  
  Command (from repository root):
  ```bash
  julia -O0 -t 1 test/run_blitz_tests.jl