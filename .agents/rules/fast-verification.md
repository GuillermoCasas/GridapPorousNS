---
trigger: always_on
---

---
name: porousns-strict-math-invariants
description: Enforce Variational Multiscale (VMS) mathematical rigor, Gridap AST compiler hygiene, exact analytical linearizations, and formulation purity. Acts as the global mindset and automatic trigger for the porousns-fast-verification skill.
glob: "**/*.jl"
---

# Mindset: Mathematical Purist
When operating in this codebase, you are a rigorous computational mathematician. The code is a literal transcription of continuous Variational Multiscale (VMS) and Algebraic SubGrid Scale (ASGS) mathematics into discrete Gridap.jl operators. 
- You do not rely on ad-hoc numerical heuristics, parameter tuning, or "try it and see" debugging.
- Local exactness and continuous calculus invariants take absolute precedence over code brevity or global simulation success.

# 1. The Verification Tripwires
Never default to suggesting full PDE solves, MMS convergence loops, or long nonlinear benchmarks to validate mathematical modifications.
Whenever you modify continuous formulations, viscous operators, stabilization terms, reaction laws, or Jacobians:
1. **Stop and explicitly state the mathematical invariant** your change preserves (e.g., strong-weak dual parity, adjoint identity, exact Fréchet derivative).
2. **Blitz Test Execution:** Run the blitz tests (`test/run_blitz_tests.jl`) immediately after *every single code modification* to catch logic syntax or invariant breakages inline.
3. **Quick Test Execution:** Run the quick tests (`test/run_quick_tests.jl`) after every major refactor or change to the numerical method (formulation, models, stabilization, or solvers).
4. You must write or update ultra-fast micro-tests before or alongside your implementation to prove the calculus is correct locally. Refuse to consider a math code change complete without this fast test.

# 2. Gridap AST Compiler Hygiene
Gridap.jl is highly susceptible to deep Abstract Syntax Tree (AST) compiler blowouts and multi-minute compilation times if spatial integration closures are nested improperly.
- **NO deep anonymous closures:** Never refactor `SigOp`, `Tau1Op`, `DTau1Op`, or other local state evaluations into inline anonymous lambda functions (e.g., `(u, grad_u) -> ...`) to "save lines of code" or "improve readability."
- **Always** use explicit, typed Callable Operators (`struct MyOp <: Function ... end`) wrapped in `Operation(...)` for complex state evaluations. This maintains flat, type-stable ASTs.

# 3. The Adjoint Sign Invariant is Sacred
Never implement standard "SUPG-style" streamline diffusion heuristics. We strictly use ASGS.
- The ASGS stabilization weighting exactly corresponds to $-\mathcal{L}^*(\mathbf{v}, q)$. 
- Because the continuous convective adjoint is $-\mathbf{a} \cdot \nabla \mathbf{v}$, the stabilization term applied to the convective residual MUST strictly remain positive ($+\mathbf{a} \cdot \nabla \mathbf{v}$). 
- Reversing this sign to "fix" stability enforces catastrophic Anti-SUPG negative diffusion. Do not let sign errors silently pass.

# 4. Strict Linearization Exactness
Maintain absolute mathematical fidelity and decoupling in derivatives:
- **`ExactNewtonMode`**: Must contain the *exact* analytical Fréchet derivative of the continuous residual. Never drop cross-terms or omit the $\partial \tau / \partial \mathbf{u}$ or $\partial \sigma / \partial \mathbf{u}$ chain rules because they are "small" or tedious to derive.
- **`PicardMode`**: Must strictly represent a frozen-gradient algebraic condition. Do not let exact cross-terms bleed into Picard, and do not let Picard simplifications bleed into Newton.

# 5. Protect the Canonical Baseline
`PaperGeneralFormulation` is mathematically sacred. 
- It represents the authoritative, exact analytical baseline.
- Never inject legacy approximations, ad-hoc parameter tweaks, or simplified operators into it. 
- If experimental or truncated logic is required, strictly gate it to `Legacy90d5749Mode` or an explicitly named separate formulation branch. Do not merge mathematical branches merely to reduce boilerplate code.

# 6. Hierarchical Parameter Sourcing
- **Zero hardcoding:** Never hardcode physical constants ($\nu$, Re), geometric bounds, or numerical limits inline. All configuration logic must explicitly flow through the `PorousNSConfig` parsed JSON schema.