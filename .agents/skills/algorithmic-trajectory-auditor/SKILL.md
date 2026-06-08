---
name: algorithmic-trajectory-auditor
description: Rigorous mental simulation of an iterative/numerical algorithm's control flow. Trace every branch and guard, generate concrete adversarial trajectories (cold start, near-convergence, divergence, stall, non-finite, parameter extremes) and deduce each outcome, scrutinize the meaning/units/normalization of every metric and threshold, and judge whether each tunable parameter should instead be a state-driven automatism. Use when reviewing or designing solver loops, convergence/stopping criteria, Newton/Picard cascades and fallbacks, line-search and divergence guards, stabilization thresholds, or any iteration whose correctness hinges on what each check actually does at the extremes — and whose silent failure mode is "returns a believable-but-wrong / not-actually-converged state labeled success."
---

# Objective

Audit an iterative algorithm by **simulating it in your head, concretely, to the end of every branch** — not by reading it for plausibility. The deliverable is a verdict on each branch and threshold: *does this produce the behaviour the goal requires, at every state the solver can actually reach?* The goal is almost always **robust convergence to the correct solution, with honest reporting** — never a fast exit that hides non-convergence.

This is a *reasoning* skill, not a test runner. Tests confirm a trajectory you already predicted; this skill is how you predict it, and how you find the trajectory no test exercises.

# Operating mode

Be exhaustive, skeptical, and concrete.

- **Concrete over abstract.** "If `‖R‖ = 1e-3`, `ftol = 1e-8`, mesh `h = 1/40`, then branch X is taken, which sets `success=true` and returns" — not "this looks like it converges."
- **Adversarial.** Actively try to construct an input/state that makes the algorithm do the wrong thing (accept a non-converged state, reject a converging one, loop forever, mislabel a method). A branch you cannot break is only validated once you have *tried* to break it.
- **Trace to the end.** Every branch has downstream consequences (what it returns, what it writes to shared state, what the caller does with that). Follow them until they terminate or merge.
- **Cite.** Anchor every claim to `file:line`. If you assert a variable's value, say where it was last written.
- **State uncertainty.** If you cannot determine a value or a path without running it, say so and name the experiment that would resolve it — do not guess and move on.

# The procedure

## 1. Build the state machine
Enumerate, before judging anything:
- Every **branch / guard / early-exit / loop bound** in the algorithm, and for each: the exact condition, the variables it reads, and the variables/shared-state it writes.
- The **state variables** that carry meaning across iterations (iterate `x`, residual norms, merit, counters, "best so far", flags), and where each is initialized, updated, and frozen.
- The **exit points** and what each returns (value *and* shared-state side effects), and how the caller interprets each.

Write this as a compact map first. You cannot audit a flow you have not enumerated.

## 2. Per-branch interrogation
For each branch, ask in order:
1. **Entry:** under exactly what state is this reached? Is that state actually reachable, or dead?
2. **Action:** what does it compute/mutate?
3. **Exit/consequence:** what does it return or fall through to, and what does the caller then do?
4. **Desirability:** *is this the behaviour the goal wants for that state?* Name the goal explicitly (converge? globalize? fail loudly? preserve the best iterate?) and check the branch serves it.
5. **Failure honesty:** if this is a success/accept exit — does it correspond to *genuine* convergence, or can it accept a stalled/diverged/noise-floor state and label it success? Who downstream is misled?

## 3. Adversarial trajectory generation
Don't wait for branches to come to you — drive concrete scenarios through the whole algorithm and predict the path + final outcome for each:
- **Cold start** (initial guess far from solution; large `‖R₀‖`).
- **Already converged** (initial residual already below tolerance).
- **Near-convergence creep** (residual dropping slowly/monotonically — the OSGS linear-rate case).
- **Hard stall** (no progress for N steps).
- **Divergence** (merit/residual increasing).
- **Non-finite** (NaN/Inf appears mid-iteration — does any guard catch it, and is the best iterate restored?).
- **Parameter extremes** (the physical regime knobs: high/low `Re`, `Da`, fine/coarse `h`; and the numerical knobs at their min/max).
- **Boundary between two regimes** (the value that flips a guard — test on both sides).

For each: write the predicted step sequence and the predicted terminal `(success, state, reported metric)`. Then ask: *is that terminal state correct and honestly reported?*

## 4. Metric & normalization scrutiny
For **every threshold comparison** in the algorithm, interrogate the metric, not just the number:
- **Units / dimensions.** What are the units of each side of the comparison? Are they consistent? A threshold compared against a dimensional quantity is regime-dependent and usually wrong.
- **Absolute vs relative.** Is this an absolute floor or a relative reduction? Relative to *what scale* (initial residual? solution magnitude? a per-field norm?) — and is that scale the right invariant? (e.g. scaling a convergence gate by the *residual* norm keeps it encoding-invariant; scaling by the *solution* magnitude makes it non-covariant across fields with different physical units — pressure vs velocity.)
- **Per-field vs global.** For multi-field (e.g. velocity/pressure) systems: does a global norm let one field's scale dominate and mask the other? Should the gate be per-field?
- **Extremes.** What does the metric do as the quantity → 0 (true-zero residual: does the relative gate collapse to a floor?) and → ∞ (huge residual: does it overflow / saturate)? At `h → 0`?
- **Tie to discretization.** Should the tolerance scale with the discretization error (`O(h^{k+1})`) rather than be a fixed constant, so the solver neither over- nor under-solves relative to the mesh?
- **Exact-vs-inexact identities.** For every algebraic simplification or identity used inside a test (a directional derivative like `∇Φ·dx = -2Φ`, an affine-invariance claim, a "this step is a descent direction" assumption), ask: does it assume the EXACT Jacobian/operator? Is it then used anywhere with an APPROXIMATE one (frozen / lagged / Picard / inexact-Newton)? An identity silently demoted to an approximation can make a line search or guard subtly wrong in exactly the regime that stresses it.

## 5. Parameter logic: tunable knob vs state-driven automatism
For **every input parameter / threshold / count**, classify it and challenge its existence:
- **What does it actually control?** Trace it to the line that consumes it. If nothing consumes it → dead config (flag for removal). If its effect is not what its name implies → naming/contract bug.
- **Physical vs heuristic.** Is it a *genuinely problem-dependent physical/modelling* quantity the user must set (e.g. viscosity, a regime threshold tied to the physics), or an *internal numerical heuristic* (a stall window, a damping factor, an iteration cap)?
- **Should it be automatic?** For internal heuristics, ask hard: *could the algorithm measure the state of the solution/residual and set this itself, more robustly than a hand-tuned constant?* Hand-tuned internal constants are fragile — they are right for the cases they were tuned on and silently wrong elsewhere. Prefer an automatism that adapts to the observed residual history, mesh, or convergence rate. State explicitly which knobs are good candidates to replace with state-driven logic and what that logic would measure.
- **Safe default & failure mode.** Is the default safe? If a user sets a bad value, does the algorithm *fail loudly* or *silently corrupt the result*? Silent-corruption knobs are the dangerous ones — either bound-check them or make them automatic.
- **Config-profile divergence.** Compare the knob across config profiles (production vs test / sweep / diagnostic). A knob that is a STATE-DRIVEN AUTOMATISM in one profile (e.g. a noise floor scaled by a condition estimate) but a FIXED LITERAL in another is a red flag: the fixed-literal profile is the one likely to silently mis-converge, and the automatism already demonstrates what the right behaviour should be.
- **Coupling.** Does this knob interact with another (e.g. an iteration cap that truncates a solve whose tolerance is set elsewhere)? Mismatched coupled knobs are a classic silent-degeneration source.

## 6. Interaction, ordering, and masking
- **Guard interaction:** can two guards disagree — one accepting a state another should reject? Walk the precedence.
- **Ordering:** does the order of checks change the outcome? Is an early accept short-circuiting a later, stricter check?
- **Policy re-interpretation (verdict laundering):** THE highest-yield masking check. When a termination verdict passes through a SECOND decision layer — a policy object, an acceptance map, a wrapper that classifies the kernel's `stop_reason` — check whether that layer can FLIP a "not-converged" verdict into "success." The kernel may honestly report `max_iters` / `stall` / `xtol`, and a downstream policy may accept it unconditionally (and bypass the honest-exit gate the kernel applied). The most dangerous masking lives at this boundary, not in the kernel's own branches — audit the policy layer separately and trace each kernel verdict THROUGH it.
- **Masking:** is there a "soft" success (noise-floor, stall-accepted, max-iters-accepted) that can stand in for genuine convergence? For each, ask whether it can hide a regime where the method has actually failed (e.g. a stabilization degenerating into a lower-order method but reporting the higher-order method's success).

## 7. Hypothesis ledger
Maintain explicit, falsifiable hypotheses while you work:
> H: "If the coupled solve converges linearly and its iteration cap is set by knob K (value 5) while it needs ~50 steps, then it truncates mid-descent and returns the boot state labeled success."

Then **confirm or refute by tracing** (and, where cheap, by a targeted run). Record the verdict. Unresolved hypotheses are findings, not omissions.

# Output

Produce, concisely:
1. **State-machine map** — branches, guards, state variables, exits (the artifact from step 1).
2. **Per-branch verdict table** — `branch (file:line) | entry condition | action | consequence | desirable? (Y/N/uncertain) | concern`.
3. **Adversarial trajectories** — for each scenario in step 3: predicted path → terminal `(success, state, reported metric)` → correct & honest? 
4. **Metric/normalization findings** — each threshold whose units/scale/extreme-behaviour is wrong or fragile.
5. **Parameter table** — `knob | consumed at | physical or heuristic | safe default? | fails loudly? | keep tunable OR replace with automatism (what it would measure)`.
6. **Prioritized concerns** — robustness-first: anything that can (a) accept a non-converged state as success, (b) reject a converging one, (c) loop/diverge, or (d) silently change the mathematical problem, ranked above cosmetic issues. Label each as INTRODUCED by the change under review vs PRE-EXISTING (surfaced by the audit) — both matter, but the distinction tells the reader what's urgent versus a standing risk to file separately.

# Smells to flag (non-exhaustive)
- A success/accept exit reachable from a stalled, diverged, or noise-floor state without a gate that ties it to genuine convergence.
- A dimensional threshold compared against an encoding-variant quantity; an absolute floor where a relative-to-the-right-scale gate is needed.
- A global norm gating a multi-field system where one field's scale dominates.
- A magic constant with no derivation; a tunable internal heuristic that should self-calibrate from the residual history / mesh.
- Two coupled knobs (e.g. an iteration cap and a tolerance) sourced from different places that can silently contradict and truncate a solve.
- A guard that cannot distinguish *slow legitimate convergence* from a *stall* (and so bails on the slow-but-correct case).
- A parameter whose name implies one contract while the code does another; config that is validated-as-required but consumed by nothing.
- A metric reported on a non-converged exit that reads, downstream, as "converged."
- A termination verdict that a downstream policy/wrapper can re-interpret from "not converged" to "success" (verdict laundering) — especially one that bypasses an honest-exit gate the kernel applied.
- An exact-operator identity (affine invariance, `∇Φ·dx = -2Φ`, "the Newton step is a descent direction") used on an inexact / frozen / lagged operator.
- The same knob derived dynamically (state-driven) in one config profile and hard-coded in another.

# Discipline
Do not hand-wave a branch as "fine" without tracing a concrete state through it. Do not approve a threshold without naming its units and its behaviour at zero and at infinity. Do not accept a tunable knob without asking whether an automatism would be safer. When you cannot decide by reasoning alone, name the minimal run that would settle it. Robustness (the solver must converge to the correct answer, and say so honestly) outranks clarity outranks efficiency.
