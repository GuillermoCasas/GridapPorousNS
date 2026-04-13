---
name: vms-regression-guard
description: guard the porous navier-stokes / darcy-brinkman-forchheimer gridap.jl repository against repeated mathematical and numerical regressions. use when reviewing, proposing, or editing changes related to weak forms, vms stabilization, asgs vs osgs projection logic, exactnewtonmode vs picardmode jacobians, nonlinear solver safeguards, stopping criteria, drift metrics, manufactured-solution verification, or any paper-faithful porous solver implementation detail in this codebase.
---

# Goal

Prevent the assistant from repeating known mathematical, numerical, and architectural mistakes in this porous VMS solver project.

Treat this skill as persistent anti-regression memory for a repository whose behavior depends on:
- paper-faithful continuous VMS formulation
- exact handling of ASGS vs OSGS projection logic
- explicit ExactNewtonMode vs PicardMode linearization behavior
- safeguarded nonlinear solves
- meaningful MMS plateau verification

Do not optimize for brevity or code neatness at the expense of mathematical correctness or diagnostic power.

# Session startup

At the start of every relevant session:

1. Read `docs/lessons_learned.md` in full.
2. Extract a compact internal checklist of the highest-risk anti-patterns.
3. Keep that checklist active for the rest of the session.
4. Before proposing solver or formulation changes, compare the request against the known lessons.

If `docs/lessons_learned.md` does not exist, recommend creating it immediately using the exact table format defined below.

# What to protect

Focus on these surfaces only:

- weak form consistency
- stabilized residual construction
- ASGS vs OSGS projection handling
- formal adjoint sign correctness
- ExactNewtonMode vs PicardMode Jacobian consistency
- nonlinear convergence safeguards
- stopping criteria and scaling choices
- MMS / manufactured-solution verification logic
- authoritative vs legacy formulation boundaries

Avoid generic programming advice unless it directly affects one of the above.

# Repository assumptions

Assume all of the following unless the user explicitly says otherwise:

## 1. Mathematical primacy
The repository is organized around continuous VMS mathematics and paper-faithful operators. Favor mathematically authoritative behavior over implementation shortcuts.

## 2. ASGS and OSGS are not interchangeable
Treat ASGS and OSGS as different execution contracts.

- ASGS uses identity-style projection behavior and zero projection state in practice.
- OSGS requires iterative projected residual quantities and staggered fixed-point style updates.
- Do not silently reuse one branch's assumptions in the other.
- Do not treat projection state as optional when the formulation requires it.

## 3. Linearization mode is part of the contract
ExactNewtonMode and PicardMode are distinct mathematical choices, not mere performance toggles.

- Do not silently drop terms that belong to ExactNewtonMode.
- Do not claim Exact Newton behavior if the implemented Jacobian is Picard-like.
- Be explicit about which derivatives are kept, frozen, or removed.

## 4. Solver safeguards are intentional
Backtracking, merit logic, stagnation guards, damping, regularization shifts, and tolerance coupling are part of the solver design, not incidental clutter.

Do not propose removing or weakening them casually.

## 5. Verification logic is part of correctness
Manufactured-solution plateau checks are part of the project’s correctness policy.

Do not weaken plateau logic, hide relative-change diagnostics, or redefine success in a way that makes regressions harder to detect.

## 6. Legacy branches must stay explicitly labeled
Legacy or reduced branches may exist for regression comparison, debugging, or historical reasons.

Never present a legacy or reduced operator as the canonical formulation unless the user explicitly asks for that tradeoff.

# Mandatory review workflow

Follow this workflow whenever reviewing or proposing changes.

## Step 1: classify the change
Decide which category the request touches:

- weak form
- stabilization
- projection / OSGS loop
- Jacobian / linearization
- nonlinear solver
- stopping criteria
- MMS verification
- legacy branch / regression comparison

If multiple categories apply, review all of them.

## Step 2: cross-check against lessons learned
Check `docs/lessons_learned.md` for similar prior mistakes.

If a similar mistake exists:
- warn before proposing a change
- name the anti-pattern plainly
- summarize why it was wrong
- explain how the current request could repeat it

## Step 3: classify the formulation status
Whenever you describe a design choice, explicitly classify it as one of:

- **paper-faithful**
- **code-actual approximation**
- **legacy / deprecated**
- **debugging workaround**

Do not blur these categories.

## Step 4: run the relevant checks
Use the checklists below.

# Checklists

## Weak form and stabilization checklist

Check all of the following:

- Is the stabilized term using the correct formal adjoint weighting?
- Has any sign been flipped in a way that could turn stabilization into anti-diffusion?
- Are ASGS and OSGS residual treatments kept distinct?
- Is projection being applied only where that branch expects it?
- Has a canonical operator been replaced with a reduced one without explicit labeling?
- Has any change weakened consistency between strong residuals, weak form terms, and stabilization terms?

Warn immediately if the change risks:
- adjoint-sign reversal
- ASGS/OSGS projection confusion
- silent operator downgrading
- hidden consistency loss

## Jacobian and linearization checklist

Check all of the following:

- Does the Jacobian match the stated linearization mode?
- Are ExactNewtonMode-only terms being incorrectly dropped?
- Are Picard simplifications being mislabeled as exact derivatives?
- Are derivative terms for stabilization parameters, reaction terms, or adjoint terms being altered inconsistently?
- Does the proposed change preserve correspondence between residual and Jacobian structure?

Warn immediately if the change introduces a mismatch between residual definition and Jacobian definition.

## Nonlinear solver checklist

Check all of the following:

- Are Newton safeguards still active and meaningful?
- Are merit-function and backtracking rules preserved?
- Are damping or regularization changes justified numerically?
- Are stagnation guards and noise-floor logic still interpretable?
- Does the proposal accidentally favor aggressive steps over robust convergence?

Warn immediately against changes that:
- over-aggressively force full Newton steps
- weaken line search or merit checks
- redefine stagnation in a misleading way
- make failure modes harder to diagnose

## OSGS / projection loop checklist

Check all of the following:

- Is the initialization state consistent with the intended branch?
- Is projection drift distinguished from state drift?
- Are inner nonlinear solves and outer OSGS fixed-point iterations coupled sensibly?
- Are cached mass-matrix or L2 projection assumptions still valid?
- Are projection updates being mixed, accelerated, or stopped in a way that changes mathematical meaning?

Warn immediately if the change:
- confuses inner solve convergence with outer fixed-point convergence
- uses the wrong drift metric
- removes required physical or mass-matrix scaling
- treats OSGS as if it were a monolithic projection-free solve

## Stopping criteria checklist

Check all of the following:

- Is the stopping metric physically meaningful?
- Is the norm/scaling appropriate for mixed velocity-pressure state?
- Is mass-matrix weighting required here?
- Are state drift and projection drift being conflated?
- Are tolerances coupled correctly across outer and inner loops?
- Is the stop rule compatible with the formulation’s intended accuracy and diagnostics?

Warn immediately if the proposal:
- uses raw infinity norms where scaled or weighted drift is required
- couples tolerances incoherently
- declares convergence from the wrong signal
- hides physically important imbalance between variables

## MMS / verification checklist

Check all of the following:

- Does the proposal preserve plateau verification logic?
- Are error histories and relative-change histories still available?
- Are consecutive-pass rules still meaningful?
- Does the change make false convergence easier to declare?
- Are diagnostic outputs still sufficient to distinguish true plateau from budget exhaustion or solver failure?

Warn immediately if the proposal:
- weakens plateau criteria without strong justification
- removes useful diagnostics
- conflates base convergence with verification completion
- redefines success to mask regression

# Default warning style

When you detect a likely repeated mistake, say it plainly before continuing.

Use this structure:

**Regression risk**
- **Anti-pattern:** [short name]
- **Why it is risky here:** [project-specific reason]
- **What to preserve instead:** [correct policy]
- **Relevant prior lesson:** [brief pointer to matching lesson entry]

# Archival protocol

When a real error or anti-pattern is confirmed, append a new row to `docs/lessons_learned.md`.

Never delete history.
Never rewrite old lessons to make them look cleaner.
Prefer precise, factual entries over long explanations.

Only log lessons that are actually useful for preventing future mistakes:
- mathematical sign or consistency errors
- projection-policy mistakes
- Jacobian / residual mismatches
- incorrect convergence or stopping logic
- verification logic regressions
- silent canonical-to-legacy substitutions
- solver safeguard mistakes
- misleading diagnostics or scaling choices

Do not log trivial style issues.

# Required file format

The file `docs/lessons_learned.md` must use this exact table structure:

```markdown
# Lessons Learned – Porous VMS Solver

| Date | Error / Anti-pattern | Why it was wrong | Correct action taken | Related files | Tags |
|------|----------------------|------------------|----------------------|---------------|------|