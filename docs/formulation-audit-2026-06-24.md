# Formulation & Solver Audit — 2026-06-24

**Scope.** A deep, independent re-audit of the whole `src/` tree against the theory
(`theory/paper/article.tex` and the `theory/*` notes), plus a forensic re-examination of the
convergence *results* and the test harness/reporting that produces them. Three questions:

1. Is the theory faithfully transcribed into the code? Where not, what are the remaining inconsistencies and the recommended fixes?
2. Where does the formulation/solver make the algorithm **fragile or inaccurate**, and what is a viable alternative?
3. Where can it be **simplified / reorganised** for clarity, efficiency, or elegance?

**Method.** (a) A line-by-line independent read of every core file in `src/`. (b) A 9-domain
multi-agent audit of `src/` against the paper, with every finding adversarially re-verified by an
independent code-grounding skeptic *and* a theory-grounding skeptic (53 raw findings → 34 upheld, 19
refuted/trivial). (c) An independent recomputation of the 2D and 3D convergence rates straight from the
raw result files (HDF5 + JSON), *not* trusting the summary docs. (d) A fresh control experiment
(`/tmp/audit_3d_structured.jl`, log under `results/debug_results/`) run on a structured Kuhn tet mesh of
uniform, refinement-invariant quality, to separate "mesh quality" from "method/formulation" as the cause
of the 3D rate anomalies.

> **Provenance of this document.** Every code finding below was confirmed by reading the cited
> `file:line`. Every results claim was recomputed from the raw error arrays. The control-experiment
> numbers come from the run logged at
> `test/extended/ManufacturedSolutions3D/results/debug_results/audit_3d_structured.log`.

---

## 0. Executive summary

**The continuous VMS formulation is faithfully transcribed.** The strong residual, both Jacobians, the
adjoint operators and their sign conventions, the deviatoric/symmetric viscous expansions (2D & 3D), the
τ₁/τ₂ definitions, the OSGS projection policies, the Forchheimer-Ergun σ(α,u), the porosity field and
*all four* of its analytic derivatives, and both MMS oracles (2D and 3D) were each re-derived and match
the paper. The previously-catalogued divergences in `docs/solver/paper-code-divergences.md` (adjoint
sign, the omitted `(1/α)∇·(αa)v` term, the simplified τ forms, the unconstrained projection space, the
ε_num Jacobian-only penalty) hold up under re-scrutiny. **No correctness bug was found in the weak-form
assembly.**

**The substantive findings are not in the equations — they are in (i) a few documentation↔code
contracts that no longer match the code, (ii) the convergence *results* and the harness/reporting that
produces them, and (iii) hygiene/validation/fragility gaps.** In order of importance:

| # | Finding | Severity | Kind |
|---|---|---|---|
| **B-1** | **3D P1-ASGS is genuinely sub-optimal (L²u rate ≈ 1.2, not 2) even on a perfect uniform mesh** — reproduced on the structured Kuhn control. It is **method-intrinsic, not mesh quality.** OSGS recovers accuracy on the same mesh. | High | results/theory |
| **B-5** | **P2-3D has a genuine under-stabilization defect at the paper's c₁=4k⁴, confirmed on a perfect uniform mesh** (converged fine pairs show errors *growing*). It is **not mesh quality** — c₁=4k⁴ is structurally too small for P2 tetrahedra; **c₁×4 fixes it** (optimal rates, ≈40× smaller errors). The solver line-search failures at paper c₁ are a *symptom* of this. | High | results/theory |
| **B-2** | **The committed "3D-P2 divergence" tables conflate the real B-5 defect with `success=False` solves plotted as if valid.** The reporting tool does not gate on `success`, overstating the divergence (the genuine signal is the growing error on *converged* fine pairs). | High | reporting |
| **B-3 / B-6** | At every shared-failed level ASGS and OSGS report **byte-identical** errors. **Mechanism (control-confirmed): a failed OSGS solve silently reports the ASGS Stage-I boot state under the OSGS label** — so a sweep can record an ASGS error in an OSGS column. | Med | reporting/harness |
| **A-1** | ✅ **RESOLVED 2026-06-25.** `dynamic_*` Re/Da budget knobs were declared & validated in `config.jl` but had ZERO `src/` consumers. Relocated to the harness-frame `test/extended/harness_dynamic_budget.jl`; `SolverConfig`/schema/`base_config.json` no longer carry them. Behavior-preserving (2D MMS bit-identical). See F1. | High | doc↔code |
| **A-2** | ✅ **RESOLVED 2026-06-25 (Batch 1, `2f50d6d`).** The scale-free ε_M/ε_C gate is unconditionally attached by `solve_system` as THE production success test; the three comments that mislabeled it "diagnostic / not yet wired in" were corrected to state it is authoritative, the per-iteration cost is documented as intentional (with a `[future]` opt-out note), and `eps_tol_momentum`/`eps_tol_mass` now fail loud at load (C.2). | High | doc↔code / perf |
| **B-4** | Provenance gap: the most recent committed 3D result files cannot be reproduced from the current (uncommitted-modified) harness; their `mesh_algorithm`/level-set match no function now in `smoke3d.jl`. | Med | reproducibility |
| **C-x** | Fragility: silent ILU→identity preconditioner fallback + no GMRES convergence check (F3, open); missing load-time validation (Batch 1); **✅ the `1e-12` `|u|`-floor magic number → config input `velocity_magnitude_derivative_floor` (C.4, done 2026-06-25)**; a very loose 3D mass gate `eps_tol_mass=0.8` (F4, open). | Med/Low | fragility |
| **D-x** | Simplification: **✅ `build_picard_jacobian` now a wrapper (FORM-01); `_build_coeffs`/`_tau_ns_inv`/`_grad_div` helpers dedup the setup/τ/grad-div (FORM-05, TAU-05, VISC-01); dead `gridap_extensions.jl` deleted (VISC-02); dynamic-budget knobs relocated (F1); `accelerators.jl` wired into OSGS (NONL-03); `p_ex` dedup'd (DRIV-07) — all 2026-06-25.** `diagnostics_helpers.jl` removed in Batch 1. | Low | cleanup |

The rest of this document expands each part with `file:line` evidence, the recommended action, and (for
fragility items) a viable alternative.

---

## Deferred follow-ups — implementation checklist (post Batch-1)

**Batch 1 landed 2026-06-25** (branch `cleanup/contract-drift-and-fragility`, commit `2f50d6d`,
behavior-preserving — 2D MMS bit-identical, Blitz 194/194, Quick 57/57): A.2 (conv-gate comment
corrections), C.2 (fail-loud `validate!` asserts), C.3 (loose-gate doc), C.4 (`1e-12`→named const),
C.5 (`is_sigma_constant` trait), NONL-05 (ILU ctor defaults removed), the dead-code removals
(`diagnostics_helpers.jl`, FORM-03 dup) + `[unused]` labels, the A.1 **documentation** fix, and two new
tests. The items below were intentionally deferred; each is self-contained and ordered roughly by value.

### F1 — A.1 field relocation: move the `dynamic_*` Re/Da knobs out of production `SolverConfig` → §A.1
✅ **Landed 2026-06-25** (branch `cleanup/contract-drift-and-fragility`), behavior-preserving. The 8 knobs
were removed from `SolverConfig` (`src/config.jl`) + their 4 `validate!` asserts, from
`config/porous_ns.schema.json`, and from `config/base_config.json`. Their defaults now live in the new
harness-frame helper `test/extended/harness_dynamic_budget.jl` (the single source of truth +
`read_mms_dynamic_budget`, which reads an optional top-level `mms_dynamic_budget` block and **fails loud**
if a legacy `dynamic_*` key is still under `numerical_method.solver`, so a forgotten migration can't
silently fall back to a default). All 6 harness read sites (the two `run_test.jl`, `probe_stiff_diagnose`,
the two diagnostics probes, `smoke3d.jl`) consume the reader. The only ever-overridden value
(`dynamic_newton_re_iterations: 150`, in **13** configs — not 38) moved to a top-level `mms_dynamic_budget`
block; the inert base-valued `dynamic_ftol_*` keys were dropped from the other 21 (incl. the
Cocquet-EXPERIMENT configs whose harness never read them — provenance-safe, they were never on a read
path). **Verified:** 2D MMS bit-identical (every solution dataset), reader returns 150 for all 13 live
configs and defaults elsewhere, fail-loud guard fires, 3D smoke clean, Blitz 194/194, Quick 57/57.
- **Alternative (still open, bigger):** a production element-wise **cell-Péclet** adaptive budget in
  `src/solvers` (the frame-independent analogue) — changes production behavior, needs its own baselines.

### F2 — FORM-01: collapse `build_picard_jacobian` to a wrapper → §D
✅ **Landed 2026-06-25** (with the C.4 + D-x cleanup, behavior-preserving). Following the recipe exactly:
(1) added `test/blitz/picard_jacobian_equivalence_blitz_test.jl`, which assembles BOTH Jacobians on a tiny
mesh at a non-trivial Forchheimer iterate and asserts the ASGS *and* OSGS matrices are byte-identical
(`Array(A_picard) == Array(A_general)`); (2) once it passed, replaced `build_picard_jacobian`'s body with a
one-line wrapper over `build_stabilized_weak_form_jacobian(…, false, PicardMode())`. Verified by the new
equality test + a 2D Re=1e6 ASGS/OSGS run (bit-identical errors; OSGS exercises the Picard fallback) +
Blitz 196/196 + Quick 54/54.

### F3 — Batch 2 / C.1: ILU-GMRES honesty (behavior-CHANGING, 3D-only) → §C.1
- `src/solvers/linear_solvers.jl`: call `gmres!(…; log=true)`, inspect `ch.isconverged`; on non-convergence
  signal failure through `eval_linear_system_resolution!` (cascade → Picard) or loud-warn with the achieved
  residual. Make the ILU→identity fallback an explicit config-gated `allow_unpreconditioned_fallback`
  (schema + `LinearSolverConfig` + JSON).
- **Scope:** only the `ILU_GMRES` path (3D meshes > `LU_DOF_LIMIT`); the 2D harnesses use `LUSolver`, so 2D
  is unaffected. **Re-baseline the 3D structured-Kuhn control before/after** (it is *expected* to change
  which fine-mesh OSGS solves report success — that is the point).

### F4 — C.3: tighten / supplement the mass gate → §C.3
- Investigate why `eps_tol_mass` cannot be < 0.8 for 3D k=2; if it genuinely must stay loose, add a
  separate tighter check on the pure-divergence ratio `‖∇·(αu)‖/‖∇(αu)‖ ≤ √d` (already computed in
  `convergence_criterion.jl` as the self-check) so continuity is still gated. No behavior change landed.

### F5 — element-type-aware c₁ for P2 tets (the B.5 fix) → §B.5
- Add a config-driven `c1_multiplier` (or a per-`(element, order)` table), explicitly `[code-actual]`, so
  P2 tetrahedra use a larger c₁ (the control showed **c₁×4 fixes P2-3D**); 2D/quad stays paper-faithful at
  `4k⁴`. Also reconcile `docs/mms/3d-p2-convergence-investigation.md`'s TL;DR (still says "mesh quality")
  with its README pointer (says falsified) using the B.5 c₁×1-vs-c₁×4 numbers (the doc itself is committed,
  but its TL;DR predates the structured-Kuhn control).

### F6 — make the 3D MMS test config-driven + add an official 3D-MMS test
- **Why:** `test/extended/ManufacturedSolutions3D/smoke3d.jl` is a manual sweep driver with hard-coded
  study params (`RE`/`DA`/`ALPHA0`/`R1`/`R2`/`L`, the `mesh_sequence`, `c1_mult`, the `kv`/`method` loops),
  the 3D analogue of the 2D `run_test.jl` — but unlike the 2D side it has **no committed config JSON and no
  automated guard**, so every 3D study is a hand-edited driver (the source of the deleted one-off scrap).
- **Recipe:** (1) lift the hard-coded study knobs into a JSON config (mirror the 2D
  `ManufacturedSolutions/data/*.json` shape, plus the 3D-specific `mesh_sequence` / `c1_mult` / domain slab
  z-extent); (2) add a `config <path>` entry mode to `smoke3d.jl` that reads it and drives the existing
  `solve_one`/`run_sweep*` machinery (reuses `mesh3d.jl`'s `structured_kuhn_model` / `build_sequence`);
  (3) commit a lean smoke config (`data/smoke3d_p1.json`: §5.2, structured-Kuhn, 2-3 coarse levels, k=1,
  ASGS+OSGS, paper c₁); (4) wire a lean **official** extended test into `test/run_extended_tests.jl` that
  runs that config and asserts the solve succeeds and the P1 rate is within tolerance of optimal — making
  the 3D MMS path the official debug reference, with the bigger studies (structured control, c₁×4) as
  committed configs instead of ad-hoc drivers.
- **Verify:** the new extended test (a 3D solve — minutes of compile+solve), plus a manual config run
  reproducing one cell of the committed §5.2 numbers.

### Minor / opportunistic
- `_inv_centered.json` latent fragility: the official `test/quick/encoding_invariance_quick_test.jl` reads
  a config it must generate first — fine today, but a stale leftover can confuse a clean checkout.
- NONL-01/NONL-04 (Anderson guards) only matter if `accelerators.jl` is ever wired in (currently `[unused]`).

**Bit-identity verification recipe (reused for F1/F2):** capture a clean pre-change baseline by
`git stash push -- src/`, run a couple of 2D cells
(`run_test.jl phase1_quad_k1.json --filter kv=1,kp=1,etype=QUAD,Re=1.0,Da=1.0,alpha0=0.5 --max-N 40 --h5 debug_results/baseline_pre.h5`),
`git checkout -- src/` to restore edits, re-run to `baseline_post.h5`, and compare the `err_*` arrays for
exact equality.

---

## Part A — Theory ↔ code consistency

### A.0 What is faithful (re-verified, not assumed)

- **Strong momentum/mass residual** (`continuous_problem.jl` `eval_strong_residual_u/p`) matches
  `eq:StrongMomentumEquation`/`eq:StrongMassEquation` term-by-term, including the IBP pressure form
  `−p(α∇·v + ∇α·v)` (= `−∫ p ∇·(αv)`) and `∇·(αu) = α∇·u + u·∇α`.
- **Adjoint sign discipline.** `strong_adjoint_momentum` returns `+α(∇v)'·u` and `B_S` subtracts the
  adjoint; the `σv` term is subtracted. Matches Eq.39/Eq.50 and the documented A²−B² symmetry. The
  `(1/α)∇·(αa)v` omission is the paper's own simplification (article.tex ~L800).
- **Viscous operators** (`viscous_operators.jl`). `∇·ε^d(u)=½Δu+(½−1/d)∇(∇·u)` with coefficient `0` in
  2D and `+1/6` in 3D; `∇·ε(u)=½Δu+½∇(∇·u)`. The 2αν factor (μ=αν), the weak Jacobian (linear ⇒ `du`
  for `u`), and the self-adjoint reuse on `v` are all correct. The 3D MMS oracle (`mms3d.jl`) uses the
  matching `∇·(2αν∇ᵈu)=2ν(∇ˢu·∇α−⅓(∇·u)∇α)+ανΔu+(αν/3)∇(∇·u)`.
- **τ₁/τ₂** (`tau.jl`) implement `eq:Tau1Final`/`eq:Tau2Final` exactly: `τ₁=1/(α·τ_NS⁻¹+σ)`,
  `τ₂=h²/(c₁ α τ_NS)`, `τ_NS=(c₁ν/h²+c₂|w|/h)⁻¹`, `c₁=4k⁴, c₂=2k²`. The dropped `εh²` and `C_α` terms
  are the paper's own §4.2 simplifications. The dτ/du derivatives are the correct chain rule.
- **OSGS projection policies** (`projection.jl`) implement the §4.4 `(1−Π)` trim correctly; the
  unconstrained `V_free/Q_free` projection space matches the divergences-ledger requirement.
- **Reaction** (`reaction.jl`): `σ=a(α)+b(α)|u|`, `a=σ_lin((1−α)/α)²`, `b=σ_nonlin(1−α)/α`; `dσ/du`
  is the exact `b·(u·du)/|u|`.
- **Porosity** (`porosity.jl`): `α`, `∇α`, the scalar Laplacian, and the full Hessian were each derived
  by hand and match; the Hessian trace equals the Laplacian (mutually consistent), and the logistic is
  genuinely C∞ across the annulus boundaries. The MMS oracles are exact for the same operators the
  solver assembles.
- **ε_num** is Jacobian-pressure-block only in *both* Jacobian builders and absent from the residual and
  the VMS subscale — cancels in the residual, vanishes at convergence, as documented.

### A.1 [HIGH] Dynamic Re/Da iteration-budget knobs are documented as production behaviour but have no `src/` consumer

CLAUDE.md states: *"`dynamic_picard_re_threshold` / `dynamic_picard_da_threshold` automatically widen
Picard's iteration budget at high Re / Da. `dynamic_ftol_ceiling` couples the residual tolerance to mesh
resolution."* These fields are declared and `validate!`'d in [config.jl](src/config.jl), but:

```
grep -rn 'dynamic_picard_re_threshold|dynamic_picard_da_threshold|dynamic_newton_re_threshold' src/
   → only src/config.jl (declaration). NO consumer in src/solvers/ or run_simulation.jl.
   Consumers are test/extended/*/run_test.jl (the MMS/Cocquet harnesses).
```

So the dynamic-budget *logic* lives in the harnesses, which build their own solvers; the production
`solve_system` path never reads these knobs. A production user must nonetheless supply them (mandatory,
no-default struct fields) where they are provably inert, and the CLAUDE.md/​docstring description of live
solver behaviour does not exist in `src/`.

**Recommended action.** Decide the intended home. Either (a) lift the dynamic-budget logic into
`src/solvers` (so the documented behaviour is real on the production path), or (b) move these fields out
of the production `SolverConfig` and into a harness-only config block, and correct CLAUDE.md +
the config docstrings to say the dynamic budgeting is a harness feature. Do not leave required production
knobs that nothing in `src/` reads.

> **Implementation status (2026-06-25, branch `cleanup/contract-drift-and-fragility`).** ✅ **Fully
> landed** (option (b) in full). Both halves are done: (a) the documentation correction (CLAUDE.md, the
> `SolverConfig` field comments, the cell-Péclet note) and (b) the field relocation. The 8 knobs were
> removed from `SolverConfig` (+ their 4 `validate!` asserts), `config/porous_ns.schema.json`, and
> `config/base_config.json`; their defaults + reader now live in the harness-frame
> `test/extended/harness_dynamic_budget.jl`, consumed by all 6 harness read sites. The one
> ever-overridden value (`dynamic_newton_re_iterations: 150`, **13** configs — the audit's earlier "38"
> counted every file carrying the inert base-valued `dynamic_ftol_*` keys, none of which override
> anything) moved to a top-level `mms_dynamic_budget` block; the inert keys were dropped from the other
> 21. Behavior-preserving: 2D MMS bit-identical (every solution dataset), the reader returns 150 for all
> 13 live configs and defaults elsewhere, a fail-loud guard rejects a stray legacy key, and 3D smoke +
> Blitz 194/194 + Quick 57/57 are clean. See F1 in the deferred-follow-ups checklist for the full record.

### A.2 [HIGH] ✅ RESOLVED — The scale-free convergence gate's "diagnostic only / not yet wired in" contract was false

> **Resolved 2026-06-25 (Batch 1, commit `2f50d6d`).** All three stale comments were corrected: the
> scale-free ε_M/ε_C criterion is now documented everywhere as the **authoritative** production success
> gate (not a diagnostic) — [PorousNSSolver.jl:47](src/PorousNSSolver.jl#L47), the
> [build_convergence_probe](src/solvers/solver_core.jl#L121) docstring, and the `[authoritative gate]`
> label at [nonlinear.jl:74](src/solvers/nonlinear.jl#L74) / [:654](src/solvers/nonlinear.jl#L654). The
> per-iteration field-assembly cost is documented as **intentional**, with a `[future]` opt-out
> config-flag note ([solver_core.jl:123-125](src/solvers/solver_core.jl#L123)) rather than implemented (a
> flag would silently downgrade production to the weaker scalar-ftol fallback). The SOLV-04 elevation is
> covered: the `eps_tol_momentum`/`eps_tol_mass` gate tolerances now fail loud at load
> ([config.jl:307-308](src/config.jl#L307), C.2). The original finding, as audited, follows.

[PorousNSSolver.jl:45](src/PorousNSSolver.jl#L45) still says the criterion is *"not yet wired into
solve_system."* [build_convergence_probe](src/solvers/solver_core.jl#L121)'s docstring says attach it
*"only for trajectory diagnostics — the per-iteration field assembly is too costly for production,"* and
[nonlinear.jl:738](src/solvers/nonlinear.jl#L738) labels ε_M/ε_C `[diagnostic]`. **But**
[solver_core.jl:462-464](src/solvers/solver_core.jl#L462) unconditionally builds `conv_eval` and injects
it into *both* stage solvers, where [nonlinear.jl](src/solvers/nonlinear.jl) makes it the **authoritative
success gate** (`scale_free = solver.conv_probe !== nothing` → ε_M/ε_C decide every accept). So in both
production *and* the harness it is THE gate, not a diagnostic — and the per-iteration field re-assembly
(`momentum_force_envelope` builds five load vectors + rebuilds σ each accepted iterate) is paid on every
run.

Two real consequences: (1) the documentation is actively misleading about how convergence is decided;
(2) it elevates **SOLV-04** below from "a fallback" to "the only gate," so the unvalidated
`eps_tol_momentum`/`eps_tol_mass` deserve hard validation.

**Recommended action.** Update the three stale comments to state that the scale-free criterion *is* the
authoritative gate. If the per-iteration cost is unwanted in production, gate the attachment behind a
config flag; otherwise document the cost as intentional. Then add the validation in C.2.

### A.3 [LOW] The OSGS ExactNewton Jacobian's `dτ·R_old` / `dL*·R_old` terms use `R`, not `R−π` (and the theory note's wording is imprecise)

`theory/osgs_algorithm/osgs_algorithm.tex` (≈L1330) states: *"Passing a literal zero for π_h when
assembling the Jacobian is intentional: the projection value does not enter the frozen-π_h tangent at
all."* This is true for the **leading** term — `apply_jacobian_projection_u(ProjectFullResidual,…)`
returns `R_du` and ignores π ([projection.jl:59](src/stabilization/projection.jl#L59)). It is **not**
true for the ExactNewton product-rule terms: at
[continuous_problem.jl:520](src/formulations/continuous_problem.jl#L520),
`stab_R_old_u = apply_projection_u(ProjectFullResidual, R_u_old, …, proj_pi_u, is_osgs)` returns
`R_u_old − proj_pi_u`, and with the zero placeholder `proj_pi_u = 0` this is `R_u_old`. That quantity
feeds `dτ_1·R_old` and `dL*·R_old` at [continuous_problem.jl:528](src/formulations/continuous_problem.jl#L528).
The exact frozen-π tangent wants `(R−π)` there, not `R`.

This is **not a correctness bug** — the residual uses the live π, so the converged solution is correct,
and the tangent is *already* deliberately inexact (∂π/∂u is dropped; the doc acknowledges a
linear/superlinear rate). It is a precision-of-claim issue plus a small extra inexactness in the dτ/dL*
terms that can only affect the Newton *rate*.

**Recommended action.** Fix the wording in the tex note and the inline comment
([osgs_solver.jl:155-156](src/solvers/osgs_solver.jl#L155)) to say the projection value *does* enter the
`dτ`/`dL*` terms and is being deliberately approximated by zero. Optionally pass the live frozen π into
the Jacobian builder for a slightly tighter tangent (cheap; π is already computed for the residual).

### A.4 [LOW] Config-strictness gaps relative to the repo's own hard rule

- The JSON schema declares `required` for exactly one object (`linear_solver`); presence is enforced in
  practice by the no-default `@kwdef` structs, but the schema does not encode it
  ([porous_ns.schema.json](config/porous_ns.schema.json)). Add `required` arrays mirroring the structs.
- `config/base_config.json` omits the now-required `eps_val` and fails `load_frozen_config`
  (documented; **DRIV-01**). Either add `eps_val` to the canonical example or document that it is
  intentionally incomplete.
- Doc bug: the `PhysicalProperties` struct docstring ([config.jl:23](src/config.jl#L23)) calls `eps_val`
  *"porosity ε of the medium (>0)"* — it is the compressibility/pressure-penalty ε (α is porosity), and
  it may be 0. The inline field comment is correct; the struct docstring is not.

---

## Part C — Fragility / inaccuracy, with viable alternatives

### C.1 [Med] ILU-GMRES silently degrades to unpreconditioned, and never checks GMRES convergence

[linear_solvers.jl:90-114](src/solvers/linear_solvers.jl#L90) catches any ILU factorization exception and
substitutes the **identity** preconditioner with only a `println` warning; the saddle-point (u,p)
Jacobian then very likely will not converge under plain GMRES. [linear_solvers.jl:118-123](src/solvers/linear_solvers.jl#L118)
calls `gmres!(…; log=false)` and returns `x` **without inspecting convergence**, so a non-converged
linear solve is handed back as if exact. This is directly relevant to the **3D fine-mesh OSGS** path,
which auto-switches to ILU-GMRES above 80k DOF ([smoke3d.jl:160](test/extended/ManufacturedSolutions3D/smoke3d.jl#L160))
— exactly where a poor ILU is most likely.

**Alternative.** Pass `log=true`, capture the convergence history, and on `!converged` signal failure
back through `eval_linear_system_resolution!` (so the nonlinear cascade rolls back / falls to Picard)
or at minimum emit a loud warning with the achieved relative residual. Make the identity fallback an
explicit config-gated policy (`allow_unpreconditioned_fallback`) rather than a silent default.

### C.2 [Med] Missing load-time validation lets physically/numerically invalid configs through

`validate!` ([config.jl](src/config.jl)) checks the solver tolerances but **`DomainConfig` is entirely
unvalidated** and the reaction coefficients are unchecked:

- No `sigma_linear ≥ 0`, `sigma_nonlinear ≥ 0`, `sigma_constant ≥ 0` ⇒ σ can be made negative,
  violating the paper's SPSD assumption ([reaction.jl:100-104](src/models/reaction.jl#L100)). **MODE-03.**
- No `0 < alpha_0 ≤ 1`, no `r_1 < r_2`, no `bounding_box` length/parity check ⇒ `α≤0` makes `((1−α)/α)²`
  and the MMS `u_ex = α⁻¹·S` blow up; `check_mms_parameters` guards this in the harness only.
- No `eps_tol_momentum > 0`, no `0 < eps_tol_mass` on the **authoritative** production gate (A.2).
  **SOLV-04.**
- No positivity guarantee on the velocity floor: `SmoothVelocityFloor` is C∞ only when `u_floor>0`; at
  `u_base_floor_ref=0` (with `h_floor_weight=0`, as the harness forces) it degenerates to `sqrt(u·u)`,
  non-differentiable at 0 — the very failure the floor exists to prevent. **MODE-02.**

**Alternative.** Add these asserts to `validate!` so a malformed config fails loudly at load, mirroring
the existing `ftol`/`xtol` checks and the no-silent-defaults policy.

### C.3 [Med] The 3D k=2 mass-convergence gate `eps_tol_mass = 0.8` is extremely loose

ε_C = ‖εp+∇·(αu)−g‖ / (‖∇(αu)‖+‖g‖). A tolerance of **0.8** accepts an iterate whose mass-equation
residual is 80% of the flux-gradient envelope. The momentum gate `eps_tol_momentum=1e-9` is tight; the
mass gate is not. This plausibly contributes to the large "converged" P2 pressure errors (H¹p ≈ 2.3 at
the coarsest converged level). The looseness is intentional (the k=2-gate lesson cites mass-residual
scale sensitivity), but 0.8 is a wide margin to leave on the *only* check of the pressure/continuity
balance.

**Alternative.** Investigate why ε_C cannot be tightened (is the `‖∇(αu)‖` envelope the right scale for
the P2 pressure block?) rather than accept 0.8 as a fixed constant; if it genuinely must stay loose,
add a separate, tighter check on the pure-divergence ratio `‖∇·(αu)‖/‖∇(αu)‖` (already computed as the
√d self-check) so continuity is still gated.

### C.4 [Low] ✅ RESOLVED — A `1e-12` `|u|`-floor magic number was copy-pasted across formulation sites

`mag_u_reg = mag_u + 1e-12` appeared in `reaction.jl` (dσ/du) and `tau.jl` ×2 (dτ derivatives) — a
no-magic-numbers violation; the copies could drift.

> **Resolved 2026-06-25.** Batch 1 first collapsed the literals to one named const
> `VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR`; this change promotes it to a documented **config input**
> `PhysicalProperties.velocity_magnitude_derivative_floor` (schema + `base_config.json` + fail-loud
> `validate!`), carried on `SmoothVelocityFloor`, read via the `velocity_magnitude_derivative_floor(reg)`
> accessor, and threaded through `DSigOp`/`DTau1Op`/`DTau2Op` into the three leaf sites — one source of
> truth. Default `1e-12` ⇒ behavior-preserving (2D MMS bit-identical). Fully documented in
> `theory/velocity_floor_regularization/` (including the double-flooring nuance: it is kept a *separate*,
> independently-tunable parameter from `epsilon_floor`/`u_base_floor_ref`).

### C.5 [Low] `sanitize_projection_policy` guards the §4.4 trim by concrete type, not by a nonlinearity trait

[projection.jl:157](src/stabilization/projection.jl#L157) rejects the constant-σ trim only for the
concrete `ForchheimerErgunLaw`, with a permissive default. A future nonlinear reaction law would slip
through and silently corrupt the OSGS stabilization (the trim `(1−Π)(σu)=0` only holds for constant σ).

**Alternative.** Define `is_sigma_constant(::AbstractReactionLaw)=false` (true only for
`ConstantSigmaLaw`) and gate the trim on that trait, so any new nonlinear law is rejected by default.

---

## Part D — Simplification / reorganisation

> **✅ Landed 2026-06-25 (behavior-preserving, 2D MMS bit-identical; Blitz 196/196, Quick 54/54):**
> FORM-01 (`build_picard_jacobian` → wrapper + byte-equality guard test), FORM-03 (already Batch 1),
> FORM-05 (`_build_coeffs`), TAU-05 (`_tau_ns_inv`), VISC-01 + VISC-02 (`_grad_div` helper unifies the 4
> inline grad-div copies; dead `gridap_extensions.jl` deleted). Also 2026-06-25: ✅ NONL-03
> (`AndersonAccelerator` wired into OSGS behind `osgs_anderson_enabled`, OFF by default — verified to
> accelerate, docs/solver/osgs-anderson-acceleration.md) and ✅ DRIV-07 (`PExFunc` with registered ∇).
> Still open: only the per-dimension grad-div *coefficient* note (VISC-01 `0.5−1/D`; the contraction
> itself is already unified in `_grad_div`).

- **`build_picard_jacobian` is a byte-equivalent duplicate** of
  `build_stabilized_weak_form_jacobian(…, PicardMode())` — every term coincides (traced). Both are live
  in the OSGS coupled solve ([osgs_solver.jl:159](src/solvers/osgs_solver.jl#L159) Newton vs
  [osgs_solver.jl:178](src/solvers/osgs_solver.jl#L178) Picard), so a future edit to one path silently
  desyncs the Newton/Picard pair. **Make `build_picard_jacobian` a one-line wrapper** over the general
  builder in `PicardMode()`. (**FORM-01**)
- **`dL_du_star_v` is computed twice** with identical args at
  [continuous_problem.jl:479](src/formulations/continuous_problem.jl#L479) and
  [:509](src/formulations/continuous_problem.jl#L509). Delete the second. (**FORM-03**)
- **The σ/τ₁/τ₂ + α/f/g/h/dΩ setup block is triplicated** across the residual and both Jacobian builders
  (~20 lines each). Extract one `_build_coeffs(...)` helper. (**FORM-05**)
- **τ_NS inverse-denominator is hand-written in four `tau.jl` functions**; extract `_tau_ns_inv` /
  `_tau1_inv` so the primal and its derivative share one source of truth. (**TAU-05**)
- **Dead code to remove or label:**
  - `gridap_extensions.jl` (`ContractGradDivOp`/`grad_div_op`) — no caller; `∇(∇·u)` is re-implemented
    inline in `viscous_operators.jl`. (**VISC-02**)
  - ✅ `accelerators.jl` (`AndersonAccelerator`) — 2026-06-25 wired into the OSGS stage
    (`_osgs_anderson_outer!`) behind the opt-in `osgs_anderson_enabled` (OFF by default), with
    depth/relaxation/safety/max-outer config + a unit test; verified to cut the staggered outer-iteration
    count ≈1.4–2.2× (docs/solver/osgs-anderson-acceleration.md). (**NONL-03**)
  - `diagnostics_helpers.jl` — references the removed `config.phys.*` / `ThermodynamicState` /
    `build_reaction_model` / 7-arg `compute_tau_1` API; it cannot run. Delete or rewrite + test.
    (**DRIV-02**)
  - ✅ The dynamic Re/Da budget knobs (A.1) — relocated 2026-06-25 to `test/extended/harness_dynamic_budget.jl` (F1); no longer in `src/`.
- **`EvalDivDevSymOp`/`EvalStrongViscSymOp` hand-transcribe the grad-div coefficient per dimension**
  (`0.0` for 2D, `0.5-1/3` for 3D). Compute `0.5 − 1/D` from the tensor's `D` and reuse the single
  `grad_div_op` contraction so a new dimension can't drift. (**VISC-01/VISC-02**)
- ✅ **DRIV-07 (done 2026-06-25):** `mms_paper_2d.jl`'s three inline copies of `p_ex`/`∇p_ex` are unified
  into a `PExFunc` with a registered analytic `∇` (mirroring `UExFunc`). Solution bit-identical; the only
  change is the harness H1-pressure error metric, which shifts ≤2 ULP because it now uses the exact
  analytic `∇p_ex` rather than Gridap's ForwardDiff autodiff of the former closure — consistent with how
  the velocity H1 error already used `UExFunc`'s analytic ∇. (**DRIV-07**)
- **No tracked analytic regression test** for the canonical deviatoric strong/adjoint operator's 3D
  grad-div coefficient (the only such check is an untracked debug script). Promote it to a blitz/quick
  test. (**VISC-03**)

---

## Part B — Convergence results vs theoretical expectations

> This is the part where the user's intuition ("the clues are in the results") pays off. **Do not trust
> the summary docs** — every number below was recomputed from the raw error arrays, and the headline
> conclusions of the existing 3D investigation are re-tested against a clean control mesh.

### B.0 2D (k=1 QUAD) is honest and optimal — recomputed

Recomputing finest-pair rates straight from
`previous_results/validated_k1_quad_N640/phase1_quad_k1.h5` (48 configs): **every** cell has L²u ≥ 1.93
(median 2.03), H¹u ≥ 1.00 (median 1.01), L²p median 2.01. Zero sub-optimal at the finest pair. The
HDF5-stored `rate_u_l2` attribute is a *global* least-squares fit (≈1.7–1.96, dragged down by coarse
meshes) and is systematically lower than the finest-pair rate — so a report's headline depends on which
estimator it cites — but the method is genuinely optimal. **The 2D story is sound; the problem is
3D-specific.** (Minor reporting note: state which rate estimator a table uses.)

### B.1 [HIGH] 3D P1-ASGS is sub-optimal even on a perfect uniform mesh — method-intrinsic, not mesh quality

**The control experiment.** The structured Kuhn simplex mesh has uniform, refinement-invariant element
quality (constant inverse-estimate constant `C_inv` across levels), so it removes "mesh quality" as a
variable. Driving the *same* paper §5.2 case (Deviatoric, Constant-σ, Re=Da=1, α₀=0.5, paper c₁) through
the *same* `solve_one` the committed sweeps use:

| method | h-pair | L²u rate | H¹u rate | L²p rate |
|---|---|---|---|---|
| **P1 ASGS** | 0.149→0.099 | 1.24 | 0.92 | 0.84 |
| | 0.099→0.075 | 1.06 | 0.67 | 0.83 |
| | 0.075→0.060 | 1.27 | 0.88 | 0.95 |
| **P1 OSGS** | 0.149→0.099 | **2.21** | **0.99** | 1.66 |
| | 0.099→0.075 | **1.77** | 0.85 | 1.93 |

(Optimal: L²u = 2, H¹u = 1, L²p = 1.) On the **identical mesh and c₁**, ASGS sits at L²u ≈ 1.2 across
all three pairs while OSGS is at 2.21. ASGS's H¹u (≈ 0.9) is near-optimal — so the deficiency is
specifically the **L²-velocity extra (Aubin–Nitsche) order**, which ASGS fails to achieve in 3D and OSGS
recovers. This reproduces the committed `frontal_c1x1` numbers (recomputed: OSGS L²u ≈ 2.0 optimal,
ASGS L²u ≈ 1.3 sub-optimal) — but the structured-Kuhn control proves the gap **cannot be a mesh-quality
artifact**, because a same-mesh ASGS↔OSGS difference is by construction method-dependent.

> **Prior art / corroboration.** A separate structured-Kuhn control run earlier on 2026-06-24 (recorded
> in project memory and already reflected in [docs/README.md](README.md) line 24, which states the
> control "falsifies the old gmsh mesh-quality hypothesis") reached the same P1 conclusion: ASGS L²u
> ≈ 1.24, OSGS ≈ 1.91. **This run independently reproduces those numbers** (different driver, fresh
> meshes) — strong corroboration — and **completes the P2 case** (B.5), which the prior run could not
> (its P2 coarse solve failed and its fine P2 OOM'd). Note the *canonical* doc
> [3d-p2-convergence-investigation.md](mms/3d-p2-convergence-investigation.md) still leads with the
> mesh-quality hypothesis in its TL;DR, contradicting its own README pointer — that doc needs updating
> (see B.5 recommended action).

**Interpretation.** This is consistent with standard VMS theory: OSGS stabilises only the part of the
strong residual *orthogonal* to the FE space, removing the FE-projectable "consistency error" that ASGS's
full-residual stabilisation retains. That retained error is adjoint-inconsistent and caps the L²
superconvergence at the H¹ order. In 2D the codebase shows ASGS *is* optimal (B.0, median L²u 2.00); the
3D loss appears tied to the genuinely-3D tetrahedral discretisation of the (z-extruded) solution and the
deviatoric trace coupling — but the precise mechanism is secondary to the reproducible empirical fact.

**Why this matters for the existing narrative.** `docs/mms/3d-p2-convergence-investigation.md` and the
project memory frame 3D sub-optimality as *mesh quality / `C_inv` vs the c₁=4k⁴ budget*, "fixed by the
Frontal mesh." That framing is correct **for OSGS** (a better mesh lifts OSGS to optimal) but **wrong for
ASGS**: the ASGS L²-order loss is method-intrinsic and survives a perfect uniform mesh. Reports that
present "3D P1 optimal" should qualify it as *OSGS-optimal, ASGS-suboptimal*.

**Recommended action.** (1) Document the ASGS 3D L²-order deficiency honestly as a method property, not a
mesh defect. (2) Treat OSGS as the load-bearing 3D method (as the code already leans). (3) If ASGS 3D
optimality is wanted, that is a genuine formulation research question (the orthogonal-projection
consistency term), worth a paper footnote — not a c₁ tuning.

> **Solver note (same runs).** Both P1-OSGS control solves hit the **iteration cap (20)** and were
> accepted as success via the OSGS soft-stall policy (`accept_soft_stall=true`,
> [osgs_solver.jl:28](src/solvers/osgs_solver.jl#L28)), *not* by reaching the ε_M/ε_C gate. The accepted
> iterate is nonetheless optimal-rate and more accurate than ASGS — but "success=True" here means
> "budget-exhausted soft stall accepted," not "ε-converged." This is the documented OSGS linear-rate
> coupling (the JFNK motivation), but it means the OSGS success flag is weaker than it looks and the
> iteration cap (a config value) is silently load-bearing for OSGS accuracy.

### B.2 [HIGH] The committed "3D-P2 divergence" is dominated by `success=False` solves plotted as valid data

Recomputing the committed P2 rates straight from
`previous_results/convergence3d/convergence3d_results_frontal_c1x1_20260623.json` (Frontal mesh, paper
c₁):

| level | h | success | L²u | H¹u | L²p |
|---|---|---|---|---|---|
| 0 | 0.123 | **True** | 0.0272 | 0.905 | 0.268 |
| 1 | 0.094 | **True** | 0.0123 | 0.501 | 0.126 |
| 2 | 0.078 | **False** | 0.0620 | 2.863 | 0.656 |

The level 0→1 rates (both `success=True`) are **L²u = 2.92, H¹u = 2.17, L²p = 2.77 — fully optimal for
P2** (opt 3/2/2). The "divergence" is entirely the level-2 step, which **did not converge**
(`success=False`) and whose error values (0.062, 2.863) are whatever the failed solve happened to stop
at. The `pre_frontal` (Delaunay) file is the same story inverted: its level-1 solve *fails* and spikes,
while the *finer* level-2 solve (`success=True`) recovers to L²u 0.0062.

**The reporting does not gate on `success`.** [plot_convergence3d.py](test/extended/ManufacturedSolutions3D/plot_convergence3d.py)
`_plot_by_kv` reads the top-level `hs`/`l2u`/`h1u`/`l2p` arrays and `_seg_slope` computes a slope across
*every* consecutive pair, with no `success` filter; the top-level error arrays in the JSON drop the
per-level `success` flag (kept only in `levels[]`). So a failed solve is plotted as a data point and its
wild slope is annotated as a convergence rate. The "P2 diverges under refinement" headline is largely an
artifact of mixing non-converged solves into the rate table.

This does **not** fully exonerate the formulation. The committed files conflate two effects: (i) the
failed-solve contamination above, and (ii) a **genuine** under-stabilization defect. The "optimal"
frontal pair (level 0→1, L²u ≈ 2.9) is at *coarse* h (0.12→0.094); the clean-mesh control (B.5) goes
finer and shows the P2 error **turning around and growing** on converged fine pairs — the classic
loss-of-coercivity signature (error optimal-looking at coarse h, then diverging as h→0). So the right
reading is: **the reporting overstates the divergence (failed solves), but a real P2-3D coercivity defect
exists underneath it** — quantified and resolved in **B.5**. The fix is the c₁ budget, not the mesh.

**Recommended action.** (1) Make the plotter and any rate computation **exclude or visibly flag**
`success=False` levels (the data is in `levels[]`). (2) Re-derive the "errors grow under refinement"
claim using only converged levels on a fixed-quality mesh before attributing it to coercivity. (3) Never
compute a convergence rate across a non-converged solve.

### B.3 [Med] ASGS and OSGS report byte-identical errors at every shared-failed level

Across both committed P2 files, the *only* levels where ASGS and OSGS agree to all 16 digits are exactly
the levels where **both** report `success=False`:

```
frontal   lvl2: ASGS(False, l2u=0.06198525425452791) == OSGS(False, l2u=0.06198525425452791)
pre_frontal lvl1: ASGS(False, l2u=0.06196690717763982) == OSGS(False, l2u=0.06196690717763982)
```

At every mixed- or both-success level the two methods differ (as independent solves must). Two
independent nonlinear solves with different stabilisation cannot produce bit-identical errors unless they
returned the *same field* — i.e. a shared failed/fallback state (most likely a degenerate independent
remesh at that level on which both solves fail identically, before the method-specific terms can
diverge), or a data-handling artifact. Either way the failed-level numbers are doubly untrustworthy.

**Recommended action.** Add an assertion/diagnostic that flags identical ASGS/OSGS error tuples as a
red flag; surface the per-level mesh quality (min dihedral/radius-ratio) so a bad remesh is visible; and
(with B.2) exclude these levels from the rate table.

### B.4 [Med] The most recent committed 3D results are not reproducible from the current harness

The committed `convergence3d_results_frontal_c1x1_20260623.json` records
`mesh_algorithm = "gmsh_Frontal_alg4_independent_remesh"` and a 6-level P1 ladder, but **no function now
in `smoke3d.jl` produces that string or that ladder**: the sweep paths use `build_nested_family` and
write `mesh="nested_red"`; `build_sequence("frontal")` ([mesh3d.jl](test/extended/ManufacturedSolutions3D/mesh3d.jl))
defines only three lcs `[0.137,0.098,0.070]`. All of `mesh3d.jl`, `smoke3d.jl`, `plot_convergence3d.py`
are uncommitted-modified in the working tree. So the exact driver that produced the most recent committed
numbers is not recoverable from `HEAD`+worktree — a violation of the project's own
parameters→results-traceability rule.

A related minor provenance smell: `smoke3d.jl`'s `build_config` sets `physical_properties.eps_val = 1e-8`,
but `solve_one` builds the formulation with `eps_phys = 0.0` (default), so the formulation's ε is 0 and
the config's `1e-8` is dead for the solve — the stored config value is not the one used.

**Recommended action.** Restore/commit the exact driver (the Frontal independent-remesh sweep function)
that produced the committed 3D files, or re-run them with a committed, named function and archive the
config snapshot alongside, per `reproducible-results.md`. Make `build_config`'s `eps_val` and the
formulation's `eps_phys` agree (or document why they differ).

### B.5 [HIGH] P2-3D at paper c₁ has a genuine under-stabilization defect (confirmed on a clean uniform mesh) — and it is **not** mesh quality; c₁×4 fixes it

The structured-Kuhn control (uniform, refinement-invariant quality) ran P2 ASGS+OSGS at paper c₁ and
ASGS at c₁×4. Results (paper §5.2 case, all on the same mesh ladder):

**P2 ASGS, paper c₁ (=4k⁴=64):**

| h | success | L²u | H¹u | L²p |
|---|---|---|---|---|
| 0.149 | **False** | 9.58e-2 | 2.341 | 0.557 |
| 0.099 | True | 4.93e-2 | 1.816 | 0.464 |
| 0.075 | True | 1.09e-2 | 0.562 | 0.106 |
| 0.060 | True | **1.31e-2** | **0.871** | **0.136** |

The finest **two `success=True`** levels (0.075→0.060) give rates **L²u = −0.82, H¹u = −1.96,
L²p = −1.14 — the errors GROW under refinement, on converged solves, on a perfect uniform mesh.** Most
coarse P2 solves at paper c₁ also fail outright (`linesearch_failed` / `no_progress_stall`). So the
P2-3D defect is **real**, not merely the B.2 reporting artifact.

**P2 ASGS, c₁×4 (=256):** every level `success=True`, errors decrease monotonically, rates
**L²u = +1.67, +2.67, +2.30** (climbing toward the optimal 3), H¹u → +1.26, L²p → +1.48 — and the
**absolute errors are ≈ 40× smaller** than at paper c₁. The c₁ knob is exactly the coercivity lever
(`c₁ > 2ξ C_inv²`, article.tex `eq:conditions_on_num_param`).

**Verdict — what is right and wrong in the existing 3D-P2 narrative.**

1. ✅ **The c₁/coercivity hypothesis is correct.** Scaling c₁ up is the single lever that turns P2-3D from
   diverging-and-failing to converging-and-optimal, exactly as `docs/mms/3d-p2-convergence-investigation.md`
   claims, and it matches the paper's own remark that the effective c₁ is element-type dependent.
2. ❌ **It is NOT "mesh quality" / a bad-element tail.** The structured Kuhn mesh has uniform,
   refinement-invariant quality, yet P2 still fails / grows at paper c₁ and is fixed by c₁×4. So the
   deficiency is that **c₁ = 4k⁴ is structurally too small for P2 tetrahedra in general** (the geometric
   part of `C_inv` for tets — even uniform ones — exceeds the `4k⁴` budget at k=2 in 3D), *not* that the
   gmsh generator produces an occasional sliver. The "Frontal-mesh fix" helps P1-OSGS, but it cannot be
   the explanation for P2, which fails on a flawless mesh.
3. ⚠️ **The committed "divergence" tables conflate two things** (B.2): a genuine under-stabilization
   signal *and* failed-solve reporting contamination. The control separates them: the genuine signal is
   the growing error on *converged* fine pairs; the contamination is the `success=False` spikes.
4. ⚠️ **The P2 solve failures at paper c₁ are a symptom, not a separate solver bug.** Under-stabilization
   (too-small c₁) leaves the discrete operator poorly conditioned/near-non-coercive, so Newton's line
   search depletes. Fixing c₁ fixes the solver failures too (every c₁×4 solve converges).

**Recommended action.** (1) Adopt an **element-type-aware c₁** for P2 tetrahedra — a config-driven
`c1_multiplier` (or a per-(element, order) table), explicitly labelled `[code-actual]` as a deviation
from the paper's uniform `4k⁴`, with 2D/quad staying paper-faithful. This is the principled fix the
investigation doc already floats as option (3); the control shows it is *necessary*, not a band-aid.
(2) **Reconcile the canonical doc with its own README pointer:** `docs/README.md` (line 24) already
states the structured-Kuhn control *falsifies* the mesh-quality hypothesis, but
`docs/mms/3d-p2-convergence-investigation.md` still leads with "mesh quality / `C_inv` tail" in its TL;DR
(its 2026-06-21 body predates the control). Update that doc's TL;DR to the **c₁-budget-vs-tet-`C_inv`**
verdict (reproduced on a uniform mesh, *not* a bad-element tail), with this run's clean P2 c₁×1-vs-c₁×4
numbers as the evidence. (3) Measure the actual `C_inv` for P2 Kuhn vs P2 gmsh tets to calibrate the
multiplier.

> **Run log:** `test/extended/ManufacturedSolutions3D/results/debug_results/audit_3d_structured.log`
> (full P1/P2, ASGS/OSGS, c₁×1 and c₁×4; ~50 min wall). Driver: `/tmp/audit_3d_structured.jl`.

### B.6 [Med] A failed OSGS solve silently reports the ASGS Stage-I state under the OSGS label

The control reproduced the byte-identical anomaly (B.3) **and revealed its mechanism.** Every P2-OSGS
solve at paper c₁ failed (`success=False`) and reported an error **bit-identical to the standalone
P2-ASGS run at the same level** — including at (20,20,5) where ASGS *succeeded* (1.3101e-2) and OSGS
*failed* but still reported 1.3101e-2. Because the OSGS path runs the **ASGS Stage-I boot first**
([solver_core.jl:511](src/solvers/solver_core.jl#L511)) and the coupled solve, on failure, leaves the
iterate at that ASGS state, **a non-converging OSGS solve degenerates to ASGS and reports ASGS's error
under the OSGS label.** [osgs_solver.jl:163-168](src/solvers/osgs_solver.jl#L163) already flags this exact
failure mode as a `[known-fragility]` for the stall sensor; the control shows it also happens via
`linesearch_failed`. When OSGS *does* converge (all P1 cases), it produces genuinely different, more
accurate results — so the degeneration is silent and only bites on the hard (P2) cases that most need
OSGS.

**Recommended action.** When the OSGS coupled solve does not reach its own convergence verdict, mark the
result distinctly (e.g. `method="OSGS(degenerated→ASGS)"` or a separate flag) so a sweep cannot record an
ASGS error in an OSGS column. Combined with B.2/B.3, this stops the failed-solve numbers from masquerading
as method comparisons.


---

## Appendix 1 — Control experiment: full 3D structured-Kuhn data

Paper §5.2 case (Deviatoric, Constant-σ, Re=Da=1, α₀=0.5), structured Kuhn simplex mesh, domain
(0,1)×(0,1)×(0,0.3). Per-segment rates between consecutive levels (optimal: L²u=k+1, H¹u=k, L²p=k).
`†` = a segment that includes a `success=False` solve (rate is meaningless).

| case | h-pairs | L²u rates | H¹u rates | L²p rates | verdict |
|---|---|---|---|---|---|
| P1 ASGS c₁×1 | 0.149/0.099/0.075/0.060 | 1.24, 1.06, 1.27 | 0.92, 0.67, 0.88 | 0.84, 0.83, 0.95 | **sub-optimal** (L²u≈1.2) |
| P1 OSGS c₁×1 | same | 2.20, 1.78, 2.08 | 0.99, 0.86, 0.98 | 1.67, 1.93, 1.60 | **optimal** |
| P2 ASGS c₁×1 | same | 1.64†, 5.24, **−0.82** | 0.63†, 4.08, **−1.96** | 0.45†, 5.14, **−1.14** | **defect** (errors grow on converged fine pair; coarse solves fail) |
| P2 OSGS c₁×1 | same | all †† | all †† | all †† | all solves failed → reports ASGS state (B.6) |
| P2 ASGS c₁×4 | same | 1.67, 2.67, 2.30 | 0.83, 1.76, 1.26 | 0.94, 1.73, 1.48 | **converges, climbing to optimal**; ≈40× smaller errors |

Headline reads: P1 ASGS sub-optimal vs P1 OSGS optimal on the *same* mesh (B.1); P2 at paper c₁ has a
genuine coercivity defect on a *uniform* mesh, fixed by c₁×4 (B.5); a failed OSGS solve reports the ASGS
state (B.6).

## Appendix 2 — Index of the 34 upheld code findings (by domain)

Each was confirmed by an independent code-grounding skeptic and re-checked against the theory. Full
descriptions + recommended actions are in the audit-run record; the high-value ones are written up in
Parts A/C/D above.


**convergence-criterion**
- `CONV-05` (fragility, medium): MMS verifier numerical parameters (tau_err, eps_*, max_extra_cycles, require_consecutive_passes, rate_check_factor) are  — `test/extended/CocquetFormMMS/run_test.jl:401-412`
- `CONV-02` (inconsistency, low): Inline literal `1e-2` √d self-check margin in evaluate_convergence violates the repo no-magic-numbers / config-strictnes — `src/solvers/convergence_criterion.jl:211`
- `CONV-04` (fragility, low): Sub-optimal-rate budget uses a bare power of h with NO reference-error / leading-constant normalization, so the rate_che — `src/solvers/mms_verification.jl:130-131`

**driver-mms-io**
- `DRIV-02` (inconsistency, low): diagnostics_helpers.jl references config.phys and a removed reaction/tau API; it cannot run — `src/diagnostics_helpers.jl:46,51,56,94,96,97,99`
- `DRIV-03` (inconsistency, low): JSON schema declares no `required` fields for physical_properties/solver/etc., so it does not enforce the no-implicit-de — `config/porous_ns.schema.json`
- `DRIV-05` (inconsistency, low): Production export_results writes no provenance (Re/Da/α₀/params) into or alongside the VTK output — `src/io.jl:28-51`
- `DRIV-07` (simplification, low): MMS oracle re-derives p_ex and ∇p_ex inline instead of reusing the exact-field overloads, risking drift — `src/problems/mms_paper_2d.jl:294-295`
- `DRIV-08` (fragility, low): ForchheimerErgunLaw.dsigma_du uses an inline 1e-12 velocity-floor literal (magic number) in formulation/model code — `src/models/reaction.jl:114`

**formulation-core**
- `FORM-01` (simplification, low): build_picard_jacobian is a full byte-for-byte duplicate of build_stabilized_weak_form_jacobian(..., PicardMode()) and ca — `src/formulations/continuous_problem.jl:371-437`
- `FORM-02` (fragility, low): dsigma_du regularization floor 1e-12 is a hard-coded magic number in the Exact-Newton reaction derivative — `src/models/reaction.jl:114`
- `FORM-03` (simplification, low): dL_du_star_v is computed twice in build_stabilized_weak_form_jacobian (redundant recomputation) — `src/formulations/continuous_problem.jl:479`
- `FORM-04` (fragility, low): τ rational/Forchheimer |u| nonlinearity is under-integrated by the polynomial quadrature rule (inexact, not just inexact — `src/formulations/continuous_problem.jl:196-218`
- `FORM-05` (simplification, low): Residual and both Jacobians repeat ~20 lines of identical setup boilerplate (α/f/g/h/dΩ unpack, σ/τ₁/τ₂ operator constru — `src/formulations/continuous_problem.jl:297-322,`

**models**
- `MODE-03` (fragility, medium): No SPSD / range validation on reaction coefficients and porosity bounds — σ can be made non-positive-semidefinite, viola — `src/config.jl:45-65`
- `MODE-01` (inconsistency, low): Hard-coded 1e-12 magic number in Forchheimer dsigma_du denominator (no-magic-numbers rule violation + double-flooring) — `src/models/reaction.jl:114`
- `MODE-02` (fragility, low): No validation that the velocity floor is strictly positive — SmoothVelocityFloor is only C¹ when u_floor>0; u_base_floor — `src/models/regularization.jl:55-58;`
- `MODE-04` (fragility, low): Porosity logistic saturation guard uses hard-coded ±100 thresholds (magic numbers) and α is silently 2D-only — `src/models/porosity.jl:70-74,`

**nonlinear-safeguards**
- `NONL-01` (fragility, low): ILU-GMRES silently returns a possibly-unconverged solution: no convergence check after gmres! — `src/solvers/linear_solvers.jl:118-123`
- `NONL-02` (fragility, low): ILU factorization failure silently falls back to identity preconditioner (unpreconditioned GMRES) with only a println — `src/solvers/linear_solvers.jl:90-114`
- `NONL-03` (simplification, low): AndersonAccelerator is fully implemented and exported but has zero consumers and zero config — dead code — `src/solvers/accelerators.jl:1-127`
- `NONL-04` (fragility, low): Anderson update! has no zero/near-zero residual guard before solving the least-squares system — `src/solvers/accelerators.jl:53-127`
- `NONL-05` (inconsistency, low): Hard-rule violation: ILUGMRESSolver keyword constructor backfills magic-number defaults (m=30, τ=1e-4, rel_tol=1e-11, ma — `src/solvers/linear_solvers.jl:72-74`
- `NONL-06` (fragility, low): Picard line-search reuses Armijo c1 as a multiplicative residual-reduction factor (1 - c1·α), a different mathematical r — `src/solvers/nonlinear.jl:434-438`

**projection**
- `PROJ-01` (fragility, low): sanitize_projection_policy guards the constant-σ trim by concrete law type, not by a nonlinearity trait — a future nonli — `src/stabilization/projection.jl:151-164`
- `PROJ-02` (simplification, low): ProjectResidualWithoutPressurePenalty is never instantiated in production code — dead policy reachable only from tests — `src/stabilization/projection.jl:34,121-143`

**solver-orchestration**
- `SOLV-04` (fragility, medium): Production convergence-gate tolerances eps_tol_momentum / eps_tol_mass are never validated (no fail-loud), violating the — `src/config.jl:166-167`
- `SOLV-03` (inconsistency, low): One-way ASGS Picard success uses scalar ℓ∞ ftol, not the scale-free cascade_step_outcome the ping-pong path and Newton s — `src/solvers/asgs_solver.jl:144`
- `SOLV-05` (fragility, low): discrete_l2_projection re-solves a fresh allocate_in_domain RHS each residual eval; b_vec/x_solve scratch is re-allocate — `src/solvers/osgs_solver.jl:59-64`

**tau**
- `TAU-01` (inconsistency, low): Hard-coded 1e-12 |u|-floor in dτ derivatives violates the no-magic-numbers rule (and duplicates/contradicts tau_reg_lim' — `src/stabilization/tau.jl:77`
- `TAU-02` (fragility, low): σ inside τ₁ is evaluated at effective_speed (mesh-dependent diffusive floor), not at the physical reaction_speed used in — `src/stabilization/tau.jl:37`
- `TAU-05` (simplification, low): τ_NS (A_NS) inverse-denominator is recomputed inline in four functions instead of a shared helper — `src/stabilization/tau.jl:36,`

**viscous**
- `VISC-03` (fragility, medium): No committed analytic regression test for the canonical (deviatoric) strong/adjoint operator's 3D grad-div coefficient;  — `test/quick/viscous_operators_quick_test.jl:52-88`
- `VISC-01` (fragility, low): 3D deviatoric grad-div coefficient (1/6) hard-codes d=3 instead of computing 0.5 − 1/D, and the per-dimension dispatch d — `src/formulations/viscous_operators.jl:159-170`
- `VISC-02` (simplification, low): gridap_extensions.jl (ContractGradDivOp / grad_div_op) is entirely dead code; the same ∇(∇·u) contraction is re-implemen — `src/formulations/gridap_extensions.jl:17-37`

**Refuted / trivial (19, not carried):** the second-pass verifiers rejected, among others: "all-Dirichlet
pressure block is singular" (refuted — the τ₂/εp stabilisation regularises it), "homotopy dilution changes
the converged solution" (refuted — `mult_mom/mult_mass` are always 1.0 in the core), "Da=0 mis-sizes the
Forchheimer MMS pressure scale" (refuted — the MMS sweeps use Constant-σ, which *does* expose Da),
"`base_config.json` missing `eps_val`" (already documented), and several `[trivial]` cosmetics
(unused imports, the 2D `0.0*grad_div` multiply, the ρ_{Λ⁻¹}(σ)→scalar simplification).

## Appendix 3 — Method & reproducibility

- Multi-agent audit run record (53 findings, per-finding two-lens verdicts): workflow `wf_9dc5a56d-800`.
- 2D rate recomputation: `previous_results/validated_k1_quad_N640/phase1_quad_k1.h5` (48 configs).
- 3D committed-data reanalysis: `previous_results/convergence3d/convergence3d_results_{frontal_c1x1,pre_frontal}_*.json`.
- 3D control experiment: driver `/tmp/audit_3d_structured.jl`, log
  `test/extended/ManufacturedSolutions3D/results/debug_results/audit_3d_structured.log`.
- Every code `file:line` in this document was opened and read; every results claim was recomputed from
  raw error arrays, per the "do not trust the reports" directive.
