# you asked

message time: 2026-05-18 22:26:06

Please read the documentation in artilcle.tex and osgs_algorithm.tex and the code (packaged by repomix) which implemnets it and look for inconsistencies or mistakes in the implementation. Pleae use the highest standard of rigor and analyze every part in fulll detail

---

# chatgpt response

I’ll treat this as a paper-to-code consistency audit, not a superficial code review. I’ll first establish the algorithmic contracts from the two LaTeX files, then trace the implementation file-by-file and classify any deviations by severity.

# 🚨 ALERT SUMMARY 🚨

**Audit scope:** `article.tex`, `osgs_algorithm.tex`, and the Repomix-packed Julia implementation. I focused on the paper-to-code contract: continuous PDE, VMS residuals, ASGS/OSGS projection semantics, stabilization parameters, adjoint operators, Exact Newton/Picard behavior, stopping logic, and configuration strictness. I did **not** execute the Julia test suite; this is a static mathematical/code-consistency audit.

**Traceability/Test Mandate:** The codebase has been marked for inconsistencies. Traceability entries and fast tests are explicitly required before considering these resolved. The repository already encodes this norm in its paper-consistency auditing framework and no-hard-coded-parameters rule set. 

## Issue counts by severity

- 🚨 **CRITICAL INCONSISTENCY:** 4
- 🐛 **LIKELY BUG:** 3
- ⚠️ **UNDOCUMENTED DEVIATION:** 5
- 📝 **DOCUMENTED APPROXIMATION:** 2
- ⏳ **NEEDS DERIVATION / TEST SEALING:** 3

## Most important findings

1. **The pressure stabilization parameter `τ₂` is missing the paper’s `ε h²` contribution.** The paper’s stability analysis explicitly contains the compressibility contribution through the pressure term and the `τ₂` stability bound, but the code computes `τ₂` using only `c₁ α τ_NS` plus `tau_reg_lim`. This is a direct mathematical mismatch. 

2. **`PaperGeneralFormulation` silently replaces `eps_val` by `max(eps_val, eps_floor)`.** This changes the mass equation from incompressible or user-selected compressible to forcibly compressible. The paper treats `ε > 0` as a deliberate penalty choice, especially in 3D manufactured tests, not as a hidden constructor correction. 

3. **OSGS inner-cascade failure handling appears too permissive for `max_iters_stagnation` / `xtol_stagnation`.** The algorithm text says iteration-cap and non-noise-floor stagnation are failures, and only true tolerance or noise-floor exits should be accepted. The implementation path, as reviewed, only treats a narrower subset of stop reasons as structural failures. 

4. **The 3D formal adjoint for deviatoric/symmetric viscous stabilization is not exact.** The code drops the `∇(∇·v)` contribution in adjoint-like operators. This is harmless only in special limits, but not for the general 3D VMS adjoint contract. 

5. **The core OSGS architecture is largely aligned with the current `osgs_algorithm.tex`:** zero projection bootstrap, frozen projection in the inner cascade, re-projection, mass-weighted projection drift, and mode-dependent Picard line search are all conceptually present. 

---

# TRACEABLE EXPLANATION

## P-001 — `τ₂` omits the paper’s compressibility contribution

**Paper reference:** pressure-stability part of `article.tex`; VMS pressure stabilization and compressibility bound.  
**Code location:** `src/stabilization/tau.jl`, `compute_tau_2`.

- **Severity:** 🚨 CRITICAL INCONSISTENCY
- **Consistency status:** Checked & Failing
- **Confidence:** High

**Logic followed:**  
The article’s stability estimate explicitly contains the pressure penalty contribution `ε ||p_h||²` and the negative stabilized pressure term `- ε² ||τ₂^{1/2} p_h||²`, so `τ₂` must be bounded consistently with `ε`; the text later discusses choosing `ε > 0` conservatively for well-posed manufactured tests. 

The implementation computes:

```julia
return (med.h * med.h) / ((c_1 * med.alpha * τ_1_NS_val) + tau_reg_lim)
```

with no `eps_val * h^2` contribution. 

**Why this is inconsistent:**  
The paper’s pressure stabilization relies on ensuring the compressibility part does not become anti-stabilizing. Replacing the `ε h²` denominator contribution by an unrelated `tau_reg_lim` changes the scaling of pressure stabilization. In regimes with `ε > 0`, especially the documented 3D manufactured setting, this is not a harmless regularization: it changes the bound that controls the pressure term.

**Minimal correction:**

```julia
function compute_tau_2(kin, med, ν, c_1, c_2, tau_reg_lim, eps_val)
    mag_u = kin.mag_u
    τ_1_NS_val = 1.0 / ((c_1 * ν / (med.h * med.h)) + (c_2 * mag_u / med.h) + tau_reg_lim)
    return (med.h * med.h) / ((c_1 * med.alpha * τ_1_NS_val) + eps_val * med.h * med.h + tau_reg_lim)
end
```

Then thread `eps_val` through `Tau2Op`, `DTau2Op`, residual assembly, and all tests. The derivative `compute_dtau_2_du` must be updated consistently.

---

## P-002 — Silent `eps_val` flooring changes the PDE

**Paper reference:** incompressible/penalized mass equation treatment; manufactured 3D use of deliberate `ε > 0`.  
**Code location:** `src/formulations/continuous_problem.jl`, `PaperGeneralFormulation` constructor.

- **Severity:** 🚨 CRITICAL INCONSISTENCY
- **Consistency status:** Checked & Failing
- **Confidence:** High

**Logic followed:**  
The algorithm and article distinguish model equations and optional compressibility/penalty choices. In the 3D manufactured case, `ε > 0` is deliberately selected to make the problem well-posed and to remove pressure indeterminacy; it is not presented as something the formulation constructor may silently impose. 

The code constructor has a default `eps_floor=1e-8` and computes `safe_eps = max(eps_val, eps_floor)`. 

**Why this is inconsistent:**  
If the user requests `eps_val = 0`, the executable mass residual becomes `eps_floor * p + ∇·(αu) - g`, not the requested incompressible equation. That modifies the PDE, the pressure gauge, and the stabilization balance. It also violates the repository’s own no-hidden-defaults policy, which explicitly rejects silent fallback numerical values. 

**Minimal correction:**

```julia
function PaperGeneralFormulation(v::V, r::R, p_in::P, reg::Reg,
                                 ν::Float64, eps_val::Float64;
                                 autocorrect_policy=false) where {V,R,P,Reg}
    eps_val >= 0 || throw(ArgumentError("eps_val must be nonnegative."))
    valid_policy = sanitize_projection_policy(p_in, r; autocorrect=autocorrect_policy)
    new{V,R,typeof(valid_policy),Reg}(v, r, valid_policy, reg, ν, eps_val)
end
```

If an `eps_floor` is scientifically desired, it must be an explicit named configuration policy, not a constructor-side clamp.

---

## P-003 — OSGS inner-cascade stop reasons are not strict enough

**Paper / algorithm reference:** Algorithm A.3 and OSGS failure handling.  
**Code location:** `src/solvers/porous_solver.jl`, OSGS inner cascade result handling.

- **Severity:** 🚨 CRITICAL INCONSISTENCY
- **Consistency status:** Checked & Failing
- **Confidence:** Medium-High

**Logic followed:**  
`osgs_algorithm.tex` explicitly classifies `Xtol` as failure unless the residual has slipped below the noise floor, and `IterationCap` as budget-exhausted failure. It also warns that coercing unresolved iterates into convergence is forbidden because unresolved `U_h` produces stale projections that Stage II cannot reliably recover. 

The reviewed implementation logic treats only a narrower set of stop reasons as failure-like in OSGS inner solves: `linesearch_failed`, `merit_divergence_escaped`, and `linear_solve_nan`. The concerning omissions are `max_iters_stagnation` and `xtol_stagnation`.

**Why this is inconsistent:**  
OSGS is a fixed point on `(U_h, π_h)`. If an inner solve stops because of iteration cap or step stagnation while the residual is still above the noise floor, the next projection is computed from a non-root. That is exactly the failure mode the algorithm text says must not be accepted.

**Minimal correction:**

```julia
function is_structural_inner_failure(reason)
    return reason in (
        "linesearch_failed",
        "merit_divergence_escaped",
        "linear_solve_nan",
        "xtol_stagnation",
        "max_iters_stagnation",
    )
end

function is_inner_success(reason)
    return reason in (
        "ftol_reached",
        "initial_ftol",
        "stagnation_noise_floor_reached",
    )
end
```

Then require `is_inner_success(reason)` before re-projecting as a valid OSGS step.

---

## P-004 — Formal viscous adjoints are incomplete for the general 3D VMS contract

**Paper reference:** VMS stabilized term uses the formal adjoint `L*V_h` in the subgrid term.  
**Code location:** `src/formulations/viscous_operators.jl`, `adjoint_viscous_operator`.

- **Severity:** 🚨 CRITICAL INCONSISTENCY for general 3D canonical use; 📝 DOCUMENTED APPROXIMATION in restricted 2D/divergence-free limits
- **Consistency status:** Checked & Failing for 3D; documented approximation for some paths
- **Confidence:** High

**Logic followed:**  
The stabilized VMS form is built from a formal adjoint weighting `L*V_h` paired with `τ ΠR`. The article states the OSGS/VMS stabilized system in terms of `L*V_h` and the projection of the residual. 

The symmetric-gradient adjoint code explicitly sets:

```julia
div_eps_v = 0.5 * Δ(v) # + 0.5 * ∇(∇⋅v)
```

and the deviatoric adjoint uses `div_ViscProj_v = 0.5 * Δ(v)` with a comment acknowledging simplification. 

**Why this is inconsistent:**  
For the symmetric-gradient operator, omitting `0.5 ∇(∇·v)` is not the formal adjoint except in divergence-free or otherwise special test-function regimes. For the deviatoric operator, the omitted term is zero in 2D because `0.5 - 1/d = 0`, but in 3D the coefficient is `1/6`, so the formal adjoint is not represented. The code’s strong deviatoric operator does include dimension-aware behavior, but the adjoint does not. 

**Minimal correction:**  
Use the same Hessian contraction machinery already available for strong operators to implement exact adjoint divergence terms, at least for the canonical paper path:

```julia
function adjoint_viscous_operator(::DeviatoricSymmetricViscosity, v, α, ν)
    ViscProj_v = Operation(EvalDevSymOp())(∇(v))
    div_ViscProj_v = Operation(EvalDivDevSymOp())(Δ(v), ∇∇(v))
    return 2.0 * ν * (ViscProj_v ⋅ ∇(α)) + 2.0 * α * ν * div_ViscProj_v
end
```

If Gridap limitations force approximation, expose this as a named operator mode and forbid calling it “paper-faithful” in 3D.

---

## P-005 — The OSGS stopping short-circuit is inconsistent between documentation and implementation

**Paper / algorithm reference:** OSGS stopping mode section.  
**Code location:** `src/solvers/porous_solver.jl`, OSGS convergence predicate.

- **Severity:** ⚠️ UNDOCUMENTED DEVIATION / doc-code inconsistency
- **Consistency status:** Checked & Failing as documentation equivalence
- **Confidence:** High

**Logic followed:**  
`osgs_algorithm.tex` says that in all three stopping modes an additional raw `ℓ∞` short-circuit fires when DOF drift falls below `stag` for `m ≥ 2`. 

The reviewed code gates this short-circuit only under `osgs_stopping_mode == "state_drift"`.

**Why this matters:**  
The implementation is probably *safer* than the text, because allowing raw state drift to override `projection_drift` or `both` can return a non-fixed point in projection space. But the documentation claims equivalence that the code does not implement. Because the algorithm document is meant to be the executable contract, this must be resolved one way or the other.

**Minimal correction:**  
Prefer updating the document to match the safer code:

```text
The raw Linf short-circuit is allowed only in state_drift mode.
It is deliberately disabled in projection_drift and both modes because
the projection may still move when the primary state appears stagnant.
```

---

## P-006 — OSGS iteration budget is silently expanded

**Paper / algorithm reference:** Algorithm C parameter `N_OSGS`.  
**Code location:** `src/solvers/porous_solver.jl`, effective OSGS iteration calculation.

- **Severity:** 🐛 LIKELY BUG / configuration semantics violation
- **Consistency status:** Checked & Failing
- **Confidence:** Medium

**Logic followed:**  
Algorithm C treats `N_OSGS` as a declared solver parameter.  The implementation computes an effective OSGS iteration count by taking a maximum involving `sol_cfg.newton_iterations + 5`, and may further expand it under MMS extension logic.

**Why this is dangerous:**  
A configured maximum iteration count is a numerical policy. Silently increasing it changes runtime, failure behavior, and convergence interpretation. It can also make an invalid configuration appear to succeed by spending more iterations than the user requested.

**Minimal correction:**  
Either use exactly `stab_cfg.osgs_iterations`, or introduce a separately named config field:

```json
"allow_osgs_budget_extension_for_mms": true,
"osgs_mms_extra_cycles": 5
```

and document that this is an MMS extension, not the core Algorithm C budget.

---

## P-007 — MMS plateau exhaustion can be conflated with solver success

**Paper / algorithm reference:** Algorithm D, MMS plateau verifier.  
**Code location:** `src/solvers/porous_solver.jl`, MMS-enabled OSGS completion path.

- **Severity:** 🐛 LIKELY BUG
- **Consistency status:** Needs runtime confirmation, but high-risk
- **Confidence:** Medium

**Logic followed:**  
Algorithm D declares plateau only after all monitored error quantities satisfy the relative-change bound for `N_plat` consecutive passes. The text explicitly says the counter resets to zero on failure. 

The implementation appears to return base solver success even when the MMS plateau budget is exhausted, while tagging the reason separately as budget exhaustion.

**Why this matters:**  
It is acceptable for the nonlinear solver to succeed while the MMS verifier fails. It is not acceptable for a verification run to report “success” without making the failure distinction impossible to miss. This is exactly the historical regression class flagged in the lessons: weakening plateau criteria can mask formulation regressions.

**Minimal correction:**  
Split the return contract:

```julia
solver_success::Bool
mms_plateau_success::Bool
overall_verification_success = solver_success && (!mms_enabled || mms_plateau_success)
```

Do not overload one `success` flag.

---

## P-008 — `τ₁` uses an `α / τ_NS + σ` simplification without explicit traceability to the paper’s `C_α`-dependent expression

**Paper reference:** stabilization parameter robustness and dependence on porosity variation.  
**Code location:** `src/stabilization/tau.jl`, `compute_tau_1`.

- **Severity:** ⚠️ UNDOCUMENTED DEVIATION
- **Consistency status:** Needs derivation
- **Confidence:** Medium

**Logic followed:**  
The implementation computes:

```julia
τ1 = 1 / (α / τ_NS + σ + tau_reg_lim)
```

with no explicit `C_α` or porosity-gradient correction.  The article’s robustness discussion treats porosity and parameter scaling carefully, including terms controlled by `α` and the physical parameter regimes. 

**Why this matters:**  
This may be a valid simplified stabilization choice under well-resolved porosity variation, but it is not neutral. If the derivation assumes neglected porosity-gradient constants, that assumption must be recorded as a paper-code divergence and covered by MMS tests with variable `α`.

**Minimal correction:**  
Add a traceability entry documenting the simplification and a variable-porosity `τ₁` sensitivity test.

---

## P-009 — Pressure projection space is unconstrained in all-Dirichlet regimes

**Paper reference:** pressure space / gauge behavior.  
**Code location:** OSGS pressure projection using `Q_free`.

- **Severity:** ⚠️ UNDOCUMENTED / DEFERRED DEVIATION
- **Consistency status:** Documented as deferred, not a current correctness failure
- **Confidence:** High

**Logic followed:**  
The project’s own critical-analysis notes state that in all-Dirichlet velocity regimes the continuous pressure space is `L²_0`, but `Q_free` is unconstrained and the pressure projection may carry a constant mode. The same note correctly observes that this is not necessarily wrong for the stabilization gradient but pollutes projection drift. 

**Why this matters:**  
This is not as severe as the `τ₂` or `eps` issues because the constant mode is mostly inert in gradients. But it can degrade convergence diagnostics and pressure-rate interpretation.

**Minimal correction:**  
Leave the current behavior only if the divergence remains explicitly documented. If fixed, make mean-removal conditional on boundary-condition regime, not unconditional.

---

## P-010 — Projection policy trims are broader than the article’s explicit constant-σ trim

**Paper / algorithm reference:** projection policy and constant-σ trim.  
**Code location:** `src/stabilization/projection.jl`.

- **Severity:** ⚠️ UNDOCUMENTED DEVIATION
- **Consistency status:** Needs derivation
- **Confidence:** Medium

**Logic followed:**  
The algorithm document justifies removing the reaction term from the projected residual only when `σ` is constant, because `σ u_h` lies in the FE velocity space and the projection/re-subtraction is algebraically redundant. For non-constant `σ`, the full residual must be projected. 

The implementation also includes `ProjectResidualWithoutMassPenalty`, which delegates pressure projection to a policy that removes the mass penalty contribution. 

**Why this matters:**  
Removing `εp` from pressure projection may be mathematically defensible by the same FE-subspace argument when `ε` is constant and `p_h ∈ Q_h`, but this is not the same as the documented constant-σ reaction trim. It needs its own derivation and projection-policy legality test.

**Minimal correction:**  
Document this as a separate policy:

```julia
ProjectMassResidualWithoutPressurePenaltyWhenConstantEpsilon
```

and test that it is equivalent to full projection up to projection noise for constant `ε`, while forbidden or carefully handled for variable penalties.

---

## P-011 — Configuration loading appears to violate the repository’s fail-fast doctrine

**Paper / algorithm reference:** configuration and parameter taxonomy.  
**Code location:** `src/config.jl`, `base_config.json`, schema handling.

- **Severity:** ⚠️ UNDOCUMENTED DEVIATION / reproducibility risk
- **Consistency status:** Checked conceptually; needs full config-line audit
- **Confidence:** Medium

**Logic followed:**  
The repository rules explicitly prohibit hidden numerical defaults, fallback parameter injection, and silently inventing values when the user omits them. 

The implementation architecture described in the README says `load_config_with_defaults(path)` parses JSON into typed configs, while also claiming “No silent defaults.”  A loader that deep-merges a user config with `base_config.json` is not automatically wrong, but it contradicts the phrase “missing numerical fields are configuration errors” unless every inherited value is explicitly reported and treated as an intentional template default.

**Why this matters:**  
If a user omits a solver tolerance or physical parameter and the loader silently imports it from `base_config.json`, the run is no longer fully specified by the user’s config.

**Minimal correction:**  
Either rename the behavior honestly as “template inheritance” and print the inherited numerical fields, or require complete configs for production runs.

---

# CHECKED-PASSING AREAS

These are important because they prevent false positives:

1. **The current OSGS high-level loop matches the updated algorithm structure.** The document specifies Stage I ASGS, Stage II OSGS, frozen `π_h` during inner solves, re-projection afterward, and mass-weighted projection drift. The reviewed code follows this architecture. 

2. **Picard line-search mode branching is mathematically sound in the algorithm.** The document correctly avoids using the Newton merit directional identity for Picard steps and instead uses monotone residual reduction. 

3. **The base residual/weak-form sign structure appears internally consistent.** The code builds base weak terms, strong residuals, adjoint weights, projection-adjusted residuals, and stabilization terms in a way that matches the stabilized VMS residual structure, subject to the specific issues above. 

4. **The 2D deviatoric strong viscous operator is not wrong merely because it drops `∇(∇·u)`.** For the deviatoric symmetric projection, the coefficient `0.5 - 1/d` is zero in 2D, and the code’s 2D branch reflects that. The 3D branch includes the `1/6` coefficient in the strong operator. 

---

# TRACEABILITY MATRIX ENTRIES TO ADD

Add these rows to `docs/paper_traceability.md`:

| Issue ID | Paper Ref | Code Location | Status | Severity | Confidence | Impact | Invariant | Test Category | Recommended Test | Remediation Status |
|---|---|---|---|---|---|---|---|---|---|---|
| P-001 | Pressure stabilization / `τ₂` stability bound | `src/stabilization/tau.jl:compute_tau_2` | Failing | CRITICAL INCONSISTENCY | High | Pressure stabilization scaled incorrectly when `ε > 0` | `τ₂` denominator includes compressibility contribution | Tau exactness | `test/blitz/tau_blitz_test.jl:test_tau2_epsilon_term` | Unresolved |
| P-002 | Mass equation / pressure penalty | `PaperGeneralFormulation` constructor | Failing | CRITICAL INCONSISTENCY | High | User-requested incompressible mass equation silently altered | `eps_val` is exactly user/config value | Config/formulation validation | `test/blitz/config_strictness_test.jl:test_no_eps_floor` | Unresolved |
| P-003 | Algorithm A.3 / OSGS failure propagation | `src/solvers/porous_solver.jl` | Failing | CRITICAL INCONSISTENCY | Medium-High | OSGS may re-project from unresolved inner iterate | Only `ftol`, `initial_ftol`, or noise-floor exits are acceptable | Solver stop-reason logic | `test/blitz/osgs_inner_failure_test.jl` | Unresolved |
| P-004 | VMS formal adjoint | `src/formulations/viscous_operators.jl` | Failing for 3D | CRITICAL INCONSISTENCY | High | Stabilization adjoint is not formal adjoint in 3D | `L*` matches weak-form operator under stated boundary assumptions | Adjoint consistency | `test/quick/viscous_operators_quick_test.jl:test_3d_deviatoric_adjoint` | Unresolved |
| P-005 | Algorithm C budget | `src/solvers/porous_solver.jl` | Failing | LIKELY BUG | Medium | User OSGS iteration budget silently changed | `N_OSGS` means configured maximum unless extension explicit | Config semantics | `test/blitz/config_budget_test.jl` | Unresolved |
| P-006 | Algorithm D plateau verifier | MMS OSGS completion path | Investigating | LIKELY BUG | Medium | Verification may report solver success despite plateau failure | MMS success requires plateau success | Verification logic | `test/extended/ManufacturedSolutions/mms_plateau_failure_test.jl` | Investigating |
| P-007 | Stabilization parameter simplification | `compute_tau_1` | Needs Derivation | UNDOCUMENTED DEVIATION | Medium | Variable-porosity scaling may drift from derivation | `τ₁` simplification assumptions are explicit | Tau/porosity sensitivity | `test/blitz/tau_variable_alpha_test.jl` | Investigating |
| P-008 | OSGS pressure gauge | `Q_free` pressure projection | Deferred | DOCUMENTED APPROXIMATION | High | Projection drift polluted by pressure constant mode | Projection space matches pressure gauge regime | Projection drift | `test/quick/osgs_pressure_mean_projection_test.jl` | Deferred |
| P-009 | Projection policy mass trim | `projection.jl` | Needs Derivation | UNDOCUMENTED DEVIATION | Medium | Pressure projection policy broader than documented reaction trim | Any projected-residual trim must be FE-subspace-exact | Projection policy legality | `test/blitz/projection_policy_legality_test.jl` | Investigating |

---

# FAST-TEST RECOMMENDATION BLOCK

## Verification anchor for P-001

- **Invariant to test:** `τ₂(ε)` decreases consistently with the paper’s pressure penalty contribution; at fixed `u,h,α`, increasing `ε` must affect `τ₂`.
- **Suggested test file:** `test/blitz/tau_blitz_test.jl`
- **Suggested test name:** `@testset "Tau2 includes epsilon pressure penalty [P-001]"`
- **Expected pass condition:** finite-difference and closed-form `τ₂` agree with denominator `(c₁ α τ_NS + ε h² + regularization)`.
- **Expected failure mode today:** `τ₂` is unchanged when `eps_val` changes.

## Verification anchor for P-002

- **Invariant to test:** constructing a formulation with `eps_val = 0.0` stores exactly `0.0`.
- **Suggested test file:** `test/blitz/config_strictness_test.jl`
- **Suggested test name:** `@testset "No silent epsilon floor [P-002]"`
- **Expected pass condition:** `form.eps_val == 0.0`.
- **Expected failure mode today:** `form.eps_val == 1e-8`.

## Verification anchor for P-003

- **Invariant to test:** OSGS inner cascade aborts or marks structural failure for `xtol_stagnation` and `max_iters_stagnation` unless residual is below `stagnation_noise_floor`.
- **Suggested test file:** `test/blitz/nonlinear_blitz_test.jl`
- **Suggested test name:** `@testset "OSGS rejects unresolved inner stop reasons [P-003]"`
- **Expected pass condition:** OSGS does not re-project after unresolved stop.
- **Expected failure mode today:** unresolved stop may proceed to projection.

## Verification anchor for P-004

- **Invariant to test:** for smooth 3D fields, the discrete adjoint identity holds:
  `∫ (L u)·v ≈ ∫ u·(L* v)` under the stated boundary simplifications.
- **Suggested test file:** `test/quick/viscous_operators_quick_test.jl`
- **Suggested test name:** `@testset "3D formal viscous adjoint invariant [P-004]"`
- **Expected pass condition:** adjoint mismatch decreases to quadrature tolerance for polynomial fields.
- **Expected failure mode today:** nonzero mismatch proportional to the missing `∇(∇·v)` term.

---

# PRIORITY ORDER

1. **Fix P-002 first** because silent `ε` flooring changes every mass residual and can contaminate interpretation of P-001.
2. **Fix P-001 next** because `τ₂` is analytically central to pressure stability and convergence.
3. **Fix P-003 before trusting OSGS convergence diagnostics** in stiff regimes.
4. **Fix or explicitly demote P-004** before claiming general 3D paper-faithful VMS behavior.
5. Then address configuration strictness, projection-policy documentation, and MMS success semantics.

My bottom-line assessment: the implementation is architecturally close to the intended ASGS/OSGS solver, and several sophisticated pieces are correctly aligned with `osgs_algorithm.tex`. The main correctness risks are not the high-level OSGS loop; they are hidden model alteration (`eps` floor), incomplete pressure-stabilization scaling (`τ₂`), permissive solver stop-reason propagation, and adjoint exactness in the general 3D VMS contract.