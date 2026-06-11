# Symbolic verification suite for `article.tex`

This folder holds **SymPy scripts that re-derive and check the symbolic claims of
the paper** *"A stabilized finite element method for incompressible, inertial
flows in inhomogeneous porous media"* (Casas, González-Usúa, Codina,
de-Pouplana). The goal is to harden the analytical results: every matrix
identity, eigenvalue, asymptotic limit and algebraic manipulation that a referee
would otherwise have to check by hand is verified here by computer algebra.

```
# one-off setup (sympy is the only dependency)
python3 -m venv /tmp/sympy_venv && /tmp/sympy_venv/bin/pip install sympy
# run everything
/tmp/sympy_venv/bin/python "run_all.py"
```

Current status: **87/87 checks pass across 7 scripts.**

| Script | Paper location | What it verifies | Checks |
|---|---|---|---|
| `cdr_operator_verification.py` | §2 (`eq:matrices_stationary_strong_problem`, `eq:AdjointDifferentialOperator`) | The CDR matrices `K_ij, A_c, A_f, S` reproduce the strong momentum+mass operator (incl. `-2∇·(ανΠ̃∇u)`, the `∇·(αu)` split and `εp`); `L*` is the formal adjoint of `L`. | 4 |
| `fourier_tau_verification.py` | §4 / App. C (`eq:StabilizationParameters`, `eq:Tau1`, `eq:Tau2`) | The Fourier symbols of each operator piece, the viscous eigenvalues (`4/3` at `d=3`), the `√λ` design pair, and the assembly recovering `τ₁`, `τ₂`. | 19 |
| `stability_estimate_verification.py` | §5 (`eq:SigmaAlpha`, `eq:StabilityEstimateFinal`, `eq:UpperBoundOnEpsilon`) | The four forms of `σ̃_α`; the Young perfect square; the viscous & velocity coefficient expansions and their reduction to `Cν`, `Cσ̃_α`; the `ε`-smallness chain (`ετ₂ ≤ C₂`). | 9 |
| `robustness_asymptotics_verification.py` | §6 (`eq:GeneralAsymptoticBehaviourOfParameters`, regime blocks, `eq:DimensionlessMomentumEquation`) | The exact `τ₁,τ₂,σ̃_α` forms and every dominant-regime limit (incl. the A6 corrections); the nondimensionalization coefficients (incl. the A7 forcing scaling). | 14 |
| `manufactured_solution_verification.py` | §7 (`eq:ManufacturedProblem`, `eq:PlateauBumpFunction`, `eq:Gamma`) | `∇·(αu)=0` for any porosity; the boundary-trace point (A11); the plateau-bump `dγ/dη>0`, the limits `α₀`/`1`, monotonicity and `C^∞` joining. | 10 |
| `elemental_matrices_verification.py` | App. A Galerkin terms | Each Galerkin elemental component `T_(ai)(bj)=∂T/∂U_j^b` re-derived by symbolic differentiation and matched to the printed formula. | 19 |
| `elemental_bilinear_form_verification.py` | App. A stabilization (`eq:AALHSStabilizationTerm` … `eq:GLHSStabilizationTerm`) | **All ~60 stabilization elemental terms**: each family of printed results is summed and checked against the corresponding piece of the bilinear form `eq:StabilizationLVLU`. | 12 |

---

## Audit: what was checked, what is left, what is out of scope

### Fully verified (symbolic, in the scripts above)
- **CDR operator form (§2).** `K_ij` reproduces the deviatoric-symmetric viscous
  operator with a *varying* `α`; the full `L_wU` reproduces the strong system;
  `L*` is the formal adjoint (periodic-cell integral).
- **Fourier design of `τ` (§4 / App. C).** All five symbols, the viscous symbol
  eigenvalues, the `τ_b` generalized-eigenproblem `√λ` pair, and the assembly
  `→ τ₁, τ₂` under `λ = h²/(|k₀|²τ₁,NS²)`.
- **Robustness asymptotics (§6).** The general `τ₁,τ₂,σ̃_α` forms and the
  dominant viscosity/convection/reaction limits — this directly re-checks the
  A6.1–A6.4 corrections (the `α_K` factors and `τ₁∼1/σ`). The full
  nondimensionalization, including the corrected forcing scaling (A7).
- **Stability algebra (§5).** The `σ̃_α` identities, the Young inequality, the two
  coefficient expansions of `eq:StabilityEstimateFinal`, and the `ε` condition.
- **Manufactured solution + porosity (§7).** Exact divergence-freeness, the
  boundary-trace correction (A11), and the smoothness/monotonicity of the bump.
- **Elemental matrices (App. A) — complete.** A Gâteaux-derivative framework
  re-derives *all* Galerkin terms (`G_S, D_νD, V, R_σ, Q_D, G_αD, P, G_P, P_Q`),
  and the **whole stabilization set (≈60 terms)** is verified family-by-family
  against the bilinear form `eq:StabilizationLVLU`: the K_VU families
  `A_•, L_•, C_•, G_β•, D_β•, R_•` and mass terms, the K_QU `G_•` + mass, the
  K_VP block, and K_QP. This pass **found and fixed two transcription typos** in
  the appendix:
  - `D_U` (`eq:DULHSStabilizationTerm`) was missing a factor `α`
    (`τ₂ ∂ᵢNᵃ ∂ⱼα Nᵇ` → `τ₂ α ∂ᵢNᵃ ∂ⱼα Nᵇ`);
  - `G_P` in K_VP (`eq:GPLHSStabilizationTerm`) was a duplicate of `D_P`
    (`τ₂ α ε ∂ᵢNᵃ Nᵇ` → `τ₂ ε Nᵃ Nᵇ ∂ᵢα`, integrand also missing `Nᵃ`).

### Verifiable, not yet encoded (good next steps)
- **The Galerkin stability identity `eq:StabilityEstimate` itself** (the
  integration-by-parts that turns `B_S(a;U_h,U_h)` into the sum of squares plus
  cross terms). This can be done with the same periodic-cell trick used for the
  adjoint in `cdr_operator_verification.py`.
- **`eq:BoundProjectionOfLBySubscales` / the `|·|_Λ` operator-norm bound** (§4):
  the generalized spectral-radius inequality can be checked numerically on random
  SPD `Λ` and random `k`.

### Out of scope for symbolic verification (verified elsewhere or non-symbolic)
- **Convergence-rate tables (§7).** These are *numerical* results; they are
  exercised by the Method-of-Manufactured-Solutions harness in
  `test/extended/ManufacturedSolutions/`, not by computer algebra.
- **The inverse estimate `eq:InverseEstimateFiniteOrderNorm`, coercivity, inf–sup,
  trace and interpolation theorems.** Functional-analysis results (citations to
  Brenner–Scott / Codina); not algebraic identities.
- **Physical/empirical modelling** — the Forchheimer coefficients
  `a(α), b(α)` of the literature example and the DBF reduction are definitions
  matched to their references, not derivations.
- **The nonlinear solver orchestration** (Newton/Picard/homotopy) — algorithmic,
  covered by the test suite.

---

## Notes
- The scripts are self-contained and depend only on `sympy` (tested with 1.14).
- They are deliberately written to mirror the paper's notation and cite the
  relevant `eq:` labels in comments, so a reader can line up each check with the
  manuscript.
- Equation numbers (e.g. `eq:847`, `eq:855`) refer to the unnumbered displays in
  §5; `eq:*` names are the LaTeX labels in `theory/paper/`.
