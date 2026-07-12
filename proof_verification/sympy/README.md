# Symbolic verification suite for `article.tex`

This folder holds **SymPy scripts that re-derive and check the symbolic claims of
the paper** *"A stabilized finite element method for incompressible, inertial
flows in inhomogeneous porous media"* (Casas, González-Usúa, Codina,
de-Pouplana). The goal is to harden the analytical results: every matrix
identity, eigenvalue, asymptotic limit and algebraic manipulation that a referee
would otherwise have to check by hand is verified here by computer algebra.

```
# one-off setup (sympy + numpy)
python3 -m venv /tmp/sympy_venv && /tmp/sympy_venv/bin/pip install sympy numpy
# run everything
/tmp/sympy_venv/bin/python "run_all.py"
```

Current status: **110/110 checks pass across 9 scripts.**

| Script | Paper location | What it verifies | Checks |
|---|---|---|---|
| `cdr_operator_verification.py` | §2 (`eq:matrices_stationary_strong_problem`, `eq:AdjointDifferentialOperator`, `eq:AdjointFlux`, `eq:neumann_bc`) | The CDR matrices reproduce the strong operator; `L*` is the formal adjoint; the **natural Neumann co-normal** `n·K∇u = 2αν Π̃∇u·n` (the A3 factor 2); and the full **Green's identity** on a box (verifies `D*_N` and `D_N`). | 6 |
| `fourier_tau_verification.py` | §4 / App. C (`eq:StabilizationParameters`, `eq:Tau1`, `eq:Tau2`) | The Fourier symbols of each operator piece, the viscous eigenvalues (`4/3` at `d=3`), the `√λ` design pair, and the assembly recovering `τ₁`, `τ₂`. | 19 |
| `stability_estimate_verification.py` | §5 (`eq:SigmaAlpha`, `eq:StabilityEstimateFinal`, `eq:UpperBoundOnEpsilon`) | The four forms of `σ̃_α`; the Young perfect square; the viscous & velocity coefficient expansions and their reduction to `Cν`, `Cσ̃_α`; the `ε`-smallness chain (`ετ₂ ≤ C₂`). | 9 |
| `robustness_asymptotics_verification.py` | §6 (`eq:GeneralAsymptoticBehaviourOfParameters`, regime blocks, `eq:DimensionlessMomentumEquation`) | The exact `τ₁,τ₂,σ̃_α` forms and every dominant-regime limit (incl. the A6 corrections); the nondimensionalization coefficients (incl. the A7 forcing scaling). | 14 |
| `manufactured_solution_verification.py` | §7 (`eq:ManufacturedProblem`, `eq:PlateauBumpFunction`, `eq:Gamma`, `eq:EpsilonRef`, literature example) | `∇·(αu)=0` (2D and the 3D z-extruded field); the boundary-trace point (A11); the plateau-bump `dγ/dη>0`, limits/monotonicity/`C^∞` joining; the `eq:EpsilonRef` chain and that `ε=10⁻⁴ε_ref` meets A1 with `C₂=10⁻²`; the literature DBF `a(α),b(α)≥0` and porosity profile. | 17 |
| `elemental_matrices_verification.py` | App. A Galerkin terms | Each Galerkin elemental component `T_(ai)(bj)=∂T/∂U_j^b` re-derived by symbolic differentiation and matched to the printed formula. | 19 |
| `elemental_bilinear_form_verification.py` | App. A stabilization LHS+RHS (`eq:StabilizationLVLU`, `eq:StabilizationLVF`) | **All ~60 stabilization matrix terms** (by family vs. the bilinear form) **plus the RHS vectors** `F_V` (`A_F…V_φ`) and `F_Q` (`Q_αF, Q_φ`). | 14 |
| `assembly_consistency_verification.py` | App. A assembly (`\mathbf{K}, \mathbf{K}_S, \mathbf{F}, \mathbf{F}_S`) | **Structural (bookkeeping)** cross-check the two scripts above cannot see: every matrix *named* in the assembled `K/K_S/F/F_S` must be *defined*, and every *defined* matrix must appear in the assembly. Parses the appendix directly. | 4 |
| `subscale_norm_verification.py` | §4 (`eq:BoundProjectionOfLBySubscales`, B9) | The operator-norm bound `|L̂û|²_Λ ≤ |L̂|²_Λ|û|²_{Λ⁻¹}` (Monte-Carlo, `n=3,4`), the B9 formula `|L̂|²_Λ=ρ_{Λ⁻¹}(L̂^†ΛL̂)`, and tightness at the maximizing eigenvector. | 8 |

---

## Audit: what was checked, what is left, what is out of scope

### Fully verified (symbolic / numeric, in the scripts above)
- **CDR operator form + boundary (§2).** `K_ij` reproduces the deviatoric-symmetric
  viscous operator with a *varying* `α`; the full `L_wU` reproduces the strong
  system; `L*` is the formal adjoint; the natural Neumann co-normal carries the
  factor 2 (A3); and the Green's identity on a box verifies the adjoint flux
  `D*_N` (`eq:AdjointFlux`) and the natural BC `D_N` (`eq:neumann_bc`) together.
- **Fourier design of `τ` (§4 / App. C).** All five symbols, the viscous symbol
  eigenvalues, the `τ_b` generalized-eigenproblem `√λ` pair, and the assembly
  `→ τ₁, τ₂` under `λ = h²/(|k₀|²τ₁,NS²)`.
- **Subscale-norm bound (§4).** The operator-norm inequality
  `eq:BoundProjectionOfLBySubscales` and the B9 formula, checked numerically.
- **Robustness asymptotics (§6).** The general `τ₁,τ₂,σ̃_α` forms and the
  dominant viscosity/convection/reaction limits — re-checks the A6.1–A6.4
  corrections; the full nondimensionalization, including the A7 forcing scaling.
- **Stability algebra (§5).** The `σ̃_α` identities, the Young inequality, the two
  coefficient expansions of `eq:StabilityEstimateFinal`, and the `ε` condition.
- **Manufactured solution + porosity (§7).** Exact divergence-freeness (2D and
  3D extruded), the boundary-trace correction (A11), the bump smoothness/
  monotonicity, the `eq:EpsilonRef` chain, and the literature DBF coefficients.
- **Elemental matrices (App. A) — complete (LHS + RHS).** A Gâteaux-derivative
  framework re-derives *all* Galerkin terms; the **whole stabilization set
  (≈60 matrix terms)** is verified family-by-family against `eq:StabilizationLVLU`
  (K_VU `A_•,L_•,C_•,G_β•,D_β•,R_•` + mass; K_QU `G_•` + mass; K_VP; K_QP); and
  the **RHS vectors** `F_V` and `F_Q` against `eq:StabilizationLVF`. This pass
  **found and fixed four transcription typos** in the appendix:
  - `D_U` (`eq:DULHSStabilizationTerm`) missing a factor `α`
    (`τ₂ ∂ᵢNᵃ ∂ⱼα Nᵇ` → `τ₂ α ∂ᵢNᵃ ∂ⱼα Nᵇ`);
  - `G_P` in K_VP (`eq:GPLHSStabilizationTerm`) was a duplicate of `D_P`
    (`τ₂ α ε ∂ᵢNᵃ Nᵇ` → `τ₂ ε Nᵃ Nᵇ ∂ᵢα`);
  - `G_βF` (`eq:GBetaFTerm`) — its second forcing index should be `f̄ᵢ`, not `f̄ₖ`;
  - `Q_φ` (`eq:QPhiTerm`) — sign should be `−τ₂ ε Nᵃ φ̄` (to match `eq:StabilizationLVF`'s
    own `−(ε/α)q`, consistent with the verified `Q_P`).
- **Assembly bookkeeping (App. A "Putting together the results") — complete.**
  The content checks above verify each matrix *in isolation* but never look at
  the assembled displays. `assembly_consistency_verification.py` closes that gap:
  it enforces that every matrix *named* in `K, K_S, F, F_S` is *defined* and
  every *defined* matrix is *used*. Running it against the July-2026 revision
  **found and fixed four assembly-display defects** — three *named-but-undefined*
  (`I`, a transient mass matrix, identically zero in this stationary paper; and
  the bare `G_β`, `D_β`, which exist only as two-subscript stabilization
  cross-terms, never as Galerkin blocks) and one *defined-but-unused* (`V_T`, the
  Neumann traction load, silently dropped from `F`). Since the Galerkin viscous
  term `2ν(α∇v,∇ˢu) − (2ν/3)(α∇·v,∇·u)` differentiates to exactly `G_S + D_νD`
  (no `∇α`; porosity gradients arise only in the stabilization), the bare
  `G_β, D_β` were mathematically spurious rather than mere typos. Full write-up:
  [`docs/part_i_erratum.md`](../../docs/part_i_erratum.md).

### Verifiable, not yet encoded (one remaining)
- **The Galerkin coercivity identity `eq:StabilityEstimate` itself** (the
  integration-by-parts that turns `B_S(a;U_h,U_h)` into the sum of squares plus
  cross terms) — doable with the periodic-cell / box-IBP technique already in
  `cdr_operator_verification.py`. Its algebraic *consequences* are already
  covered by `stability_estimate_verification.py`.

### Out of scope for symbolic verification (verified elsewhere or non-symbolic)
- **Convergence-rate tables (§7).** These are *numerical* results; they are
  exercised by the Method-of-Manufactured-Solutions harness in
  `test/extended/ManufacturedSolutions/`, not by computer algebra.
- **The inverse estimate `eq:InverseEstimateFiniteOrderNorm`, coercivity, inf–sup,
  trace and interpolation theorems.** Functional-analysis results (citations to
  Brenner–Scott / Codina); not algebraic identities.
- **The nonlinear solver orchestration** (Newton/Picard/homotopy) — algorithmic,
  covered by the test suite.

---

## Notes
- The scripts are self-contained and depend only on `sympy` (tested with 1.14);
  `subscale_norm_verification.py` additionally uses `numpy` for its Monte-Carlo check.
- They are deliberately written to mirror the paper's notation and cite the
  relevant `eq:` labels in comments, so a reader can line up each check with the
  manuscript.
- Equation numbers (e.g. `eq:ViscousCoefficientBound`, `eq:VelocityCoefficientBound`) refer to the unnumbered displays in
  §5; `eq:*` names are the LaTeX labels in `theory/paper/`.
