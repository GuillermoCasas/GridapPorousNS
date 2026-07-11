# Machine-checked verification of the paper's mathematical claims (Coq)

This directory contains a Coq formalisation of the algebraic and analytic
claims underlying *A stabilized finite element method for incompressible,
inertial flows in inhomogeneous porous media* (Casas, González-Usúa, Codina,
de-Pouplana). It is the proof-assistant counterpart of the SymPy suite in
`theory/verification scripts/`: where SymPy checks the identities by symbolic
computation, the files here prove them from the axioms of the real numbers,
and the proofs are re-verified by Coq's trusted kernel (`coqchk`).

**Status: 12 files, ~384 theorems, all proved. No `Admitted`, no `Axiom`,
no `admit` anywhere** (grep the sources to confirm). Developed and tested
against Coq 8.18.0 using **only the standard library** — no Mathcomp, no
Coquelicot, no external packages. Every definition has been checked against
the manuscript sources (`article.tex`, `continuity_appendix.tex`, July 2026
revision).

**Headline of this revision: `lemma:Stability` and `lemma:Continuity` are
now complete machine-checked theorems** (`abstract_stability`,
`abstract_continuity_sharp`, `abstract_continuity`), proved over an
abstract inner-product/mesh interface from a short, named, quantitative
trusted base of analytic facts — see the abstract layer section and the
scope ledger below. The elemental Cauchy–Schwarz inequalities are *derived*
from four inner-product axioms, not assumed; the discrete Cauchy–Schwarz
over element and face sums is proved by induction; every estimate
manipulation, coefficient collection, jump treatment and absorption of the
papers' proofs is inside the kernel-checked development. The residual
trusted base has a Lean 4 formalisation roadmap (`LEAN_ROADMAP.md`,
`PorousNSToolbox.lean`). A hand audit of the manuscript accompanies this
suite in `AUDIT.md`.

## Why Coq (and not Lean 4 + Mathlib)

Lean 4 with Mathlib would be the more comfortable vehicle for parts of this
material (in particular the Rayleigh-quotient spectral theory noted under
*Out of scope* below). It was not used for a practical reason: the sandbox in
which this formalisation was developed could not fetch the Mathlib build
cache, and building Mathlib from source was infeasible, so Lean proofs could
not have been machine-checked before delivery. Unverified formal proofs are
worth little. Coq's standard library (`Reals`, `micromega`) is self-contained,
installs from any distribution's package manager, and suffices for everything
in scope, so every theorem shipped here has actually been compiled and
kernel-checked. The trade-off is more manual epsilon–delta work (visible in
`PlateauBump.v`) and no off-the-shelf spectral theory.

## What "verified" means here — and what it does not

Provenance: an earlier revision of this suite was reconstructed from the
SymPy scripts alone, because the repository snapshot then excluded
`theory/paper/*.tex`. That caveat is now discharged: the current statements
have been diffed against the manuscript, and the definitions in
`StabilityAlgebra.v` and `ContinuityAlgebra.v` match eq:TauNavierStokes,
eq:Tau1Final, eq:Tau2Final, eq:SigmaAlpha, eq:UpperBoundOnEpsilon and the
appendix's eq:taus, eq:phi1, eq:sigmatilde, eq:epscond, eq:jumpcond verbatim.
One hygiene note: the SymPy script headers still cite the labels `eq:847` and
`eq:855`, which no longer exist in the tex (the corresponding displays after
eq:UpperBoundOnEpsilon are currently unlabelled); see AUDIT.md, finding F3.

A caveat that remains: Coq verifies the *mathematics as stated*, i.e. that each
stated identity/inequality/limit follows from the real-number axioms. It does
not verify that the statements correctly model the physics, nor that the
Julia/Kratos implementations match them — the latter is what the manufactured
-solution convergence tests in the codebase are for.

## File inventory and crosswalk

Dependency order: `Limits.v` and `DerivKit.v` are shared prerequisites for
the concrete files; `AbstractSums.v` and `InnerSpace.v` are prerequisites of
the two abstract files, which also import the closed algebraic theorems of
`StabilityAlgebra.v` and `ContinuityAlgebra.v` respectively and apply them
per element. All other files are mutually independent.

### `Limits.v` — one-sided limit definitions and generic limit lemmas

Defines `tendsto_at_top`, `tendsto_at_0plus`, `tendsto_at_1minus` (explicit
epsilon–delta, no filters) with transfer lemmas for functions that agree on
positives, and the generic rational limits used by `Asymptotics.v`:
`x/(C+x) -> 1`, `C/(C+x)` behaviour, `(A+x)/(B+x) -> 1`.

### `DerivKit.v` — derivative toolkit

`derivable_pt_lim` helpers: value rewriting (`dpl_val_eq`), extensionality
(`dpl_ext`), and derivatives of `a·t`, `sin(a t)`, `cos(a t)`, `c·f(t)`,
`e^{t-1}`. Nothing paper-specific.

### `StabilityAlgebra.v` — Section 5 stability estimate

Mirrors `stability_estimate_verification.py`.

| Coq statement | Paper | SymPy check |
|---|---|---|
| `sigt_form1/2/3` (σ̃ = σφ₁/(φ₁+σ) = σ − σ²τ₁ = σφ₁τ₁) | eq:SigmaAlpha | [1] |
| `young_identity`, `young_inequality` | Young step in §5 | [2] |
| `viscous_coefficient_expansion` | eq:847 = coefficient in eq:StabilityEstimateFinal | [3] |
| `velocity_coefficient_expansion`, `velocity_slack_identity`, `velocity_coefficient_lower_bound`, `u_final_lower_bound` | eq:855 ⇒ ≥ C·σ̃ | [4] |
| `eps_max_tau2_identity`, `tau1NS_minus_alphaK_tau1`, `alphaK_tau1_lt_tau1NS`, `eps_tau2_le_C2`, `pressure_term_coercive`, `epsilon_smallness_chain` | eq:UpperBoundOnEpsilon (amendment A1) | [5] |
| `viscous_coefficient_lower_bound`, `visc_final_lower_bound` (bracket ≥ C·ν with C = min{2−4C_inv²/c₁, 2(1−2/ξ)}) | unlabelled display after eq:UpperBoundOnEpsilon | — |
| `stability_constants_positive` (c₁ > 2ξC_inv², ξ > 2 ⟹ C_visc, C_u > 0) | eq:conditions_on_num_param | — |
| `elemental_coercivity` (the five coefficient bounds dominate C_stab × the triple-norm contributions, C_stab = min{C_visc, C_u, 1−C₂, 1}) | lemma:Stability, algebraic skeleton | — |

The last three complete the formal skeleton of lemma:Stability: what remains
outside Coq is only the functional-analytic layer (testing with U_h, the
inverse estimate eq:winv-divvisc, and the summation over elements), each of
which is hand-audited in AUDIT.md.

The slack identity is the sharpest statement: the gap between the eq:855
coefficient and `C_u·σ̃` equals `α_K τ₁ σ (ξ C_inv²/c₁)(c₂|a|/h) ≥ 0`
exactly, not merely non-negatively.

### `ContinuityAlgebra.v` — Appendix B continuity proof, algebraic core

New in this revision; no SymPy counterpart (the appendix postdates the script
suite). Formalises every purely algebraic step of app:Continuity, with the
appendix's generality (σ ≥ 0, α_K ≤ 1) and explicit constants wherever the
appendix writes a generic C.

| Coq statement | Appendix |
|---|---|
| `P1_sigma_tau1`, `P1_phi1_tau1`, `P1_tau1_le_inv_phi1` | lem:parameters (P1) |
| `P2_visc`, `P2_conv`, `P2_key`, `visc_le_phi1`, `conv_le_phi1` | (P2) |
| `P3_id`, `P3_sqrt` (φ₁^{1/2}h = c₁^{1/2}α_Kτ₂^{1/2}), `P3_sqrt_le` | (P3) |
| `P4_le_sigma`, `P4_le_phi1`, `P4_sigt_tau1`, `sigt_id_tau`, `sigt_id_sub`, `one_minus_sigma_tau1` | (P4), eq:sigmatilde, the 1−στ₁ = φ₁τ₁ workhorse of Steps 6b–6c |
| `P5_eps_tau2_phi/_C2/_lt1`, `P5_eps_h2`, `P5_sqrt`, `step7_scalar` | (P5), eq:epscond, Step 7's ετ₂^{1/2} ≤ C₂^{1/2}ε^{1/2} |
| `keyvisc_sq`, `keyvisc_sqrt` | eq:keyvisc (Step 4) |
| `T8_chain`, `T13_chain` | the σφ₁τ₁ = σ̃ absorptions of eq:T8bound and eq:T13conv |
| `volpart_scalar` (φ₁τ₁ ≤ φ₁^{1/2}τ₁^{1/2}), `sqrtphi_tau_le` (φ₁^{1/2}τ₁ ≤ τ₁^{1/2}), `phi_sqrttau_le`, `sqrtphi_sqrttau_le_1` | eq:volpart (Step 6b), Step 6c, Step 6d |
| `absorb1_core/_sqrt`, `absorb2_sqrt/_full`, `absorb4_vel/_full` | eq:absorb1–eq:absorb4 coefficient chains (Step 9) |
| `jump_formula`, `phi_diff_bound`, `DB_lower/_upper`, `tau_comp_AB/_BA`, `sg_comp_A_le_B`, `jump_bound_A` (σ\|[τ₁]\| ≤ (C_J/c_J)σ̃τ₁ with C_J = max{c_J′−1, 1−c_J}), `jumpsplit_AB`, `III_absorb_AB/_BA` | lem:jump, eq:jumpest, eq:jumpsplit, Step 6d absorption — all constants explicit |
| `delta_alpha_bound`, `winv_ratio` (α_∞/α₀^{1/2} ≤ δ^{1/2}α_∞^{1/2}) | H:porosity's derived bound; the lem:winv coefficient step |
| `lagrange2/3`, `CS2`, `CS3`, `step1_bound` | the finite Cauchy–Schwarz used in Steps 1 and 8 (eq:easystep) |
| `norm5_absorption` | Step 9's squaring-and-summing (eq:absorb1–eq:normconv), with the explicit aggregate constant |
| `tau2impl_le_tau2`, `tau2_le_scaled_impl` | analysis-τ₂ (eq:Tau2Final) vs implemented τ₂ (eq:Tau2): equivalent up to 1+C₂ under eq:epscond — supports AUDIT.md finding F6 |

### The abstract functional-analytic layer

Four files upgrade the two main lemmas from "algebra checked" to "theorem
checked". `AbstractSums.v` builds finite sums over a list of elements or
faces with linearity, monotonicity, the triangle inequality, and the
Cauchy–Schwarz inequality for sums proved by induction, plus subadditivity
of the square root. `InnerSpace.v` defines a minimal real pre-Hilbert
record (symmetric, left-bilinear, positive-semidefinite form) and *derives*
full bilinearity, Cauchy–Schwarz (`CS`, by the discriminant argument with
the degenerate direction handled), the norm expansions, the triangle
inequality, the parametrized Young inequality at vector level, and the
difference-of-squares identity `⟨−a+b, a+b⟩ = ‖b‖² − ‖a‖²` that drives the
ASGS cancellation.

`AbstractStability.v` then proves **lemma:Stability**: modelling each
field's elemental restriction as an abstract vector and the parameters via
the closed formulas, the theorem `abstract_stability` derives
`B_S(U_h,U_h) ≥ C_stab · |||U_h|||²` with the explicit
`C_stab = min{C_visc, C_u, 1−C₂, 1} > 0` from exactly two named analytic
hypotheses: (HBS) the Galerkin-testing/adjoint-expansion identity, and (S3)
the weighted inverse estimate eq:winv-divvisc — everything else (the
difference-of-squares step, the expansion of ‖G(u)‖², elemental
Cauchy–Schwarz, the Young step with parameter ξC̄²να_K/h² and the exact
cancellation of the inverse-estimate constant, the coefficient collection
into `visc_final`/`u_final`, the elemental coercivity, and the summation
over the mesh) is proved.

`AbstractContinuity.v` proves **lemma:Continuity** in full: the eighteen
terms of eq:Bstab are the definition of B_S; Steps 1–5 and 7 are the
fourteen per-term bound lemmas; Step 6a's five-term rewriting
(eq:fiveterms) is the exact identity `step6a`, consuming the three
integration-by-parts identities as hypotheses; Step 6b splits exactly into
the volumetric part (`bound_VolP`, via `φ₁τ₁ ≤ √φ₁√τ₁` and (P3)) and the
jump part (`bound_JmpP`), where lem:jump is applied through the four
face-resolved coefficient bounds `Jb11–Jb22` with explicit constants,
facewise Cauchy–Schwarz, and the bounded-multiplicity hypothesis; Step 6c
is `bound_II_V`; Step 6d is `bound_III` with the four `JD` bounds; Step 8
is `abstract_continuity_sharp` (eq:assembly, the sharp porosity-weighted
form); and Step 9's absorption chain eq:absorb1–eq:normconv is
`absorb_elem` + `step9`, yielding the final
`abstract_continuity : B_S(U,V) ≤ C_tot (BrU + BrP) |||V|||` with a fully
explicit constant. The trusted base is the named hypothesis list in the
file header (integration-by-parts identities, facewise assembly, the two
face-integral estimates, H:jump, bounded face multiplicity, the eight
weighted inverse estimates of lem:winv, and eq:epscond); the classification
and the Lean 4 attack plan for each provable item are in `LEAN_ROADMAP.md`.

### `TauDesign.v` — Section 4 / Appendix C Fourier design of τ

Mirrors `fourier_tau_verification.py`.

| Coq statement | Paper | SymPy check |
|---|---|---|
| `viscous_symbol_closed_form` (Σkᵢkⱼ·K_ij = αν(|k|²I + ⅓kkᵀ), d=3) | eq:matrices_stationary_strong_problem | [1] |
| `parallel_eigenpair` (4/3·αν|k|²), `transverse_eigenpair`, `parallel_factor_d3/d2` | viscous spectrum | [1] |
| `convective_symbol_*`, spectral radius (α/h)|w·k| | convective symbol | [2] |
| `coupling_eigen_pos/neg/kernel`, `coupling_det4`, `coupling_spectrum`, `coupling_spectral_radius` | pressure–velocity coupling C, spectrum {0, ±|k|} | [3] |
| `taub_pair_solves`, `taub_pair_unique` | X Y = c², X/Y = λ has the unique positive solution (c√λ, c/√λ) | [5] |
| `sqrt_lam_choice`, `taub1/taub2/tau_grada` inverses, `visc_plus_conv_momentum`, `tau1_inv_assembled`, `tau2_assembled` | eq:Tau1, eq:Tau2, eq:CAlpha, eq:TauNavierStokes | [6] |

### `Asymptotics.v` — Section 6 robustness asymptotics

Mirrors `robustness_asymptotics_verification.py`. General closed forms of
τ₁, τ₂, σ̃ in (Re, Da) (eq:GeneralAsymptoticBehaviourOfParameters, checks
A6.1); exact dominant-viscosity values and the Da→0⁺ limit of the σ̃ ratio
(A6.2); dominant-convection limits as Re→∞ for τ₁·α_K|a|/h, the τ₂ ratio and
the σ̃ ratio (A6.3, genuine `tendsto_at_top` statements); dominant-reaction
τ₁σ → 1 and Da-independence of τ₂ (A6.4); and the nondimensionalisation
coefficients Re, 1, (1+Re+Da), Da with unit forcing
(eq:DimensionlessMomentumEquation, eq:PressureScale, amendment A7).

### `ManufacturedSolution.v` — Section 7 manufactured solution

Mirrors `manufactured_solution_verification.py`, checks [1]–[5] except the
bump.

| Coq statement | Content |
|---|---|
| `manufactured_mass_conservation_2d/3d` | div(αu) = 0 for **arbitrary** differentiable positive α, since αu = Uα₀S with S solenoidal (eq:ManufacturedProblem). Stated via uniqueness of the derivative: whatever the partials are, they sum to zero. Genuine `derivable_pt_lim` derivatives, not symbolic ones. |
| `boundary_trace_nonzero` | u₂(0,0) = Uα₀/α(0,0) > 0 (amendment A11: the trace does not vanish). |
| `rhs_closed_form`, `denominator_gap`, `mid_le_rhs`, `eps_choice_margin` | the eq:EpsilonRef reporting chain: closed form of the eq:UpperBoundOnEpsilon bound at C₂ = 10⁻², the polynomial gap `99 + 99(c₂/c₁)Re + (9999/c₁)Da ≥ 0` dominating the mid expression, and ε := 10⁻⁴ε_ref sitting two orders inside the bound. |
| `dbf_a_nonneg`, `dbf_b_nonneg` | Darcy–Brinkman–Forchheimer literature coefficients admissible on 0 < α ≤ 1. |
| `alpha_lit_at_1`, `alpha_lit_deriv(_pos)`, `alpha_lit_at_0_range` | literature profile α(y) = 0.45 + 0.55e^{y−1}: α(1) = 1, α' > 0, 0 < α(0) < 1. |

### `PlateauBump.v` — Section 7 plateau bump (eq:PlateauBumpFunction, eq:Gamma)

The analytically substantial file. With γ(η) = (2η−1)/(η(1−η)) and
α_mid = 1 − (1−α₀)/(1+e^γ):

| Coq statement | Content |
|---|---|
| `gam_deriv`, `gam'_pos` | γ′ = (2η²−2η+1)/(η(1−η))² as a genuine derivative; positive on (0,1). |
| `q_deriv`, `am_deriv`, `am'_pos` | α_mid′ = (1−α₀)e^γγ′/(1+e^γ)² as a genuine derivative; the transition is strictly monotone. |
| `am_range` | α₀ < α_mid < 1, unconditionally in η. |
| `gam_upper_small`, `neg_gam_lower_small`, `exp_gam_small`, `exp_gam_cubic_small` (+ `_big` mirrors) | quantitative decay: e^γ ≤ 2η and e^γ ≤ 216η³ on (0,¼]; symmetric bounds in (1−η) on [¾,1). |
| `gam'_upper_small/_big` | γ′ ≤ 16/(9η²) resp. 16/(9(1−η)²). |
| `am_limit_0`, `am_limit_1` | α_mid → α₀ (η→0⁺) and → 1 (η→1⁻): the joins are continuous. Explicit δ = min(¼, ε/(2(1−α₀))). |
| `am'_limit_0`, `am'_limit_1` | α_mid′ → 0 at both ends: the joins are C¹. Explicit δ = min(¼, ε/(384(1−α₀))), from the bound α_mid′ ≤ 384(1−α₀)η resp. 384(1−α₀)(1−η). |

All limits are proved through explicit epsilon–delta arguments built on the
elementary inequalities t < e^t and t³/27 ≤ e^t (t ≥ 0); no logarithms, no
filter library. The paper's C^∞ claim rests on every derivative vanishing at
the joins by the same e^{−1/x}-type mechanism; the first-order case — which is
what the SymPy script also verifies symbolically — is the one formalised.
Extending to order n would follow the identical pattern (γ⁽ⁿ⁾ is a rational
function with denominator (η(1−η))^{n+1}, dominated by the cubic-in-fact-
any-power decay of e^γ) but is not included.

## Scope ledger

**Formalised here** (see crosswalk above): the §5 stability-estimate algebra
including the ε-smallness chain, the viscous/velocity coefficient lower
bounds and the elemental coercivity assembly of lemma:Stability; the
Appendix B parameter inequalities, jump lemma, step scalar cores and
Cauchy–Schwarz aggregation of the continuity proof; the §4/App. C Fourier
symbol analysis, coupling spectrum and τ assembly; the §6 asymptotics and
nondimensionalisation; the §7 manufactured solution, ε_ref chain, DBF
admissibility, and the plateau bump through C¹.

**Formalised in this revision** (the abstract layer): the complete proofs
of lemma:Stability and lemma:Continuity — including the summation over
elements, the elemental and discrete Cauchy–Schwarz inequalities (derived,
not assumed), the Young step, all eighteen term bounds, the exact
rewritings of Step 6, the jump treatment with explicit constants, the
facewise Cauchy–Schwarz with bounded multiplicity, and the Step-9
absorption — as theorems over a named trusted base.

**The residual trusted base** (each item a hypothesis of the abstract
theorems, stated quantitatively in the file headers): (a) *data/mesh
assumptions* — H:jump, eq:epscond, bounded face multiplicity, positivity of
the coefficients — which remain hypotheses in any formal system; (b)
*Green-type identities* — the Galerkin/adjoint identity (HBS), eq:skew,
eq:globalibp, eq:elemibp, the elemental pressure IBP, and the two facewise
assembly identities; (c) *inverse-type estimates* — eq:winv-divvisc (S3)
and the eight weighted inverse estimates of lem:winv, plus (inside the two
face-integral hypotheses) the L^∞ inverse estimate, meas(Γᵇ) ≤ Chᵈ⁻¹, and
Hölder on the face. Classes (b) and (c) are textbook material
(Brenner–Scott; Ern–Guermond) verified by hand against the manuscript
(AUDIT.md §4); their Lean 4 formalisation plan is `LEAN_ROADMAP.md` with
statement skeleton `PorousNSToolbox.lean`. The interpolation-theoretic
replacements of lem:continterp remain hand-audited only.

Two precision amendments to the appendix's hypotheses (AUDIT.md, F1–F2), plus
the hygiene/notation/strengthening items F3, F4, F6, F7, have been **applied**
to `theory/paper/` (see AUDIT.md's amendment-status banner and
`../coq_coverage.tex` §7).

**Deliberately left to the SymPy suite** — CAS-appropriate transcription
checks where a proof assistant adds cost but no assurance beyond what
symbolic expansion already gives:
`elemental_matrices_verification.py` and
`elemental_bilinear_form_verification.py` (Appendix A elemental matrices via
symbolic differentiation of shape functions).

**Out of scope for stdlib Coq** — would be natural next steps in Lean 4 +
Mathlib:
* the subscale-norm equivalence of `subscale_norm_verification.py` /
  Appendix B9, which needs Rayleigh-quotient spectral machinery
  (`Mathlib.Analysis.InnerProductSpace.Rayleigh`);
* the CDR adjoint/Green's-identity computation of
  `cdr_operator_verification.py`, which needs the divergence theorem.

Nothing in these two items is doubted by the SymPy checks; they are simply
beyond what is reasonable to hand-roll on the bare standard library.

## Building

Requires only Coq (8.16–8.20; developed against 8.18.0):

```
apt install coq          # or: opam install coq
```

Then either

```
./run_all.sh             # per-file PASS report + optional coqchk kernel re-check
```

or

```
make                     # compile everything
make check               # additionally re-verify with the trusted kernel (coqchk)
make clean
```

An editor with a Coq plugin (VsCoq, Proof General, CoqIDE) will pick up
`_CoqProject` automatically for interactive stepping.

## Conventions

* One file per SymPy script (plus the two shared prerequisite files); each
  file header states exactly which script checks it mirrors.
* Section variables carry the paper's positivity hypotheses (h, ν, α_K, c₁,
  c₂ > 0; Re, Da ≥ 0; 0 < α₀ < 1); every theorem is closed under its section.
* Inequalities are stated with explicit slack where the mathematics provides
  it (e.g. `velocity_slack_identity`), so the formal statement is strictly
  stronger than "≥ 0".
* `R`'s total division (x/0 = 0) is never relied upon: every division in a
  hypothesis-free statement (e.g. `am_range`) is harmless by construction,
  and all others carry non-vanishing hypotheses.
