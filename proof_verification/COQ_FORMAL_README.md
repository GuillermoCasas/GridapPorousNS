# Machine-checked verification of the paper's mathematical claims (Coq)

This directory contains a Coq formalisation of the algebraic and analytic
claims underlying *A stabilized finite element method for incompressible,
inertial flows in inhomogeneous porous media* (Casas, González-Usúa, Codina,
de-Pouplana). It is the proof-assistant counterpart of the SymPy suite in
`theory/verification scripts/`: where SymPy checks the identities by symbolic
computation, the files here prove them from the axioms of the real numbers,
and the proofs are re-verified by Coq's trusted kernel (`coqchk`).

**Status: 17 files, ~789 theorems, all proved. No `Admitted`, no `Axiom`,
no `admit` anywhere** (grep the sources to confirm — and, more strongly,
`Print Assumptions`: every top-level theorem depends on exactly three axioms,
all from the standard library's *construction* of the reals
(`ClassicalDedekindReals.sig_not_dec`, `ClassicalDedekindReals.sig_forall_dec`,
`FunctionalExtensionality.functional_extensionality_dep`), and on nothing else).
Compiled and independently kernel-checked (`coqchk`) on Coq 8.18.0 using **only
the standard library** — no Mathcomp, no Coquelicot, no external packages. Every
definition has been checked against the manuscript sources (`article.tex`,
`continuity_appendix.tex`, July 2026 revision).

**This revision reduces the named trusted base from 53 items to 50.** The count
is the weakest part of the claim and is stated here first so it can be
discounted: what the two refactors below buy is *structural*, and in one case
the row that was removed carried no logical content at all. Exactly **three**
items that were *hypotheses* are now *theorems*, and **no** item is added:

- **`HBS`** -- the tested identity for the diagonal `B_S(U_h,U_h)`. It is proved
  from the two *diagonal* Green identities `H_skew_diag`/`H_ibp_diag` that it
  silently bundled, so `B_S(U_h,U_h)` is now *defined* as the eighteen-term form
  (`AbstractInterpolation.BS` at the diagonal atoms) and the two encodings of
  `B_S` are reconciled inside the kernel.

  Two honest qualifications, because the previous revision's label for this row
  (*"the single largest assumption in the development"*) will not survive
  inspection. First, **`HBS` was formally vacuous.** It read `Variable BS : R.`
  `Hypothesis HBS : BS = <five-term expression in free atoms>` -- the eliminable
  pattern `forall x, x = t -> P x`. A free real constrained only by an equation
  defining it assumes nothing: the old `abstract_stability` could be instantiated
  at `BS := t` with `eq_refl` and the hypothesis discharged for free. So deleting
  it removes no logical content from the theorem, and the 53 -> 50 count includes
  one phantom row. Second, of the four things `HBS` *informally* bundled, only
  the annihilation of `T2 + T4` was ever provable; the Galerkin form, the
  substitution `V_h := U_h` and `P^T P = P` did not vanish but *moved* into the
  eighteen-term reading obligation.

  What the refactor genuinely buys is therefore not a smaller list but a
  reconciliation: the bespoke five-term encoding of `B_S` is destroyed, and the
  number this file bounds from below is now *the same closed expression* the
  other three files bound from above, checked by the kernel rather than by a
  reader comparing two displays. The eighteen-term reading obligation it inherits
  is not new -- it already underpinned `BSWW` in `AbstractConvergence.v` -- so at
  the level of the union of all files, trust strictly decreases. Two consequences
  worth knowing: `AbstractStability.v` now imports `AbstractInterpolation.v`
  (hence the build order), and the atom `xu = alpha X(U_h)` was split into `cxu`,
  `gpu` with `xu` *defined* as their sum -- one cannot state "the convective part
  is skew" without naming it.
- **`IU_nonneg` / `IP_nonneg`** -- nonnegativity of the interpolation-error
  functionals, now derived from `HI_uu`/`HI_pp` plus `nrm_nonneg`: a norm is
  nonnegative and `CI > 0`, so `0 <= ||.|| <= CI * IU` forces `IU >= 0`. The
  price is strengthening `CI >= 0` to **`CI > 0`**, and `{CI > 0}` is logically
  *strictly stronger* than `{CI >= 0, IU >= 0, IP >= 0}` -- the old set does not
  imply the new one (take `CI = 0` with all atoms zero: every old hypothesis
  holds, `CI > 0` fails). So this is a trade (-2 rows, +1 strengthened), not a
  free reduction, and it is flagged as such in the coverage note.

  Two reasons the trade is nevertheless credibility-neutral. The *justification*
  burden does not grow: `IU_k = h_K^{k+1} |u|_{H^{k+1}(K)} >= 0` holds by
  definition of the seminorm, and `C_I > 0` holds for any interpolation estimate
  anyone would actually cite -- a Bramble--Hilbert constant of zero would assert
  exact interpolation. And the *strengthening is monotone in the model*: `CI`
  occurs only on the large side of upper-bound hypotheses (`HI_*`) and in the
  output constant `KPP_I = sqrt c1 * CI + K6b`, so any model of the old base
  survives at `CI + 1`, yielding the same theorem with a weaker constant. Nothing
  is excluded that the analysis wanted to admit.

Three further discharges are part of the development but **not** of this
revision; they belong to the previous one, which established the 53-item base,
and are recorded here only so the standing picture is complete:

- **`HBS_W`** -- the same tested identity for the discrete-error diagonal
  `B_S(W,W)`. It stopped being a *hypothesis* one revision ago, which replaced it
  with a `Theorem HBS_W` proved from the same two diagonal identities. What *this*
  revision adds is that even that theorem is retired: `BSWW` is definitionally
  `AbstractStability.BS` at the W atoms (`BSWW_is_ASBS`, by `reflexivity`), so
  `AbstractConvergence.v` simply *applies* `abstract_stability` there (`stab_W`),
  which proves the tested identity internally rather than demanding it. There is
  now no `Theorem HBS_W` in the development and no `Hypothesis HBS_W` either. The
  two diagonal identities the convergence file states are the same pair the
  stability file asks for, so they are *shared* rather than added on top.
- **`H_elem_conv_ibp` / `H_elem_p_ibp`** -- whose face terms were free variables
  and so constrained nothing; they are now *defined* as the integration-by-parts
  defect (`Lemma H_elem_conv_ibp` / `H_elem_p_ibp` in `AbstractContinuity.v` and
  `AbstractInterpolation.v`).

The strictness `sigma > 0`, `eps > 0`, `C2 > 0` is gone -- the development now
assumes exactly `H:data` (`sigma >= 0`, `eps >= 0`, `0 <= C2 < 1`), so the
**reaction-free case sigma = 0 is a genuine instance**. The numerical-parameter
condition is assumed in its *sharp* form `c1 > xi * Cbar^2`
(`stability_constants_positive_sharp`), with `C_stab_margin` recording what the
manuscript's two-fold stronger condition buys. And the analysed-vs-implemented
tau_2 bridge is wired end to end: `abstract_convergence_implemented` states
thm:convergence for the parameter the solver actually forms, with the constant
inflated by at most sqrt(1 + C2) < sqrt 2.

Per-theorem hypothesis counts (each equal to the number of `Hypothesis`
declarations in the file, so nothing is declared and left unused):
`abstract_stability` 16 -> **17** (-`HBS`, +`H_skew_diag`, +`H_ibp_diag`),
`abstract_continuity` **36** (type byte-identical to the previous revision),
`abstract_continterp` 41 -> **39**, `abstract_convergence` 46 -> **44**.
The conclusions of `abstract_continterp` and `abstract_convergence` are
unchanged; `abstract_stability`'s changed only in that `BS` is now the defined
eighteen-term expression rather than a free real, and the norm is expressed in
`cxu`,`gpu` rather than the combined `xu`.

Read that first count carefully, because it cuts against the headline:
**`abstract_stability`'s own analytic base went up, not down** -- it traded one
vacuous row for two genuine Green identities. The 53 -> 50 reduction is a
statement about the *union* over the four files, and it holds only because
`AbstractConvergence.v` already declared `H_skew_diag` and `H_ibp_diag` for its
own `BSWW` reasoning; the stability file now shares them rather than adding
them. Taken standalone, this file assumes strictly more than it did, and assumes
it honestly.

**Non-vacuity is machine-checked** for three of the four abstract theorems.
`NonVacuity.v` exhibits an explicit instance for `abstract_stability` (the
carrier `R` with `<x,y> := x*y`, a small mesh, rational data), discharges
*every* hypothesis — including `H_skew_diag`, `H_ibp_diag`, `S3` and `Heps` —
and applies the theorem inside the kernel, obtaining the non-degenerate
conclusion `B_S = 7/8 >= C_stab * |||U_h|||^2 = 7/16` with both sides strictly
positive. That witness is now *stronger* than it was: `B_S = 7/8` used to be a
number it *chose* (a free real, set by fiat, then justified against the assumed
`HBS`); it is now a number it *computes* from the eighteen-term definition. A
second witness in the same file (`am = 1`, a two-element mesh, `5069/567 >=
167/42`) adds a **strictly positive advection field**, and `NonVacuityInterp.v`
and `NonVacuityConv.v` do the same for `abstract_continterp` (39 hypotheses) and
`abstract_convergence` (44) — each on a genuine two-element interior-face mesh
(not `Empty_set`, not `Fl := []`), each with a concrete rational lower bound on
the right-hand side (`|B_S| = 72181/8400 <= 131 <= RHS` for interpolation;
`NErr <= C_conv * Psi` with `NErr, Psi > 0` for convergence), not merely
`RHS > 0`. So three of the four hypothesis bundles are provably *consistent*.

What remains, stated plainly. `abstract_continuity` has **no witness of its
own** — its consistency still rests on the hand argument. Each witness also has
disclosed non-sharp corners, recorded in the file banners: the advection witness
uses a uniform mesh and meets `Heps` at equality (the extremal admissible
choice, not a strict one); and the interpolation and convergence witnesses
discharge `H_skew`/`H_ibp_vp`/`H_skew_diag` as `0 = ±0`. That last is *forced*
(on a skew or antisymmetric diagonal both sides must vanish), but is met by
*nonzero cancelling* summands — it is exactly that cancellation which pins the
face data `FB_p`, `FB_c` the rest of the bundle then controls, so those
hypotheses are not idle. The previously-flagged gap around `CI_pos` — the one
hypothesis this revision made strictly stronger, and which lives only in the
interpolation and convergence bundles — is now **closed**: both files discharge
`0 < CI` strictly (`CI = 1`, `CI = 4`) and jointly with the full bundle. All of
this settles *consistency*, not *soundness*: that the finite element objects
satisfy the hypotheses is the other direction, and is what the hand audit in
`AUDIT.md` §4 establishes.

**Headline of this revision: the paper's entire a priori chain --
`lemma:Stability`, `lemma:Continuity`, `lem:continterp` and
`thm:convergence` -- consists of complete machine-checked theorems**
(`abstract_stability`, `abstract_continuity` with its sharp form
`abstract_continuity_sharp`, `abstract_continterp`, `abstract_convergence`),
proved over an
abstract inner-product/mesh interface from a short, named, quantitative
trusted base of analytic facts — see the abstract layer section and the
scope ledger below. The elemental Cauchy–Schwarz inequalities are *derived*
from four inner-product axioms, not assumed; the discrete Cauchy–Schwarz
over element and face sums is proved by induction; every estimate
manipulation, coefficient collection, jump treatment and absorption of the
papers' proofs is inside the kernel-checked development.

Two further files close the convergence analysis. `AbstractInterpolation.v`
proves **lem:continterp** by rerunning the continuity machinery with the
first argument equal to the interpolation error, exactly as the appendix
prescribes: the discrete inverse estimates on the first slot are replaced
by the interpolation estimates eq:interpdivvisc--eq:interpinftyE (seven
named hypotheses `HI_*`, one constant CI), the V-side stays discrete and
keeps its hypotheses verbatim, and the conclusions are
`abstract_continterp` (|B_S(E,V)| <= CtotI (PsiU + PsiP) |||V|||, the
sharp l2 form of psi(h)), its l1 corollary, and `interp_triple_norm`
(eq:Enorm). `AbstractConvergence.v` then proves **thm:convergence** as a
genuine composition *inside the kernel*: the closed `abstract_stability`
is applied to the discrete error W (its inverse-estimate input is
literally the V-side hypothesis `Hw_dv`, and its eps-condition is
definitionally the same eq:epscond; the cross-module agreement of the
tau1/tau2/sigma-tilde formulas is itself machine-checked --
`tau1_agree`/`tau2_agree`/`sigt_agree`), the closed `abstract_continterp`
is applied to the pair (E, W), the two are glued by the single genuinely
new trusted item `Horth` (Galerkin orthogonality eq:consistency plus
bilinearity of B_S in its first argument) -- the tested identity for the W-pair,
formerly the hypothesis `HBS_W`, is not needed at all, since
`abstract_stability` is applied at those atoms and carries it --
and the triangle inequality for the mesh-dependent triple
norm is **proved** from the pre-Hilbert axioms (per-element five-fold
Cauchy--Schwarz via the discrete C--S over an explicit five-element index
list, then the discrete C--S over the mesh; the error family's
components are the componentwise sums of the two families, which is
faithful because every component map is linear in U). Conclusion:
`abstract_convergence`, |||U - U_h||| <= (KabsI + CtotI/C_stab)
(PsiU + PsiP), with the l1 form of eq:convergence as a corollary; the
porosity-weighted sharp form of rem:sharperconv is what is proved.

The residual
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
`C_stab = min{C_visc, C_u, 1−C₂, 1} > 0` from exactly three named analytic
hypotheses: (H_skew_diag, H_ibp_diag) the two *diagonal* Green identities,
and (S3) the weighted inverse estimate eq:winv-divvisc. `B_S(U_h,U_h)` is
**not** among them: it is a `Definition` (the eighteen-term form at the
diagonal atoms) and the Galerkin-testing/adjoint-expansion identity is the
*theorem* `HBS`, proved from the first two — everything else (the
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

**The residual trusted base** — 50 hypotheses in total, enumerated row by row
in Table `tab:inventory` of `../coq_coverage.tex`, which is the authoritative
list (each item a hypothesis of the abstract theorems, stated quantitatively in
the file headers): (a) *data/mesh assumptions*, items 1–24 — H:jump, eq:epscond,
bounded face multiplicity, positivity of the coefficients — which remain
hypotheses in any formal system; (b) *Green-type identities*, items 25–32 —
eq:skew, eq:globalibp, their two *diagonal* instances `H_skew_diag`/`H_ibp_diag`,
the two facewise assembly identities, and `Horth`; (c) *inverse-type estimates*,
items 33–50 — eq:winv-divvisc (S3) and the eight weighted inverse estimates of
lem:winv, the seven `HI_*` interpolation estimates, plus (inside the two
face-integral hypotheses) the L^∞ inverse estimate, meas(Γᵇ) ≤ Chᵈ⁻¹, and
Hölder on the face. Classes (b) and (c) are textbook material
(Brenner–Scott; Ern–Guermond) verified by hand against the manuscript
(AUDIT.md §4); their Lean 4 formalisation plan is `LEAN_ROADMAP.md` with
statement skeleton `PorousNSToolbox.lean`. The interpolation estimates of
eq:interp/eq:interpinfty (Bramble--Hilbert technology) enter
lem:continterp as the named Class-(c) hypotheses `HI_*`, hand-audited
like the rest; with them, lem:continterp and thm:convergence are fully
formalised (`AbstractInterpolation.v`, `AbstractConvergence.v`), and the
only class-(b) addition of the convergence file is `Horth` (consistency +
bilinearity) — `HBS_W` is gone entirely, and the file's two diagonal identities
are the stability file's own, shared rather than added.

**Note what is *not* on that list any more, and what that is worth.** The
Galerkin/adjoint identity `HBS` used to head class (b), and this file used to
call it the single largest assumption in the development. That label was wrong:
as stated it was a free real pinned by its own defining equation, hence
eliminable, and it constrained nothing. It is now a theorem, proved from the two
diagonal Green identities, and its convergence twin `HBS_W` is gone outright.
The gain is that `B_S` has one encoding instead of two, not that a large
assumption was discharged. `Horth` (consistency + bilinearity) is now the
largest class-(b) item and the face bundle `H_face_p`/`H_face_c` the second —
and those two, together with the eighteen-term reading obligation that no row of
the table can express, are where the residual trust actually sits.

Two precision amendments to the appendix's hypotheses are proposed
(AUDIT.md, F1–F2).

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

The `Makefile` derives the inter-file dependency graph with `coqdep` (written to
a gitignored `.coq-deps`, regenerated automatically), so `make -jN` is
parallel-safe — the build order is no longer implied by the file listing.

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
