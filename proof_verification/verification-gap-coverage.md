# Covering what the formal machinery missed

**Written 2026-07-19, after the external review caught two hand-transcription defects
(findings F1 and F2) that the existing formal machinery did not.** This note diagnoses
*precisely why* they slipped through, and proposes a layered plan so no further errors of
that class survive. Layer 0 is already implemented (this session).

## What is already machine-checked

Three independent layers already cover most of the paper:

1. **Coq (`coq-formal/`)** — four abstract a-priori theorems (`abstract_stability`,
   `abstract_continuity`, `abstract_continterp`, `abstract_convergence`) proved from a
   trusted base of ~50 named hypotheses that transcribe the paper's assumptions. `Print
   Assumptions` on the headline theorems returns only the three stdlib axioms; 3 of the 4
   theorems carry non-vacuity witnesses (`NonVacuity*.v`). Map: `coq_coverage.tex`.
2. **SymPy (`sympy/`)** — **115/115** symbolic checks across 10 scripts: the CDR operator
   and its formal adjoint, the Fourier τ design, the σ̃_α / stability algebra, the §6
   robustness asymptotics, the manufactured solution, the ~60 elemental stabilization
   matrices + RHS, assembly bookkeeping, the subscale-norm bound, and (new, §Layer 0
   below) the β-factored strong-residual display and the eliminated-subscale sign. This
   pass has itself **already found and fixed 8 transcription typos** in the appendix.
3. **MMS numerics** (`test/extended/ManufacturedSolutions*`) — the convergence-rate
   tables; a factor-2 error in an *assembled* term would spoil variable-porosity
   convergence, which is how we know the **implementation** is correct.

The two variants are complementary: SymPy checks the *concrete algebra*; Coq checks the
*abstract a-priori chain* that the algebra feeds.

## Why F1 and F2 still slipped through — the coverage boundary

Both defects live in a precise blind spot **between** the checked objects.

The SymPy suite verifies the pipeline **(symbolic operator) → (assembled matrices) →
(implementation)**. It does *not* parse every equation the paper *prints*; where it can
reconstruct an object symbolically, it compares to the *assembled/implemented* form, not to
the printed intermediate display.

- **F2** (`eq:StabilizationLVLU`/`eq:StabilizationLVF`, missing factor 2 on `ν Π̃∇u·∇β`)
  lives in the **β-factored compact display** of the strong residual, which sits *between*
  the un-factored operator (checked by `cdr_operator_verification.py`, via `K_ij`) and the
  assembled matrices (checked by `elemental_bilinear_form_verification.py`). The
  β-factoring identity itself was never checked. Crucially, the assembled matrices absorb
  the factor 2 into their symmetrized index structure (`A_Gβ` carries the symmetric
  two-term `(∂_i N^b ∂_j α + ∂_m α ∂_m N^b δ_ij)`, `A_Dβ` the `2/3` deviatoric
  coefficient), so **the assembly stayed correct and the suite passed while the display was
  wrong.**
- **F1** (`eq:weak_form_eliminated_subscales`, a minus where the split forces a plus) is a
  **motivational** identity, off the computational path the suite walks — the suite starts
  at the CDR operator and the final method, skipping the scale-elimination motivation.

The Coq layer cannot catch either: it is *abstract*, reasoning from hypotheses about an
abstract bilinear form; it never sees concrete algebra, and it does not verify that the
paper's concrete objects satisfy its hypotheses.

So the gap is **not** "no symbolic verification" (there is a lot) but **coverage
boundaries**: printed displays reconstructed-but-not-parsed, motivational equations off the
path, and the missing abstract↔concrete bridge.

## Proposed layers (prioritized)

### Layer 0 — close the two specific gaps  ✅ DONE (2026-07-19)
Added `sympy/display_consistency_verification.py` (5 checks, suite now 115/115):
- verifies the β-factoring identity `(1/α)(−2 div(ανΠ̃∇u)) = −νΔ̄u − 2ν Π̃∇u·∇β`
  component-by-component (would have caught **F2**; certifies the applied `2ν` fix), and
  asserts the pre-fix coefficient `ν` *fails* it (the check is discriminating);
- verifies the **plus** sign of the eliminated fine-scale term on an exact finite-dimensional
  static-condensation analogue of the VMS split (would have caught **F1**; certifies the
  applied `+` fix), and asserts the printed **minus** is not satisfied.

### Layer 1 — a "every displayed equation is checked" sweep  (recommended; cheap)
Generalize Layer 0: for **every** `eq:` the paper *displays*, either the suite reconstructs
it symbolically **and compares to the printed tokens**, or it is listed as
*displayed-but-unchecked*. Highest-value targets (each a known fragility home):
- `eq:AdjointDifferentialOperator` / `eq:AdjointFlux` — the sign home of the documented
  "Anti-SUPG" failure. `cdr_operator_verification.py` already builds the symbolic adjoint;
  add a printed-vs-symbolic token comparison of `𝓛*V`.
- `eq:Tau1`, `eq:Tau2`, `eq:TauNavierStokes` — `fourier_tau_verification.py` derives them;
  add a parse of the printed closed forms.
- the σ̃_α forms and the §6 estimate displays.
Deliverable: a short coverage report enumerating any *displayed-but-unchecked* equation, so
the boundary is explicit rather than implicit.

### Layer 2 — bridge the abstract Coq to the concrete instantiation  (high value)
The Coq proves "theorem holds **if** the ~50 hypotheses hold." Add a module proving the
paper's **concrete** objects satisfy the *algebraic* hypotheses:
- skew-symmetry (`H_skew_diag`) and integration-by-parts (`H_ibp_diag`) — provable
  symbolically with the box-IBP technique already in `cdr_operator_verification.py`;
- the weighted inverse estimate `S3` / coercivity threshold `c₁ > 2ξC̄_inv²` — the
  geometric constant is discharged numerically by the **already-validated**
  `test/extended/ManufacturedSolutions3D/element_c1.jl` (per-shape coercivity floor);
  wire its certified per-shape value into the check.
This upgrades non-vacuity ("hypotheses satisfiable on *some* data") to instantiation ("the
paper's *actual* method satisfies them, hence the theorem applies to it").

### Layer 3 — close the `abstract_continuity` non-vacuity witness gap  (small)
Provide the missing `NonVacuity` witness (checklist §8) so **all four** headline theorems
are proven on jointly-satisfiable, non-degenerate data — removing the "possibly vacuous"
caveat on the continuity theorem.

### Layer 4 — hypothesis-transcription audit  (ongoing; the one irreducible risk)
The ~50 Coq hypotheses are *hand-transcribed* from the paper (`coq_coverage.tex`, Table
`tab:inventory`). A mis-transcription — a hypothesis that says subtly more or less than the
paper assumes — is the one class **neither** SymPy **nor** Coq can catch (garbage-in). Add
an explicit "over/under-claim" column to the inventory and an adversarial second read
mapping each hypothesis to the exact paper assumption (line-referenced).

### Layer 5 — encode the one remaining "verifiable, not yet encoded" item
The Galerkin coercivity identity `eq:StabilityEstimate` (the box-IBP turning `B_S` into a
sum of squares plus cross terms) — flagged doable in `sympy/README.md` with the box-IBP
technique already present. Its algebraic *consequences* are already checked by
`stability_estimate_verification.py`; encode the identity itself.

### Scope honesty (not a gap to close, a boundary to state)
Enumerate in the paper what remains outside **any** machine check, so "machine-checked" is
correctly scoped: the OSGS theory, nonlinear discrete existence/uniqueness, the
velocity-dependent σ, and Neumann boundaries are all out of scope of both the Coq and the
SymPy layers.

## Suggested order
Layer 0 (done) → **Layer 1** (display sweep — cheapest, prevents the exact recurrence) →
Layer 3 (small) → Layer 5 (small) → **Layer 2** (bridge — highest structural value) →
Layer 4 (continuous discipline).
