# Part I erratum — elemental-matrix assembly (`elemental_matrices_appendix.tex`)

**Status: RESOLVED (2026-07-12).** Fixes applied to the appendix; a durable
structural check now guards the class of defect. Canonical verifier:
[`proof_verification/sympy/assembly_consistency_verification.py`](../proof_verification/sympy/assembly_consistency_verification.py).

This note records an exposition/implementability bug found in the "Putting
together the results…" assembly of Appendix A (the elemental matrices) and the
gap in the proof-verification suite that let it through. It does **not** touch
any theorem: the defect lived only in the assembly *display*.

---

## 1. What was wrong — four defects in the assembled `K` and `F`

The block "Putting together the results…" assembled the elemental matrices as

```
K = [[I, 0],[0, 0]]
  + [[ G_S + D_νD + V + G_β + D_β + R_σ ,  P + G_P ],
     [ Q_D + G_αD                       ,  P_Q     ]]
  + K_S
F = [ V_F ; 0 ] + F_S
```

Four things are wrong with this display (verified against the appendix source;
each independently re-confirmed by an adversarial reviewer that tried to refute
it):

| # | Symbol | Class | Problem |
|---|--------|-------|---------|
| 1 | `I` | named-but-undefined | The transient **mass matrix** `I = α(θ₀/Δt)NᵃNᵇδ_ij`. Part I is the **stationary** paper (article.tex §2, l.218: *"we will consider only the stationary form of the equations"*; the strong momentum eq. `eq:StrongMomentumEquation` has no `∂ₜ`; the appendix has **zero** occurrences of `Δt`/`∂ₜ`/`θ`). With `M = 0` the block is identically zero, and `I` is defined nowhere. |
| 2 | `G_β` | named-but-undefined | The appendix defines only the **two-subscript** stabilization cross-terms `G_{βA}, G_{βL}, G_{βC}, G_{βG}, G_{βD}, G_{βσ}` (all in `K_S`). A bare, single-subscript `G_β` exists nowhere. |
| 3 | `D_β` | named-but-undefined | Same as #2 for `D_{βA…βσ}`; a bare `D_β` exists nowhere. |
| 4 | `V_T` | defined-but-unused | The Neumann traction load `V_{T,(ia)} = Nᵃδ_ik t_{N,k}` (`eq:VTerm`, from `eq:NeumannBCsTerm`) is defined and then **never enters `F`**. Anyone implementing from the paper would silently drop the Neumann load. |

## 2. Why #2/#3 are a *mathematics* error, not a typo

The Galerkin viscous term is

```
⟨∂ᵢV, K_ij ∂ⱼU⟩ = 2ν(α ∇v, ∇ˢu) − (2ν/3)(α ∇·v, ∇·u).      (eq:ViscousTerm)
```

The porosity `α` sits **inside** the bilinear form and **no integration by parts
is performed on it**, so the Gateaux derivative w.r.t. the nodal unknown `U_j^b`
yields **exactly** two element matrices and nothing else:

```
G_S  = αν(∂_l Nᵃ ∂_l Nᵇ δ_ij + ∂_j Nᵃ ∂_i Nᵇ)      (eq:GSComponents)
D_νD = −(2/3) αν ∂_i Nᵃ ∂_j Nᵇ                       (eq:DDComponents)
```

There is **no `∇α` (`∇β`) contribution** in the Galerkin block. Porosity-gradient
terms arise **only in the stabilization**, where one expands `∇·(αν Π^DΠ^S∇u)`
and pulls `α` out — which is exactly why they appear in `K_S` with **two**
subscripts. So `G_β, D_β` in the Galerkin block assert a Galerkin operator that
does not exist. (This is machine-checked: `elemental_matrices_verification.py`
re-derives the full Gateaux derivative of `eq:ViscousTerm` and gets precisely
`G_S + D_νD`.)

Provenance: all three stray objects (`I`, `G_β`, `D_β`) carry `θ₀/Δt` or
`θ̃₀/Δt` in the transient sibling. The stationary and transient appendices share
an ancestor; when `M` was set to zero the three transient matrices were dropped
from the *definitions* but survived in the *assembly display*.

## 3. The fix (applied)

```
K = [[ G_S + D_νD + V + R_σ ,  P + G_P ],
     [ Q_D + G_αD           ,  P_Q     ]]  + K_S      # I-block, G_β, D_β removed
F = [ V_F + V_T ; 0 ] + F_S                            # V_T restored (marked \amend)
```

Diff is confined to the assembly display in `elemental_matrices_appendix.tex`
(the `\mathbf{K}`/`\mathbf{F}` align). `latexmk` recompiles clean.

**Do NOT "fix" Part I's component definitions.** They are correct — nine were
re-derived symbolically (`C_L, C_σ, G_{βC}, G_{βσ}, D_{βL}, R_{σL}, R_{σC}, D_U,
G_L`) and `D_U` already carries an `\amend{\alpha}` (that one was caught here and
the fix simply never propagated to the display). The four bugs are display-only.

### Severity
It **does not invalidate anything** in Part I. Lemma 5.3, Lemma 5.5 and Theorem
5.6 never touch the assembled matrix; the ~60-term stabilization content and the
Galerkin components are all verified correct (`elemental_bilinear_form_verification.py`,
`elemental_matrices_verification.py`, 33 checks). This was an
implementability/exposition bug.

## 4. Why the proof-verification suite missed it — and the durable check

The sympy suite verified every matrix **in isolation**: each component formula
(`elemental_matrices_verification.py`) and each family-sum against the
bilinear form (`elemental_bilinear_form_verification.py`). **Neither ever looked
at the assembled `K/K_S/F/F_S` displays.** A matrix could therefore be perfectly
correct on its own yet be named-without-definition or defined-then-dropped in the
assembly — precisely these four defects.

**The durable check now adopted** (`assembly_consistency_verification.py`, wired
into `run_all.py`): parse the appendix and enforce the invariant

> every matrix **named** in the general assembly must be **defined**, and every
> matrix **defined** must appear in the assembly.

Run against the pre-fix revision it reported exactly `named-but-undefined = {I,
G_β, D_β}` and `defined-but-unused = {V_T}`; post-fix it is green (4/4). Suite
total: **110/110 across 9 scripts.**

## 5. Additional (pre-existing, cosmetic — not fixed here)

Surfaced by the audit; none is a correctness defect, all orthogonal to the four
above:

- **`G_P` name collision (benign).** The symbol `G_P` denotes two *different*
  matrices: the Galerkin flux block `−∂_iα NᵃNᵇ` (`eq:PComponents`, in the
  Galerkin `K_{V,P}` block) and the τ₂ stabilization block `τ₂ε NᵃNᵇ ∂_iα`
  (`eq:GPLHSStabilizationTerm`, in `K_S`). Each is used exactly once and placed
  correctly; the consistency check reports it as informational, not a failure.
- **F_V prose enumeration undercount (l.283):** the sentence lists six
  stabilization contributions (`A_F…R_{σF}`) but the align below also defines
  `D_φ`; `V_φ, Q_φ` are then added in the next sentence. All are defined and
  assembled — the prose sentence is just loose.
- **Subscript punctuation:** definitions write `G_{βF}` / `R_{σF}`, the `F_S`
  assembly writes `G_{β,F}` / `R_{σ,F}` — same matrices (the checker normalizes
  the comma away).
- **`D_βG` integrand display (l.183):** the *pre-derivative integrand* over-uses
  the index `l` (`∂_l Nᵃ δ_il … (∂_l N^c U_k^c + ∂_k N^c U_l^c) ∂_l β`), violating
  the summation convention. Its *printed result* `(4/3)τ₁ν² ∂_iNᵃ ∂_lα ∂_lNᵇ ∂_jα`
  is **machine-verified correct** (`elemental_bilinear_form_verification.py`,
  `Db_family`), so this is an intermediate-display index typo, not a wrong matrix
  — author's call, left untouched.
- **`\amend`-marked terms are verified, not open.** The five `\amend` markers in
  this appendix are `D_U` (l.197), `G_P`-stab (l.251), `G_βF` (l.289), `Q_φ` sign
  (l.298) — the four transcription typos the sympy suite already caught and
  confirmed — plus the `V_T` addition above (l.325). None is an unresolved
  correctness question.
