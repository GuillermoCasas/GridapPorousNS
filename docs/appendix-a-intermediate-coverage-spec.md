# Covering the paper's *intermediate* derivations

**Status:** Appendix A — **CLOSED 2026-07-29**. App. C's Step-6 η-optimisation — **CLOSED**
(§6 C2). The rest of the paper — **OPEN**, inventoried in §6.
**Blind spot it closes:** BS-2 ("endpoint-only verification") of
[lessons_learned.md](lessons_learned.md), 2026-07-29 row.
**What ships:** `proof_verification/sympy/latex_index_notation.py` (new) +
the "READING APPENDIX A's PRINTED DERIVATION" section of
`proof_verification/sympy/elemental_matrices_verification.py` (23 → 118 checks) +
section [6] of `stability_estimate_verification.py` (9 → 24). Suite: **498/498 → 608/608**.

---

## 1. The gap, precisely

`elemental_matrices_verification.py` and `elemental_bilinear_form_verification.py` certify the
elemental matrices of `theory/paper/elemental_matrices_appendix.tex` (App. A) by re-deriving them
**from their own sympy encoding**. Neither opened the appendix. `elemental_bilinear_form_verification.py`
transcribes only the printed *results* (the RHS of each display) and checks their family sums.

So the appendix's printed **path** — the pre-differentiation integrands sitting inside
`\frac{\partial}{\partial U_j^b}\left( \dots \right)` — was verified by nothing. On 2026-07-29 a
whole-paper read found **nine** defects there while the suite was green at 454/454:

| line | term | defect |
|---|---|---|
| 147 | `A_Gβ` | `\partial_{ik}` where the symmetric-gradient pair needs `\partial_k` |
| 150 | `A_σ` | trial factor reuses the test dummy `l` |
| 160 | `L_Dβ` | trial factor **already differentiated** (`\partial_m N^b \delta_{mj}`), so the argument is `U`-free and `∂/∂U_j^b` of it is literally 0 |
| 162 | `L_σ` | dummy reuse |
| 186 | `D_βL` | `\delta_{ik}` should be `\delta_{il}` (dangling `l`, `k` three times) |
| 189 | `D_βG` | trial `G_β` piece borrows the test dummy `l` |
| 194 | `R_σL` | dummy reuse on the viscous trial factor |
| 203 | `G_U` | trial `u·∇β` reuses `l` |
| 204 | `G_D` | trial divergence reuses `l` |

All nine were fixed in `fdc7507`. Every printed *final result* was correct, so nothing downstream
changed — which is exactly why the endpoint checks stayed green.

---

## 2. Two cheap approaches that were tried and REJECTED

Do not re-attempt these; the numbers below are measured, not guessed.

1. **Per-row index census.** Count each index letter over a whole display row; flag any letter
   that is neither an LHS free index nor paired. → **52 of 68 rows flagged.** A display row holds
   several additive terms and often both sides of an `=`, so the counts conflate independent
   expressions. No discriminating power.
2. **Per-summand census** (split the differentiation argument on top-level `+`). → **22 false
   positives on the already-repaired file.** Nested sums such as
   `\left( \partial_m N^c U_k^c + \partial_k N^c U_m^c \right)` are one "summand" at top level, so
   an index legitimately summed inside each branch is counted twice.

Both failures have the same root: **the census is only meaningful on a fully distributed
monomial.** The shipped gate distributes first (§3), which is why its false-positive rate on the
repaired appendix is zero.

---

## 3. What was built

**A parser, not a transcription.** The original spec proposed hand-transcribing each printed
intermediate into sympy. That was rejected during execution, for a reason worth recording: the
transcriber is the same reader who must notice the typo, and the dominant defect class — a trial
factor silently re-using a test dummy index — is exactly the kind a human eye repairs without
registering that it did. A transcription-based gate would have certified the transcription, not the
paper. `latex_index_notation.py` therefore reads the `.tex`: **both** the intermediate and the
printed result come from the source, so the check holds the paper to its own claim.

The grammar is small because App. A's token set is closed (33 macros in the component region):

```
expr    := term (('+'|'-') term)*
term    := factor+                                   (implicit product)
factor  := number | atom | '(' expr ')' | deriv | \frac{expr}{expr}
deriv   := \partial_x <operand> | \partial^2_{xy} <operand>
operand := '(' expr ')' | deriv | <factors up to and including the first N^., \alpha or \beta>
```

The operand rule is the only subtle part: it makes `\partial^2_{km} U_m^c N^c` (the derivative
reaching *past* the constant nodal coefficient to the shape function) parse the same way as
`\partial_m N^c U_k^c` (where it does not need to).

**Two independent arms**, either of which can fail alone:

* **P2 — index census.** Distribute every product over every (nested) sum, then require each
  monomial to obey the Einstein convention exactly: a declared free index occurs **once**, every
  other letter occurs **twice**. Three occurrences = a re-used dummy; one occurrence of an
  undeclared letter = a dangling index. A third rule — every monomial of a differentiation argument
  carries exactly one nodal trial coefficient — catches the `:160` class (an argument that is
  already differentiated, or has no unknown in it).
* **P4 — differentiation.** Differentiate the parsed integrand w.r.t. the nodal unknown
  (`N^c → N^b`, `U_k^c → δ_kj`, `P^c → 1`) and compare against the parsed printed result for every
  `(i,j)` in `{1..3}²`, with `β = log α` so the `α²` prefactors reduce correctly.

**Measured result:** 79 component displays extracted (67 carrying a `∂/∂U` or `∂/∂P` operator,
6 generic `T(...)` templates skipped), 147 expressions parsed, 214 monomials censused,
67 differentiation checks, 1 printed chain — **all pass as written**, in ~7 s. The whole file goes
23 → 118 checks; the suite 498 → 593.

**Where it went:** `elemental_matrices_verification.py`, per the original spec — the file that
certifies the endpoints now also reads the path. The parser lives in a sibling module
(`latex_index_notation.py`, deliberately *not* named `*_verification.py`, so `run_all.py`'s glob
does not execute it standalone).

---

## 4. Acceptance criteria — met

| criterion | outcome |
|---|---|
| every intermediate checked, count printed and **asserted** | P0 asserts ≥79 displays / ≥67 with a derivative / exactly 6 templates; a second check asserts the parse, census, differentiation and non-zero counters. A rule whose input silently empties is the failure mode this suite has been bitten by twice (2026-07-28 Rule 1, 2026-07-29 D3), so the counters are assertions, not decoration. |
| control siblings pass **as written** | Superseded by something stronger: *all 79* displays pass as written, including the four siblings (`D_βA`, `L_Gβ`, `A_Dβ`, `D_U`) the original investigation used as controls. A transcription gate needs controls to distinguish reader error from paper error; a parser that reproduces the whole appendix has already made that distinction 79 times. |
| negative controls: re-inject two historical defects | Superseded: **all nine** are re-injected **verbatim from the pre-fix source** (the `-` side of `fdc7507`'s hunks), and each is asserted to be rejected. Synthetic look-alikes would not have proved the class is closed. |
| — | **Added:** five **index-balanced** mutations (a convective index moved onto the test factor, a sign flip inside a symmetric-gradient pair, a repeated branch, a rewired second-derivative contraction, a dropped power of `α`). Each is asserted to pass the census and fail P4 — proving the differentiation arm does independent work rather than restating the census. |
| `run_all.py` still green | **593/593 across 23 scripts** (was 498/498). Both mains rebuilt first, so `document_hygiene`'s freshness rule reads live logs: article 78 pp / 776 labels / 0 undefined, article_v2 112 pp / 974 labels / 0 undefined. |
| S1/S2 kept | Kept. They are cheap and orthogonal; S1's convention is now *also* enforced structurally (the tokenizer rejects a multi-letter `\partial` subscript without `^2`), which is how the `:147` defect is caught. |

---

## 5. Do not

* Do not delete S1/S2 — they lint a textual convention the parser enforces only as a side effect.
* Do not "cover" an intermediate by asserting it against a *second* hand transcription of the same
  intermediate: that verifies the transcription, not the paper.
* Do not widen the census to work per-row or per-summand. See §2.
* Do not relax the P0 counters to "≥ 1". Their whole job is to fail if the extraction stops finding
  the appendix.

---

## 6. The rest of the paper: intermediates still uncovered

A six-agent survey (2026-07-29) inventoried every printed intermediate in App. A, App. B, App. C,
App. D and the main text, and mapped what each of the 23 verification scripts actually **reads**.
That map is the denominator, and it is stark:

> **17 of 23 scripts open no file at all** — they re-derive the mathematics from their own encoding.
> 5 open a `.tex`; 1 (`document_hygiene`) reads only build logs. Of the `.tex` that are opened,
> **App. A is the only one whose printed mathematics is parsed**; the others are opened for label
> and wording lints (`projector_algebra` D-rules, `theorem_statement`, `appendix_twins`) that never
> read a display's content.

So BS-2 is closed for App. A and **open everywhere else**. The clusters below are ranked by
(risk × count) / difficulty. Counts are uncovered items, not total displays.

### C1 — Main-text ↔ appendix duplicated *collections* (≈3 items, difficulty **low**, do this next)

`eq:CollectedCoercivity` (article_v2 l.1169–1180) is a main-text copy of App. C's `eq:coerccollect`;
`eq:OSGSCollected` (l.1305–1315) is a main-text copy of App. D's Step-4 collection; and l.1318–1322
asserts inline that each of the four coefficients is positive under the design conditions. **No gate
couples any copy to its original** — the same shape as BS-3, which `appendix_twins` closes for the
App. D twins and `projector_algebra` D12 closes for the duplicated §2.1.

*Gate:* extract the coefficient list from both copies and compare; then evaluate the four
coefficients symbolically under the design conditions and assert positivity (this second half is a
genuine equality/inequality check, not a diff). Cheap because the coefficients are short rational
expressions in `η, t, C_inv, c₁, c₂, β₀, ψ, γ`.

### C2 — App. C's Step-6 η-optimisation — **CLOSED 2026-07-29** (the rest of App. C still open)

Most of App. C's ~46 uncovered intermediates are *inequality* chains (Cauchy–Schwarz, Young,
absorption), where "differentiate and compare" does not apply. But one sub-chain is pure equality
algebra: `eq:coerccollect` → the η-optimisation (l.610–617) → `eq:coercoeff` → the equivalence
`c₁ > 2ξC̄_inv,α² for some ξ>2 ⟺ c₁ > 4C̄_inv,α²` (l.621–625). It is the chain the 3D `c₁` work
leans on (see [findings.md](findings.md) §3), so an error here would propagate into solver
decisions, not just prose.

**It is now covered** — section [6] of `stability_estimate_verification.py` (E0–E10b + 4 negative
controls; the file goes 9 → 24 checks). It reads the printed expressions out of
`asgs_convergence.tex` via a deliberately narrow `_tex2sym` converter that *raises* on any
construct it was not built for, so a rewritten display fails loudly instead of being mis-read.

**Two defects were found in the process, and both are the same shape — a check that looks like
coverage and is not:**

1. **Stale anchors = coverage illusion.** Sections [3]/[4] of that script were written against
   `eq:StabilityEstimateFinal`, `eq:ViscousCoefficientBound`, `eq:VelocityCoefficientBound`.
   **None of those labels is defined in any `.tex` in the repository** — App. C was rewritten from
   the ξ split to the η parameterisation, and the script never followed. Anyone grepping "is
   `eq:coerccollect` covered?" got a plausible-looking hit on a script anchored to a display that
   no longer exists. The identities encoded are still true, so nothing failed; they are kept, now
   labelled as the superseded ξ-form, and E0 asserts the *live* labels resolve.
2. **A tautological check.** Check [5]'s second assertion read
   `sp.simplify((1 - C2) - (1 - C2)) == 0` — i.e. `0 == 0`, printing `[PASS]` while certifying
   nothing (its own comment said "recorded for completeness"). Replaced by the real slack identity
   `ε(1−ετ₂) − (1−C₂)ε = ε(C₂ − ετ₂)`, plus a witness showing it goes **negative** when the
   hypothesis fails — so the check now depends on the hypothesis it is about.

*Measurement worth keeping:* a crude scan for `eq:`/`lem:` anchors named in the suite that resolve
in no `.tex` returns **16 candidates across 9 scripts**; after discarding regex artefacts
(trailing colons in prose, deliberate names for *unnumbered* displays such as
`eq:StabilityEstimate`) roughly 4 are real. A suite-wide **resolvability lint** in the D3'' idiom
would close the class, but it needs a convention for naming unnumbered displays first — otherwise
it becomes the kind of gate that cries wolf (§2).

*Still open in App. C:* the Step 0 term decomposition, Steps 1–9's bound chains, `lem:winv`'s
proof, and the interpolation chain — see C3.

### C3 — App. C's term-accounting for the 18-term decomposition (≈1 structural item, difficulty **low**)

`eq:T1`–`eq:T18` (l.699–724) decompose `B_ASGS` into 18 named terms, and Steps 1–9 then bound them
in groups. `continuity_grouping_verification.py` already does exactly this kind of accounting for
*one* collected display (it caught the `T13 = T13^c + R` double-count). Extending it to assert that
each of `T1…T18` is consumed **exactly once** across the Step 1–9 groups is the same technique
applied to the whole appendix, and it catches the highest-value structural error class (a term
bounded twice, or silently dropped).

### C4 — App. B's symbol chain (≈9 items, difficulty **medium**)

`fourier_tau_verification.py` certifies the τ endpoints from its own encoding; nothing reads
`fourier_appendix.tex`'s displays. The 2026-07-29 units slip lived here, and is now guarded by
`projector_algebra` D11′ — but as a *quantity lint on one sentence*, not as coverage of the chain.
Highest-risk remaining: the collapse of the viscous + convective momentum entries onto
`α τ_{1,NS}^{-1}` (l.146, parenthetical), and the announced sum `τ_K^{-1} = Σ(five contributions)`
at l.140/146 which omits `τ_{b,1}^{-1}`.

*Gate:* the App. A treatment does not transfer directly (matrix/vector notation, not index
notation), but the *shape* does — parse the printed symbol expressions and check each substitution
step, rather than only the final τ.

### C5 — App. D's OSGS inf-sup Step 2–4 chains (≈50 items, difficulty **high**)

The densest concentration of uncovered intermediates in the paper: the seven-term expansion
`oa:eq:T1`–`oa:eq:T7`, the Step-3 Young splits per term, the Step-4 collection with its seven
coefficients, and the numeric audit (`γ ≥ 10(1+M₂)`, `ψ ≤ min{1/8, β₀²/16}`). Same character as C2:
mostly inequalities, with the coefficient arithmetic of Step 4 mechanisable in isolation.
`appendix_twins` guards that the built and linted copies agree — it says nothing about content.

### C6 — Documents no script opens at all (difficulty **n/a**; a *different*, larger gap)

Worse than BS-2, because here the *endpoint* is unverified too:

* `theory/osgs_a_priori/osgs_convergence.tex` — the ungated near-twin of App. D. Memory records
  **10 of 14 proof bodies have drifted** and no gate couples them, unlike the
  `osgs_appendix`/`osgs_appendix_commented` pair.
* Every companion note except `viscous_projector_note` (whose build log is gated but whose content
  is not): `centered_encoding/`, `numerical_constants/`, `osgs_algorithm/`, `osgs_reaction_note/`,
  `pressure_recentering_note/`, `scale_free_gate_note/`, `tau_saturation_note/`,
  `velocity_floor_regularization/`, `theory/cocquet/`.
* `theory/paper/shared.tex`, `genuine3d_table.tex`, and the *content* of `coq_coverage.tex`.

The cheapest first move here is not a content gate but adding these documents to
`document_hygiene_verification.py`'s `DOCS` list, so at least their builds are health-checked.

### Not mechanisable

Chains whose middle member is a *choice* (which Young parameter, which term to absorb into which)
rather than an identity cannot be machine-checked into correctness — only their arithmetic can.
For those, the durable artefact is the periodic **hypothesis/transcription audit**
(`docs/ChatGPT audit/hypothesis_transcription_audit.md`), not another script. Say so explicitly
when scoping, rather than writing a gate that appears to cover them.
