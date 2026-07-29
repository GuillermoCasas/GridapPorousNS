# Spec: covering Appendix A's *intermediate* derivations

**Status:** OPEN. Written 2026-07-29 for a future session to execute.
**Blind spot it closes:** BS-2 ("endpoint-only verification") of
[lessons_learned.md](lessons_learned.md), 2026-07-29 row.
**Prerequisite reading:** that row, plus `proof_verification/sympy/elemental_matrices_verification.py`.

---

## 1. The gap, precisely

`elemental_matrices_verification.py` and `elemental_bilinear_form_verification.py` certify the
elemental matrices of `theory/paper/elemental_matrices_appendix.tex` (App. A) by re-deriving them
**from their own sympy encoding**. Neither opens the appendix. `elemental_bilinear_form_verification.py`
transcribes only the printed *results* (the RHS of each display) and checks their family sums.

So the appendix's printed **path** — the ~54 pre-differentiation integrands sitting inside
`\frac{\partial}{\partial U_j^b}\left( \dots \right)` — is verified by nothing. On 2026-07-29 a
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

All nine are **now fixed**. Every printed *final result* was correct, so nothing downstream
changed — which is exactly why the endpoint checks stayed green.

**Partially closed already.** `elemental_matrices_verification.py` now carries a source lint:

* **S1** — every multi-index `\partial` subscript in App. A carries an explicit `^2` (65 inspected,
  0 violations). Catches the `:147` class decisively.
* **S2** — every symmetric-gradient pair uses single-index derivatives (15 pairs inspected).

Both ship with negative controls built from the pre-fix text. **The dummy-index-reuse class —
eight of the nine — is still uncovered.**

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
   an index legitimately summed inside each branch is counted twice. Correct handling requires
   recursively distributing products over nested sums — i.e. a small CAS, at which point approach
   §3 below is strictly better.

A third rule (**S3**: the argument of `∂/∂U_j^b` must contain a trial `U_·^c` and must **not**
contain `N^b` or a `\delta` carrying the free index `j`) is *conceptually* right — it is exactly
what `:160` violated — but a naive implementation produced **20 false positives**, almost all from
`\vphantom{\frac{\partial}{\partial U_j^b}}`, a spacing hack that looks like a real derivative.
S3 becomes viable once the parser of §3 exists (it must skip `\vphantom` anyway).

---

## 3. The approach that works, and what it costs

This is the method that *found* the nine defects, mechanised. For each display:

1. **Transcribe the printed intermediate** — the integrand inside `\frac{\partial}{\partial U_j^b}\left(\dots\right)` — into sympy, in the same index-explicit style the script already uses for
   the endpoints (`d(Na, i)` for `∂_i N^a`, `U[k][c]`, `sigma[i][j]`, `beta`, …).
2. **Transcribe the printed result** (the RHS after the final `=`).
3. **Differentiate** the intermediate with respect to `U_j^b` symbolically and compare to the
   printed result, for every `(i,j)` in `{1..d}²`, at `d = 3` (and `d = 2` where the display is
   dimension-independent).
4. **Assert equality.** A typo on the path makes step 3 fail even though the endpoint is right.

**The validation protocol is not optional.** Include as *control siblings* several displays known
to be correct as written — `:185 D_βA`, `:158 L_Gβ`, `:149 A_Dβ`, `:201 D_U` were used in the
original investigation and all four passed as written. A run in which the controls also fail means
the transcription convention is wrong, not the paper. This asymmetry is what distinguishes a real
defect from reader error, and it is what made the nine findings credible.

**Scope:** 54 differentiation arguments. Roughly a dozen distinct structural shapes (convective,
viscous Laplacian, viscous symmetric-gradient, `∇β` families, reaction, penalty), so most
transcriptions are copies of a sibling with indices changed.

**Effort:** on the order of a day for a session that already knows the notation. Front-load the
~12 shapes; the rest is mechanical.

**Where it goes:** extend `elemental_matrices_verification.py` (it already has the shape-function
machinery, `d()`, `Dpre()` and the `check()` harness) rather than adding a new script — the whole
point is that the file which certifies the endpoints should also read the path.

---

## 4. Acceptance criteria

* Every one of the ~54 intermediates transcribed and asserted; the count printed in the PASS line
  so the rule cannot go vacuous (`N intermediates checked`, `N > 0` enforced).
* At least four control siblings included and passing **as written**.
* Negative controls: re-inject two of the nine historical defects (`:147`'s `\partial_{ik}` and
  `:160`'s pre-differentiated factor) and assert the check fails on each. Without these the suite
  cannot prove the new rule discriminates.
* `run_all.py` still green; the grand total rises by the number of new checks.
* This file updated to **CLOSED**, and the BS-2 row in `lessons_learned.md` amended to say the
  dummy-reuse class is covered.

## 5. Do not

* Do not delete S1/S2 when the full check lands — they are cheap, orthogonal, and catch a
  convention violation the differentiation check would only catch indirectly.
* Do not "cover" the intermediates by asserting them against a *second* hand transcription of the
  same intermediate: that verifies the transcription, not the paper. The comparison must be
  against the printed **result**, which is independently corroborated by
  `elemental_bilinear_form_verification.py`'s family sums.
* Do not widen S1/S2 into a general index-balance lint. See §2.
