# 2026-07-23 audit of the commented appendix — verified disposition

An external LLM audit of `article_v2.tex` + the live commented App D (`osgs_appendix_commented.tex`). It is
~90% a re-run of the 2026-07-22 audit (see `ChatGPT audit/latest_audit_response.md`), now on the commented
file. Every claim was verified against the live source with ≥2 independent angles; the verbose raw audit has
been distilled to this record. **Verdict: viable proofs, no fatal defect; the real items are small and local.**

## Applied this session (verified + built clean: v2 + v1 + note build with 0 undefined refs; SymPy 262/262)

- **Codina citation (new, high-value).** `Codina2008FiniteEA` was the wrong 2008 paper (hyperbolic wave);
  the OSGS machinery (mesh-size Lemma 1, smoothing Lemma 2, condition (35), Methods I/II, Oseen inf–sup) is
  `codina2008analysis`. Fixed in both appendices + `article_v2:897`. Left `:274` (intro Fourier cite — genuinely
  ambiguous, author judgment).
- **Three real proof gaps** (all invisible to Coq — trusted-base content; see
  `proof_verification/hypothesis_transcription_audit.md`): §2.1 added the `ω_χ(h)→0` smooth-grading hypothesis
  (`eq:SmoothGrading` on `H:patch`) and reworded the smoothing proof; §2.3 strengthened `H:advectionsmooth` to
  global `a∈W^{kᵤ,∞}(Ω)`; §4.2 replaced the additive→multiplicative patch leap with the contrast-free
  resolved-porosity chain (`δ_α²`).
- **Smaller confirmed items:** §3.4 dev-sym definiteness lemma added to App C (`eq:devsymidentity`), App D's
  conformal-Killing Korn re-pointed to it; §3.1 abs-value bars on the three ASGS continuity displays; §3.2
  mean-corrected pressure interpolant; §4.5 (`ε‖q‖²`, `c₁≥1`, the "once more" clause, the **I2–I5** Damköhler
  count); §4.7 numeric over-claim removed (`(1+Da_h/c₁)^{1/2}≈1.9`, not 3.5); A17 inf–sup sup domain;
  §6.1 fold language and §6.3 `1/α` claim tempered. Pedagogical boxes reconciled with the fixes; self-flagged
  "Audit note" markers resolved.

All formal edits mirrored across the live commented App D, the clean backing `osgs_appendix.tex`, and the
standalone note; commented/clean stay verbatim-identical in their math.

## Do NOT "fix" — verified false positives

- **§4.6** "restore `(1+c₁/Da_h)` in the main-text L² formula" — the display is correctly scoped to the
  reaction limit `Da_h→∞` (factor →1); the general form is `oa:cor:Ltwo`. Restoring it would be wrong.
- **§6.4** P₁ weighted viscous residual for variable α — already scoped to the unweighted operator, and the
  variable-α term is already written out.
- **§2.4** "`m≥2` is false for P₁/P₁" — `H:porositysmooth` already gates the extra smoothness to `m≥2`, and
  `C_{α,m}` already reduces to the audit's proposed `C_{α,1}` at `m=1`.
- **§10** "add `AinsworthOden`, `ScottZhang`" — both already in `references.bib`; the auditor's package (and
  `references(4).bib`, `§12`) were incomplete-download artifacts.

## Residual — author judgment (not applied)

Broader scope/prose tempering already tracked in `ChatGPT audit/latest_audit_response.md` §2–§3 (β₀/h₀
parameter-uniformity wording; abstract/conclusion robustness qualifiers; the `:274` Fourier cite). The
pedagogical boxes are temporary and will be dropped for submission (swap `\input` to the clean
`osgs_appendix.tex`) — the auditor's §8–§10 "two-document" recommendation is already the plan.

## Why the proof gaps escaped machine checking — and the new guards

Coq proves the *assembly* from ~50 abstract trusted-base hypotheses; it never unfolds their content or proves
the concrete FE objects satisfy them, so a hypothesis whose *stated proof doesn't establish it* is invisible
by design (it even certified the one-sided `abstract_continuity`). Two gates now close the reachable half:
`proof_verification/sympy/theorem_statement_verification.py` (statement-decoration lint — abs-bars, inf–sup
domains — in the `run_all.py` gate) and `proof_verification/hypothesis_transcription_audit.md` (the recurring
trusted-base read that catches the `ω_χ`/patch/advection class).
