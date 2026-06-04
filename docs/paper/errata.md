# `article.tex` — errata & open editorial items (author judgment)

Items found during the 2026-06-04 documentation audit that need an author decision rather than a
mechanical fix. The one unambiguous typo (the duplicate Fourier-operator label `eq:728`,
`\widehat{\mathcal{L}}_{\sigma}` → `\widehat{\mathcal{L}}_{\nabla\alpha}`) has already been corrected
in `article.tex`. Cross-references and citations otherwise resolve cleanly (verified by a full
`latexmk` cycle with `shared.tex` present: 0 undefined refs, all 41 `\cite` keys in `references.bib`).
The `\Guillermo{}`/`\Joaquin{}` review markup (16 + 3 occurrences) is intentionally preserved.

## Build status
The paper compiles cleanly (`latexmk` exit 0, 0 undefined references/citations, 43 pages): `shared.tex`,
the figure `figures/bump_plateau.pdf`, `elemental_matrices_appendix.tex`, and `references.bib` are all
present, and the SIAM class `siamart190516.cls` / `siamplain.bst` are local in `theory/`.

- **`supplement.tex` is the SIAM template boilerplate** — it `\input{ex_shared}` / `\externaldocument{ex_article}`
  (template defaults, absent) and contains example `\lipsum`/`thm:bigthm` content. It is optional for the
  article (used only via `\externaldocument{supplement}` for cross-refs). Either replace it with the real
  supplement or drop the `\externaldocument{supplement}` line if there is no supplement.

## Factual claim to confirm
- **"Kratos Multiphysics" implementation claim** ([line 1015](../../theory/paper/article.tex#L1015),
  `\Guillermo{All the implementations have been made within the Kratos Multiphysics software…}`). This
  repository is a **Gridap.jl** solver. Confirm whether the paper's numerical experiments were run in
  Kratos (historical) or should now read Gridap.jl. Do not silently flip — author call.

## Unfinished content
- **Results-section figures** ([line 1436](../../theory/paper/article.tex#L1436), `The results are presented in\Guillermo{Add figures}.`).
  The convergence results are currently presented as tables; figure environments are not yet added.
  (Convergence-plot PDFs were staged in a `figures_NEW/` folder during the audit and removed at the
  author's request, as they were unreferenced.)

## Pending structural merge (ready, but needs author placement)
- **Merge `centered_encoding.tex` into `article.tex`.** `centered_encoding.tex` self-describes as a
  "Working note (not yet merged into `article.tex`)". Safety checked: **no label clashes** between the
  two files. To merge: strip its standalone preamble (lines 1–25, through `\begin{document}\maketitle`)
  and the trailing `\end{document}`; **drop its `\Reyn`/`\Damk`/`\code` `\newcommand`s** (they would
  duplicate the article/shared definitions); inline the body near the verification / numerical-results
  section. Left for the authors to choose placement (and whether to condense rather than paste wholesale).

## Softer math points to verify (not corrected)
- Reaction matrix $\mathbf{S}(\boldsymbol{w})$ mass-equation row: confirm the porosity-gradient entries
  are consistent with $\nabla\cdot(\alpha\boldsymbol{u}) = \alpha\nabla\cdot\boldsymbol{u} + \nabla\alpha\cdot\boldsymbol{u}$.
- `cocquet_formulation.tex` uses the full symmetric gradient $\mathbf{S}(\boldsymbol{u})$, while
  `article.tex` uses the deviatoric-symmetric operator — confirm the `CocquetExperiment` Galerkin run
  matches the operator the paper claims to compare against.
