# `theory/` — LaTeX sources

This directory holds **only the LaTeX sources** — the current state of the theory. All
meta-documentation (observations, to-dos, process notes, and the paper↔code references) lives under
[`../docs/`](../docs/); see [`../docs/README.md`](../docs/README.md).

## `paper/` — the SIAM article

Build from `theory/paper/` with its local `latexmkrc` (which adds `paper/siam/` to
`TEXINPUTS`/`BSTINPUTS`). Verified clean: `latexmk` exit 0, 0 undefined refs/citations, 43 pp.

- [`paper/article.tex`](paper/article.tex) — the paper; the authoritative theory anchor for the codebase.
- [`paper/elemental_matrices_appendix.tex`](paper/elemental_matrices_appendix.tex), [`paper/shared.tex`](paper/shared.tex), [`paper/supplement.tex`](paper/supplement.tex), [`paper/references.bib`](paper/references.bib) — `\input`/bibliography dependencies.
- `paper/figures/bump_plateau.pdf` — the one figure the article references.
- [`paper/siam/`](paper/siam/) — the SIAM class (`siamart190516.cls`) and bib style (`siamplain.bst`).
- Open editorial items for the paper: [`../docs/paper/errata.md`](../docs/paper/errata.md).

The `\Guillermo{}` / `\Joaquin{}` author-review macros are defined in `article.tex` (lines ~101–102).

## `cocquet/` — Cocquet-et-al. material

- [`cocquet/cocquet_formulation.tex`](cocquet/cocquet_formulation.tex) — the exact (unstabilized Galerkin) Cocquet formulation.
- [`cocquet/cocquet_form_mms_manufactured_solution.tex`](cocquet/cocquet_form_mms_manufactured_solution.tex) — manufactured-solution sibling of the Cocquet form.
- `cocquet/Cocquet et al. - 2021 - Error analysis ... .pdf` — the source paper.

## Root

- [`osgs_algorithm.tex`](osgs_algorithm.tex) — OSGS algorithm derivation and pseudocode driving the solver.
- `centered_encoding.tex` — verification working note, **pending merge** into `paper/article.tex` (mechanics in [`../docs/paper/errata.md`](../docs/paper/errata.md)).

## Where the meta-docs went

- Paper↔code map + divergence ledger → [`../docs/solver/`](../docs/solver/).
- Cocquet investigation (synthesis, analysis, transcripts) → [`../docs/cocquet/`](../docs/cocquet/).
- MMS status / fold-recovery / next-actions → [`../docs/mms/`](../docs/mms/).
- Algorithm-improvement history, audit transcripts, refactor brief → [`../docs/solver/`](../docs/solver/).
