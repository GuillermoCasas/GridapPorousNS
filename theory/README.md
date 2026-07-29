# `theory/` — LaTeX sources

This directory holds **only the LaTeX sources** — the current state of the theory. All
meta-documentation (observations, to-dos, process notes, and the paper↔code references) lives under
[`../docs/`](../docs/); see [`../docs/README.md`](../docs/README.md).

## `paper/` — the SIAM article

Build from `theory/paper/` with its local `latexmkrc` (which adds `paper/siam/` to
`TEXINPUTS`/`BSTINPUTS`). Verified clean 2026-07-29: `latexmk` exit 0, 0 undefined
refs/citations, 0 multiply-defined labels — `article.tex` 76 pp / 768 labels,
`article_v2.tex` 111 pp / 964 labels (v2 carries the commented OSGS appendix).

- [`paper/article.tex`](paper/article.tex) — the paper; the authoritative theory anchor for the codebase.
- [`paper/article_v2.tex`](paper/article_v2.tex) — the submission-track revision (adds the OSGS linear theory, Appendix D).
- [`paper/asgs_convergence.tex`](paper/asgs_convergence.tex) (App. C), [`paper/fourier_appendix.tex`](paper/fourier_appendix.tex) (App. B) — **`\input` by BOTH articles**, so they may cite only labels defined in both; each carries a `[known-fragility]` header listing the guaranteed labels. Machine-guarded by `sympy/projector_algebra_verification.py` rule D3.
- [`paper/osgs_appendix.tex`](paper/osgs_appendix.tex) / [`paper/osgs_appendix_commented.tex`](paper/osgs_appendix_commented.tex) (App. D, v2 only) — the clean and the pedagogy-annotated copies; edits go to **both**.
- **The viscous projector is generic** (2026-07-29): the a priori theory is stated for any pointwise, constant-coefficient orthogonal projection of the velocity gradient that is Korn-compatible on `H¹₀` (standing assumption `H:projector`, display `eq:PiKorn`); the two required properties are numbered `(V1)` *Orthogonality* and `(V2)` *Korn compatibility*, and `{𝕀, sym, dev, dev∘sym}` are verified instances with the uniform constant `√2`, in **§2.1 `sec:ViscousProjector`** ("The viscous projector") — one main-text home for the definition, the assumed properties, the variants and the Korn proofs, duplicated verbatim in both mains because App. B and App. C cite its labels (relocated there from App. C on 2026-07-29). `\ViscProj` therefore denotes the *generic* operator; the deviatoric–symmetric default is `\DSPi`.
- [`paper/elemental_matrices_appendix.tex`](paper/elemental_matrices_appendix.tex), [`paper/shared.tex`](paper/shared.tex), [`paper/references.bib`](paper/references.bib) — `\input`/bibliography dependencies.
- `paper/figures/bump_plateau.pdf` — the one figure the article references.
- [`paper/siam/`](paper/siam/) — the SIAM class (`siamart190516.cls`) and bib style (`siamplain.bst`).
- Open editorial items for the paper: [`../docs/open-questions.md`](../docs/open-questions.md) §5.

The `\Guillermo{}` / `\Joaquin{}` author-review macros are defined in `article.tex` (lines ~101–102).

## `cocquet/` — Cocquet-et-al. material

- [`cocquet/cocquet_formulation.tex`](cocquet/cocquet_formulation.tex) — the exact (unstabilized Galerkin) Cocquet formulation.
- [`cocquet/cocquet_form_mms_manufactured_solution.tex`](cocquet/cocquet_form_mms_manufactured_solution.tex) — manufactured-solution sibling of the Cocquet form.
- `cocquet/Cocquet et al. - 2021 - Error analysis ... .pdf` — the source paper.

## Dedicated Notes

- [`osgs_algorithm/osgs_algorithm.tex`](osgs_algorithm/osgs_algorithm.tex) — OSGS algorithm derivation and pseudocode driving the solver.
- [`centered_encoding/centered_encoding.tex`](centered_encoding/centered_encoding.tex) — verification working note, **pending merge** into `paper/article.tex` (mechanics in [`../docs/open-questions.md`](../docs/open-questions.md) §5).
- [`osgs_reaction_note/osgs_reaction_note.tex`](osgs_reaction_note/osgs_reaction_note.tex) — OSGS reaction-dominated convergence analysis note.
- [`tau_saturation_note/tau_saturation_note.tex`](tau_saturation_note/tau_saturation_note.tex) — τ-saturation / stabilization-parameter analysis note.
- [`viscous_projector_note/`](viscous_projector_note/viscous_projector_note.tex) — **companion note (2026-07-29)**: the *sharp* characterization of the Korn compatibility `(V2)` — admissible ⟺ the ellipticity constant `c_Π` of `eq:cPi` is positive, i.e. the kernel contains no rank-one tensor — with the Plancherel/oscillating-profile proof, the admissible projector `𝕀−Π^DS` outside the paper's family, the inadmissible skew and spherical parts, and the tilted family `Π_θ` showing the `[1,2]` τ-window is not uniform. Removed from the article's §2.1 (which needs only the sufficient criterion) and kept here; it cites the paper, not the reverse.
- [`velocity_floor_regularization/velocity_floor_regularization.tex`](velocity_floor_regularization/velocity_floor_regularization.tex) — smooth velocity-floor regularization of the Forchheimer `|u|` term.
- [`continuity appendix/continuity_appendix.tex`](continuity%20appendix/continuity_appendix.tex) — the standalone continuity (**boundedness**) proof for the ASGS bilinear form `B_ASGS`; the source of Appendix C (`paper/asgs_convergence.tex`). Nothing to do with mass conservation.
- [`osgs_a_priori/osgs_convergence.tex`](osgs_a_priori/osgs_convergence.tex) — the standalone OSGS a priori note (working source behind Appendix D).
- [`numerical_constants/`](numerical_constants/) — the `c1` dimension note (element-aware coercivity floor).
- [`scale_free_gate_note/`](scale_free_gate_note/) — the scale-free convergence gate.
- [`projection_space_note/`](projection_space_note/) — the OSGS projection-space note.
- [`paper novelties/paper_novelties.tex`](paper%20novelties/paper_novelties.tex) — running record of what is new in the paper relative to the literature.
- [`pressure_recentering_note/pressure_recentering_note.tex`](pressure_recentering_note/pressure_recentering_note.tex) — pressure-mean drift under the iterated (Codina) penalty in the all-Dirichlet setting, and a **re-centering** hardening. **Implemented** behind the default-off config flag `recenter_pressure_between_penalty_passes` (A/B-verified behavior-preserving); *adoption-by-default* still pending. Provenance: the iterated penalty is Codina's, the re-centering is *this note's* proposal (not attributed to Codina).

## Where the meta-docs went

- Paper↔code map + divergence ledger → [`../docs/solver/`](../docs/solver/).
- Cocquet investigation (synthesis, analysis, transcripts) → [`../docs/cocquet/`](../docs/cocquet/).
- MMS status / convergence-2d / p2-3d → [`../docs/mms/`](../docs/mms/).
- Algorithm-improvement history, audit transcripts, refactor brief → [`../docs/solver/`](../docs/solver/).
