---
name: porousns-clean-latex-compilation
description: Whenever you create or edit a LaTeX document, keep its build clean — every source .tex directory carries a latexmkrc that routes ALL auxiliary/intermediate files into a per-target 'latex compilation/<basename>/' subfolder (gitignored) and leaves ONLY the PDF next to the source. Build with latexmk so the rc applies; never commit aux files or scatter them in the source directory; never delete or weaken a directory's latexmkrc.
glob: "**/{*.tex,latexmkrc}"
---
# LaTeX builds stay clean: aux → 'latex compilation/%B/', PDF stays put

A LaTeX compile normally sprays a dozen intermediate files (`.aux`, `.bbl`, `.blg`, `.log`,
`.out`, `.toc`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, `.thm`, …) into the directory it runs
in. In this repo that directory is a **source** directory (`theory/*`, `proof_verification/`),
so an untamed build litters tracked sources and makes `git status` unreadable. The fix is
already in place across the repo (commit `d6bddaf`): **every source `.tex` document directory
carries a `latexmkrc`** that routes all intermediates into a per-target
`latex compilation/<basename>/` subfolder and keeps only the output PDF beside the source. Your
job is to **preserve that invariant** on every LaTeX edit — never regress it.

Treat "left aux files in a source dir" or "deleted a dir's latexmkrc" the same as any other
tree-dirtying regression.

## Canonical `latexmkrc` (drop this in every source `.tex` directory)

```perl
# latexmk configuration file for porous_NS_with_Gridap

# Route auxiliary files to a target-specific subdirectory (%B = job basename)
$aux_dir = 'latex compilation/%B';

# Keep the output PDF in the source directory
$out_dir = '.';

# Clean up intermediate files that latexmk knows about
$clean_ext = 'synctex.gz synctex\(busy\) thm bbl blg';

print "latexmk: Routing auxiliary files to 'latex compilation/%B/' and PDF to './'\n";
```

Reference copies: [`theory/paper/latexmkrc`](../../theory/paper/latexmkrc) is canonical. The
`theory/paper/` copy additionally prepends `./siam//` to `TEXINPUTS`/`BSTINPUTS` for the SIAM
class/bst — that per-document addition is fine; the three routing lines above are the part that
must be identical everywhere.

**`%B`-expansion caveat.** Some `latexmk` versions do **not** interpolate `%B` inside
`$aux_dir`, so `latex compilation/%B/` ends up literal. When that bites, compute the basename
explicitly from `@ARGV` instead of relying on `%B` — see
[`proof_verification/latexmkrc`](../../proof_verification/latexmkrc) for the exact, working
fallback. Same routing target, same `$out_dir`/`$clean_ext`; only the `$aux_dir` line changes.

## Do
- **Every source `.tex` document directory must contain a `latexmkrc`** with the routing block
  above. When you add a new LaTeX document in a new directory, add its `latexmkrc` **in the same
  change** — a `.tex` without a sibling `latexmkrc` is an incomplete addition.
- **Preserve the existing `latexmkrc`** when you edit a `.tex` in that directory. Do not delete,
  rename, empty, or "simplify" it away.
- **Build with `latexmk`** (which reads the rc), not bare `pdflatex`/`xelatex`/`lualatex` — a bare
  engine ignores `latexmkrc` and dumps aux files in place.
- **Rely on the root `.gitignore`.** It already ignores `latex\ compilation/`, `compilation/`, and
  every loose aux extension globally (`.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.toc`, `.fls`,
  `.fdb_latexmk`, `.synctex.gz`, `.synctex(busy)`, `.thm`, `.bcf`, `.run.xml`, …). Trust it; don't
  re-declare these per directory.

## Don't
- **Don't commit intermediate/aux files** or the `latex compilation/` folder. Only the `.tex`
  sources (and, where the repo tracks it, the built `.pdf`) belong in git.
- **Don't route the PDF into the aux subfolder.** `$out_dir = '.'` is deliberate — the PDF stays
  next to its source. Don't change it to `latex compilation/…` or a `build/` dir.
- **Don't invent a different layout** (`build/`, `out/`, in-place aux, a global aux dir). The one
  layout above is what the whole repo and `.gitignore` are wired for.
- **Don't add per-directory `.gitignore` stanzas** for aux files — the root `.gitignore` owns that.
- **Out of scope:** generated report dirs under gitignored `results/` (e.g.
  `test/extended/ManufacturedSolutions/results/.../latex_compilation/`) are build *output*, not
  source documents — this rule governs source `.tex` directories only.

## The tell — verify before you call a LaTeX change done
After building, `git status` in the document's directory shows **only** the intended `.tex` (and,
if tracked, `.pdf`) as changed — **no** loose `.aux`/`.bbl`/`.log`/`.synctex.gz`/… at the source-dir
top level, and **no** tracked `latex compilation/`. If stray aux files appear beside the source, the
`latexmkrc` is missing/ignored or you built with a bare engine instead of `latexmk` — fix that, don't
`git add`-and-ignore it.
