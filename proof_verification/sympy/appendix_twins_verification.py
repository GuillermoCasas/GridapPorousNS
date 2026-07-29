#!/usr/bin/env python3
# =============================================================================
# appendix_twins_verification.py
#
# WHY THIS SCRIPT EXISTS.  Appendix D ships as two files that are supposed to be
# the same appendix:
#
#   theory/paper/osgs_appendix_commented.tex   -- the one article_v2.tex BUILDS,
#                                                 carrying \begin{pedagogy} boxes
#   theory/paper/osgs_appendix.tex             -- the clean submission copy, and
#                                                 the one theorem_statement_verification.py LINTS
#
# So the file that is rendered is not the file that is linted, and until
# 2026-07-29 nothing compared them.  They had drifted: 18 lines differed outside
# the pedagogy boxes (\bigl\lVert...\bigr\rVert in the clean copy vs \norm{...} in
# the built one -- a difference that RENDERS differently, not a whitespace nit).
# The drift was invisible to every other gate because each reads one file only.
#
# This is a third structural blind spot, distinct from the two already recorded:
#   * "vacuous rule"        - a rule that passes while inspecting zero items
#                             (lessons_learned 2026-07-28, and the D3/D9' rows)
#   * "endpoint-only"       - the mathematics is re-derived from the script's own
#                             encoding, so the paper's printed path is never read
#                             (elemental_matrices_verification.py's S1/S2 lint)
#   * "no pair consistency" - THIS one: every gate reads one file at a time, so a
#                             divergence between two files that must agree is
#                             structurally undetectable.
#
# WHAT IS CHECKED
#   T1  the two twins agree line for line outside \begin{pedagogy}...\end{pedagogy}
#   T2  the pedagogy environment is balanced in the annotated copy (an unbalanced
#       \begin would silently swallow real content into the stripped region and
#       make T1 pass vacuously)
#   T3  the stripper actually removed something (>0 pedagogy blocks) and left a
#       substantial body (>0 lines) -- anti-vacuity on both ends
#   T4  the built file and the linted file are still the pair this script names,
#       read from article_v2.tex's \input and the lint's FILES list, so that
#       renaming either file fails here instead of silently un-gating the pair
#
# Run:  python3 appendix_twins_verification.py
# =============================================================================
import os
import re
import sys
import difflib

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", "..", "theory", "paper"))

CLEAN = "osgs_appendix.tex"
ANNOTATED = "osgs_appendix_commented.tex"
SHARED_ANCHOR = r"\section{Convergence analysis of the OSGS method}"

results = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    line = f"  [{tag}] {name}"
    if detail and not ok:
        line += f"\n         -> {detail}"
    print(line)
    return ok


def read(fn):
    with open(os.path.join(PAPER, fn), encoding="utf-8") as f:
        return f.read()


def body_lines(text, strip_pedagogy):
    """Non-blank lines of the appendix proper, optionally without pedagogy boxes."""
    text = text[text.index(SHARED_ANCHOR):]
    if not strip_pedagogy:
        return [l for l in text.split("\n") if l.strip()], 0, 0
    out, depth, opened, maxdepth = [], 0, 0, 0
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith(r"\begin{pedagogy}"):
            depth += 1
            opened += 1
            maxdepth = max(maxdepth, depth)
            continue
        if s.startswith(r"\end{pedagogy}"):
            depth -= 1
            continue
        if depth == 0:
            out.append(line)
    return [l for l in out if l.strip()], opened, depth


print("=" * 72)
print("APPENDIX D TWINS: the BUILT file and the LINTED file must be the same appendix")
print("=" * 72)

clean_txt, annot_txt = read(CLEAN), read(ANNOTATED)
clean_lines, _, _ = body_lines(clean_txt, False)
annot_lines, n_boxes, residual_depth = body_lines(annot_txt, True)

check(f"T2 the pedagogy environment is balanced in {ANNOTATED} "
      f"({n_boxes} blocks opened, closing depth {residual_depth})",
      residual_depth == 0,
      "an unbalanced \\begin{pedagogy} would hide real content and make T1 vacuous")

check(f"T3 the stripper is non-vacuous ({n_boxes} pedagogy blocks removed, "
      f"{len(annot_lines)} body lines left, {len(clean_lines)} in the clean twin)",
      n_boxes > 0 and len(annot_lines) > 100 and len(clean_lines) > 100)

diff = [l for l in difflib.unified_diff(clean_lines, annot_lines,
                                        CLEAN, ANNOTATED, lineterm="", n=0)
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
check(f"T1 the twins agree line for line outside the pedagogy boxes "
      f"({len(clean_lines)} vs {len(annot_lines)} lines compared, {len(diff)} differ)",
      not diff,
      " | ".join(x.strip()[:90] for x in diff[:6]))

# T4  the pair is still the built/linted pair this script is named for.
main = read("article_v2.tex")
built = re.findall(r"(?m)^\s*\\input\{(osgs_appendix[a-z_]*)\.tex\}", main)
with open(os.path.join(HERE, "theorem_statement_verification.py"), encoding="utf-8") as f:
    lint_src = f.read()
linted = re.findall(r'"(osgs_appendix[a-z_]*)\.tex"', lint_src)
check(f"T4 article_v2 builds {built} and the statement lint reads {linted}: this script "
      f"guards exactly that pair",
      built == [ANNOTATED[:-4]] and linted == [CLEAN[:-4]],
      f"built={built} linted={linted}; if either was renamed, update this gate")

# Discriminating negative: an injected divergence must be caught.
_a = ["\\begin{lemma}", "  x = 1", "\\end{lemma}"]
_b = ["\\begin{lemma}", "  x = 2", "\\end{lemma}"]
check("negative: an injected one-line divergence is detected",
      len([l for l in difflib.unified_diff(_a, _b, lineterm="", n=0)
           if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]) == 2)

print("=" * 72)
n_fail = sum(1 for tag, _ in results if tag == "FAIL")
print(f"SUMMARY: {len(results) - n_fail}/{len(results)}")
print("=" * 72)
sys.exit(1 if n_fail else 0)
