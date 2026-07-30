#!/usr/bin/env python3
# =============================================================================
# document_hygiene_verification.py
#
# WHY THIS SCRIPT EXISTS.  Until 2026-07-29 not one script in this suite read a
# LaTeX BUILD LOG.  Every other *_verification.py either re-derives the paper's
# mathematics from its own encoding (and so never looks at the document at all)
# or lints a .tex SOURCE for textual anchors.  The consequence is a structural
# blind spot: the *rendered* document was unverified.  A whole-paper referee read
# on 2026-07-29 found, in a paper that had been reviewed many times and was green
# on 454/454 symbolic checks:
#
#   * an Overfull \hbox of 19.09pt in article.tex printing a VISIBLE black rule in
#     the margin next to eq. (5.12) on p. 22 (the SIAM class leaves \overfullrule
#     at its 5pt default outside the draft/final options),
#   * two more overfull boxes of 8.68pt and 20.79pt in article_v2.tex (table rows),
#   * a "Float too large for page by 81.27pt" in BOTH mains,
#   * four "Command \small invalid in math mode" font warnings from a \small\vert
#     written inside a math environment.
#
# None of these is visible to a symbolic check, and none of them is a "vacuous
# rule" in the 2026-07-28 sense -- there was simply NO rule, because no input to
# the suite carried the information.  The log is that input.
#
# WHAT IS CHECKED, per document, from the LIVE log under "latex compilation/<base>/":
#   H1  zero undefined references
#   H2  zero undefined citations
#   H3  zero multiply-defined labels
#   H4  zero "Float too large for page"
#   H5  zero "invalid in math mode" font warnings
#   H6  no Overfull \hbox wider than OVERFULL_PT_LIMIT (these print a margin rule)
#   H8  exactly ONE aux directory holds a <base>.log -- two means a misrouting
#       latexmkrc is writing the live log somewhere other than the correctly-named
#       sibling, leaving a stale log for this gate (or a reader) to trust.
#   H7  the log is FRESH -- newer than every .tex it could have been built from.
#       Without this the gate would happily certify a stale build, which is the
#       failure mode already recorded for the Coq harness (lessons_learned,
#       2026-07-17: "silently green on a stale/incomplete tree").
#
# Every rule reports the number of items it inspected; the per-document counter
# and the document counter can both reach zero and FAIL, so the gate cannot go
# silently vacuous.  A discriminating negative control at the end feeds each rule
# a synthetic bad log and asserts it fires.
#
# Run:  python3 document_hygiene_verification.py
#       Build FIRST, and build each document in its OWN latexmk invocation -- the
#       latexmkrc computes $aux_dir from @ARGV's first .tex, so
#       `latexmk -pdf article.tex article_v2.tex` routes article_v2's intermediates
#       into 'latex compilation/article/' and H8 fails.  The documents this gate
#       needs are the DOCS list below:
#           cd theory/paper                    && latexmk -pdf article.tex
#           cd theory/paper                    && latexmk -pdf article_v2.tex
#           cd theory/viscous_projector_note   && latexmk -pdf viscous_projector_note.tex
#           cd theory/mesh_regularity_note     && latexmk -pdf mesh_regularity_note.tex
#           cd theory/centered_encoding        && latexmk -pdf centered_encoding.tex
#           cd theory/cocquet                  && latexmk -pdf cocquet_form_mms_manufactured_solution.tex
#           cd proof_verification              && latexmk -pdf coq_coverage.tex
# =============================================================================
import os
import re
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# Overfull boxes below this are not visible and do not draw the margin rule in
# practice; above it they do.  Raising this to hide a real defect is a regression.
OVERFULL_PT_LIMIT = 1.0

# (directory relative to ROOT, document basename, overfull-box DEBT BUDGET)
#
# The budget is a RECORDED DEBT, not a licence.  The three submission-track documents carry 0:
# any overfull box wider than OVERFULL_PT_LIMIT is a defect.  coq_coverage is an internal
# verification-coverage report, not a submission artefact; it is full of deliberately wide
# alignment tables and carries its current count so that the gate cannot go red on known debt
# while still failing the moment an 18th box appears.  A budget may only ever be REDUCED --
# the gate prints a nudge when the actual count comes in under it.
DOCS = [
    ("theory/paper", "article", 0),
    ("theory/paper", "article_v2", 0),
    ("theory/viscous_projector_note", "viscous_projector_note", 0),
    # 2026-07-29: 17 -> 7 (text block widened, all tables set \footnotesize, three column
    # specs tightened; worst box 133pt -> 56pt).  The residue needs per-cell content work
    # in an internal audit report -- low value.  This number may only ever go DOWN.
    ("proof_verification", "coq_coverage", 7),
    # 2026-07-30: the standalone note that discharges the global-quasi-uniformity claim the two
    # mains now assume in (A1) and its "more than the proofs need" footnote. The papers do not
    # cite it (it is internal), which is exactly why it needs a build gate: nothing else would
    # notice if its proofs stopped compiling while (A1) still leaned on them.
    ("theory/mesh_regularity_note", "mesh_regularity_note", 0),
    # 2026-07-30: the two notes this pass edited.  Both were UNGATED, and both silently acquired a
    # catastrophic box from that edit -- a provenance table 344.92pt (~12cm) past the margin here,
    # and two 124/127-character verbatim lines (243/260pt) in the Cocquet note.  Nothing would have
    # caught either.  Both are fixed; the budgets below are the PRE-EXISTING debt of prose that was
    # never gated, and may only ever go DOWN.  NB: neither directory's latexmkrc worked before this
    # pass -- both used the literal-'%B' aux_dir, so the live log sat in 'latex compilation/%B/'
    # while a months-old stale log sat in the correctly-named sibling that this gate reads.
    ("theory/centered_encoding", "centered_encoding", 10),
    ("theory/cocquet", "cocquet_form_mms_manufactured_solution", 2),
]

results = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    line = f"  [{tag}] {name}"
    if detail and not ok:
        line += f"\n         -> {detail}"
    print(line)
    return ok


def find_logs(dirpath, base):
    """Every candidate log under '<dir>/latex compilation/<something>/<base>.log'.

    [known-fragility]  The aux subdirectory is '<base>' with the @ARGV latexmkrc
    files, but a LITERAL '%B' with the older ones (TeX Live 2023's latexmk 4.79
    does not expand %B inside $aux_dir).  A directory with BOTH therefore holds a
    live log and a stale one -- and on 2026-07-30 this cost a real defect: the
    previous version of this function returned sorted(...)[0], and '%B' sorts
    BEFORE any letter, so for theory/centered_encoding it would have read a
    months-old log while the build that actually shipped the PDF -- carrying a
    344.92pt overfull table, ~12cm past the margin -- wrote to '%B/'.  So return
    all candidates NEWEST FIRST, and let the caller fail on multiplicity."""
    pat = os.path.join(ROOT, dirpath, "latex compilation", "*", base + ".log")
    return sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)


def read_log(path):
    # LaTeX logs are latin-1-ish; decoding as utf-8 raises and reading as bytes
    # makes every regex a bytes regex.  Decode permissively.
    with open(path, encoding="latin-1") as f:
        return f.read()


OVERFULL = re.compile(r"^Overfull \\hbox \(([0-9.]+)pt too wide\)(.*)$", re.M)
# [known-fragility]  A LaTeX log's "at line N" is relative to the file OPEN at that moment, NOT
# to the main document.  Reporting the bare number sends the reader to the wrong file: on
# 2026-07-29 two boxes reported "line 1776"/"line 1886" were taken for result tables in
# article_v2.tex and were in fact displays in osgs_appendix_commented.tex, 100 pages away.  The
# log records the nesting as "(./file.tex ... )", so track the stack and attribute properly.
FILE_OPEN = re.compile(r"\((\.?/?[\w./-]+\.tex)")


def attribute(txt, pos):
    """Innermost .tex file open at character offset `pos` of the log."""
    stack = []
    for m in re.finditer(r"\((\.?/?[\w./-]+\.tex)|(\))", txt[:pos]):
        if m.group(1):
            stack.append(os.path.basename(m.group(1)))
        elif stack:
            stack.pop()
    return stack[-1] if stack else "<main>"
UNDEF_REF = re.compile(r"^LaTeX Warning: Reference .* undefined", re.M)
UNDEF_CITE = re.compile(r"^LaTeX Warning: Citation .* undefined", re.M)
MULTIDEF = re.compile(r"multiply defined", re.M)
FLOAT_BIG = re.compile(r"^LaTeX Warning: Float too large for page.*$", re.M)
MATHMODE = re.compile(r"^LaTeX Font Warning: Command .* invalid in math mode.*$", re.M)
OUTPUT_OK = re.compile(r"^Output written on ", re.M)


def newest_dependency(logpath, dirpath):
    """mtime of the newest .tex the document ACTUALLY depends on.

    Taken from latexmk's .fls beside the log, which records one "INPUT <path>" line per file
    the engine read.  A coarser rule -- "the newest .tex in the directory" -- produces false
    staleness: theory/paper holds both mains and four appendices, and editing an appendix that
    only article_v2 inputs would wrongly mark article's build stale.  A gate that cries wolf is
    a gate people learn to ignore."""
    fls = os.path.splitext(logpath)[0] + ".fls"
    newest = 0.0
    if os.path.exists(fls):
        seen = set()
        with open(fls, encoding="latin-1") as f:
            for line in f:
                if not line.startswith("INPUT "):
                    continue
                src = line[6:].strip()
                if not src.endswith(".tex") or src in seen:
                    continue
                seen.add(src)
                cand = src if os.path.isabs(src) else os.path.join(ROOT, dirpath, src)
                if os.path.exists(cand):
                    newest = max(newest, os.path.getmtime(cand))
    if newest == 0.0:                      # no .fls: fall back to the directory scan
        for f in glob.glob(os.path.join(ROOT, dirpath, "*.tex")):
            newest = max(newest, os.path.getmtime(f))
    return newest


print("=" * 72)
print("DOCUMENT HYGIENE: the RENDERED document, read from the live build logs")
print("=" * 72)

n_docs = 0
for dirpath, base, budget in DOCS:
    logs = find_logs(dirpath, base)
    if not logs:
        check(f"{base}: a live build log exists under '{dirpath}/latex compilation/'",
              False,
              f"no log found; build it first (cd '{dirpath}' && latexmk -pdf {base}.tex)")
        continue
    log = logs[0]  # newest
    # H8: exactly ONE aux directory. More than one means a misrouting latexmkrc
    # (the literal-'%B' form) is writing the live log somewhere other than the
    # correctly-named sibling, leaving a stale log for a reader to trust.
    if not check(f"H8 {base}: exactly one aux directory holds a {base}.log "
                 f"({len(logs)} found)", len(logs) == 1,
                 "candidates: " + ", ".join(os.path.relpath(p, ROOT) for p in logs)
                 + f"  -- fix {dirpath}/latexmkrc to compute $aux_dir from @ARGV "
                   "(see theory/paper/latexmkrc) and delete the stale directory"):
        continue
    txt = read_log(log)
    if not OUTPUT_OK.search(txt):
        check(f"{base}: the log records a completed run ('Output written on')", False,
              f"{log} looks truncated or the run failed")
        continue
    n_docs += 1

    # H7 first: everything below is meaningless if the log predates the sources.
    fresh = os.path.getmtime(log) >= newest_dependency(log, dirpath) - 1.0
    check(f"H7 {base}: the build log is newer than the sources beside it", fresh,
          f"log is older than a .tex it INPUTs; rebuild: cd '{dirpath}' && latexmk -pdf {base}.tex")

    n_ref = len(UNDEF_REF.findall(txt))
    n_cite = len(UNDEF_CITE.findall(txt))
    n_multi = len(MULTIDEF.findall(txt))
    n_float = len(FLOAT_BIG.findall(txt))
    n_math = len(MATHMODE.findall(txt))
    boxes = [(float(m.group(1)), m.group(2).strip(), attribute(txt, m.start()))
             for m in OVERFULL.finditer(txt)]
    bad_boxes = [b for b in boxes if b[0] > OVERFULL_PT_LIMIT]

    check(f"H1 {base}: no undefined references ({n_ref} found)", n_ref == 0)
    check(f"H2 {base}: no undefined citations ({n_cite} found)", n_cite == 0)
    check(f"H3 {base}: no multiply-defined labels ({n_multi} found)", n_multi == 0)
    check(f"H4 {base}: no float too large for the page ({n_float} found)", n_float == 0,
          "; ".join(FLOAT_BIG.findall(txt)[:3]))
    check(f"H5 {base}: no 'invalid in math mode' font warnings ({n_math} found)",
          n_math == 0, "; ".join(sorted(set(MATHMODE.findall(txt)))[:3]))
    nudge = (f"  [budget {budget} can be tightened to {len(bad_boxes)}]"
             if len(bad_boxes) < budget else "")
    check(f"H6 {base}: Overfull \\hbox wider than {OVERFULL_PT_LIMIT}pt within the recorded "
          f"debt budget ({len(bad_boxes)} of {len(boxes)} boxes exceed it, budget {budget})"
          + nudge,
          len(bad_boxes) <= budget,
          "; ".join(f"{pt}pt in {src} {where}" for pt, where, src in bad_boxes[:4]))

# Anti-vacuity: the loop must actually have inspected the documents.
check(f"H0 every required document was inspected ({n_docs} of {len(DOCS)})",
      n_docs == len(DOCS),
      "a missing or truncated log silently removes a document from this gate")

# ---------------------------------------------------------------------------
# Discriminating negatives: each pattern must fire on a synthetic bad log.
# ---------------------------------------------------------------------------
print("-" * 72)
BAD = (
    "Output written on x.pdf (1 page).\n"
    "LaTeX Warning: Reference `foo' on page 1 undefined on input line 3.\n"
    "LaTeX Warning: Citation `bar' on page 1 undefined on input line 4.\n"
    "LaTeX Warning: Label `baz' multiply defined.\n"
    "LaTeX Warning: Float too large for page by 81.26614pt on input line 1441.\n"
    "LaTeX Font Warning: Command \\small invalid in math mode on input line 580.\n"
    "Overfull \\hbox (19.08522pt too wide) detected at line 1151\n"
    "Overfull \\hbox (0.11728pt too wide) detected at line 358\n"
)
check("negative: an undefined reference is detected", len(UNDEF_REF.findall(BAD)) == 1)
check("negative: an undefined citation is detected", len(UNDEF_CITE.findall(BAD)) == 1)
check("negative: a multiply-defined label is detected", len(MULTIDEF.findall(BAD)) == 1)
check("negative: a too-large float is detected", len(FLOAT_BIG.findall(BAD)) == 1)
check("negative: a math-mode font warning is detected", len(MATHMODE.findall(BAD)) == 1)
_b = [float(m.group(1)) for m in OVERFULL.finditer(BAD)]
check(f"negative: the 19.09pt box is caught and the 0.12pt one is not "
      f"(threshold {OVERFULL_PT_LIMIT}pt)",
      len(_b) == 2 and len([x for x in _b if x > OVERFULL_PT_LIMIT]) == 1)

print("=" * 72)
n_fail = sum(1 for tag, _ in results if tag == "FAIL")
print(f"SUMMARY: {len(results) - n_fail}/{len(results)}")
print("=" * 72)
sys.exit(1 if n_fail else 0)
