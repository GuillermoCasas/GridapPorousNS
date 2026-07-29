#!/usr/bin/env python3
# =============================================================================
# theorem_statement_verification.py
#
# Extends the display-consistency net (2026-07-22 external audit, issues M13/M14/
# M15) from INTERMEDIATE PROOF DISPLAYS to the PRINTED LEMMA / THEOREM STATEMENTS
# themselves. It guards the two decoration classes a symbolic check cannot see
# because they are TEXTUAL, not algebraic:
#
#   (1) Absolute-value bars on a bilinear-form bound. A continuity / boundedness
#       lemma must state  |B(a;U,V)| <= (...) ,  not the one-sided  B(a;U,V) <= (...) .
#       A one-sided bound is TRUE but strictly weaker than a continuity estimate,
#       so neither SymPy (self-contained algebra) nor Coq flags it -- indeed Coq's
#       own abstract_continuity was transcribed one-sided (BS <= Ctot*(...), no
#       Rabs) and the kernel certified it happily (see AbstractContinuity.v:46).
#       This is the 2026-07-23 audit's section 3.1 / M13 defect.
#
#   (2) An inf-sup sup/inf whose trial/test domain fails to exclude 0
#       (\sup_{V_h \in Xhz}  must be  \sup_{V_h \in Xhz\setminus\{0\}}), else the
#       Rayleigh quotient is 0/0 at V_h = 0. This is the audit's A17 residual.
#
# SOURCE-COUPLED BY DESIGN: unlike the symbolic scripts (which hard-code the
# paper's algebra), this one reads the LIVE submission-track appendices so that a
# regression in a printed statement is caught. Each rule ships with a synthetic
# NEGATIVE (a hand-built bad statement the rule must reject), so a rule that
# silently stops discriminating is itself caught.
#
# Run:  python3 theorem_statement_verification.py      (no sympy needed)
# =============================================================================
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", "..", "theory", "paper"))
# Submission-track appendices: App C (ASGS/continuity) and App D (OSGS, clean), PLUS
# both mains.  The mains are in scope because printed lemma/theorem statements live
# there too -- lemma:Continuity states the barred bound |B_ASGS(...)| <= ... in the
# body, and the 2026-07-29 relocation moved lem:projfamily / lem:projinstances out of
# App. C and into the main-text subsection sec:ViscousProjector.  A lint that reads
# only the appendices would not follow them.
FILES = ["asgs_convergence.tex", "osgs_appendix.tex", "article.tex", "article_v2.tex"]

results = []
def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    line = f"  [{tag}] {name}"
    if detail and not ok:
        line += f"\n         -> {detail}"
    print(line)
    return ok

# A bilinear stabilized form as it appears in the two appendices.  Both the macro
# spellings (\BS, \BASGS, \BOSGS) and the expanded ones (B_{\mathrm{ASGS}}, ...) must be
# listed: App C writes the ASGS form through the macro \BASGS while App D writes the OSGS
# form expanded as B_{\mathrm{OSGS}}.  [known-fragility]  Miss a spelling and Rule 1 below
# silently checks NOTHING while still reporting PASS -- which is exactly what happened
# between the B_S -> B_ASGS/B_OSGS rename and 2026-07-28.  The "(N bound(s) checked)"
# counter in the PASS line is the tell: it must stay > 0 for both files.
BILINEAR = re.compile(r'\\BASGS|\\BOSGS|\\BS|\\BO'
                      r'|B_\{\\mathrm\{(?:osgs|asgs|ASGS|OSGS|S|O)\}\}')
# Any "opening absolute value" delimiter (\lvert / \lVert with any size prefix, or a bar |).
ABS_OPEN = re.compile(r'\\(?:bigl|Bigl|biggl|Biggl|left)?\s*(?:\\lvert|\\lVert|\|)')
# An UPPER-bound relation. Rule 1 targets continuity/boundedness (|B| <= ...) ONLY:
# a lower bound (B(U,U) >= C||U||^2 coercivity, or an inf-sup quotient >= beta) is a
# statement about the SIGNED value and must NOT be barred, so \ge/\geq are excluded.
UPPER_REL = re.compile(r'\\le\b|\\leq\b')
STMT_ENV = re.compile(r'\\begin\{(lemma|theorem|proposition|corollary)\}')


def read(fn):
    with open(os.path.join(PAPER, fn), encoding="utf-8") as f:
        return f.read()


def statement_regions(text):
    """
    Yield the STATEMENT text of every lemma/theorem/proposition/corollary: from
    \\begin{env} to whichever comes first, its \\begin{proof} or its \\end{env}.
    Proof bodies (covered by the other display-consistency scripts) are excluded.
    """
    for m in STMT_ENV.finditer(text):
        start = m.start()
        env = m.group(1)
        tail = text[m.end():]
        stops = [t.start() for t in
                 (re.search(r'\\begin\{proof\}', tail), re.search(r'\\end\{' + env + r'\}', tail))
                 if t]
        end = m.end() + (min(stops) if stops else len(tail))
        yield text[start:end]


def equation_blocks(text):
    """Yield the body of every equation / align / gather (starred or not) block."""
    for m in re.finditer(r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}'
                         r'(.*?)\\end\{\1\}', text, re.S):
        yield m.group(2)


def lhs_of(block):
    """LHS = text up to the first UPPER-bound relation; None if none present."""
    m = UPPER_REL.search(block)
    return block[:m.start()] if m else None


def bilinear_bound_is_barred(block):
    """
    For a block that BOUNDS a bilinear form, return True iff the form on the LHS is
    wrapped in absolute-value bars (an abs-open delimiter occurs before the form).
    Returns None if the block is not a bilinear-form bound (nothing to check).
    """
    lhs = lhs_of(block)
    if lhs is None:
        return None
    bm = BILINEAR.search(lhs)
    if not bm:
        return None
    # An abs-open delimiter must appear strictly before the form on the LHS.
    return any(am.start() < bm.start() for am in ABS_OPEN.finditer(lhs))


# --- inf-sup domain rule ----------------------------------------------------
# Flag any  \sup_{ X_h \in \Xh(z) }  /  \inf_{...}  quotient that omits \setminus\{0\}.
# [known-fragility]  BOTH spellings of the space must be in the pattern: \Xhz is only an
# alias (\newcommand{\Xhz}{\Xh}), and the printed inf-sup of the MAIN TEXT writes \Xh.
# A pattern hard-wired to \Xhz matches nothing outside App. D and prints "(0 guarded)"
# while still passing -- the same silent vacuity as the 2026-07-28 Rule 1 regression.
SUP_BARE = re.compile(r'\\(?:sup|inf)_\{\s*[UV]_h\s*\\in\s*\\Xhz?\s*\}')
SUP_OK   = re.compile(r'\\(?:sup|inf)_\{\s*[UV]_h\s*\\in\s*\\Xhz?\\setminus\\\{0\\\}\s*\}')


print("=" * 72)
print("Theorem/lemma STATEMENT lint: abs-value bars + inf-sup domains")
print("=" * 72)

# ---------------------------------------------------------------------------
# Rule 1: every bilinear-form UPPER bound in a lemma/theorem STATEMENT is barred.
# ---------------------------------------------------------------------------
for fn in FILES:
    text = read(fn)
    offenders = []
    checked = 0
    for region in statement_regions(text):
        for blk in equation_blocks(region):
            barred = bilinear_bound_is_barred(blk)
            if barred is None:
                continue
            checked += 1
            if not barred:
                snippet = re.sub(r'\s+', ' ', (lhs_of(blk) or '').strip())[:80]
                offenders.append(snippet)
    check(f"{fn}: every bilinear-form upper bound in a statement carries |.| "
          f"({checked} continuity bound(s) checked)",
          not offenders,
          detail="one-sided (barless) bound(s): " + " | ".join(offenders))
    # Anti-vacuity: a rule that matches nothing is a rule that has stopped biting.
    check(f"{fn}: the bilinear-form rule is non-vacuous (>=1 bound reached it)",
          checked > 0,
          detail="BILINEAR did not match any statement -- the form's spelling probably "
                 "changed; extend the regex rather than accepting the empty PASS")

# ---------------------------------------------------------------------------
# Rule 2: no inf-sup sup/inf over Xhz omits \setminus\{0\}.
# ---------------------------------------------------------------------------
n_guarded_total = 0
for fn in FILES:
    text = read(fn)
    bare = SUP_BARE.findall(text)
    n_guarded = len(SUP_OK.findall(text))
    n_guarded_total += n_guarded
    check(f"{fn}: every inf-sup sup/inf over Xh excludes 0 "
          f"({n_guarded} guarded)",
          len(bare) == 0,
          detail=f"{len(bare)} sup/inf over Xh without \\setminus\\{{0\\}}")
# Anti-vacuity, GLOBAL: per-file would fail legitimately (article.tex states no inf-sup,
# and App. C is the coercivity route). But if NO file in the whole set has a guarded
# quotient, the pattern has stopped matching the paper and the rule is inspecting nothing.
check(f"the inf-sup domain rule is non-vacuous across the set "
      f"({n_guarded_total} guarded quotient(s) found in {len(FILES)} files)",
      n_guarded_total > 0,
      detail="SUP_OK matched nowhere -- the space's spelling probably changed; extend the "
             "pattern rather than accepting the empty PASS")

# ---------------------------------------------------------------------------
# Discriminating NEGATIVES: the rules must REJECT hand-built bad statements.
# (If a rule stops biting, these fail and the gate goes red.)
# ---------------------------------------------------------------------------
bad_bound = r"\BS(\ba; U_h,V_h) \le C\Bigl( \norm{x} \Bigr)\,\triplenorm{V_h}"
check("negative: a barless bilinear bound is rejected",
      bilinear_bound_is_barred(bad_bound) is False)

good_bound = r"\bigl\lvert\BS(\ba; U_h,V_h)\bigr\rvert \le C\,\triplenorm{V_h}"
check("negative: a properly barred bound is accepted",
      bilinear_bound_is_barred(good_bound) is True)

good_bound_pipe = r"\bigl| B_{\mathrm{osgs}}(\ba; U-U_h, V_h) \bigr| \le C\,\Psi"
check("negative: the \\bigl|...\\bigr| pipe form is accepted",
      bilinear_bound_is_barred(good_bound_pipe) is True)

# The post-rename spellings actually used by the two appendices (App C: \BASGS macro;
# App D: expanded B_{\mathrm{OSGS}}).  These guard the 2026-07-28 vacuity regression.
check("negative: a barless \\BASGS bound is rejected",
      bilinear_bound_is_barred(r"\BASGS(\ba; U_h,V_h) \le C\,\triplenorm{V_h}") is False)
check("negative: a barred B_{\\mathrm{OSGS}} bound is accepted",
      bilinear_bound_is_barred(
          r"\bigl| B_{\mathrm{OSGS}}(\ba; U-U_h, V_h) \bigr| \le C\,\Psi") is True)

bad_sup = r"\sup_{V_h \in \Xhz} \frac{B_{\mathrm{osgs}}(\ba; U_h, V_h)}{\tnorm{V_h}}"
check("negative: a sup over Xhz missing \\setminus\\{0\\} is rejected",
      len(SUP_BARE.findall(bad_sup)) == 1)

good_sup = r"\sup_{V_h \in \Xhz\setminus\{0\}} \frac{B(\ba; U_h, V_h)}{\tnorm{V_h}}"
check("negative: a sup over Xhz\\setminus\\{0\\} is accepted",
      len(SUP_BARE.findall(good_sup)) == 0)

n_pass = sum(1 for t, _ in results if t == "PASS")
n_tot = len(results)
print("=" * 72)
print(f"SUMMARY: {n_pass}/{n_tot}")
sys.exit(0 if n_pass == n_tot else 1)
