#!/usr/bin/env python3
# =============================================================================
# assembly_consistency_verification.py
#
# STRUCTURAL (bookkeeping) verification of the "Putting together the results"
# assembly in theory/paper/elemental_matrices_appendix.tex.
#
# The sibling scripts (elemental_matrices_verification.py,
# elemental_bilinear_form_verification.py) verify the *content* of every
# elemental matrix -- each component formula and each family-sum against the
# bilinear form. They do NOT look at the final assembled displays
#     K = [[I,0],[0,0]] + [[...Galerkin...]] + K_S ,   F = [V_F;0] + F_S ,
#     K_S = [[...]] ,   F_S = [[...]] .
# A matrix can therefore be *correct in isolation* yet be (a) named in the
# assembly without ever being defined, or (b) defined and then silently
# dropped from the assembly. Those are exactly the four defects the July-2026
# audit found in Part I:
#     named-but-undefined :  I  (transient mass matrix -- 0 in the stationary
#                                 paper),  G_beta,  D_beta  (bare porosity-
#                                 gradient blocks that only exist as two-
#                                 subscript stabilization cross-terms)
#     defined-but-unused  :  V_T (the Neumann traction load vector -- dropping
#                                 it silently discards the Neumann load)
#
# This script implements the durable invariant proposed by that audit:
#   * every matrix NAMED in the general assembly must be DEFINED, and
#   * every matrix DEFINED must appear in the general assembly.
#
# It parses the appendix directly, so it stays true if the appendix changes.
# Pure text processing -- no sympy needed. Emits the same "SUMMARY: n/m" line
# as the other scripts so run_all.py picks it up.
#
# Run:  python3 assembly_consistency_verification.py
# =============================================================================
import os
import re
import sys

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    print(f"  [{tag}] {name}")
    return ok

HERE = os.path.dirname(os.path.abspath(__file__))
APPENDIX = os.path.normpath(os.path.join(
    HERE, "..", "..", "theory", "paper", "elemental_matrices_appendix.tex"))

# --- Greek / macro normalisation so "\nu D", "\sigma", "\beta", "\text{S}" etc.
#     collapse to a stable canonical spelling ---------------------------------
_GREEK = {
    r"\nu": "nu", r"\sigma": "sig", r"\alpha": "al", r"\beta": "be",
    r"\phi": "phi", r"\varepsilon": "eps",
}
# Index tuples that terminate a component-definition subscript, longest first.
_INDEX_TUPLES = ["(ai)(bj)", "a(bj)", "(ai)b", "(ia)", "ab", "a"]


def _canon_subscript(sub):
    """Normalise a LaTeX subscript body to a canonical, whitespace-free name."""
    s = sub
    for k, v in _GREEK.items():
        s = s.replace(k, v + " ")          # keep a boundary so "\nu D" -> "nu D"
    s = s.replace(r"\text", "").replace(r"\,", " ").replace(r"\;", " ")
    s = s.replace("{", " ").replace("}", " ").replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "")


def _strip_index_tuple(sub):
    """Remove a trailing (ai)(bj)/a(bj)/(ai)b/(ia)/ab/a index tuple, if present.

    Returns (subscript_without_indices, matched) so callers can require that a
    genuine component definition actually ended in an index tuple.
    """
    s = sub.rstrip()
    # drop trailing separators like  "\," or spaces just before the tuple
    for tup in _INDEX_TUPLES:
        if s.endswith(tup):
            s2 = s[: -len(tup)]
            # a lone trailing "a"/"ab" must be a real index, i.e. preceded by a
            # separator or brace-space, never glued onto subscript letters.
            if tup in ("a", "ab") and s2 and s2[-1].isalnum():
                continue
            return s2, True
    return s, False


# ---------------------------------------------------------------------------
# 1. DEFINED matrices: LHS of every component-definition equation.
#    Pattern:  <Letter>_{ <subscript><index-tuple> }  &?=    (start of an align
#    row / equation).  We also accept the no-subscript form  V_{(ai)(bj)} = ...
#    and, for the boundary term, V_{T\, (ia)} = ...
# ---------------------------------------------------------------------------
_DEF_RE = re.compile(
    r"(?:^|[\s&])"                       # row / cell boundary
    r"([A-Za-z])_\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"   # X_{ subscript } (1 nesting)
    r"\s*(?:&\s*)?=",                    # optional align '&' then '='
)

# Generic template symbols and finite-element-function LHSs that are NOT
# elemental matrices (they use the same X_{...} = shape but denote the abstract
# quadrature rule or the FE interpolation, not an assembled block).
_DEF_EXCLUDE_LETTERS = {"T", "u", "p"}


def parse_defined(text):
    names = {}
    for m in _DEF_RE.finditer(text):
        letter, sub = m.group(1), m.group(2)
        if letter in _DEF_EXCLUDE_LETTERS:
            continue
        body, had_tuple = _strip_index_tuple(sub)
        if not had_tuple:
            continue                       # not a component definition
        canon = letter + ("_" + _canon_subscript(body) if _canon_subscript(body) else "")
        names.setdefault(canon, m.start())
    return names


# ---------------------------------------------------------------------------
# 2. USED matrices: every \mathbf{..}/\boldsymbol{..} block symbol appearing in
#    the GENERAL assembly (K, K_S, F, F_S) -- not the first-order variant.
# ---------------------------------------------------------------------------
# A \mathbf{}/\boldsymbol{} symbol whose subscript is either a braced group
# {..}, a bare Greek macro (\sigma), or a single char -- covering \mathbf{A}_A,
# \mathbf{A}_\sigma and \mathbf{A}_{G\beta} alike.
_USE_RE = re.compile(
    r"\\(?:mathbf|boldsymbol)\{([A-Za-z0])\}"
    r"(?:_\{([^{}]*)\}|_(\\[A-Za-z]+)|_([A-Za-z0-9]))?")

# \vphantom{..} etc. are invisible spacing struts, not real usages -- strip the
# (single-nesting) argument so struts like \vphantom{\mathbf{A}_A} do not read
# as assembly entries.
_PHANTOM_RE = re.compile(r"\\[vh]?phantom\{(?:[^{}]|\{[^{}]*\})*\}")

# Structural block symbols that are containers, not assembled contributions.
_USE_EXCLUDE = {"0", "K", "F", "K_S", "F_S"}


def _canon_use(letter, sub):
    if sub is None:
        return letter
    return letter + ("_" + _canon_subscript(sub) if _canon_subscript(sub) else "")


def parse_used(assembly_text):
    assembly_text = _PHANTOM_RE.sub(" ", assembly_text)
    used = []
    for m in _USE_RE.finditer(assembly_text):
        letter = m.group(1)
        sub = next((g for g in (m.group(2), m.group(3), m.group(4)) if g is not None), None)
        canon = _canon_use(letter, sub)
        if canon in _USE_EXCLUDE:
            continue
        used.append(canon)
    return used


def slice_general_assembly(text):
    """From 'Putting together the results' up to (but excluding) the first-order
    variant that starts at 'If second-order contributions can be neglected'."""
    start = text.index("Putting together the results")
    end = text.index("If second-order contributions can be neglected")
    return text[start:end]


# ---------------------------------------------------------------------------
def main():
    with open(APPENDIX, encoding="utf-8") as fh:
        text = fh.read()

    # Tolerate the review \amend{...} markup on definition LHSs and assembly
    # names (commit 8a644d2 wrapped e.g. \amend{\mathbf{G}_{\alpha P}}): unwrap
    # it so the real matrix symbols are parsed. This checks the math content,
    # not the review coloring, so the invariant is unchanged.
    text = re.sub(r"\\amend\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", text)

    asm = slice_general_assembly(text)
    definitions_text = text[: text.index("Putting together the results")]

    defined = parse_defined(definitions_text)
    used_list = parse_used(asm)
    used = set(used_list)

    print("=" * 70)
    print("Assembly consistency (Appendix: elemental_matrices_appendix.tex)")
    print("=" * 70)
    print(f"\nDefined component matrices ({len(defined)}):")
    print("  " + ", ".join(sorted(defined)))
    print(f"\nMatrices named in the general assembly ({len(used)} distinct):")
    print("  " + ", ".join(sorted(used)))

    # --- parser sanity guard: a broken regex must not silently pass by
    #     producing empty/degenerate sets. Require known anchors. ---
    anchors_def = {"G_S", "D_nuD", "V", "R_sig", "A_A", "G_beA", "V_F",
                   "Q_alF", "V_T", "P_Q", "G"}
    anchors_use = {"G_S", "D_nuD", "V", "R_sig", "P_Q", "G"}
    check(f"parser sanity: all {len(anchors_def)} anchor definitions detected",
          anchors_def <= set(defined))
    check(f"parser sanity: all {len(anchors_use)} anchor assembly symbols detected",
          anchors_use <= used)

    named_but_undefined = sorted(used - set(defined))
    defined_but_unused = sorted(set(defined) - used)

    print(f"\nNamed-but-undefined (in assembly, never defined): {named_but_undefined or '(none)'}")
    print(f"Defined-but-unused   (defined, never assembled):  {defined_but_unused or '(none)'}")

    # informational: symbols repeated in the assembly (e.g. the G_P name is
    # reused for both the Galerkin flux block and the tau2 stabilization block).
    dups = sorted({n for n in used_list if used_list.count(n) > 1})
    if dups:
        print(f"(info) symbols appearing >1x in the assembly: {dups}")

    check("no matrix is NAMED in the assembly without being DEFINED",
          not named_but_undefined)
    check("no matrix is DEFINED without appearing in the assembly",
          not defined_but_unused)

    print("\n" + "=" * 70)
    npass = sum(1 for t, _ in results if t == "PASS")
    print(f"SUMMARY: {npass}/{len(results)} assembly-consistency checks passed.")
    for t, nme in results:
        if t == "FAIL":
            print(f"   FAILED: {nme}")
    print("=" * 70)
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
