#!/usr/bin/env python3
# =============================================================================
# latex_index_notation.py
#
# A small reader that turns the INDEX NOTATION actually printed in Appendix A of
#   "A stabilized finite element method for incompressible, inertial flows in
#    inhomogeneous porous media"  (theory/paper/elemental_matrices_appendix.tex)
# into sympy expressions.
#
# WHY IT EXISTS.  Every other script in this suite re-derives the appendix's
# elemental matrices from its OWN sympy encoding and never opens the .tex.  That
# certifies the destination while leaving the printed PATH unread: the ~60
# pre-differentiation integrands that sit inside
#     T_{(ai)(bj)} = <prefactor> d/dU_j^b ( <integrand> ) = <printed result>
# are verified by nothing.  On 2026-07-29 a whole-paper read found nine index
# defects there while the suite was green (docs/lessons_learned.md, BS-2).
#
# A hand transcription of those integrands would not close the gap honestly: the
# transcriber is the same reader who would have to notice the typo, and the
# characteristic defect (a trial factor silently re-using a test dummy index) is
# exactly what a human eye repairs without noticing.  So this module PARSES the
# source instead.  Two independent things then hold the paper to account:
#
#   (1) INDEX CENSUS.  After distributing all products over all sums -- which is
#       what the two rejected cheap lints could not do (see the spec in
#       docs/appendix-a-intermediate-coverage-spec.md Sec. 2) -- every monomial
#       must satisfy the Einstein convention exactly: each index letter occurs
#       once (and is then one of the display's declared free indices) or twice
#       (and is summed).  Three occurrences = a re-used dummy; one occurrence of
#       an undeclared letter = a dangling index.
#   (2) DIFFERENTIATION.  The parsed integrand is differentiated w.r.t. the nodal
#       unknown and compared against the parsed printed RESULT.  Both sides come
#       from the paper, so this checks the paper against itself, not against a
#       transcription.
#
# GRAMMAR (all of it; the appendix uses a closed token set):
#     expr    := term (('+'|'-') term)*
#     term    := factor+                                  (implicit product)
#     factor  := number | atom | '(' expr ')' | deriv | \frac{expr}{expr}
#     deriv   := \partial_x <operand> | \partial^2_{xy} <operand>
#     operand := '(' expr ')' | deriv | <factors up to and including the first
#                shape function N^., \alpha or \beta>
# The operand rule is what makes  "\partial^2_{km} U_m^c N^c"  (the derivative
# reaching PAST the constant nodal coefficient to the shape function) parse the
# same way as  "\partial_m N^c U_k^c"  (where it does not need to).
# =============================================================================
import re
from fractions import Fraction

import sympy as sp

# --------------------------------------------------------------------- symbols
X = sp.symbols('x0 x1 x2', real=True)
NA = sp.Function('Na')(*X)
NB = sp.Function('Nb')(*X)
NC = sp.Function('Nc')(*X)                      # trial shape function, undifferentiated
ALPHA = sp.Function('alpha', positive=True)(*X)
NU, EPS, TAU1, TAU2 = sp.symbols('nu varepsilon tau1 tau2', positive=True)
AVEC = sp.symbols('a0 a1 a2', real=True)
SIG = sp.Matrix(3, 3, lambda p, q: sp.Symbol(f'sig{min(p, q)}{max(p, q)}', real=True))
UDOF = sp.symbols('U0 U1 U2', real=True)        # nodal velocity coefficients U_k^c
PDOF = sp.Symbol('Pc', real=True)               # nodal pressure coefficient P^c
FBAR = sp.symbols('fbar0 fbar1 fbar2', real=True)
FVEC = sp.symbols('f0 f1 f2', real=True)
TN = sp.symbols('tN0 tN1 tN2', real=True)
PHIBAR = sp.Symbol('phibar', real=True)


class ParseError(Exception):
    """Raised when the printed source does not fit the appendix's own grammar."""


class IndexError_(Exception):
    """Raised when a monomial violates the Einstein summation convention."""


# --------------------------------------------------------------- normalisation
_DROP_WITH_ARG = ("vphantom", "hphantom", "phantom", "label", "smash", "makebox")
_UNWRAP = ("amend", "mathrm", "mathbf", "boldsymbol", "mathbb", "text")
_DROP_BARE = ("Bigl", "Bigr", "bigl", "bigr", "Big", "big", "Bigg", "biggl", "biggr",
              "notag", "nonumber", "displaystyle", "quad", "qquad", "hfill",
              "VertSpace", "HorSpace", "footnotesize", "begingroup", "endgroup")


def _strip_cmd(s, cmd, keep_body):
    """Remove \\cmd{...} (balanced braces); keep or drop the body."""
    out, i, tok = [], 0, "\\" + cmd
    while True:
        k = s.find(tok, i)
        # do not match a longer command name that merely starts with `cmd`
        while k >= 0 and k + len(tok) < len(s) and s[k + len(tok)].isalpha():
            k = s.find(tok, k + 1)
        if k < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:k])
        j = k + len(tok)
        while j < len(s) and s[j] in " \t\n":
            j += 1
        if j >= len(s) or s[j] != '{':
            i = j                                   # command without an argument
            continue
        depth, m = 0, j
        while m < len(s):
            if s[m] == '{':
                depth += 1
            elif s[m] == '}':
                depth -= 1
                if depth == 0:
                    break
            m += 1
        if depth != 0:
            raise ParseError(f"unbalanced braces after \\{cmd}")
        if keep_body:
            out.append(" " + s[j + 1:m] + " ")
        i = m + 1


def normalize(s):
    """Strip layout/markup so only the mathematics is left."""
    s = re.sub(r'(?<!\\)%.*', '', s)
    for c in _UNWRAP:
        s = _strip_cmd(s, c, True)
    for c in _DROP_WITH_ARG:
        s = _strip_cmd(s, c, False)
    s = s.replace(r'\left.', ' ').replace(r'\right.', ' ')
    s = re.sub(r'\\(?:left|right)\s*', ' ', s)
    s = re.sub(r'\\(?:' + '|'.join(_DROP_BARE) + r')(?![A-Za-z])', ' ', s)
    s = re.sub(r'\\[,;:!>]', ' ', s)
    s = s.replace('\\ ', ' ').replace('{}', ' ').replace('&', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s


# ------------------------------------------------------------------- AST nodes
class Num:
    def __init__(self, v):
        self.v = Fraction(v)


class Atom:
    __slots__ = ("kind", "idx", "power")

    def __init__(self, kind, idx=(), power=1):
        self.kind, self.idx, self.power = kind, tuple(idx), power

    def __repr__(self):
        return f"{self.kind}{list(self.idx)}^{self.power}"


class Prod:
    def __init__(self, fs):
        self.fs = fs


class Sum:
    def __init__(self, terms):        # list of (sign, node)
        self.terms = terms


class Deriv:
    def __init__(self, idx, node):
        self.idx, self.node = tuple(idx), node


class Div:
    def __init__(self, num, den):
        self.num, self.den = num, den


# ------------------------------------------------------------------ tokenizer
_IDX = r'[a-z]'
_ATOM_RULES = [
    # second derivatives -- both spellings the appendix uses
    (r'\\partial\s*\^\s*\{?2\}?\s*_\s*\{(' + _IDX + r')(' + _IDX + r')\}', 'D2'),
    (r'\\partial\s*_\s*\{(' + _IDX + r')(' + _IDX + r')\}\s*\^\s*\{?2\}?', 'D2'),
    # a multi-letter \partial subscript WITHOUT ^2 is the 2026-07-29 defect class
    (r'\\partial\s*_\s*\{(' + _IDX + r'{2,})\}(?!\s*\^)', 'DBAD'),
    (r'\\partial\s*_\s*\{?(' + _IDX + r')\}?', 'D1'),
    (r'N\s*\^\s*\{?([abc])\}?', 'N'),
    (r'U\s*_\s*\{?(' + _IDX + r')\}?\s*\^\s*\{?c\}?', 'U'),
    (r'U\s*\^\s*\{?c\}?\s*_\s*\{?(' + _IDX + r')\}?', 'U'),
    (r'P\s*\^\s*\{?c\}?', 'P'),
    (r'\\delta\s*_\s*\{(' + _IDX + r')(' + _IDX + r')\}', 'DELTA'),
    (r'\\sigma\s*_\s*\{(' + _IDX + r')(' + _IDX + r')\}', 'SIGMA'),
    (r'\\alpha\s*\^\s*\{?(\d)\}?', 'ALPHAP'),
    (r'\\alpha', 'ALPHA'),
    (r'\\beta', 'BETA'),
    (r'\\nu\s*\^\s*\{?(\d)\}?', 'NUP'),
    (r'\\nu', 'NU'),
    (r'\\tau\s*_\s*\{?([12])\}?', 'TAU'),
    (r'\\varepsilon\s*\^\s*\{?(\d)\}?', 'EPSP'),
    (r'\\varepsilon', 'EPS'),
    (r'a\s*_\s*\{?(' + _IDX + r')\}?', 'A'),
    (r'\\overline\s*\{\s*f\s*\}\s*_\s*\{?(' + _IDX + r')\}?', 'FBAR'),
    (r'\\overline\s*\{\s*\\phi\s*\}', 'PHIBAR'),
    (r't\s*_\s*\{\s*N\s*,\s*(' + _IDX + r')\}', 'TN'),
    (r'f\s*_\s*\{?(' + _IDX + r')\}?', 'F'),
]
_ATOM_RES = [(re.compile(p), k) for p, k in _ATOM_RULES]
_NUM_RE = re.compile(r'\d+')
_FRAC_RE = re.compile(r'\\frac\s*')


def tokenize(s):
    """(kind, payload) stream.  kind in {'(' ,')' ,'+','-','num','frac','atom'}."""
    toks, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch in ' \t\n':
            i += 1
            continue
        if ch in '()':
            toks.append((ch, None))
            i += 1
            continue
        if ch in '+-':
            toks.append((ch, None))
            i += 1
            continue
        m = _FRAC_RE.match(s, i)
        if m:
            toks.append(('frac', None))
            i = m.end()
            continue
        if ch == '{':
            toks.append(('{', None))
            i += 1
            continue
        if ch == '}':
            toks.append(('}', None))
            i += 1
            continue
        hit = None
        for rex, kind in _ATOM_RES:
            m = rex.match(s, i)
            if m:
                hit = (kind, m)
                break
        if hit:
            kind, m = hit
            if kind == 'DBAD':
                raise ParseError(
                    f"multi-index \\partial subscript without ^2: '{m.group(0)}' "
                    "(App. A writes every second derivative with an explicit ^2)")
            toks.append(('atom', (kind, m.groups())))
            i = m.end()
            continue
        m = _NUM_RE.match(s, i)
        if m:
            toks.append(('num', int(m.group(0))))
            i = m.end()
            continue
        raise ParseError(f"unrecognised token at {i!r}: {s[i:i + 30]!r}")
    return toks


def _mkatom(kind, groups):
    if kind == 'D2':
        return ('deriv', (groups[0], groups[1]))
    if kind == 'D1':
        return ('deriv', (groups[0],))
    if kind == 'N':
        return Atom('N' + groups[0])
    if kind == 'U':
        return Atom('U', (groups[0],))
    if kind == 'P':
        return Atom('P')
    if kind in ('DELTA', 'SIGMA'):
        return Atom(kind, (groups[0], groups[1]))
    if kind == 'ALPHAP':
        return Atom('ALPHA', (), int(groups[0]))
    if kind == 'NUP':
        return Atom('NU', (), int(groups[0]))
    if kind == 'EPSP':
        return Atom('EPS', (), int(groups[0]))
    if kind == 'TAU':
        return Atom('TAU' + groups[0])
    if kind in ('A', 'FBAR', 'F', 'TN'):
        return Atom(kind, (groups[0],))
    return Atom(kind)


# ------------------------------------------------------------------- the parser
_TERMINATORS = ('Na', 'Nb', 'Nc', 'ALPHA', 'BETA')


class _Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse_expr(self):
        terms = []
        sign = 1
        if self.peek()[0] in ('+', '-'):
            sign = -1 if self.next()[0] == '-' else 1
        terms.append((sign, self.parse_term()))
        while self.peek()[0] in ('+', '-'):
            sign = -1 if self.next()[0] == '-' else 1
            terms.append((sign, self.parse_term()))
        return terms[0][1] if len(terms) == 1 and terms[0][0] == 1 else Sum(terms)

    def parse_term(self):
        fs = []
        while True:
            k, _ = self.peek()
            if k is None or k in ('+', '-', ')', '}'):
                break
            fs.append(self.parse_factor())
        if not fs:
            raise ParseError("empty term")
        return fs[0] if len(fs) == 1 else Prod(fs)

    def parse_group(self, open_tok, close_tok):
        k, _ = self.next()
        if k != open_tok:
            raise ParseError(f"expected {open_tok!r}, got {k!r}")
        e = self.parse_expr()
        k, _ = self.next()
        if k != close_tok:
            raise ParseError(f"expected {close_tok!r}, got {k!r}")
        return e

    def parse_factor(self):
        k, payload = self.peek()
        if k == '(':
            return self.parse_group('(', ')')
        if k == 'num':
            self.next()
            return Num(payload)
        if k == 'frac':
            self.next()
            num = self.parse_group('{', '}')
            den = self.parse_group('{', '}')
            return Div(num, den)
        if k == 'atom':
            kind, groups = payload
            node = _mkatom(kind, groups)
            self.next()
            if isinstance(node, tuple) and node[0] == 'deriv':
                return Deriv(node[1], self.parse_operand())
            return node
        raise ParseError(f"unexpected token {k!r}")

    def parse_operand(self):
        """What a \\partial acts on -- see the grammar note at the top of the file."""
        k, payload = self.peek()
        if k == '(':
            return self.parse_group('(', ')')
        fs = []
        while True:
            k, payload = self.peek()
            if k is None or k in ('+', '-', ')', '}'):
                raise ParseError("derivative with no field to act on")
            if k != 'atom':
                raise ParseError(f"derivative operand hit {k!r}")
            kind, groups = payload
            node = _mkatom(kind, groups)
            self.next()
            if isinstance(node, tuple) and node[0] == 'deriv':
                fs.append(Deriv(node[1], self.parse_operand()))
                break
            fs.append(node)
            if node.kind in _TERMINATORS:
                break
        return fs[0] if len(fs) == 1 else Prod(fs)


def parse(latex):
    toks = tokenize(normalize(latex))
    p = _Parser(toks)
    node = p.parse_expr()
    if p.i != len(p.t):
        raise ParseError(f"trailing tokens from {p.t[p.i]!r}")
    return node


# ---------------------------------------------------- distribution to monomials
# A monomial is (Fraction coefficient, tuple of factors); a factor is an Atom or a
# Deriv whose .node is itself a *monomial factor tuple* (already distributed).
def expand(node):
    if isinstance(node, Num):
        return [(node.v, ())]
    if isinstance(node, Atom):
        return [(Fraction(1), (node,))]
    if isinstance(node, Sum):
        out = []
        for sign, sub in node.terms:
            out += [(sign * c, f) for c, f in expand(sub)]
        return out
    if isinstance(node, Prod):
        out = [(Fraction(1), ())]
        for sub in node.fs:
            out = [(c1 * c2, f1 + f2) for c1, f1 in out for c2, f2 in expand(sub)]
        return out
    if isinstance(node, Deriv):
        return [(c, (Deriv(node.idx, f),)) for c, f in expand(node.node)]
    if isinstance(node, Div):
        num, den = expand(node.num), expand(node.den)
        if len(den) != 1:
            raise ParseError("only single-monomial denominators are supported")
        dc, df = den[0]
        if df:                                     # e.g. 1/alpha
            inv = tuple(Atom(a.kind, a.idx, -a.power) for a in df
                        if isinstance(a, Atom))
            if len(inv) != len(df):
                raise ParseError("cannot invert a differentiated denominator")
        else:
            inv = ()
        return [(c / dc, f + inv) for c, f in num]
    raise ParseError(f"cannot expand {node!r}")


def monomial_indices(factors):
    """Every index letter occurring in a monomial, with multiplicity."""
    out = []
    for f in factors:
        if isinstance(f, Deriv):
            out += list(f.idx) + monomial_indices(f.node if isinstance(f.node, tuple)
                                                  else (f.node,))
        else:
            out += list(f.idx)
    return out


def trial_factors(factors):
    """Count the nodal trial coefficients (U_k^c / P^c) inside one monomial."""
    n = 0
    for f in factors:
        if isinstance(f, Deriv):
            n += trial_factors(f.node if isinstance(f.node, tuple) else (f.node,))
        elif f.kind in ('U', 'P'):
            n += 1
    return n


def census(node, free):
    """Einstein-convention audit of every monomial after full distribution.

    Returns (n_monomials, list_of_violation_strings).  `free` is the set of index
    letters the display DECLARES free (read off its left-hand side).

    Distribution first is the whole point: a display row holds several additive
    terms and often a nested sum inside a product, so a letter legitimately summed
    inside each branch of  (d_k N^c U_m^c + d_m N^c U_k^c)  is counted twice at row
    level.  Per-row and per-summand censuses were both measured and rejected for
    exactly this reason (52/68 rows and 22 false positives respectively).
    """
    free = set(free)
    viol, monos = [], expand(node)
    for coeff, factors in monos:
        counts = {}
        for L in monomial_indices(factors):
            counts[L] = counts.get(L, 0) + 1
        for L, n in sorted(counts.items()):
            if L in free:
                if n != 1:
                    viol.append(f"free index '{L}' occurs {n} times in one monomial "
                                f"(a free index must occur exactly once)")
            elif n == 1:
                viol.append(f"index '{L}' occurs once but is not a declared free "
                            f"index {sorted(free)} (dangling index)")
            elif n > 2:
                viol.append(f"index '{L}' occurs {n} times in one monomial "
                            f"(re-used dummy; the convention allows 1 free or 2 summed)")
    return len(monos), viol


# -------------------------------------------------------------------- evaluation
def _atom_value(a, env, mode):
    p = a.power
    if a.kind == 'Na':
        return NA
    if a.kind == 'Nb':
        return NB
    if a.kind == 'Nc':
        return NB if mode else NC
    if a.kind == 'U':
        k = env[a.idx[0]]
        return (sp.Integer(1) if k == mode[1] else sp.Integer(0)) if mode else UDOF[k]
    if a.kind == 'P':
        return sp.Integer(1) if mode else PDOF
    if a.kind == 'DELTA':
        return sp.Integer(1) if env[a.idx[0]] == env[a.idx[1]] else sp.Integer(0)
    if a.kind == 'SIGMA':
        return SIG[env[a.idx[0]], env[a.idx[1]]]
    if a.kind == 'ALPHA':
        return ALPHA ** p
    if a.kind == 'BETA':
        return sp.log(ALPHA)
    if a.kind == 'NU':
        return NU ** p
    if a.kind == 'EPS':
        return EPS ** p
    if a.kind == 'TAU1':
        return TAU1
    if a.kind == 'TAU2':
        return TAU2
    if a.kind == 'A':
        return AVEC[env[a.idx[0]]]
    if a.kind == 'FBAR':
        return FBAR[env[a.idx[0]]]
    if a.kind == 'F':
        return FVEC[env[a.idx[0]]]
    if a.kind == 'TN':
        return TN[env[a.idx[0]]]
    if a.kind == 'PHIBAR':
        return PHIBAR
    raise ParseError(f"no value for atom {a.kind}")


def _mono_value(factors, env, mode):
    val = sp.Integer(1)
    for f in factors:
        if isinstance(f, Deriv):
            sub = _mono_value(f.node if isinstance(f.node, tuple) else (f.node,), env, mode)
            for L in f.idx:
                sub = sp.diff(sub, X[env[L]])
            val *= sub
        else:
            val *= _atom_value(f, env, mode)
        if val == 0:
            return sp.Integer(0)
    return val


def evaluate(node, free_values, d=3, mode=None):
    """Sum the parsed expression over its dummy indices.

    free_values : {letter: 0..d-1} for the display's free indices.
    mode        : None -> literal value (trial field kept as N^c U_k^c);
                  ('U', j) -> the Gateaux derivative d/dU_j^b (N^c -> N^b, U_k -> delta_kj);
                  ('P', None) -> the Gateaux derivative d/dP^b (N^c -> N^b, P^c -> 1).
    """
    total = sp.Integer(0)
    for coeff, factors in expand(node):
        letters = sorted(set(monomial_indices(factors)) - set(free_values))
        for combo in _tuples(len(letters), d):
            env = dict(free_values)
            env.update(dict(zip(letters, combo)))
            total += sp.Rational(coeff.numerator, coeff.denominator) * _mono_value(factors, env, mode)
    return total


def _tuples(n, d):
    if n == 0:
        yield ()
        return
    for head in range(d):
        for rest in _tuples(n - 1, d):
            yield (head,) + rest


def zero(expr):
    """Robust `is this identically zero` for the polynomial-in-derivatives forms here."""
    e = sp.expand(expr)
    if e == 0:
        return True
    return sp.simplify(sp.together(e)) == 0
