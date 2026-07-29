#!/usr/bin/env python3
# =============================================================================
# elemental_matrices_verification.py
#
# Symbolic (sympy) verification of the elemental matrix contributions of the
# Appendix "Elemental matrices" (eq:GSComponents ... eq:GLHSStabilizationTerm),
#   "A stabilized finite element method ... porous media".
#
# Each elemental component is, per eq:GenericMatrixVU/QU/VP/QP,
#     T_{(ai)(bj)} = d T( N^a delta_ik, U_k^c N^c ) / d U_j^b
# (and analogues for the pressure / mass blocks). For every term below we
#   (a) transcribe the *integrand* T (the expression inside the derivative)
#       from the appendix, written with real shape functions N^a, N^b and the
#       trial fields u, p, and a varying porosity alpha(x);
#   (b) take the Gateaux derivative w.r.t. the trial DOF (b,j) [resp. b];
#   (c) check that it equals the *printed result* of the appendix.
#
# The Gateaux derivative is exact because every left-hand-side term is linear
# in the trial unknown: d/dU_j^b just replaces the trial component k by
# delta_kj N^b (and its derivatives). Second derivatives are handled by genuine
# symbolic differentiation, so the d^2_{lm}=d^2_{ml} symmetry is automatic.
#
# Run:  python3 elemental_matrices_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok
def kron(p, q): return 1 if p == q else 0

x = sp.symbols('x0 x1 x2', real=True)
def d(f, i): return sp.diff(f, x[i])

# Shape functions (real functions => exact, symmetric derivatives)
Na = sp.Function('Na')(*x)
Nb = sp.Function('Nb')(*x)
# Trial fields
u = [sp.Function(f'u{k}')(*x) for k in range(3)]     # velocity trial
pp = sp.Function('pp')(*x)                            # pressure trial
# Porosity and physical parameters
alpha = sp.Function('alpha', positive=True)(*x)
beta = sp.log(alpha)                                  # beta = log alpha (d_l beta = d_l alpha/alpha)
nu, eps, tau1, tau2 = sp.symbols('nu varepsilon tau1 tau2', positive=True)
a = sp.symbols('a0 a1 a2', real=True)                 # advection a_l (frozen, constant)
sig = sp.Matrix(3, 3, lambda p, q: sp.Symbol(f'sig{min(p,q)}{max(p,q)}', real=True))  # symmetric

I3 = range(3)

def Dvel(integrand, j):
    """Gateaux derivative w.r.t. velocity DOF (b,j): trial u_k -> delta_kj N^b."""
    sub = {u[k]: (Nb if k == j else sp.Integer(0)) for k in I3}
    return sp.expand(integrand.xreplace(sub).doit())

def Dpre(integrand):
    """Gateaux derivative w.r.t. pressure DOF b: trial p -> N^b."""
    return sp.expand(integrand.xreplace({pp: Nb}).doit())

def verify_vel(name, integrand_of_i, formula_of_ij):
    ok = all(sp.simplify(Dvel(integrand_of_i(i), j) - formula_of_ij(i, j)) == 0
             for i in I3 for j in I3)
    check(name, ok)

def verify_mass(name, integrand, formula_of_j):   # mass-row test (scalar N^a), trial velocity
    ok = all(sp.simplify(Dvel(integrand, j) - formula_of_j(j)) == 0 for j in I3)
    check(name, ok)

def verify_velP(name, integrand_of_i, formula_of_i):   # velocity test i, pressure trial
    ok = all(sp.simplify(Dpre(integrand_of_i(i)) - formula_of_i(i)) == 0 for i in I3)
    check(name, ok)

print("=" * 70)
print("Elemental matrix contributions (Appendix) -- symbolic checks")
print("=" * 70)

# -------------------------------------------------------------------------
# Galerkin block K_{V,U}  (test velocity comp. i, trial velocity comp. j)
# -------------------------------------------------------------------------
print("\n[K_VU] Galerkin velocity-momentum terms")
# G_S  (eq:GSComponents): alpha nu (d_l u_i + d_i u_l) d_l N^a   ->  alpha nu(delta_ij d_l N^a d_l N^b + d_j N^a d_i N^b)
verify_vel("G_S  = alpha nu (d_l N^a d_l N^b delta_ij + d_j N^a d_i N^b)",
           lambda i: nu*alpha*sum((d(u[i], l) + d(u[l], i))*d(Na, l) for l in I3),
           lambda i, j: nu*alpha*(kron(i, j)*sum(d(Na, l)*d(Nb, l) for l in I3) + d(Na, j)*d(Nb, i)))
# D_nuD (eq:DDComponents): -2/3 alpha nu d_i N^a (div u)  ->  -2/3 alpha nu d_i N^a d_j N^b
verify_vel("D_nuD = -2/3 alpha nu d_i N^a d_j N^b",
           lambda i: -sp.Rational(2, 3)*nu*alpha*d(Na, i)*sum(d(u[m], m) for m in I3),
           lambda i, j: -sp.Rational(2, 3)*nu*alpha*d(Na, i)*d(Nb, j))
# V (eq:VComponents): alpha N^a a_l d_l u_i  ->  alpha N^a a_l d_l N^b delta_ij
verify_vel("V    = alpha N^a a_l d_l N^b delta_ij  (convection)",
           lambda i: alpha*Na*sum(a[l]*d(u[i], l) for l in I3),
           lambda i, j: alpha*Na*sum(a[l]*d(Nb, l) for l in I3)*kron(i, j))
# R_sigma (eq:ReactionTermComponents): N^a sigma_ik u_k  ->  N^a N^b sigma_ij
verify_vel("R_sigma = N^a N^b sigma_ij  (reaction)",
           lambda i: Na*sum(sig[i, k]*u[k] for k in I3),
           lambda i, j: Na*Nb*sig[i, j])

# -------------------------------------------------------------------------
# Galerkin block K_{Q,U}  (mass test, trial velocity comp. j)
# -------------------------------------------------------------------------
print("\n[K_QU] Galerkin mass-velocity terms")
# Q_D (eq:QComponents): alpha N^a div u  ->  alpha N^a d_j N^b
verify_mass("Q_D    = alpha N^a d_j N^b  (alpha div u)",
            alpha*Na*sum(d(u[l], l) for l in I3),
            lambda j: alpha*Na*d(Nb, j))
# G_alphaD (eq:GAlphaComponents): N^a d_k alpha u_k  ->  N^a N^b d_j alpha
verify_mass("G_alphaD = N^a N^b d_j alpha  (u . grad alpha)",
            Na*sum(d(alpha, k)*u[k] for k in I3),
            lambda j: Na*Nb*d(alpha, j))

# -------------------------------------------------------------------------
# Galerkin block K_{V,P}  (velocity test i, pressure trial)  and  K_{Q,P}
# -------------------------------------------------------------------------
print("\n[K_VP / K_QP] Galerkin pressure terms")
# P (eq:PComponents): -alpha p d_i N^a  ->  -alpha d_i N^a N^b
verify_velP("P    = -alpha d_i N^a N^b  (pressure gradient)",
            lambda i: -alpha*pp*d(Na, i),
            lambda i: -alpha*d(Na, i)*Nb)
# G_P (eq:PComponents): -d_i alpha N^a p  ->  -d_i alpha N^a N^b
verify_velP("G_P  = -d_i alpha N^a N^b",
            lambda i: -d(alpha, i)*Na*pp,
            lambda i: -d(alpha, i)*Na*Nb)
# P_Q (eq:PQLHS): eps N^a p  ->  eps N^a N^b
check("P_Q  = eps N^a N^b  (compressibility)",
      sp.simplify(Dpre(eps*Na*pp) - eps*Na*Nb) == 0)

# -------------------------------------------------------------------------
# Representative stabilization terms  (-sum_K <L* V, tau L U>), eq:StabilizationLVLU
#   adjoint test operator pieces:  A = a.grad,  L = nu Lap-bar,  reaction = sigma
# -------------------------------------------------------------------------
print("\n[K_VU stab] Representative stabilization terms")
# A_A (eq:AALHSStabilizationTerm): tau1 alpha^2 (a_l d_l N^a)(a_m d_m u_i)
#   -> tau1 alpha^2 a_l d_l N^a a_m d_m N^b delta_ij
verify_vel("A_A   = tau1 alpha^2 (a.grad N^a)(a.grad N^b) delta_ij",
           lambda i: tau1*alpha**2*(sum(a[l]*d(Na, l) for l in I3))*(sum(a[m]*d(u[i], m) for m in I3)),
           lambda i, j: tau1*alpha**2*sum(a[l]*d(Na, l) for l in I3)*sum(a[m]*d(Nb, m) for m in I3)*kron(i, j))
# A_sigma (eq:ASigmaLHSStabilizationTerm): tau1 alpha (a_l d_l N^a) sigma_ik u_k
#   -> tau1 alpha (a.grad N^a) N^b sigma_ij
verify_vel("A_sigma = tau1 alpha (a.grad N^a) N^b sigma_ij",
           lambda i: tau1*alpha*sum(a[l]*d(Na, l) for l in I3)*sum(sig[i, k]*u[k] for k in I3),
           lambda i, j: tau1*alpha*sum(a[l]*d(Na, l) for l in I3)*Nb*sig[i, j])
# R_Rsigma (eq:RRSigmaLHSStabilizationTerm): -tau1 sigma_ik N^a sigma_km u_m
#   -> -tau1 sigma_ik N^a sigma_kj N^b
verify_vel("R_Rsigma = -tau1 sigma_ik N^a sigma_kj N^b",
           lambda i: -tau1*Na*sum(sig[i, k]*sum(sig[k, m]*u[m] for m in I3) for k in I3),
           lambda i, j: -tau1*sum(sig[i, k]*sig[k, j] for k in I3)*Na*Nb)
# A_L (eq:ALLHSStabilizationTerm): -tau1 alpha^2 nu (a.grad N^a)(Lap u_i + d_i div u)
#   -> -tau1 alpha^2 nu (a.grad N^a)(d^2_mm N^b delta_ij + d^2_ij N^b)
verify_vel("A_L   = -tau1 alpha^2 nu (a.grad N^a)(Lap N^b delta_ij + d^2_ij N^b)",
           lambda i: -tau1*alpha**2*nu*sum(a[l]*d(Na, l) for l in I3)
                     * (sum(d(d(u[i], m), m) for m in I3) + sum(d(d(u[m], i), m) for m in I3)),
           lambda i, j: -tau1*alpha**2*nu*sum(a[l]*d(Na, l) for l in I3)
                     * (sum(d(d(Nb, m), m) for m in I3)*kron(i, j) + d(d(Nb, i), j)))

print("\n[K_QU stab] Representative mass-row stabilization terms")
# G_A (eq:GALHSStabilizationTerm): tau1 alpha^2 (d_k N^a)(a_l d_l u_k) -> tau1 alpha^2 d_j N^a (a.grad N^b)
verify_mass("G_A  = tau1 alpha^2 d_j N^a (a.grad N^b)",
            tau1*alpha**2*sum(d(Na, k)*sum(a[l]*d(u[k], l) for l in I3) for k in I3),
            lambda j: tau1*alpha**2*d(Na, j)*sum(a[l]*d(Nb, l) for l in I3))
# G_R (eq:GRLHSStabilizationTerm): tau1 alpha (d_k N^a) sigma_kl u_l -> tau1 alpha (d_k N^a) sigma_kj N^b
verify_mass("G_R  = tau1 alpha d_k N^a sigma_kj N^b",
            tau1*alpha*sum(d(Na, k)*sum(sig[k, l]*u[l] for l in I3) for k in I3),
            lambda j: tau1*alpha*sum(d(Na, k)*sig[k, j] for k in I3)*Nb)
# D_D (eq:DAlphaDLHSStabilizationTerm): tau2 alpha^2 d_i N^a div u -> tau2 alpha^2 d_i N^a d_j N^b
verify_vel("D_D   = tau2 alpha^2 d_i N^a d_j N^b  (tau2 block)",
           lambda i: tau2*alpha**2*d(Na, i)*sum(d(u[m], m) for m in I3),
           lambda i, j: tau2*alpha**2*d(Na, i)*d(Nb, j))

print("\n[K_VP / K_QP stab] Representative pressure-column stabilization terms")
# A_G (eq:AGLHSStabilizationTerm): tau1 alpha^2 (a.grad N^a)(d_i p) -> tau1 alpha^2 (a.grad N^a) d_i N^b
verify_velP("A_G  = tau1 alpha^2 (a.grad N^a) d_i N^b",
            lambda i: tau1*alpha**2*sum(a[l]*d(Na, l) for l in I3)*d(pp, i),
            lambda i: tau1*alpha**2*sum(a[l]*d(Na, l) for l in I3)*d(Nb, i))
# G (eq:GLHSStabilizationTerm, K_QP): tau1 alpha^2 d_k N^a d_k p -> tau1 alpha^2 d_k N^a d_k N^b
check("G    = tau1 alpha^2 d_k N^a d_k N^b  (pressure Laplacian, K_QP)",
      sp.simplify(Dpre(tau1*alpha**2*sum(d(Na, k)*d(pp, k) for k in I3))
                  - tau1*alpha**2*sum(d(Na, k)*d(Nb, k) for k in I3)) == 0)
# Q_P (eq:QPLHSStabilizationTerm): -tau2 eps^2 N^a p -> -tau2 eps^2 N^a N^b
check("Q_P  = -tau2 eps^2 N^a N^b",
      sp.simplify(Dpre(-tau2*eps**2*Na*pp) - (-tau2*eps**2*Na*Nb)) == 0)


# =========================================================================
# SOURCE LINT OF THE PRINTED DERIVATION  (added 2026-07-29)
#
# Everything above re-derives the elemental matrices from THIS script's own
# encoding and never opens the appendix.  That is a structural blind spot: the
# endpoint is certified while the printed PATH to it goes unread, so a typo in an
# intermediate display of App. A survives every symbolic check in the suite.  On
# 2026-07-29 a whole-paper referee read found exactly that -- "\partial_{ik} N^c
# U_m^c" inside eq:AGBetaLHSStabilizationTerm, where the symmetric-gradient pair
# requires "\partial_k N^c U_m^c" (App. A prints it correctly in five siblings,
# e.g. eq:LGBetaLHSStabilizationTerm, which carries the identical sub-expression).
#
# The two rules below are deliberately NARROW.  A broader "index balance" lint was
# prototyped and REJECTED: counting index occurrences per display row conflates
# independent additive terms and flagged 52 of 68 rows, i.e. it had no
# discriminating power.  A gate that cries wolf is a gate people learn to ignore.
# These rules key on a convention App. A follows without exception, so their false
# positive rate on the current source is zero, and each ships with a negative
# control built from the pre-fix text.
# =========================================================================
print("\n" + "=" * 70)
print("SOURCE LINT OF APPENDIX A's PRINTED DERIVATION")
print("=" * 70)

import os as _os
import re as _re

_PAPER = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                         "..", "..", "theory", "paper"))
with open(_os.path.join(_PAPER, "elemental_matrices_appendix.tex"), encoding="utf-8") as _f:
    _APPA = _f.read()

# S1  Second-derivative convention: App. A writes every second derivative with an
# explicit ^2 (\partial^2_{kl} or \partial_{kl}^2).  A multi-letter \partial subscript
# WITHOUT it is a first derivative that has absorbed a neighbouring index.
_MULTI = _re.compile(r"\\partial(\^2)?_\{([a-z]{2,})\}(\^2)?")
_multi = _MULTI.findall(_APPA)
_bad1 = [m for m in _multi if not (m[0] or m[2])]
check(f"S1 every multi-index \\partial subscript in App. A carries an explicit ^2 "
      f"({len(_multi)} inspected, {len(_bad1)} missing it)",
      len(_multi) > 0 and not _bad1)

# S2  The symmetric-gradient pair is always printed as
# (\partial_k N^c U_m^c + \partial_m N^c U_k^c).  The derivative index must be a SINGLE
# letter.  The pattern must accept BOTH \partial_k and \partial_{ik}: the braced form is
# what the 2026-07-29 typo looked like, so a rule blind to it cannot guard against it.
_DERIV = r"\\partial_(?:\{([a-z]+)\}|([a-z]))"
_PAIR = _re.compile(_DERIV + r" N\^c U_\{?([a-z])\}?\^c\s*\+\s*"
                    + _DERIV + r" N\^c U_\{?([a-z])\}?\^c")


def _pair_deriv_indices(m):
    """The two derivative indices of one matched symmetric-gradient pair."""
    return (m.group(1) or m.group(2), m.group(4) or m.group(5))


_pairs = list(_PAIR.finditer(_APPA))
_bad2 = [m for m in _pairs if any(len(x) != 1 for x in _pair_deriv_indices(m))]
check(f"S2 every symmetric-gradient pair in App. A uses single-index derivatives "
      f"({len(_pairs)} pairs inspected, {len(_bad2)} malformed)",
      len(_pairs) > 0 and not _bad2)

# Discriminating negatives, built from the text exactly as it stood before the fix.
_PREFIX = (r"\delta_{ik} \left( \partial_{ik} N^c U_m^c "
           r"+ \partial_m N^c U_k^c \right) \partial_m \beta")
check("negative: S1 rejects the pre-fix \\partial_{ik} (multi-index, no ^2)",
      any(not (m[0] or m[2]) for m in _MULTI.findall(_PREFIX)))
check("negative: S2 rejects the pre-fix symmetric-gradient pair",
      any(len(x) != 1 for m in _PAIR.finditer(_PREFIX)
          for x in _pair_deriv_indices(m)))


# =========================================================================
# READING APPENDIX A's PRINTED DERIVATION  (added 2026-07-29)
#
# The checks at the top of this file certify App. A's printed RESULTS from this
# script's own sympy encoding.  S1/S2 above lint two textual conventions.  Neither
# reads the ~67 pre-differentiation INTEGRANDS that App. A prints inside
#     T_{(ai)(bj)} = <prefactor> d/dU_j^b ( <integrand> ) = <printed result>,
# so the printed PATH to the certified destination was verified by nothing.  Nine
# index defects lived there while the suite was green at 454/454 (BS-2 in
# docs/lessons_learned.md, 2026-07-29).
#
# This section closes that gap by PARSING the appendix rather than transcribing
# it.  Transcription would not be honest closure: the transcriber is the same
# reader who must notice the typo, and the characteristic defect -- a trial factor
# silently re-using a test dummy index -- is exactly the kind a human eye repairs
# without registering that it did.  Parsing removes the reader from the loop:
# BOTH the intermediate and the printed result come from the .tex, so the check
# holds the paper to its own claim.
#
# Two independent arms, either of which can fail alone:
#
#   P2 INDEX CENSUS.  Distribute every product over every (nested) sum, then
#      require each monomial to obey the Einstein convention exactly: a declared
#      free index occurs once, every other letter occurs twice.  Distribution is
#      what the two cheap lints measured and rejected in the spec could not do
#      (52/68 rows flagged; 22 false positives) -- see
#      docs/appendix-a-intermediate-coverage-spec.md Sec. 2.
#   P4 DIFFERENTIATION.  Differentiate the parsed integrand w.r.t. the nodal
#      unknown and compare, index by index, against the parsed printed result.
#
# The two arms catch different things, and the negative controls below prove it:
# the nine HISTORICAL defects (recovered verbatim from the pre-fix source, commit
# fdc7507) are all caught by P1/P2, while five index-BALANCED mutations -- which
# no census can see -- are caught only by P4.
# =========================================================================
print("\n" + "=" * 70)
print("READING APPENDIX A's PRINTED DERIVATION (parse -> census -> differentiate)")
print("=" * 70)

import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import latex_index_notation as _lin

_ENV = _re.compile(r'\\begin\{(align|equation)\*?\}(.*?)\\end\{\1\*?\}', _re.S)
_ROWSPLIT = _re.compile(r'\\\\\s*(?:\[[^\]]*\])?')
_DERIVOP = _re.compile(r'\\frac\s*\{\s*\\partial\s*\}\s*\{\s*\\partial\s*'
                       r'(U\s*_\s*\{?j\}?\s*\^\s*\{?b\}?|P\s*\^\s*\{?b\}?)\s*\}')
# Which spatial indices each block's left-hand side DECLARES free, read off the
# subscript pattern of the printed component name (e.g. "A_{A\, (ai)(bj)}").
# The differentiation argument never carries j: j enters through d/dU_j^b.
_BLOCKS = [(r'\(ai\)\(bj\)', 'VU'), (r'a\(bj\)', 'QU'), (r'\(ai\)b', 'VP'),
           (r'\(ia\)', 'FV'), (r'ab', 'QP')]
_ARGFREE = {'VU': ['i'], 'QU': [], 'VP': ['i'], 'QP': []}
_RESFREE = {'VU': ['i', 'j'], 'QU': ['j'], 'VP': ['i'], 'QP': [], 'FV': ['i'], 'FQ': []}


def _logical_rows(region):
    """App. A's align rows, joined across the \\notag continuations."""
    out = []
    for m in _ENV.finditer(region):
        body, off = m.group(2), m.start(2)
        phys, pos = [], 0
        for rm in _ROWSPLIT.finditer(body):
            phys.append((body[pos:rm.start()], off + pos))
            pos = rm.end()
        phys.append((body[pos:], off + pos))
        cur, curoff = [], None
        for txt, o in phys:
            if not txt.strip():
                continue
            if curoff is None:
                curoff = o
            cur.append(txt)
            if r'\notag' not in txt and r'\nonumber' not in txt:
                out.append((" ".join(cur), curoff))
                cur, curoff = [], None
        if cur:
            out.append((" ".join(cur), curoff))
    return out


_i0 = _APPA.index(r'\subsection{Components of $\mathbf{K}_{V, U}$}')
_i1 = _APPA.index('Putting together the results')
_REGION = _APPA[_i0:_i1]
_BASELINE = _APPA[:_i0].count('\n') + 1

_rows = _logical_rows(_REGION)
_displays, _templates = [], 0
for _raw, _off in _rows:
    _ln = _BASELINE + _REGION[:_off].count('\n')
    _m = _re.search(r'\\label\{([^}]*)\}', _raw)
    _lab = _m.group(1) if _m else '(unnumbered)'
    _norm = _lin.normalize(_raw)
    _parts = [_re.sub(r'[,.;]+$', '', p.strip()).strip() for p in _norm.split('=')]
    _parts = [p for p in _parts if p]
    if _re.match(r'T\s*_', _parts[0]):        # the generic T(...) templates
        _templates += 1
        continue
    _blk = next((b for pat, b in _BLOCKS if _re.search(pat, _parts[0])), 'FQ')
    _displays.append((_ln, _lab, _blk, _parts, _DERIVOP.search(_norm) is not None))

# P0  Extraction must actually have found the appendix.  A rule whose input silently
# empties is the failure mode this suite has been bitten by twice (2026-07-28 Rule 1,
# 2026-07-29 D3): the counts below are asserted, not merely printed.
_nderiv = sum(1 for *_r, hasd in _displays if hasd)
check(f"P0 App. A component displays extracted "
      f"({len(_displays)} displays, {_nderiv} with a d/dU or d/dP operator, "
      f"{_templates} generic templates skipped)",
      len(_displays) >= 79 and _nderiv >= 67 and _templates == 6)


def _split_deriv(part):
    k = _DERIVOP.search(part)
    return part[:k.start()], part[k.end():], ('P' if k.group(1).startswith('P') else 'U')


_nparsed = _nmono = _ndiff = _nchain = 0
_nonzero_seen = 0
for _ln, _lab, _blk, _parts, _hasd in _displays:
    _name = f"A.{_lab} (l.{_ln})"
    try:
        if _hasd:
            _pre, _arg, _wrt = _split_deriv(_parts[1])
            # `1` keeps a bare leading sign in the prefactor grammatical
            _inter = _lin.parse(f"{_pre} 1 ( {_arg} )")
            _res = _lin.parse(_parts[-1])
            _nparsed += 2
            _af, _rf = _ARGFREE[_blk], _RESFREE[_blk]
            _n1, _v1 = _lin.census(_inter, _af)
            _n2, _v2 = _lin.census(_res, _rf)
            _nmono += _n1 + _n2
            if _v1 or _v2:
                check(f"{_name} index census", False)
                print(f"        argument: {_v1}")
                print(f"        result  : {_v2}")
                continue
            # every monomial of a differentiation argument carries exactly one
            # nodal trial coefficient -- an argument that is already differentiated
            # (or has no unknown in it) is the ':160' defect class
            _tc = {_lin.trial_factors(f) for _, f in _lin.expand(_inter)}
            if _tc != {1}:
                check(f"{_name} argument is linear in exactly one nodal unknown", False)
                print(f"        trial-coefficient counts per monomial: {sorted(_tc)}")
                continue
            _ok, _nz = True, False
            for _i in ([0, 1, 2] if 'i' in _rf else [None]):
                for _j in ([0, 1, 2] if 'j' in _rf else [None]):
                    _fa = {'i': _i} if 'i' in _af else {}
                    _fr = {}
                    if 'i' in _rf:
                        _fr['i'] = _i
                    if 'j' in _rf:
                        _fr['j'] = _j
                    _lv = _lin.evaluate(_inter, _fa, mode=(_wrt, _j) if _wrt == 'U' else ('P', None))
                    _rv = _lin.evaluate(_res, _fr)
                    _nz = _nz or _lv != 0
                    if not _lin.zero(_lv - _rv):
                        _ok = False
                        print(f"        i={_i} j={_j}")
                        print(f"        d/d{_wrt} of the printed intermediate: {sp.simplify(_lv)}")
                        print(f"        printed result                      : {sp.simplify(_rv)}")
                        break
                if not _ok:
                    break
            _ndiff += 1
            _nonzero_seen += 1 if _nz else 0
            check(f"{_name} printed intermediate --d/d{_wrt}--> printed result", _ok and _nz)
        else:
            _rf = _RESFREE[_blk]
            _chain = [_lin.parse(p) for p in _parts[1:]]
            _nparsed += len(_chain)
            _viol = []
            for _c in _chain:
                _n, _v = _lin.census(_c, _rf)
                _nmono += _n
                _viol += _v
            if _viol:
                check(f"{_name} index census", False)
                print(f"        {_viol}")
                continue
            _ok, _nz = True, False
            for _i in ([0, 1, 2] if 'i' in _rf else [None]):
                _fv = {'i': _i} if 'i' in _rf else {}
                _vals = [_lin.evaluate(_c, _fv) for _c in _chain]
                _nz = _nz or any(v != 0 for v in _vals)
                if any(not _lin.zero(_vals[0] - v) for v in _vals[1:]):
                    _ok = False
            if len(_chain) > 1:
                _nchain += 1
            check(f"{_name} index census"
                  + (" + printed chain is self-consistent" if len(_chain) > 1 else ""),
                  _ok and _nz)
    except Exception as _e:                                   # noqa: BLE001
        check(f"{_name} parses under App. A's own grammar", False)
        print(f"        {type(_e).__name__}: {_e}")

check(f"P1/P2/P4 counters are non-vacuous "
      f"({_nparsed} expressions parsed, {_nmono} monomials censused, "
      f"{_ndiff} differentiation checks, {_nchain} printed chains, "
      f"{_nonzero_seen} intermediates verified not identically zero)",
      _nparsed >= 140 and _nmono >= 200 and _ndiff >= 67 and _nchain >= 1
      and _nonzero_seen == _ndiff)

# -------------------------------------------------------------------------
# NEGATIVE CONTROLS I -- the nine HISTORICAL defects, verbatim from the '-' side
# of commit fdc7507's hunks.  These are the real pre-fix intermediates, not
# synthetic look-alikes: if the gate does not reject exactly the text the paper
# used to carry, it does not close the class it claims to close.
# -------------------------------------------------------------------------
print("\n[negative controls] the nine 2026-07-29 defects, as they were printed")
_HISTORICAL = [
    ("eq:AGBetaLHSStabilizationTerm",
     r"a_l \partial_l N^a \delta_{ik} \left( \partial_{ik} N^c U_m^c "
     r"+ \partial_m N^c U_k^c \right) \partial_m \beta"),
    ("eq:ASigmaLHSStabilizationTerm",
     r"a_l \partial_l N^a \delta_{ik} \sigma_{kl} U_l^c N^c"),
    ("eq:LDBetaLHSStabilizationTerm",
     r"\partial_l \left( \partial_l N^a \delta_{ik} + \partial_k N^a \delta_{il} \right) "
     r"\partial_m N^b \delta_{mj} \partial_k \beta"),
    ("eq:LSigmaLHSStabilizationTerm",
     r"\partial_l \left( \partial_l N^a \delta_{ik} + \partial_k N^a \delta_{il} \right) "
     r"\sigma_{kl} U_l^c N^c"),
    ("eq:DBetaLLHSStabilizationTerm",
     r"\partial_l N^a \delta_{ik} \partial_k \beta \partial_m "
     r"\left( \partial_m N^c U_k^c +\partial_k N^c U_m^c \right)"),
    ("eq:DBetaGLHSStabilizationTerm",
     r"\partial_l N^a \delta_{il} \partial_k \beta "
     r"\left( \partial_l N^c U_k^c + \partial_k N^c U_l^c \right) \partial_l \beta"),
    ("eq:RSigmaLLHSStabilizationTerm",
     r"\sigma_{lk} N^a \delta_{il} \partial_l "
     r"\left( \partial_l N^c U_k^c + \partial_k N^c U_l^c \right)"),
    ("eq:GULHSStabilizationTerm",
     r"N^a \delta_{il} \partial_l \beta U_l^c N^c \partial_l \beta"),
    ("eq:GAlphaDLHSStabilizationTerm",
     r"N^a \delta_{il} \partial_l \beta \partial_l N^c U_l^c"),
]


def _rejects(arg, free=('i',)):
    """Would the structural arm (P1 grammar / P2 census / trial-count) reject this?"""
    try:
        node = _lin.parse(arg)
    except _lin.ParseError as e:
        return f"grammar: {e}"
    n, viol = _lin.census(node, list(free))
    if viol:
        return f"census: {viol[0]}"
    if {_lin.trial_factors(f) for _, f in _lin.expand(node)} != {1}:
        return "argument is not linear in exactly one nodal unknown"
    return None


for _lab, _arg in _HISTORICAL:
    _why = _rejects(_arg)
    check(f"negative: rejects the pre-fix {_lab}"
          + (f"  [{_why[:72]}]" if _why else ""), _why is not None)

# -------------------------------------------------------------------------
# NEGATIVE CONTROLS II -- index-BALANCED mutations.  Every one of these satisfies
# the Einstein convention perfectly, so the census arm passes them; only the
# differentiation arm can tell they are wrong.  Without these the suite could not
# distinguish "P4 verifies the algebra" from "P4 re-states the census".
# -------------------------------------------------------------------------
print("\n[negative controls] index-balanced mutations (census-invisible, P4 only)")
_MUTANTS = [
    ("A_A: convective trial index moved onto the test factor", 'VU', 'U',
     r"\tau_1 \alpha^2", r"a_l \partial_l N^a \delta_{ik} a_m \partial_k N^c U_m^c",
     r"\tau_1 \alpha^2 a_l \partial_l N^a a_m \partial_m N^b \delta_{ij}"),
    ("A_L: sign flip inside the symmetric-gradient pair", 'VU', 'U',
     r"- \tau_1 \alpha^2 \nu",
     r"a_l \partial_l N^a \delta_{ik} \partial_m \left(\partial_m N^c U_k^c "
     r"- \partial_k N^c U_m^c \right)",
     r"- \tau_1 \alpha^2 \nu a_l \partial_l N^a \left( \partial^2_{mm}N^b \delta_{ij} "
     r"+ \partial^2_{ij} N^b \right)"),
    ("G_S: symmetric-gradient pair with a repeated branch", 'VU', 'U',
     r"\alpha \nu",
     r"\left( \partial_l N^c U_k^c + \partial_l N^c U_k^c \right) \partial_l N^a \delta_{ik}",
     r"\alpha \nu \left( \partial_l N^a \partial_l N^b \delta_{ij} "
     r"+ \partial_j N^a \partial_i N^b \right)"),
    ("C_C: second-derivative contraction rewired", 'VU', 'U',
     r"- \frac{4}{9} \tau_1 \alpha^2 \nu^2",
     r"\partial^2_{kl} N^a \delta_{il} \partial^2_{mm} N^c U_k^c",
     r"- \frac{4}{9} \tau_1 \alpha^2 \nu^2 \partial^2_{ki} N^a \partial^2_{kj} N^b"),
    ("A_C: one power of alpha dropped from the prefactor", 'VU', 'U',
     r"\frac{2}{3} \tau_1 \alpha \nu",
     r"a_l \partial_l N^a \delta_{ik} \partial^2_{km} U_m^c N^c",
     r"\frac{2}{3} \tau_1 \alpha^2 \nu a_l \partial_l N^a \partial^2_{ij} N^b"),
]
for _desc, _blk, _wrt, _pre, _arg, _res in _MUTANTS:
    _structural = _rejects(_arg)
    _inter = _lin.parse(f"{_pre} 1 ( {_arg} )")
    _resn = _lin.parse(_res)
    _differs = any(
        not _lin.zero(_lin.evaluate(_inter, {'i': _i}, mode=('U', _j))
                      - _lin.evaluate(_resn, {'i': _i, 'j': _j}))
        for _i in range(3) for _j in range(3))
    check(f"negative: P4 rejects [{_desc}] and the census does NOT (as intended)",
          _differs and _structural is None)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} elemental-component checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
