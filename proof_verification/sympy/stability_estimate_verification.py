#!/usr/bin/env python3
# =============================================================================
# stability_estimate_verification.py
#
# Symbolic (sympy) verification of the algebraic identities underlying the
# coercivity / stability estimate of Section 5
#   "A stabilized finite element method ... porous media".
#
# (The full Galerkin identity eq:StabilityEstimate is an integration-by-parts
#  statement; here we verify the algebra that the estimate rests on, which is
#  where transcription errors hide.)  With
#     tau1NS^{-1} = c1 nu/h^2 + c2 |a|/h            (eq:TauNavierStokes)
#     tau1 = (alpha_K tau1NS^{-1} + sigma)^{-1}     (eq:Tau1Final)
#     tau2 = h^2/(c1 alpha_K tau1NS)                (eq:Tau2Final)
#     phi1 = alpha_K tau1NS^{-1},  sigtilde = tau1NS^{-1} sigma/(tau1NS^{-1}+sigma/alpha_K)  (eq:SigmaAlpha)
# it checks:
#   (1) the four equivalent forms of sigtilde, incl. the key sigtilde = sigma phi1 tau1;
#   (2) the Young inequality -2xy >= -x^2/xi - xi y^2 (perfect-square identity);
#   (3) the viscous-coefficient expansion (SUPERSEDED xi-form, see the note below);
#   (4) the velocity-coefficient expansion (SUPERSEDED xi-form), and its reduction to >= C sigtilde
#       with C = 1 - xi Cinv^2/c1 (using sigtilde = sigma phi1 tau1 and |a| >= 0);
#   (5) the epsilon smallness condition (amendment A1): eq:UpperBoundOnEpsilon
#       gives eps*tau2 <= C2, hence eps(1-eps*tau2) >= (1-C2)eps;
#   (6) the CURRENT Step-5/Step-6 coercivity collection of App. C, read from the source.
#
# STALE ANCHORS, and what replaced them (2026-07-29).  Checks (3) and (4) were written
# against displays named eq:StabilityEstimateFinal / eq:ViscousCoefficientBound /
# eq:VelocityCoefficientBound.  Those labels exist in NO .tex in this repository: App. C
# was rewritten and its Step 5/6 is now the eta-parameterised eq:coerccollect ->
# eq:coercoeff, with t := Cinva^2/c1 in place of the xi split.  The identities (3)/(4)
# encode remain true and are the algebra behind the xi reading of H:coercivity, so they
# are kept -- but they were CLAIMING coverage of printed displays that no longer exist,
# which is the coverage-illusion form of a vacuous rule.  Section (6) below covers what
# the appendix actually prints today, anchors itself to LIVE labels, and asserts those
# labels resolve, so the next rename fails loudly instead of drifting silently.
#
# Run:  python3 stability_estimate_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

def _raises(fn):
    """True iff fn() raises -- used by the negative controls below."""
    try:
        fn()
    except Exception:                                     # noqa: BLE001
        return True
    return False

print("=" * 70)
print("Stability-estimate algebra (Section 5) -- symbolic checks")
print("=" * 70)

nu, h, alphaK, sigma, amag = sp.symbols('nu h alpha_K sigma a', positive=True)
c1, c2, Cinv, xi = sp.symbols('c1 c2 C_inv xi', positive=True)
eps, C2 = sp.symbols('varepsilon C2', positive=True)

tau1NS_inv = c1*nu/h**2 + c2*amag/h
tau1 = 1/(alphaK*tau1NS_inv + sigma)
tau2 = h**2/(c1*alphaK/tau1NS_inv)                  # = h^2 tau1NS^{-1}/(c1 alpha_K)
phi1 = alphaK*tau1NS_inv
sigtilde = tau1NS_inv*sigma/(tau1NS_inv + sigma/alphaK)

# -------------------------------------------------------------------------
# (1) sigtilde : four equivalent forms
# -------------------------------------------------------------------------
print("\n[1] sigtilde identities (eq:SigmaAlpha)")
check("sigtilde = sigma phi1/(phi1+sigma)",
      sp.simplify(sigtilde - sigma*phi1/(phi1 + sigma)) == 0)
check("sigtilde = sigma - sigma^2 tau1",
      sp.simplify(sigtilde - (sigma - sigma**2*tau1)) == 0)
check("sigtilde = sigma phi1 tau1   (key form for the velocity coefficient)",
      sp.simplify(sigtilde - sigma*phi1*tau1) == 0)

# -------------------------------------------------------------------------
# (2) Young's inequality used throughout:  -2xy >= -x^2/xi - xi y^2
# -------------------------------------------------------------------------
print("\n[2] Young inequality (perfect square)")
xx, yy = sp.symbols('x y', real=True)
check("x^2/xi - 2 x y + xi y^2 = (x/sqrt(xi) - sqrt(xi) y)^2  >= 0",
      sp.simplify(xx**2/xi - 2*xx*yy + xi*yy**2 - (xx/sp.sqrt(xi) - sp.sqrt(xi)*yy)**2) == 0)

# -------------------------------------------------------------------------
# (3) viscous coefficient -- SUPERSEDED xi-form (see the header note): the display it
#     was written against, eq:ViscousCoefficientBound, is defined in no .tex today.
# -------------------------------------------------------------------------
print("\n[3] Viscous coefficient expansion (superseded xi-form)")
visc_final = nu*tau1*(2/tau1 - 4*Cinv**2*alphaK*nu/h**2 - sp.Rational(4)*sigma/xi)
visc_847 = nu*tau1*(alphaK*(2 - 4*Cinv**2/c1)*(c1*nu/h**2) + 2*alphaK*c2*amag/h + 2*(1 - 2/xi)*sigma)
check("nu tau1 (2/tau1 - 4Cinv^2 alpha_K nu/h^2 - 4 sigma/xi) == eq:ViscousCoefficientBound expansion",
      sp.simplify(visc_final - visc_847) == 0)

# -------------------------------------------------------------------------
# (4) velocity coefficient -- SUPERSEDED xi-form (see the header note), kept because the
#     reduction to C*sigtilde is the algebra behind the xi reading of H:coercivity.
# -------------------------------------------------------------------------
print("\n[4] Velocity coefficient expansion (superseded xi-form) and reduction to C sigtilde")
u_final = sigma*tau1*(1/tau1 - sigma - xi*Cinv**2*alphaK*nu/h**2)
u_855 = alphaK*tau1*sigma*((1 - xi*Cinv**2/c1)*(c1*nu/h**2) + c2*amag/h)
check("sigma tau1 (1/tau1 - sigma - xi Cinv^2 alpha_K nu/h^2) == eq:VelocityCoefficientBound expansion",
      sp.simplify(u_final - u_855) == 0)
# reduction: u_855 - C*sigtilde = alpha_K tau1 sigma * (1 - xi Cinv^2/c1) * c2 |a|/h * (slack >= 0),
# using sigtilde = sigma phi1 tau1 = alpha_K sigma tau1NS^{-1} tau1.
C_u = 1 - xi*Cinv**2/c1
slack = sp.simplify(u_855 - C_u*sigtilde)
check("eq:VelocityCoefficientBound - (1 - xi Cinv^2/c1) sigtilde = alpha_K tau1 sigma (xi Cinv^2/c1) c2 |a|/h  >= 0",
      sp.simplify(slack - alphaK*tau1*sigma*(xi*Cinv**2/c1)*(c2*amag/h)) == 0)

# -------------------------------------------------------------------------
# (5) epsilon smallness condition (amendment A1)
# -------------------------------------------------------------------------
print("\n[5] Epsilon condition (eq:UpperBoundOnEpsilon -> eps tau2 <= C2)")
# eq:UpperBoundOnEpsilon:  eps <= C2 c1 alpha_K^2 tau1 / h^2   (using tau_{1,K}=tau1).
# tau2 <= h^2/(c1 alpha_K tau1NS) and tau_{1,K} <= tau1NS/alpha_K give eps*tau2 <= C2.
# Symbolic chain with the bound eps_max:
eps_max = C2*c1*alphaK**2*tau1/h**2
# tau2 written out and tau1 <= tau1NS/alpha_K  (since tau1^{-1} = alpha_K tau1NS^{-1}+sigma >= alpha_K tau1NS^{-1})
ratio = sp.simplify(eps_max*tau2)                  # = C2 * alpha_K * tau1 / tau1NS  (<= C2 since tau1 <= tau1NS/alpha_K)
tau1NS = 1/tau1NS_inv
check("eps_max * tau2 = C2 alpha_K tau1/tau1NS, and alpha_K tau1 <= tau1NS  =>  eps tau2 <= C2",
      sp.simplify(ratio - C2*alphaK*tau1/tau1NS) == 0
      and sp.simplify((tau1NS - alphaK*tau1)) == sp.simplify(sigma*tau1*tau1NS))   # >=0 since sigma,tau1,tau1NS>0
# coercivity of the pressure term once eps tau2 <= C2.  This check used to read
#   sp.simplify((1 - C2) - (1 - C2)) == 0
# i.e. 0 == 0 -- a tautology that certified nothing while printing [PASS] (the same
# vacuous-rule class as the 2026-07-28 Rule-1 and 2026-07-29 D3 incidents).  The real
# content is that the SLACK is exactly eps*(C2 - eps*tau2), hence non-negative precisely
# under the hypothesis eps*tau2 <= C2 -- and nothing weaker.
_e, _t2 = sp.symbols('eps_ tau2_', positive=True)
_slack = sp.simplify(_e*(1 - _e*_t2) - (1 - C2)*_e)
check("eps(1 - eps tau2) - (1 - C2) eps = eps (C2 - eps tau2) >= 0 iff eps tau2 <= C2  (A1)",
      sp.simplify(_slack - _e*(C2 - _e*_t2)) == 0
      and _slack.subs({_e: 1, _t2: 1, C2: sp.Rational(1, 2)}) < 0)   # negative when the hypothesis fails

# -------------------------------------------------------------------------
# (6) THE CURRENT App. C Step 5/6 -- eq:coerccollect -> eq:coercoeff.
#
# What this section is for.  The eta-optimisation printed in Step 6 is a chain whose
# MIDDLE members are printed and whose endpoint (eta = 4) is what the paper carries
# forward.  Before 2026-07-29 nothing in the suite read any of it: the labels the
# sections above name were stale, and no other script greps eq:coerccollect.  The chain
# also feeds a code-facing decision -- the c1 > 4 Cinva^2 threshold behind the 3D
# element-aware-c1 work (docs/findings.md 3) -- so an error here would not stay in prose.
#
# The printed expressions are read from asgs_convergence.tex and converted to sympy, so
# this checks the paper's own claim rather than a transcription of it (the design
# argument written up in docs/appendix-a-intermediate-coverage-spec.md 3).
# -------------------------------------------------------------------------
print("\n[6] Current App. C coercivity collection: eta-optimisation (eq:coerccollect -> eq:coercoeff)")

import os as _os
import re as _re

_APPC_PATH = _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..",
    "theory", "paper", "asgs_convergence.tex"))
with open(_APPC_PATH, encoding="utf-8") as _f:
    _APPC = _f.read()


def _tex2sym(s):
    """Convert one of App. C's short rational expressions in (eta, t, Cinva, c1) to sympy.

    Deliberately narrow: it accepts the handful of constructs these three displays use and
    raises on anything else, so a rewritten display fails loudly instead of being silently
    mis-read.  Its own negative control is asserted below.
    """
    s = _re.sub(r'\\(?:bigl|bigr|Bigl|Bigr|left|right|quad)(?![A-Za-z])', ' ', s)
    s = _re.sub(r'\\[,;!:]', ' ', s)
    s = s.replace(r'\coloneqq', '=')
    for _ in range(4):                                   # nested \tfrac{}{} -> ( )/( )
        s2 = _re.sub(r'\\t?frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
        if s2 == s:
            break
        s = s2
    # pad each symbol so juxtaposition like  t\eta  cannot glue into one name
    s = s.replace(r'\eta', ' eta ').replace(r'\Cinva', ' Cinva ').replace('c_1', ' c1 ')
    s = s.replace('{', '(').replace('}', ')')
    if '\\' in s:
        raise ValueError(f"unconverted LaTeX left in {s!r}")
    s = _re.sub(r'\s+', ' ', s).strip()
    # implicit multiplication (done BEFORE ^ -> **, so exponents are never touched):
    #   't eta^2' -> 't*eta^2',  '4t' -> '4*t',  ')(' -> ')*('
    s = _re.sub(r'(?<=[\w\)])\s+(?=[\w\(])', '*', s)
    s = _re.sub(r'(?<=\d)(?=[A-Za-z(])', '*', s)
    s = _re.sub(r'(?<=\))(?=\()', '*', s)
    s = s.replace('^', '**')
    return sp.sympify(s, locals={'eta': sp.Symbol('eta', positive=True),
                                 't': sp.Symbol('t', positive=True),
                                 'Cinva': sp.Symbol('Cinva', positive=True),
                                 'c1': c1})


eta = sp.Symbol('eta', positive=True)
t = sp.Symbol('t', positive=True)
Cinva = sp.Symbol('Cinva', positive=True)


def _grab(pattern, what):
    m = _re.search(pattern, _APPC)
    if not m:
        raise AssertionError(f"could not locate {what} in asgs_convergence.tex")
    return m.group(1)


# E0  the labels this section anchors to must RESOLVE (the failure mode of [3]/[4] above)
_labels = ['eq:coerccollect', 'eq:coercoeff', 'eq:coerconstant']
_missing = [L for L in _labels if f'\\label{{{L}}}' not in _APPC]
check(f"E0 the labels this section reads are defined in App. C "
      f"({len(_labels) - len(_missing)} of {len(_labels)} resolve)", not _missing)

# E1/E2  the two eta-dependent coefficients as PRINTED in eq:coerccollect
_visc_printed = _tex2sym(_grab(r'\\Bigl\(\s*(2\\bigl\(1-\\tfrac\{2\}\{\\eta\}\\bigr\)'
                               r'-\\tfrac\{4\\Cinva\^2\}\{c_1\})\s*\\Bigr\)',
                               "the printed viscous coefficient of eq:coerccollect"))
_reac_printed = _tex2sym(_grab(r'\\Bigl\(\s*(1-\\tfrac\{\\eta\\Cinva\^2\}\{c_1\})\s*\\Bigr\)',
                               "the printed reaction coefficient of eq:coerccollect"))
# E3  the two definitions Step 6 introduces, with t := Cinva^2/c1
_cv_def = _tex2sym(_grab(r'c_\{\\mathrm v\}\(\\eta\)\\coloneqq\s*([^$]*?)\$', "c_v definition"))
_cr_def = _tex2sym(_grab(r'c_\{\\mathrm r\}\(\\eta\)\\coloneqq\s*([^$]*?)\$', "c_r definition"))
check("E1 printed viscous coefficient of eq:coerccollect == c_v(eta) with t = Cinva^2/c1",
      sp.simplify(_visc_printed - _cv_def.subs(t, Cinva**2/c1)) == 0)
check("E2 printed reaction coefficient of eq:coerccollect == c_r(eta) with t = Cinva^2/c1",
      sp.simplify(_reac_printed - _cr_def.subs(t, Cinva**2/c1)) == 0)

# E4  monotonicity, which is what licenses "their minimum is largest where the two coincide"
check("E4 c_v increases and c_r decreases in eta (d/d eta > 0 and < 0 for t > 0)",
      sp.simplify(sp.diff(_cv_def, eta) - 4/eta**2) == 0 and sp.diff(_cr_def, eta) == -t)

# E5  the printed quadratic:  c_v(eta) = c_r(eta)  "reads"  t eta^2 + (1-4t) eta - 4 = 0
_quad_printed = _tex2sym(_grab(r'reads \$\s*(t\\eta\^2\+\(1-4t\)\\eta-4)\s*=0\$', "printed quadratic"))
check("E5 the printed quadratic t eta^2 + (1-4t) eta - 4 is eta (c_v(eta) - c_r(eta))",
      sp.simplify(sp.expand(eta*(_cv_def - _cr_def)) - sp.expand(_quad_printed)) == 0)

# E6  the printed factorisation
_fact_printed = _tex2sym(_grab(r'i\.e\.\\?\s*\n?\$\s*(\(\\eta-4\)\(t\\eta\+1\))\s*=0\$', "printed factorisation"))
check("E6 the printed factorisation (eta-4)(t eta+1) expands to that quadratic",
      sp.simplify(sp.expand(_fact_printed) - sp.expand(_quad_printed)) == 0)

# E7  ... hence eta = 4 is the unique admissible root, "whatever the value of t".
#     Solved over an UNRESTRICTED eta, so the claim rests on the algebra rather than on a
#     positivity assumption baked into the symbol.
_e = sp.Symbol('eta_free')
_roots = sp.solve(sp.Eq(_quad_printed.subs(eta, _e), 0), _e)
check(f"E7 the roots are exactly 4 and -1/t, so eta = 4 is the unique positive one for "
      f"every t > 0 (roots {_roots})",
      len(_roots) == 2 and sp.Integer(4) in _roots
      and any(sp.simplify(r + 1/t) == 0 for r in _roots))

# E8  eq:coercoeff:  c_v(4) = c_r(4) = 1 - 4 Cinva^2/c1
_coeff_printed = _tex2sym(_grab(r'c_\{\\mathrm v\}\(4\)=c_\{\\mathrm r\}\(4\)=\s*'
                                r'(1-\\frac\{4\\Cinva\^2\}\{c_1\})', "eq:coercoeff"))
check("E8 c_v(4) = c_r(4) = printed 1 - 4 Cinva^2/c1",
      sp.simplify(_cv_def.subs({eta: 4, t: Cinva**2/c1}) - _coeff_printed) == 0
      and sp.simplify(_cr_def.subs({eta: 4, t: Cinva**2/c1}) - _coeff_printed) == 0)

# E9  positivity of that common value is exactly c1 > 4 Cinva^2 -- the threshold the 3D
#     element-aware-c1 work leans on
#     (stated as a sign identity: multiplying by the positive c1 preserves the sign, so the
#      printed coefficient is positive exactly where c1 - 4 Cinva^2 is)
check("E9 c1 * (printed coefficient) = c1 - 4 Cinva^2, so it is positive precisely when c1 > 4 Cinva^2",
      sp.simplify(sp.expand(c1*_coeff_printed) - (c1 - 4*Cinva**2)) == 0)

# E10 App. C asserts that H:coercivity's "c1 > 2 xi Cinva^2 for some xi > 2" and
#     "c1 > 4 Cinva^2" are equivalent requirements on c1.  The forward direction is
#     immediate; the reverse needs a WITNESS xi in (2, c1/(2 Cinva^2)), and the midpoint
#     xi* = (2 + c1/(2 Cinva^2))/2 is one exactly when c1 > 4 Cinva^2.  Both defining
#     inequalities for xi* reduce to a positive multiple of (c1 - 4 Cinva^2), which is the
#     whole content of the claim.
_xistar = (2 + c1/(2*Cinva**2))/2
check("E10a the witness xi* = (2 + c1/(2 Cinva^2))/2 satisfies xi* - 2 = (c1 - 4 Cinva^2)/(4 Cinva^2) "
      "and c1 - 2 xi* Cinva^2 = (c1 - 4 Cinva^2)/2, so it is admissible exactly when c1 > 4 Cinva^2",
      sp.simplify(sp.expand(_xistar - 2) - (c1 - 4*Cinva**2)/(4*Cinva**2)) == 0
      and sp.simplify(sp.expand(c1 - 2*_xistar*Cinva**2) - (c1 - 4*Cinva**2)/2) == 0)
# E10b the xi-explicit consequence quoted for eq:coerconstant: 4 Cinva^2/c1 < 2/xi.
#      Content: the difference is a positive multiple of (2 xi Cinva^2 - c1), i.e. it is
#      negative precisely under the hypothesis and NOT unconditionally.
_d10 = sp.simplify(4*Cinva**2/c1 - 2/xi)
check("E10b 4 Cinva^2/c1 - 2/xi = 2(2 xi Cinva^2 - c1)/(c1 xi), hence < 0 exactly when c1 > 2 xi Cinva^2",
      sp.simplify(_d10 - 2*(2*xi*Cinva**2 - c1)/(c1*xi)) == 0
      and _d10.subs({Cinva: 1, c1: 1, xi: 3}) > 0)      # fails when the hypothesis fails

# ---- negative controls: the converter and the algebra must both discriminate ----
check("negative: _tex2sym rejects an unconverted LaTeX construct",
      _raises(lambda: _tex2sym(r'\sqrt{\eta}')))
check("negative: a wrong factorisation (eta-4)(t eta-1) fails E6",
      sp.simplify(sp.expand(_tex2sym(r'(\eta-4)(t\eta-1)')) - sp.expand(_quad_printed)) != 0)
check("negative: a viscous coefficient missing the outer factor 2 fails E1",
      sp.simplify(_tex2sym(r'(1-\tfrac{2}{\eta})-\tfrac{4\Cinva^2}{c_1}')
                  - _cv_def.subs(t, Cinva**2/c1)) != 0)
check("negative: eta = 2 is not a root of the printed quadratic for general t",
      sp.simplify(_quad_printed.subs(eta, 2)) != 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
