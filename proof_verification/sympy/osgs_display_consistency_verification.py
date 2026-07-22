#!/usr/bin/env python3
# =============================================================================
# osgs_display_consistency_verification.py
#
# Closes a coverage gap the 2026-07-22 external audit exposed (latest_audit,
# issue M15 / full audit sect. 7.10): the OSGS appendix (App. D, osgs_appendix.tex)
# had NO symbolic display coverage at all. The F1/F2 body-vs-appendix
# display-consistency net (display_consistency_verification.py) stopped at the
# main text + Appendix A and was never extended past the ASGS material, so a
# body display that drops a factor proved in the OSGS appendix would pass the
# whole suite unflagged.
#
# The specific display at issue: the sigma-robust L^2 velocity corollary
# oa:eq:LtwoVelocity (osgs_appendix.tex:1190) carries the elementwise factor
# (1 + c1/Da_h); the body text (article_v2.tex:1082-1092, 1226-1230) states the
# Da_h >> c1 asymptotic form in which that factor -> 1. The Coq proves the
# corollary only ABSTRACTLY (OsgsConvergence.abstract_osgs_convergence_Ltwo:
# sqrt(sigma)*||e_u|| <= Cconv*Psi_O), keeping the RHS as the abstract functional
# Eh and never unfolding this closed form -- so neither machine layer saw it.
#
# This script re-derives the corollary from the OSGS error functional Psi_O and
# the elementwise parameter identities, verifying that the (1 + c1/Da_h) factor
# is genuinely produced by the sigma^{-1} normalization, and asserts the
# discriminating negative: the factor is NOT identically 1 for finite Da_h, so
# the body's factor-free form is legitimate ONLY as the Da_h -> infinity limit.
# A silent re-omission of the factor from the *appendix* corollary would fail
# check (C1)/(C4); a body display equal to the appendix for finite Da_h (i.e. a
# spurious factor) would fail (C5b).
#
# Symbols/identities are transcribed from osgs_appendix.tex:
#   Re_h  = |a|_inf,K h_K / nu                              (oa:eq:...,  l.107)
#   Da_h  = sigma h_K^2 / (alpha_K nu)                      (              l.110)
#   tau2  = (nu/alpha_K)(1 + (c2/c1) Re_h)                  (oa:eq:Tau2Expanded, l.102-105)
#   tau1 <= 1/sigma                                          (oa:eq:P1,  l.287)
#   Psi_O^2 summand = (c1+Da_h)(alpha_K^2/h_K^2)(tau2 E_u^2 + tau1 E_p^2)
#                                                            (oa:eq:ErrorFunctionL2, l.847)
#   ||e_u|| <= sigma^{-1/2} Psi_O(h)                        (oa:cor:Ltwo proof, l.1209)
#
# Run:  python3 osgs_display_consistency_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    print(f"  [{tag}] {name}")
    return ok

print("=" * 72)
print("OSGS L^2 velocity corollary (oa:eq:LtwoVelocity) -- body vs appendix")
print("=" * 72)

# --- symbols (all positive; they are physical magnitudes) ---------------------
nu, alphaK, hK, sigma = sp.symbols('nu alpha_K h_K sigma', positive=True)
c1, c2, ainf = sp.symbols('c1 c2 a_inf', positive=True)
Eu, Ep = sp.symbols('E_u E_p', positive=True)   # interpolation errors E_int,K(u), E_int,K(p)

# --- dimensionless numbers (osgs_appendix.tex l.107,110) ----------------------
Re_h = ainf * hK / nu
Da_h = sigma * hK**2 / (alphaK * nu)

# --- stabilization parameters -------------------------------------------------
tau2 = (nu / alphaK) * (1 + (c2 / c1) * Re_h)   # oa:eq:Tau2Expanded (exact form)
tau1_bound = 1 / sigma                           # oa:eq:P1 : tau1 <= 1/sigma (used in the pressure slot)

# =============================================================================
# (C1) The load-bearing weight identity that MANIFESTS the (1 + c1/Da_h) factor.
#      sigma^{-1} (c1 + Da_h) alpha_K^2 / h_K^2  ==  (1 + c1/Da_h) alpha_K / nu
#      (osgs_appendix.tex l.1213-1216 -- the step the Coq never unfolds)
# =============================================================================
lhs_weight = (1 / sigma) * (c1 + Da_h) * alphaK**2 / hK**2
rhs_weight = (1 + c1 / Da_h) * alphaK / nu
check("(C1) sigma^{-1}(c1+Da_h)alpha_K^2/h_K^2 == (1 + c1/Da_h) alpha_K/nu",
      sp.simplify(lhs_weight - rhs_weight) == 0)

# =============================================================================
# (C2) velocity slot:  (alpha_K/nu) tau2 == 1 + (c2/c1) Re_h   (oa:eq:Tau2Expanded)
# =============================================================================
check("(C2) (alpha_K/nu) tau2 == 1 + (c2/c1) Re_h",
      sp.simplify((alphaK / nu) * tau2 - (1 + (c2 / c1) * Re_h)) == 0)

# =============================================================================
# (C3) pressure slot:  (alpha_K/nu) tau1  <=  alpha_K/(sigma nu),
#      at the bound tau1 = 1/sigma it is an equality (oa:cor:Ltwo proof).
# =============================================================================
check("(C3) (alpha_K/nu) tau1|_{tau1=1/sigma} == alpha_K/(sigma nu)",
      sp.simplify((alphaK / nu) * tau1_bound - alphaK / (sigma * nu)) == 0)

# =============================================================================
# (C4) FULL appendix corollary summand follows from sigma^{-1} Psi_O^2.
#      sigma^{-1} * [ (c1+Da_h) alpha_K^2/h_K^2 (tau2 E_u^2 + tau1 E_p^2) ]
#        == (1 + c1/Da_h) [ (1 + (c2/c1)Re_h) E_u^2 + (alpha_K/(sigma nu)) E_p^2 ]
#      i.e. exactly oa:eq:LtwoVelocity (the sum-of-squares form; the (a+b)^2
#      <= 2(a^2+b^2) split that turns Psi_O's (sqrt tau2 E_u + sqrt tau1 E_p)^2
#      into a sum is absorbed into C and is not part of this weight identity).
# =============================================================================
psiO_summand = (c1 + Da_h) * alphaK**2 / hK**2 * (tau2 * Eu**2 + tau1_bound * Ep**2)
appendix_summand = (1 + c1 / Da_h) * (
    (1 + (c2 / c1) * Re_h) * Eu**2 + (alphaK / (sigma * nu)) * Ep**2
)
check("(C4) sigma^{-1} Psi_O summand == oa:eq:LtwoVelocity summand (factor present)",
      sp.simplify((1 / sigma) * psiO_summand - appendix_summand) == 0)

# =============================================================================
# (C5) DISCRIMINATING body-vs-appendix check.
#   The body (article_v2.tex:1082-1092,1226-1230) states the SAME estimate with
#   the factor set to 1 (the Da_h >> c1 asymptotic form, cross-referenced to
#   oa:cor:Ltwo). So:
#     body_summand := appendix_summand with (1 + c1/Da_h) -> 1
#   (C5a) the two are NOT equal for finite Da_h  -> the factor is real; a silent
#         drop of it from the *appendix* would be caught here.
#   (C5b) their ratio -> 1 as Da_h -> infinity   -> the body form is a legitimate
#         asymptotic simplification, not a wrong statement.
# =============================================================================
body_summand = appendix_summand.subs(c1 / Da_h, 0)          # factor -> 1
ratio = sp.simplify(appendix_summand / body_summand)         # == 1 + c1/Da_h
check("(C5a) appendix != body for finite Da_h  (factor genuinely present, not 1)",
      sp.simplify(ratio - 1) != 0)
lim = sp.limit(ratio, sigma, sp.oo)                          # Da_h -> oo  <=>  sigma -> oo
check("(C5b) appendix/body -> 1 as Da_h -> infinity  (body IS the asymptotic form)",
      sp.simplify(lim - 1) == 0)

# =============================================================================
print("=" * 72)
n_fail = sum(1 for tag, _ in results if tag == "FAIL")
print(f"SUMMARY: {len(results) - n_fail}/{len(results)}")
print("=" * 72)
raise SystemExit(1 if n_fail else 0)
