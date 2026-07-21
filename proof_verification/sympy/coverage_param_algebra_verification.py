#!/usr/bin/env python3
# =============================================================================
# coverage_param_algebra_verification.py
#
# Symbolic (sympy) verification of the CONTINUITY-APPENDIX PARAMETER ALGEBRA of
#   "A stabilized finite element method ... porous media"
#   (theory/paper/continuity_appendix.tex).
#
# WHY THIS SCRIPT EXISTS (coverage-gap closure, 2026-07-21)
# --------------------------------------------------------
# The parameter definitions eq:taus / eq:phi1 / eq:sigmatilde and the elementary
# identities of Lemma "Parameter inequalities" (lem:parameters) are algebraically
# correct but had NO permanent encoded check.  This script reconstructs each
# identity FROM ITS UPSTREAM DEFINITIONS (not from the printed form) and encodes:
#
#   * P:27         (l.229-233) : phi1 h^2 = c1 alpha_K^2 tau2, and the sqrt form
#                                phi1^{1/2} h = c1^{1/2} alpha_K tau2^{1/2}
#                                            <= c1^{1/2} tau2^{1/2}  (alpha_K<=1)
#   * unnumbered@248 (l.248-252): the phi1 h^2 = alpha_K h^2 tau1NS_inv
#                                = c1 alpha_K^2 (h^2/(c1 alpha_K tau1NS))
#                                = c1 alpha_K^2 tau2  proof chain, per equality.
#   * eq:sigmatilde (P:sigmatilde) : sigma - sigt = sigma^2/(phi1+sigma), and the
#                                full harmonic-mean chain
#                                tau1NS_inv sigma/(tau1NS_inv + sigma/alpha_K)
#                                = phi1 sigma/(phi1+sigma) = sigma - sigma^2 tau1
#                                = sigma phi1 tau1, plus sigt >= 0,
#                                sigt <= min(sigma,phi1), sigt tau1 <= 1.
#
# Definitions (eq:taus, eq:phi1):
#   tau1NS_inv = c1 nu/h^2 + c2 |a|/h ,   tau1NS = 1/tau1NS_inv
#   phi1       = alpha_K tau1NS_inv
#   tau1       = 1/(phi1 + sigma)
#   tau2       = h^2/(c1 alpha_K tau1NS)
#   sigt       = sigma phi1 tau1  ( = sigma phi1/(phi1+sigma) )
#
# Several checks are DISCRIMINATING: a plausible wrong variant (dropped alpha,
# missing sigma in a denominator, wrong power) is asserted to FAIL, so the check
# has teeth against a regression.
#
# Run:  python3 coverage_param_algebra_verification.py     (requires sympy)
# =============================================================================
import sympy as sp
import random

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

print("=" * 70)
print("Continuity-appendix parameter algebra (eq:taus/phi1/sigmatilde) -- symbolic")
print("=" * 70)

# Positive physical symbols.  c1,c2>0 algorithmic constants; nu,h,amag>0;
# alphaK = alpha_{inf,K} in (0,1] (H:porosity); sigma>=0 dimensional reaction.
c1, c2, nu, h, amag, alphaK, sigma = sp.symbols(
    'c1 c2 nu h amag alpha_K sigma', positive=True)

# ---- Definitions (eq:taus, eq:phi1) -------------------------------------
tau1NS_inv = c1*nu/h**2 + c2*amag/h          # eq:taus, tau_{1,NS}^{-1}
tau1NS     = 1/tau1NS_inv
phi1       = alphaK*tau1NS_inv               # eq:phi1
tau1       = 1/(phi1 + sigma)                # eq:taus tau_{1,K} = (phi1+sigma)^{-1}
tau2       = h**2/(c1*alphaK*tau1NS)         # eq:taus tau_{2,K}
sigt       = sigma*phi1*tau1                 # eq:sigmatilde (final closed form)

Z = lambda e: sp.simplify(e) == 0            # "is identically zero"

# =========================================================================
# P:27  (continuity_appendix.tex l.229-233)
#   phi1 h^2 = c1 alpha_K^2 tau2
# =========================================================================
check("P:27  phi1 h^2 = c1 alpha_K^2 tau2",
      Z(phi1*h**2 - c1*alphaK**2*tau2))
# Discriminating: dropping one alpha_K power must NOT hold.
check("P:27  DISCRIMINATING: c1 alpha_K^1 tau2 (dropped alpha) FAILS",
      not Z(phi1*h**2 - c1*alphaK*tau2))
# sqrt form:  phi1^{1/2} h = c1^{1/2} alpha_K tau2^{1/2}
check("P:27  phi1^{1/2} h = c1^{1/2} alpha_K tau2^{1/2}",
      Z(sp.sqrt(phi1)*h - sp.sqrt(c1)*alphaK*sp.sqrt(tau2)))

# =========================================================================
# unnumbered@248  (l.248-252): the P:27 proof chain, equality by equality.
#   phi1 h^2 = alpha_K h^2 tau1NS_inv
#            = c1 alpha_K^2 (h^2/(c1 alpha_K tau1NS))
#            = c1 alpha_K^2 tau2
# =========================================================================
step_a = alphaK*h**2*tau1NS_inv
step_b = c1*alphaK**2*(h**2/(c1*alphaK*tau1NS))
check("@248  eq(1): phi1 h^2 = alpha_K h^2 tau1NS_inv",
      Z(phi1*h**2 - step_a))
check("@248  eq(2): alpha_K h^2 tau1NS_inv = c1 alpha_K^2 (h^2/(c1 alpha_K tau1NS))",
      Z(step_a - step_b))
check("@248  eq(3): c1 alpha_K^2 (h^2/(c1 alpha_K tau1NS)) = c1 alpha_K^2 tau2",
      Z(step_b - c1*alphaK**2*tau2))

# =========================================================================
# eq:sigmatilde / P:sigmatilde  (l.239-247 def; l.255-260 proof)
# =========================================================================
# Headline task identity:  sigma - sigt = sigma^2/(phi1+sigma)
check("sigmatilde  sigma - sigt = sigma^2/(phi1+sigma)",
      Z((sigma - sigt) - sigma**2/(phi1 + sigma)))
# Equivalent closed form with tau1:  sigma - sigt = sigma^2 tau1
check("sigmatilde  sigma - sigt = sigma^2 tau1",
      Z((sigma - sigt) - sigma**2*tau1))
# Discriminating: dropping sigma from the denominator must NOT hold.
check("sigmatilde  DISCRIMINATING: sigma^2/phi1 (dropped sigma in denom) FAILS",
      not Z((sigma - sigt) - sigma**2/phi1))

# Full harmonic-mean chain of eq:sigmatilde:
#   tau1NS_inv sigma/(tau1NS_inv + sigma/alpha_K)
#     = phi1 sigma/(phi1+sigma) = sigma - sigma^2 tau1 = sigma phi1 tau1
sigt_raw  = tau1NS_inv*sigma/(tau1NS_inv + sigma/alphaK)   # printed first form
sigt_phi  = phi1*sigma/(phi1 + sigma)                      # harmonic mean
check("sigmatilde  chain(1): tau1NS_inv sigma/(tau1NS_inv+sigma/alpha_K) = phi1 sigma/(phi1+sigma)",
      Z(sigt_raw - sigt_phi))
check("sigmatilde  chain(2): phi1 sigma/(phi1+sigma) = sigma - sigma^2 tau1",
      Z(sigt_phi - (sigma - sigma**2*tau1)))
check("sigmatilde  chain(3): sigma - sigma^2 tau1 = sigma phi1 tau1",
      Z((sigma - sigma**2*tau1) - sigt))
# Discriminating: the raw form with a WRONG denominator (sigma instead of
# sigma/alpha_K, i.e. dropping the /alpha_K) must NOT collapse to phi1 mean.
sigt_raw_bad = tau1NS_inv*sigma/(tau1NS_inv + sigma)
check("sigmatilde  DISCRIMINATING: raw form missing /alpha_K FAILS chain(1)",
      not Z(sigt_raw_bad - sigt_phi))

# sigt <= min(sigma, phi1):  sigma - sigt and phi1 - sigt are perfect squares/(sum)
check("sigmatilde  sigma - sigt = sigma^2/(phi1+sigma) >= 0  (=> sigt <= sigma)",
      Z((sigma - sigt) - sigma**2/(phi1 + sigma)))
check("sigmatilde  phi1  - sigt = phi1^2/(phi1+sigma)  >= 0  (=> sigt <= phi1)",
      Z((phi1 - sigt) - phi1**2/(phi1 + sigma)))

# ---- Numeric confirmation of the inequalities (P:27 sqrt bound; P:sigmatilde) --
# alpha_K in (0,1], everything else > 0, sigma >= 0.  Random dense sample.
syms = (c1, c2, nu, h, amag, alphaK, sigma)
def sample():
    return {c1: random.uniform(0.1, 20), c2: random.uniform(0.1, 20),
            nu: random.uniform(1e-4, 5), h: random.uniform(1e-3, 2),
            amag: random.uniform(0, 10), alphaK: random.uniform(1e-3, 1.0),
            sigma: random.uniform(0, 50)}
random.seed(20260721)
N = 4000
ok_p27  = ok_min = ok_tau = ok_nonneg = True
f_p27lhs = sp.lambdify(syms, sp.sqrt(phi1)*h, 'math')
f_p27rhs = sp.lambdify(syms, sp.sqrt(c1)*sp.sqrt(tau2), 'math')
f_sigt   = sp.lambdify(syms, sigt, 'math')
f_sigma  = sp.lambdify(syms, sigma, 'math')
f_phi1   = sp.lambdify(syms, phi1, 'math')
f_sttau1 = sp.lambdify(syms, sigt*tau1, 'math')
for _ in range(N):
    s = sample(); args = tuple(s[v] for v in syms)
    if not (f_p27lhs(*args) <= f_p27rhs(*args) + 1e-12):            ok_p27 = False
    st = f_sigt(*args)
    if not (st >= -1e-12):                                          ok_nonneg = False
    if not (st <= min(f_sigma(*args), f_phi1(*args)) + 1e-12):      ok_min = False
    if not (f_sttau1(*args) <= 1 + 1e-12):                          ok_tau = False
check(f"P:27  phi1^{{1/2}} h <= c1^{{1/2}} tau2^{{1/2}}  (alpha_K<=1), {N} samples", ok_p27)
check(f"sigmatilde  sigt >= 0, {N} samples", ok_nonneg)
check(f"sigmatilde  sigt <= min(sigma,phi1), {N} samples", ok_min)
check(f"sigmatilde  sigt tau1 <= 1, {N} samples", ok_tau)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
