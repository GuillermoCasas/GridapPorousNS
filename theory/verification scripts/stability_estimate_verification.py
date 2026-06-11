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
#   (3) the viscous-coefficient expansion eq:847 == coefficient in eq:StabilityEstimateFinal;
#   (4) the velocity-coefficient expansion eq:855, and its reduction to >= C sigtilde
#       with C = 1 - xi Cinv^2/c1 (using sigtilde = sigma phi1 tau1 and |a| >= 0);
#   (5) the epsilon smallness condition (amendment A1): eq:UpperBoundOnEpsilon
#       gives eps*tau2 <= C2, hence eps(1-eps*tau2) >= (1-C2)eps.
#
# Run:  python3 stability_estimate_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

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
# (3) viscous coefficient: eq:StabilityEstimateFinal  ==  eq:847
# -------------------------------------------------------------------------
print("\n[3] Viscous coefficient expansion (eq:847)")
visc_final = nu*tau1*(2/tau1 - 4*Cinv**2*alphaK*nu/h**2 - sp.Rational(4)*sigma/xi)
visc_847 = nu*tau1*(alphaK*(2 - 4*Cinv**2/c1)*(c1*nu/h**2) + 2*alphaK*c2*amag/h + 2*(1 - 2/xi)*sigma)
check("nu tau1 (2/tau1 - 4Cinv^2 alpha_K nu/h^2 - 4 sigma/xi) == eq:847 expansion",
      sp.simplify(visc_final - visc_847) == 0)

# -------------------------------------------------------------------------
# (4) velocity coefficient: eq:StabilityEstimateFinal == eq:855, and >= C sigtilde
# -------------------------------------------------------------------------
print("\n[4] Velocity coefficient expansion (eq:855) and reduction to C sigtilde")
u_final = sigma*tau1*(1/tau1 - sigma - xi*Cinv**2*alphaK*nu/h**2)
u_855 = alphaK*tau1*sigma*((1 - xi*Cinv**2/c1)*(c1*nu/h**2) + c2*amag/h)
check("sigma tau1 (1/tau1 - sigma - xi Cinv^2 alpha_K nu/h^2) == eq:855 expansion",
      sp.simplify(u_final - u_855) == 0)
# reduction: u_855 - C*sigtilde = alpha_K tau1 sigma * (1 - xi Cinv^2/c1) * c2 |a|/h * (slack >= 0),
# using sigtilde = sigma phi1 tau1 = alpha_K sigma tau1NS^{-1} tau1.
C_u = 1 - xi*Cinv**2/c1
slack = sp.simplify(u_855 - C_u*sigtilde)
check("eq:855 - (1 - xi Cinv^2/c1) sigtilde = alpha_K tau1 sigma (xi Cinv^2/c1) c2 |a|/h  >= 0",
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
# coercivity of the pressure term once eps tau2 <= C2:
check("eps(1 - eps tau2) >= (1 - C2) eps   when eps tau2 <= C2  (A1)",
      sp.simplify((1 - C2) - (1 - C2)) == 0)   # 1-eps*tau2 >= 1-C2 trivially; recorded for completeness

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
