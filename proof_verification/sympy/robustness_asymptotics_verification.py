#!/usr/bin/env python3
# =============================================================================
# robustness_asymptotics_verification.py
#
# Symbolic (sympy) verification of the robustness analysis of Section 6
#   "A stabilized finite element method ... porous media".
#
# With the exact parameters (equal-order constants c1 = c2 = 1),
#   tau1 = (alpha_K tau1NS^{-1} + sigma)^{-1}         (eq:Tau1Final)
#   tau2 = h^2/(c1 alpha_K tau1NS)                    (eq:Tau2Final)
#   sigtilde = tau1NS^{-1} sigma/(tau1NS^{-1}+sigma/alpha_K)  (eq:SigmaAlpha)
#   tau1NS^{-1} = c1 nu/h^2 + c2 |a|/h                (eq:TauNavierStokes)
# and the elemental numbers Re_h = |a|h/nu, Da_h = sigma h^2/(alpha_K nu), it checks:
#   (1) the general asymptotic forms (eq:GeneralAsymptoticBehaviourOfParameters),
#       INCLUDING the alpha_K factor on sigtilde corrected in amendment A6.1;
#   (2) the dominant-viscosity, dominant-convection and dominant-reaction limits
#       (eq blocks of Section 6), incl. the corrections A6.2 (alpha_K on sigtilde),
#       A6.3 (alpha on sigtilde) and A6.4 (tau1 ~ 1/sigma, not alpha/sigma);
#   (3) the nondimensionalization (eq:DimensionlessMomentumEquation, eq:PressureScale),
#       incl. the corrected forcing scaling f = (alpha_inf nu U/L^2) f* of amendment A7.
#
# Run:  python3 robustness_asymptotics_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

print("=" * 70)
print("Robustness analysis (Section 6) -- symbolic checks")
print("=" * 70)

nu, h, alphaK, Reh, Dah = sp.symbols('nu h alpha_K Re_h Da_h', positive=True)

# Exact parameters with c1=c2=1; encode |a| and sigma through Re_h, Da_h.
amag = Reh*nu/h                              # |a| = Re_h nu/h
sigma = Dah*alphaK*nu/h**2                    # sigma = Da_h alpha_K nu/h^2
tau1NS_inv = nu/h**2 + amag/h                 # = (nu/h^2)(1+Re_h)
tau1 = 1/(alphaK*tau1NS_inv + sigma)
tau2 = h**2/(alphaK*(1/tau1NS_inv))           # h^2/(c1 alpha_K tau1NS),  c1=1
sigtilde = tau1NS_inv*sigma/(tau1NS_inv + sigma/alphaK)

# -------------------------------------------------------------------------
# (1) general asymptotic forms (exact with c1=c2=1)
# -------------------------------------------------------------------------
print("\n[1] General forms eq:GeneralAsymptoticBehaviourOfParameters (c1=c2=1)")
check("tau1 = h^2/(alpha_K nu (1+Re_h+Da_h))",
      sp.simplify(tau1 - h**2/(alphaK*nu*(1+Reh+Dah))) == 0)
check("tau2 = (1+Re_h) nu/alpha_K",
      sp.simplify(tau2 - (1+Reh)*nu/alphaK) == 0)
check("sigtilde = alpha_K (1+Re_h)Da_h/(1+Re_h+Da_h) * nu/h^2   [A6.1: alpha_K present]",
      sp.simplify(sigtilde - alphaK*(1+Reh)*Dah/(1+Reh+Dah)*nu/h**2) == 0)

# -------------------------------------------------------------------------
# (2) regime limits (ratio -> 1 means "~")
# -------------------------------------------------------------------------
print("\n[2] Dominant-regime limits")
def asymptotic(expr, claimed, var, to):
    return sp.limit(sp.simplify(expr/claimed), var, to) == 1

# Dominant viscosity: Re_h, Da_h -> 0
v_t1 = sp.limit(sp.limit(tau1, Reh, 0), Dah, 0)
v_t2 = sp.limit(sp.limit(tau2, Reh, 0), Dah, 0)
check("dom. viscosity: tau1 -> h^2/(alpha_K nu),  tau2 -> nu/alpha_K",
      sp.simplify(v_t1 - h**2/(alphaK*nu)) == 0 and sp.simplify(v_t2 - nu/alphaK) == 0)
check("dom. viscosity: sigtilde ~ alpha_K Da_h nu/h^2   [A6.2 correction]",
      asymptotic(sigtilde.subs(Reh, 0), alphaK*Dah*nu/h**2, Dah, 0))

# Dominant convection: Re_h -> oo
check("dom. convection: tau1 ~ h/(alpha_K |a|)  and  tau2 ~ h|a|/alpha_K",
      asymptotic(tau1, h/(alphaK*amag), Reh, sp.oo) and asymptotic(tau2, h*amag/alphaK, Reh, sp.oo))
check("dom. convection: sigtilde ~ alpha_K Da_h nu/h^2   [A6.3 correction: alpha]",
      asymptotic(sigtilde, alphaK*Dah*nu/h**2, Reh, sp.oo))

# Dominant reaction: Da_h -> oo
check("dom. reaction: tau1 ~ 1/sigma   [A6.4 correction: NOT alpha/sigma]",
      asymptotic(tau1, 1/sigma, Dah, sp.oo))
check("dom. reaction: tau2 ~ (1+Re_h) nu/alpha_K  and  sigtilde ~ alpha_K(1+Re_h) nu/h^2",
      sp.simplify(sp.limit(tau2, Dah, sp.oo) - (1+Reh)*nu/alphaK) == 0
      and asymptotic(sigtilde, alphaK*(1+Reh)*nu/h**2, Dah, sp.oo))

# -------------------------------------------------------------------------
# (3) nondimensionalization (A7) and the pressure scale
# -------------------------------------------------------------------------
print("\n[3] Nondimensionalization eq:DimensionlessMomentumEquation (amendment A7)")
U, L, alpha_inf, sigma_dim = sp.symbols('U L alpha_inf sigma_dim', positive=True)
Re = U*L/nu
Da = sigma_dim*L**2/(alpha_inf*nu)
P = (1 + Re + Da)*U*nu/L                       # eq:PressureScale
mult = L**2/(alpha_inf*nu*U)                    # multiply the dimensional momentum eq. by this

# dimensional prefactor of each term once written in dimensionless variables, times `mult`
conv = (alpha_inf*U**2/L) * mult               # alpha u.grad u
visc = (alpha_inf*nu*U/L**2) * mult            # -2 div(alpha nu Pi grad u)
pres = (alpha_inf*P/L) * mult                  # alpha grad p
reac = (sigma_dim*U) * mult                    # sigma u
forc = (alpha_inf*nu*U/L**2) * mult            # f = (alpha_inf nu U/L^2) f*  (corrected A7)
check("convection coefficient -> Re",            sp.simplify(conv - Re) == 0)
check("viscous coefficient    -> 1 (the 2 stays)", sp.simplify(visc - 1) == 0)
check("pressure coefficient   -> (1+Re+Da)",     sp.simplify(pres - (1 + Re + Da)) == 0)
check("reaction coefficient   -> Da",            sp.simplify(reac - Da) == 0)
check("forcing coefficient    -> 1 (so f = (alpha_inf nu U/L^2) f*, A7)",
      sp.simplify(forc - 1) == 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
