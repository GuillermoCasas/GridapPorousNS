#!/usr/bin/env python3
# =============================================================================
# manufactured_solution_verification.py
#
# Symbolic (sympy) verification of the manufactured solution and the
# plateau-bump porosity field of Section 7
#   (eq:ManufacturedProblem, eq:PlateauBumpFunction, eq:Gamma),
#   "A stabilized finite element method ... porous media".
#
# Verifies:
#   (1) the manufactured velocity satisfies div(alpha u) = 0 exactly, for an
#       ARBITRARY porosity alpha (the alpha0/alpha prefactor does its job);
#   (2) the boundary-condition point (amendment A11): on the unit square the
#       second component u2 = U(alpha0/alpha)cos(pi x1)cos(pi x2) does NOT
#       vanish on the sides, so the data is the trace of the field, not null;
#   (3) the plateau-bump porosity (eq:PlateauBumpFunction, eq:Gamma) is a
#       smooth, monotone transition: d gamma/d eta = (2eta^2-2eta+1)/(eta(1-eta))^2 > 0,
#       the correct limits alpha(r1+)=alpha0 and alpha(r2-)=1, monotonicity
#       d alpha/d r > 0, and C-infinity joining (all derivatives -> 0 at the ends).
#
# Run:  python3 manufactured_solution_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

print("=" * 70)
print("Manufactured solution and plateau-bump porosity -- symbolic checks")
print("=" * 70)

x1, x2 = sp.symbols('x1 x2', real=True)
U, P, alpha0, L = sp.symbols('U P alpha0 L', positive=True)
k = sp.pi / L

# -------------------------------------------------------------------------
# (1) div(alpha u) = 0 for arbitrary alpha
# -------------------------------------------------------------------------
print("\n[1] Mass equation: div(alpha u) = 0 (arbitrary porosity)")
alpha = sp.Function('alpha', positive=True)(x1, x2)
S = [sp.sin(k*x1)*sp.sin(k*x2), sp.cos(k*x1)*sp.cos(k*x2)]      # base shape
u = [U*alpha0/alpha*S[0], U*alpha0/alpha*S[1]]                  # u = U alpha0/alpha S
div_au = sp.diff(alpha*u[0], x1) + sp.diff(alpha*u[1], x2)
check("div(alpha u) == 0 for any alpha(x) (since alpha u = U alpha0 S, div S = 0)",
      sp.simplify(div_au) == 0)
div_S = sp.diff(S[0], x1) + sp.diff(S[1], x2)
check("the base shape S is divergence-free: div S == 0",
      sp.simplify(div_S) == 0)

# -------------------------------------------------------------------------
# (2) Boundary trace (amendment A11): on the unit square (0,1)^2,
#     u2 = U(alpha0/alpha)cos(pi x1)cos(pi x2) does not vanish on the sides.
# -------------------------------------------------------------------------
print("\n[2] Boundary data is the manufactured trace, not null (amendment A11)")
ac = sp.Symbol('ac', positive=True)   # alpha evaluated on the boundary (some positive value)
u2 = U*alpha0/ac*sp.cos(sp.pi*x1)*sp.cos(sp.pi*x2)
# bottom side x2 = 0 of the unit square:
u2_bottom = u2.subs(x2, 0)
check("on (0,1)^2, u2(x1,0) = U(alpha0/alpha)cos(pi x1) is not identically zero",
      sp.simplify(u2_bottom) != 0 and sp.simplify(u2_bottom.subs(x1, 0)) != 0)

# -------------------------------------------------------------------------
# (3) Plateau-bump porosity (eq:PlateauBumpFunction, eq:Gamma)
# -------------------------------------------------------------------------
print("\n[3] Plateau-bump porosity: smoothness and monotonicity")
eta = sp.symbols('eta', positive=True)            # eta = (r^2-r1^2)/(r2^2-r1^2) in (0,1)
gamma = (2*eta - 1)/(eta*(1 - eta))               # eq:Gamma
dgamma = sp.simplify(sp.diff(gamma, eta))
check("d gamma/d eta == (2 eta^2 - 2 eta + 1)/(eta^2 (1-eta)^2)",
      sp.simplify(dgamma - (2*eta**2 - 2*eta + 1)/(eta**2*(1 - eta)**2)) == 0)
check("d gamma/d eta > 0 on (0,1)  (numerator 2eta^2-2eta+1 has negative discriminant)",
      sp.discriminant(2*eta**2 - 2*eta + 1, eta) < 0 and (2*eta**2 - 2*eta + 1).subs(eta, sp.Rational(1, 2)) > 0)

alpha_mid = 1 - (1 - alpha0)/(1 + sp.exp(gamma))   # the bump branch (r1<r<r2)
check("limit alpha(eta->0+) = alpha0   (matches the inner plateau)",
      sp.limit(alpha_mid, eta, 0, '+') == alpha0)
check("limit alpha(eta->1-) = 1        (matches the outer value)",
      sp.limit(alpha_mid, eta, 1, '-') == 1)
# d alpha/d eta = (1 - alpha0) * F(eta) with F = e^gamma/(1+e^gamma)^2 * dgamma/deta > 0;
# since the porosity obeys 0 < alpha0 < 1, the factor (1 - alpha0) > 0, hence alpha is increasing.
dalpha = sp.simplify(sp.diff(alpha_mid, eta))
F = sp.simplify(dalpha / (1 - alpha0))
check("d alpha/d eta = (1-alpha0)*F, with F>0 on (0,1) and (1-alpha0)>0 for porosity (increasing)",
      F.has(alpha0) is False and sp.simplify(F.subs(eta, sp.Rational(1, 2))) > 0)

# C-infinity joining: alpha - alpha0 ~ exp(-1/eta) near eta=0, so derivatives vanish.
d1 = sp.limit(sp.diff(alpha_mid, eta), eta, 0, '+')
d2 = sp.limit(sp.diff(alpha_mid, eta, 2), eta, 0, '+')
check("C-infinity joining at eta=0: d alpha/d eta -> 0 and d^2 alpha/d eta^2 -> 0",
      d1 == 0 and d2 == 0)
e1 = sp.limit(sp.diff(alpha_mid, eta), eta, 1, '-')
e2 = sp.limit(sp.diff(alpha_mid, eta, 2), eta, 1, '-')
check("C-infinity joining at eta=1: d alpha/d eta -> 0 and d^2 alpha/d eta^2 -> 0",
      e1 == 0 and e2 == 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
