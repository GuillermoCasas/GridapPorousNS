#!/usr/bin/env python3
# =============================================================================
# robustness_isolation_verification.py
#
# Symbolic (sympy) verification of the PER-TERM ISOLATION DISPLAYS of Section 6
#   "A stabilized finite element method ... porous media".
#
# WHY THIS SCRIPT EXISTS (coverage-gap closure, 2026-07-21)
# --------------------------------------------------------
# robustness_asymptotics_verification.py checks the tau closed forms and the
# regime LIMITS (tau1,tau2,sigma-tilde as Re_h,Da_h -> 0/oo) and the
# nondimensionalization.  It does NOT reconstruct the *isolation displays*: the
# steps that take a coupled convergence bound (eq:ConvergenceResult specialized
# to a regime), isolate a single left-hand-side term, and re-express the bound in
# terms of E_int(u) or E_int(p) using E_int(psi) = <scale_psi> * E*_int(psi) with
# the equal-order identification E*_int(u) = E*_int(p) = E*_int.
#
# That blind spot is exactly where the external audit found an ALGEBRAIC ERROR
# in eq:DominantPressureGradientXTermEstimate (line ~1052): the printed factor
#       ||a||_inf / sqrt(P) + 1                         (WRONG)
# does not follow from eq:DominantConvectionEstimate; the correct factor is
#       ||a||_inf * U / P + 1                           (CORRECT)
# (both reduce to O(1) under P ~ U^2, so the final "~" conclusion is unchanged,
#  but the intermediate display is wrong and did not follow from its predecessor).
#
# This script encodes the coupled regime bounds (eq:ConvergenceResultDominant*)
# and mechanically performs each isolation, asserting the printed factor of
#   eq:DominantViscosityVelocityGradientEstimate   (1023)
#   eq:DominantViscosityPressureGradientEstimate   (1027)
#   eq:DominantConvectionEstimate                  (1043)
#   eq:DominantConvectionXTermEstimate             (1048)
#   eq:DominantPressureGradientXTermEstimate       (1052)  <-- the corrected one
#   eq:DominantReactionVelocityGradientEstimate    (1068-69)
#   eq:DominantReactionPressureGradientEstimate    (1075)
# Each isolation is compared to the coupled bound it is derived from.  For 1052
# the check is DISCRIMINATING: it asserts the corrected ||a|| U / P + 1 form
# holds AND that the printed ||a|| / sqrt(P) + 1 form FAILS to follow from the
# predecessor (so a regression to the wrong display is caught).
#
# Run:  python3 robustness_isolation_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

print("=" * 70)
print("Robustness per-term isolation displays (Section 6) -- symbolic checks")
print("=" * 70)

# Positive physical/interpolation symbols.
#   U,P     : velocity & pressure characteristic scales, E_int(u)=U Es, E_int(p)=P Es
#   amag    : ||a||_inf     nu,h : viscosity, mesh size
#   sig     : sigma (dimensional reaction)   Reh : elemental Reynolds number
#   a0      : alpha_0 (min porosity)   Es : the common dimensionless interp error E*_int
#   Eu,Ep   : E_int(u), E_int(p)   (Eu = U Es, Ep = P Es for equal order)
U, P, amag, nu, h, sig, Reh, a0, Es = sp.symbols('U P amag nu h sigma Re_h alpha0 Es', positive=True)
Eu = U * Es
Ep = P * Es

# The isolation mechanic:  a coupled bound  (LHS terms) <~ RHS,  where
#   RHS = (1/a0**q) * (sum of  coef_k * E_int(psi_k)/h).
# To bound a single LHS term T (possibly with a fixed prefactor m so that the
# printed left side is m*T) and express the result "in E_int(target)":
#   m*T <~ RHS  =>  T <~ (1/m) * RHS,   then substitute Es -> E_target/scale_target
#   so the whole RHS is a multiple of E_target/h; return that coefficient.
def isolate_coeff(RHS_over_Es_times_h, prefactor_m, target_scale):
    """Coupled bound:  m_coupled * ||term|| <~ RHS = RHS_over_Es_times_h * Es/h.
    Printed display:   m_printed * ||term|| <~ factor * E_target/h,  E_target = target_scale*Es.
    Then  m_printed*||term|| = (m_printed/m_coupled)*(m_coupled*||term||) <~ (m_printed/m_coupled)*RHS,
    and substituting Es = E_target/target_scale gives
        factor = (m_printed/m_coupled) * RHS_over_Es_times_h / target_scale.
    Pass prefactor_m := m_coupled / m_printed (the RHS is DIVIDED by it)."""
    return sp.simplify(RHS_over_Es_times_h / prefactor_m / target_scale)

# =====================================================================
# (A) Dominant viscosity.  Coupled eq:ConvergenceResultDominantViscosity (1015):
#   ||Pi grad e_u|| + (h/nu)||grad e_p||_h + a0^{-1/2}||div(a e_u)||_h
#        <~ a0^{-1/2} ( E_int(u)/h + (1/nu) E_int(p) )
#   RHS = a0^{-1/2} ( U/h + P/nu ) Es       [E_int(u)/h = U Es/h ; (1/nu)E_int(p)= (P/nu)Es = (P/nu)*h*(Es/h)]
# Note the pressure RHS term (1/nu)E_int(p) has NO 1/h, so as a multiple of Es/h it is (P h/nu).
print("\n[A] Dominant viscosity (Re_h,Da_h -> 0)")
RHS_visc = (U + P*h/nu) / a0**sp.Rational(1,2)          # coefficient of Es/h
# 1023: isolate ||Pi grad e_u|| (prefactor 1), express in E_int(u) (scale U)
c1023 = isolate_coeff(RHS_visc, 1, U)
check("eq 1023  factor = a0^{-1/2}(1 + P h/(U nu))",
      sp.simplify(c1023 - (1 + P*h/(U*nu))/a0**sp.Rational(1,2)) == 0)
# 1027: isolate ||grad e_p|| with printed prefactor (h/nu), express in E_int(p) (scale P)
c1027 = isolate_coeff(RHS_visc, h/nu, P)
check("eq 1027  factor = a0^{-1/2}(U nu/(P h) + 1)   [CORRECTED display]",
      sp.simplify(c1027 - (U*nu/(P*h) + 1)/a0**sp.Rational(1,2)) == 0)

# =====================================================================
# (B) Dominant convection.  Coupled eq:ConvergenceResultDominantConvection (1039):
#   (1/||a||)||a.grad e_u + grad e_p||_h + a0^{-1/2}||div(a e_u)||_h
#        <~ a0^{-1/2} ( E_int(u)/h + (1/||a||) E_int(p)/h )
#   RHS = a0^{-1/2} ( U + P/||a|| ) Es/h
print("\n[B] Dominant convection (Re_h -> oo)")
RHS_conv = (U + P/amag) / a0**sp.Rational(1,2)          # coefficient of Es/h
# 1043: the combined term, prefactor 1, kept in E*_int (scale = 1, i.e. leave as Es)
c1043 = isolate_coeff(RHS_conv, 1, 1)
check("eq 1043  factor = a0^{-1/2}(U + P/||a||)   (coupled, in E*_int)",
      sp.simplify(c1043 - (U + P/amag)/a0**sp.Rational(1,2)) == 0)
# 1048: dominant a.grad e_u.  The 1/||a|| prefactor is kept on BOTH sides
# (m_coupled = m_printed = 1/||a||), so m := m_coupled/m_printed = 1; express in E_int(u) (scale U).
c1048 = isolate_coeff(RHS_conv, 1, U)
check("eq 1048  pre-sub factor = a0^{-1/2}(1 + P/(U||a||)); with P~U^2 -> (1+U/||a||)",
      sp.simplify(c1048 - (1 + P/(U*amag))/a0**sp.Rational(1,2)) == 0
      and sp.simplify(c1048.subs(P, U**2) - (1 + U/amag)/a0**sp.Rational(1,2)) == 0)
# 1052: dominant grad e_p, multiply through by ||a|| (prefactor 1/||a||), express in E_int(p) (scale P)
c1052 = isolate_coeff(RHS_conv, 1/amag, P)
correct_1052 = (amag*U/P + 1) / a0**sp.Rational(1,2)
printed_1052 = (amag/sp.sqrt(P) + 1) / a0**sp.Rational(1,2)   # the WRONG printed display
check("eq 1052  factor = a0^{-1/2}(||a|| U/P + 1)   [CORRECTED; follows from 1043]",
      sp.simplify(c1052 - correct_1052) == 0)
check("eq 1052  printed a0^{-1/2}(||a||/sqrt(P)+1) does NOT follow from 1043 (discriminating)",
      sp.simplify(c1052 - printed_1052) != 0)
check("eq 1052  both corrected & printed reduce to a0^{-1/2}(1+||a||/U) under P~U^2 "
      "(so the final ~ conclusion is unaffected)",
      sp.simplify(correct_1052.subs(P, U**2) - printed_1052.subs(P, U**2)) == 0)

# =====================================================================
# (C) Dominant reaction.  Coupled eq (1063-64):
#   ||Pi grad e_u|| + (1+Re_h)^{1/2}||e_u||/h + (a0^{1/2}/(sig^{1/2} nu^{1/2}))||grad e_p||_h
#        + ((1+Re_h)^{1/2}/a0^{1/2})||div(a e_u)||_h
#        <~ a0^{-1/2} ( (1+Re_h)^{1/2} E_int(u)/h + (1/(sig^{1/2} nu^{1/2})) E_int(p)/h )
#   RHS = a0^{-1/2} ( (1+Re_h)^{1/2} U + P/(sig^{1/2} nu^{1/2}) ) Es/h
print("\n[C] Dominant reaction (Da_h -> oo)")
rr = sp.sqrt(1 + Reh)
RHS_reac = (rr*U + P/(sp.sqrt(sig)*sp.sqrt(nu))) / a0**sp.Rational(1,2)   # coeff of Es/h
# 1068: isolate ||Pi grad e_u|| (prefactor 1), express in E_int(u) (scale U)
c1068 = isolate_coeff(RHS_reac, 1, U)
check("eq 1068  factor = a0^{-1/2}((1+Re_h)^{1/2} + (1/(sig^{1/2}nu^{1/2}))(P/U))",
      sp.simplify(c1068 - (rr + P/(U*sp.sqrt(sig)*sp.sqrt(nu)))/a0**sp.Rational(1,2)) == 0)
# 1069: the "~" step uses P ~ Da*U*nu/L with sigma = Da*alpha_inf*nu/L^2, i.e.
#   sig = Da*ainf*nu/L^2 and P = Da*U*nu/L  =>  (1/(sig^{1/2}nu^{1/2}))(P/U) = Da^{1/2}/ainf^{1/2}
Da, ainf, L = sp.symbols('Da alpha_inf L', positive=True)
sub_second = (P/(U*sp.sqrt(sig)*sp.sqrt(nu))).subs({sig: Da*ainf*nu/L**2, P: Da*U*nu/L})
check("eq 1069  second term (1/(sig^{1/2}nu^{1/2}))(P/U) -> Da^{1/2}/alpha_inf^{1/2}",
      sp.simplify(sub_second - sp.sqrt(Da)/sp.sqrt(ainf)) == 0)
# 1075: isolate ||grad e_p|| with printed prefactor a0^{1/2}/(sig^{1/2}nu^{1/2}), in E_int(p) (scale P)
m1075 = a0**sp.Rational(1,2)/(sp.sqrt(sig)*sp.sqrt(nu))
c1075 = isolate_coeff(RHS_reac, m1075, P)
check("eq 1075  factor = a0^{-1}( sig^{1/2} nu^{1/2}(1+Re_h)^{1/2} U/P + 1 )",
      sp.simplify(c1075 - (sp.sqrt(sig)*sp.sqrt(nu)*rr*U/P + 1)/a0) == 0)
# 1075 "~": with the reaction-limit scalings, sig^{1/2}nu^{1/2} U/P -> alpha_inf^{1/2} Da^{-1/2}
sub_1075 = (sp.sqrt(sig)*sp.sqrt(nu)*U/P).subs({sig: Da*ainf*nu/L**2, P: Da*U*nu/L})
check("eq 1075  printed a0^{-1}((1+Re_h)^{1/2} alpha_inf^{1/2} Da^{-1/2} + 1) matches after scalings",
      sp.simplify(sub_1075 - sp.sqrt(ainf)/sp.sqrt(Da)) == 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
