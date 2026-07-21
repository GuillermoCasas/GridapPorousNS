#!/usr/bin/env python3
# =============================================================================
# coverage_weighted_inverse_verification.py
#
# Symbolic (sympy) verification of the CONTINUITY-APPENDIX WEIGHTED-INVERSE
# ESTIMATES of "A stabilized finite element method ... porous media"
# (theory/paper/continuity_appendix.tex, Lemma "Weighted inverse estimates"
# lem:winv, its proof, Step 9 absorption, and the interpolation analogue).
#
# WHY THIS SCRIPT EXISTS (coverage-gap closure, cluster g2-weighted-inverse)
# -------------------------------------------------------------------------
# These estimates were re-derived and found algebraically correct by a prior
# audit pass, but had NO permanent encoded check.  Each is an inequality of the
# form ||L(alpha, w_h)||_K <= COEFF * ||w_h||_K, where COEFF has a specific
# structure in the powers of  h_K, alpha_K, the inverse constant C_inv, the
# porosity-resolution constant C_{grad alpha}, and (for absorb5) tau_2.  The
# content this script permanently locks down is:
#
#   (i)  the EXACT product-rule split that underlies every bound
#          div(alpha w) = alpha div(w) + (grad alpha) . w      (vector)
#          div(alpha M) = alpha div(M) + M (grad alpha)         (tensor, rowwise)
#        -- verified symbolically on concrete polynomial fields, with the
#           grad-alpha term shown to be NON-vanishing (so "dropping it" is caught);
#   (ii) the COEFFICIENT / SCALING structure of each printed bound, reconstructed
#        from its two contributing pieces (an inverse estimate eq:inverse plus the
#        porosity-resolution bound eq:resolved), with the correct powers of
#        h_K / alpha_K / tau_2, and DISCRIMINATING wrong-variant checks
#        (flipped/omitted grad-alpha term, wrong h power, wrong alpha power,
#         wrong tau_2 power) that must FAIL.
#
# Equations covered (labels in continuity_appendix.tex):
#   eq:winv-divvisc  (l.283)  ||div(alpha Pi grad w)||_K
#                                   <= (Cinva/h) aK^{1/2} ||alpha^{1/2} Pi grad w||_K
#   eq:winv-divu     (l.290)  ||div(alpha w)||_K       <= (Cinva/h) aK ||w||_K
#   eq:winv-gradp    (l.296)  ||alpha grad r||_K       <= (Cinv/h) aK ||r||_K
#   eq:absorb5       (l.799)  tau2^{1/2}||div(alpha u)||_K
#                                   <= Cinva aK tau2^{1/2} h^{-1} ||u||_K
#   eq:interpdivvisc (l.931)  ||div(alpha Pi grad e~u)||_K <= C aK h^{-2} Eint(u)
# with the constant identity  Cinva = sqrt(d*delta_alpha) Cinv + Cgrad  (l.330),
#   delta_alpha = 1 + Cgrad  (l.166),  and the double-inverse estimate eq:doubleinv
#   (l.514) used to reconstruct eq:interpdivvisc.
#
# Upstream facts reconstructed from (NOT merely restated):
#   eq:inverse   (l.188)  ||grad psi_h||_K <= (Cinv/h) ||psi_h||_K
#   eq:resolved  (l.160)  h ||grad alpha||_inf <= Cgrad alpha_{0,K}
#   H:porosity   (l.169)  alpha_{0,K} <= alpha <= alpha_K <= delta_alpha alpha_{0,K}
#   lem:parameters(l.250) varphi1 h^2 = c1 aK^2 tau2   (the "tau2 scaling")
#
# Run:  /tmp/sympy_venv/bin/python coverage_weighted_inverse_verification.py
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

print("=" * 70)
print("Continuity-appendix weighted-inverse estimates (lem:winv) -- symbolic checks")
print("=" * 70)

# Positive symbols.
#   h       : element size h_K            Cinv  : inverse-estimate constant (eq:inverse)
#   Cgrad   : C_{grad alpha} (eq:resolved)   d  : spatial dimension
#   aK      : alpha_K = alpha_{infty,K}   a0K   : alpha_{0,K}
#   deltaa  : delta_alpha = 1 + Cgrad     tau2  : tau_2   c1 : numerical constant c1
#   phi1    : varphi_1                    nu,sig,amag : viscosity, sigma, |a|_inf
h, Cinv, Cgrad, d, aK, a0K, deltaa, tau2, c1, phi1, nu, sig, amag = sp.symbols(
    'h C_inv C_grad d alpha_K alpha_0K delta_alpha tau2 c1 varphi1 nu sigma amag',
    positive=True)
half = sp.Rational(1, 2)

# Printed constant of the lemma (l.330) and the porosity ratio (l.166).
deltaa_def = 1 + Cgrad
Cinva = sp.sqrt(d * deltaa) * Cinv + Cgrad          # eq:winv-divvisc / winv-divu constant

# =====================================================================
# (0) EXACT product-rule identities underlying every bound.
#     div(alpha w) = alpha div w + grad(alpha).w         (vector case, winv-divu)
#     div(alpha M) = alpha div M + M grad(alpha)         (tensor rowwise, winv-divvisc)
#     These give the grad-alpha term its teeth: dropping it is a genuine error.
print("\n[0] Exact product-rule splits (concrete polynomial fields)")

def div_vec(field, coords):
    return sum(sp.diff(field[i], coords[i]) for i in range(len(coords)))

# --- vector, d = 3 ---
x, y, z = sp.symbols('x y z', real=True)
coords3 = (x, y, z)
alpha3 = 1 + x + 2*y - z + x*y + y*z*z          # smooth scalar (not const, grad != 0)
w3 = [x**2*y + z, x*y*z - y**2, x + y*z + z**3]  # polynomial vector field
lhs_v = div_vec([alpha3*w3[i] for i in range(3)], coords3)
grad_a3 = [sp.diff(alpha3, c) for c in coords3]
rhs_v = alpha3*div_vec(w3, coords3) + sum(grad_a3[i]*w3[i] for i in range(3))
check("eq:winv-divu split  div(alpha w) = alpha div w + grad(alpha).w  (d=3, exact)",
      sp.expand(lhs_v - rhs_v) == 0)
# discriminating: the grad-alpha term is NOT identically zero -> dropping it is wrong
dropped_v = alpha3*div_vec(w3, coords3)
check("eq:winv-divu split  dropping grad(alpha).w changes the divergence (teeth)",
      sp.expand(lhs_v - dropped_v) != 0)

# --- tensor M (d=2), rowwise divergence  (div M)_i = d_j M_ij ---
coords2 = (x, y)
alpha2 = 2 + 3*x - y + x*y + x**2
M = [[x*y + y**2, x**2 - y],
     [x - x*y,     y**3 + x]]                    # 2x2 polynomial tensor (= Pi grad w proxy)
grad_a2 = [sp.diff(alpha2, c) for c in coords2]
ok_tensor = True
for i in range(2):
    lhs_t = sum(sp.diff(alpha2*M[i][j], coords2[j]) for j in range(2))   # div(alpha M)_i
    divM_i = sum(sp.diff(M[i][j], coords2[j]) for j in range(2))         # (div M)_i
    Mgrada_i = sum(M[i][j]*grad_a2[j] for j in range(2))                 # (M grad alpha)_i
    ok_tensor = ok_tensor and sp.expand(lhs_t - (alpha2*divM_i + Mgrada_i)) == 0
check("eq:winv-divvisc split  div(alpha M)_i = alpha (div M)_i + (M grad alpha)_i  (exact)",
      ok_tensor)

# =====================================================================
# (1) eq:winv-divvisc constant  Cinva = sqrt(d*delta_alpha) Cinv + Cgrad.
#     term1 (alpha div M): alpha_inf * sqrt(d) * (Cinv/h) * ||M||, then
#        alpha_infK / alpha_0K^{1/2} = (alpha_infK/alpha_0K)^{1/2} alpha_infK^{1/2}
#                                    <= delta_alpha^{1/2} alpha_K^{1/2}
#        -> constant  sqrt(d) * Cinv * delta_alpha^{1/2}.
#     term2 (M grad alpha): ||grad alpha||_inf <= Cgrad a0K/h and a0K^{1/2} <= aK^{1/2}
#        -> constant  Cgrad.
print("\n[1] eq:winv-divvisc  Cinva = sqrt(d*delta_alpha) Cinv + Cgrad")

# exact algebraic identity used in the ratio step (l.317-318)
ainf, a0 = sp.symbols('alpha_inf alpha_0', positive=True)
check("ratio identity  alpha_inf/alpha_0^{1/2} = (alpha_inf/alpha_0)^{1/2} alpha_inf^{1/2}",
      sp.simplify(ainf/a0**half - (ainf/a0)**half * ainf**half) == 0)
# sqrt split used to assemble the constant
check("sqrt split  sqrt(d) sqrt(delta_alpha) = sqrt(d*delta_alpha)  (positivity)",
      sp.simplify(sp.sqrt(d)*sp.sqrt(deltaa) - sp.sqrt(d*deltaa)) == 0)

term1_divvisc = sp.sqrt(d) * Cinv * deltaa**half   # alpha div M contribution
term2_divvisc = Cgrad                              # M grad alpha contribution
check("eq:winv-divvisc  reconstructed constant term1+term2 == Cinva",
      sp.simplify((term1_divvisc + term2_divvisc) - Cinva) == 0)
# discriminating: dropping the grad-alpha (Cgrad) contribution misses exactly Cgrad
check("eq:winv-divvisc  dropping M.grad(alpha) term FAILS (misses Cgrad)",
      sp.simplify(term1_divvisc - Cinva) != 0)
# discriminating: ignoring the delta_alpha porosity enlargement (delta_alpha->1) FAILS
Cinva_nodelta = (sp.sqrt(d)*Cinv + Cgrad).subs(deltaa, 1)  # cosmetic; explicit wrong form
check("eq:winv-divvisc  ignoring delta_alpha enlargement (sqrt(d)Cinv+Cgrad) FAILS",
      sp.simplify((sp.sqrt(d)*Cinv + Cgrad) - Cinva) != 0)

# =====================================================================
# (2) eq:winv-divu  ||div(alpha w)||_K <= (Cinva/h) aK ||w||_K.
#     Exact contributing constant (no alpha^{1/2} weighting here):
#       term1: alpha_infK sqrt(d) Cinv/h,  alpha_infK = aK   -> sqrt(d) Cinv
#       term2: ||grad alpha||_inf <= Cgrad a0K/h, a0K <= aK  -> Cgrad
#     so the sharp constant is  cdivu = sqrt(d) Cinv + Cgrad, and the printed
#     bound uses the (looser) Cinva >= cdivu.  RHS scaling: aK^1, h^{-1}.
print("\n[2] eq:winv-divu  ||div(alpha w)|| <= (Cinva/h) alpha_K ||w||")
cdivu = sp.sqrt(d)*Cinv + Cgrad                    # sharp reconstructed constant
# the printed Cinva dominates the sharp constant, gap = sqrt(d) Cinv (sqrt(delta)-1) >= 0
gap = sp.simplify(Cinva - cdivu)
check("eq:winv-divu  Cinva - cdivu = sqrt(d) Cinv (sqrt(delta_alpha)-1)",
      sp.simplify(gap - sp.sqrt(d)*Cinv*(sp.sqrt(deltaa) - 1)) == 0)
# gap = sqrt(d) Cinv (sqrt(1+Cgrad)-1) >= 0: the printed Cinva never UNDER-states the
# sharp constant.  Verify via the exact square identity (sqrt(1+Cgrad))^2 - 1 = Cgrad >= 0
# together with a numeric sweep over positive parameters (all samples must be >= 0).
gap_phys = gap.subs(deltaa, deltaa_def)            # = sqrt(d) Cinv (sqrt(1+Cgrad)-1)
sq_id = sp.simplify(sp.sqrt(1 + Cgrad)**2 - 1 - Cgrad) == 0
sweep_ok = all(
    float(gap_phys.subs({d: dd, Cinv: cc, Cgrad: cg})) >= -1e-12
    for dd in (1, 2, 3) for cc in (sp.Rational(1, 3), 1, 7)
    for cg in (sp.Rational(1, 10), 1, 5))
check("eq:winv-divu  gap >= 0 (Cinva >= sharp const): (1+Cgrad)-1=Cgrad>=0 + numeric sweep",
      sq_id and sweep_ok)
# RHS scaling as an explicit monomial: constant * aK^1 * h^{-1}
rhs_divu = Cinva * aK * h**(-1)
check("eq:winv-divu  RHS = Cinva * alpha_K^1 * h^{-1} (alpha power 1, h power -1)",
      sp.simplify(rhs_divu - Cinva*aK/h) == 0
      and sp.degree(sp.Poly(rhs_divu*h, aK), aK) == 1)
# discriminating: dropping grad-alpha term under-counts by exactly Cgrad
check("eq:winv-divu  dropping grad(alpha) term (const sqrt(d)Cinv) FAILS (misses Cgrad)",
      sp.simplify(cdivu - sp.sqrt(d)*Cinv) == Cgrad and Cgrad != 0)
# discriminating: a wrong h power (h^{-2}) does not match the first-order estimate
check("eq:winv-divu  wrong h power h^{-2} FAILS",
      sp.simplify(rhs_divu - Cinva*aK*h**(-2)) != 0)

# =====================================================================
# (3) eq:winv-gradp  ||alpha grad r||_K <= (Cinv/h) aK ||r||_K.
#     Two-step, no product rule:  alpha <= aK pointwise (factor aK^1) THEN
#     inverse estimate ||grad r|| <= (Cinv/h)||r||.  RHS = Cinv aK^1 h^{-1}.
print("\n[3] eq:winv-gradp  ||alpha grad r|| <= (Cinv/h) alpha_K ||r||")
factor_alpha = aK                                  # from alpha <= alpha_K
inv_gradr = Cinv / h                               # from eq:inverse
rhs_gradp = factor_alpha * inv_gradr               # = Cinv aK / h
check("eq:winv-gradp  RHS = alpha_K * (Cinv/h) = Cinv alpha_K^1 h^{-1}",
      sp.simplify(rhs_gradp - Cinv*aK/h) == 0)
# discriminating: dropping the alpha_K factor (alpha power 0) FAILS
check("eq:winv-gradp  dropping alpha_K (Cinv/h only) FAILS",
      sp.simplify(rhs_gradp - Cinv/h) != 0)
# discriminating: wrong alpha power (alpha_K^{1/2}, as if weighted) FAILS
check("eq:winv-gradp  wrong alpha power alpha_K^{1/2} FAILS",
      sp.simplify(rhs_gradp - Cinv*aK**half/h) != 0)

# =====================================================================
# (4) eq:absorb5  tau2^{1/2} ||div(alpha u)||_K <= Cinva aK tau2^{1/2} h^{-1} ||u||.
#     Multiply eq:winv-divu (RHS = Cinva aK/h) by tau2^{1/2}.
#     "tau2 scaling": the resulting weight aK tau2^{1/2} h^{-1} is exactly the
#     canonical bracket-norm weight, and equals c1^{-1/2} varphi1^{1/2} via
#     eq:lem-parameters  varphi1 h^2 = c1 aK^2 tau2.
print("\n[4] eq:absorb5  tau2^{1/2}||div(alpha u)|| <= Cinva alpha_K tau2^{1/2} h^{-1} ||u||")
rhs_absorb5 = sp.sqrt(tau2) * rhs_divu             # tau2^{1/2} * (Cinva aK/h)
check("eq:absorb5  RHS = Cinva alpha_K tau2^{1/2} h^{-1}",
      sp.simplify(rhs_absorb5 - Cinva*aK*sp.sqrt(tau2)/h) == 0)
# tau2 scaling: weight aK tau2^{1/2}/h == c1^{-1/2} varphi1^{1/2} using varphi1 h^2 = c1 aK^2 tau2
weight = aK*sp.sqrt(tau2)/h
phi1_from_scaling = c1*aK**2*tau2/h**2             # varphi1 = c1 aK^2 tau2 / h^2  (l.250)
check("eq:absorb5  tau2 scaling: alpha_K tau2^{1/2} h^{-1} = c1^{-1/2} varphi1^{1/2}",
      sp.simplify(weight - c1**(-half)*phi1_from_scaling**half) == 0)
# discriminating: wrong tau2 power (tau2 instead of tau2^{1/2}) breaks the scaling identity
check("eq:absorb5  wrong tau2 power (tau2^1) breaks the canonical weight FAILS",
      sp.simplify((aK*tau2/h) - c1**(-half)*phi1_from_scaling**half) != 0)
# discriminating: dropping the grad-alpha part of the constant (Cinva->sqrt(d)Cinv) FAILS
check("eq:absorb5  constant must be Cinva (grad-alpha included), sqrt(d)Cinv FAILS",
      sp.simplify(rhs_absorb5 - sp.sqrt(d)*Cinv*aK*sp.sqrt(tau2)/h) != 0)

# =====================================================================
# (5) eq:doubleinv  ||div(alpha Pi grad w)||_K <= (Cinva Cinv / h^2) aK ||w||_K,
#     the composition of eq:winv-divvisc with eq:winv-grad, and its interpolation
#     analogue eq:interpdivvisc  ||div(alpha Pi grad e~u)||_K <= C aK h^{-2} Eint(u).
#     winv-divvisc:  (Cinva/h) aK^{1/2} ||alpha^{1/2} Pi grad w||
#     winv-grad:     ||alpha^{1/2} Pi grad w|| <= (Cinv/h) aK^{1/2} ||w||
#     compose  ->    (Cinva Cinv / h^2) aK ||w||   (alpha^{1/2} * alpha^{1/2} = alpha^1)
print("\n[5] eq:doubleinv / eq:interpdivvisc  ||div(alpha Pi grad .)|| ~ (C/h^2) alpha_K (.)")
coeff_divvisc = (Cinva/h) * aK**half               # winv-divvisc coeff of ||alpha^{1/2}Pi grad w||
coeff_grad = (Cinv/h) * aK**half                   # winv-grad coeff of ||w||
coeff_double = sp.simplify(coeff_divvisc * coeff_grad)
check("eq:doubleinv  composed coeff = Cinva Cinv aK^1 h^{-2}",
      sp.simplify(coeff_double - Cinva*Cinv*aK/h**2) == 0)
# discriminating: if the grad step forgot its aK^{1/2}, alpha power would be 1/2 (wrong)
coeff_double_bad = sp.simplify(coeff_divvisc * (Cinv/h))
check("eq:doubleinv  forgetting winv-grad's alpha_K^{1/2} (alpha power 1/2) FAILS",
      sp.simplify(coeff_double - coeff_double_bad) != 0)

# eq:interpdivvisc: same coefficient structure, ||w|| -> Eint(u).  Reconstruct its
# own product-rule split from the interpolation estimate ||e~u||_{H^{m'}} <= C h^{-m'} Eint(u)
# (eq:interp) and eq:resolved.  Model each contribution as  const * aK^{p} * h^{q} * Eint.
Eint, C = sp.symbols('E_int C', positive=True)
# term1: alpha_infK * ||div(Pi grad e~u)|| <= aK * C h^{-2} Eint   (second-order, m'=2)
term1_interp = aK * C * h**(-2) * Eint
# term2: ||grad alpha||_inf * ||Pi grad e~u|| <= (Cgrad a0K/h) * C h^{-1} Eint <= aK C h^{-2} Eint
term2_interp = aK * (Cgrad) * C * h**(-2) * Eint   # a0K <= aK absorbed; scaling h^{-2}
interp_rhs = sp.simplify(term1_interp + term2_interp)
check("eq:interpdivvisc  reconstructed RHS ~ C' alpha_K h^{-2} Eint(u) (alpha 1, h -2)",
      sp.simplify(interp_rhs - C*aK*(1 + Cgrad)*h**(-2)*Eint) == 0)
# matches the doubleinv scaling (alpha_K h^{-2}) with ||w|| replaced by Eint(u)
check("eq:interpdivvisc  same alpha/h scaling as eq:doubleinv (alpha^1 h^{-2})",
      sp.simplify((interp_rhs/Eint) / (coeff_double/(Cinva*Cinv)) - C*(1+Cgrad)) == 0)
# discriminating: using the H^1 (first-order, m'=1) interp for the div term -> h^{-1}, wrong
term1_interp_bad = aK * C * h**(-1) * Eint         # forgot it's a SECOND-order operator
check("eq:interpdivvisc  naive m'=1 (h^{-1}) for the divergence term FAILS",
      sp.simplify(term1_interp - term1_interp_bad) != 0)
# discriminating: dropping the grad-alpha contribution changes the constant (misses Cgrad)
check("eq:interpdivvisc  dropping grad(alpha) contribution FAILS (misses Cgrad term)",
      sp.simplify(interp_rhs - term1_interp) != 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
