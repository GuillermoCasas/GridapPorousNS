#!/usr/bin/env python3
# =============================================================================
# coverage_ibp_verification.py
#
# Symbolic (sympy) verification of the CONTINUITY-APPENDIX integration-by-parts
# identities of
#   "A stabilized finite element method for incompressible, inertial flows in
#    inhomogeneous porous media" (Casas, Gonzalez-Usua, Codina, de-Pouplana).
#
# WHY THIS SCRIPT EXISTS (coverage-gap closure, 2026-07-21)
# --------------------------------------------------------
# continuity_appendix.tex Step 6a records three IBP identities (lines ~576-606)
# used to rewrite the convective/pressure group N.  They are algebraically
# correct (re-verified by a prior pass) but had NO permanent encoded check.
# This script reconstructs each FROM ITS UPSTREAM product-rule identity on a
# symbolic box cell, so a future regression (sign flip, dropped alpha, dropped
# div-free hypothesis) is caught.  Encoded:
#
#   eq:skew       (~line 573)  Global skew-symmetry of convection.
#                 Pointwise:  (alpha a.grad u).v + (alpha a.grad v).u
#                                = alpha a.grad(u.v),
#                 and, with div(alpha a)=0 (H:advection) on a periodic cell,
#                 the symmetric convective form integrates to a boundary term
#                 (= 0 here):  (v, alpha a.grad u) = -(u, alpha a.grad v).
#
#   eq:globalibp  (~line 577)  Global pressure/mass IBP:
#                 (v, alpha grad p) = -(p, div(alpha v)),
#                 and the analogue  (q, div(alpha u)) = -(u, alpha grad q).
#
#   eq:elemibp    (~line 583)  Elementwise convective IBP, tau1 constant on K,
#                 a continuous, div(alpha a)=0 in K:
#                 (tau1 alpha a.grad u, v)_K = -(tau1 alpha a.grad v, u)_K
#                                + tau1 * int_{dK} alpha (n.a)(u.v) dGamma.
#
# The interior identities are made EXACT by working with symbolic fields (for
# the pointwise part) and with periodic trig fields on [0,2pi]^3 / polynomial
# fields on the box [0,1]^3 (for the integrated parts).  Every identity is also
# checked to be DISCRIMINATING: a plausible WRONG variant (flipped sign, dropped
# alpha, dropped div-free hypothesis) is asserted to FAIL.
#
# Run:  python3 coverage_ibp_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

x = sp.symbols('x1 x2 x3', real=True)
D = 3
def d(f, i): return sp.diff(f, x[i])
def div(vec): return sum(d(vec[i], i) for i in range(D))
def adv(b, f):                      # (b.grad f):  f scalar -> scalar,  f vector -> vector
    if isinstance(f, (list, tuple)):
        return [sum(b[i]*d(f[k], i) for i in range(D)) for k in range(D)]
    return sum(b[i]*d(f, i) for i in range(D))
def dot(a, b): return sum(a[k]*b[k] for k in range(D))

print("=" * 70)
print("Continuity-appendix integration-by-parts identities -- symbolic checks")
print("=" * 70)

# =====================================================================
# (1) eq:skew -- POINTWISE product-rule identity (exact, symbolic fields).
#     (alpha a.grad u).v + (alpha a.grad v).u == alpha a.grad(u.v)
# =====================================================================
print("\n[1] eq:skew  pointwise:  (alpha a.grad u).v + (alpha a.grad v).u = alpha a.grad(u.v)")
alpha = sp.Function('alpha', positive=True)(*x)
a  = [sp.Function(f'a{i+1}')(*x) for i in range(D)]      # advection field a(x)
uu = [sp.Function(f'u{i+1}')(*x) for i in range(D)]      # velocity u(x)
vv = [sp.Function(f'v{i+1}')(*x) for i in range(D)]      # velocity v(x)

# b := alpha a  (the product that actually appears; both entries carry alpha)
lhs_pt = (dot([alpha*c for c in adv(a, uu)], vv)
          + dot([alpha*c for c in adv(a, vv)], uu))
rhs_pt = alpha * adv(a, dot(uu, vv))                     # alpha a.grad(u.v)
check("eq:skew pointwise identity holds (varying alpha, a, u, v)",
      sp.simplify(lhs_pt - rhs_pt) == 0)

# Discriminating: a sign flip on the second convective term must NOT reproduce RHS.
lhs_pt_flip = (dot([alpha*c for c in adv(a, uu)], vv)
               - dot([alpha*c for c in adv(a, vv)], uu))
check("eq:skew pointwise: FLIPPED sign (u-term minus v-term) does NOT equal alpha a.grad(u.v)",
      sp.simplify(lhs_pt_flip - rhs_pt) != 0)
# Discriminating: dropping alpha off the second term must NOT reproduce RHS (alpha varies).
lhs_pt_noalpha = (dot([alpha*c for c in adv(a, uu)], vv)
                  + dot(adv(a, vv), uu))                 # alpha missing on 2nd term
check("eq:skew pointwise: DROPPING alpha on one term does NOT equal alpha a.grad(u.v)",
      sp.simplify(lhs_pt_noalpha - rhs_pt) != 0)

# =====================================================================
# (2) eq:skew -- GLOBAL skew form on a periodic cell, using div(alpha a)=0.
#     int (v, b.grad u) + int (u, b.grad v) = int b.grad(u.v)
#                                           = -int div(b)(u.v) + bdry = 0
#     (bdry vanishes by periodicity; div(b)=0 by H:advection)  =>
#     (v, alpha a.grad u) = -(u, alpha a.grad v).
# =====================================================================
print("\n[2] eq:skew  global on periodic cell (div(alpha a)=0):  (v,alpha a.grad u) = -(u,alpha a.grad v)")
def box_int(expr):
    return sp.integrate(sp.integrate(sp.integrate(sp.expand_trig(sp.expand(expr)),
                        (x[0], 0, 2*sp.pi)), (x[1], 0, 2*sp.pi)), (x[2], 0, 2*sp.pi))

# b = alpha a, built DIVERGENCE-FREE:  b_i independent of x_i => div b = 0.  Periodic on [0,2pi]^3.
b_df = [sp.sin(x[1]) + sp.cos(x[2]),
        sp.sin(x[2]) + sp.cos(x[0]),
        sp.sin(x[0]) + sp.cos(x[1])]
assert sp.simplify(div(b_df)) == 0                       # sanity: really divergence-free
# periodic trial/test velocity fields
up = [sp.sin(x[0])*sp.cos(x[1]), sp.sin(x[1])*sp.cos(x[2]), sp.sin(x[2])*sp.cos(x[0])]
vp = [sp.cos(x[0])*sp.sin(x[2]), sp.cos(x[1])*sp.sin(x[0]), sp.cos(x[2])*sp.sin(x[1])]

skew_sum = box_int(dot(vp, adv(b_df, up)) + dot(up, adv(b_df, vp)))
check("eq:skew global: (v,alpha a.grad u) + (u,alpha a.grad v) = 0  on periodic cell, div(alpha a)=0",
      sp.simplify(skew_sum) == 0)

# Discriminating: WITHOUT the div-free hypothesis the symmetric form does NOT vanish.
# The symmetric sum equals -int div(b)(u.v); pick fields whose overlap is guaranteed
# nonzero (u.v = sin^2(x1) has a cos(2 x1) mode that pairs with div(b_bad) = 2 cos(2 x1)).
b_bad  = [sp.sin(2*x[0]), sp.Integer(0), sp.Integer(0)]  # div = 2 cos(2 x1) != 0
u_bad  = [sp.sin(x[0]), sp.Integer(0), sp.Integer(0)]
v_bad  = [sp.sin(x[0]), sp.Integer(0), sp.Integer(0)]
assert sp.simplify(div(b_bad)) != 0
skew_sum_bad = box_int(dot(v_bad, adv(b_bad, u_bad)) + dot(u_bad, adv(b_bad, v_bad)))
check("eq:skew global: dropping div(alpha a)=0 (non-solenoidal b) makes the sum NONZERO (discriminating)",
      sp.simplify(skew_sum_bad) != 0)

# =====================================================================
# (3) eq:globalibp -- pressure/mass IBP on the periodic cell.
#     div(alpha p v) = alpha v.grad p + p div(alpha v)  =>
#     int v.(alpha grad p) = -int p div(alpha v)   (bdry=0, periodic).
#     Analogue:  int q div(alpha u) = -int u.(alpha grad q).
# =====================================================================
print("\n[3] eq:globalibp  (v,alpha grad p) = -(p,div(alpha v))  and  (q,div(alpha u)) = -(u,alpha grad q)")
alp = 2 + sp.sin(x[0]) + sp.Rational(1,3)*sp.cos(x[1]) + sp.Rational(1,4)*sp.sin(x[2])   # alpha(x) > 0, periodic
pp = sp.sin(x[0])*sp.sin(x[1]) + sp.cos(x[2])
qq = sp.cos(x[0]) + sp.sin(x[1])*sp.cos(x[2])

grad_p = [d(pp, i) for i in range(D)]
lhs_gp = box_int(dot(vp, [alp*g for g in grad_p]))                 # (v, alpha grad p)
rhs_gp = box_int(-pp * div([alp*vp[i] for i in range(D)]))         # -(p, div(alpha v))
check("eq:globalibp  (v,alpha grad p) = -(p,div(alpha v))",
      sp.simplify(lhs_gp - rhs_gp) == 0)

grad_q = [d(qq, i) for i in range(D)]
lhs_gm = box_int(qq * div([alp*up[i] for i in range(D)]))          # (q, div(alpha u))
rhs_gm = box_int(-dot(up, [alp*g for g in grad_q]))                # -(u, alpha grad q)
check("eq:globalibp  analogue  (q,div(alpha u)) = -(u,alpha grad q)",
      sp.simplify(lhs_gm - rhs_gm) == 0)

# Discriminating: the PLUS-sign variant must fail.
check("eq:globalibp  PLUS-sign variant (v,alpha grad p) = +(p,div(alpha v)) FAILS (discriminating)",
      sp.simplify(lhs_gp - (-rhs_gp)) != 0)
# Discriminating: dropping alpha inside the divergence must fail (alpha varies).
rhs_noalpha = box_int(-pp * div(vp))                              # -(p, div v), alpha dropped
check("eq:globalibp  dropping alpha inside div (-(p,div v)) FAILS (discriminating)",
      sp.simplify(lhs_gp - rhs_noalpha) != 0)

# =====================================================================
# (4) eq:elemibp -- ELEMENTWISE convective IBP WITH the boundary term retained.
#     On K=[0,1]^3, tau1 constant, div(b)=0 (b=alpha a):
#     (b.grad u, v)_K = -(b.grad v, u)_K + int_{dK} (n.b)(u.v) dGamma.
#     Multiplying by the constant tau1 gives eq:elemibp verbatim.
# =====================================================================
print("\n[4] eq:elemibp  (tau1 alpha a.grad u,v)_K = -(tau1 alpha a.grad v,u)_K + tau1 int_dK alpha(n.a)(u.v)")
tau1 = sp.Symbol('tau1', positive=True)                           # constant on K
# polynomial fields on the unit box; b divergence-free (b_i independent of x_i).
b_el = [x[1]**2 + x[2],  x[0]*x[2] + x[2]**2,  x[0] - x[1]**2]     # div = 0 exactly
assert sp.simplify(div(b_el)) == 0
u_el = [x[0]**2 - x[1]*x[2], x[1]**2 + x[0]*x[2], x[2]**2 - x[0]*x[1]]
v_el = [x[0]*x[1] + x[2],    x[1]*x[2] - x[0]**2, x[0]**2 + x[1]]

def cell_int(expr):
    return sp.integrate(sp.integrate(sp.integrate(sp.expand(expr),
                        (x[0], 0, 1)), (x[1], 0, 1)), (x[2], 0, 1))

lhs_el = cell_int(tau1 * dot(adv(b_el, u_el), v_el))              # (tau1 b.grad u, v)_K
rhs_vol = cell_int(-tau1 * dot(adv(b_el, v_el), u_el))            # -(tau1 b.grad v, u)_K
# boundary term  tau1 * int_{dK} (n.b)(u.v),  assembled over the 6 faces of [0,1]^3
uv = dot(u_el, v_el)
bdry_el = 0
for k in range(D):
    other = [t for t in range(D) if t != k]
    for face_val, sgn in ((0, -1), (1, +1)):                     # x_k = 0 (n=-e_k), x_k = 1 (n=+e_k)
        n_dot_b = (sgn * b_el[k])                                # n.b on the face
        integrand = (n_dot_b * uv).subs(x[k], face_val)
        bdry_el += sp.integrate(sp.integrate(integrand,
                        (x[other[0]], 0, 1)), (x[other[1]], 0, 1))
bdry_el = tau1 * bdry_el
check("eq:elemibp holds with the boundary term retained (tau1 const, div(alpha a)=0 on K)",
      sp.simplify(lhs_el - (rhs_vol + bdry_el)) == 0)

# Discriminating: a FLIPPED boundary-term sign must break the identity.
check("eq:elemibp  FLIPPING the boundary-term sign breaks the identity (discriminating)",
      sp.simplify(lhs_el - (rhs_vol - bdry_el)) != 0)
# Discriminating: dropping the boundary term entirely (naive global skew on K) must fail.
check("eq:elemibp  DROPPING the boundary term (skew on K) fails -- the flux is genuinely nonzero",
      sp.simplify(lhs_el - rhs_vol) != 0)
# Discriminating: with a NON-solenoidal b there is an extra volume term, so the
# boundary-corrected identity no longer closes.
b_el_bad = [x[0]**2, x[1], x[2]]                                  # div = 2 x1 + 2 != 0
assert sp.simplify(div(b_el_bad)) != 0
lhs_bad = cell_int(tau1 * dot(adv(b_el_bad, u_el), v_el))
rhs_bad = cell_int(-tau1 * dot(adv(b_el_bad, v_el), u_el))
uv_b = dot(u_el, v_el)
bdry_bad = 0
for k in range(D):
    other = [t for t in range(D) if t != k]
    for face_val, sgn in ((0, -1), (1, +1)):
        integrand = (sgn * b_el_bad[k] * uv_b).subs(x[k], face_val)
        bdry_bad += sp.integrate(sp.integrate(integrand,
                        (x[other[0]], 0, 1)), (x[other[1]], 0, 1))
bdry_bad = tau1 * bdry_bad
check("eq:elemibp  a NON-solenoidal b leaves a residual volume term (div-free hypothesis has teeth)",
      sp.simplify(lhs_bad - (rhs_bad + bdry_bad)) != 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
