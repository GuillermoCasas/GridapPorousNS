#!/usr/bin/env python3
# =============================================================================
# coverage_coercivity_numeric_verification.py
#
# Symbolic (sympy) verification of the GALERKIN COERCIVITY identity and the
# NUMERICAL-SETUP identities of
#   "A stabilized finite element method for incompressible, inertial flows in
#    inhomogeneous porous media" (Casas, Gonzalez-Usua, Codina, de-Pouplana).
#
# WHY THIS SCRIPT EXISTS (coverage-gap closure, cluster g5-coercivity-numeric)
# ---------------------------------------------------------------------------
# Three algebraically-correct displays had no permanent encoded check.  They are
# all "numeric setup"/coercivity identities that a regression could silently
# break:
#
#  (A) eq:StabilityEstimate  (unnumbered display @ l.850): the Galerkin
#      coercivity identity
#        B(a;U_h,U_h) = 2 nu ||alpha^{1/2} Pi grad u_h||^2
#                       + ||sigma^{1/2} u_h||^2 + eps ||p_h||^2 ,
#      obtained by taking V_h=U_h in the bilinear form B (l.845-847) under the
#      hypotheses div(alpha a)=0 and u_h=0 on the boundary.  The convective term
#      and the pressure-gradient/porous-divergence cross terms cancel by
#      integration by parts; the viscous term collapses because Pi is a
#      symmetric idempotent projector.  README flags this as "verifiable, not
#      yet encoded".  We RECONSTRUCT B(U_h,U_h) on a unit box (the box-IBP
#      technique already used in cdr_operator_verification.py) and check it
#      equals the sum of squares -- with DISCRIMINATING variants proving each
#      hypothesis (div(alpha a)=0, the alpha inside div(alpha u), the projector,
#      the factor 2) is load-bearing.
#
#  (B) Centered dimensional encoding (unnumbered display @ l.1132-1138):
#      from Re = U L/nu, Da = sigma L^2/(alpha_inf nu), L=1 and the centering
#      constraint nu*sigma=1, the paper solves
#        nu = 1/sqrt(alpha_inf Da),  sigma = sqrt(alpha_inf Da),
#        U  = Re/sqrt(alpha_inf Da).
#      We re-derive these from the three constraints, verify they reproduce
#      (Re,Da) and the centering, and check the naive-vs-centered magnitude
#      claims at the worst cell.
#
#  (C) Footnote @ l.1430: the elementwise viscous coercivity threshold
#      c1*(K) = 2 chat^2(K); its Kuhn-vs-right-triangle ratio
#      (100 + 5 sqrt2)/24 ~ 4.5; and c1 = 16 k^4 (3D) being four times
#      4 k^4 (2D).  The chosen integer multiplier 4 = floor(4.5) is precisely
#      why "this round value ... sits just below the ... Kuhn threshold" while
#      4 k^4 is sub-coercive.
#
# SKIPPED (see equations_skipped in the task report):
#   - DBF inline @ l.1551 (C_a=0.30, C_b=1.75, Da(alpha_0)): the DBF/porosity
#     coefficients are owned by manufactured_solution_verification.py (literature
#     DBF a(alpha),b(alpha)); not duplicated here (out of this cluster's scope).
#   - Cocquet inline @ l.1645: prose ("viscosity-dominated flows"), carries no
#     standalone algebraic identity to encode.
#
# Run:  python3 coverage_coercivity_numeric_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

print("=" * 70)
print("Galerkin coercivity + numerical-setup identities -- symbolic checks")
print("=" * 70)

# =====================================================================
# (A) Galerkin coercivity identity  eq:StabilityEstimate (l.850)
#
# Bilinear form (l.845-847), d=2, V_h=U_h:
#   B(a;U,U) = (u, alpha a.grad u)          [convection]
#            + 2 (grad u, alpha nu Pi grad u)[viscous]
#            + (u, alpha grad p)             [pressure gradient]
#            + (u, sigma u)                  [reaction]
#            + (p, eps p)                    [compressibility]
#            + (p, div(alpha u))             [porous divergence]
# Claim (l.851):  = 2 nu ||alpha^{1/2} Pi grad u||^2 + ||sigma^{1/2} u||^2
#                   + eps ||p||^2 .
# Reconstructed on the unit box [0,1]^2 with:
#   * u vanishing on the whole boundary (bubble * poly),
#   * m := alpha a a DIVERGENCE-FREE field (stream function) so div(alpha a)=0,
#   * nu, sigma, eps uniform constants (as in Sec. 5), alpha(x) varying.
# =====================================================================
print("\n[A] Galerkin coercivity identity  B(a;U,U) = sum of squares (l.850)")
y0, y1 = sp.symbols('y0 y1', real=True)
Y = (y0, y1)
nu, sig, eps = sp.symbols('nu sigma varepsilon', positive=True)   # uniform constants
def dy(f, i): return sp.diff(f, Y[i])

# varying porosity, strictly positive on the box
al = 2 + y0/3 + y1**2/4

# velocity vanishing on the whole boundary of [0,1]^2  (bubble * poly)
bub = y0*(1 - y0)*y1*(1 - y1)
u = [bub*(1 + y0 + 2*y1), bub*(2 - y0 + y1)]

# pressure: arbitrary polynomial (no BC needed -- its boundary term is killed by u=0)
p = 1 + y0 - y1 + y0*y1 + y0**2/2

# m = alpha a : divergence-free via a stream function psi  ->  m=(psi_y1, -psi_y0)
psi = y0**2*y1**2 + y0*y1
m = [dy(psi, 1), -dy(psi, 0)]
check("A-hyp  m=alpha a is divergence-free (div(alpha a)=0)",
      sp.simplify(dy(m[0], 0) + dy(m[1], 1)) == 0)
check("A-hyp  velocity vanishes on the whole boundary of [0,1]^2",
      all(sp.simplify(u[i].subs(y0, 0)) == 0 and sp.simplify(u[i].subs(y0, 1)) == 0
          and sp.simplify(u[i].subs(y1, 0)) == 0 and sp.simplify(u[i].subs(y1, 1)) == 0
          for i in range(2)))

# deviatoric symmetric gradient projector  (Pi grad u)_{a i},  d=2
divu = dy(u[0], 0) + dy(u[1], 1)
def Pi(a, i):
    return sp.Rational(1, 2)*(dy(u[i], a) + dy(u[a], i)) - sp.Rational(1, 2)*divu*(1 if a == i else 0)

def box_int(f):
    return sp.integrate(sp.integrate(sp.expand(f), (y0, 0, 1)), (y1, 0, 1))

# --- individual bilinear-form integrands (v=u, q=p) ---
conv_ig  = sum(u[i]*sum(m[j]*dy(u[i], j) for j in range(2)) for i in range(2))
visc_ig  = 2*nu*al*sum(dy(u[i], a)*Pi(a, i) for a in range(2) for i in range(2))
pgrad_ig = sum(u[i]*al*dy(p, i) for i in range(2))
reac_ig  = sig*sum(u[i]**2 for i in range(2))
eps_ig   = eps*p**2
div_ig   = p*sum(dy(al*u[i], i) for i in range(2))            # p * div(alpha u)

B_UU = box_int(conv_ig + visc_ig + pgrad_ig + reac_ig + eps_ig + div_ig)

# --- claimed RHS: 2 nu ||alpha^{1/2} Pi grad u||^2 + ||sigma^{1/2} u||^2 + eps||p||^2
rhs_visc_ig = 2*nu*al*sum(Pi(a, i)**2 for a in range(2) for i in range(2))
RHS = box_int(rhs_visc_ig + reac_ig + eps_ig)

check("eq:StabilityEstimate  B(a;U,U) == 2nu||a^{1/2}Pi grad u||^2 + ||s^{1/2}u||^2 + eps||p||^2",
      sp.simplify(B_UU - RHS) == 0)

# --- why each hypothesis matters (DISCRIMINATING) ---
# (i) convective term vanishes ONLY because div(alpha a)=0 (with u=0 on bdry)
check("A-cancel  convective term (u, alpha a.grad u) integrates to 0",
      sp.simplify(box_int(conv_ig)) == 0)
m_bad = [y0, sp.Integer(0)]                                    # div = 1 != 0
conv_bad = box_int(sum(u[i]*sum(m_bad[j]*dy(u[i], j) for j in range(2)) for i in range(2)))
check("A-cancel  WRONG: a non-divergence-free alpha a leaves a nonzero convective term",
      sp.simplify(conv_bad) != 0)

# (ii) pressure-gradient + porous-divergence cross terms cancel;
#      the alpha INSIDE div(alpha u) is required for the cancellation
check("A-cancel  (u,alpha grad p) + (p, div(alpha u)) == 0  (IBP, u=0 on bdry)",
      sp.simplify(box_int(pgrad_ig + div_ig)) == 0)
div_noalpha = p*sum(dy(u[i], i) for i in range(2))             # p * div(u): alpha dropped
check("A-cancel  WRONG: dropping alpha from div(alpha u) breaks the cross-term cancellation",
      sp.simplify(box_int(pgrad_ig + div_noalpha)) != 0)

# (iii) viscous collapse needs Pi symmetric+idempotent: <grad u, Pi grad u> = |Pi grad u|^2
g = sp.Matrix(2, 2, lambda a, b: sp.Symbol(f'G{a}{b}', real=True))     # generic gradient tensor
def PiT(G):
    sym = (G + G.T)/2
    return sym - sp.Rational(1, 2)*sp.trace(G)*sp.eye(2)
check("A-visc  Pi is idempotent (Pi(Pi G) = Pi G) on a generic tensor",
      sp.simplify(PiT(PiT(g)) - PiT(g)) == sp.zeros(2, 2))
frob = lambda A, B: sum(A[i, j]*B[i, j] for i in range(2) for j in range(2))
check("A-visc  <G, Pi G> = <Pi G, Pi G> = |Pi G|^2  (symmetric orthogonal projector)",
      sp.simplify(frob(g, PiT(g)) - frob(PiT(g), PiT(g))) == 0)
check("A-visc  WRONG: |grad u|^2 (no projector) differs from |Pi grad u|^2 in general",
      sp.simplify(frob(g, g) - frob(PiT(g), PiT(g))) != 0)

# (iv) the coefficient 2 on the viscous term is load-bearing
RHS_nofactor2 = box_int(nu*al*sum(Pi(a, i)**2 for a in range(2) for i in range(2)) + reac_ig + eps_ig)
check("A-coef   WRONG: dropping the factor 2 (nu instead of 2nu) breaks the identity",
      sp.simplify(B_UU - RHS_nofactor2) != 0)

# =====================================================================
# (B) Centered dimensional encoding (l.1132-1138)
#   Re = U L/nu ,  Da = sigma L^2/(alpha_inf nu) ,  L=1 ,  centering nu*sigma=1.
#   Paper solution:  nu=1/sqrt(a_inf Da), sigma=sqrt(a_inf Da), U=Re/sqrt(a_inf Da).
# =====================================================================
print("\n[B] Centered dimensional encoding (l.1132)")
Re, Da, ainf = sp.symbols('Re Da alpha_inf', positive=True)
L = sp.Integer(1)

# paper closed forms (l.1134-1136)
nu_c  = 1/sp.sqrt(ainf*Da)
sig_c = sp.sqrt(ainf*Da)
U_c   = Re/sp.sqrt(ainf*Da)

# re-derive them from the three constraints (independent of the printed forms)
Us, nus, sigs = sp.symbols('Us nus sigs', positive=True)
sol = sp.solve([sp.Eq(Re, Us*L/nus),
                sp.Eq(Da, sigs*L**2/(ainf*nus)),
                sp.Eq(nus*sigs, 1)], [Us, nus, sigs], dict=True)
check("B-derive  the 3 constraints have a unique positive solution",
      len(sol) == 1)
s = sol[0]
check("B-derive  solved  nu = 1/sqrt(alpha_inf Da)   (matches paper)",
      sp.simplify(s[nus] - nu_c) == 0)
check("B-derive  solved  sigma = sqrt(alpha_inf Da)  (matches paper)",
      sp.simplify(s[sigs] - sig_c) == 0)
check("B-derive  solved  U = Re/sqrt(alpha_inf Da)   (matches paper)",
      sp.simplify(s[Us] - U_c) == 0)

# closed forms reproduce (Re, Da) and satisfy the centering
check("B-reprod  U L/nu  reproduces Re",
      sp.simplify(U_c*L/nu_c - Re) == 0)
check("B-reprod  sigma L^2/(alpha_inf nu)  reproduces Da",
      sp.simplify(sig_c*L**2/(ainf*nu_c) - Da) == 0)
check("B-center  nu*sigma = 1  (coefficients centered about unity)",
      sp.simplify(nu_c*sig_c - 1) == 0)
check("B-spread  sigma/nu = alpha_inf Da  (intrinsic, non-removable spread)",
      sp.simplify(sig_c/nu_c - ainf*Da) == 0)

# DISCRIMINATING: the (task-stated) variant nu=sqrt(Da a_inf)/Re does NOT reproduce Re
nu_wrong = sp.sqrt(Da*ainf)/Re
check("B-wrong   nu = sqrt(Da alpha_inf)/Re does NOT reproduce Re (gives Re^2/(a_inf Da))",
      sp.simplify(U_c*L/nu_wrong - Re) != 0
      and sp.simplify(U_c*L/nu_wrong - Re**2/(ainf*Da)) == 0)

# numeric magnitude claims at the worst reaction cell (Re,Da,a_inf)=(1e-6,1e6,1)
subs = {Re: sp.Rational(1, 10**6), Da: sp.Integer(10**6), ainf: sp.Integer(1)}
nu_naive  = sp.Integer(1)/Re                     # naive U=1 -> nu = U/Re = 1/Re
sig_naive = Da*ainf*nu_naive                     # sigma = Da*a_inf*nu
check("B-naive   naive U=1 drives nu~1e6, sigma~1e12 at (Re,Da)=(1e-6,1e6)",
      nu_naive.subs(subs) == 10**6 and sig_naive.subs(subs) == 10**12)
check("B-cent-num centered keeps sigma<=1e3 (and nu=1e-3) at the same cell",
      sp.nsimplify(sig_c.subs(subs)) == 1000 and sp.nsimplify(nu_c.subs(subs)) == sp.Rational(1, 1000))
check("B-cent-num centered geometric mean sqrt(nu*sigma)=1 at the same cell",
      sp.simplify(sp.sqrt(nu_c*sig_c).subs(subs) - 1) == 0)

# =====================================================================
# (C) Coercivity-threshold footnote (l.1430)
#   c1*(K) = 2 chat^2(K);  Kuhn/triangle ratio = (100 + 5 sqrt2)/24 ~ 4.5;
#   c1 = 16 k^4 (3D) = 4 * 4 k^4 (2D).  The integer multiplier 4 = floor(4.5)
#   is why 16 k^4 "sits just below" the Kuhn threshold while 4 k^4 is sub-coercive.
# =====================================================================
print("\n[C] Coercivity-threshold footnote (l.1430)")
ratio = (100 + 5*sp.sqrt(2))/24
check("C-ratio  (100 + 5 sqrt2)/24 == 100/24 + 5 sqrt2/24  (exact form)",
      sp.simplify(ratio - (sp.Rational(100, 24) + 5*sp.sqrt(2)/24)) == 0)
rv = float(ratio.evalf())
check("C-ratio  numeric value ~ 4.5  (|value - 4.5| < 0.05)",
      abs(rv - 4.5) < 0.05)
# DISCRIMINATING: the 5*sqrt2 term is needed -- dropping it (100/24~4.17) misses 4.5,
# and flipping its sign ((100-5sqrt2)/24~3.87) would push the ratio below 4,
# which would make the chosen 4x multiplier EXCEED the threshold (breaking "just below").
check("C-ratio  WRONG: 100/24 (no sqrt2 term) ~ 4.17 is not within 0.05 of 4.5",
      abs(float((sp.Rational(100, 24)).evalf()) - 4.5) >= 0.05)
check("C-ratio  sign of sqrt2 term matters: (100-5sqrt2)/24 < 4 (would break 'just below')",
      float(((100 - 5*sp.sqrt(2))/24).evalf()) < 4)

k = sp.Symbol('k', positive=True)
c1_2d = 4*k**4
c1_3d = 16*k**4
check("C-4x     c1(3D)=16k^4 == 4 * (4k^4)=c1(2D)  for all k  (four times the 2D value)",
      sp.simplify(c1_3d - 4*c1_2d) == 0 and sp.simplify(c1_3d/c1_2d - 4) == 0)
check("C-4x     k=2:  16k^4 = 256,  4k^4 = 64,  ratio = 4",
      c1_3d.subs(k, 2) == 256 and c1_2d.subs(k, 2) == 64
      and sp.Rational(c1_3d.subs(k, 2), c1_2d.subs(k, 2)) == 4)

# threshold form c1*(K)=2 chat^2(K): the factor 2 cancels in the element ratio,
# which is WHY the paper can state the Kuhn/triangle ratio purely geometrically.
ct, cK = sp.symbols('chat_tri chat_Kuhn', positive=True)
c1star_tri  = 2*ct**2
c1star_Kuhn = 2*cK**2
# impose the paper's geometric ratio on chat^2, then the c1* ratio must coincide with it
gate = sp.Eq(cK**2/ct**2, ratio)
c1star_ratio = sp.simplify((c1star_Kuhn/c1star_tri).subs(sp.solve(gate, cK**2, dict=True)[0]))
check("C-thresh c1*(K)=2chat^2(K): factor 2 cancels, c1*(Kuhn)/c1*(tri) == (100+5sqrt2)/24",
      sp.simplify(c1star_ratio - ratio) == 0)

# "round value ... sits just below the Kuhn threshold": the chosen integer
# multiplier equals floor(ratio)=4, i.e. 4 < ratio < 5.  Taking the 2D marginal
# 4k^4 as the right-triangle reference threshold, the Kuhn threshold is
# ratio*4k^4; then 16k^4 = 4*4k^4 < ratio*4k^4 (just below) while 4k^4 sits a
# factor 'ratio' below it (deeply sub-coercive).
check("C-floor  multiplier 4 = floor((100+5sqrt2)/24):  4 < ratio < 5",
      (4 < rv) and (rv < 5) and int(sp.floor(ratio)) == 4)
kuhn_thr = ratio*c1_2d                                  # ~4.46 * 4k^4
check("C-below  16k^4 sits just below the Kuhn threshold (16k^4 < ratio*4k^4)",
      sp.simplify(c1_3d - kuhn_thr) != 0
      and float((c1_3d/kuhn_thr).subs(k, 2).evalf()) < 1
      and float((c1_3d/kuhn_thr).subs(k, 2).evalf()) > 0.85)      # close to 1 => "just below"
check("C-below  4k^4 is sub-coercive: a factor 'ratio' below the Kuhn threshold",
      float((c1_2d/kuhn_thr).subs(k, 2).evalf()) < 0.3
      and sp.simplify(kuhn_thr/c1_2d - ratio) == 0)
# DISCRIMINATING: a hypothetical 5x multiplier (20k^4) would EXCEED the threshold,
# confirming 4 is the largest admissible round value strictly below it.
check("C-below  WRONG: a 5x multiplier (20k^4) would exceed the Kuhn threshold",
      float((5*c1_2d/kuhn_thr).subs(k, 2).evalf()) > 1)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
