#!/usr/bin/env python3
# =============================================================================
# cdr_operator_verification.py
#
# Symbolic (sympy) verification of the convection-diffusion-reaction (CDR)
# matrix form of the porous Navier-Stokes operator, Section 2 of
#   "A stabilized finite element method for incompressible, inertial flows in
#    inhomogeneous porous media" (Casas, Gonzalez-Usua, Codina, de-Pouplana).
#
# Verifies, for d = 3 and with a spatially varying porosity alpha(x):
#   (1) the diffusion matrix K_ij (eq:matrices_stationary_strong_problem)
#       reproduces the deviatoric-symmetric viscous operator
#       -2 div(alpha nu Pi grad u)  of eq:StrongMomentumEquation;
#   (2) the full operator  L_w U = -d_i(K_ij d_j U) + A_{c,i} d_i U
#       + A_{f,i} d_i U + S U  (eq:CDR_equation) reproduces the strong
#       momentum + mass system, including the split div(alpha u)=alpha div u
#       + u.grad alpha and the epsilon*p compressibility term;
#   (3) the adjoint operator L^* (eq:AdjointDifferentialOperator) is the formal
#       adjoint of L_w: int (V.L_w U - (L^*V).U) = 0 on a periodic cell for
#       constant coefficients (the matrix transposes and first-order sign
#       flips are correct).
#
# Run:  python3 cdr_operator_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

x = sp.symbols('x1 x2 x3', real=True)
nu, eps = sp.symbols('nu varepsilon', positive=True)
def d(f, i): return sp.diff(f, x[i])
def kron(p, q): return 1 if p == q else 0
D = 3

print("=" * 70)
print("CDR matrix form of the porous Navier-Stokes operator -- symbolic checks")
print("=" * 70)

# Spatially varying fields
alpha = sp.Function('alpha', positive=True)(*x)
u = [sp.Function(f'u{i+1}')(*x) for i in range(D)]
p = sp.Function('p')(*x)
w = [sp.Function(f'w{i+1}')(*x) for i in range(D)]                 # advection field
sig = sp.Matrix(D, D, lambda a, b: sp.Function(f'sig{min(a,b)+1}{max(a,b)+1}')(*x))  # symmetric

# -------------------------------------------------------------------------
# (1) Viscous block:  -d_i( K_ij^{(a,b)} d_j u_b )  ==  -2 div(alpha nu Pi grad u)
# -------------------------------------------------------------------------
print("\n[1] Diffusion matrix K_ij vs. -2 div(alpha nu Pi grad u)")
divu = sum(d(u[k], k) for k in range(D))
def Pi(a, i):                       # deviatoric symmetric gradient (Pi grad u)_{a i}
    return sp.Rational(1, 2)*(d(u[i], a) + d(u[a], i)) - sp.Rational(1, D)*divu*kron(a, i)
strong_visc = [-2*sum(d(alpha*nu*Pi(a, i), i) for i in range(D)) for a in range(D)]

def Kblk(i, j, a, b):               # velocity (a,b in 0..2) block of K_ij
    if a == b:
        return nu*alpha*(kron(i, j) + sp.Rational(1, D)*kron(a, i)*kron(a, j))
    return nu*alpha*(kron(b, i)*kron(a, j) - sp.Rational(2, D)*kron(a, i)*kron(b, j))
cdr_visc = [-sum(d(sum(Kblk(i, j, a, b)*d(u[b], j) for j in range(D) for b in range(D)), i)
                 for i in range(D)) for a in range(D)]
check("K_ij reproduces -2 div(alpha nu Pi grad u) for all 3 components (varying alpha)",
      all(sp.simplify(strong_visc[a] - cdr_visc[a]) == 0 for a in range(D)))

# -------------------------------------------------------------------------
# (2) Full operator L_w U == strong momentum + mass system
# -------------------------------------------------------------------------
print("\n[2] Full operator L_w U vs. the strong system")
U = u + [p]                                     # 4-vector [u1,u2,u3,p]
n = D + 1

def Ac(i):                                      # convection + divergence row
    M = sp.zeros(n, n)
    for a in range(D):
        M[a, a] = alpha*w[i]                     # alpha w_i on the velocity diagonal
    for b in range(D):
        M[D, b] = alpha*kron(i, b)               # alpha delta_{ib} : divergence in the mass row
    return M
def Af(i):                                      # pressure gradient
    M = sp.zeros(n, n)
    for a in range(D):
        M[a, D] = alpha*kron(i, a)               # alpha delta_{ia} in the pressure column
    return M
def Smat():                                     # reaction + grad-alpha + eps
    M = sp.zeros(n, n)
    for a in range(D):
        for b in range(D):
            M[a, b] = sig[a, b]                  # sigma on the velocity block
    for b in range(D):
        M[D, b] = d(alpha, b)                    # grad alpha . u  in the mass row
    M[D, D] = eps                                # eps p
    return M
def Kfull(i, j):                                # 4x4 diffusion (pressure row/col are zero)
    M = sp.zeros(n, n)
    for a in range(D):
        for b in range(D):
            M[a, b] = Kblk(i, j, a, b)
    return M

# L_w U, component by component
LU = sp.zeros(n, 1)
for a in range(n):
    visc = -sum(d(sum(Kfull(i, j)[a, b]*d(U[b], j) for j in range(D) for b in range(n)), i) for i in range(D))
    conv = sum(sum(Ac(i)[a, b]*d(U[b], i) for b in range(n)) for i in range(D))
    flux = sum(sum(Af(i)[a, b]*d(U[b], i) for b in range(n)) for i in range(D))
    reac = sum(Smat()[a, b]*U[b] for b in range(n))
    LU[a] = sp.expand(visc + conv + flux + reac)

strong_mom = [sp.expand(strong_visc[a] + alpha*sum(w[i]*d(u[a], i) for i in range(D))
                        + alpha*d(p, a) + sum(sig[a, b]*u[b] for b in range(D))) for a in range(D)]
strong_mass = sp.expand(eps*p + sum(d(alpha*u[k], k) for k in range(D)))   # eps p + div(alpha u)

check("momentum rows of L_w U match -2div(..)+alpha w.grad u+alpha grad p+sigma u",
      all(sp.simplify(LU[a] - strong_mom[a]) == 0 for a in range(D)))
check("mass row of L_w U equals eps p + div(alpha u) = eps p + alpha div u + u.grad alpha",
      sp.simplify(LU[D] - strong_mass) == 0)

# -------------------------------------------------------------------------
# (3) Adjoint L^* (eq:AdjointDifferentialOperator), constant-coefficient
#     periodic-cell test:  int_cell ( V . L_w U - (L^*V) . U ) = 0.
# -------------------------------------------------------------------------
print("\n[3] Adjoint operator L^* = formal adjoint of L_w (periodic-cell integral)")
# constant coefficients
a0, nu0, e0 = sp.symbols('a0 nu0 e0', positive=True)
w0 = sp.symbols('w01 w02 w03', real=True)
S0 = sp.Matrix(D, D, lambda a, b: sp.Symbol(f'S0_{min(a,b)}{max(a,b)}', real=True))  # symmetric const
# periodic trial/test fields on [0,2pi]^3 (integer frequencies)
def trig(seed):
    return (sp.sin((seed % 3 + 1)*x[0]) * sp.cos((seed % 2 + 1)*x[1]) * sp.sin((seed % 4 + 1)*x[2] + seed))
uu = [trig(s) for s in (1, 2, 3)]; pp = trig(7)
vv = [trig(s) for s in (4, 5, 6)]; qq = trig(8)
UU = uu + [pp]; VV = vv + [qq]

def Kc(i, j, a, b):
    if a >= D or b >= D: return 0
    if a == b: return nu0*a0*(kron(i, j) + sp.Rational(1, D)*kron(a, i)*kron(a, j))
    return nu0*a0*(kron(b, i)*kron(a, j) - sp.Rational(2, D)*kron(a, i)*kron(b, j))
def Acc(i, a, b):
    if a < D and b < D: return a0*w0[i]*kron(a, b)
    if a == D and b < D: return a0*kron(i, b)
    return 0
def Afc(i, a, b):
    if a < D and b == D: return a0*kron(i, a)
    return 0
def Sc(a, b):
    if a < D and b < D: return S0[a, b]
    if a == D and b == D: return e0
    return 0   # constant alpha => grad alpha = 0

def Lw(field):                                   # L_w applied to a 4-field
    out = []
    for a in range(n):
        visc = -sum(d(sum(Kc(i, j, a, b)*d(field[b], j) for j in range(D) for b in range(n)), i) for i in range(D))
        conv = sum(sum(Acc(i, a, b)*d(field[b], i) for b in range(n)) for i in range(D))
        flux = sum(sum(Afc(i, a, b)*d(field[b], i) for b in range(n)) for i in range(D))
        reac = sum(Sc(a, b)*field[b] for b in range(n))
        out.append(visc + conv + flux + reac)
    return out
def Ladj(field):                                 # L^* : -d_i(K_ji^T d_j V) - d_i(A_c,i^T V) - d_i(A_f,i^T V) + S^T V
    out = []
    for a in range(n):
        visc = -sum(d(sum(Kc(i, j, b, a)*d(field[b], j) for j in range(D) for b in range(n)), i) for i in range(D))
        conv = -sum(d(sum(Acc(i, b, a)*field[b] for b in range(n)), i) for i in range(D))
        flux = -sum(d(sum(Afc(i, b, a)*field[b] for b in range(n)), i) for i in range(D))
        reac = sum(Sc(b, a)*field[b] for b in range(n))
        out.append(visc + conv + flux + reac)
    return out

LU3 = Lw(UU); LSV = Ladj(VV)
integrand = sp.expand(sum(VV[a]*LU3[a] for a in range(n)) - sum(LSV[a]*UU[a] for a in range(n)))
box = sp.integrate(sp.integrate(sp.integrate(integrand, (x[0], 0, 2*sp.pi)),
                                (x[1], 0, 2*sp.pi)), (x[2], 0, 2*sp.pi))
check("int_cell ( V.L_w U - (L^*V).U ) = 0  (L^* is the formal adjoint)",
      sp.simplify(box) == 0)

# -------------------------------------------------------------------------
# (4) Natural Neumann co-normal: n_i K_ij d_j u = 2 alpha nu (Pi grad u).n
#     (the factor-2 viscous traction of eq:neumann_bc, amendment A3).
# -------------------------------------------------------------------------
print("\n[4] Natural Neumann co-normal n_i K_ij d_j u = 2 a nu (Pi grad u).n   (A3 factor 2)")
nrm = sp.symbols('n0 n1 n2', real=True)
visc_conormal = [sum(nrm[i]*sum(Kblk(i, j, dd, b)*d(u[b], j) for j in range(D) for b in range(D))
                     for i in range(D)) for dd in range(D)]
traction = [2*alpha*nu*sum(Pi(dd, m)*nrm[m] for m in range(D)) for dd in range(D)]
check("n_i K_ij d_j u == 2 alpha nu (Pi grad u).n   (all components, varying alpha)",
      all(sp.simplify(visc_conormal[dd] - traction[dd]) == 0 for dd in range(D)))

# -------------------------------------------------------------------------
# (5) Green's identity on the unit box, with VARYING alpha(x), a(x):
#       int_Omega ( V.L_w U - (L^*V).U ) = int_dOmega ( U.D*_N V - V.D_N U ),
#     with  D_N U  = n_i ( K_ij d_j U - A_f,i U ) = alpha(2 nu Pi grad u - p I).n  (eq:neumann_bc, A3)
#      and  D*_N V = n_i ( K_ji^T d_j V + A_c,i^T V )                               (eq:AdjointFlux).
# -------------------------------------------------------------------------
print("\n[5] Green's identity on the box -> verifies D*_N (eq:AdjointFlux) and D_N (A3)")
y = sp.symbols('y0 y1 y2', real=True)
def dy(f, i): return sp.diff(f, y[i])
al = 2 + y[0]/3 + y[1]**2/4 + y[2]/5                                    # alpha(x) > 0 on the box
av = [sp.Rational(1, 2) + y[0]/4 - y[1]/6, sp.Rational(1, 3) + y[2]/5, sp.Rational(1, 5) - y[0]/7]
Up = [y[0]**2 - y[1]*y[2], y[1]**2 + y[0]/2, y[2]**2 - y[0]*y[1], y[0]*y[1] + y[2]**2/2]    # U=[u;p]
Vp = [y[1]*y[2] + y[0], y[0]**2 - y[2], y[1]**2 + y[0]*y[2], y[0] - y[1]*y[2]]              # V=[v;q]
sg = sp.Matrix(D, D, lambda p, q: sp.Integer(1 + (p == q)))                                # const symmetric sigma
nn = D + 1
def Km(i, j):
    M = sp.zeros(nn, nn)
    for dd in range(D):
        for b in range(D):
            M[dd, b] = (nu*al*(int(i == j) + sp.Rational(1, D)*int(dd == i)*int(dd == j)) if dd == b
                        else nu*al*(int(b == i)*int(dd == j) - sp.Rational(2, D)*int(dd == i)*int(b == j)))
    return M
def Acm(i):
    M = sp.zeros(nn, nn)
    for dd in range(D): M[dd, dd] = al*av[i]
    for b in range(D): M[D, b] = al*int(i == b)
    return M
def Afm(i):
    M = sp.zeros(nn, nn)
    for dd in range(D): M[dd, D] = al*int(i == dd)
    return M
def Sm():
    M = sp.zeros(nn, nn)
    for dd in range(D):
        for b in range(D): M[dd, b] = sg[dd, b]
    for b in range(D): M[D, b] = dy(al, b)
    M[D, D] = eps
    return M
def LU_box():
    out = sp.zeros(nn, 1)
    for e in range(nn):
        visc = -sum(dy(sum(Km(i, j)[e, b]*dy(Up[b], j) for j in range(D) for b in range(nn)), i) for i in range(D))
        conv = sum(sum(Acm(i)[e, b]*dy(Up[b], i) for b in range(nn)) for i in range(D))
        flux = sum(sum(Afm(i)[e, b]*dy(Up[b], i) for b in range(nn)) for i in range(D))
        out[e] = visc + conv + flux + sum(Sm()[e, b]*Up[b] for b in range(nn))
    return out
def LsV_box():
    out = sp.zeros(nn, 1)
    for e in range(nn):
        visc = -sum(dy(sum(Km(j, i)[b, e]*dy(Vp[b], j) for j in range(D) for b in range(nn)), i) for i in range(D))
        conv = -sum(dy(sum(Acm(i)[b, e]*Vp[b] for b in range(nn)), i) for i in range(D))
        flux = -sum(dy(sum(Afm(i)[b, e]*Vp[b] for b in range(nn)), i) for i in range(D))
        out[e] = visc + conv + flux + sum(Sm()[b, e]*Vp[b] for b in range(nn))
    return out
LUb, LsVb = LU_box(), LsV_box()
vol_integrand = sp.expand(sum(Vp[e]*LUb[e] - LsVb[e]*Up[e] for e in range(nn)))
vol = sp.integrate(sp.integrate(sp.integrate(vol_integrand, (y[0], 0, 1)), (y[1], 0, 1)), (y[2], 0, 1))
def DstarN(nv):    # n_i ( K_ji^T d_j V + A_c,i^T V )
    return [sum(nv[i]*(sum(Km(j, i)[b, e]*dy(Vp[b], j) for j in range(D) for b in range(nn))
                       + sum(Acm(i)[b, e]*Vp[b] for b in range(nn))) for i in range(D)) for e in range(nn)]
def DN(nv):        # n_i ( K_ij d_j U - A_f,i U )
    return [sum(nv[i]*(sum(Km(i, j)[dd, b]*dy(Up[b], j) for j in range(D) for b in range(nn))
                       - sum(Afm(i)[dd, b]*Up[b] for b in range(nn))) for i in range(D)) for dd in range(nn)]
bdry = 0
for k in range(D):
    other = [t for t in range(D) if t != k]
    for s, sgn in ((0, -1), (1, +1)):
        nv = [sgn if t == k else 0 for t in range(D)]
        dsn, dn = DstarN(nv), DN(nv)
        face = sp.expand(sum(Up[e]*dsn[e] - Vp[e]*dn[e] for e in range(nn))).subs(y[k], s)
        bdry += sp.integrate(sp.integrate(face, (y[other[0]], 0, 1)), (y[other[1]], 0, 1))
check("int_box (V.L_w U - (L*V).U) == int_bdry (U.D*_N V - V.D_N U)   (D*_N eq:AdjointFlux, D_N=A3 flux)",
      sp.simplify(vol - bdry) == 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
