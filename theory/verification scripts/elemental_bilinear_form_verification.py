#!/usr/bin/env python3
# =============================================================================
# elemental_bilinear_form_verification.py
#
# Verifies the *assembled* stabilization elemental matrices of the Appendix
# against the stabilization bilinear form eq:StabilizationLVLU itself,
#   -sum_K <L*V, tau L U> = ( tau1 alpha^2 L*_mom(V), L_mom(U) )
#                         + ( tau2 alpha^2 L*_mass(V), L_mass(U) ),
# with
#   L_mom(u)  = a.grad u  - nu Lapbar u - nu (Pi grad u).grad beta + grad p + sigma u/alpha
#   L*_mom(v) = a.grad v  + nu Lapbar v + nu (Pi grad v).grad beta + grad q - sigma v/alpha
#   L_mass(u) = div u + u.grad beta + eps p/alpha ,   L*_mass(v)= div v + v.grad beta - eps q/alpha
#   Lapbar w  = 2 div(sym grad w) - 2/3 grad(div w) ,  (Pi grad w)_{dm}=sym(d_d w_m)-1/3 div(w) delta_dm ,
#   beta = log alpha .
# The appendix groups the 40+8+8+3 contributions into "families" X_Y, where X is
# the piece of L*_mom(V) and Y the piece of L_mom(U); each family's printed
# results must therefore sum to  tau1 alpha^2 (piece X of L*_mom(V)) . L_mom(U)
# (and similarly for the mass / pressure blocks).  We transcribe the printed
# results, sum each family, and check it against that bilinear-form piece.
#
# Run:  python3 elemental_bilinear_form_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok
def kron(p, q): return 1 if p == q else 0

x = sp.symbols('x0 x1 x2', real=True)
def d(f, i): return sp.diff(f, x[i])
I3 = range(3)

alpha = sp.Function('alpha', positive=True)(*x)
beta = sp.log(alpha)
nu, eps, tau1, tau2 = sp.symbols('nu varepsilon tau1 tau2', positive=True)
a = sp.symbols('a0 a1 a2', real=True)
sig = sp.Matrix(3, 3, lambda p, q: sp.Symbol(f'sig{min(p,q)}{max(p,q)}', real=True))
Na = sp.Function('Na')(*x); Nb = sp.Function('Nb')(*x)
u = [sp.Function(f'u{k}')(*x) for k in I3]
pp = sp.Function('pp')(*x)

# ---- operator pieces (act on a generic 3-vector field "w" and scalar "r") ----
def divw(w): return sum(d(w[k], k) for k in I3)
def symg(w, dd, m): return sp.Rational(1, 2)*(d(w[dd], m) + d(w[m], dd))
def Pi(w, dd, m): return symg(w, dd, m) - sp.Rational(1, 3)*divw(w)*kron(dd, m)
# L*_mom(V) pieces (with the +signs of the adjoint), V=(v,q):
def pA(v, dd):  return sum(a[l]*d(v[dd], l) for l in I3)                                  # a.grad v
def pL(v, dd):  return nu*2*sum(d(symg(v, dd, m), m) for m in I3)                          # nu 2 div(sym grad v)
def pC(v, dd):  return -nu*sp.Rational(2, 3)*d(divw(v), dd)                                # -nu 2/3 grad(div v)
# ViscProj grad w = 2 Pi grad w = (grad w + grad w^T) - 2/3 div(w) I  (factor 2 absorbed, as in Lapbar);
# its grad-beta contraction splits into the symmetric (Gbeta) and divergence (Dbeta) pieces:
def pGb(v, dd): return nu*sum((d(v[dd], m) + d(v[m], dd))*d(beta, m) for m in I3)           # nu (grad v+grad v^T).grad beta
def pDb(v, dd): return -nu*sp.Rational(2, 3)*divw(v)*d(beta, dd)                            # -2nu/3 div(v) grad beta
def pR(v, dd):  return -sum(sig[dd, k]*v[k] for k in I3)/alpha                             # -sigma v/alpha
def Lstar_mom(v, q, dd):
    return pA(v, dd) + pL(v, dd) + pC(v, dd) + pGb(v, dd) + pDb(v, dd) + pR(v, dd) + d(q, dd)
# L_mom(U) pieces (operator signs), U=(u,p):
def L_mom(uu, p_, dd):
    visc = -nu*2*sum(d(symg(uu, dd, m), m) for m in I3) + nu*sp.Rational(2, 3)*d(divw(uu), dd)
    gb = -nu*sum((d(uu[dd], m) + d(uu[m], dd))*d(beta, m) for m in I3) + nu*sp.Rational(2, 3)*divw(uu)*d(beta, dd)
    return sum(a[l]*d(uu[dd], l) for l in I3) + visc + gb + d(p_, dd) + sum(sig[dd, k]*uu[k] for k in I3)/alpha
def L_mass(uu, p_): return divw(uu) + sum(uu[k]*d(beta, k) for k in I3) + eps*p_/alpha
def Lstar_mass(v, q): return divw(v) + sum(v[k]*d(beta, k) for k in I3) - eps*q/alpha

# ---- Gateaux derivatives ----
def Dvel(expr, j):
    return sp.expand(expr.xreplace({u[k]: (Nb if k == j else sp.Integer(0)) for k in I3}).doit())
def Dpre(expr):
    return sp.expand(expr.xreplace({pp: Nb}).doit())

# velocity test in component i:  v_d = Na delta_{di}
def vtest(i): return [Na if dd == i else sp.Integer(0) for dd in I3]

print("=" * 70)
print("Assembled stabilization matrices vs. bilinear form eq:StabilizationLVLU")
print("=" * 70)

# =====================================================================
# K_VU : test velocity comp i, trial velocity comp j.
#   family X_• (printed results) must equal  tau1 alpha^2 pX(v_i) . L_mom(u, p=0)
# =====================================================================
print("\n[K_VU] families X_• = tau1 alpha^2 (piece X of L*_mom) . L_mom(u)")

def famcheck(name, piece_fun, printed_sum_ij):
    """piece_fun(v,dd): a piece of L*_mom; verify sum_Y X_Y == tau1 a^2 piece . L_mom(u)."""
    ok = True
    for i in I3:
        v = vtest(i)
        bil = tau1*alpha**2*sum(piece_fun(v, dd)*L_mom(u, sp.Integer(0), dd) for dd in I3)
        for j in I3:
            if sp.simplify(Dvel(bil, j) - printed_sum_ij(i, j)) != 0:
                ok = False
    check(name, ok)

def S(f):  # helper: sum over a spatial index 0..2
    return sum(f(t) for t in I3)

# --- A_• : A_A, A_L, A_C, A_Gβ, A_Dβ, A_σ ---
def A_family(i, j):
    aNa = S(lambda l: a[l]*d(Na, l))
    A_A = tau1*alpha**2*aNa*S(lambda m: a[m]*d(Nb, m))*kron(i, j)
    A_L = -tau1*alpha**2*nu*aNa*(S(lambda m: d(d(Nb, m), m))*kron(i, j) + d(d(Nb, i), j))
    A_C = sp.Rational(2, 3)*tau1*alpha**2*nu*aNa*d(d(Nb, i), j)
    A_Gb = -tau1*alpha*nu*aNa*(d(Nb, i)*d(alpha, j) + S(lambda m: d(alpha, m)*d(Nb, m))*kron(i, j))
    A_Db = sp.Rational(2, 3)*tau1*alpha*nu*aNa*d(alpha, i)*d(Nb, j)
    A_s = tau1*alpha*aNa*Nb*sig[i, j]
    return A_A + A_L + A_C + A_Gb + A_Db + A_s
famcheck("A_•  (A_A,A_L,A_C,A_Gbeta,A_Dbeta,A_sigma)", pA, A_family)

def L_family(i, j):
    ll = lambda f: S(lambda l: d(d(f, l), l))
    aNb = S(lambda m: a[m]*d(Nb, m))
    L_A = tau1*alpha**2*nu*(ll(Na)*kron(i, j) + d(d(Na, i), j))*aNb
    L_L = -tau1*alpha**2*nu**2*(ll(Na)*ll(Nb)*kron(i, j) + ll(Na)*d(d(Nb, i), j)
                               + d(d(Na, i), j)*ll(Nb) + S(lambda k: d(d(Na, i), k)*d(d(Nb, j), k)))
    L_C = sp.Rational(2, 3)*tau1*alpha**2*nu**2*(ll(Na)*d(d(Nb, i), j) + S(lambda k: d(d(Na, i), k)*d(d(Nb, k), j)))
    L_Gb = -tau1*alpha*nu**2*(ll(Na)*(d(Nb, i)*d(alpha, j) + S(lambda m: d(alpha, m)*d(Nb, m))*kron(i, j))
                              + S(lambda k: d(d(Na, i), k)*d(Nb, k))*d(alpha, j) + d(d(Na, i), j)*S(lambda m: d(alpha, m)*d(Nb, m)))
    L_Db = sp.Rational(2, 3)*tau1*alpha*nu**2*(ll(Na)*d(alpha, i) + S(lambda k: d(alpha, k)*d(d(Na, i), k)))*d(Nb, j)
    L_s = tau1*alpha*nu*(ll(Na)*sig[i, j]*Nb + S(lambda k: d(d(Na, i), k)*sig[k, j])*Nb)
    return L_A + L_L + L_C + L_Gb + L_Db + L_s
famcheck("L_•  (L_A,L_L,L_C,L_Gbeta,L_Dbeta,L_sigma)", pL, L_family)

def C_family(i, j):
    aNb = S(lambda m: a[m]*d(Nb, m))
    C_A = -sp.Rational(2, 3)*tau1*alpha**2*nu*d(d(Na, i), j)*aNb
    C_L = sp.Rational(2, 3)*tau1*alpha**2*nu**2*(d(d(Na, i), j)*S(lambda m: d(d(Nb, m), m))
                                                 + S(lambda k: d(d(Na, i), k)*d(d(Nb, k), j)))
    C_C = -sp.Rational(4, 9)*tau1*alpha**2*nu**2*S(lambda k: d(d(Na, k), i)*d(d(Nb, k), j))
    C_Gb = sp.Rational(2, 3)*tau1*alpha*nu**2*(S(lambda k: d(d(Na, i), k)*d(Nb, k))*d(alpha, j)
                                               + d(d(Na, i), j)*S(lambda m: d(alpha, m)*d(Nb, m)))
    C_Db = -sp.Rational(4, 9)*tau1*alpha*nu**2*S(lambda k: d(alpha, k)*d(d(Na, i), k))*d(Nb, j)
    C_s = -sp.Rational(2, 3)*tau1*alpha*nu*S(lambda k: d(d(Na, k), i)*sig[k, j])*Nb
    return C_A + C_L + C_C + C_Gb + C_Db + C_s
famcheck("C_•  (C_A,C_L,C_C,C_Gbeta,C_Dbeta,C_sigma)", pC, C_family)

def Gb_family(i, j):
    gNa = lambda jj: d(alpha, jj)            # shorthand
    aNb = S(lambda m: a[m]*d(Nb, m))
    Gb_A = tau1*alpha*nu*(d(alpha, i)*d(Na, j) + S(lambda l: d(alpha, l)*d(Na, l))*kron(i, j))*aNb
    Gb_L = -tau1*alpha*nu**2*(d(alpha, i)*(d(Na, j)*S(lambda m: d(d(Nb, m), m)) + S(lambda k: d(Na, k)*d(d(Nb, k), j)))
                              + S(lambda l: d(alpha, l)*d(Na, l))*(d(d(Nb, i), j) + S(lambda m: d(d(Nb, m), m))*kron(i, j)))
    Gb_C = sp.Rational(2, 3)*tau1*alpha*nu**2*(d(alpha, i)*S(lambda k: d(Na, k)*d(d(Nb, k), j))
                                               + S(lambda l: d(alpha, l)*d(Na, l))*d(d(Nb, i), j))
    Gb_G = -tau1*nu**2*(d(alpha, i)*(S(lambda k: d(Na, k)*d(Nb, k))*d(alpha, j) + d(Na, j)*S(lambda m: d(alpha, m)*d(Nb, m)))
                        + S(lambda l: d(alpha, l)*d(Na, l))*(d(Nb, i)*d(alpha, j) + S(lambda m: d(alpha, m)*d(Nb, m))*kron(i, j)))
    Gb_D = sp.Rational(2, 3)*tau1*nu**2*(S(lambda k: d(alpha, k)*d(Na, k)) + S(lambda l: d(alpha, l)*d(Na, l)))*d(alpha, i)*d(Nb, j)
    Gb_s = tau1*nu*(d(alpha, i)*S(lambda k: d(Na, k)*sig[k, j]) + S(lambda l: d(alpha, l)*d(Na, l))*sig[i, j])*Nb
    return Gb_A + Gb_L + Gb_C + Gb_G + Gb_D + Gb_s
famcheck("Gbeta_•  (G_bA,G_bL,G_bC,G_bG,G_bD,G_bsigma)", pGb, Gb_family)

def Db_family(i, j):
    aNb = S(lambda m: a[m]*d(Nb, m))
    Db_A = -sp.Rational(2, 3)*tau1*alpha*nu*d(Na, i)*d(alpha, j)*aNb
    Db_L = sp.Rational(2, 3)*tau1*alpha*nu**2*d(Na, i)*(d(alpha, j)*S(lambda m: d(d(Nb, m), m)) + S(lambda k: d(alpha, k)*d(d(Nb, k), j)))
    Db_C = -sp.Rational(4, 9)*tau1*alpha*nu**2*d(Na, i)*S(lambda k: d(alpha, k)*d(d(Nb, j), k))
    Db_G = sp.Rational(4, 3)*tau1*nu**2*d(Na, i)*S(lambda l: d(alpha, l)*d(Nb, l))*d(alpha, j)
    Db_D = -sp.Rational(4, 9)*tau1*nu**2*S(lambda k: d(alpha, k)*d(alpha, k))*d(Na, i)*d(Nb, j)
    Db_s = -sp.Rational(2, 3)*tau1*nu*d(Na, i)*Nb*S(lambda k: d(alpha, k)*sig[k, j])
    return Db_A + Db_L + Db_C + Db_G + Db_D + Db_s
famcheck("Dbeta_•  (D_bA,D_bL,D_bC,D_bG,D_bD,D_bsigma)", pDb, Db_family)

def R_family(i, j):
    R_sA = -tau1*alpha*Na*S(lambda m: a[m]*d(Nb, m))*sig[i, j]
    R_sL = tau1*alpha*nu*Na*(S(lambda l: d(d(Nb, l), l))*sig[i, j] + S(lambda k: sig[i, k]*d(d(Nb, j), k)))
    R_sC = -sp.Rational(2, 3)*tau1*alpha*nu*S(lambda k: sig[i, k]*d(d(Nb, k), j))*Na
    R_Gb = tau1*nu*(Na*S(lambda k: sig[i, k]*d(Nb, k))*d(alpha, j) + Na*S(lambda m: d(alpha, m)*d(Nb, m))*sig[i, j])
    R_Db = -sp.Rational(2, 3)*tau1*nu*Na*S(lambda k: sig[i, k]*d(alpha, k))*d(Nb, j)
    R_Rs = -tau1*S(lambda k: sig[i, k]*sig[k, j])*Na*Nb
    return R_sA + R_sL + R_sC + R_Gb + R_Db + R_Rs
famcheck("R_•  (R_sA,R_sL,R_sC,R_Gbeta,R_Dbeta,R_Rsigma)", pR, R_family)

# K_VU mass terms (tau2 part): verify against tau2 alpha^2 L*_mass(v) L_mass(u)
def massVU(i, j):
    G_U = tau2*Na*Nb*d(alpha, i)*d(alpha, j)
    G_D = tau2*alpha*Na*d(alpha, i)*d(Nb, j)
    D_U = tau2*alpha*d(Na, i)*d(alpha, j)*Nb     # eq:DULHSStabilizationTerm (an alpha typo here was found and fixed by this check)
    D_D = tau2*alpha**2*d(Na, i)*d(Nb, j)
    return G_U + G_D + D_U + D_D
okm = True
for i in I3:
    v = vtest(i); bil = tau2*alpha**2*Lstar_mass(v, sp.Integer(0))*L_mass(u, sp.Integer(0))
    for j in I3:
        if sp.simplify(Dvel(bil, j) - massVU(i, j)) != 0: okm = False
check("mass_•  (G_U,G_D,D_U,D_D)   [found+fixed an alpha typo in D_U]", okm)

# =====================================================================
# K_QU : mass-row test (q=Na), trial velocity comp j.
#   G_• = tau1 alpha^2 grad(Na) . L_mom(u) ;  mass = tau2 alpha^2 L*_mass(q=Na) L_mass(u)
# =====================================================================
print("\n[K_QU] G_• and mass terms")
def vmasstest():
    return ([sp.Integer(0)]*3, Na)            # (v=0, q=Na)
def G_family(j):
    G_A = tau1*alpha**2*d(Na, j)*S(lambda l: a[l]*d(Nb, l))
    G_L = -tau1*nu*alpha**2*(d(Na, j)*S(lambda l: d(d(Nb, l), l)) + S(lambda k: d(Na, k)*d(d(Nb, j), k)))
    G_C = sp.Rational(2, 3)*tau1*nu*alpha**2*S(lambda k: d(Na, k)*d(d(Nb, k), j))
    G_Gb = -tau1*alpha*nu*(S(lambda k: d(Na, k)*d(Nb, k))*d(alpha, j) + d(Na, j)*S(lambda l: d(alpha, l)*d(Nb, l)))
    G_Db = sp.Rational(2, 3)*tau1*alpha*nu*S(lambda k: d(alpha, k)*d(Na, k))*d(Nb, j)
    G_R = tau1*alpha*S(lambda k: d(Na, k)*sig[k, j])*Nb
    return G_A + G_L + G_C + G_Gb + G_Db + G_R
vq = vmasstest()
okq = True
for j in I3:
    bil = tau1*alpha**2*sum(d(Na, dd)*L_mom(u, sp.Integer(0), dd) for dd in I3)  # L*_mom(q=Na)=grad Na
    if sp.simplify(Dvel(bil, j) - G_family(j)) != 0: okq = False
check("G_•  (G_A,G_L,G_C,G_Gbeta,G_Dbeta,G_R)", okq)
def massQU(j):
    Q_eD = -tau2*alpha*eps*Na*d(Nb, j)
    Q_U = -tau2*eps*Na*Nb*d(alpha, j)
    return Q_eD + Q_U
okqm = True
for j in I3:
    bil = tau2*alpha**2*Lstar_mass([sp.Integer(0)]*3, Na)*L_mass(u, sp.Integer(0))
    if sp.simplify(Dvel(bil, j) - massQU(j)) != 0: okqm = False
check("mass_•  (Q_epsD, Q_U)", okqm)

# =====================================================================
# K_VP : velocity test comp i, pressure trial.
#   {A_G,L_G,C_G,G_G,D_G,R_sigmaG} = tau1 alpha^2 L*_mom(v_i) . grad(Nb)
#   {D_P,G_P} from tau2 mass part.
# =====================================================================
print("\n[K_VP] L*_mom(v).grad p  and  mass terms")
def VPfamily(i):
    AG = tau1*alpha**2*S(lambda l: a[l]*d(Na, l))*d(Nb, i)
    LG = tau1*alpha**2*nu*(S(lambda l: d(d(Na, l), l))*d(Nb, i) + S(lambda k: d(d(Na, i), k)*d(Nb, k)))
    CG = -sp.Rational(2, 3)*tau1*alpha**2*nu*S(lambda k: d(d(Na, i), k)*d(Nb, k))
    GG = tau1*alpha*nu*(d(alpha, i)*S(lambda k: d(Na, k)*d(Nb, k)) + S(lambda l: d(alpha, l)*d(Na, l))*d(Nb, i))
    DG = -sp.Rational(2, 3)*tau1*alpha*nu*d(Na, i)*S(lambda k: d(alpha, k)*d(Nb, k))
    RsG = -tau1*alpha*S(lambda k: sig[i, k]*d(Nb, k))*Na
    return AG + LG + CG + GG + DG + RsG
okvp = True
for i in I3:
    v = vtest(i); bil = tau1*alpha**2*sum(Lstar_mom(v, sp.Integer(0), dd)*d(pp, dd) for dd in I3)
    if sp.simplify(Dpre(bil) - VPfamily(i)) != 0: okvp = False
check("A_G,L_G,C_G,G_G,D_G,R_sigmaG  = tau1 a^2 L*_mom(v).grad p", okvp)
def VPmass(i):
    D_P = tau2*alpha*eps*d(Na, i)*Nb
    G_P = tau2*eps*Na*Nb*d(alpha, i)        # eq:GPLHSStabilizationTerm (a N^a/alpha typo here was found and fixed by this check)
    return D_P + G_P
okvpm = True
for i in I3:
    v = vtest(i); bil = tau2*alpha**2*Lstar_mass(v, sp.Integer(0))*L_mass([sp.Integer(0)]*3, pp)
    if sp.simplify(Dpre(bil) - VPmass(i)) != 0: okvpm = False
check("mass_•  (D_P, G_P)   [found+fixed a typo in G_P]", okvpm)

# =====================================================================
# K_QP : mass test (q=Na), pressure trial.  G + Q_P  (P_Q is Galerkin)
# =====================================================================
print("\n[K_QP] G and Q_P")
def QPstab():
    G = tau1*alpha**2*S(lambda k: d(Na, k)*d(Nb, k))
    Q_P = -tau2*eps**2*Na*Nb
    return G + Q_P
bil = (tau1*alpha**2*sum(d(Na, dd)*d(pp, dd) for dd in I3)
       + tau2*alpha**2*Lstar_mass([sp.Integer(0)]*3, Na)*L_mass([sp.Integer(0)]*3, pp))
check("G + Q_P  = tau1 a^2 grad(Na).grad p + tau2 a^2 L*_mass(q) L_mass(p)",
      sp.simplify(Dpre(bil) - QPstab()) == 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} family/block checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
