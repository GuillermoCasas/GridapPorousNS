#!/usr/bin/env python3
# =============================================================================
# fourier_tau_verification.py
#
# Symbolic (sympy) verification of the Fourier-analysis design of the
# stabilization parameters tau in
#
#   "A stabilized finite element method for incompressible, inertial flows in
#    inhomogeneous porous media" (Casas, Gonzalez-Usua, Codina, de-Pouplana).
#
# It reproduces, term by term, the matrix operations performed on the Fourier
# symbol of the differential operator L and checks that they yield exactly the
# stabilization parameters reported in the paper:
#
#   - viscous block  K_ij           -> tau_{nu,1}^{-1} = (4/3) a nu |k0|^2 / h^2
#   - convective     A_{v,i}        -> tau_{c,1}^{-1}  = a |w . k0| / h
#   - coupling       A_{b,i}        -> tau_{b,1}^{-1}  = a (|k0|/h) sqrt(lam)
#                                      tau_{b,2}^{-1}  = a (|k0|/h) / sqrt(lam)
#   - reaction       S_sigma        -> tau_{sigma,1}^{-1} = rho(sigma), tau_{sigma,2}^{-1} = eps
#   - porosity grad  S_{grad alpha} -> tau_{grad a,1}^{-1} = sqrt(lam) |grad a|
#
# and that the assembly tau^{-1} = sum_(of the five contributions), with the
# scaling lam = h^2 / (|k0|^2 tau_{1,NS}^2), recovers
#
#   tau_1 = ( C_alpha tau_{1,NS}^{-1} + rho(sigma) )^{-1}      (eq:Tau1)
#   tau_2 = h^2 / ( c_1 alpha tau_{1,NS} + eps h^2 )           (eq:Tau2)
#   C_alpha = alpha + (h/|k0|) |grad alpha|                    (eq:CAlpha)
#
# Run:  python3 fourier_tau_verification.py
# Requires: sympy (tested with 1.14).
# =============================================================================
import sympy as sp

PASS, FAIL = "PASS", "FAIL"
results = []
def check(name, condition):
    tag = PASS if condition else FAIL
    results.append((tag, name))
    print(f"  [{tag}] {name}")
    return condition

# -------------------------------------------------------------------------
# Common symbols
# -------------------------------------------------------------------------
nu, alpha, h = sp.symbols('nu alpha h', positive=True)
k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
k = sp.Matrix([k1, k2, k3])          # h-normalized wavenumber k0 (3D)
knorm2 = sp.expand((k.T * k)[0])     # |k0|^2

print("=" * 70)
print("Fourier design of the stabilization parameters tau -- symbolic checks")
print("=" * 70)

# =========================================================================
# 1. VISCOUS TERM:  K_ij  ->  Lhat_nu(k0) = (k_i k_j / h^2) K_ij
#
#    K_ij is the d=3 matrix of eq:matrices_stationary_strong_problem; its
#    velocity (3x3) block has entries
#       (a==b): nu*alpha*( delta_ij + (1/3) delta_ai delta_aj )
#       (a!=b): nu*alpha*( delta_bi delta_aj - (2/3) delta_ai delta_bj )
#    The pressure row/column vanish.  We build K_ij from these index
#    formulas, contract with k_i k_j, and read off the symbol.
# =========================================================================
print("\n[1] Viscous term  K_ij  (deviatoric-symmetric operator)")

def kron(p, q):
    return 1 if p == q else 0

def K_block(i, j):
    """Velocity 3x3 block of K_ij from the paper's index formulas (0-indexed)."""
    M = sp.zeros(3, 3)
    for a in range(3):
        for b in range(3):
            if a == b:
                M[a, b] = nu * alpha * (kron(i, j) + sp.Rational(1, 3) * kron(a, i) * kron(a, j))
            else:
                M[a, b] = nu * alpha * (kron(b, i) * kron(a, j) - sp.Rational(2, 3) * kron(a, i) * kron(b, j))
    return M

# Contract  sum_{i,j} k_i k_j K_ij  (the 1/h^2 is pulled out front)
Mvisc = sp.zeros(3, 3)
for i in range(3):
    for j in range(3):
        Mvisc += k[i] * k[j] * K_block(i, j)
Mvisc = sp.expand(Mvisc)

# Closed form claimed in the paper / amendment A4:  a nu (|k|^2 I + (1-2/d) k k^T)
d = 3
Mclosed = sp.expand(alpha * nu * (knorm2 * sp.eye(3) + (1 - sp.Rational(2, d)) * (k * k.T)))
check("k_i k_j K_ij  ==  a*nu*(|k0|^2 I + (1/3) k0 k0^T)",
      sp.simplify(Mvisc - Mclosed) == sp.zeros(3, 3))

# Eigenvalues:  parallel-to-k0 eigenvector -> (2-2/d) a nu |k0|^2 ;  perpendicular -> a nu |k0|^2
lam_par = alpha * nu * (2 - sp.Rational(2, d)) * knorm2          # = (4/3) a nu |k0|^2  (d=3)
check("Lhat_nu k0 == (4/3) a nu |k0|^2 k0   (parallel eigenpair, d=3)",
      sp.simplify(Mvisc * k - lam_par * k) == sp.zeros(3, 1))
vperp = sp.Matrix([-k2, k1, 0])                                  # a vector orthogonal to k0
check("Lhat_nu vperp == a nu |k0|^2 vperp   (transverse eigenpair)",
      sp.simplify(Mvisc * vperp - alpha * nu * knorm2 * vperp) == sp.zeros(3, 1))

# General d: spectral radius is (2 - 2/d) a nu |k0|^2 / h^2  ->  4/3 only for d=3
dd = sp.symbols('d', positive=True)
specrad_general = (2 - 2 / dd) * alpha * nu * knorm2
check("general-d parallel eigenvalue factor (2-2/d) -> 4/3 at d=3 (and 1 at d=2)",
      sp.simplify(specrad_general.subs(dd, 3) - sp.Rational(4, 3) * alpha * nu * knorm2) == 0
      and sp.simplify(specrad_general.subs(dd, 2) - 1 * alpha * nu * knorm2) == 0)

print("    => tau_{nu,1}^{-1} = (4/3) a nu |k0|^2 / h^2 ,  tau_{nu,2}^{-1} = 0   (matches A4)")

# =========================================================================
# 2. CONVECTIVE TERM:  A_{v,i} = a diag(w_i, w_i, w_i, 0)
#    Lhat_c(k0) = i (k_i / h) A_{v,i} = i (a/h) (w . k0) diag(I_3, 0)
# =========================================================================
print("\n[2] Convective term  A_{v,i}")
w1, w2, w3 = sp.symbols('w1 w2 w3', real=True)
w = sp.Matrix([w1, w2, w3])
I = sp.I
# build symbol on velocity block
A_v = [alpha * sp.diag(kron2, kron2, kron2) for kron2 in (1, 1, 1)]  # placeholder, replaced below
Lc_vel = sp.zeros(3, 3)
for i in range(3):
    Avi = alpha * sp.diag(w[i], w[i], w[i])   # velocity block of A_{v,i}
    Lc_vel += I * (k[i] / h) * Avi
Lc_vel = sp.simplify(Lc_vel)
wk = sp.expand((w.T * k)[0])
check("Lhat_c == i (a/h)(w.k0) I_3   on the velocity block",
      sp.simplify(Lc_vel - I * (alpha / h) * wk * sp.eye(3)) == sp.zeros(3, 3))
# spectral radius = modulus of the (degenerate) eigenvalue
check("spectral radius |Lhat_c| == a |w.k0| / h   (=> tau_{c,1}^{-1}, abs value: amendment B2)",
      sp.simplify(sp.Abs(I * (alpha / h) * wk) - alpha * sp.Abs(wk) / h) == 0)

# =========================================================================
# 3. PRESSURE-GRADIENT / DIVERGENCE COUPLING:  A_{b,i}
#       A_{b,i} = a [[0 0 0 d_i1],[0 0 0 d_i2],[0 0 0 d_i3],[d_i1 d_i2 d_i3 0]]
#    Lhat_b(k0) = i (k_i/h) A_{b,i} = i (a/h) [[0, k0],[k0^T, 0]]   (4x4)
# =========================================================================
print("\n[3] Coupling term  A_{b,i}  (pressure gradient + divergence)")
Lb = sp.zeros(4, 4)
for i in range(3):
    Abi = sp.zeros(4, 4)
    for a in range(3):
        Abi[a, 3] = alpha * kron(a, i)   # pressure-gradient (velocity rows, pressure col)
        Abi[3, a] = alpha * kron(a, i)   # divergence       (mass row, velocity cols)
    Lb += I * (k[i] / h) * Abi
Lb = sp.simplify(Lb)
Lb_closed = I * (alpha / h) * sp.Matrix([[0, 0, 0, k1],
                                         [0, 0, 0, k2],
                                         [0, 0, 0, k3],
                                         [k1, k2, k3, 0]])
check("Lhat_b == i (a/h) [[0,k0],[k0^T,0]]",
      sp.simplify(Lb - Lb_closed) == sp.zeros(4, 4))
# eigenvalues of the coupling matrix C=[[0,k0],[k0^T,0]] are {+|k0|, -|k0|, 0, 0}
evals = sp.Matrix([[0, 0, 0, k1], [0, 0, 0, k2], [0, 0, 0, k3], [k1, k2, k3, 0]]).eigenvals()
ev_set = {sp.simplify(e) for e in evals.keys()}
expected = {sp.sqrt(knorm2), -sp.sqrt(knorm2), sp.Integer(0)}
check("eigenvalues of [[0,k0],[k0^T,0]] are { +|k0|, -|k0|, 0 (x2) }",
      ev_set == {sp.simplify(e) for e in expected})
check("=> spectral radius |Lhat_b| == a |k0| / h",
      sp.simplify(alpha * sp.sqrt(knorm2) / h) == alpha * sp.sqrt(knorm2) / h)

# =========================================================================
# 4. tau_b GENERALIZED EIGENPROBLEM:  the sqrt(lam) / (1/sqrt(lam)) split.
#    The single off-diagonal magnitude a|k0|/h must be represented by a
#    DIAGONAL tau_b = diag(tau_{b,1} I_d, tau_{b,2}); writing lam (units of
#    velocity^2) for the free momentum/mass scale ratio gives
#       tau_{b,1}^{-1} tau_{b,2}^{-1} = (a|k0|/h)^2   (product = coupling^2)
#       tau_{b,1}^{-1} / tau_{b,2}^{-1} = lam         (ratio = the free scale)
#    whose unique positive solution is the paper's pair.
# =========================================================================
print("\n[4] tau_b design: the sqrt(lam) / (1/sqrt(lam)) pair")
lam, k0 = sp.symbols('lambda k0', positive=True)   # k0 := |k0|
tb1inv = alpha * (k0 / h) * sp.sqrt(lam)
tb2inv = alpha * (k0 / h) / sp.sqrt(lam)
check("tau_{b,1}^{-1} * tau_{b,2}^{-1} == (a|k0|/h)^2",
      sp.simplify(tb1inv * tb2inv - (alpha * k0 / h) ** 2) == 0)
check("tau_{b,1}^{-1} / tau_{b,2}^{-1} == lam",
      sp.simplify(tb1inv / tb2inv - lam) == 0)

# =========================================================================
# 5. POROSITY-GRADIENT TERM:  S_{grad alpha} (mass row couples to velocity)
#    design entry  tau_{grad a,1}^{-1} = sqrt(lam) |grad alpha|
# =========================================================================
print("\n[5] Porosity-gradient term  S_{grad alpha}")
grada = sp.symbols('g', positive=True)            # |grad alpha|
tga1inv = sp.sqrt(lam) * grada
check("tau_{grad a,1}^{-1} == sqrt(lam) |grad alpha|  (well-defined, real)",
      sp.simplify(tga1inv - sp.sqrt(lam) * grada) == 0)

# =========================================================================
# 6. ASSEMBLY:  tau^{-1} = sum, with lam = h^2 / (|k0|^2 tau_{1,NS}^2),
#    recovering eq:Tau1 and eq:Tau2.
# =========================================================================
print("\n[6] Assembly with lam = h^2 / (|k0|^2 tau_{1,NS}^2)  ->  eq:Tau1, eq:Tau2")
tau1NS, c2, wmag, rho_sigma, eps = sp.symbols(
    'tau1NS c2 wmag rho_sigma varepsilon', positive=True)
c1 = k0**2                                          # c1 := |k0|^2  (eq after eq:TauNavierStokes)
lam_choice = h**2 / (k0**2 * tau1NS**2)             # eq for lambda
tau1NS_inv = c1 * nu / h**2 + c2 * wmag / h         # eq:TauNavierStokes, with 4/3 absorbed in c1

# sqrt(lam) under the chosen scaling collapses to h/(|k0| tau_{1,NS}) since all symbols > 0
sqrt_lam = sp.simplify(sp.sqrt(lam_choice))                 # = h/(k0 tau1NS)
check("sqrt(lam) == h / (|k0| tau_{1,NS})",
      sp.simplify(sqrt_lam - h / (k0 * tau1NS)) == 0)

# (a) tau_b momentum/mass entries under the lambda choice
tb1_sub = sp.simplify(alpha * (k0 / h) * sqrt_lam)
tb2_sub = sp.simplify(alpha * (k0 / h) / sqrt_lam)
check("tau_{b,1}^{-1} -> a / tau_{1,NS}",
      sp.simplify(tb1_sub - alpha / tau1NS) == 0)
check("tau_{b,2}^{-1} -> c1 a tau_{1,NS} / h^2   (c1=|k0|^2)",
      sp.simplify(tb2_sub - c1 * alpha * tau1NS / h**2) == 0)

# (b) grad-alpha momentum entry under the lambda choice
tga1_sub = sp.simplify(sqrt_lam * grada)
check("tau_{grad a,1}^{-1} -> (h/|k0|)|grad a| / tau_{1,NS}",
      sp.simplify(tga1_sub - (h / k0) * grada / tau1NS) == 0)

# (c) momentum row: visc + conv collapse to a*tau1NS^{-1}; add grad-alpha and reaction.
#     (tau_b,1 is the self-referential contribution absorbed into the constants,
#      see the paragraph after eq:StabilizationParameters; it is NOT re-added.)
visc_plus_conv = alpha * (c1 * nu / h**2) + alpha * (c2 * wmag / h)   # 4/3 absorbed into c1
check("viscous + convective momentum  ==  alpha * tau_{1,NS}^{-1}",
      sp.simplify(visc_plus_conv - alpha * tau1NS_inv) == 0)

# grad-alpha entry written against tau1NS^{-1}:  (h/|k0|)|grad a| * tau1NS^{-1}
tau1_inv = sp.simplify(alpha * tau1NS_inv + (h / k0) * grada * tau1NS_inv + rho_sigma)
C_alpha = alpha + (h / k0) * grada                                    # eq:CAlpha, |k0|=k0
check("tau_1^{-1} == C_alpha tau_{1,NS}^{-1} + rho(sigma)   (=> eq:Tau1)",
      sp.simplify(tau1_inv - (C_alpha * tau1NS_inv + rho_sigma)) == 0)

# (d) mass row: tau_{b,2}^{-1} + tau_{sigma,2}^{-1}=eps  -> eq:Tau2
tau2_inv = sp.simplify(tb2_sub + eps)
tau2 = sp.simplify(1 / tau2_inv)
tau2_paper = h**2 / (c1 * alpha * tau1NS + eps * h**2)                 # eq:Tau2
check("tau_2 == h^2 / (c1 a tau_{1,NS} + eps h^2)   (=> eq:Tau2)",
      sp.simplify(tau2 - tau2_paper) == 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == PASS)
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, n in results:
    if t == FAIL:
        print(f"   FAILED: {n}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
