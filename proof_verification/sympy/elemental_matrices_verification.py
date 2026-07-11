#!/usr/bin/env python3
# =============================================================================
# elemental_matrices_verification.py
#
# Symbolic (sympy) verification of the elemental matrix contributions of the
# Appendix "Elemental matrices" (eq:GSComponents ... eq:GLHSStabilizationTerm),
#   "A stabilized finite element method ... porous media".
#
# Each elemental component is, per eq:GenericMatrixVU/QU/VP/QP,
#     T_{(ai)(bj)} = d T( N^a delta_ik, U_k^c N^c ) / d U_j^b
# (and analogues for the pressure / mass blocks). For every term below we
#   (a) transcribe the *integrand* T (the expression inside the derivative)
#       from the appendix, written with real shape functions N^a, N^b and the
#       trial fields u, p, and a varying porosity alpha(x);
#   (b) take the Gateaux derivative w.r.t. the trial DOF (b,j) [resp. b];
#   (c) check that it equals the *printed result* of the appendix.
#
# The Gateaux derivative is exact because every left-hand-side term is linear
# in the trial unknown: d/dU_j^b just replaces the trial component k by
# delta_kj N^b (and its derivatives). Second derivatives are handled by genuine
# symbolic differentiation, so the d^2_{lm}=d^2_{ml} symmetry is automatic.
#
# Run:  python3 elemental_matrices_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok
def kron(p, q): return 1 if p == q else 0

x = sp.symbols('x0 x1 x2', real=True)
def d(f, i): return sp.diff(f, x[i])

# Shape functions (real functions => exact, symmetric derivatives)
Na = sp.Function('Na')(*x)
Nb = sp.Function('Nb')(*x)
# Trial fields
u = [sp.Function(f'u{k}')(*x) for k in range(3)]     # velocity trial
pp = sp.Function('pp')(*x)                            # pressure trial
# Porosity and physical parameters
alpha = sp.Function('alpha', positive=True)(*x)
beta = sp.log(alpha)                                  # beta = log alpha (d_l beta = d_l alpha/alpha)
nu, eps, tau1, tau2 = sp.symbols('nu varepsilon tau1 tau2', positive=True)
a = sp.symbols('a0 a1 a2', real=True)                 # advection a_l (frozen, constant)
sig = sp.Matrix(3, 3, lambda p, q: sp.Symbol(f'sig{min(p,q)}{max(p,q)}', real=True))  # symmetric

I3 = range(3)

def Dvel(integrand, j):
    """Gateaux derivative w.r.t. velocity DOF (b,j): trial u_k -> delta_kj N^b."""
    sub = {u[k]: (Nb if k == j else sp.Integer(0)) for k in I3}
    return sp.expand(integrand.xreplace(sub).doit())

def Dpre(integrand):
    """Gateaux derivative w.r.t. pressure DOF b: trial p -> N^b."""
    return sp.expand(integrand.xreplace({pp: Nb}).doit())

def verify_vel(name, integrand_of_i, formula_of_ij):
    ok = all(sp.simplify(Dvel(integrand_of_i(i), j) - formula_of_ij(i, j)) == 0
             for i in I3 for j in I3)
    check(name, ok)

def verify_mass(name, integrand, formula_of_j):   # mass-row test (scalar N^a), trial velocity
    ok = all(sp.simplify(Dvel(integrand, j) - formula_of_j(j)) == 0 for j in I3)
    check(name, ok)

def verify_velP(name, integrand_of_i, formula_of_i):   # velocity test i, pressure trial
    ok = all(sp.simplify(Dpre(integrand_of_i(i)) - formula_of_i(i)) == 0 for i in I3)
    check(name, ok)

print("=" * 70)
print("Elemental matrix contributions (Appendix) -- symbolic checks")
print("=" * 70)

# -------------------------------------------------------------------------
# Galerkin block K_{V,U}  (test velocity comp. i, trial velocity comp. j)
# -------------------------------------------------------------------------
print("\n[K_VU] Galerkin velocity-momentum terms")
# G_S  (eq:GSComponents): alpha nu (d_l u_i + d_i u_l) d_l N^a   ->  alpha nu(delta_ij d_l N^a d_l N^b + d_j N^a d_i N^b)
verify_vel("G_S  = alpha nu (d_l N^a d_l N^b delta_ij + d_j N^a d_i N^b)",
           lambda i: nu*alpha*sum((d(u[i], l) + d(u[l], i))*d(Na, l) for l in I3),
           lambda i, j: nu*alpha*(kron(i, j)*sum(d(Na, l)*d(Nb, l) for l in I3) + d(Na, j)*d(Nb, i)))
# D_nuD (eq:DDComponents): -2/3 alpha nu d_i N^a (div u)  ->  -2/3 alpha nu d_i N^a d_j N^b
verify_vel("D_nuD = -2/3 alpha nu d_i N^a d_j N^b",
           lambda i: -sp.Rational(2, 3)*nu*alpha*d(Na, i)*sum(d(u[m], m) for m in I3),
           lambda i, j: -sp.Rational(2, 3)*nu*alpha*d(Na, i)*d(Nb, j))
# V (eq:VComponents): alpha N^a a_l d_l u_i  ->  alpha N^a a_l d_l N^b delta_ij
verify_vel("V    = alpha N^a a_l d_l N^b delta_ij  (convection)",
           lambda i: alpha*Na*sum(a[l]*d(u[i], l) for l in I3),
           lambda i, j: alpha*Na*sum(a[l]*d(Nb, l) for l in I3)*kron(i, j))
# R_sigma (eq:ReactionTermComponents): N^a sigma_ik u_k  ->  N^a N^b sigma_ij
verify_vel("R_sigma = N^a N^b sigma_ij  (reaction)",
           lambda i: Na*sum(sig[i, k]*u[k] for k in I3),
           lambda i, j: Na*Nb*sig[i, j])

# -------------------------------------------------------------------------
# Galerkin block K_{Q,U}  (mass test, trial velocity comp. j)
# -------------------------------------------------------------------------
print("\n[K_QU] Galerkin mass-velocity terms")
# Q_D (eq:QComponents): alpha N^a div u  ->  alpha N^a d_j N^b
verify_mass("Q_D    = alpha N^a d_j N^b  (alpha div u)",
            alpha*Na*sum(d(u[l], l) for l in I3),
            lambda j: alpha*Na*d(Nb, j))
# G_alphaD (eq:GAlphaComponents): N^a d_k alpha u_k  ->  N^a N^b d_j alpha
verify_mass("G_alphaD = N^a N^b d_j alpha  (u . grad alpha)",
            Na*sum(d(alpha, k)*u[k] for k in I3),
            lambda j: Na*Nb*d(alpha, j))

# -------------------------------------------------------------------------
# Galerkin block K_{V,P}  (velocity test i, pressure trial)  and  K_{Q,P}
# -------------------------------------------------------------------------
print("\n[K_VP / K_QP] Galerkin pressure terms")
# P (eq:PComponents): -alpha p d_i N^a  ->  -alpha d_i N^a N^b
verify_velP("P    = -alpha d_i N^a N^b  (pressure gradient)",
            lambda i: -alpha*pp*d(Na, i),
            lambda i: -alpha*d(Na, i)*Nb)
# G_P (eq:PComponents): -d_i alpha N^a p  ->  -d_i alpha N^a N^b
verify_velP("G_P  = -d_i alpha N^a N^b",
            lambda i: -d(alpha, i)*Na*pp,
            lambda i: -d(alpha, i)*Na*Nb)
# P_Q (eq:PQLHS): eps N^a p  ->  eps N^a N^b
check("P_Q  = eps N^a N^b  (compressibility)",
      sp.simplify(Dpre(eps*Na*pp) - eps*Na*Nb) == 0)

# -------------------------------------------------------------------------
# Representative stabilization terms  (-sum_K <L* V, tau L U>), eq:StabilizationLVLU
#   adjoint test operator pieces:  A = a.grad,  L = nu Lap-bar,  reaction = sigma
# -------------------------------------------------------------------------
print("\n[K_VU stab] Representative stabilization terms")
# A_A (eq:AALHSStabilizationTerm): tau1 alpha^2 (a_l d_l N^a)(a_m d_m u_i)
#   -> tau1 alpha^2 a_l d_l N^a a_m d_m N^b delta_ij
verify_vel("A_A   = tau1 alpha^2 (a.grad N^a)(a.grad N^b) delta_ij",
           lambda i: tau1*alpha**2*(sum(a[l]*d(Na, l) for l in I3))*(sum(a[m]*d(u[i], m) for m in I3)),
           lambda i, j: tau1*alpha**2*sum(a[l]*d(Na, l) for l in I3)*sum(a[m]*d(Nb, m) for m in I3)*kron(i, j))
# A_sigma (eq:ASigmaLHSStabilizationTerm): tau1 alpha (a_l d_l N^a) sigma_ik u_k
#   -> tau1 alpha (a.grad N^a) N^b sigma_ij
verify_vel("A_sigma = tau1 alpha (a.grad N^a) N^b sigma_ij",
           lambda i: tau1*alpha*sum(a[l]*d(Na, l) for l in I3)*sum(sig[i, k]*u[k] for k in I3),
           lambda i, j: tau1*alpha*sum(a[l]*d(Na, l) for l in I3)*Nb*sig[i, j])
# R_Rsigma (eq:RRSigmaLHSStabilizationTerm): -tau1 sigma_ik N^a sigma_km u_m
#   -> -tau1 sigma_ik N^a sigma_kj N^b
verify_vel("R_Rsigma = -tau1 sigma_ik N^a sigma_kj N^b",
           lambda i: -tau1*Na*sum(sig[i, k]*sum(sig[k, m]*u[m] for m in I3) for k in I3),
           lambda i, j: -tau1*sum(sig[i, k]*sig[k, j] for k in I3)*Na*Nb)
# A_L (eq:ALLHSStabilizationTerm): -tau1 alpha^2 nu (a.grad N^a)(Lap u_i + d_i div u)
#   -> -tau1 alpha^2 nu (a.grad N^a)(d^2_mm N^b delta_ij + d^2_ij N^b)
verify_vel("A_L   = -tau1 alpha^2 nu (a.grad N^a)(Lap N^b delta_ij + d^2_ij N^b)",
           lambda i: -tau1*alpha**2*nu*sum(a[l]*d(Na, l) for l in I3)
                     * (sum(d(d(u[i], m), m) for m in I3) + sum(d(d(u[m], i), m) for m in I3)),
           lambda i, j: -tau1*alpha**2*nu*sum(a[l]*d(Na, l) for l in I3)
                     * (sum(d(d(Nb, m), m) for m in I3)*kron(i, j) + d(d(Nb, i), j)))

print("\n[K_QU stab] Representative mass-row stabilization terms")
# G_A (eq:GALHSStabilizationTerm): tau1 alpha^2 (d_k N^a)(a_l d_l u_k) -> tau1 alpha^2 d_j N^a (a.grad N^b)
verify_mass("G_A  = tau1 alpha^2 d_j N^a (a.grad N^b)",
            tau1*alpha**2*sum(d(Na, k)*sum(a[l]*d(u[k], l) for l in I3) for k in I3),
            lambda j: tau1*alpha**2*d(Na, j)*sum(a[l]*d(Nb, l) for l in I3))
# G_R (eq:GRLHSStabilizationTerm): tau1 alpha (d_k N^a) sigma_kl u_l -> tau1 alpha (d_k N^a) sigma_kj N^b
verify_mass("G_R  = tau1 alpha d_k N^a sigma_kj N^b",
            tau1*alpha*sum(d(Na, k)*sum(sig[k, l]*u[l] for l in I3) for k in I3),
            lambda j: tau1*alpha*sum(d(Na, k)*sig[k, j] for k in I3)*Nb)
# D_D (eq:DAlphaDLHSStabilizationTerm): tau2 alpha^2 d_i N^a div u -> tau2 alpha^2 d_i N^a d_j N^b
verify_vel("D_D   = tau2 alpha^2 d_i N^a d_j N^b  (tau2 block)",
           lambda i: tau2*alpha**2*d(Na, i)*sum(d(u[m], m) for m in I3),
           lambda i, j: tau2*alpha**2*d(Na, i)*d(Nb, j))

print("\n[K_VP / K_QP stab] Representative pressure-column stabilization terms")
# A_G (eq:AGLHSStabilizationTerm): tau1 alpha^2 (a.grad N^a)(d_i p) -> tau1 alpha^2 (a.grad N^a) d_i N^b
verify_velP("A_G  = tau1 alpha^2 (a.grad N^a) d_i N^b",
            lambda i: tau1*alpha**2*sum(a[l]*d(Na, l) for l in I3)*d(pp, i),
            lambda i: tau1*alpha**2*sum(a[l]*d(Na, l) for l in I3)*d(Nb, i))
# G (eq:GLHSStabilizationTerm, K_QP): tau1 alpha^2 d_k N^a d_k p -> tau1 alpha^2 d_k N^a d_k N^b
check("G    = tau1 alpha^2 d_k N^a d_k N^b  (pressure Laplacian, K_QP)",
      sp.simplify(Dpre(tau1*alpha**2*sum(d(Na, k)*d(pp, k) for k in I3))
                  - tau1*alpha**2*sum(d(Na, k)*d(Nb, k) for k in I3)) == 0)
# Q_P (eq:QPLHSStabilizationTerm): -tau2 eps^2 N^a p -> -tau2 eps^2 N^a N^b
check("Q_P  = -tau2 eps^2 N^a N^b",
      sp.simplify(Dpre(-tau2*eps**2*Na*pp) - (-tau2*eps**2*Na*Nb)) == 0)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} elemental-component checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
