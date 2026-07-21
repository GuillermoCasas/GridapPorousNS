#!/usr/bin/env python3
# =============================================================================
# coverage_vms_operators_verification.py
#
# Symbolic (sympy) verification of the VMS / operator matrix identities of
#   "A stabilized finite element method ... porous media".
#
# WHY THIS SCRIPT EXISTS (coverage-gap closure, 2026-07-21)
# --------------------------------------------------------
# A coverage audit found three algebraically-correct operator identities that
# had NO permanent encoded check.  They are reconstructed here FROM THEIR
# UPSTREAM definitions (not restated), and each is guarded by a DISCRIMINATING
# negative (a plausible wrong variant is asserted to FAIL):
#
#   (A) deviatoric split, article.tex unnumbered display @ line 258:
#         -2 div(alpha nu Pi^DS grad u)
#            = -2 div(alpha nu grad^S u) + (2/d) grad(alpha nu div u)
#       with the deviatoric-of-symmetric projector
#         Pi^DS grad u = grad^S u - (1/d)(div u) I           (line 256-258).
#       Checked for d=2 AND d=3.  Negative: a wrong 1/d factor fails.
#
#   (B) eq:ftSplit  (fourier_appendix.tex ~l.18-22):
#         A_{c,i}(w) + A_{f,i} = A_{v,i} + A_{b,i},
#         A_{v,i} = alpha diag(w_i,...,w_i,0),
#         A_{b,i} = alpha (e_i e_n^T + e_n e_i^T),  n = d+1.
#       A_{c,i}, A_{f,i} are reconstructed from eq:matrices_stationary_strong_problem
#       (article.tex l.311).  Checked for d=2 AND d=3.  Negatives: dropping the
#       symmetric partner e_n e_i^T fails; putting w_i on the pressure diagonal
#       fails; dropping alpha fails.
#
#   (C) eq:FTOfDifferentialOperator  (article.tex l.776):
#         |tau_K^{-1}|_Lambda^2 <= 5 ( |tau_nu|^2 + |tau_c|^2 + |tau_b|^2
#                                       + |tau_sigma|^2 + |tau_gradalpha|^2 ),
#       the triangle + Cauchy-Schwarz (power-mean) bound on the FIVE
#       contributions, sharp with equality when all five coincide.  Encoded via
#       the exact polarization identity  m*sum|v_i|^2 - |sum v_i|^2
#       = sum_{i<j} |v_i - v_j|^2  in an arbitrary positive-definite Lambda-metric,
#       which proves BOTH the >=0 gap (the bound) AND its sharpness.  Negative:
#       the 4-term bound (factor 4 for 5 vectors) is violated by five equal vectors.
#
# SKIPPED: eq:ExplicitExactSubscales and eq:NonlinearStabilizedEquation -- these
# are full nonlinear operator/subscale identities involving the (frozen) inverse
# stabilization operator applied to the strong residual and the projection Pi_h;
# encoding them robustly would require modelling the whole discrete residual and
# the L2 projection, well beyond a self-contained matrix identity, so a faithful
# check is out of reasonable scope here (a vacuous restatement is worse than a skip).
#
# Run:  /tmp/sympy_venv/bin/python coverage_vms_operators_verification.py
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

print("=" * 70)
print("VMS / operator matrix identities -- symbolic checks")
print("=" * 70)


# =====================================================================
# (A) Deviatoric split of the viscous term (article.tex line 258)
#     Pi^DS grad u = grad^S u - (1/d)(div u) I           (line 256-258)
#     -2 div(alpha nu Pi^DS grad u)
#        = -2 div(alpha nu grad^S u) + (2/d) grad(alpha nu div u)
# Reconstructed with genuine differential operators on symbolic fields.
# =====================================================================
print("\n[A] Deviatoric split of the viscous term (l.258)")

def deviatoric_split_residual(d, dfac_num=2, dfac_den=None):
    """Return LHS-RHS (a d-vector of expressions) of the split, using
    (dfac_num/dfac_den) in place of the RHS gradient coefficient 2/d and the
    projector trace coefficient 1/d.  dfac_den defaults to d (the correct case).
    A nonzero residual means the identity fails for that coefficient choice."""
    if dfac_den is None:
        dfac_den = d
    coords = sp.symbols(f'x0:{d}', real=True)
    # arbitrary (nonlinear) velocity field and spatially-varying alpha, nu
    u = [sp.Function(f'u{a}')(*coords) for a in range(d)]
    alpha = sp.Function('alpha')(*coords)
    nu = sp.Function('nu')(*coords)
    def dd(f, b):
        return sp.diff(f, coords[b])
    # velocity gradient G[a][b] = d u_a / d x_b ; grad^S = (G+G^T)/2
    G = [[dd(u[a], b) for b in range(d)] for a in range(d)]
    gradS = [[(G[a][b] + G[b][a]) / 2 for b in range(d)] for a in range(d)]
    divu = sum(G[a][a] for a in range(d))
    # projector: Pi^DS grad u = grad^S u - (1/d)(div u) I   (trace coeff = 1/d here)
    trace_coeff = sp.Rational(1, d)
    PiDS = [[gradS[a][b] - (trace_coeff * divu if a == b else 0) for b in range(d)]
            for a in range(d)]
    # row-wise tensor divergence:  (div T)_a = sum_b d T_ab / d x_b
    def tensor_div(T):
        return [sum(dd(T[a][b], b) for b in range(d)) for a in range(d)]
    def scale(T):  # multiply tensor by alpha*nu
        return [[alpha * nu * T[a][b] for b in range(d)] for a in range(d)]
    LHS = [-2 * v for v in tensor_div(scale(PiDS))]
    div_gradS = tensor_div(scale(gradS))
    grad_divterm = [dd(alpha * nu * divu, a) for a in range(d)]  # grad(alpha nu div u)
    coeff = sp.Rational(dfac_num, dfac_den)
    RHS = [-2 * div_gradS[a] + coeff * grad_divterm[a] for a in range(d)]
    return [sp.simplify(LHS[a] - RHS[a]) for a in range(d)]

for d in (2, 3):
    res = deviatoric_split_residual(d)   # correct 2/d
    check(f"l.258 deviatoric split holds componentwise (d={d})",
          all(r == 0 for r in res))

# Discriminating negatives: a wrong RHS gradient coefficient must FAIL.
for d in (2, 3):
    # use 1/d instead of 2/d on the gradient term
    res_bad = deviatoric_split_residual(d, dfac_num=1, dfac_den=d)
    check(f"l.258 WRONG coeff 1/d (not 2/d) fails (d={d})",
          any(r != 0 for r in res_bad))
    # use 2/(d-1) instead of 2/d
    res_bad2 = deviatoric_split_residual(d, dfac_num=2, dfac_den=d - 1)
    check(f"l.258 WRONG coeff 2/(d-1) fails (d={d})",
          any(r != 0 for r in res_bad2))


# =====================================================================
# (B) eq:ftSplit  (fourier_appendix.tex l.18-22)
#     A_{c,i}(w) + A_{f,i} = A_{v,i} + A_{b,i}
# A_{c,i}, A_{f,i} reconstructed from eq:matrices_stationary_strong_problem
# (article.tex l.311); A_{v,i}, A_{b,i} as printed in eq:ftSplit.
# =====================================================================
print("\n[B] eq:ftSplit  first-order symbol split")

def e(m, n):
    """m-th canonical column vector of R^n (1-indexed m)."""
    v = sp.zeros(n, 1); v[m - 1] = 1; return v

def A_c(i, d, alpha, w):
    """A_{c,i}(w), n x n with n=d+1, from eq:matrices_stationary_strong_problem:
       velocity diagonal w_i, plus divergence row (n, a)=delta_{ia}."""
    n = d + 1
    M = sp.zeros(n, n)
    for a in range(d):
        M[a, a] = w[i - 1]           # w_i on velocity diagonal
    for a in range(1, d + 1):
        M[n - 1, a - 1] = 1 if a == i else 0   # bottom row: delta_{i a} (divergence)
    return alpha * M

def A_f(i, d, alpha):
    """A_{f,i}: pressure-gradient column (a, n)=delta_{ia}."""
    n = d + 1
    M = sp.zeros(n, n)
    for a in range(1, d + 1):
        M[a - 1, n - 1] = 1 if a == i else 0   # right column: delta_{i a}
    return alpha * M

def A_v(i, d, alpha, w, pressure_diag=False):
    """A_{v,i} = alpha diag(w_i,...,w_i,0).  pressure_diag=True is the WRONG
    variant that also puts w_i on the pressure entry."""
    n = d + 1
    diag = [w[i - 1]] * d + [w[i - 1] if pressure_diag else 0]
    return alpha * sp.diag(*diag)

def A_b(i, d, alpha, symmetric=True):
    """A_{b,i} = alpha (e_i e_n^T + e_n e_i^T).  symmetric=False drops the
    e_n e_i^T partner (WRONG variant)."""
    n = d + 1
    M = e(i, n) * e(n, n).T
    if symmetric:
        M = M + e(n, n) * e(i, n).T
    return alpha * M

for d in (2, 3):
    n = d + 1
    alpha = sp.Symbol('alpha')
    w = sp.symbols(f'w1:{d + 1}')  # w1..wd
    ok = True
    for i in range(1, d + 1):
        lhs = A_c(i, d, alpha, w) + A_f(i, d, alpha)
        rhs = A_v(i, d, alpha, w) + A_b(i, d, alpha)
        ok = ok and (sp.simplify(lhs - rhs) == sp.zeros(n, n))
    check(f"eq:ftSplit  A_c+A_f = A_v+A_b for all i (d={d})", ok)

# Discriminating negatives on d=3 (i chosen so the flaw is exposed).
d = 3; n = d + 1; alpha = sp.Symbol('alpha'); w = sp.symbols('w1:4')
i = 2
lhs = A_c(i, d, alpha, w) + A_f(i, d, alpha)
# (b1) drop the symmetric partner e_n e_i^T  -> loses the divergence row coupling
bad1 = A_v(i, d, alpha, w) + A_b(i, d, alpha, symmetric=False)
check("eq:ftSplit  dropping e_n e_i^T partner fails (d=3)",
      sp.simplify(lhs - bad1) != sp.zeros(n, n))
# (b2) put w_i on the pressure diagonal too
bad2 = A_v(i, d, alpha, w, pressure_diag=True) + A_b(i, d, alpha)
check("eq:ftSplit  w_i on pressure diagonal fails (d=3)",
      sp.simplify(lhs - bad2) != sp.zeros(n, n))
# (b3) drop the alpha prefactor on the RHS
bad3 = (A_v(i, d, alpha, w) + A_b(i, d, alpha)) / alpha
check("eq:ftSplit  dropping alpha prefactor fails (d=3)",
      sp.simplify(lhs - bad3) != sp.zeros(n, n))


# =====================================================================
# (C) eq:FTOfDifferentialOperator  (article.tex l.776)
#     |tau_K^{-1}|_Lambda^2 <= 5 * sum_{k=1}^5 |contribution_k|_Lambda^2
# The triangle + Cauchy-Schwarz (power-mean) bound on the FIVE contributions,
# in an arbitrary positive-definite Lambda-metric.  Encoded via the EXACT
# polarization identity  m*sum|v_i|^2 - |sum v_i|^2 = sum_{i<j}|v_i - v_j|^2,
# which is >= 0 (=> the bound) and vanishes iff all v_i coincide (=> sharpness).
# =====================================================================
print("\n[C] eq:FTOfDifferentialOperator  five-term triangle/power-mean bound")

m = 5           # the five contributions: nu, c, b, sigma, grad-alpha
dimv = 3        # arbitrary vector length carrying each contribution
# arbitrary positive Lambda-metric weights (symbolic, assumed > 0)
lam = sp.symbols(f'lam1:{dimv + 1}', positive=True)
Lambda = sp.diag(*lam)
# five arbitrary symbolic vectors
V = [sp.Matrix(sp.symbols(f'v{k}_1:{dimv + 1}', real=True)) for k in range(m)]
def nrm2(v):
    return (v.T * Lambda * v)[0, 0]

S = sp.zeros(dimv, 1)
for v in V:
    S = S + v
sum_sq = sum(nrm2(v) for v in V)
gap = sp.expand(m * sum_sq - nrm2(S))                 # 5*sum|v|^2 - |sum v|^2
pairwise = sum(nrm2(V[a] - V[b]) for a in range(m) for b in range(a + 1, m))
# Polarization identity: gap == sum_{i<j}|v_i - v_j|^2  (exact, hence bound + sharpness)
check("eq:FTOfDifferentialOperator  gap = sum_{i<j}|v_i-v_j|^2 (bound & sharpness)",
      sp.expand(gap - sp.expand(pairwise)) == 0)

# Sharpness: all five vectors equal => equality (gap = 0).
u = sp.Matrix(sp.symbols('u_1:4', real=True))
subs_equal = {}
for k in range(m):
    for a in range(dimv):
        subs_equal[V[k][a]] = u[a]
check("eq:FTOfDifferentialOperator  equality when all five coincide",
      sp.simplify(gap.subs(subs_equal)) == 0)

# Discriminating negative: the 4-term bound (factor 4 for FIVE vectors) is
# violated -- five equal nonzero vectors give 4*5|u|^2 - 25|u|^2 = -5|u|^2 < 0.
gap4 = 4 * sum_sq - nrm2(S)
gap4_equal = sp.simplify(gap4.subs(subs_equal))       # = -5 |u|^2_Lambda
# pick concrete positive weights and a nonzero u to certify strict negativity
concrete = {lam[0]: 1, lam[1]: 1, lam[2]: 1, u[0]: 1, u[1]: 0, u[2]: 0}
check("eq:FTOfDifferentialOperator  4-term bound FAILS (factor 4 < 5)",
      sp.simplify(gap4_equal.subs(concrete)) < 0)
# and factor 5 is exactly tight there (gap = 0, not > 0): any factor < 5 fails
check("eq:FTOfDifferentialOperator  factor 5 is the tight (minimal) constant",
      sp.simplify(gap.subs(subs_equal)) == 0 and
      sp.simplify((5 * sum_sq - nrm2(S)).subs(subs_equal)) == 0)


# =============================================================================
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
