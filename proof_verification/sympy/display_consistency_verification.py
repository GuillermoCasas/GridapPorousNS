#!/usr/bin/env python3
# =============================================================================
# display_consistency_verification.py
#
# Closes two coverage gaps in the symbolic suite -- the exact gaps that let the
# 2026-07 external review catch two hand-transcription defects the *existing*
# scripts could not see:
#
#   (F2) eq:StabilizationLVLU / eq:StabilizationLVF -- the *beta-factored compact
#        display* of the strong residual L U. The suite verifies the un-factored
#        operator (K_ij, in cdr_operator_verification.py) AND the *assembled*
#        elemental matrices (elemental_bilinear_form_verification.py), but never
#        the printed beta-factored display that sits *between* them. That display
#        carried a missing factor 2 on the porosity-gradient viscous cross-term
#        (`nu Pi grad u . grad beta`  ->  `2 nu Pi grad u . grad beta`). Because
#        the assembled matrices absorb the 2 into their symmetrized index
#        structure, they stayed correct and the suite passed while the display
#        was wrong. This script checks the beta-factoring identity directly.
#
#   (F1) eq:weak_form_eliminated_subscales -- the *motivational* scale-elimination
#        identity, off the computational path the suite walks. Its fine-scale
#        term was printed with a MINUS where the split equations force a PLUS.
#        We verify the sign on an exact finite-dimensional analogue of the VMS
#        coarse/fine (static-condensation) splitting.
#
# Both checks PASS on the corrected manuscript and each additionally asserts that
# the *pre-fix* form would have FAILED, i.e. the check has discriminating power.
#
# Run:  python3 display_consistency_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    print(f"  [{tag}] {name}")
    return ok

# ---------------------------------------------------------------------------
# (F2) beta-factored strong-residual display  (eq:StabilizationLVLU, LVF)
#
#   strong viscous momentum term      =  -2 div(alpha nu Pi grad u)          (cdr_operator)
#   pulling alpha out (beta = log alpha, grad alpha = alpha grad beta):
#   (1/alpha)(-2 div(alpha nu Pi grad u)) = - nu Deltabar u - 2 nu (Pi grad u).grad beta
#   with  Deltabar u := 2 div(grad^S u) - (2/3) grad(div u).
#
#   The printed display had coefficient `nu` (not `2 nu`) on the grad-beta term.
# ---------------------------------------------------------------------------
print("=" * 72)
print("(F2) Display consistency: beta-factored strong residual (eq:StabilizationLVLU)")
print("=" * 72)

x = sp.symbols('x1 x2 x3', real=True)
nu = sp.symbols('nu', positive=True)
D = 3
def d(f, i): return sp.diff(f, x[i])
def kron(p, q): return 1 if p == q else 0

alpha = sp.Function('alpha', positive=True)(*x)
u = [sp.Function(f'u{i+1}')(*x) for i in range(D)]
beta = sp.log(alpha)
divu = sum(d(u[k], k) for k in range(D))

def Pi(a, i):                       # (Pi grad u)_{a i}: deviatoric symmetric gradient
    return sp.Rational(1, 2) * (d(u[i], a) + d(u[a], i)) - sp.Rational(1, D) * divu * kron(a, i)

def gradS(a, i):                    # (grad^S u)_{a i} = 1/2 (d_a u_i + d_i u_a)
    return sp.Rational(1, 2) * (d(u[i], a) + d(u[a], i))

def Deltabar(a):                    # (Deltabar u)_a = 2 div(grad^S u)_a - (2/3) d_a(div u)
    return 2 * sum(d(gradS(a, i), i) for i in range(D)) - sp.Rational(2, 3) * d(divu, a)

# un-factored strong viscous term, per component a
strong_visc = [-2 * sum(d(alpha * nu * Pi(a, i), i) for i in range(D)) for a in range(D)]

def cross(a):                       # (Pi grad u . grad beta)_a = sum_i Pi(a,i) d_i beta
    return sum(Pi(a, i) * d(beta, i) for i in range(D))

def factored(a, coeff):             # -nu Deltabar u_a - coeff*nu (Pi grad u . grad beta)_a
    return -nu * Deltabar(a) - coeff * nu * cross(a)

# corrected display (coeff = 2) must reproduce the un-factored operator / alpha
ok_correct = all(sp.simplify(strong_visc[a] / alpha - factored(a, 2)) == 0 for a in range(D))
check("(1/alpha)(-2 div(alpha nu Pi grad u)) == -nu Deltabar u - 2 nu Pi grad u . grad beta "
      "[corrected eq:StabilizationLVLU, all 3 components]", ok_correct)

# the pre-fix display (coeff = 1) must NOT reproduce it -> the check is discriminating
typo_mismatch = any(sp.simplify(strong_visc[a] / alpha - factored(a, 1)) != 0 for a in range(D))
check("pre-fix coefficient `nu` (factor 1) fails the same identity "
      "[the F2 typo would have been caught]", typo_mismatch)

# ---------------------------------------------------------------------------
# (F1) sign of the eliminated fine-scale term  (eq:weak_form_eliminated_subscales)
#
#   Split L U = F into coarse (U_h) and fine (Utilde) by static condensation.
#   With residual R U_h := F - L U_h and fine Green operator Ltilde^{-1} on the
#   fine block, the elimination gives, for coarse test V_h,
#       B(u, U_h, V_h) + <L Ltilde^{-1} R U_h, V_h> = L(V_h)              (PLUS).
#   The manuscript printed a MINUS. We verify the sign on an exact 5x5 instance
#   (coarse dim 3, fine dim 2), using rational arithmetic so residuals are exact.
# ---------------------------------------------------------------------------
print("=" * 72)
print("(F1) Sign of the eliminated fine-scale term (eq:weak_form_eliminated_subscales)")
print("=" * 72)

L = sp.Matrix([[4, 1, 0, 1, 0],
               [1, 5, 1, 0, 1],
               [0, 1, 6, 1, 0],
               [1, 0, 1, 7, 1],
               [0, 1, 0, 1, 8]])          # non-degenerate, invertible
F = sp.Matrix([1, 2, 3, 4, 5])
kc = 3                                     # coarse dimension (indices 0..2); fine = 3..4

A = L[:kc, :kc]; B = L[:kc, kc:]           # coarse-coarse, coarse-fine
C = L[kc:, :kc]; Dff = L[kc:, kc:]         # fine-coarse, fine-fine (= Ltilde)
Fc = F[:kc, :]; Ff = F[kc:, :]
Dinv = Dff.inv()

# eliminated coarse equation: (A - B Dinv C) U_h = Fc - B Dinv Ff
Uh = (A - B * Dinv * C).inv() * (Fc - B * Dinv * Ff)
Utilde = Dinv * (Ff - C * Uh)              # fine scale from the coarse residual R U_h = Ff - C Uh

# (i) the split reconstructs the exact solution  L [Uh; Utilde] = F
Ufull = sp.Matrix.vstack(Uh, Utilde)
check("VMS split reconstructs the exact solution: L [U_h; Utilde] == F",
      sp.simplify(L * Ufull - F) == sp.zeros(5, 1))

# The coarse-test image of the eliminated fine-scale term  <L Ltilde^{-1} R U_h, V_h>
# is the coarse part of L*Utilde, i.e. B*Utilde. B(u,U_h,V_h) is A*U_h; L(V_h) is Fc.
plus_residual = A * Uh + B * Utilde - Fc          # eliminated form with PLUS
minus_residual = A * Uh - B * Utilde - Fc          # eliminated form with the printed MINUS

check("eliminated form with PLUS is exact: A U_h + <L Ltilde^{-1} R U_h> - L(V_h) == 0",
      sp.simplify(plus_residual) == sp.zeros(kc, 1))
check("eliminated form with MINUS is NOT satisfied [the F1 sign typo would have been caught]",
      sp.simplify(minus_residual) != sp.zeros(kc, 1))

# ---------------------------------------------------------------------------
print("=" * 72)
n_fail = sum(1 for tag, _ in results if tag == "FAIL")
print(f"SUMMARY: {len(results) - n_fail}/{len(results)}")
print("=" * 72)
raise SystemExit(1 if n_fail else 0)
