#!/usr/bin/env python3
# =============================================================================
# subscale_norm_verification.py
#
# Numerical (Monte-Carlo) verification of the operator-norm inequality that
# underlies the stabilization-parameter design, eq:BoundProjectionOfLBySubscales
# of "A stabilized finite element method ... porous media", together with the
# clarification added in amendment B9.
#
# The design rests on the pointwise bound (an equality up to the projection,
# which is a contraction):
#       | Lhat(k) Uhat |_Lambda^2  <=  | Lhat(k) |_Lambda^2  | Uhat |_{Lambda^{-1}}^2 ,
# where, for a vector v,  |v|_Lambda^2 = v^dagger Lambda v  (Lambda real SPD),
# and |Lhat|_Lambda is the operator norm of the (complex) matrix Lhat from
# (C^n, |.|_{Lambda^{-1}}) to (C^n, |.|_Lambda), which -- per B9 -- equals
#       |Lhat|_Lambda^2 = rho_{Lambda^{-1}}( Lhat^dagger Lambda Lhat )
#                       = lambda_max( Lambda Lhat^dagger Lambda Lhat ).
#
# This is exactly the inequality used to motivate the design condition
# |tau_K^{-1}|_Lambda^2 <= |Lhat(k0)|_Lambda^2.  Being an operator-norm bound it
# holds for ANY Lhat; we confirm it over many random (Lambda, Lhat, Uhat),
# verify the operator-norm formula, and confirm tightness (equality at the
# maximizing eigenvector).
#
# Run:  python3 subscale_norm_verification.py     (requires numpy)
# =============================================================================
import numpy as np

rng = np.random.default_rng(20260611)
results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

def rand_spd(n):
    A = rng.standard_normal((n, n)) + 1j*0          # real SPD metric
    A = rng.standard_normal((n, n))
    return A @ A.T + n*np.eye(n)                      # symmetric positive definite
def rand_cplx(n, m=None):
    m = m or n
    return rng.standard_normal((n, m)) + 1j*rng.standard_normal((n, m))

print("=" * 70)
print("Operator-norm bound eq:BoundProjectionOfLBySubscales -- numeric checks")
print("=" * 70)

def rayleigh(M, Lami, w):                            # (w^H M w)/(w^H Lambda^-1 w)
    return float((w.conj().T @ M @ w).real) / float((w.conj().T @ Lami @ w).real)

TRIALS = 3000
for n in (4, 3):                                     # n = d+1 (the porous NS system) and d
    worst_ineq = -np.inf                             # max( |Lh u|^2_L - |Lh|^2_L |u|^2_{L^-1} )  -> must be <= 0
    worst_probe = -np.inf                            # max( RQ(random w) - |Lh|^2_L )              -> must be <= 0
    worst_attain = 0.0                               # | RQ(v_max) - |Lh|^2_L | / |Lh|^2_L
    worst_tight = 0.0                                # tightness gap at v_max
    for _ in range(TRIALS):
        Lam = rand_spd(n); Lami = np.linalg.inv(Lam)
        Lh = rand_cplx(n);  u = rand_cplx(n, 1)
        M = Lh.conj().T @ Lam @ Lh                                   # Hermitian PSD
        # |Lh|_Lambda^2 = lambda_max of the Hermitian-definite pencil (M, Lambda^-1) = max real eig of Lambda M
        evals, evecs = np.linalg.eig(Lam @ M)
        kmax = int(np.argmax(evals.real)); opn2 = float(evals[kmax].real)
        vmax = evecs[:, kmax].reshape(n, 1)
        Lu = Lh @ u
        lhs = float((Lu.conj().T @ Lam @ Lu).real)
        rhs = opn2 * float((u.conj().T @ Lami @ u).real)
        worst_ineq = max(worst_ineq, lhs - rhs)
        worst_probe = max(worst_probe, max(rayleigh(M, Lami, rand_cplx(n, 1)) - opn2 for _ in range(25)))
        worst_attain = max(worst_attain, abs(rayleigh(M, Lami, vmax) - opn2)/opn2)
        Lvm = Lh @ vmax
        worst_tight = max(worst_tight, abs(float((Lvm.conj().T @ Lam @ Lvm).real)
                                           - opn2*float((vmax.conj().T @ Lami @ vmax).real))
                          / max(opn2*float((vmax.conj().T @ Lami @ vmax).real), 1e-12))
    rel = lambda v: v / max(abs(opn2), 1.0)
    check(f"n={n}: |Lhat u|^2_Lambda <= |Lhat|^2_Lambda |u|^2_(Lambda^-1)   ({TRIALS} trials, worst {worst_ineq:.1e})",
          worst_ineq <= 1e-6)
    check(f"n={n}: |Lhat|^2_Lambda = sup of the generalized Rayleigh quotient (random probes never exceed it)",
          worst_probe <= 1e-6)
    check(f"n={n}: |Lhat|^2_Lambda = rho_(Lambda^-1)(Lhat^H Lambda Lhat) attained at v_max  (B9; gap {worst_attain:.1e})",
          worst_attain <= 1e-8)
    check(f"n={n}: bound is tight -- equality at v_max (rel.gap {worst_tight:.1e})", worst_tight <= 1e-8)

# -------------------------------------------------------------------------
print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
