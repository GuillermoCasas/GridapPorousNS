#!/usr/bin/env python3
# =============================================================================
# projector_algebra_verification.py
#
# Verification of the GENERALIZED VISCOUS PROJECTOR material added to the paper
# on 2026-07-29:
#
#   * standing assumption  H:projector  (A3) with displays eq:PiOrthogonal, eq:PiKorn,
#   * main-text subsection "The viscous projector" (sec:ViscousProjector, Sec. 2.1):
#       lem:projfamily   (eq:projmonotone, eq:devsymidentity, eq:crossibp),
#       lem:projinstances(eq:projconstants),
#   * Appendix B  rem:ftGenericPi  (eq:ftGenericQuad, eq:cPi and the [1,2] factor),
#   * the standalone note theory/viscous_projector_note/, which carries the SHARP
#     characterization (c_Pi > 0) and the witnesses of section E below.  It left the
#     paper on 2026-07-29; sections C and E here remain its machine check.
#
# 2026-07-29 RELOCATION.  The three results above used to live in an Appendix C
# subsubsection (sec:projectors) of asgs_convergence.tex; they were moved into the
# main-text subsection sec:ViscousProjector, which is duplicated VERBATIM in both
# mains.  The source lint of part D was retargeted accordingly: what used to be
# "defined once in App. C" is now "defined once in EACH main", which is what keeps
# the shared appendices (App. B and App. C, \input by both articles) resolvable.
#
# WHY THIS SCRIPT EXISTS.  The Coq development covers the ABSTRACT algebra of the
# generalization (ViscousProjector.v: nested-projection Pythagoras, contraction,
# the Korn chaining and the definiteness step) but deliberately not the CONCRETE
# facts about the four canonical operators -- their idempotence and Frobenius
# self-adjointness, the integration-by-parts identity behind eq:devsymidentity,
# the sharp constants of eq:projconstants and the Fourier symbols.  Those are
# checked here, exactly, so that the two layers together cover the whole of the
# new material.
#
# WHAT IS CHECKED
#   A. Pointwise tensor algebra (exact, symbolic, d = 2, 3) for
#      I, S = sym, D = dev, DS = dev.sym, C = I - DS, skew and sph:
#      idempotence, Frobenius self-adjointness, non-expansiveness, which of them
#      fix the deviatoric-symmetric tensors, the nested-range Pythagoras
#      identity eq:projmonotone, the major symmetry P_{aibj} = P_{bjai} with its
#      consequence K_ji^T = K_ij, and agreement of the d = 3 dev-sym fourth-order
#      tensor with the K_ij displayed in eq:matrices_stationary_strong_problem.
#   B. The H_0^1 identities.  eq:crossibp is verified in the strong, exact form
#      that makes it true -- the integrand difference is a pointwise DIVERGENCE
#      of a compactly supported field -- and the four norm identities are then
#      derived from it by exact pointwise algebra.  The sharp constants of
#      eq:projconstants are exhibited by exact symbolic integration of explicit
#      divergence-free and gradient fields on the unit box, and the failure
#      inadmissibility witnesses (skew, sph) likewise; these now live in the standalone note theory/viscous_projector_note/.
#   C. Fourier symbols: eq:ftGenericQuad, the closed forms, the family bounds
#      1/2|k|^2 I <= T_Pi <= |k|^2 I, the eq:ftViscEig eigenvalues, the resulting
#      spectral-radius factors (2-2/d for DS; 2 for I, S, D) and the ellipticity
#      constants c_Pi of eq:cPi.
#   D. SOURCE LINT (source-coupled, like theorem_statement_verification.py): the
#      constants and coefficients above are re-read from the LIVE .tex sources,
#      so that editing a constant in the paper without editing it here is caught.
#      It also guards the shared-file constraint that made this generalization
#      delicate: asgs_convergence.tex and fourier_appendix.tex are \input by BOTH
#      article.tex and article_v2.tex, so H:projector / eq:PiKorn must be defined
#      in BOTH mains.  Each lint rule reports the number of items it matched; a
#      zero count fails, so a rule cannot go silently vacuous.
#
# Run:  python3 projector_algebra_verification.py
# =============================================================================
import os
import re
import sys
import itertools
import sympy as sp

results = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    line = f"  [{tag}] {name}"
    if detail and not ok:
        line += f"\n         -> {detail}"
    print(line)
    return ok


# =============================================================================
# A. POINTWISE TENSOR ALGEBRA
# =============================================================================
print("=" * 72)
print("A. POINTWISE TENSOR ALGEBRA (exact, symbolic)")
print("=" * 72)


def sym(T):
    return (T + T.T) / 2


def dev(T):
    d = T.shape[0]
    return T - (T.trace() / d) * sp.eye(d)


def sph(T):
    d = T.shape[0]
    return (T.trace() / d) * sp.eye(d)


def skew(T):
    return (T - T.T) / 2


PROJ = {
    "I":    lambda T: T,
    "S":    sym,
    "D":    dev,
    "DS":   lambda T: dev(sym(T)),
    "C":    lambda T: T - dev(sym(T)),
    "skew": skew,
    "sph":  sph,
}
# The four operators the paper's family consists of (lem:projinstances).
FAMILY = ("I", "S", "D", "DS")


def frob(A, B=None):
    if B is None:
        B = A
    return sp.expand(sum(A[i, j] * B[i, j]
                         for i in range(A.shape[0]) for j in range(A.shape[1])))


def generic(d, tag="t"):
    return sp.Matrix(d, d, lambda i, j: sp.Symbol(f"{tag}{i}{j}", real=True))


for d in (2, 3):
    T = generic(d, "t")
    U = generic(d, "u")
    for name, P in PROJ.items():
        idem = sp.simplify(P(P(T)) - P(T)) == sp.zeros(d, d)
        sa = sp.expand(frob(P(T), U) - frob(T, P(U))) == 0
        # Non-expansiveness in its exact form: |T|^2 - |P T|^2 = |T - P T|^2 >= 0.
        gap = sp.expand(frob(T) - frob(P(T)) - frob(T - P(T)))
        check(f"A1 d={d} {name:>4}: idempotent, Frobenius self-adjoint, "
              f"|T|^2-|PT|^2 = |T-PT|^2 (>= 0)",
              idem and sa and gap == 0)

    # A3: which operators fix every deviatoric-symmetric tensor?
    Tds = dev(sym(T))
    for name, P in PROJ.items():
        fixes = sp.simplify(P(Tds) - Tds) == sp.zeros(d, d)
        check(f"A3 d={d} {name:>4}: fixes deviatoric-symmetric tensors "
              f"= {str(fixes):<5} (expected {str(name in FAMILY):<5})",
              fixes == (name in FAMILY))

    # A4: eq:projmonotone for every nested pair inside the family.
    for big, small in (("S", "DS"), ("D", "DS"), ("I", "DS"), ("I", "S"), ("I", "D")):
        Pb, Ps = PROJ[big], PROJ[small]
        # nesting really holds:
        nested = sp.simplify(Pb(Ps(T)) - Ps(T)) == sp.zeros(d, d)
        pyth = sp.expand(frob(Pb(T)) - frob(Ps(T)) - frob(Pb(T) - Ps(T))) == 0
        check(f"A4 d={d} eq:projmonotone |{big}T|^2 = |{small}T|^2 "
              f"+ |({big}-{small})T|^2, and {big}o{small} = {small}",
              nested and pyth)


# A5: fourth-order representation, major symmetry, K_ji^T = K_ij.
def fourth_order(P, d):
    """Pt[a][i][b][j] with (P M)_{ai} = Pt[a,i,b,j] M_{bj}."""
    Pt = {}
    for b in range(d):
        for j in range(d):
            E = sp.zeros(d, d)
            E[b, j] = 1
            PE = P(E)
            for a in range(d):
                for i in range(d):
                    Pt[(a, i, b, j)] = sp.nsimplify(PE[a, i])
    return Pt


for d in (2, 3):
    for name in ("I", "S", "D", "DS", "C"):
        Pt = fourth_order(PROJ[name], d)
        major = all(Pt[(a, i, b, j)] == Pt[(b, j, a, i)]
                    for a in range(d) for i in range(d)
                    for b in range(d) for j in range(d))
        # [K_ij]_{ab} = 2 nu alpha P_{aibj}; K_ji^T = K_ij means P_{aibj} = P_{bjai}
        ksym = all(Pt[(b, j, a, i)] == Pt[(a, i, b, j)]
                   for a in range(d) for i in range(d)
                   for b in range(d) for j in range(d))
        check(f"A5 d={d} {name:>3}: major symmetry P_(aibj)=P_(bjai) "
              f"and hence K_ji^T = K_ij", major and ksym)

# A5b: the d = 3 dev-sym fourth-order tensor reproduces the K_ij entries printed
# in eq:matrices_stationary_strong_problem (in units of nu*alpha):
#   a == b : delta_ij + (1/3) delta_ai delta_aj
#   a != b : delta_bi delta_aj - (2/3) delta_ai delta_bj
d = 3
Pt = fourth_order(PROJ["DS"], d)
ok = True
for a in range(d):
    for b in range(d):
        for i in range(d):
            for j in range(d):
                val = 2 * Pt[(a, i, b, j)]          # K = 2 nu alpha P
                if a == b:
                    ref = (1 if i == j else 0) + sp.Rational(1, 3) * int(a == i) * int(a == j)
                else:
                    ref = (1 if (b == i and a == j) else 0) \
                        - sp.Rational(2, 3) * int((a == i) and (b == j))
                ok &= sp.simplify(val - ref) == 0
check("A5b d=3 dev-sym K_ij matches eq:matrices_stationary_strong_problem exactly", ok)

# A5c: the general-d entries quoted in the Fourier appendix (sec:ftViscous):
#   (K_ij)_ab = nu alpha (delta_ij delta_ab + (1 - 2/d) delta_ai delta_aj)  [a = b]
#             = nu alpha (delta_bi delta_aj - (2/d) delta_ai delta_bj)      [a != b]
for d in (2, 3):
    Pt = fourth_order(PROJ["DS"], d)
    ok = True
    for a in range(d):
        for b in range(d):
            for i in range(d):
                for j in range(d):
                    val = 2 * Pt[(a, i, b, j)]
                    if a == b:
                        ref = (1 if i == j else 0) \
                            + (1 - sp.Rational(2, d)) * int(a == i) * int(a == j)
                    else:
                        ref = (1 if (b == i and a == j) else 0) \
                            - sp.Rational(2, d) * int((a == i) and (b == j))
                    ok &= sp.simplify(val - ref) == 0
    check(f"A5c d={d} dev-sym K_ij matches the general-d entries of sec:ftViscous", ok)


# =============================================================================
# B. THE H_0^1 IDENTITIES
# =============================================================================
print("=" * 72)
print("B. H_0^1 IDENTITIES AND SHARP CONSTANTS (exact)")
print("=" * 72)

for d in (2, 3):
    X = sp.symbols(f"x0:{d}", real=True)
    v = [sp.Function(f"v{i}")(*X) for i in range(d)]
    G = sp.Matrix(d, d, lambda i, l: sp.diff(v[i], X[l]))   # G[i,l] = d_l v_i
    div = sum(G[i, i] for i in range(d))

    # B1 (eq:crossibp).  The two integrands differ by an exact divergence,
    #     d_j (v_i d_i v_j) - d_i (v_i d_j v_j) = d_j v_i d_i v_j - (div v)^2,
    # whose left side integrates to zero for compactly supported (H_0^1) fields.
    # This is the identity that makes eq:crossibp true, verified pointwise.
    F1 = [sum(v[i] * sp.diff(v[j], X[i]) for i in range(d)) for j in range(d)]
    F2 = [v[j] * div for j in range(d)]
    lhs = sum(sp.diff(F1[j], X[j]) - sp.diff(F2[j], X[j]) for j in range(d))
    rhs = frob(G, G.T) - div ** 2
    check(f"B1 d={d} eq:crossibp: (grad v : grad v^T) - (div v)^2 is an exact "
          f"divergence", sp.simplify(sp.expand(lhs - rhs)) == 0)

    # B2-B5.  Pointwise algebra relating each projected square to |grad v|^2,
    # grad v : grad v^T and (div v)^2.  Combined with B1 these ARE the displayed
    # identities of lem:projfamily / lem:projinstances and of the standalone note.
    T = generic(d, "t")
    tr = T.trace()
    cross = frob(T, T.T)
    n2 = frob(T)
    for name, expr in (
        ("S  (|S T|^2 = 1/2|T|^2 + 1/2 T:T^T)",
         n2 / 2 + cross / 2),
        ("DS (|DS T|^2 = 1/2|T|^2 + 1/2 T:T^T - (1/d) tr^2)",
         n2 / 2 + cross / 2 - tr ** 2 / d),
        ("D  (|D T|^2 = |T|^2 - (1/d) tr^2)",
         n2 - tr ** 2 / d),
        ("C  (|C T|^2 = 1/2|T|^2 - 1/2 T:T^T + (1/d) tr^2)",
         n2 / 2 - cross / 2 + tr ** 2 / d),
    ):
        key = name.split()[0]
        check(f"B2 d={d} pointwise {name}",
              sp.expand(frob(PROJ[key](T)) - sp.expand(expr)) == 0)

    # B3.  Substituting T = grad v in B2 and then eliminating the cross term with B1
    # (int grad v : grad v^T = int (div v)^2) must reproduce, coefficient by
    # coefficient, the identities the paper displays.  The reduction is performed here
    # -- it is not a comparison of two hand-typed spellings: the left-hand sides are
    # the B2 expressions with the symbols A = int|grad v|^2, X = int grad v:grad v^T,
    # G = int (div v)^2 substituted, and B1 is applied as the rewrite X -> G.
    A_, X_, G_ = sp.symbols("A X G", positive=True)
    from_B2 = {                      # exactly the right-hand sides checked in B2
        "S":  A_ / 2 + X_ / 2,
        "DS": A_ / 2 + X_ / 2 - G_ / d,
        "D":  A_ - G_ / d,
        "C":  A_ / 2 - X_ / 2 + G_ / d,
    }
    printed = {                      # exactly what the paper prints
        "S":  A_ / 2 + G_ / 2,                                          # lem:projinstances
        "DS": A_ / 2 + (sp.Rational(1, 2) - sp.Rational(1, d)) * G_,    # eq:devsymidentity
        "D":  A_ - G_ / d,                                              # lem:projinstances
        "C":  A_ / 2 + (sp.Rational(1, d) - sp.Rational(1, 2)) * G_,    # note, Ex. 3.1
    }
    for kk in printed:
        reduced = sp.expand(from_B2[kk].subs(X_, G_))       # apply B1
        check(f"B3 d={d} integrated form for {kk}: B2 + B1 gives "
              f"{sp.nsimplify(reduced)}, paper prints {sp.nsimplify(printed[kk])}",
              sp.simplify(reduced - printed[kk]) == 0)
    # DISCRIMINATING: without B1 the reduction must NOT reproduce the printed form for
    # the projectors whose identity actually consumes the cross term (S, DS, C).
    check(f"B3' d={d} B1 is load-bearing (dropping it breaks S, DS and C)",
          all(sp.simplify(from_B2[kk] - printed[kk]) != 0 for kk in ("S", "DS", "C")))

# B6-B8.  Sharpness witnesses, by exact symbolic integration on the unit box.
#   - a divergence-free field  (attains sqrt2 for S and DS),
#   - a gradient field         (attains sqrt(d/(d-1)) for D, and kills skew),
#   - and the inadmissibility witnesses of the standalone note (Ex. 3.2).
def box_integrate(expr, X):
    for xk in X:
        expr = sp.integrate(sp.expand_trig(sp.expand(expr)), (xk, 0, 1))
    return sp.simplify(expr)


def grad_of(vs, X):
    d = len(X)
    return sp.Matrix(d, d, lambda i, l: sp.diff(vs[i], X[l]))


for d in (2, 3):
    X = sp.symbols(f"x0:{d}", real=True)
    pi = sp.pi
    # phi vanishes to second order on the boundary, so grad phi is in H_0^1.
    phi = sp.prod([sp.sin(pi * xk) ** 2 for xk in X])

    # (i) gradient field v = grad phi  -> grad v is the (symmetric) Hessian.
    vg = [sp.diff(phi, xk) for xk in X]
    Gg = grad_of(vg, X)
    Ag = box_integrate(frob(Gg), X)
    Bg = box_integrate(sum(Gg[i, i] for i in range(d)) ** 2, X)
    check(f"B6 d={d} gradient field: int (div v)^2 = int |grad v|^2 "
          f"(equality case of ||div v|| <= ||grad v||)",
          sp.simplify(Ag - Bg) == 0)
    ratio_D = sp.sqrt(Ag / box_integrate(frob(PROJ['D'](Gg)), X))
    check(f"B7 d={d} gradient field attains C_K(D) = sqrt(d/(d-1)) "
          f"= {sp.nsimplify(sp.sqrt(sp.Rational(d, d-1)))}",
          sp.simplify(ratio_D - sp.sqrt(sp.Rational(d, d - 1))) == 0)
    check(f"B8 d={d} gradient field kills the skew part (note Ex. 3.2 witness)",
          sp.simplify(box_integrate(frob(PROJ['skew'](Gg)), X)) == 0)

    # (ii) divergence-free field v = curl-type field built from phi.
    vd = [sp.diff(phi, X[1]), -sp.diff(phi, X[0])] + [sp.Integer(0)] * (d - 2)
    Gd = grad_of(vd, X)
    Ad = box_integrate(frob(Gd), X)
    Bd = box_integrate(sum(Gd[i, i] for i in range(d)) ** 2, X)
    check(f"B9 d={d} the built field is divergence-free", sp.simplify(Bd) == 0)
    for key in ("S", "DS"):
        ratio = sp.sqrt(Ad / box_integrate(frob(PROJ[key](Gd)), X))
        check(f"B10 d={d} divergence-free field attains C_K({key}) = sqrt(2)",
              sp.simplify(ratio - sp.sqrt(2)) == 0)
    check(f"B11 d={d} divergence-free field kills the spherical part "
          f"(note Ex. 3.2 witness)",
          sp.simplify(box_integrate(frob(PROJ['sph'](Gd)), X)) == 0)
    # B12: C_K(I) = 1 -- checked as an identity of the PROJECTED integral against the
    # unprojected one (not as the tautology sqrt(A/A) = 1).
    I_of_G = box_integrate(frob(PROJ["I"](Gd)), X)
    check(f"B12 d={d} C_K(I) = 1: int |I grad v|^2 equals int |grad v|^2 "
          f"(both {sp.nsimplify(Ad)})",
          sp.simplify(I_of_G - Ad) == 0 and sp.simplify(Ad) != 0)


# =============================================================================
# C. FOURIER SYMBOLS  (rem:ftGenericPi)
# =============================================================================
print("=" * 72)
print("C. FOURIER SYMBOLS (exact)")
print("=" * 72)

for d in (2, 3):
    k = sp.Matrix(d, 1, lambda i, _: sp.Symbol(f"k{i}", real=True))
    vv = sp.Matrix(d, 1, lambda i, _: sp.Symbol(f"w{i}", real=True))
    k2 = sp.expand(sum(k[i] ** 2 for i in range(d)))

    def T_of(P):
        """T[a,b] = P_{aibj} k_i k_j, i.e. <P(e_a (x) k), P(e_b (x) k)>."""
        cols = []
        for a in range(d):
            E = sp.zeros(d, d)
            for i in range(d):
                E[a, i] = k[i]
            cols.append(P(E))
        return sp.Matrix(d, d, lambda a, b: sp.expand(frob(cols[a], cols[b])))

    for key, ref in (
        ("I",  k2 * sp.eye(d)),
        ("S",  (k2 * sp.eye(d) + k * k.T) / 2),
        ("DS", (k2 * sp.eye(d) + (1 - sp.Rational(2, d)) * k * k.T) / 2),
        ("D",  k2 * sp.eye(d) - (sp.Rational(1, d)) * k * k.T),
    ):
        T = T_of(PROJ[key])
        check(f"C1 d={d} closed form T_{key}",
              sp.simplify(T - ref) == sp.zeros(d, d))
        # eq:ftGenericQuad: v^T T v = |Pi(v (x) k)|^2
        E = sp.Matrix(d, d, lambda a, i: vv[a] * k[i])
        quad = sp.expand((vv.T * T * vv)[0, 0] - frob(PROJ[key](E)))
        check(f"C2 d={d} eq:ftGenericQuad for {key}: v.T_Pi.v = |Pi(v (x) k)|^2",
              sp.simplify(quad) == 0)
        # C3/C4: the spectrum, DERIVED FROM THE COMPUTED MATRIX T (not hand-typed).
        # Every T here is isotropic-plus-rank-one, T = lam_perp |k|^2 I
        # + (lam_par - lam_perp) k k^T, so the longitudinal eigenvalue comes from the
        # Rayleigh quotient at v = k and the transverse one from the trace; the
        # reconstruction below is what makes the two genuinely checked.
        lam_par = sp.simplify(((k.T * T * k)[0, 0]) / k2 ** 2)
        lam_perp = sp.simplify((T.trace() - lam_par * k2) / ((d - 1) * k2))
        recon = sp.simplify(T - (lam_perp * k2 * sp.eye(d)
                                 + (lam_par - lam_perp) * k * k.T))
        check(f"C3a d={d} spectrum of T_{key} reconstructed from the computed matrix "
              f"(lam_perp={lam_perp}, lam_par={lam_par}, both x |k|^2)",
              recon == sp.zeros(d, d))
        lo, hi = min(lam_perp, lam_par), max(lam_perp, lam_par)
        check(f"C3 d={d} family bounds for {key}: 1/2 <= {lo} and {hi} <= 1 "
              f"(so 1/2|k|^2 I <= T_Pi <= |k|^2 I)",
              sp.Rational(1, 2) <= lo and hi <= 1)
        # C4 spectral radius factor of the viscous symbol 2 T_Pi.
        factor = sp.simplify(2 * hi)
        expected = (2 - sp.Rational(2, d)) if key == "DS" else 2
        check(f"C4 d={d} spectral-radius factor of 2 T_{key} is {factor} "
              f"(rem:ftGenericPi expects {expected})", sp.simplify(factor - expected) == 0)
        if key == "DS":
            # C5 the eq:ftViscEig eigenvalues, read off the SAME derived spectrum.
            check(f"C5 d={d} eq:ftViscEig: eig(2 T_DS)/|k|^2 = "
                  f"{{1 (x {d-1}), {2 - sp.Rational(2, d)}}}",
                  sp.simplify(2 * lam_perp - 1) == 0
                  and sp.simplify(2 * lam_par - (2 - sp.Rational(2, d))) == 0)

# C6 ellipticity constants c_Pi of eq:cPi, by exact minimization over the
# rank-one tensors v (x) k with |v| = |k| = 1.  Every quadratic form below
# depends on (v.k)^2 only, so the minimum is attained at an endpoint of [0, 1].
for d in (2, 3):
    c = sp.Symbol("c", nonnegative=True)      # c = (v.k)^2 in [0, 1]
    # |Pi(v (x) k)|^2 for |v| = |k| = 1, as a function of c:
    forms = {
        "I":    sp.Integer(1),
        "S":    (1 + c) / 2,
        "D":    1 - c / d,
        "DS":   (1 + c) / 2 - c / d,
        "C":    (1 - c) / 2 + c / d,
        "skew": (1 - c) / 2,
        "sph":  c / d,
    }
    # confirm the forms against the symbolic tensors
    k = sp.Matrix(d, 1, lambda i, _: sp.Symbol(f"kk{i}", real=True))
    vv = sp.Matrix(d, 1, lambda i, _: sp.Symbol(f"ww{i}", real=True))
    E = sp.Matrix(d, d, lambda a, i: vv[a] * k[i])
    kn = sum(k[i] ** 2 for i in range(d))
    vn = sum(vv[i] ** 2 for i in range(d))
    dot = sum(vv[i] * k[i] for i in range(d))
    for key, f in forms.items():
        raw = sp.expand(frob(PROJ[key](E)))
        # raw must equal |v|^2|k|^2 * f(c) with c = (v.k)^2/(|v|^2|k|^2);
        # check the homogeneous identity by clearing denominators.
        target = sp.expand(f.subs(c, dot ** 2 / (vn * kn)) * vn * kn)
        check(f"C6 d={d} {key:>4}: |Pi(v (x) k)|^2 as a function of (v.k)^2",
              sp.simplify(raw - target) == 0)
    claims = {"I": 1, "S": sp.sqrt(2) / 2, "D": sp.sqrt(1 - sp.Rational(1, d)),
              "DS": sp.sqrt(2) / 2, "C": sp.sqrt(sp.Rational(1, d)),
              "skew": 0, "sph": 0}
    for key, f in forms.items():
        vals = [sp.simplify(f.subs(c, 0)), sp.simplify(f.subs(c, 1))]
        cmin = sp.sqrt(min(vals))
        check(f"C6b d={d} {key:>4}: c_Pi = {cmin} (claimed {claims[key]})",
              sp.simplify(cmin - claims[key]) == 0)


# =============================================================================
# D. SOURCE LINT  (source-coupled; guards the printed statements)
# =============================================================================
print("=" * 72)
print("D. SOURCE LINT OF THE PRINTED STATEMENTS")
print("=" * 72)

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", "..", "theory", "paper"))


def read(fn):
    with open(os.path.join(PAPER, fn), encoding="utf-8") as f:
        return f.read()


def uncommented(txt):
    """Source with %-comment lines removed: a \\label named in a comment is prose."""
    return re.sub(r"(?m)^\s*%.*$", "", txt)


APP_C = read("asgs_convergence.tex")
APP_B = read("fourier_appendix.tex")
MAINS = {"article.tex": read("article.tex"),
         "article_v2.tex": read("article_v2.tex")}

# D1  H:projector and eq:PiKorn must be DEFINED in BOTH mains.  asgs_convergence
# and fourier_appendix are \input by both articles and cite these labels, so a
# definition present in only one main silently becomes an undefined reference in
# the other build.  This is the single fragility the generalization introduced.
n = 0
ok = True
for fn, txt in MAINS.items():
    has_lbl = len(re.findall(r"\\label\{H:projector\}", txt)) == 1
    has_eq = len(re.findall(r"\\label\{eq:PiKorn\}", txt)) == 1
    ok &= has_lbl and has_eq
    n += 1
check(f"D1 H:projector and eq:PiKorn defined exactly once in each main "
      f"({n} mains checked)", ok and n == 2)

# D2  The insertion point: H:projector must come AFTER H:data and BEFORE
# H:porosity in both mains, so that every existing \crefrange over the standing
# assumptions keeps bracketing it without edits.
n = 0
ok = True
for fn, txt in MAINS.items():
    i_data = txt.index(r"\label{H:data}")
    i_proj = txt.index(r"\label{H:projector}")
    i_por = txt.index(r"\label{H:porosity}")
    ok &= i_data < i_proj < i_por
    n += 1
check(f"D2 H:projector sits between H:data and H:porosity in both mains "
      f"({n} mains checked)", ok and n == 2)

# D3  Shared appendices must not cite a label that exists in only one main.
V2_ONLY = ("sec:CommonSetting", "app:osgs", "sec:StabilityOSGS",
           "eq:TripleNormOSGS", "H:design", "H:projection", "H:patch",
           "H:porositysmooth", "H:advectionsmooth", "eq:BosgsMain",
           "th:StabilityOSGS", "th:ConvergenceOSGS")
bad = []
nchk = 0
# \eqref must be in the alternation: App. C cites >100 displays with \eqref, and a
# blacklist that cannot see them is a blacklist with a hole.
CREF = re.compile(r"\\(?:eq|c|C)?ref[a-z]*\*?\{([^}]*)\}")
SHARED = (("asgs_convergence.tex", APP_C), ("fourier_appendix.tex", APP_B))


def cited_labels(txt):
    """Every label cited from `txt`, comment lines stripped."""
    body = re.sub(r"(?m)^\s*%.*$", "", txt)
    for m in CREF.finditer(body):
        for lbl in m.group(1).split(","):
            lbl = lbl.strip()
            if lbl:
                yield lbl


for fn, txt in SHARED:
    for lbl in cited_labels(txt):
        nchk += 1
        # the whole oa:* namespace lives in App. D, which only v2 inputs
        if lbl.startswith("oa:") or lbl in V2_ONLY:
            bad.append((fn, lbl))
check(f"D3 shared appendices cite no v2-only label: every cited label screened "
      f"({nchk} citations checked, none in the oa:* namespace or the v2-only list)",
      not bad and nchk > 0, str(bad))
# DISCRIMINATING negative: the rule must reject a synthetic bad citation.
check("D3' the D3 rule rejects a synthetic v2-only citation",
      any(l.startswith("oa:") or l in V2_ONLY
          for l in cited_labels(r"see \cref{lem:winv,oa:th:stability}")))
# D3'' POSITIVE resolvability.  D3 is a blacklist: it can only reject names someone
# thought to list.  The 2026-07-29 relocation moved nine labels OUT of App. C and into
# the main text, a move a blacklist cannot see at all.  So: every label a shared
# appendix cites but does NOT define itself must be defined in BOTH mains (or in the
# other shared appendices, which both mains also \input).  This count can go to zero,
# and then the rule fails instead of passing vacuously.
APP_A = read("elemental_matrices_appendix.tex")
SIBLING = {"elemental_matrices_appendix.tex": APP_A,
           "asgs_convergence.tex": APP_C,
           "fourier_appendix.tex": APP_B}
LABEL = re.compile(r"\\label\{([^}]*)\}")


def defined_labels(txt):
    """Labels DEFINED by `txt`.  Comment lines are stripped first: a \\label named inside a
    %-comment (the relocation note in App. C names \\label{sec:projectors}) defines nothing,
    and treating it as a definition would let a genuinely dangling citation to that name be
    written off below as 'internal, trivially fine'."""
    return set(LABEL.findall(uncommented(txt)))


unresolved = []
ncross = 0
for fn, txt in SHARED:
    own = defined_labels(txt)
    for lbl in cited_labels(txt):
        if lbl in own:
            continue                                  # internal, trivially fine
        ncross += 1
        homes = [m for m, t in MAINS.items() if lbl in defined_labels(t)]
        if len(homes) == 2:
            continue                                  # in both mains: safe
        if any(lbl in defined_labels(t) for s, t in SIBLING.items() if s != fn):
            continue                                  # in another shared appendix
        unresolved.append((fn, lbl, homes))
check(f"D3'' every cross-file label cited by a shared appendix is defined in BOTH mains "
      f"(or another shared appendix) ({ncross} cross-file citations resolved)",
      not unresolved and ncross > 0, str(unresolved[:12]))

# D4  eq:projconstants states the four sharp constants -- in EACH main, since the
# lemma now lives in the main text (duplicated verbatim in both articles).
# 2026-07-31: the identity member is now \IPi (the sans-serif FOURTH-order identity),
# not \mathbb{I}, which the mains keep for the second-order identity of
# sigma = sigma I and I_d.  Same four constants, same strictness -- only the glyph
# the source spells them with changed.
PATS_D4 = [r"C_\{\\mathrm\{K\}\}\(\\IPi\)\s*=\s*1",
           r"C_\{\\mathrm\{K\}\}\(\\SPi\)\s*=\s*\\sqrt2",
           r"C_\{\\mathrm\{K\}\}\(\\DPi\)\s*=\s*\\sqrt\{\\tfrac\{d\}\{d-1\}\}",
           r"C_\{\\mathrm\{K\}\}\(\\DSPi\)\s*=\s*\\sqrt2"]
nmain = 0
hits_total = 0
for fn, txt in MAINS.items():
    m = re.search(r"\\label\{eq:projconstants\}(.{0,600}?)\\end\{equation\}", txt, re.S)
    if m is None:
        continue
    nmain += 1
    hits_total += sum(1 for p in PATS_D4 if re.search(p, m.group(1)))
check(f"D4 eq:projconstants prints 1, sqrt2, sqrt(d/(d-1)), sqrt2 in each main "
      f"({nmain}/2 mains, {hits_total}/8 constants matched)",
      nmain == 2 and hits_total == 8)

# D5  eq:devsymidentity keeps its coefficients (1/2 and 1/2 - 1/d) and its tail,
# in EACH main.
nmain = 0
nok = 0
for fn, txt in MAINS.items():
    m = re.search(r"\\label\{eq:devsymidentity\}(.{0,400}?)\\end\{equation\}", txt, re.S)
    if m is None:
        continue
    nmain += 1
    body = re.sub(r"\s+", "", m.group(1))
    if (r"\tfrac12\lVert\nabla\bv\rVert^2" in body
            and r"\Bigl(\tfrac12-\tfrac1d\Bigr)\lVert\nabla\cdot\bv\rVert^2" in body
            and r"\ge\\tfrac12\lVert\nabla\bv\rVert^2" in body):
        nok += 1
check(f"D5 eq:devsymidentity keeps 1/2, (1/2 - 1/d) and the >= 1/2 tail "
      f"({nok}/{nmain} mains matched, 2 expected)", nmain == 2 and nok == 2)

# D6  The Fourier remark states the two family bounds and the [1,2] range.
ok = (r"\label{rem:ftGenericPi}" in APP_B
      and re.search(r"\\tfrac12\|\\boldsymbol\{k\}_0\|\^2\s*\\,\\mathbb\{I\}_d", APP_B)
      and r"2-\tfrac2d" in APP_B)
check("D6 rem:ftGenericPi states T_Pi >= 1/2|k|^2 I and the 2-2/d factor", ok)
ok2 = ("[1,2]" in MAINS["article.tex"]) and ("[1,2]" in MAINS["article_v2.tex"])
check("D7 both mains record that the dropped O(1) factor lies in [1,2]", ok2)

# D8  The definiteness lemma must invoke H:projector, and must NOT re-derive the
# dev-sym identity (which now lives once, in lem:projfamily).
m = re.search(r"\\begin\{lemma\}\[Definiteness of the working norm\]"
              r"(.*?)\\end\{proof\}", APP_C, re.S)
ok = (m is not None
      and r"\cref{H:data,H:projector,H:porosity,H:advection,H:spaces}" in m.group(1)
      and r"eq:PiKorn" in m.group(1) and r"\operatorname{sym}" not in m.group(1))
check("D8 lem:definiteness cites H:projector/eq:PiKorn (and H:advection, which bounds "
      "|a| and so keeps tau_1 positive) and re-derives nothing", ok)

# D10  The tau-design sentence must NOT claim the [1,2] window for a general admissible
# projector -- see the negative control E1 below, which exhibits an (A3)-admissible
# projector whose factor falls below 1.  It must attribute the window to the
# lem:projfamily family instead.
ok = True
n = 0
for fn, txt in MAINS.items():
    n += 1
    seg = re.search(r"the discarded factor still lies in \$\[1,2\]\$", txt)
    ok &= seg is not None
    # the 200 characters before the window claim must name the family, not "admissible"
    if seg:
        before = txt[max(0, seg.start() - 200):seg.start()]
        ok &= ("lem:projfamily" in before or "lem:projinstances" in before)
        ok &= "general admissible" not in before
check(f"D10 both mains scope the [1,2] window to the lem:projfamily family, not to a "
      f"general admissible projector ({n} mains checked)", ok and n == 2)

# D11  rem:ftGenericPi must likewise not conclude the two-sided bound "for every
# admissible projector", and must record the general lower bound 2 c_Pi^2.
ok = ("for every admissible projector" not in APP_B
      and "For every projector of that family" in APP_B
      and "2c_{\ViscProj}^2" in APP_B)
check("D11 rem:ftGenericPi scopes its two-sided bound and records the general "
      "2 c_Pi^2 lower bound", ok)

# D11' UNITS.  rem:ftGenericPi compares SPECTRAL RADII of the viscous symbol, which carry the
# factor alpha*nu/h^2 (the family bounds are written that way two sentences earlier).  The
# general lower bound must carry it too: written as "2 c^2 |k_0|^2" it is a bare tensor bound
# in a sentence about spectral radii, off by alpha*nu/h^2.  That slip lived in the source
# undetected because D11 only tests for the substring "2c_{\ViscProj}^2".
m = re.search(r"the lower one becomes \$([^$]*)\$", APP_B)
ok = m is not None and all(tok in m.group(1)
                           for tok in (r"2c_{\ViscProj}^2", r"\alpha\nu", "h^2"))
check("D11' the general lower bound in rem:ftGenericPi is a spectral radius, carrying "
      "alpha*nu/h^2 like the family bounds it is compared with", ok,
      f"found: {m.group(1) if m else '<no match>'}")

# D9  Every relocated label is DEFINED exactly once in EACH main and NOWHERE in the
# shared appendices (a second definition would be multiply-defined in both builds).
RELOCATED = ("sec:ViscousProjector", "V:orthogonality", "V:korn",
             "eq:PiOrthogonal", "eq:PiKorn", "eq:ProjExtractors", "eq:ProjFamily",
             "eq:DevSymDecomposition",
             "lem:projfamily", "eq:projmonotone", "eq:devsymidentity", "eq:crossibp",
             "lem:projinstances", "eq:projconstants")
bad9 = []
n9 = 0
for lbl in RELOCATED:
    for fn, txt in MAINS.items():
        n9 += 1
        if len(re.findall(r"\\label\{" + re.escape(lbl) + r"\}", uncommented(txt))) != 1:
            bad9.append(("not-once-in-" + fn, lbl))
    for fn, txt in SHARED:
        if re.search(r"\\label\{" + re.escape(lbl) + r"\}", uncommented(txt)):
            bad9.append(("still-defined-in-" + fn, lbl))
check(f"D9 the relocated projector labels are defined exactly once in each main and "
      f"nowhere in the shared appendices ({n9} label/main pairs checked)",
      not bad9 and n9 == 2 * len(RELOCATED), str(bad9))

# D9' eq:devsymidentity is still cited from outside its new home, so the relocation did
# not orphan it.  Count only LIVE citations (comment lines stripped) made from OUTSIDE the
# sec:ViscousProjector block -- the lemma's own proof cites it, and counting those, or
# counting the relocation comment in App. C, makes the rule unfailable.  As of the
# relocation the genuine outside citers are the Cocquet/Korn footnote in each main.
D9_CITERS = ("asgs_convergence.tex", "fourier_appendix.tex",
             "osgs_appendix.tex", "osgs_appendix_commented.tex")
outside = 0
for f in D9_CITERS:
    outside += len(re.findall(r"eq:devsymidentity", uncommented(read(f))))
for fn, txt in MAINS.items():
    body = uncommented(txt)
    i = body.index(r"\subsection{The viscous projector}")
    j = body.index(r"\subsection{Abstract reformulation of the problem (Strong form)}")
    outside += len(re.findall(r"eq:devsymidentity", body[:i] + body[j:]))
check(f"D9' eq:devsymidentity is still cited LIVE from outside sec:ViscousProjector "
      f"({outside} such citations; the relocation must not orphan it)", outside >= 2)

# D12  The sec:ViscousProjector block is DUPLICATED in the two mains -- the price of
# defining, in the main text, labels that the shared appendices (App. B, App. C) cite.
# Nothing else diffs the two copies, so they drift silently.  This rule requires them to
# agree line for line EXCEPT at the two sentences flagged "v1/v2 DIVERGENCE" in the source:
#   (1) v2 names the second working norm and cites oa:lem:definiteness, a v2-only label
#       that must NEVER appear in article.tex;
#   (2) "the stabilized method" (v1, ASGS only) vs "...methods" (v2, ASGS and OSGS).
BLOCK_OPEN = r"\subsection{The viscous projector}"
BLOCK_CLOSE = r"\subsection{Abstract reformulation of the problem (Strong form)}"


def projector_block(txt):
    return txt[txt.index(BLOCK_OPEN):txt.index(BLOCK_CLOSE)].split("\n")


b1 = projector_block(MAINS["article.tex"])
b2 = projector_block(MAINS["article_v2.tex"])
# The two sanctioned v1 -> v2 rewrites, applied verbatim.  After them the blocks must be
# EXACTLY equal: a count-and-label check is too weak, because a stray edit landing ON one
# of the two divergent lines leaves the count at 2 and slips through.
V1_TO_V2 = (
    ("runs the stabilized method.", "runs the stabilized methods."),
)
norm1 = "\n".join(b1)
n_applied = 0
for old, new in V1_TO_V2:
    if old in norm1:
        norm1 = norm1.replace(old, new, 1)
        n_applied += 1
equal = norm1 == "\n".join(b2)
first_bad = next((i for i, (x, y) in enumerate(zip(norm1.split("\n"), b2))
                  if x != y), None)
check(f"D12 the two mains' sec:ViscousProjector copies are identical after the "
      f"{len(V1_TO_V2)} sanctioned v1->v2 rewrites ({len(b1)} lines compared, "
      f"{n_applied}/{len(V1_TO_V2)} rewrites applicable)",
      equal and n_applied == len(V1_TO_V2),
      f"first differing line offset {first_bad}: "
      f"v1(normalized)={norm1.split(chr(10))[first_bad][:110]!r} vs "
      f"v2={b2[first_bad][:110]!r}" if first_bad is not None
      else f"line counts {len(b1)} vs {len(b2)}")
# The v1 copy must CITE no v2-only label, whatever else changes.  Comment lines are
# stripped: the "v1/v2 DIVERGENCE" marker deliberately NAMES oa:lem:definiteness in prose
# to warn against copying it, and that warning must not read as the violation it prevents.
v1_live = uncommented("\n".join(b1))
check("D12' the v1 copy of the block cites no oa:* (App. D, v2-only) label",
      not [l for l in cited_labels(v1_live) if l.startswith("oa:")],
      str([l for l in cited_labels(v1_live) if l.startswith("oa:")]))

# =============================================================================
# E. NEGATIVE CONTROL: (A3) ALONE DOES NOT GIVE THE [1,2] WINDOW
#
# The 2026-07-29 audit caught the paper claiming the dropped O(1) viscous factor lies
# in [1,2] "for a general admissible projector".  That is FALSE: (A3) asks only for an
# orthogonal projection with SOME Korn constant, i.e. c_Pi > 0 (eq:cPi, now in App. B;
# the sharp characterization is Thm. 2.2 of theory/viscous_projector_note/), and the
# lower bound T_Pi >= 1/2|k|^2 I needs the strictly stronger lem:projfamily hypothesis.
# The family below is the witness, and it is kept here so the claim can never be
# re-broadened without the suite failing.
#
# In d = 2, in the orthonormal tensor basis {I/sqrt2, E1, E2, W} with
# E1 = diag(1,-1)/sqrt2, E2 = [[0,1],[1,0]]/sqrt2, W = [[0,1],[-1,0]]/sqrt2, let
# Pi_theta project orthogonally onto span{cos(th) I/sqrt2 + sin(th) E1,
#                                        cos(th) W       - sin(th) E2}.
# Then |Pi_theta(v (x) k)|^2 = (1 + sin(2 th) cos(2 b))/2 for |v| = |k| = 1, b the
# polar angle of k -- INDEPENDENT of v.  Hence c_Pi^2 = (1 - |sin 2 th|)/2 > 0 for
# |th| < pi/4 (so (A3) holds), while the factor 2 lambda_max = 1 + sin(2 th) cos(2 b)
# dips below 1 and tends to 0 as th -> -pi/4.
# =============================================================================
print("=" * 72)
print("E. NEGATIVE CONTROL: (A3) DOES NOT IMPLY THE [1,2] WINDOW")
print("=" * 72)

th, bb = sp.symbols("theta beta", real=True)
r2 = sp.sqrt(2)
Bas = {
    "I2": sp.Matrix([[1, 0], [0, 1]]) / r2,
    "E1": sp.Matrix([[1, 0], [0, -1]]) / r2,
    "E2": sp.Matrix([[0, 1], [1, 0]]) / r2,
    "W":  sp.Matrix([[0, 1], [-1, 0]]) / r2,
}
u1 = sp.cos(th) * Bas["I2"] + sp.sin(th) * Bas["E1"]
u2 = sp.cos(th) * Bas["W"] - sp.sin(th) * Bas["E2"]
check("E0 the two spanning tensors are orthonormal",
      sp.simplify(frob(u1) - 1) == 0 and sp.simplify(frob(u2) - 1) == 0
      and sp.simplify(frob(u1, u2)) == 0)


def P_theta(T):
    return frob(T, u1) * u1 + frob(T, u2) * u2


# E1: Pi_theta is a genuine constant-coefficient orthogonal projection.
Tg = generic(2, "z")
Ug = generic(2, "y")
check("E1 Pi_theta is idempotent and Frobenius self-adjoint (an orthogonal projection)",
      sp.simplify(P_theta(P_theta(Tg)) - P_theta(Tg)) == sp.zeros(2, 2)
      and sp.simplify(frob(P_theta(Tg), Ug) - frob(Tg, P_theta(Ug))) == 0)

# E2: the closed form of |Pi_theta(v (x) k)|^2 on the unit sphere.
aa = sp.Symbol("a_ang", real=True)
vv2 = sp.Matrix([sp.cos(aa), sp.sin(aa)])
kk2 = sp.Matrix([sp.cos(bb), sp.sin(bb)])
E = sp.Matrix(2, 2, lambda i, j: vv2[i] * kk2[j])
q = sp.simplify(sp.trigsimp(frob(P_theta(E))))
target = (1 + sp.sin(2 * th) * sp.cos(2 * bb)) / 2
check("E2 |Pi_theta(v (x) k)|^2 = (1 + sin 2th cos 2b)/2, independent of v",
      sp.simplify(sp.trigsimp(q - target)) == 0)

# E3: at theta = -0.3 the projector satisfies (A3) but the factor drops below 1.
th0 = sp.Rational(-3, 10)
cmin2 = sp.simplify((1 - sp.Abs(sp.sin(2 * th0))) / 2)          # c_Pi^2
factor_min = sp.simplify(1 + sp.sin(2 * th0) * sp.cos(2 * sp.Integer(0)))  # k along e1
check(f"E3 theta=-0.3: c_Pi^2 = {sp.N(cmin2, 6)} > 0, so (A3) holds "
      f"(C_K = {sp.N(1/sp.sqrt(cmin2), 6)})", sp.N(cmin2) > 0)
check(f"E4 theta=-0.3: the viscous factor 2*lambda_max = {sp.N(factor_min, 6)} < 1, "
      f"so the [1,2] window FAILS for an (A3)-admissible projector",
      sp.N(factor_min) < 1)
# E5: and it is not a boundary artefact -- the factor tends to 0 as theta -> -pi/4.
th1 = sp.Rational(-3, 4)
check(f"E5 theta=-0.75: factor = {sp.N(1 + sp.sin(2*th1), 6)}, "
      f"c_Pi^2 = {sp.N((1 - sp.Abs(sp.sin(2*th1)))/2, 6)} (still (A3)-admissible)",
      sp.N(1 + sp.sin(2 * th1)) < sp.Rational(1, 100)
      and sp.N((1 - sp.Abs(sp.sin(2 * th1))) / 2) > 0)
# E6: consistency -- Pi_theta does NOT fix the deviatoric-symmetric tensors (it is
# outside lem:projfamily's hypothesis), which is exactly why the bound may fail.
Tds2 = dev(sym(generic(2, "q")))
check("E6 Pi_theta does not fix the deviatoric-symmetric tensors "
      "(outside lem:projfamily, as it must be)",
      sp.simplify(P_theta(Tds2).subs(th, th0) - Tds2) != sp.zeros(2, 2))

# =============================================================================
print("=" * 72)
n_fail = sum(1 for tag, _ in results if tag == "FAIL")
print(f"SUMMARY: {len(results) - n_fail}/{len(results)}")
print("=" * 72)
raise SystemExit(1 if n_fail else 0)
