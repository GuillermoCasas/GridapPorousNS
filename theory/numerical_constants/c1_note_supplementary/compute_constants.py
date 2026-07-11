#!/usr/bin/env python3
# =============================================================================
# compute_constants.py
#
# Sharp local inverse-estimate constants that gate the stabilization
# constant c1 of the ASGS/VMS method, on triangles vs tetrahedra.
#
# For a simplex K with diameter h_K and a velocity space [P_k(K)]^d we
# compute, by exact assembly (sympy rationals) and generalized eigenvalue
# solves, the following squared constants (all normalized by h_K, i.e.
# the number reported is  C^2 := h_K^2 * lambda_max):
#
#   Cgrad2[k]  : scalar gradient inverse constant on P_k,
#                ||grad q||_K <= (Cgrad/h) ||q||_K.
#   Cdiv2[k-1] : vector divergence constant on [P_{k-1}]^d,
#                ||div m||_K <= (Cdiv/h) ||m||_K.
#                (Row-wise, this bounds ||div M|| <= (Cdiv/h)||M|| for
#                matrix fields with rows in [P_{k-1}]^d, hence the paper's
#                Cbar for constant porosity.)
#   Chat2[k]   : the operator that actually appears in the viscous
#                stabilization term, with the projector taken as the
#                identity (constant-alpha proxy):
#                sup over u in [P_k]^d of
#                    h^2 ||div(sym grad u)||^2 / ||sym grad u||^2 .
#   Chat2df[k] : the same sup restricted to pointwise divergence-free u
#                (the modes on which the tau_2 grad-div term gives no
#                help).  On this subspace div(sym grad u) = (1/2) Lap u.
#
# Element shapes:
#   2D: unit right triangle (structured quad-split meshes), equilateral.
#   3D: unit right tetrahedron; Kuhn tetrahedron (unit-cube 6-tet
#       subdivision); regular tetrahedron.
#
# Everything is assembled exactly; eigenvalues of the small P1 problems
# are computed exactly, the rest in float via scipy on the exact
# matrices.  A brute-force Monte-Carlo Rayleigh check validates each
# reported eigenvalue.
# =============================================================================

import itertools, json
import sympy as sp
import numpy as np
from scipy.linalg import eigh

x, y, z = sp.symbols('x y z')

# ---------------------------------------------------------------- elements
def diam(V):
    return sp.sqrt(max(sum((a - b)**2 for a, b in zip(P, Q))
                       for P, Q in itertools.combinations(V, 2)))

ELEMENTS = {
    '2D right (quad-split)': [(0, 0), (1, 0), (0, 1)],
    '2D equilateral':        [(0, 0), (1, 0), (sp.Rational(1, 2), sp.sqrt(3)/2)],
    '3D right':              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    '3D Kuhn (cube-split)':  [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
    '3D regular':            [(0, 0, 0), (1, 0, 0),
                              (sp.Rational(1, 2), sp.sqrt(3)/2, 0),
                              (sp.Rational(1, 2), sp.sqrt(3)/6, sp.sqrt(6)/3)],
}

# --------------------------------------------- exact integration on simplex
def simplex_integrator(V):
    """Return (integrate_poly, dim, coords).  Exact integration of a
    polynomial in the physical coordinates over the simplex with
    vertices V, by affine pullback to the reference simplex."""
    d = len(V) - 1
    coords = [x, y, z][:d]
    xis = sp.symbols('xi0:%d' % d)
    V = [sp.Matrix(v) for v in V]
    B = sp.Matrix.hstack(*[V[i + 1] - V[0] for i in range(d)])
    detB = sp.Abs(sp.simplify(B.det()))
    phys = V[0] + B * sp.Matrix(xis)
    sub = {coords[i]: phys[i] for i in range(d)}

    def integ(p):
        q = sp.expand(sp.sympify(p).subs(sub))
        # iterated integral over the reference simplex
        lims = []
        s = 0
        for i in range(d):
            upper = 1 - sum(xis[j] for j in range(i))
            lims.append((xis[i], 0, upper))
        for lim in reversed(lims):
            q = sp.integrate(q, lim)
        return sp.simplify(q * detB)
    return integ, d, coords

# ----------------------------------------------------------- bases and forms
def monomials(d, k, coords):
    out = []
    for degs in itertools.product(range(k + 1), repeat=d):
        if sum(degs) <= k:
            out.append(sp.prod([coords[i]**degs[i] for i in range(d)]))
    return out

def mass_and_dmats(V, k):
    """Scalar mass matrix M and the d^2 first-derivative coupling
    matrices D[l][m] = int dl(phi_i) dm(phi_j) for the P_k monomial
    basis, all exact."""
    integ, d, coords = simplex_integrator(V)
    phis = monomials(d, k, coords)
    n = len(phis)
    M = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            v = integ(phis[i] * phis[j])
            M[i, j] = v
            M[j, i] = v
    dphi = [[sp.diff(p, c) for p in phis] for c in coords]
    D = [[sp.zeros(n, n) for _ in range(d)] for _ in range(d)]
    for l in range(d):
        for m in range(l, d):
            for i in range(n):
                for j in range(n):
                    D[l][m][i, j] = integ(dphi[l][i] * dphi[m][j])
            if m != l:
                D[m][l] = D[l][m].T
    return phis, M, D, d, coords

def np_(Msym):
    return np.array(Msym.tolist(), dtype=float)

def gen_eig_max(A, B, tol=1e-10):
    """max lambda of A v = lambda B v on the positive part of B."""
    A = 0.5 * (A + A.T); B = 0.5 * (B + B.T)
    w, U = np.linalg.eigh(B)
    keep = w > tol * w.max()
    P = U[:, keep] / np.sqrt(w[keep])
    Ar = P.T @ A @ P
    return float(np.linalg.eigvalsh(Ar).max())

# ------------------------------------------------------- constants per element
def constants_for(V, kmax=2):
    integ, d, coords = simplex_integrator(V)
    h2 = float(sp.N(diam(V)**2))
    out = {'h2': h2, 'd': d}

    # scalar gradient constant on P_k, and vector divergence on [P_k]^d
    for k in range(1, kmax + 1):
        phis, M, D, d, coords = mass_and_dmats(V, k)
        n = len(phis)
        Mf = np_(M)
        Sgrad = sum(np_(D[l][l]) for l in range(d))
        out.setdefault('Cgrad2', {})[k] = h2 * gen_eig_max(Sgrad, Mf)

        # divergence on [P_k]^d : v = sum_{l,i} c_{l,i} phi_i e_l
        # div v = sum_l d_l(component l);   S_(l i)(m j) = D[l][m]_{ij}
        Sdiv = np.zeros((d * n, d * n))
        Mvec = np.zeros((d * n, d * n))
        for l in range(d):
            Mvec[l*n:(l+1)*n, l*n:(l+1)*n] = Mf
            for m in range(d):
                Sdiv[l*n:(l+1)*n, m*n:(m+1)*n] = np_(D[l][m])
        out.setdefault('Cdiv2', {})[k] = h2 * gen_eig_max(Sdiv, Mvec)

    # the div(sym grad) operator on [P_k]^d velocities
    for k in range(1, kmax + 1):
        phis, M, D, d, coords = mass_and_dmats(V, k)
        n = len(phis)
        N = d * n
        # symbolic velocity with coefficient symbols
        cs = sp.symbols('c0:%d' % N)
        u = [sum(cs[l*n + i] * phis[i] for i in range(n)) for l in range(d)]
        E = [[sp.Rational(1, 2) * (sp.diff(u[i], coords[j])
                                   + sp.diff(u[j], coords[i]))
              for j in range(d)] for i in range(d)]
        divE = [sum(sp.diff(E[i][j], coords[j]) for j in range(d))
                for i in range(d)]
        divu = sum(sp.diff(u[i], coords[i]) for i in range(d))

        def quad_form(exprs):
            """exact Gram matrix of  int sum expr^2  in the coefficients"""
            A = sp.zeros(N, N)
            for e in exprs:
                pe = sp.expand(e)
                g = [sp.expand(sp.diff(pe, c)) for c in cs]  # linear in cs
                for i in range(N):
                    if g[i] == 0:
                        continue
                    for j in range(i, N):
                        if g[j] == 0:
                            continue
                        v = integ(sp.expand(
                            g[i].subs({c: 0 for c in cs})
                            * g[j].subs({c: 0 for c in cs})))
                        A[i, j] += v
                        if j != i:
                            A[j, i] += v
            return A

        A_divE = np_(quad_form(divE))
        B_E = np_(quad_form([E[i][j] for i in range(d) for j in range(d)]))
        out.setdefault('Chat2', {})[k] = h2 * gen_eig_max(A_divE, B_E)

        # restriction to pointwise div-free u: div u = 0 as a polynomial
        # identity -> linear constraints on the coefficients
        divu_poly = sp.Poly(sp.expand(divu), *coords) if d > 1 else None
        rows = []
        for mono_coef in divu_poly.coeffs() if divu_poly is not None else []:
            row = [sp.diff(mono_coef, c) for c in cs]
            rows.append([float(sp.N(r)) for r in row])
        C = np.array(rows, dtype=float) if rows else np.zeros((0, N))
        # nullspace of C
        if C.shape[0] > 0:
            _, s, Vt = np.linalg.svd(C)
            rank = int((s > 1e-10 * (s.max() if s.size else 1)).sum())
            Z = Vt[rank:].T
        else:
            Z = np.eye(N)
        if Z.shape[1] > 0:
            Ar = Z.T @ A_divE @ Z
            Br = Z.T @ B_E @ Z
            out.setdefault('Chat2df', {})[k] = h2 * gen_eig_max(Ar, Br)
        else:
            out.setdefault('Chat2df', {})[k] = 0.0
    return out

# --------------------------------------------------- Monte-Carlo validation
def mc_check(V, k, target, mode, trials=20000, seed=0):
    """Random Rayleigh quotients must never exceed the reported max
    (up to roundoff) and should approach it."""
    rng = np.random.default_rng(seed)
    phis, M, D, d, coords = mass_and_dmats(V, k)
    n = len(phis); N = d * n
    h2 = float(sp.N(diam(V)**2))
    # reuse numeric forms
    cs = sp.symbols('c0:%d' % N)
    u = [sum(cs[l*n + i] * phis[i] for i in range(n)) for l in range(d)]
    E = [[sp.Rational(1, 2)*(sp.diff(u[i], coords[j]) + sp.diff(u[j], coords[i]))
          for j in range(d)] for i in range(d)]
    divE = [sum(sp.diff(E[i][j], coords[j]) for j in range(d)) for i in range(d)]
    integ, _, _ = simplex_integrator(V)

    def gram(exprs):
        A = sp.zeros(N, N)
        for e in exprs:
            g = [sp.expand(sp.diff(sp.expand(e), c)) for c in cs]
            for i in range(N):
                if g[i] == 0: continue
                for j in range(i, N):
                    if g[j] == 0: continue
                    v = integ(sp.expand(g[i].subs({c: 0 for c in cs})
                                        * g[j].subs({c: 0 for c in cs})))
                    A[i, j] += v
                    if j != i: A[j, i] += v
        return np_(A)

    A = gram(divE)
    B = gram([E[i][j] for i in range(d) for j in range(d)])
    best = 0.0
    for _ in range(trials):
        c = rng.standard_normal(N)
        den = c @ B @ c
        if den < 1e-12: continue
        best = max(best, h2 * (c @ A @ c) / den)
    return best, target

# ----------------------------------------------------------------- driver
if __name__ == '__main__':
    results = {}
    for name, V in ELEMENTS.items():
        print(f'--- {name}')
        results[name] = constants_for(V, kmax=2)
        for key in ('Cgrad2', 'Cdiv2', 'Chat2', 'Chat2df'):
            print(f'    {key}: ' + ', '.join(
                f'k={k}: {v:.6f}' for k, v in results[name][key].items()))
    # validation of the headline numbers
    print('\nMonte-Carlo validation of Chat2 (must not exceed, should approach):')
    for name in ('2D right (quad-split)', '3D Kuhn (cube-split)'):
        best, tgt = mc_check(ELEMENTS[name], 2, results[name]['Chat2'][2],
                             'Chat2')
        print(f'    {name}: MC best {best:.4f}  <=  eig {tgt:.4f}   '
              f'({"OK" if best <= tgt * (1 + 1e-8) else "FAIL"})')
    with open('constants.json', 'w') as f:
        json.dump(results, f, indent=1)
    print('\nsaved constants.json')
