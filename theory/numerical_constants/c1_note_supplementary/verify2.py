import itertools
import sympy as sp
import numpy as np
import mpmath as mp

x, y, z = sp.symbols('x y z')

def build(V, k, dfree):
    d = len(V) - 1
    coords = [x, y, z][:d]
    xis = sp.symbols('xi0:%d' % d)
    Vm = [sp.Matrix(v) for v in V]
    B = sp.Matrix.hstack(*[Vm[i+1] - Vm[0] for i in range(d)])
    detB = sp.Abs(sp.simplify(B.det()))
    phys = Vm[0] + B * sp.Matrix(xis)
    sub = {coords[i]: phys[i] for i in range(d)}
    def integ(p):
        q = sp.expand(sp.sympify(p).subs(sub))
        for i in reversed(range(d)):
            q = sp.integrate(q, (xis[i], 0, 1 - sum(xis[j] for j in range(i))))
        return sp.simplify(q * detB)
    mono = [sp.prod([coords[i]**e[i] for i in range(d)])
            for e in itertools.product(range(k+1), repeat=d) if sum(e) <= k]
    n = len(mono); N = d*n
    cs = sp.symbols('c0:%d' % N)
    u = [sum(cs[l*n+i]*mono[i] for i in range(n)) for l in range(d)]
    E = [[sp.Rational(1,2)*(sp.diff(u[i],coords[j])+sp.diff(u[j],coords[i]))
          for j in range(d)] for i in range(d)]
    divE = [sum(sp.diff(E[i][j],coords[j]) for j in range(d)) for i in range(d)]
    def gram(exprs):
        A = sp.zeros(N,N)
        for e in exprs:
            g = [sp.diff(sp.expand(e), c) for c in cs]
            for i in range(N):
                if g[i]==0: continue
                for j in range(i,N):
                    if g[j]==0: continue
                    v = integ(g[i]*g[j]); A[i,j]+=v
                    if j!=i: A[j,i]+=v
        return A
    A = gram(divE)
    Bm = gram([E[i][j] for i in range(d) for j in range(d)])
    Z = sp.eye(N)
    if dfree:
        divu = sp.expand(sum(sp.diff(u[i],coords[i]) for i in range(d)))
        P = sp.Poly(divu, *coords)
        C = sp.Matrix([[sp.diff(cf, c) for c in cs] for cf in P.coeffs()])
        Z = sp.Matrix.hstack(*C.nullspace())
    h2 = max(sum((a-b)**2 for a,b in zip(P1,P2))
             for P1,P2 in itertools.combinations(V,2))
    return sp.simplify(Z.T*A*Z), sp.simplify(Z.T*Bm*Z), h2

def exact_max(Ar, Br, h2):
    W = sp.Matrix.hstack(*Br.columnspace())
    A2 = sp.simplify(W.T*Ar*W); B2 = sp.simplify(W.T*Br*W)
    lam = sp.symbols('lam')
    p = sp.Poly((A2 - lam*B2).det(), lam)
    return sp.simplify(h2*sp.Max(*[r for r in sp.roots(p)]))

def mp50_max(Ar, Br, h2):
    mp.mp.dps = 50
    A = mp.matrix([[mp.mpf(sp.N(Ar[i,j], 60)) for j in range(Ar.cols)]
                   for i in range(Ar.rows)])
    B = mp.matrix([[mp.mpf(sp.N(Br[i,j], 60)) for j in range(Br.cols)]
                   for i in range(Br.rows)])
    ew, ev = mp.eigsy(B)
    keep = [i for i in range(len(ew)) if ew[i] > mp.mpf('1e-40')*max(ew)]
    P = mp.matrix(B.rows, len(keep))
    for jj, i in enumerate(keep):
        for r in range(B.rows):
            P[r, jj] = ev[r, i] / mp.sqrt(ew[i])
    Ar2 = P.T*A*P
    ew2 = mp.eigsy(Ar2, eigvals_only=True)
    return mp.mpf(sp.N(h2, 60)) * max(ew2)

# exact: 2D right, div-free and all
for df, name in [(True, 'div-free'), (False, 'all')]:
    Ar, Br, h2 = build([(0,0),(1,0),(0,1)], 2, df)
    print('2D right P2', name, ' exact h^2 lam_max =', exact_max(Ar, Br, h2))

# 50-digit: 3D right, Kuhn (div-free)
for V, nm in [([(0,0,0),(1,0,0),(0,1,0),(0,0,1)], '3D right'),
              ([(0,0,0),(1,0,0),(1,1,0),(1,1,1)], '3D Kuhn')]:
    Ar, Br, h2 = build(V, 2, True)
    v = mp50_max(Ar, Br, h2)
    print(nm, 'P2 div-free  50-digit:', mp.nstr(v, 35))
print('reference 440/3           =', mp.nstr(mp.mpf(440)/3, 35))
print('reference 200 + 10*sqrt2  =', mp.nstr(200 + 10*mp.sqrt(2), 35))
