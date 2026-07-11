# Exact elementwise thresholds with the paper's projector Pi = dev o sym:
#   Pi grad u = sym grad u - (div u / d) I.
# Quantities (velocity space [P2(K)]^d, h = diam K):
#   Call2_dev  := h^2 max ||div(Pi grad u)||^2 / ||Pi grad u||^2      (all u)
#   Cdf2_dev   := same restricted to div-free u  (should equal the sym values)
#   c1*_dev(K) := max 4 h^2 ||div(Pi grad u)||^2 /
#                       (2||Pi grad u||^2 + ||div u||^2)
import itertools, sympy as sp, mpmath as mp
exec(open('verify2.py').read().split("# exact:")[0])  # build utilities: mp50_max

def forms(V):
    d = len(V) - 1
    coords = [sp.symbols('x y z')[i] for i in range(d)]
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
            for e in itertools.product(range(3), repeat=d) if sum(e) <= 2]
    n = len(mono); N = d*n
    cs = sp.symbols('c0:%d' % N)
    u = [sum(cs[l*n+i]*mono[i] for i in range(n)) for l in range(d)]
    divu = sum(sp.diff(u[i], coords[i]) for i in range(d))
    Edev = [[sp.Rational(1,2)*(sp.diff(u[i],coords[j])+sp.diff(u[j],coords[i]))
             - (sp.Rational(1,d)*divu if i == j else 0)
             for j in range(d)] for i in range(d)]
    divEdev = [sum(sp.diff(Edev[i][j],coords[j]) for j in range(d))
               for i in range(d)]
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
    GdE = gram(divEdev)
    GEd = gram([Edev[i][j] for i in range(d) for j in range(d)])
    Gdu = gram([divu])
    # div-free constraint rows
    P = sp.Poly(sp.expand(divu), *coords)
    C = sp.Matrix([[sp.diff(cf, c) for c in cs] for cf in P.coeffs()])
    Z = sp.Matrix.hstack(*C.nullspace())
    h2 = max(sum((a-b)**2 for a,b in zip(P1,P2))
             for P1,P2 in itertools.combinations(V,2))
    return GdE, GEd, Gdu, Z, h2

ELEMS = {
 '2D right':  [(0,0),(1,0),(0,1)],
 '2D equil':  [(0,0),(1,0),(sp.Rational(1,2),sp.sqrt(3)/2)],
 '3D right':  [(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
 '3D Kuhn':   [(0,0,0),(1,0,0),(1,1,0),(1,1,1)],
 '3D regular':[(0,0,0),(1,0,0),(sp.Rational(1,2),sp.sqrt(3)/2,0),
               (sp.Rational(1,2),sp.sqrt(3)/6,sp.sqrt(6)/3)],
}
res = {}
for nm, V in ELEMS.items():
    GdE, GEd, Gdu, Z, h2 = forms(V)
    Call2 = mp50_max(h2*GdE, GEd, sp.Integer(1))
    Cdf2  = mp50_max(sp.simplify(Z.T*(h2*GdE)*Z), sp.simplify(Z.T*GEd*Z),
                     sp.Integer(1))
    c1s   = mp50_max(4*h2*GdE, 2*GEd + Gdu, sp.Integer(1))
    res[nm] = (Call2, Cdf2, c1s)
    print(f'{nm}:  Call2_dev = {mp.nstr(Call2,20)}   Cdf2_dev = '
          f'{mp.nstr(Cdf2,20)}   c1*_dev = {mp.nstr(c1s,20)}')
print()
print('ratios (3D/2D):')
print('  df   Kuhn/right :', mp.nstr(res['3D Kuhn'][1]/res['2D right'][1], 10))
print('  df   reg/equil  :', mp.nstr(res['3D regular'][1]/res['2D equil'][1], 10))
print('  c1*  Kuhn/right :', mp.nstr(res['3D Kuhn'][2]/res['2D right'][2], 10))
print('  c1*  right/right:', mp.nstr(res['3D right'][2]/res['2D right'][2], 10))
print('  c1*  reg/equil  :', mp.nstr(res['3D regular'][2]/res['2D equil'][2], 10))
print('  all  Kuhn/right :', mp.nstr(res['3D Kuhn'][0]/res['2D right'][0], 10))
