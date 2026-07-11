# Exact elementwise coercivity threshold c1*(K) for a=0, sigma=0, alpha
# const, Pi = Id on K, velocity space [P2(K)]^d:
#   Q(u)/(alpha nu) = 2||E||^2 - (4 h^2/c1)||div E||^2 + ||div u||^2,
#   E = sym grad u.
#   Q >= 0 for all u  <=>  c1 >= c1*(K) := max_u 4 h^2 ||div E||^2
#                                          / (2||E||^2 + ||div u||^2).
import itertools, sympy as sp, mpmath as mp
exec(open('verify2.py').read().split("# exact:")[0])  # reuse build utilities

def threshold(V):
    d = len(V) - 1
    coords = [sp.symbols('x y z')[i] for i in range(d)]
    # rebuild with the extra div-u Gram
    import sympy as sp2
    x, y, z = sp.symbols('x y z')
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
    E = [[sp.Rational(1,2)*(sp.diff(u[i],coords[j])+sp.diff(u[j],coords[i]))
          for j in range(d)] for i in range(d)]
    divE = [sum(sp.diff(E[i][j],coords[j]) for j in range(d)) for i in range(d)]
    divu = sum(sp.diff(u[i],coords[i]) for i in range(d))
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
    GdivE = gram(divE)
    GE    = gram([E[i][j] for i in range(d) for j in range(d)])
    Gdivu = gram([divu])
    h2 = max(sum((a-b)**2 for a,b in zip(P1,P2))
             for P1,P2 in itertools.combinations(V,2))
    A = 4*h2*GdivE
    Bden = 2*GE + Gdivu
    return mp50_max(sp.Matrix(A)/1, sp.Matrix(Bden)/1, sp.Integer(1))

ELEMS = {
 '2D right':  [(0,0),(1,0),(0,1)],
 '2D equil':  [(0,0),(1,0),(sp.Rational(1,2),sp.sqrt(3)/2)],
 '3D right':  [(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
 '3D Kuhn':   [(0,0,0),(1,0,0),(1,1,0),(1,1,1)],
 '3D regular':[(0,0,0),(1,0,0),(sp.Rational(1,2),sp.sqrt(3)/2,0),
               (sp.Rational(1,2),sp.sqrt(3)/6,sp.sqrt(6)/3)],
}
vals = {}
for nm, V in ELEMS.items():
    v = threshold(V)
    vals[nm] = v
    print(f'{nm}:  c1*(K) = {mp.nstr(v, 25)}')
print()
print('candidate closed forms:')
for nm, guess in [('2D right','96'), ('2D equil','48'),
                  ('3D right','880/3'), ('3D Kuhn','400+20*sqrt(2)'),
                  ('3D regular','400/3')]:
    g = mp.mpf(sp.N(sp.sympify(guess), 40))
    print(f'  {nm}: {mp.nstr(vals[nm],20)} vs {guess} = {mp.nstr(g,20)}  '
          f'{"MATCH" if abs(vals[nm]-g) < 1e-15*max(1,abs(g)) else "no"}')
print()
print('ratios c1*(3D)/c1*(2D):')
print('  Kuhn / 2D right   =', mp.nstr(vals['3D Kuhn']/vals['2D right'], 10))
print('  right / 2D right  =', mp.nstr(vals['3D right']/vals['2D right'], 10))
print('  regular / equil   =', mp.nstr(vals['3D regular']/vals['2D equil'], 10))
