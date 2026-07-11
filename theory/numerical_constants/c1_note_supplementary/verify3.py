import itertools, sympy as sp, mpmath as mp
exec(open('verify2.py').read().split("# exact:")[0])  # reuse build/mp50_max

cases = [
 ('2D equil  P2 df',  [(0,0),(1,0),(sp.Rational(1,2),sp.sqrt(3)/2)], True,  '24'),
 ('2D equil  P2 all', [(0,0),(1,0),(sp.Rational(1,2),sp.sqrt(3)/2)], False, '36'),
 ('3D right  P2 all', [(0,0,0),(1,0,0),(0,1,0),(0,0,1)],             False, '200'),
 ('3D Kuhn   P2 all', [(0,0,0),(1,0,0),(1,1,0),(1,1,1)],             False, '240+30*sqrt(2)'),
 ('3D reg    P2 df',  [(0,0,0),(1,0,0),(sp.Rational(1,2),sp.sqrt(3)/2,0),
                       (sp.Rational(1,2),sp.sqrt(3)/6,sp.sqrt(6)/3)], True, '200/3'),
 ('3D reg    P2 all', [(0,0,0),(1,0,0),(sp.Rational(1,2),sp.sqrt(3)/2,0),
                       (sp.Rational(1,2),sp.sqrt(3)/6,sp.sqrt(6)/3)], False,'80'),
]
for nm, V, df, ref in cases:
    Ar, Br, h2 = build(V, 2, df)
    v = mp50_max(Ar, Br, h2)
    r = mp.mpf(sp.N(sp.sympify(ref), 60))
    ok = abs(v - r) < mp.mpf('1e-30')*max(1, abs(r))
    print(f'{nm}: {mp.nstr(v, 25)}  vs  {ref} = {mp.nstr(r, 25)}   {"OK" if ok else "MISMATCH"}')
