import sympy as sp

x, y, x_c, y_c, r1, r2, alpha_0 = sp.symbols('x y x_c y_c r1 r2 alpha_0', real=True)
r_sq = (x - x_c)**2 + (y - y_c)**2

eta = (r_sq - r1**2) / (r2**2 - r1**2)
gamma = (2*eta - 1) / (eta * (1 - eta))
alpha_expr = 1 - (1 - alpha_0) / (1 + sp.exp(gamma))

dx = sp.diff(alpha_expr, x)
dy = sp.diff(alpha_expr, y)
dxx = sp.diff(dx, x)
dyy = sp.diff(dy, y)
lap = dxx + dyy

# simplify and convert to julia
print("dx =", sp.julia_code(dx))
print("dy =", sp.julia_code(dy))
print("lap =", sp.julia_code(lap))
