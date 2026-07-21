#!/usr/bin/env python3
# =============================================================================
# genuine3d_mms_verification.py
#
# Symbolic (sympy) verification of the GENUINELY three-dimensional manufactured
# solution used by the audit-response 3D test (R6/N19), implemented in
# src/problems/mms_paper.jl (UExFuncGenuine3D / PExFuncGenuine3D). Complements
# manufactured_solution_verification.py, which covers the default z-extruded field.
#
# The genuine field: u = U*alpha_0 * v/alpha with v = curl(A) for a smooth vector
# potential A, so div(alpha*u) = div(curl A) = 0 EXACTLY for any alpha. With equal
# wavenumber k in x,y,z (m=k) and amplitudes (a1,a2,a3)=(1,2,3):
#     v_x = -k c1 s2 s3,  v_y = 2k s1 c2 s3,  v_z = -k s1 s2 c3   (si=sin(k xi), ci=cos)
# It checks: (1) div(v)=0 (=> div(alpha u)=0 for any alpha); (2) v is genuinely 3D
# (u_z != 0, all components depend on x,y,z); (3) the closed-form gradient grad_v
# (T[i,j]=d_i v_j) transcribed into _grad_v3; (4) the Laplacian eigenvalue
# Delta v = -3 k^2 v exploited by lap_u_ex; (5) div u = -(v.grad alpha)/alpha^2.
#
# Run:  python3 genuine3d_mms_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"; results.append((tag, name)); print(f"  [{tag}] {name}"); return ok

print("=" * 70)
print("Genuinely-3D manufactured solution (R6/N19) -- symbolic checks")
print("=" * 70)

x, y, z = sp.symbols('x y z', real=True)
k = sp.symbols('k', positive=True)                 # equal wavenumber (m = k)
a1, a2, a3 = 1, 2, 3                                # amplitudes used in the code

def curl(F):
    Fx, Fy, Fz = F
    return (sp.diff(Fz, y) - sp.diff(Fy, z),
            sp.diff(Fx, z) - sp.diff(Fz, x),
            sp.diff(Fy, x) - sp.diff(Fx, y))
def div(F):
    return sp.diff(F[0], x) + sp.diff(F[1], y) + sp.diff(F[2], z)

A = (a1*sp.sin(k*x)*sp.cos(k*y)*sp.cos(k*z),
     a2*sp.cos(k*x)*sp.sin(k*y)*sp.cos(k*z),
     a3*sp.cos(k*x)*sp.cos(k*y)*sp.sin(k*z))
v = tuple(sp.simplify(c) for c in curl(A))

# The closed forms transcribed into mms_paper.jl _v3 (with m=k, a=(1,2,3)):
v_code = (-k*sp.cos(k*x)*sp.sin(k*y)*sp.sin(k*z),
          2*k*sp.sin(k*x)*sp.cos(k*y)*sp.sin(k*z),
          -k*sp.sin(k*x)*sp.sin(k*y)*sp.cos(k*z))

print("\n[1] v = curl(A): div-free and matches the code's _v3 closed form")
check("div(v) == 0  (=> div(alpha*u)=0 for any alpha)", sp.simplify(div(v)) == 0)
check("v matches _v3 (mms_paper.jl)", all(sp.simplify(v[i]-v_code[i]) == 0 for i in range(3)))

# div(alpha*u)=0 for an arbitrary alpha(x,y) (u = U a0 v/alpha):
alpha = sp.Function('alpha', positive=True)(x, y)
U, a0 = sp.symbols('U a0', positive=True)
u = tuple(U*a0*vi/alpha for vi in v)
print("\n[2] u = U a0 v/alpha inherits div(alpha*u)=0 for arbitrary alpha(x,y)")
check("div(alpha*u) == 0", sp.simplify(div(tuple(alpha*ui for ui in u))) == 0)

print("\n[3] genuinely 3D")
check("u_z != 0", sp.simplify(v[2]) != 0)
check("every component depends on x, y and z",
      all(sp.diff(v[i], x) != 0 and sp.diff(v[i], y) != 0 and sp.diff(v[i], z) != 0 for i in range(3)))

print("\n[4] grad_v (T[i,j]=d_i v_j) matches _grad_v3, and Delta v = -3 k^2 v")
# _grad_v3 entries (row-major, grouped by v-component): (d1v1,d2v1,d3v1, d1v2,d2v2,d3v2, d1v3,d2v3,d3v3)
s1,s2,s3 = sp.sin(k*x),sp.sin(k*y),sp.sin(k*z); c1,c2,c3 = sp.cos(k*x),sp.cos(k*y),sp.cos(k*z)
grad_v_code = [ k**2*s1*s2*s3,   -k**2*c1*c2*s3,  -k**2*c1*s2*c3,
                2*k**2*c1*c2*s3, -2*k**2*s1*s2*s3, 2*k**2*s1*c2*c3,
                -k**2*c1*s2*c3,  -k**2*s1*c2*c3,   k**2*s1*s2*s3 ]
grad_v_sym = [sp.diff(v_code[j], var) for j in range(3) for var in (x,y,z)]  # d_i v_j grouped by j
check("_grad_v3 matches d_i v_j exactly",
      all(sp.simplify(grad_v_sym[n]-grad_v_code[n]) == 0 for n in range(9)))
lap_v = tuple(sp.diff(vi,x,2)+sp.diff(vi,y,2)+sp.diff(vi,z,2) for vi in v_code)
check("Delta v == -3 k^2 v  (eigenfunction, used by lap_u_ex)",
      all(sp.simplify(lap_v[i] - (-3*k**2*v_code[i])) == 0 for i in range(3)))

print("\n[5] div u = -(v . grad alpha)/alpha^2  (grad_div_u_ex basis; alpha z-invariant)")
divu = sp.simplify(div(u))
expected = sp.simplify(-U*a0*(v[0]*sp.diff(alpha,x) + v[1]*sp.diff(alpha,y))/alpha**2)
check("div u matches -(U a0)(v.grad alpha)/alpha^2", sp.simplify(divu - expected) == 0)
check("div u depends on z (=> grad_div_u_ex has nonzero z-component)", sp.diff(divu, z) != 0)

print("\n" + "=" * 70)
npass = sum(1 for t, _ in results if t == "PASS")
print(f"SUMMARY: {npass}/{len(results)} checks passed.")
for t, nme in results:
    if t == "FAIL": print(f"   FAILED: {nme}")
print("=" * 70)
import sys
sys.exit(0 if npass == len(results) else 1)
