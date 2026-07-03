# verify.py — self-checks of the clean-room implementation before trusting any conclusion.
import numpy as np
from pns3d import SmoothRadialPorosity, MMS3D, ref_basis, tet_quadrature
from assemble import Problem

rng = np.random.default_rng(0)

print("== 1. Oracle self-consistency (finite differences) ==")
af = SmoothRadialPorosity(0.5, 1.0, 0.2, 0.4)
mms = MMS3D(1.0, af, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, visc="deviatoric")
X = rng.uniform([0.05,0.05,0.02],[0.95,0.95,0.28],(200,3))
eps = 1e-6
def fd_grad(fn, X, m):
    out = np.zeros((X.shape[0], m, 3))
    for j in range(3):
        dp = X.copy(); dp[:,j]+=eps; dm = X.copy(); dm[:,j]-=eps
        out[:,:,j] = (np.atleast_2d(fn(dp).T).T - np.atleast_2d(fn(dm).T).T)/(2*eps)
    return out
# grad u vs FD
Gfd = fd_grad(mms.u_ex, X, 3)
G = mms.grad_u_ex(X)
print("   max|grad_u analytic-FD| =", np.abs(G-Gfd).max())
# lap u vs FD (of grad)
lap_fd = np.zeros((X.shape[0],3))
for j in range(3):
    dp = X.copy(); dp[:,j]+=eps; dm = X.copy(); dm[:,j]-=eps
    lap_fd += (mms.grad_u_ex(dp)[:,:,j] - mms.grad_u_ex(dm)[:,:,j])/(2*eps)
print("   max|lap_u analytic-FD|  =", np.abs(mms.lap_u_ex(X)-lap_fd).max())
# grad div u vs FD
divu = lambda Y: np.einsum('nii->n', mms.grad_u_ex(Y))
gd_fd = np.zeros((X.shape[0],3))
for j in range(3):
    dp = X.copy(); dp[:,j]+=eps; dm = X.copy(); dm[:,j]-=eps
    gd_fd[:,j] = (divu(dp)-divu(dm))/(2*eps)
print("   max|graddiv analytic-FD|=", np.abs(mms.grad_div_u_ex(X)-gd_fd).max())
# alpha derivatives vs FD
A,gA,lA,H = af.eval_all(X)
gA_fd = np.zeros((X.shape[0],3))
for j in range(3):
    dp = X.copy(); dp[:,j]+=eps; dm = X.copy(); dm[:,j]-=eps
    gA_fd[:,j] = (af.eval_all(dp)[0]-af.eval_all(dm)[0])/(2*eps)
print("   max|grad_alpha - FD|    =", np.abs(gA-gA_fd).max())
H_fd = np.zeros((X.shape[0],3,3))
for j in range(3):
    dp = X.copy(); dp[:,j]+=eps; dm = X.copy(); dm[:,j]-=eps
    H_fd[:,:,j] = (af.eval_all(dp)[1]-af.eval_all(dm)[1])/(2*eps)
print("   max|hess_alpha - FD|    =", np.abs(H-H_fd).max())
print("   max|tr(H)-lap|          =", np.abs(np.einsum('nii->n',H)-lA).max())
print("   max|div(alpha u_ex)|    =", np.abs(mms.g_mass(X)).max())
# f via strong-op FD: recompute divvisc by FD of the flux T = 2 A nu (eps^d u)
def flux(Y):
    Ay,_,_,_ = af.eval_all(Y)
    Gy = mms.grad_u_ex(Y)
    Sy = 0.5*(Gy+np.swapaxes(Gy,1,2))
    dv = np.einsum('nii->n', Gy)
    T = 2*1.0*Ay[:,None,None]*(Sy - (1.0/3.0)*dv[:,None,None]*np.eye(3)[None])
    return T
divT = np.zeros((X.shape[0],3))
for j in range(3):
    dp = X.copy(); dp[:,j]+=eps; dm = X.copy(); dm[:,j]-=eps
    divT += (flux(dp)[:,:,j]-flux(dm)[:,:,j])/(2*eps)
u = mms.u_ex(X); Gu = mms.grad_u_ex(X); A,_,_,_ = af.eval_all(X)
f_fd = A[:,None]*np.einsum('nij,nj->ni',Gu,u) + A[:,None]*mms.grad_p_ex(X) + 1.0*u - divT
print("   max|f(analytic)-f(FD flux)| =", np.abs(mms.forcing(X)-f_fd).max())

print("== 2. Basis checks (P2) ==")
qp, qw = tet_quadrature(4)
phi, dphi, hess = ref_basis(2, qp)
print("   partition of unity:", np.abs(phi.sum(axis=1)-1).max(), " grads sum:", np.abs(dphi.sum(axis=1)).max())
# FD check of dphi and hess at random ref points
P = rng.uniform(0.05,0.25,(50,3))
phi0, dphi0, hess0 = ref_basis(2, P)
for j in range(3):
    Pp=P.copy(); Pp[:,j]+=eps; Pm=P.copy(); Pm[:,j]-=eps
    dfd = (ref_basis(2,Pp)[0]-ref_basis(2,Pm)[0])/(2*eps)
    assert np.abs(dfd-dphi0[:,:,j]).max()<1e-8
    hfd = (ref_basis(2,Pp)[1]-ref_basis(2,Pm)[1])/(2*eps)
    assert np.abs(hfd-hess0[:,:,:,j]).max()<1e-7
print("   dphi & hess match FD: OK")
print("   quad vol of unit tet:", tet_quadrature(6)[1].sum(), "(exact 1/6 =", 1/6, ")")

print("== 3. Discrete checks on a coarse mesh ==")
pb = Problem(n=(4,4,1), kv=2, kp=2, qorder=6)
# quadrature convergence: integral of f.f with q=6 vs q=8
pb8 = Problem(n=(4,4,1), kv=2, kp=2, qorder=8)
i6 = np.einsum('eq,eqi,eqi->', pb.wdet, pb.fq, pb.fq)
i8 = np.einsum('eq,eqi,eqi->', pb8.wdet, pb8.fq, pb8.fq)
print(f"   int f.f  q=6: {i6:.12e}  q=8: {i8:.12e}  rel diff {abs(i6-i8)/abs(i8):.2e}")
# residual at interpolant should be small (consistency) and shrink with h
for n in [(4,4,1),(8,8,2)]:
    p2 = Problem(n=n, kv=2, kp=2, qorder=6)
    x = p2.interpolant()
    R = p2.residual(x, p_prev_vec=x[3*p2.Nu:])
    print(f"   n={n}: ||R(interp)||_free = {np.linalg.norm(R[p2.free]):.3e}  ndof={p2.ndof}")
# Galerkin-only symmetric-part sanity: Jacobian finite (no NaN)
J = pb.jacobian(pb.interpolant())
print("   J finite:", np.isfinite(J.data).all(), " nnz:", J.nnz)
