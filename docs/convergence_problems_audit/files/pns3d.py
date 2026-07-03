# pns3d.py — clean-room reimplementation of the paper §5.2 3D MMS discretization.
# Independent of Gridap: P1/P2 Lagrange on Kuhn tets, analytic basis Hessians (affine cells => exact),
# ASGS stabilization exactly as src/formulations/continuous_problem.jl transcribes the paper.
# Conventions: (grad u)_ij = du_i/dx_j  (standard Jacobian; NOTE Gridap stores the transpose).
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

# ----------------------------------------------------------------------------- alpha field
class SmoothRadialPorosity:
    def __init__(self, a0, ainf, r1, r2):
        self.a0, self.ainf, self.r1, self.r2 = a0, ainf, r1, r2

    def eval_all(self, X):
        """X: (N,3). Returns alpha (N,), grad (N,3), lap (N,), hess (N,3,3). z-invariant."""
        a0, ainf, r1, r2 = self.a0, self.ainf, self.r1, self.r2
        x1, x2 = X[:, 0], X[:, 1]
        r_sq = x1**2 + x2**2
        r = np.sqrt(r_sq)
        N = X.shape[0]
        A = np.full(N, ainf)
        G = np.zeros((N, 3)); Lp = np.zeros(N); H = np.zeros((N, 3, 3))
        A[r <= r1] = a0
        mid = (r > r1) & (r < r2)
        if np.any(mid):
            rm = r[mid]; rsqm = r_sq[mid]
            eta = (rsqm - r1**2) / (r2**2 - r1**2)
            gam = (2.0*eta - 1.0) / (eta * (1.0 - eta))
            sat_hi = gam > 100.0; sat_lo = gam < -100.0
            ok = ~(sat_hi | sat_lo)
            Am = np.where(sat_hi, ainf, np.where(sat_lo, a0, 0.0))
            # values on ok
            eg = np.exp(np.clip(gam, -100, 100))
            Aok = ainf - (ainf - a0) / (1.0 + eg)
            Am = np.where(ok, Aok, Am)
            A[mid] = Am
            # derivatives (zero where saturated)
            deta = 2.0*rm/(r2**2 - r1**2)
            d2eta = 2.0/(r2**2 - r1**2)
            dgde = (2.0*eta**2 - 2.0*eta + 1.0) / (eta**2 * (1.0-eta)**2)
            d2gde2 = (4.0*eta - 2.0)/(eta**2*(1.0-eta)**2) - 2.0*(2.0*eta**2 - 2.0*eta + 1.0)*(1.0-2.0*eta)/(eta**3*(1.0-eta)**3)
            dadg = (ainf-a0)*eg/(1.0+eg)**2
            d2adg2 = (ainf-a0)*eg*(1.0-eg)/(1.0+eg)**3
            dadr = dadg*dgde*deta
            d2adr2 = d2adg2*(dgde*deta)**2 + dadg*(d2gde2*deta**2 + dgde*d2eta)
            dadr = np.where(ok, dadr, 0.0); d2adr2 = np.where(ok, d2adr2, 0.0)
            lap = d2adr2 + dadr/rm
            x1m, x2m = x1[mid], x2[mid]
            gx = dadr*(x1m/rm); gy = dadr*(x2m/rm)
            r3 = rsqm*rm
            H11 = d2adr2*(x1m*x1m)/rsqm + dadr*(1.0/rm - (x1m*x1m)/r3)
            H22 = d2adr2*(x2m*x2m)/rsqm + dadr*(1.0/rm - (x2m*x2m)/r3)
            H12 = d2adr2*(x1m*x2m)/rsqm - dadr*(x1m*x2m)/r3
            G[mid, 0] = gx; G[mid, 1] = gy
            Lp[mid] = lap
            H[mid, 0, 0] = H11; H[mid, 1, 1] = H22; H[mid, 0, 1] = H12; H[mid, 1, 0] = H12
        return A, G, Lp, H

# ----------------------------------------------------------------------------- exact solution / forcing
class MMS3D:
    """z-extruded §5.2 field. u = U a0 S / alpha, p = P cos(k x) sin(k y), k = pi/L."""
    def __init__(self, U, alpha_field, L, Re, Da, alpha_infty, nu, sigma, visc="deviatoric"):
        self.U, self.af, self.L = U, alpha_field, L
        self.nu, self.sigma = nu, sigma
        self.Pc = (1.0 + Re + Da) * U * nu / L
        self.k = np.pi / L
        self.visc = visc

    def S(self, X):
        k = self.k
        s1, s2 = np.sin(k*X[:,0]), np.sin(k*X[:,1])
        c1, c2 = np.cos(k*X[:,0]), np.cos(k*X[:,1])
        S = np.zeros((X.shape[0], 3)); S[:,0] = s1*s2; S[:,1] = c1*c2
        return S, (s1, s2, c1, c2)

    def DS(self, X):
        """DS_ij = dS_i/dx_j."""
        k = self.k
        s1, s2 = np.sin(k*X[:,0]), np.sin(k*X[:,1])
        c1, c2 = np.cos(k*X[:,0]), np.cos(k*X[:,1])
        D = np.zeros((X.shape[0], 3, 3))
        D[:,0,0] = k*c1*s2; D[:,0,1] = k*s1*c2
        D[:,1,0] = -k*s1*c2; D[:,1,1] = -k*c1*s2
        return D

    def u_ex(self, X):
        A, _, _, _ = self.af.eval_all(X)
        S, _ = self.S(X)
        return self.U * self.af.a0 * S / A[:, None]

    def grad_u_ex(self, X):
        A, gA, _, _ = self.af.eval_all(X)
        S, _ = self.S(X); DS = self.DS(X)
        c = self.U * self.af.a0
        # G_ij = c[(1/A) DS_ij - (1/A^2) S_i gA_j]
        return c * (DS / A[:, None, None] - S[:, :, None]*gA[:, None, :] / (A**2)[:, None, None])

    def lap_u_ex(self, X):
        A, gA, lA, _ = self.af.eval_all(X)
        S, _ = self.S(X); DS = self.DS(X)
        k = self.k
        lapS = -2.0 * k**2 * S
        c = self.U * self.af.a0
        P = c / A
        gP = -(c / A**2)[:, None] * gA
        lP = -(c / A**2)*lA + 2.0*(c / A**3)*np.einsum('nd,nd->n', gA, gA)
        # lap(P S)_i = P lapS_i + 2 gP_j DS_ij + lP S_i
        return P[:,None]*lapS + 2.0*np.einsum('nij,nj->ni', DS, gP) + lP[:,None]*S

    def grad_div_u_ex(self, X):
        A, gA, _, H = self.af.eval_all(X)
        c = self.U * self.af.a0
        k = self.k
        s1, s2 = np.sin(k*X[:,0]), np.sin(k*X[:,1])
        c1, c2 = np.cos(k*X[:,0]), np.cos(k*X[:,1])
        S1, S2 = s1*s2, c1*c2
        S1x, S1y = k*c1*s2, k*s1*c2
        S2x, S2y = -k*s1*c2, -k*c1*s2
        Ax, Ay = gA[:,0], gA[:,1]
        Axx, Axy, Ayy = H[:,0,0], H[:,0,1], H[:,1,1]
        phi  = S1*Ax + S2*Ay
        phix = S1x*Ax + S1*Axx + S2x*Ay + S2*Axy
        phiy = S1y*Ax + S1*Axy + S2y*Ay + S2*Ayy
        gx = -c*phix/A**2 + 2.0*c*phi*Ax/A**3
        gy = -c*phiy/A**2 + 2.0*c*phi*Ay/A**3
        out = np.zeros((X.shape[0], 3)); out[:,0] = gx; out[:,1] = gy
        return out

    def p_ex(self, X):
        k = self.k
        return self.Pc * np.cos(k*X[:,0])*np.sin(k*X[:,1])

    def grad_p_ex(self, X):
        k = self.k
        g = np.zeros((X.shape[0], 3))
        g[:,0] = -self.Pc*k*np.sin(k*X[:,0])*np.sin(k*X[:,1])
        g[:,1] =  self.Pc*k*np.cos(k*X[:,0])*np.cos(k*X[:,1])
        return g

    def forcing(self, X):
        """f = alpha (u.grad)u + alpha grad p + sigma u - visc(u)."""
        A, gA, _, _ = self.af.eval_all(X)
        u = self.u_ex(X); G = self.grad_u_ex(X); lu = self.lap_u_ex(X)
        gp = self.grad_p_ex(X)
        conv = A[:,None]*np.einsum('nij,nj->ni', G, u)
        pres = A[:,None]*gp
        rxn = self.sigma*u
        nu = self.nu
        Ssym = 0.5*(G + np.swapaxes(G, 1, 2))
        divu = np.einsum('nii->n', G)
        gdiv = self.grad_div_u_ex(X)
        if self.visc == "deviatoric":
            visc = 2.0*nu*(np.einsum('nij,nj->ni', Ssym, gA) - (1.0/3.0)*divu[:,None]*gA) \
                   + nu*A[:,None]*lu + (nu/3.0)*A[:,None]*gdiv
        elif self.visc == "symgrad":
            visc = 2.0*nu*np.einsum('nij,nj->ni', Ssym, gA) + nu*A[:,None]*lu + nu*A[:,None]*gdiv
        elif self.visc == "laplacian":
            visc = nu*np.einsum('nij,nj->ni', G, gA) + nu*A[:,None]*lu
        else:
            raise ValueError(self.visc)
        return conv + pres + rxn - visc

    def g_mass(self, X):
        """eps_phys = 0 here; div(alpha u_ex) = 0 analytically -> returns exact value for checking."""
        A, gA, _, _ = self.af.eval_all(X)
        u = self.u_ex(X); G = self.grad_u_ex(X)
        divu = np.einsum('nii->n', G)
        return A*divu + np.einsum('nd,nd->n', gA, u)

# ----------------------------------------------------------------------------- mesh (Kuhn tets)
def kuhn_mesh(n1, n2, n3, lx=1.0, ly=1.0, lz=0.3):
    """Structured Kuhn tetrahedralization of a box: 6 tets per hex (permutation construction)."""
    xs = np.linspace(0, lx, n1+1); ys = np.linspace(0, ly, n2+1); zs = np.linspace(0, lz, n3+1)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
    verts = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)
    def vid(i, j, k): return (i*(n2+1) + j)*(n3+1) + k
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    tets = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                base = np.array([i, j, k])
                for p in perms:
                    idx = [base.copy()]
                    cur = base.copy()
                    for ax in p:
                        cur = cur.copy(); cur[ax] += 1
                        idx.append(cur)
                    tets.append([vid(*v) for v in idx])
    tets = np.array(tets, dtype=np.int64)
    # positive orientation
    v = verts[tets]
    J = np.stack([v[:,1]-v[:,0], v[:,2]-v[:,0], v[:,3]-v[:,0]], axis=2)
    det = np.linalg.det(J)
    flip = det < 0
    tets[flip, 2], tets[flip, 3] = tets[flip, 3].copy(), tets[flip, 2].copy()
    return verts, tets

# ----------------------------------------------------------------------------- P1/P2 spaces
EDGES_LOC = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

class ScalarSpace:
    """Lagrange P1 or P2 on tets. Provides node coords, cell->dof map, and per-cell basis data."""
    def __init__(self, verts, tets, order):
        self.order = order
        nv = verts.shape[0]
        if order == 1:
            self.ndof = nv
            self.cell_dofs = tets.copy()
            self.node_coords = verts.copy()
        elif order == 2:
            edge_map = {}
            cell_edofs = np.zeros((tets.shape[0], 6), dtype=np.int64)
            ecoords = []
            for c in range(tets.shape[0]):
                for le, (a, b) in enumerate(EDGES_LOC):
                    va, vb = tets[c, a], tets[c, b]
                    key = (min(va, vb), max(va, vb))
                    if key not in edge_map:
                        edge_map[key] = len(ecoords)
                        ecoords.append(0.5*(verts[va] + verts[vb]))
                    cell_edofs[c, le] = nv + edge_map[key]
            self.ndof = nv + len(ecoords)
            self.cell_dofs = np.concatenate([tets, cell_edofs], axis=1)
            self.node_coords = np.concatenate([verts, np.array(ecoords)], axis=0)
        else:
            raise ValueError(order)

def ref_basis(order, pts):
    """pts: (nq,3) reference coords. Returns phi (nq,nb), dphi (nq,nb,3) wrt (l1,l2,l3)-style
    reference coords (x,y,z on the unit tet), hess (nq,nb,3,3)."""
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    l0 = 1.0 - x - y - z; l1, l2, l3 = x, y, z
    L = [l0, l1, l2, l3]
    gL = np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])  # grad of barycentric wrt ref coords
    nq = pts.shape[0]
    if order == 1:
        nb = 4
        phi = np.stack(L, axis=1)
        dphi = np.broadcast_to(gL[None,:,:], (nq, nb, 3)).copy()
        hess = np.zeros((nq, nb, 3, 3))
        return phi, dphi, hess
    nb = 10
    phi = np.zeros((nq, nb)); dphi = np.zeros((nq, nb, 3)); hess = np.zeros((nq, nb, 3, 3))
    for i in range(4):
        phi[:, i] = L[i]*(2.0*L[i] - 1.0)
        dphi[:, i, :] = (4.0*L[i] - 1.0)[:, None]*gL[i][None, :]
        hess[:, i, :, :] = 4.0*np.outer(gL[i], gL[i])[None, :, :]
    for e, (a, b) in enumerate(EDGES_LOC):
        phi[:, 4+e] = 4.0*L[a]*L[b]
        dphi[:, 4+e, :] = 4.0*(L[a][:, None]*gL[b][None, :] + L[b][:, None]*gL[a][None, :])
        hess[:, 4+e, :, :] = 4.0*(np.outer(gL[a], gL[b]) + np.outer(gL[b], gL[a]))[None, :, :]
    return phi, dphi, hess

def tet_quadrature(q):
    """Collapsed (Duffy) Gauss-Legendre rule on the unit tet; q points per direction.
    Exact for total degree <= 2q-3 at least; spectrally accurate for smooth integrands."""
    gp, gw = np.polynomial.legendre.leggauss(q)
    gp = 0.5*(gp + 1.0); gw = 0.5*gw
    a, b, c = np.meshgrid(gp, gp, gp, indexing='ij')
    wa, wb, wc = np.meshgrid(gw, gw, gw, indexing='ij')
    a, b, c = a.ravel(), b.ravel(), c.ravel()
    w = (wa*wb*wc).ravel()
    # Duffy: x = a, y = b(1-a), z = c(1-a)(1-b); Jacobian = (1-a)^2 (1-b)
    x = a; y = b*(1.0-a); z = c*(1.0-a)*(1.0-b)
    w = w*(1.0-a)**2*(1.0-b)
    return np.stack([x, y, z], axis=1), w

# ----------------------------------------------------------------------------- assembly helpers
class CellGeometry:
    def __init__(self, verts, tets):
        v = verts[tets]                    # (nel,4,3)
        self.X0 = v[:, 0, :]
        J = np.stack([v[:,1]-v[:,0], v[:,2]-v[:,0], v[:,3]-v[:,0]], axis=2)  # (nel,3,3), columns
        self.J = J
        self.detJ = np.linalg.det(J)
        self.Jinv = np.linalg.inv(J)
        self.vol = np.abs(self.detJ)/6.0
        self.h = (6.0*np.sqrt(2.0)*self.vol)**(1.0/3.0)  # regular-tet-edge from volume (as smoke3d.jl)

    def phys_points(self, ref_pts):
        # x = X0 + J @ xi
        return self.X0[:, None, :] + np.einsum('eij,qj->eqi', self.J, ref_pts)
