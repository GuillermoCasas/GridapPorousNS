# assemble.py — ASGS-stabilized residual + Picard Jacobian, mirroring
# src/formulations/continuous_problem.jl (ASGS branch, pi=0) term by term.
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
from pns3d import (SmoothRadialPorosity, MMS3D, kuhn_mesh, ScalarSpace, ref_basis,
                   tet_quadrature, CellGeometry, EDGES_LOC)

class Problem:
    def __init__(self, n=(8,8,2), kv=2, kp=2, qorder=6, alpha0=0.5, Re=1.0, Da=1.0,
                 U=1.0, L=1.0, visc="deviatoric", c1_mult=1.0,
                 stab_on=True, drop_visc_in_R=False, drop_visc_in_adj=False,
                 eps_num_mult=1.0, u_floor=1e-6, tau_lim=1e-8, domain=(1.0,1.0,0.3)):
        self.nu = U*L/Re
        alpha_infty = 1.0
        self.sigma = Da*alpha_infty*self.nu/L**2
        self.alpha0 = alpha0
        self.af = SmoothRadialPorosity(alpha0, alpha_infty, 0.2, 0.4)
        self.mms = MMS3D(U, self.af, L, Re, Da, alpha_infty, self.nu, self.sigma, visc=visc)
        self.eps_num = eps_num_mult * 1e-4 * alpha0 / (self.nu*(1.0+Re+Da))
        self.eps_phys = 0.0
        self.visc = visc
        self.cgd = {"deviatoric": 0.5-1.0/3.0, "symgrad": 0.5, "laplacian": None}[visc]
        self.kv, self.kp = kv, kp
        self.c1 = 4.0*kv**4*c1_mult
        self.c2 = 2.0*kv**2*c1_mult   # smoke3d.jl scales BOTH by c1_mult
        self.stab_on = stab_on
        self.drop_visc_in_R = drop_visc_in_R
        self.drop_visc_in_adj = drop_visc_in_adj
        self.u_floor, self.tau_lim = u_floor, tau_lim

        verts, tets = kuhn_mesh(*n, lx=domain[0], ly=domain[1], lz=domain[2])
        self.geom = CellGeometry(verts, tets)
        self.Vu = ScalarSpace(verts, tets, kv)
        self.Vp = ScalarSpace(verts, tets, kp)
        self.Nu, self.Np = self.Vu.ndof, self.Vp.ndof
        self.ndof = 3*self.Nu + self.Np
        self.nel = tets.shape[0]

        # quadrature + reference basis
        self.qp, self.qw = tet_quadrature(qorder)
        self.nq = self.qp.shape[0]
        self.phiu_ref, self.dphiu_ref, self.hphiu_ref = ref_basis(kv, self.qp)
        self.phip_ref, self.dphip_ref, _ = ref_basis(kp, self.qp)
        self.nbu = self.phiu_ref.shape[1]; self.nbp = self.phip_ref.shape[1]

        # physical qpoints, alpha and f at qpoints (fixed data)
        self.Xq = self.geom.phys_points(self.qp)      # (nel,nq,3)
        Xf = self.Xq.reshape(-1,3)
        A, gA, _, _ = self.af.eval_all(Xf)
        self.Aq = A.reshape(self.nel, self.nq)
        self.gAq = gA.reshape(self.nel, self.nq, 3)
        self.fq = self.mms.forcing(Xf).reshape(self.nel, self.nq, 3)
        self.gq = np.zeros((self.nel, self.nq))       # exact mass source is 0 (checked)
        self.wdet = self.qw[None,:]*np.abs(self.geom.detJ)[:,None]  # (nel,nq)

        # boundary u-nodes (all 6 faces Dirichlet)
        nc = self.Vu.node_coords
        tol = 1e-10
        onb = (np.abs(nc[:,0])<tol)|(np.abs(nc[:,0]-domain[0])<tol)|\
              (np.abs(nc[:,1])<tol)|(np.abs(nc[:,1]-domain[1])<tol)|\
              (np.abs(nc[:,2])<tol)|(np.abs(nc[:,2]-domain[2])<tol)
        self.bnd_u_nodes = np.where(onb)[0]
        fixed = np.concatenate([self.bnd_u_nodes + c*self.Nu for c in range(3)])
        self.fixed = np.sort(fixed)
        mask = np.ones(self.ndof, dtype=bool); mask[self.fixed] = False
        self.free = np.where(mask)[0]

        # cell dof maps (global): u comp-major then p
        cd_u = self.Vu.cell_dofs; cd_p = self.Vp.cell_dofs
        self.gdofs = np.concatenate([cd_u + c*self.Nu for c in range(3)] + [cd_p + 3*self.Nu], axis=1)  # (nel, 3nbu+nbp)
        self.nloc = self.gdofs.shape[1]

        # precompute per-element physical basis derivatives lazily in chunks (store chunk arrays only)
        self.chunk = 128

    # -------- field helpers --------
    def interpolant(self):
        x = np.zeros(self.ndof)
        uex = self.mms.u_ex(self.Vu.node_coords)
        for c in range(3):
            x[c*self.Nu:(c+1)*self.Nu] = uex[:,c]
        x[3*self.Nu:] = self.mms.p_ex(self.Vp.node_coords)
        return x

    def _phys_basis(self, sl):
        """Physical grad/hess of u-basis and grad of p-basis for element slice sl."""
        Jinv = self.geom.Jinv[sl]                        # (ne,3,3)
        # dphi_phys[e,q,b,i] = dphi_ref[q,b,j] Jinv[e,j,i]
        gU = np.einsum('qbj,eji->eqbi', self.dphiu_ref, Jinv)
        gP = np.einsum('qbj,eji->eqbi', self.dphip_ref, Jinv)
        # hess_phys[e,q,b,i,k] = Jinv[e,j,i] Href[q,b,j,l] Jinv[e,l,k]
        hU = np.einsum('eji,qbjl,elk->eqbik', Jinv, self.hphiu_ref, Jinv)
        return gU, gP, hU

    def _fields(self, x, sl, gU, gP, hU):
        """Evaluate u, grad u, lap u, graddiv u, p, grad p at qpoints for elements sl."""
        gd = self.gdofs[sl]
        ne = gd.shape[0]
        Uc = np.stack([x[gd[:, c*self.nbu:(c+1)*self.nbu]] for c in range(3)], axis=2)  # (ne,nbu,3)
        Pc = x[gd[:, 3*self.nbu:]]                                                       # (ne,nbp)
        u = np.einsum('ebi,qb->eqi', Uc, self.phiu_ref)
        G = np.einsum('ebi,eqbj->eqij', Uc, gU)          # G_ij = d u_i / dx_j
        lap = np.einsum('ebi,eqbkk->eqi', Uc, hU)
        gdiv = np.einsum('ebj,eqbij->eqi', Uc, hU)       # (grad div u)_i = sum_j d_i d_j u_j
        p = np.einsum('eb,qb->eq', Pc, self.phip_ref)
        gp = np.einsum('eb,eqbi->eqi', Pc, gP)
        return u, G, lap, gdiv, p, gp

    def _tau(self, u, A, h):
        mag = np.sqrt(np.einsum('eqi,eqi->eq', u, u) + self.u_floor**2)
        Ainv = self.c1*self.nu/h[:,None]**2 + self.c2*mag/h[:,None] + self.tau_lim
        tau1 = 1.0/(A*Ainv + self.sigma + self.tau_lim)
        tau2 = h[:,None]**2/(self.c1*A/Ainv + self.tau_lim)
        return tau1, tau2

    def _divvisc_field(self, A, gA, G, lap, gdiv):
        nu = self.nu
        if self.visc == "laplacian":
            return nu*np.einsum('eqij,eqj->eqi', G, gA) + nu*A[...,None]*lap
        S = 0.5*(G + np.swapaxes(G,2,3))
        divu = np.einsum('eqii->eq', G)
        if self.visc == "deviatoric":
            por = 2*nu*(np.einsum('eqij,eqj->eqi', S, gA) - (1.0/3.0)*divu[...,None]*gA)
        else:
            por = 2*nu*np.einsum('eqij,eqj->eqi', S, gA)
        return por + 2*A[...,None]*nu*(0.5*lap + self.cgd*gdiv)

    # -------- residual --------
    def residual(self, x, p_prev_vec=None):
        R = np.zeros(self.ndof)
        for s in range(0, self.nel, self.chunk):
            sl = slice(s, min(s+self.chunk, self.nel))
            gU, gP, hU = self._phys_basis(sl)
            u, G, lap, gdiv, p, gp = self._fields(x, sl, gU, gP, hU)
            A = self.Aq[sl]; gA = self.gAq[sl]; f = self.fq[sl]; w = self.wdet[sl]
            h = self.geom.h[sl]
            ne = u.shape[0]
            conv = A[...,None]*np.einsum('eqij,eqj->eqi', G, u)
            divvisc = self._divvisc_field(A, gA, G, lap, gdiv)
            Ru = conv + A[...,None]*gp + self.sigma*u - divvisc - f
            if self.drop_visc_in_R:
                Ru = Ru + divvisc          # i.e. remove -divvisc from Ru
            divu = np.einsum('eqii->eq', G)
            Rp = A*divu + np.einsum('eqi,eqi->eq', u, gA) - self.gq[sl]
            tau1, tau2 = self._tau(u, A, h)

            phi = self.phiu_ref; psip = self.phip_ref
            rloc = np.zeros((ne, self.nloc))
            # ---- Galerkin momentum ----
            # eps^d(u) or eps(u) @ grad(phi_a) for the viscous pairing
            S = 0.5*(G + np.swapaxes(G,2,3))
            if self.visc == "deviatoric":
                Spair = S - (1.0/3.0)*divu[...,None,None]*np.eye(3)[None,None,:,:]
                visc_gal = 2*self.nu*A[...,None,None]*np.einsum('eqik,eqak->eqai', Spair, gU)
            elif self.visc == "symgrad":
                visc_gal = 2*self.nu*A[...,None,None]*np.einsum('eqik,eqak->eqai', S, gU)
            else:
                visc_gal = self.nu*A[...,None,None]*np.einsum('eqik,eqak->eqai', G, gU)
            mom = np.einsum('eqi,qa->eqai', conv + self.sigma*u - f, phi)
            mom += visc_gal
            # pressure term: -p (alpha dphi_a_i + gA_i phi_a)
            mom -= np.einsum('eq,eqai->eqai', p, A[...,None,None]*gU + np.einsum('eqi,qa->eqai', gA, phi))
            # ---- Galerkin mass (+ iterative penalty) ----
            pen = np.zeros_like(p)
            if p_prev_vec is not None and self.eps_num > 0:
                gd = self.gdofs[sl]
                Ppc = p_prev_vec[gd[:, 3*self.nbu:] - 3*self.Nu]
                p_prev = np.einsum('eb,qb->eq', Ppc, psip)
                pen = self.eps_num*(p - p_prev)
            mass_val = self.eps_phys*p + A*divu + np.einsum('eqi,eqi->eq', u, gA) + pen - self.gq[sl]
            masr = np.einsum('eq,qm->eqm', mass_val, psip)
            if self.stab_on:
                # ---- momentum-test adjoint L*v (a,i,j) ----
                Lstar = self._Lstar_mom(u, A, gA, phi, gU, hU)
                Ru_eff = Ru
                stab_mom_rows = np.einsum('eqaij,eq,eqj->eqai', Lstar, tau1, Ru_eff)
                mom += stab_mom_rows
                # mass-adjoint on momentum rows: (alpha dphi + gA phi) tau2 Rp
                wdivv = A[...,None,None]*gU + np.einsum('eqi,qa->eqai', gA, phi)
                mom += np.einsum('eqai,eq,eq->eqai', wdivv, tau2, Rp)
                # pressure rows: (alpha grad psi) . tau1 Ru
                masr += np.einsum('eq,eqmj,eq,eqj->eqm', A, gP, tau1, Ru_eff)
                # (-eps_phys q) tau2 Rp = 0 here
            # integrate
            rl_u = np.einsum('eq,eqai->eai', w, mom)         # (ne, nbu, 3)
            rl_p = np.einsum('eq,eqm->em', w, masr)
            for c in range(3):
                rloc[:, c*self.nbu:(c+1)*self.nbu] = rl_u[:,:,c]
            rloc[:, 3*self.nbu:] = rl_p
            np.add.at(R, self.gdofs[sl].ravel(), rloc.ravel())
        return R

    def _Lstar_mom(self, u, A, gA, phi, gU, hU):
        """L* on momentum test functions: (e,q,a,i,j) with v = phi_a e_i, component j of L*v.
        L*v = alpha (u.grad)v + viscadj(v) - sigma v.  (pressure adjoint handled on p rows.)"""
        ne, nq = u.shape[0], u.shape[1]
        nbu = self.nbu
        adotg = np.einsum('eqj,eqaj->eqa', u, gU)             # u . grad phi_a
        eye = np.eye(3)
        out = np.einsum('eq,eqa,ij->eqaij', A, adotg, eye)
        out -= self.sigma*np.einsum('qa,ij->qaij', phi, eye)[None,...]
        if not self.drop_visc_in_adj:
            nu = self.nu
            if self.visc == "laplacian":
                # nu (grad v @ gA)_j + nu alpha (lap v)_j ; grad v[j,k] = delta_{ji} dphi_k
                # (grad v @ gA)_j = delta_{ji} (gphi . gA)
                gdotA = np.einsum('eqak,eqk->eqa', gU, gA)
                lapl = np.einsum('eqakk->eqa', hU)
                out += nu*np.einsum('eqa,ij->eqaij', gdotA, eye)
                out += nu*np.einsum('eq,eqa,ij->eqaij', A, lapl, eye)
            else:
                gdotA = np.einsum('eqak,eqk->eqa', gU, gA)
                lapl = np.einsum('eqakk->eqa', hU)
                # (eps(v) gA)_j = 0.5(delta_{ji} (gphi.gA) + gphi_j gA_i)
                t = 0.5*(np.einsum('eqa,ij->eqaij', gdotA, eye)
                         + np.einsum('eqaj,eqi->eqaij', gU, gA))
                if self.visc == "deviatoric":
                    t -= (1.0/3.0)*np.einsum('eqai,eqj->eqaij', gU, gA)
                out += 2*nu*t
                # bulk: 2 alpha nu (0.5 delta_{ij} lap phi + cgd Hphi[j,i])
                out += 2*nu*np.einsum('eq,eqa,ij->eqaij', A, 0.5*lapl, eye)
                out += 2*nu*self.cgd*np.einsum('eq,eqaji->eqaij', A, hU)
        return out

    def _Rdu(self, u, A, gA, phi, gU, hU):
        """Strong residual of the Picard increment for u-columns: (e,q,b,c,j)."""
        eye = np.eye(3)
        adotg = np.einsum('eqk,eqbk->eqb', u, gU)
        out = np.einsum('eq,eqb,cj->eqbcj', A, adotg, eye)
        out += self.sigma*np.einsum('qb,cj->qbcj', phi, eye)[None,...]
        if not self.drop_visc_in_R:
            nu = self.nu
            if self.visc == "laplacian":
                gdotA = np.einsum('eqbk,eqk->eqb', gU, gA)
                lapl = np.einsum('eqbkk->eqb', hU)
                out -= nu*np.einsum('eqb,cj->eqbcj', gdotA, eye)
                out -= nu*np.einsum('eq,eqb,cj->eqbcj', A, lapl, eye)
            else:
                gdotA = np.einsum('eqbk,eqk->eqb', gU, gA)
                lapl = np.einsum('eqbkk->eqb', hU)
                t = 0.5*(np.einsum('eqb,cj->eqbcj', gdotA, eye)
                         + np.einsum('eqbj,eqc->eqbcj', gU, gA))
                if self.visc == "deviatoric":
                    t -= (1.0/3.0)*np.einsum('eqbc,eqj->eqbcj', gU, gA)
                out -= 2*nu*t
                out -= 2*nu*np.einsum('eq,eqb,cj->eqbcj', A, 0.5*lapl, eye)
                out -= 2*nu*self.cgd*np.einsum('eq,eqbjc->eqbcj', A, hU)
        return out

    # -------- Picard Jacobian --------
    def jacobian(self, x):
        rows, cols, vals = [], [], []
        nbu, nbp = self.nbu, self.nbp
        for s in range(0, self.nel, self.chunk):
            sl = slice(s, min(s+self.chunk, self.nel))
            gU, gP, hU = self._phys_basis(sl)
            u, G, lap, gdiv, p, gp = self._fields(x, sl, gU, gP, hU)
            A = self.Aq[sl]; gA = self.gAq[sl]; w = self.wdet[sl]
            h = self.geom.h[sl]
            ne = u.shape[0]
            tau1, tau2 = self._tau(u, A, h)
            phi = self.phiu_ref; psip = self.phip_ref
            wA = w*A
            K = np.zeros((ne, self.nloc, self.nloc))

            # ---------- Galerkin, u rows x u cols ----------
            adotgb = np.einsum('eqk,eqbk->eqb', u, gU, optimize=True)
            Mconv = np.einsum('eq,qa,eqb->eab', wA, phi, adotgb, optimize=True)
            Mrxn  = self.sigma*np.einsum('eq,qa,qb->eab', w, phi, phi, optimize=True)
            if self.visc == "laplacian":
                Mvisc = self.nu*np.einsum('eq,eqak,eqbk->eab', wA, gU, gU, optimize=True)
                off1 = off2 = None
            else:
                Mvisc = self.nu*np.einsum('eq,eqak,eqbk->eab', wA, gU, gU, optimize=True)
                # off1[(i,a),(c,b)] = nu * sum_q wA * gU[b,i]*gU[a,c]
                off1 = self.nu*np.einsum('eq,eqac,eqbi->eacbi', wA, gU, gU, optimize=True)
                if self.visc == "deviatoric":
                    off2 = -(2.0/3.0)*self.nu*np.einsum('eq,eqai,eqbc->eaibc', wA, gU, gU, optimize=True)
                else:
                    off2 = None
            Mdiag = Mconv + Mrxn + Mvisc
            for i in range(3):
                K[:, i*nbu:(i+1)*nbu, i*nbu:(i+1)*nbu] += Mdiag
            if off1 is not None:
                for i in range(3):
                    for c in range(3):
                        K[:, i*nbu:(i+1)*nbu, c*nbu:(c+1)*nbu] += off1[:, :, c, :, i]
                        if off2 is not None:
                            K[:, i*nbu:(i+1)*nbu, c*nbu:(c+1)*nbu] += off2[:, :, i, :, c]

            # ---------- Galerkin, u rows x p cols / p rows x u cols ----------
            # -psi_n (alpha dphi_a_i + gA_i phi_a) ; psi_m (alpha dphi_b_c + gA_c phi_b)
            wdivv = np.einsum('eq,eqai->eqai', wA, gU, optimize=True) \
                    + np.einsum('eq,eqi,qa->eqai', w, gA, phi, optimize=True)   # weight-folded div(alpha v)
            for i in range(3):
                blk = np.einsum('eqa,qn->ean', wdivv[..., i], psip, optimize=True)
                K[:, i*nbu:(i+1)*nbu, 3*nbu:] += -blk
                K[:, 3*nbu:, i*nbu:(i+1)*nbu] += blk.transpose(0, 2, 1)

            # ---------- Galerkin, p rows x p cols ----------
            K[:, 3*nbu:, 3*nbu:] += (self.eps_num + self.eps_phys)*np.einsum('eq,qm,qn->emn', w, psip, psip, optimize=True)

            if self.stab_on:
                Lstar = self._Lstar_mom(u, A, gA, phi, gU, hU)   # (e,q,a,i,j)
                Rdu = self._Rdu(u, A, gA, phi, gU, hU)           # (e,q,b,c,j)
                wt1 = w*tau1; wt2 = w*tau2
                # flatten (a,i)->r = i*nbu+a and (b,c)->s = c*nbu+b to match local layout
                Lw = (Lstar*wt1[..., None, None, None]).transpose(0,1,3,2,4).reshape(ne, self.nq, 3*nbu, 3)
                Rf = Rdu.transpose(0,1,3,2,4).reshape(ne, self.nq, 3*nbu, 3)
                # K_uu stab = sum_{q,j} Lw[r,j] Rf[s,j]
                Kuu = np.einsum('eqrj,eqsj->ers', Lw, Rf, optimize=True)
                # mass stab on uu: wdivv (already w-folded)*tau2? need tau2 fold once ->
                Wd = (wdivv*tau2[..., None, None]).transpose(0,1,3,2).reshape(ne, self.nq, 3*nbu)
                Rdp = (A[...,None,None]*gU + np.einsum('eqc,qb->eqbc', gA, phi, optimize=True))
                Rdpf = Rdp.transpose(0,1,3,2).reshape(ne, self.nq, 3*nbu)
                Kuu += np.einsum('eqr,eqs->ers', Wd, Rdpf, optimize=True)
                K[:, :3*nbu, :3*nbu] += Kuu
                # u rows x p cols: Lw . (alpha gpsi)
                agP = np.einsum('eq,eqnj->eqnj', A, gP, optimize=True)
                K[:, :3*nbu, 3*nbu:] += np.einsum('eqrj,eqnj->ern', Lw, agP, optimize=True)
                # p rows x u cols: (w tau1 alpha gpsi_m) . Rdu
                agPw = np.einsum('eq,eqmj->eqmj', wt1*A, gP, optimize=True)
                K[:, 3*nbu:, :3*nbu] += np.einsum('eqmj,eqsj->ems', agPw, Rf, optimize=True)
                # p rows x p cols (PSPG)
                K[:, 3*nbu:, 3*nbu:] += np.einsum('eqmj,eqnj->emn', agPw, agP, optimize=True)

            gd = self.gdofs[sl]
            rows.append(np.repeat(gd, self.nloc, axis=1).ravel())
            cols.append(np.tile(gd, (1, self.nloc)).ravel())
            vals.append(K.ravel())
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        J = coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
        return J

    # -------- solve --------
    def solve(self, maxit=15, tol=1e-10, passes=3, verbose=True):
        """Mirrors solver_core.jl: outer penalty passes with p_prev frozen inside each pass,
        inner Picard/Newton updates, p_prev refreshed between passes."""
        x = self.interpolant()
        p_prev = x[3*self.Nu:].copy()
        hist = []
        for ps in range(passes):
            for it in range(maxit):
                R = self.residual(x, p_prev_vec=p_prev)
                rn = np.linalg.norm(R[self.free])
                hist.append(rn)
                if verbose:
                    print(f"    pass {ps} it {it}: ||R_free|| = {rn:.3e}")
                if rn < tol*max(1.0, hist[0]) or rn < 1e-12:
                    break
                J = self.jacobian(x)
                Jff = J[self.free][:, self.free].tocsc()
                dx = splu(Jff).solve(-R[self.free])
                x[self.free] += dx
                if np.linalg.norm(dx) < 1e-13*max(1.0, np.linalg.norm(x[self.free])):
                    break
            # re-centre the pressure to the exact-solution mean (constant mode is arbitrary;
            # keeps |p| O(1) so quadrature leakage of the constant into momentum stays negligible)
            m = self._pressure_mean_offset(x)
            x[3*self.Nu:] -= m
            p_new = x[3*self.Nu:]
            drift = np.linalg.norm(p_new - p_prev)/max(np.linalg.norm(p_new), 1e-30)
            p_prev = p_new.copy()
            if verbose:
                print(f"    pass {ps} done, pressure drift {drift:.3e}")
            if drift < 1e-9:
                break
        return x, hist


    def _pressure_mean_offset(self, x):
        vol = self.geom.vol.sum()
        acc = 0.0
        for s in range(0, self.nel, self.chunk):
            sl = slice(s, min(s+self.chunk, self.nel))
            gU, gP, hU = self._phys_basis(sl)
            _, _, _, _, p, _ = self._fields(x, sl, gU, gP, hU)
            pex = self.mms.p_ex(self.Xq[sl].reshape(-1,3)).reshape(p.shape)
            acc += np.einsum('eq,eq->', self.wdet[sl], p - pex)
        return acc/vol

    # -------- errors --------
    def errors(self, x):
        Uc = self.mms.U; Pc = self.mms.Pc
        vol = self.geom.vol.sum()
        el2u2 = 0.0; il2u2 = 0.0; eh1u2 = 0.0
        ep_num = 0.0; ep_mean_num = 0.0
        xi = self.interpolant()
        # accumulate pressure error with mean removal in a second pass
        e_p_int = 0.0
        for s in range(0, self.nel, self.chunk):
            sl = slice(s, min(s+self.chunk, self.nel))
            gU, gP, hU = self._phys_basis(sl)
            u, G, lap, gdiv, p, gp = self._fields(x, sl, gU, gP, hU)
            ui, _, _, _, _, _ = self._fields(xi, sl, gU, gP, hU)
            Xf = self.Xq[sl].reshape(-1,3)
            uex = self.mms.u_ex(Xf).reshape(u.shape)
            Gex = self.mms.grad_u_ex(Xf).reshape(G.shape)
            pex = self.mms.p_ex(Xf).reshape(p.shape)
            w = self.wdet[sl]
            eu = u - uex; ei = ui - uex; eg = G - Gex; ep = p - pex
            el2u2 += np.einsum('eq,eqi,eqi->', w, eu, eu)
            il2u2 += np.einsum('eq,eqi,eqi->', w, ei, ei)
            eh1u2 += np.einsum('eq,eqij,eqij->', w, eg, eg)
            e_p_int += np.einsum('eq,eq->', w, ep)
        pmean = e_p_int/vol
        el2p2 = 0.0
        for s in range(0, self.nel, self.chunk):
            sl = slice(s, min(s+self.chunk, self.nel))
            gU, gP, hU = self._phys_basis(sl)
            _, _, _, _, p, _ = self._fields(x, sl, gU, gP, hU)
            Xf = self.Xq[sl].reshape(-1,3)
            pex = self.mms.p_ex(Xf).reshape(p.shape)
            w = self.wdet[sl]
            ep = p - pex - pmean
            el2p2 += np.einsum('eq,eq->', w, ep**2)
        sq = np.sqrt(vol)
        return dict(el2u=np.sqrt(el2u2)/(Uc*sq), il2u=np.sqrt(il2u2)/(Uc*sq),
                    eh1u=np.sqrt(eh1u2)/Uc, el2p=np.sqrt(el2p2)/(Pc*sq),
                    hmean=self.geom.h.mean())
