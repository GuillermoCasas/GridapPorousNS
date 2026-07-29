"""
[SUPERSEDED — NOT PART OF THE VERIFICATION GATE.  Read this before trusting it.]

This is a hand-run NumPy scratch check from the 2026-07-29 projector generalization.
It is NOT picked up by proof_verification/sympy/run_all.py, on three counts, any one
of which is sufficient: it lives above sympy/ (the runner's glob is non-recursive),
its name lacks the *_verification.py suffix the runner matches, and it prints
"ALL CHECKS PASSED." instead of the "SUMMARY: n/m" line the runner parses -- so even
if it were moved and renamed it would contribute 0 to the grand total.

Do NOT move it into sympy/ to "add coverage": every claim it checks is already
covered, exactly and symbolically for d = 2 and 3, by
sympy/projector_algebra_verification.py (181 checks, in the gate), which additionally
carries the source lints D1-D12' and the tilted-projector negative control.  Moving
this file would only add a "(no summary)" row.

Kept as dated evidence of the original numerical check.  If you want it gated,
port its claims into projector_algebra_verification.py rather than relocating it.
"""

"""
Numerical verification of every claim to be made about the generalized viscous
projector Pi for the porous Navier-Stokes paper (article_v2.tex).

Claims checked
==============
A. Algebraic (pointwise) properties of the canonical projectors
   I, S (sym), D (dev), DS (devsym), and the complement C := I - DS:
   A1. Each is idempotent and Frobenius-self-adjoint (orthogonal projection).
   A2. Non-expansiveness |Pi T| <= |T|.
   A3. I, S, D, DS all FIX deviatoric-symmetric tensors (range >= devsym);
       C does not.
   A4. Monotonicity: if ran(Pi) >= ran(Q) then |Pi T|^2 = |Q T|^2 + |(Pi-Q)T|^2.
   A5. Major symmetry P_{aibj} = P_{bjai} and induced K_ji^T = K_ij;
       the DS K matches the paper's displayed d=3 entries.
B. Integral (H_0^1) identities on [0,1]^d, d = 2, 3, random trig fields
   vanishing on the boundary:
   B1. int grad v : grad v^T = int (div v)^2            (the key IBP identity)
   B2. ||S grad v||^2  = 1/2||grad v||^2 + 1/2||div v||^2
   B3. ||DS grad v||^2 = 1/2||grad v||^2 + (1/2 - 1/d)||div v||^2
   B4. ||D grad v||^2  = ||grad v||^2 - (1/d)||div v||^2
   B5. ||C grad v||^2  = 1/2||grad v||^2 + (1/d - 1/2)||div v||^2
   B6. ||div v|| <= ||grad v||  (consequence of B1)
   B7. Korn bounds: ||grad v|| <= sqrt(2)||S grad v||, <= sqrt(2)||DS grad v||,
       <= sqrt(d/(d-1))||D grad v||; and for C: <= sqrt(2) (d=2), <= sqrt(3) (d=3).
   B8. Sharpness probes: d=2 DS identity is an equality with no div term;
       divergence-free fields approach the sqrt(2) constant for S and DS.
   B9. FAILURE of skew-only and spherical-only projectors: explicit kernel
       fields in H_0^1.
C. Fourier symbol:
   C1. T_Pi(k) v . v = |Pi(v (x) k)|^2 and the closed forms:
       T_S  = (|k|^2 I + k k^T)/2
       T_DS = (|k|^2 I + (1-2/d) k k^T)/2
       T_D  = |k|^2 I - (1/d) k k^T
       T_I  = |k|^2 I
   C2. Viscous symbol of -2 div(alpha nu Pi grad u) = (2 alpha nu/h^2) T_Pi(k0);
       for DS this equals (alpha nu/h^2)(|k0|^2 I + (1-2/d) k0 k0^T)  [paper
       eq:ftViscClosed], eigenvalues {1,...,1, 2-2/d} * alpha nu |k0|^2/h^2.
   C3. Family bounds for range >= devsym: (1/2)|k|^2|v|^2 <= T_Pi v.v <= |k|^2|v|^2.
   C4. Ellipticity constants c_Pi = min_{|v|=|k|=1}|Pi(v x k)|:
       I: 1;  S: 1/sqrt2;  D: sqrt(1-1/d);  DS: 1/sqrt2;  C: 1/sqrt2 (d=2);
       skew: 0 (kernel e1 x e1); spherical: 0 (kernel e1 x e2).
"""
import itertools
import numpy as np

rng = np.random.default_rng(20260729)
TOL = 1e-9

def frob(A, B=None):
    if B is None:
        B = A
    return float(np.sum(A * B))

# ---------- projector actions on d x d matrices ----------
def P_I(T):   return T
def P_S(T):   return 0.5 * (T + T.T)
def P_D(T):   d = T.shape[0]; return T - (np.trace(T) / d) * np.eye(d)
def P_DS(T):  return P_D(P_S(T))
def P_C(T):   return T - P_DS(T)                      # complement of DS
def P_skew(T): return 0.5 * (T - T.T)
def P_sph(T):  d = T.shape[0]; return (np.trace(T) / d) * np.eye(d)

NAMED = {"I": P_I, "S": P_S, "D": P_D, "DS": P_DS, "C": P_C,
         "skew": P_skew, "sph": P_sph}

def as_matrix(P, d):
    """Represent the projector as a d^2 x d^2 matrix acting on vec(T)."""
    M = np.zeros((d * d, d * d))
    for a in range(d):
        for b in range(d):
            E = np.zeros((d, d)); E[a, b] = 1.0
            M[:, a * d + b] = P(E).reshape(-1)
    return M

print("=" * 78)
print("A. ALGEBRAIC PROPERTIES")
print("=" * 78)
for d in (2, 3):
    for name, P in NAMED.items():
        M = as_matrix(P, d)
        idem = np.allclose(M @ M, M, atol=TOL)
        sa   = np.allclose(M, M.T, atol=TOL)          # Frobenius self-adjoint
        # non-expansive follows from orth. projection; spot-check anyway
        ok_ne = all(frob(P(T)) <= frob(T) + TOL
                    for T in (rng.standard_normal((d, d)) for _ in range(200)))
        # fixes devsym?
        fixes = True
        for _ in range(50):
            T = rng.standard_normal((d, d))
            Tds = P_DS(T)
            if not np.allclose(P(Tds), Tds, atol=1e-10):
                fixes = False
                break
        print(f"d={d} {name:>4}: idempotent={idem} self-adjoint={sa} "
              f"non-expansive={ok_ne} fixes-devsym={fixes}")
        assert idem and sa and ok_ne
        assert fixes == (name in ("I", "S", "D", "DS")), (d, name)

# A4 monotonicity for nested ranges: DS <= S, DS <= D, DS <= I, S <= I, D <= I
print("\nA4. Pointwise monotonicity |Pi T|^2 = |Q T|^2 + |(Pi - Q) T|^2:")
pairs = [("S", "DS"), ("D", "DS"), ("I", "DS"), ("I", "S"), ("I", "D")]
for d in (2, 3):
    for big, small in pairs:
        Pb, Ps = NAMED[big], NAMED[small]
        ok = True
        for _ in range(300):
            T = rng.standard_normal((d, d))
            lhs = frob(Pb(T))
            rhs = frob(Ps(T)) + frob(Pb(T) - Ps(T))
            ok &= abs(lhs - rhs) < 1e-10
        print(f"  d={d}: |{big} T|^2 = |{small} T|^2 + |({big}-{small})T|^2 : {ok}")
        assert ok

# A5 major symmetry P_{aibj} = P_{bjai}; K_ji^T = K_ij; DS d=3 entries
print("\nA5. Major symmetry and K_ij structure:")
def fourth_order(P, d):
    Pt = np.zeros((d, d, d, d))
    for b in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[b, j] = 1.0
            Pt[:, :, b, j] = P(E)
    return Pt  # Pt[a,i,b,j]: (Pi M)_{ai} = Pt[a,i,b,j] M[b,j]

for d in (2, 3):
    for name in ("I", "S", "D", "DS", "C"):
        Pt = fourth_order(NAMED[name], d)
        major = np.allclose(Pt, np.transpose(Pt, (2, 3, 0, 1)), atol=TOL)
        # K_ij as matrices in (a,b): [K_ij]_{ab} = 2 nu alpha Pt[a,i,b,j]; take 2 nu alpha = 1
        ok_sym = True
        for i in range(d):
            for j in range(d):
                Kij = Pt[:, i, :, j]
                Kji = Pt[:, j, :, i]
                ok_sym &= np.allclose(Kji.T, Kij, atol=TOL)
        print(f"  d={d} {name:>3}: P_(aibj)=P_(bjai): {major};  K_ji^T = K_ij: {ok_sym}")
        assert major and ok_sym

# DS, d=3: compare with the paper's displayed entries (units of nu*alpha):
# diagonal a=b: delta_ij + (1/3) delta_{ai} delta_{aj}
# off-diag a!=b: delta_{bi} delta_{aj} - (2/3) delta_{ai} delta_{bj}
d = 3
Pt = fourth_order(P_DS, d)
ok = True
for a in range(d):
    for b in range(d):
        for i in range(d):
            for j in range(d):
                val = 2.0 * Pt[a, i, b, j]  # K = 2 nu alpha P; paper K in units nu*alpha
                if a == b:
                    ref = (1.0 if i == j else 0.0) + (1.0 / 3.0) * (a == i) * (a == j)
                else:
                    ref = 1.0 * ((b == i) and (a == j)) - (2.0 / 3.0) * ((a == i) and (b == j))
                ok &= abs(val - ref) < 1e-12
print(f"  DS d=3 matches paper's displayed K_ij entries: {ok}")
assert ok

# ---------- B: integral identities on [0,1]^d ----------
print("\n" + "=" * 78)
print("B. INTEGRAL IDENTITIES AND KORN CONSTANTS (H_0^1 trig fields)")
print("=" * 78)

def make_field(d, n_terms=4, fmax=3, seed=None):
    """v_i(x) = sum_m c_{i,m} prod_j sin(pi n_j x_j); returns (v, grad) callables."""
    r = np.random.default_rng(seed)
    coeffs = r.standard_normal((d, n_terms))
    freqs = r.integers(1, fmax + 1, size=(d, n_terms, d))
    def v(x):  # x: (npts, d) -> (npts, d)
        out = np.zeros((x.shape[0], d))
        for i in range(d):
            for m in range(n_terms):
                term = np.ones(x.shape[0])
                for j in range(d):
                    term *= np.sin(np.pi * freqs[i, m, j] * x[:, j])
                out[:, i] += coeffs[i, m] * term
        return out
    def grad(x):  # -> (npts, d, d) with G[p, i, l] = d_l v_i
        G = np.zeros((x.shape[0], d, d))
        for i in range(d):
            for m in range(n_terms):
                for l in range(d):
                    term = np.ones(x.shape[0]) * coeffs[i, m]
                    for j in range(d):
                        f = freqs[i, m, j]
                        if j == l:
                            term *= np.pi * f * np.cos(np.pi * f * x[:, j])
                        else:
                            term *= np.sin(np.pi * f * x[:, j])
                    G[:, i, l] += term
        return G
    return v, grad

def gauss_grid(d, n):
    pts1, w1 = np.polynomial.legendre.leggauss(n)
    pts1 = 0.5 * (pts1 + 1.0); w1 = 0.5 * w1
    pts = np.array(list(itertools.product(pts1, repeat=d)))
    w = np.prod(np.array(list(itertools.product(w1, repeat=d))), axis=1)
    return pts, w

def integrals(G, w, Ps):
    """Return dict of integrated squared norms for each projector plus basics."""
    div = np.einsum("pii->p", G)
    GT = np.transpose(G, (0, 2, 1))
    out = {
        "grad2": float(np.sum(w * np.einsum("pil,pil->p", G, G))),
        "div2": float(np.sum(w * div * div)),
        "cross": float(np.sum(w * np.einsum("pil,pil->p", G, GT))),
    }
    for name, P in Ps.items():
        PG = np.stack([P(G[p]) for p in range(G.shape[0])])
        out[name] = float(np.sum(w * np.einsum("pil,pil->p", PG, PG)))
    return out

for d, nq in ((2, 36), (3, 18)):
    pts, w = gauss_grid(d, nq)
    Ps = {k: NAMED[k] for k in ("S", "DS", "D", "C")}
    print(f"\n-- d = {d} (Gauss {nq}^d) --")
    worst = {k: 0.0 for k in
             ("B1", "B2", "B3", "B4", "B5")}
    korn_max = {"S": 0.0, "DS": 0.0, "D": 0.0, "C": 0.0}
    for trial in range(6):
        _, grad = make_field(d, seed=1000 + trial)
        G = grad(pts)
        I = integrals(G, w, Ps)
        worst["B1"] = max(worst["B1"], abs(I["cross"] - I["div2"]) / max(I["grad2"], 1))
        worst["B2"] = max(worst["B2"], abs(I["S"] - 0.5 * I["grad2"] - 0.5 * I["div2"]) / I["grad2"])
        worst["B3"] = max(worst["B3"], abs(I["DS"] - 0.5 * I["grad2"] - (0.5 - 1.0 / d) * I["div2"]) / I["grad2"])
        worst["B4"] = max(worst["B4"], abs(I["D"] - I["grad2"] + (1.0 / d) * I["div2"]) / I["grad2"])
        worst["B5"] = max(worst["B5"], abs(I["C"] - 0.5 * I["grad2"] - (1.0 / d - 0.5) * I["div2"]) / I["grad2"])
        assert I["div2"] <= I["grad2"] * (1 + 1e-10), "B6 violated"
        for k in korn_max:
            korn_max[k] = max(korn_max[k], np.sqrt(I["grad2"] / I[k]))
    for k, v in worst.items():
        print(f"  {k} residual (rel): {v:.2e}")
        assert v < 1e-8
    print(f"  B6 ||div|| <= ||grad||: OK")
    bounds = {"S": np.sqrt(2), "DS": np.sqrt(2),
              "D": np.sqrt(d / (d - 1)),
              "C": np.sqrt(2) if d == 2 else np.sqrt(3)}
    for k in korn_max:
        print(f"  B7 Korn ratio ||grad||/||{k}grad|| observed <= {korn_max[k]:.4f}"
              f"   (claimed bound {bounds[k]:.4f})")
        assert korn_max[k] <= bounds[k] + 1e-8

# B8 sharpness: d=2, DS identity exact w/o div; div-free fields approach sqrt2
d = 2
pts, w = gauss_grid(d, 40)
# stream-function field: v = curl psi = (d_y psi, -d_x psi), psi = sin(pi x)^2 sin(pi y)^2-ish
def sf_field(pts):
    x, y = pts[:, 0], pts[:, 1]
    # psi = sin(pi x) sin(pi y) * sin(2 pi x) -> v in H_0^1? v = (psi_y, -psi_x)
    # choose psi = sin(pi x)^2 sin(pi y)^2 so that grad psi = 0 on boundary -> v in H_0^1
    sx, cx = np.sin(np.pi * x), np.cos(np.pi * x)
    sy, cy = np.sin(np.pi * y), np.cos(np.pi * y)
    G = np.zeros((pts.shape[0], 2, 2))
    # v1 = d_y psi = 2 pi sx^2 sy cy ; v2 = -d_x psi = -2 pi sx cx sy^2
    # gradients:
    G[:, 0, 0] = 4 * np.pi**2 * sx * cx * sy * cy            # d_x v1
    G[:, 0, 1] = 2 * np.pi**2 * sx**2 * (cy**2 - sy**2)      # d_y v1
    G[:, 1, 0] = -2 * np.pi**2 * (cx**2 - sx**2) * sy**2     # d_x v2
    G[:, 1, 1] = -4 * np.pi**2 * sx * cx * sy * cy           # d_y v2
    return G
G = sf_field(pts)
I = integrals(G, w, {k: NAMED[k] for k in ("S", "DS")})
print(f"\nB8 (d=2, divergence-free stream field): div^2 integral = {I['div2']:.2e}")
print(f"    ||grad||/||S grad||  = {np.sqrt(I['grad2']/I['S']):.6f}  (sqrt2 = {np.sqrt(2):.6f})")
print(f"    ||grad||/||DS grad|| = {np.sqrt(I['grad2']/I['DS']):.6f}")
assert abs(np.sqrt(I['grad2'] / I['S']) - np.sqrt(2)) < 1e-6
assert abs(np.sqrt(I['grad2'] / I['DS']) - np.sqrt(2)) < 1e-6

# B9 failure fields
print("\nB9 failure of skew-only and spherical-only:")
# skew: v = grad phi with phi = (sin(pi x) sin(pi y))^2  => v in H_0^1, skew grad v = 0
x, y = pts[:, 0], pts[:, 1]
sx, cx = np.sin(np.pi * x), np.cos(np.pi * x)
sy, cy = np.sin(np.pi * y), np.cos(np.pi * y)
G = np.zeros((pts.shape[0], 2, 2))  # grad v = Hessian(phi), symmetric
G[:, 0, 0] = 2 * np.pi**2 * (cx**2 - sx**2) * sy**2
G[:, 0, 1] = 4 * np.pi**2 * sx * cx * sy * cy
G[:, 1, 0] = G[:, 0, 1]
G[:, 1, 1] = 2 * np.pi**2 * sx**2 * (cy**2 - sy**2)
I = integrals(G, w, {"skew": P_skew})
print(f"    grad phi field: ||grad v||^2 = {I['grad2']:.4f}, ||skew grad v||^2 = {I['skew']:.2e}")
assert I["skew"] < 1e-16 * I["grad2"] + 1e-12
G = sf_field(pts)  # divergence-free
I = integrals(G, w, {"sph": P_sph})
print(f"    div-free field: ||grad v||^2 = {I['grad2']:.4f}, ||sph grad v||^2 = {I['sph']:.2e}")
assert I["sph"] < 1e-16 * I["grad2"] + 1e-12

# ---------- C: Fourier symbols ----------
print("\n" + "=" * 78)
print("C. FOURIER SYMBOLS")
print("=" * 78)
def T_of(P, k):
    d = k.size
    T = np.zeros((d, d))
    for a in range(d):
        ea = np.zeros(d); ea[a] = 1.0
        for b in range(d):
            eb = np.zeros(d); eb[b] = 1.0
            T[a, b] = frob(P(np.outer(ea, k)), P(np.outer(eb, k)))
    return T

for d in (2, 3):
    ok_forms, ok_bounds = True, True
    for _ in range(200):
        k = rng.standard_normal(d)
        v = rng.standard_normal(d)
        k2 = k @ k
        TS  = T_of(P_S, k);  TS_ref  = 0.5 * (k2 * np.eye(d) + np.outer(k, k))
        TDS = T_of(P_DS, k); TDS_ref = 0.5 * (k2 * np.eye(d) + (1 - 2.0 / d) * np.outer(k, k))
        TD  = T_of(P_D, k);  TD_ref  = k2 * np.eye(d) - (1.0 / d) * np.outer(k, k)
        TI  = T_of(P_I, k);  TI_ref  = k2 * np.eye(d)
        ok_forms &= all(np.allclose(a, b, atol=1e-10) for a, b in
                        ((TS, TS_ref), (TDS, TDS_ref), (TD, TD_ref), (TI, TI_ref)))
        # quadratic-form identity and family bounds
        for P in (P_I, P_S, P_D, P_DS):
            q = v @ T_of(P, k) @ v
            ok_forms &= abs(q - frob(P(np.outer(v, k)))) < 1e-10
            ok_bounds &= (0.5 * k2 * (v @ v) - 1e-10 <= q <= k2 * (v @ v) + 1e-10)
    print(f"d={d}: closed forms T_S,T_DS,T_D,T_I and q = |Pi(v x k)|^2 : {ok_forms}")
    print(f"d={d}: family bounds (1/2)|k|^2|v|^2 <= v.T_Pi.v <= |k|^2|v|^2 : {ok_bounds}")
    assert ok_forms and ok_bounds
    # DS eigenvalues of the *viscous operator symbol* 2*T_DS (units alpha nu /h^2, |k0|=1)
    k = rng.standard_normal(d); k /= np.linalg.norm(k)
    eig = np.linalg.eigvalsh(2 * T_of(P_DS, k))
    print(f"d={d}: eig(2 T_DS) = {np.round(np.sort(eig), 6)}  "
          f"(claimed {{1 x{d-1}, {2-2.0/d:.4f}}})")
    assert np.allclose(np.sort(eig), np.array([1.0] * (d - 1) + [2 - 2.0 / d]), atol=1e-9)

# C4 ellipticity constants by dense sampling of unit v, k
print("\nC4. c_Pi = min |Pi(v x k)| over unit v,k (dense sampling):")
def c_min(P, d, N=200):
    best = np.inf
    for _ in range(N * 50):
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        k = rng.standard_normal(d); k /= np.linalg.norm(k)
        best = min(best, np.sqrt(frob(P(np.outer(v, k)))))
    return best

claims = {2: {"I": 1.0, "S": 1/np.sqrt(2), "D": np.sqrt(1 - 1/2), "DS": 1/np.sqrt(2),
              "C": 1/np.sqrt(2), "skew": 0.0, "sph": 0.0},
          3: {"I": 1.0, "S": 1/np.sqrt(2), "D": np.sqrt(1 - 1/3), "DS": 1/np.sqrt(2),
              "skew": 0.0, "sph": 0.0}}
for d in (2, 3):
    for name, cl in claims[d].items():
        c = c_min(NAMED[name], d)
        print(f"  d={d} {name:>4}: sampled c = {c:.4f}   claimed = {cl:.4f}")
        assert c >= cl - 5e-3, (d, name, c, cl)
        # for the zero cases exhibit the kernel rank-one:
        if cl == 0.0:
            if name == "skew":
                assert frob(P_skew(np.outer(np.eye(d)[0], np.eye(d)[0]))) < 1e-14
            if name == "sph":
                assert frob(P_sph(np.outer(np.eye(d)[0], np.eye(d)[1]))) < 1e-14

print("\nALL CHECKS PASSED.")
