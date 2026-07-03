# ckpt_run.py — resumable driver: loads state, advances the solve for <=BUDGET s, saves.
# Usage: python3 ckpt_run.py <case_name>
import numpy as np, json, os, sys, time
from scipy.sparse.linalg import splu
from assemble import Problem

BUDGET = 235.0
CASES = {
    # name: (mesh, kwargs)
    "P1_16":   ((16,16,4), dict(kv=1, kp=1, qorder=6)),
    "P2_8":    ((8,8,2),   dict(kv=2, kp=2, qorder=7)),
    "P2_12":   ((12,12,3), dict(kv=2, kp=2, qorder=7)),
    "P2_16":   ((16,16,4), dict(kv=2, kp=2, qorder=7)),
    "P2_8_dropR":  ((8,8,2),   dict(kv=2, kp=2, qorder=7, drop_visc_in_R=True)),
    "P2_12_dropR": ((12,12,3), dict(kv=2, kp=2, qorder=7, drop_visc_in_R=True)),
    "P2_8_c1x4":   ((8,8,2),   dict(kv=2, kp=2, qorder=7, c1_mult=4.0)),
    "P2_12_c1x4":  ((12,12,3), dict(kv=2, kp=2, qorder=7, c1_mult=4.0)),
    "TH_8":    ((8,8,2),   dict(kv=2, kp=1, qorder=7, stab_on=False)),
    "TH_12":   ((12,12,3), dict(kv=2, kp=1, qorder=7, stab_on=False)),
    "TH_16":   ((16,16,4), dict(kv=2, kp=1, qorder=7, stab_on=False)),
    "P2_12_q9":    ((12,12,3), dict(kv=2, kp=2, qorder=9)),
    "P2_8_dropBoth": ((8,8,2), dict(kv=2, kp=2, qorder=7, drop_visc_in_R=True, drop_visc_in_adj=True)),
    "P2_8_c1x1p5": ((8,8,2), dict(kv=2, kp=2, qorder=7, c1_mult=1.5)),
    "P2_8_c1x2":   ((8,8,2), dict(kv=2, kp=2, qorder=7, c1_mult=2.0)),
    "P2_8_c1x3":   ((8,8,2), dict(kv=2, kp=2, qorder=7, c1_mult=3.0)),
    "P2_12_c1x4_q9": ((12,12,3), dict(kv=2, kp=2, qorder=9, c1_mult=4.0)),
    "P2_16_c1x4":  ((16,16,4), dict(kv=2, kp=2, qorder=7, c1_mult=4.0)),
}

def main():
    name = sys.argv[1]
    mesh, kw = CASES[name]
    fstate = f"state_{name}.npz"
    fres = f"result_{name}.json"
    if os.path.exists(fres):
        print(f"[{name}] already done:", open(fres).read()); return
    t_start = time.time()
    pb = Problem(n=mesh, **kw)
    if kw.get('qorder', 7) >= 9:
        pb.chunk = 32
    if os.path.exists(fstate):
        z = np.load(fstate)
        x = z["x"]; p_prev = z["p_prev"]; ps = int(z["ps"]); it = int(z["it"])
        hist = list(z["hist"]); r0 = float(z["r0"])
        phase = str(z["phase"]) if "phase" in z else "solve"
        print(f"[{name}] resumed at pass {ps} it {it} phase {phase}")
        if phase == "errors":
            err = pb.errors(x)
            out = dict(name=name, mesh=mesh, its=len(hist), R0=hist[0], Rend=hist[-1],
                       **{k: float(v) for k, v in err.items()},
                       ratio=float(err['el2u']/err['il2u']))
            json.dump(out, open(fres, "w"), indent=1)
            os.remove(fstate)
            print(f"[{name}] DONE:", json.dumps(out)); return
    else:
        x = pb.interpolant(); p_prev = x[3*pb.Nu:].copy()
        ps, it, hist, r0 = 0, 0, [], None
        print(f"[{name}] fresh start, ndof={pb.ndof}")
    PASSES, MAXIT, TOL = 3, 8, 1e-10
    from scipy.optimize import newton_krylov
    from scipy.sparse.linalg import LinearOperator

    class Budget(Exception):
        def __init__(self, xf): self.xf = xf

    cycle_est = 0.0
    while ps < PASSES:
        # ---- stage 1: a few Picard steps (cheap contraction from the interpolant) ----
        while it < MAXIT:
            if time.time() - t_start > BUDGET - cycle_est:
                np.savez(fstate, x=x, p_prev=p_prev, ps=ps, it=it, hist=np.array(hist), r0=r0 or 0.0)
                print(f"[{name}] checkpointed at pass {ps} it {it}, last R={hist[-1] if hist else None}")
                return
            R = pb.residual(x, p_prev_vec=p_prev)
            rn = np.linalg.norm(R[pb.free]); hist.append(rn)
            if r0 is None: r0 = rn
            print(f"    [{name}] pass {ps} picard it {it}: R={rn:.3e}", flush=True)
            if rn < TOL*max(1.0, r0) or rn < 1e-12:
                it = MAXIT; break
            if it >= 3 and rn < 3e-3*r0:
                break   # hand off to JFNK
            tcy = time.time()
            J = pb.jacobian(x)
            dx = splu(J[pb.free][:, pb.free].tocsc()).solve(-R[pb.free])
            x[pb.free] += dx
            cycle_est = 1.3*(time.time() - tcy)
            it += 1
        # ---- stage 2: JFNK (exact Newton on the frozen-p_prev residual), Picard-J preconditioned ----
        R = pb.residual(x, p_prev_vec=p_prev)
        rn = np.linalg.norm(R[pb.free])
        if rn > TOL*max(1.0, r0) and rn > 1e-12:
            print(f"    [{name}] pass {ps} JFNK start: R={rn:.3e}", flush=True)
            Jp = pb.jacobian(x)
            lu = splu(Jp[pb.free][:, pb.free].tocsc())
            M = LinearOperator((len(pb.free), len(pb.free)), matvec=lu.solve)
            def F(xf):
                if time.time() - t_start > BUDGET:
                    raise Budget(xf)
                xx = x.copy(); xx[pb.free] = xf
                return pb.residual(xx, p_prev_vec=p_prev)[pb.free]
            try:
                xf = newton_krylov(F, x[pb.free].copy(), method="lgmres", inner_M=M,
                                   f_tol=max(TOL*max(1.0, r0), 1e-12), maxiter=25, verbose=True)
                x[pb.free] = xf
            except Budget as b:
                x[pb.free] = b.xf
                np.savez(fstate, x=x, p_prev=p_prev, ps=ps, it=it, hist=np.array(hist), r0=r0)
                print(f"[{name}] checkpointed inside JFNK, pass {ps}")
                return
            except Exception as e:
                print(f"    [{name}] JFNK stopped: {e}")
            R = pb.residual(x, p_prev_vec=p_prev)
            hist.append(np.linalg.norm(R[pb.free]))
            print(f"    [{name}] pass {ps} after JFNK: R={hist[-1]:.3e}", flush=True)
        m = pb._pressure_mean_offset(x); x[3*pb.Nu:] -= m
        drift = np.linalg.norm(x[3*pb.Nu:] - p_prev)/max(np.linalg.norm(x[3*pb.Nu:]), 1e-30)
        p_prev = x[3*pb.Nu:].copy()
        ps += 1; it = 0
        if drift < 1e-9: break
    np.savez(fstate, x=x, p_prev=p_prev, ps=ps, it=it, hist=np.array(hist), r0=r0, phase="errors")
    if time.time() - t_start > BUDGET*0.5:
        print(f"[{name}] converged; errors deferred to next call"); return
    err = pb.errors(x)
    out = dict(name=name, mesh=mesh, its=len(hist), R0=hist[0], Rend=hist[-1],
               **{k: float(v) for k, v in err.items()},
               ratio=float(err['el2u']/err['il2u']))
    json.dump(out, open(fres, "w"), indent=1)
    if os.path.exists(fstate): os.remove(fstate)
    print(f"[{name}] DONE:", json.dumps(out))

if __name__ == "__main__":
    main()
