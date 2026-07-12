#!/usr/bin/env python3
# test/extended/ManufacturedSolutions3D/compare_c1study.py
# =============================================================================================
# Cross-strategy comparison for the element-aware c1 study (run_c1study_nested_red). Reads every
# results/k<kv>/TET/nested_red_<strategy>/convergence3d_results.json (plus the baseline nested_red/
# if present) and tabulates, per (kv, method), the per-segment convergence rates + the finest-segment
# rate + L2/finest success flag + the effective c1 multiplier. Answers: does clearing the irregular
# mesh's worst-element coercivity floor (element_c1.jl) recover the k=2 rate the Kuhn-calibrated ×4
# leaves on the table? This is ANALYSIS (a table), not a competing plot — use plot_convergence3d.py
# <sequence> for the standard per-strategy convergence figures.
# =============================================================================================
import glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

def rate(e0, e1, h0, h1):
    if not (e0 and e1 and e0 > 0 and e1 > 0 and h0 != h1):
        return float("nan")
    return math.log(e0 / e1) / math.log(h0 / h1)

def load():
    rows = {}   # (kv, seq, method) -> block
    for jp in sorted(glob.glob(os.path.join(ROOT, "k*", "TET", "nested_red*", "convergence3d_results.json"))):
        seq = os.path.basename(os.path.dirname(jp))
        try:
            data = json.load(open(jp))
        except Exception as e:
            print(f"  [skip {jp}: {e}]"); continue
        for blk in data:
            rows[(blk["kv"], seq, blk["method"])] = blk
    return rows

def seg_rates(blk, key):
    hs = blk["hs"]; es = blk[key]
    return [rate(es[i], es[i+1], hs[i], hs[i+1]) for i in range(len(hs)-1)]

def opt(kv, key):
    return {"l2u": kv+1, "h1u": kv, "l2p": kv}[key]

def main():
    rows = load()
    if not rows:
        print(f"No nested_red* results under {ROOT} yet."); return
    seqs = sorted({k[1] for k in rows})
    print(f"\n=== element-aware c1 study — nested_red (irregular), root={ROOT} ===")
    print("strategies found:", ", ".join(seqs))
    for kv in (2, 1):
        methods = sorted({k[2] for k in rows if k[0] == kv})
        if not methods:
            continue
        print(f"\n{'='*96}\n  P{kv}   (optimal: L2u={opt(kv,'l2u')}  H1u={opt(kv,'h1u')}  L2p={opt(kv,'l2p')})\n{'='*96}")
        for method in methods:
            print(f"\n  --- {method} ---")
            hdr = f"  {'strategy':22s} {'nlvl':>4s} {'finest h':>9s} {'c1x(fin)':>9s} {'succ(fin)':>9s} "\
                  f"{'L2u_fin':>8s} {'H1u_fin':>8s} {'L2p_fin':>8s}  | all L2u seg rates"
            print(hdr); print("  " + "-"*len(hdr))
            for seq in seqs:
                blk = rows.get((kv, seq, method))
                if blk is None:
                    continue
                hs = blk["hs"]; nlvl = len(hs)
                lv = blk.get("levels", [])
                succ_fin = lv[-1].get("success") if lv else None
                c1x_fin = lv[-1].get("c1_mult_eff", blk.get("c1_mult")) if lv else blk.get("c1_mult")
                r_l2u = seg_rates(blk, "l2u"); r_h1u = seg_rates(blk, "h1u"); r_l2p = seg_rates(blk, "l2p")
                fin = lambda r: (r[-1] if r else float("nan"))
                segstr = " ".join(f"{x:5.2f}" for x in r_l2u)
                c1x_str = f"{c1x_fin:.2f}" if isinstance(c1x_fin, (int, float)) else str(c1x_fin)
                print(f"  {seq:22s} {nlvl:4d} {hs[-1]:9.4g} {c1x_str:>9s} {str(succ_fin):>9s} "
                      f"{fin(r_l2u):8.2f} {fin(r_h1u):8.2f} {fin(r_l2p):8.2f}  | {segstr}")
    # verdict helper: baseline vs best element_aware on the k=2 finest L2u rate
    print(f"\n{'='*96}\n  VERDICT (k=2 finest-segment L2u rate; optimal 3.0)\n{'='*96}")
    for method in ("ASGS", "OSGS"):
        base = rows.get((2, "nested_red_c1fixed4", method)) or rows.get((2, "nested_red", method))
        if base is None:
            print(f"  {method}: no baseline block"); continue
        br = seg_rates(base, "l2u"); bfin = br[-1] if br else float("nan")
        best_seq, best = None, -1e9
        for seq in seqs:
            if "c1aware" not in seq:
                continue
            blk = rows.get((2, seq, method))
            if blk is None:
                continue
            r = seg_rates(blk, "l2u"); v = r[-1] if r else float("nan")
            if v == v and v > best:
                best, best_seq = v, seq
        print(f"  {method}: baseline(x4) L2u_fin={bfin:.2f}   best element_aware={best:.2f} ({best_seq})   "
              f"Δ={best-bfin:+.2f}" if best_seq else f"  {method}: baseline L2u_fin={bfin:.2f}; no element_aware yet")

if __name__ == "__main__":
    main()
