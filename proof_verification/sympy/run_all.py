#!/usr/bin/env python3
# Run every *_verification.py in this folder and report a combined summary.
# Usage:  python3 "run_all.py"      (needs sympy in the active interpreter)
import subprocess, sys, glob, os, re

here = os.path.dirname(os.path.abspath(__file__))
scripts = sorted(glob.glob(os.path.join(here, "*_verification.py")))
print(f"Running {len(scripts)} verification scripts with {sys.executable}\n" + "=" * 70)
total_ok = True
grand = []
for s in scripts:
    name = os.path.basename(s)
    r = subprocess.run([sys.executable, s], capture_output=True, text=True)
    m = re.search(r"SUMMARY:\s*(\d+)/(\d+)", r.stdout)
    summary = m.group(0) if m else "(no summary)"
    ok = (r.returncode == 0)
    total_ok &= ok
    grand.append((name, summary, "OK" if ok else "FAIL"))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:42s} {summary}")
    if not ok:
        print(r.stdout[-1500:]); print(r.stderr[-800:])
print("=" * 70)
npass = sum(int(re.search(r'(\d+)/', s).group(1)) for _, s, _ in grand if 'SUMMARY' in s)
ntot = sum(int(re.search(r'/(\d+)', s).group(1)) for _, s, _ in grand if 'SUMMARY' in s)
print(f"GRAND TOTAL: {npass}/{ntot} symbolic checks across {len(scripts)} scripts "
      f"-- {'ALL PASS' if total_ok else 'SOME FAILED'}")
sys.exit(0 if total_ok else 1)
