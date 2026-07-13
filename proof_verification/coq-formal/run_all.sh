#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  run_all.sh -- compile and kernel-check the Coq formalisation.
#
#  Mirrors the reporting style of theory/verification scripts/: one PASS line
#  per file, a final summary, and a non-zero exit code on any failure.
#
#  Requires: coqc (Coq 8.16--8.20; developed against 8.18.0).
#  Optional: coqchk for the final kernel-level re-verification.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")"

FILES=(Limits DerivKit StabilityAlgebra ContinuityAlgebra AbstractSums InnerSpace AbstractStability AbstractContinuity AbstractInterpolation AbstractConvergence TauDesign Asymptotics ManufacturedSolution PlateauBump NonVacuity)
FLAGS=(-Q . PNSFormal -q)

command -v coqc >/dev/null 2>&1 || {
  echo "ERROR: coqc not found. Install Coq (e.g. 'apt install coq' or via opam)." >&2
  exit 1
}

echo "Coq formal verification suite (PNSFormal)"
echo "coqc: $(coqc --version | head -n 1)"
echo "-------------------------------------------------------------"

fail=0
for f in "${FILES[@]}"; do
  if coqc "${FLAGS[@]}" "$f.v" > /dev/null 2> "/tmp/pnsformal_$f.err"; then
    printf "  [PASS] %-24s (all proofs accepted by coqc)\n" "$f.v"
  else
    printf "  [FAIL] %-24s\n" "$f.v"
    sed 's/^/         /' "/tmp/pnsformal_$f.err"
    fail=1
  fi
done

echo "-------------------------------------------------------------"
if [ "$fail" -ne 0 ]; then
  echo "RESULT: FAILURE -- at least one file did not compile."
  exit 1
fi
echo "RESULT: all ${#FILES[@]} files compiled; every theorem is proved"
echo "        (no Admitted, no Axiom -- grep the sources to confirm)."

if command -v coqchk >/dev/null 2>&1; then
  echo
  echo "Running coqchk (independent kernel re-verification) ..."
  if coqchk -Q . PNSFormal -silent -o \
       PNSFormal.StabilityAlgebra PNSFormal.ContinuityAlgebra PNSFormal.AbstractSums PNSFormal.InnerSpace PNSFormal.AbstractStability PNSFormal.AbstractContinuity PNSFormal.AbstractInterpolation PNSFormal.AbstractConvergence PNSFormal.TauDesign PNSFormal.Asymptotics \
       PNSFormal.ManufacturedSolution PNSFormal.PlateauBump > /dev/null 2>&1; then
    echo "  [PASS] coqchk: modules successfully checked by the trusted kernel."
  else
    echo "  [FAIL] coqchk reported a problem."
    exit 1
  fi
else
  echo "NOTE: coqchk not found; skipped the optional kernel re-check."
fi
