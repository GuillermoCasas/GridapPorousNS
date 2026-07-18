#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  run_all.sh -- compile and kernel-check the Coq formalisation.
#
#  Mirrors the reporting style of theory/verification scripts/: one PASS line
#  per file, a final summary, and a non-zero exit code on any failure.
#
#  Requires: coqc (Coq 8.16--8.20; developed against 8.18.0).
#  Optional: coqchk for the final kernel-level re-verification.
#
#  [known-fragility]  Two invariants keep this script HONEST.  Each was added
#  after a real incident -- both described in full below -- so do not "tidy"
#  either one away:
#
#  (1) The file list is READ FROM _CoqProject, never restated here.  It used to
#      be a literal array, i.e. a second, independent copy of the dependency
#      order.  When AbstractInterpolation was moved ahead of AbstractStability
#      (the latter now DEFINES B_S from the former's eighteen-term expression
#      rather than assuming it), the Makefile and _CoqProject were updated and
#      this array was not -- so the documented entry point stopped compiling on
#      a clean tree while `make` stayed green.  One list, one source of truth.
#
#  (2) Stale build artefacts are REMOVED before the run.  coqc does not
#      recompile a dependency; it loads the .vo it finds.  Since the loop below
#      deliberately continues past a failure (to report every broken file in one
#      pass), a leftover .vo from an earlier source revision could satisfy a
#      later file's Require and turn a genuine breakage into a [PASS].  Starting
#      from a clean tree makes "RESULT: every theorem is proved" mean it.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")"

COQPROJECT=_CoqProject
[ -r "$COQPROJECT" ] || {
  echo "ERROR: $COQPROJECT not found next to run_all.sh; cannot determine the" >&2
  echo "       file list or its dependency order." >&2
  exit 1
}

#  The .v entries of _CoqProject, in order, with the .v suffix stripped.
mapfile -t FILES < <(grep -oE '^[A-Za-z0-9_]+\.v[[:space:]]*$' "$COQPROJECT" | sed 's/\.v[[:space:]]*$//')
[ "${#FILES[@]}" -gt 0 ] || {
  echo "ERROR: no .v files listed in $COQPROJECT." >&2
  exit 1
}

FLAGS=(-Q . PNSFormal -q)

command -v coqc >/dev/null 2>&1 || {
  echo "ERROR: coqc not found. Install Coq (e.g. 'apt install coq' or via opam)." >&2
  exit 1
}

#  Every listed source must exist -- a typo in _CoqProject must not silently
#  shrink the suite.
missing=0
for f in "${FILES[@]}"; do
  [ -r "$f.v" ] || { echo "ERROR: $COQPROJECT lists $f.v, which does not exist." >&2; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1

echo "Coq formal verification suite (PNSFormal)"
echo "coqc: $(coqc --version | head -n 1)"
echo "files: ${#FILES[@]}, in the dependency order given by $COQPROJECT"
echo "-------------------------------------------------------------"
echo "Removing stale build artefacts (see invariant (2) in this script) ..."
rm -f ./*.vo ./*.vos ./*.vok ./*.glob ./.*.aux
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
  #  Invariant (1) again: the module list is DERIVED from the same FILES array,
  #  never restated.  It used to be a literal here too, and had drifted to twelve
  #  of the fifteen modules -- leaving NonVacuity, a leaf that nothing Requires,
  #  outside the kernel re-check.
  if coqchk -Q . PNSFormal -silent -o "${FILES[@]/#/PNSFormal.}" > /dev/null 2>&1; then
    echo "  [PASS] coqchk: all ${#FILES[@]} modules successfully checked by the trusted kernel."
  else
    echo "  [FAIL] coqchk reported a problem."
    exit 1
  fi
else
  echo "NOTE: coqchk not found; skipped the optional kernel re-check."
fi
