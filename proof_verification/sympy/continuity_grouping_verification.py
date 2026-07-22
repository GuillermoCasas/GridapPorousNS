#!/usr/bin/env python3
# =============================================================================
# continuity_grouping_verification.py
#
# Closes a coverage gap the 2026-07-22 external audit exposed (latest_audit,
# issue M14 / full audit sect. 6.3): the ASGS continuity-lemma proof
# (continuity_appendix.tex, Steps 5-8) collects its term bounds into a display
# whose left-hand side must ACCOUNT for each proof term exactly once. The suite
# had no term-accounting check for that appendix (assembly_consistency_
# verification.py does the analogous named-vs-defined bookkeeping only for the
# Appendix-A elemental matrices), so a collected-display LHS that double-counts
# a subterm passed unflagged -- exactly the class of the 2026-07-12 assembly
# erratum and the 2026-07-21 robustness-isolation gap (see docs/lessons_learned.md).
#
# The specific defect: Step 5 splits  T13 = T13^c + R  (convective part + rest);
# Step 6 forms  N := R + T2 + T4 + T11 ; the four terms of this block therefore
# sum to  T13 + T2 + T4 + T11 = T13^c + N .  The collected display eq:groupstep
# originally wrote its LHS as |T13| + |N|, which -- since T13 = T13^c + R and R
# is inside N -- DOUBLE-COUNTS R. The correct (and now-printed) LHS is
# |T13^c| + |N|. The RHS and the final eq:assembly are unaffected either way.
#
# This is a term-bookkeeping invariant, not a floating-point identity: each atom
# of the block must appear in the collected accounting exactly once. We encode
# the decomposition symbolically and assert (i) the grouping identity, (ii) the
# corrected LHS accounts every atom once, and (iii) the pre-fix LHS |T13|+|N|
# double-counts R -- the discriminating negative.
#
# Run:  python3 continuity_grouping_verification.py     (requires sympy)
# =============================================================================
import sympy as sp

results = []
def check(name, ok):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    print(f"  [{tag}] {name}")
    return ok

print("=" * 72)
print("Continuity lemma: T13/N term-accounting (eq:groupstep, Steps 5-8)")
print("=" * 72)

# Atomic terms of the convective-pressure block (as formal generators).
Tc, R, T2, T4, T11 = sp.symbols('Tconv R T2 T4 T11')

# Step 5 / Step 6 decomposition (continuity_appendix.tex:602-621).
T13 = Tc + R                 # Step 5 split
N   = R + T2 + T4 + T11      # Step 6 group (absorbs R)

# The four terms of the block whose collected bound is eq:groupstep.
block_total = T13 + T2 + T4 + T11

# (G1) grouping identity: the block equals T13^c + N (R absorbed exactly once).
check("(G1) T13 + T2 + T4 + T11 == T13^c + N  (grouping identity)",
      sp.expand(block_total - (Tc + N)) == 0)

# (G2) corrected LHS |T13^c| + |N| accounts every atom exactly once.
corrected_lhs = Tc + N       # the multiset behind |T13^c| + |N|
p_corr = sp.Poly(corrected_lhs, Tc, R, T2, T4, T11)
coeffs_ok = all(
    p_corr.coeff_monomial(atom) == 1 for atom in (Tc, R, T2, T4, T11)
)
check("(G2) |T13^c|+|N| counts each atom {T13^c,R,T2,T4,T11} exactly once", coeffs_ok)
check("(G2') corrected LHS equals the block total (no term dropped or added)",
      sp.expand(corrected_lhs - block_total) == 0)

# (G3) DISCRIMINATING: the pre-fix LHS |T13| + |N| double-counts R.
buggy_lhs = T13 + N          # the multiset behind the original |T13| + |N|
p_bug = sp.Poly(buggy_lhs, Tc, R, T2, T4, T11)
check("(G3) pre-fix |T13|+|N| double-counts R  (coeff(R) == 2, not 1)",
      p_bug.coeff_monomial(R) == 2)
check("(G3') pre-fix |T13|+|N| != the block total (it overcounts by R)",
      sp.expand(buggy_lhs - block_total) == R)

# =============================================================================
print("=" * 72)
n_fail = sum(1 for tag, _ in results if tag == "FAIL")
print(f"SUMMARY: {len(results) - n_fail}/{len(results)}")
print("=" * 72)
raise SystemExit(1 if n_fail else 0)
