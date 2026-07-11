# Lean 4 roadmap for the residual trusted base

## What this document is

`AbstractStability.v` and `AbstractContinuity.v` prove `lemma:Stability` and
`lemma:Continuity` *in full* inside Coq, from a named, quantitative trusted
base. This document classifies that trusted base, states each provable item
as a Lean 4 target (see the companion skeleton `PorousNSToolbox.lean`), and
gives an honest feasibility assessment with strategy notes, for iteration
with a proof-assistant loop.

Read the labels carefully. The skeleton **does not compile as shipped** and
contains **zero proofs** — every theorem ends in `sorry`. Its value is the
task decomposition and the pinned semantics of each statement. The first
loop iterations should only make the statements elaborate against current
Mathlib; proving comes after.

## Classification of the trusted base

The hypotheses of the two abstract files fall into three classes.

**Class D — assumptions on the data and mesh (not formalisation targets).**
`H_jump` (the jump condition H:jump), `H_eps` (eq:epscond), `H_mult1/2`
(bounded face multiplicity, part of H:mesh), and the positivity/regularity
of the coefficients. These are hypotheses of the paper's theorems in any
formal system; there is nothing to prove.

**Class G — Green-type identities.** `HBS` (the Galerkin-testing/adjoint
identity of the stability proof), `H_skew`, `H_ibp_vp`, `H_ibp_qu`,
`H_elem_conv_ibp`, `H_elem_p_ibp`, and the two assembly identities
`H_assemble_p/c`. All reduce to: (a) the divergence theorem on a single
simplex for fields of the form (elementwise polynomial) x (W^{1,infty}
weight), and (b) bookkeeping over elements and faces using the continuity
of the integrands across interior faces.

**Class I — inverse-type estimates.** `S3` and the eight `Hw_*` hypotheses
(the weighted inverse estimates of lem:winv), plus, inside the face bounds
`H_face_p/c`: the L-infinity inverse estimate eq:inverse, the face-measure
bound meas <= C h^(d-1), and Hoelder on the face. All reduce to two
primitives: the **unweighted inverse estimates on a shape-regular simplex**
(L2-to-L2 for the gradient, and L-infinity-to-L2), obtained by the standard
scaling argument, and the **weighted-to-unweighted reduction**, whose scalar
chain (the alpha_infty / alpha_0 comparabilities under H:porosity) is
*already machine-checked* in `ContinuityAlgebra.v`
(`delta_alpha_bound`, `winv_ratio`) — only the norm-level gluing remains.

## State of the ecosystem (checked July 2026)

No system has FE inverse estimates or the elementwise IBP identities today.
The relevant frontier:

- **arXiv:2604.05984** (Apr 2026): De Giorgi--Nash--Moser in Lean 4. Built
  infrastructure for Sobolev spaces on bounded domains, weak solutions and
  quantitative estimates, and reports that large-scale assistant-driven
  formalisation of hard analysis is now workable. This is the strongest
  evidence that Class G/I items are reachable in Lean with an iterative
  loop; whether that project's domain-IBP is packaged reusably should be
  checked first (its repository, before Mathlib).
- **arXiv:2604.20345** (Apr 2026): simplicial Lagrange finite elements in
  Rocq (unisolvence, the interpolation operator's construction). The FE
  *spaces* now exist formally in the Coq world; estimates do not.
- **arXiv:2410.01538** (Boldo et al.): the detailed-paper-proofs roadmap for
  FEM in Coq; useful as a specification source.

Mathlib ingredients that exist and matter: equivalence of norms on
finite-dimensional spaces; change of variables for the Lebesgue integral
under affine maps (Haar-measure scaling); the divergence theorem on boxes
(not simplices); polynomial spaces (`MvPolynomial`) and their
finite-dimensionality in bounded degree.

## The targets, with strategy and effort estimates

Effort estimates assume an assistant-driven loop of the kind that produced
2604.05984, not unaided work; treat them as order-of-magnitude.

### T1. Unweighted inverse estimate on a simplex (Class I core)

Statement (skeleton `inverse_estimate_grad`): for the polynomial space P_k
on a nondegenerate simplex K with diameter h and inradius rho,
||grad p||_{L2(K)} <= C(k, d, h/rho) / h * ||p||_{L2(K)}.

Strategy: (1) on the reference simplex, p -> ||p||_{L2} and
p -> ||grad p||_{L2} are norms/seminorms on the finite-dimensional P_k, so
the ratio is bounded (Mathlib: finite-dimensional norm equivalence; the
seminorm needs the standard quotient-by-constants trick only for the
Bramble--Hilbert direction, not here — a direct bound
||grad p|| <= C ||p|| holds since both are continuous on a
finite-dimensional space and the sup over the unit sphere is attained).
(2) Transport along the affine map: Mathlib's change of variables gives the
|det| factors; the gradient transforms by the inverse Jacobian, whose norm
is controlled by h/rho (shape regularity). This is the single most valuable
target: everything in Class I reduces to it plus its L-infinity sibling.
Estimate: weeks of loop time; the reference-element half is the easy half.

### T2. L-infinity inverse estimate and face-measure bound (Class I)

`inverse_estimate_linf`: ||p||_{L-infty(K)} <= C h^(-d/2) ||p||_{L2(K)}.
Same two-step pattern as T1 (all norms equivalent on P_k over the reference
simplex; the L2 norm scales by |det|^(1/2) = (c h^d)^(1/2), the sup norm is
invariant). `face_measure_bound` is d-1 dimensional Haar scaling of a face
under the affine map. Both are corollaries of the T1 technology.

### T3. Weighted reductions (Class I, closing lem:winv)

`winv_divvisc`, `winv_grad`, `winv_conv`, `winv_divu`: given T1/T2 and
H:porosity, these are the norm-level gluing of the unweighted estimates
with the scalar comparabilities already proved in `ContinuityAlgebra.v`.
The only analytic content beyond T1 is
||alpha^(1/2) f||_{L2(K)} between alpha_0^(1/2)||f|| and
alpha_infty^(1/2)||f||, i.e. monotonicity of the integral. Days, once T1
lands.

### T4. Divergence theorem on a simplex for polynomial x W^{1,infty} data
(Class G core)

`simplex_divergence`. This is the hardest genuinely-new piece. Options, in
order of preference: (a) reuse the bounded-domain IBP of the 2604.05984
infrastructure if it covers Lipschitz polytopes; (b) prove it directly for
the *specific integrands needed* — all are products of polynomials with an
alpha in W^{1,infty}; approximate alpha by smooth functions and use a
smooth-field divergence theorem on the simplex, itself obtainable by
mapping the box result through a decomposition of the simplex, or by the
usual iterated-integral proof over a simplex written as a graph domain.
Honest assessment: this is a paper-sized formalisation in its own right if
(a) fails. Everything else in Class G is bookkeeping on top of it:
`H_skew`/`H_ibp_vp`/`H_ibp_qu` are T4 plus density/zero-trace facts on the
whole domain (which the 2604.05984 Sobolev infrastructure plausibly
provides), and the assembly identities are finite combinatorics over the
face lattice once traces of continuous piecewise polynomials are defined.

### T5. Face bounds `H_face_p/c` (Class I + G)

Hoelder on a face (Mathlib has Hoelder for integrals) + T2 on both
neighbours + `face_measure_bound` + alpha <= alpha_K. Assembly-level work
once T2/T4 exist.

### What NOT to attempt first

Do not start from `HBS`: it consumes the projector's idempotence and the
full variational setup, i.e. it presupposes a formalised discrete problem.
The dependency-minimal path is T1 -> T2 -> T3 (pure inverse-estimate
technology, no integration by parts), which already discharges `S3` and all
eight `Hw_*` — at that point *lemma:Stability's trusted base shrinks to the
single Green-type identity HBS*, and lemma:Continuity's to Class G + the
data assumptions.

## Suggested loop protocol

1. Elaboration pass: make `PorousNSToolbox.lean` compile with `sorry`s
   against current Mathlib (expect every statement to need API repair; the
   comments pin the intended meaning).
2. Reference-simplex pass: T1/T2 restricted to the reference element.
3. Transport pass: the affine scaling lemmas.
4. T3, then T5, then T4, then the assembly identities.
5. After each pass, update the Coq side's README scope ledger: each
   discharged item moves from "trusted" to "formalised (Lean)", and the
   two-system split should be stated explicitly wherever the formalisation
   is cited.
