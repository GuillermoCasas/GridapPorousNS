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

(Class D recently traded `IU_nonneg`/`IP_nonneg` — the nonnegativity of the two
interpolation-error quantities — for a strengthening of `CI_nonneg` to
`CI_pos : 0 < CI`; the two are now Coq lemmas. This is a *trade*, not a
reduction: `{CI_pos}` is logically strictly stronger than
`{CI_nonneg, IU_nonneg, IP_nonneg}`. It is credibility-neutral rather than free
— IU = h^(k+1)|u|_{H^(k+1)(K)} >= 0 by definition, and C_I > 0 holds for any
real interpolation estimate — and it changes nothing here, since C_I's value is
supplied by T6 and Class D is not a formalisation target either way.)

**Class G — Green-type identities.** `H_skew`, `H_ibp_vp`, `H_ibp_qu`, the two
*diagonal* instances `H_skew_diag`/`H_ibp_diag`, the two assembly identities
`H_assemble_p/c`, and `Horth`. All reduce to: (a) the divergence theorem on a
single simplex for fields of the form (elementwise polynomial) x (W^{1,infty}
weight), and (b) bookkeeping over elements and faces using the continuity
of the integrands across interior faces.

(`HBS` — the Galerkin-testing/adjoint identity of the stability proof — used to
head this class, and was long described as the development's single largest
assumption. It is no longer in the trusted base: it is now a Coq *theorem*,
proved from `H_skew_diag` and `H_ibp_diag`, and its convergence twin `HBS_W` is
gone outright (the convergence file applies `abstract_stability` at the
discrete-error atoms instead). Likewise `H_elem_conv_ibp`/`H_elem_p_ibp`, whose
face terms are now `Definition`s.

Two honesty notes, because the headline is easy to oversell. First, that "single
largest assumption" label was itself wrong: in its old form `HBS` was
`Variable BS : R` pinned by `Hypothesis HBS : BS = <expression in free atoms>`,
which is the eliminable `∀x, x = t → P x` pattern and carries *no* logical
content. Deleting it removed a phantom, not a burden. Second, what `HBS` did
carry survives as a reading obligation rather than a hypothesis — that the
eighteen-term expression *is* the paper's B_S, which includes the projector
idempotence `P^T P = P`. That obligation is not new and not a Lean target: the
same expression already underpinned `BSWW`, so the net effect is that two
encodings of B_S are now provably the same number and one bespoke tested form is
gone. See coq_coverage.tex §(I4) and §G1 for the full accounting; it is not
repeated here.

For this roadmap the consequence is narrow and worth stating plainly: what
changed is *which* Green identities stability rests on, not how much analysis
remains to be formalised. `H_skew_diag`/`H_ibp_diag` were already in the trusted
base via the convergence file, so the union of Lean targets below is unchanged.)

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

## State of the ecosystem (re-verified 16 July 2026 against cloned repositories)

**The hard negative.** No proof assistant has the Bramble–Hilbert lemma or FE
inverse estimates: not Coq/Rocq, Lean, Isabelle, HOL Light, HOL4, PVS, ACL2,
Mizar, or Metamath. T1/T2/T6 are not "port an existing result" tasks in any
system. Relatedly, Green's theorem is formalised in exactly two systems (HOL
Light and Isabelle), both **planar**; Stokes' theorem in none.

**The sharpest statement of the gap.** Céa's lemma *is* formalised in Coq:
`rocq-num-analysis`, file `LM/lax_milgram_cea.v`, 0 `Admitted`, 8662 `Qed`
(<https://depot.lipn.univ-paris13.fr/mayero/rocq-num-analysis> — GitLab, not
GitHub). Céa needs no Sobolev spaces: it is abstract Hilbert-space
quasi-optimality, which is precisely why it is formalizable today while
Bramble–Hilbert is not. But Céa gives **no rate** — converting
inf_{v_h} ||u - v_h|| into O(h^k) *is* the Bramble–Hilbert step, and that step
is the missing link. It is the same primitive T1/T6 need.

The relevant frontier:

- **arXiv:2604.20345** (Boldo, Clément, Martin, Mayero, Mouhcine; 22 Apr 2026):
  "A Rocq Formalization of Simplicial Lagrange Finite Elements" — unisolvence
  and the interpolation operator's construction. The FE *spaces* now exist
  formally in the Coq world; estimates do not, and the gap is structural rather
  than a matter of effort: the paper defines `local_interp`, whose only lemmas
  are algebraic (linear, idempotent, a projection). There is **no mesh and no
  element diameter**, so an O(h^k) estimate is not yet *expressible*, let alone
  proved ("bramble": 0 hits in the development). Their Figure 15 marks Sobolev
  spaces and the approximation error as "perspectives"; verbatim: "Another one
  is the bounding of the approximation error, which requires to formalize
  Sobolev spaces (and prove they are Hilbert spaces)."
- **arXiv:2604.05984** (7 Apr 2026): De Giorgi–Nash–Moser in Lean 4
  (<https://github.com/scottnarmstrong/DeGiorgi>). Self-reported first
  formalisation of Sobolev spaces on bounded domains via weak derivatives, and
  evidence that assistant-driven formalisation of hard analysis is workable at
  scale. Treat "reusable" skeptically: ~56k lines, standalone, **not upstreamed
  to Mathlib**, 3 commits total, unmaintained since April 2026. Its Sobolev
  layer is not packaged for reuse, so T4 should not be planned around it.
- **arXiv:2410.01538** (Boldo et al.): the detailed-paper-proofs roadmap for
  FEM in Coq; useful as a specification source.

**Strategic constraint: the two Coq stacks are mutually exclusive.** Per the FEM
authors themselves (Inria RR-9590 §3.3), the MathComp-Analysis and Coquelicot
hierarchies "are incompatible: they cannot both be used in the same
development", and "From MathComp-Analysis, the only part we use is boolp". So a
Coq route inherits *either* Hilbert + Lax–Milgram + Céa + Lagrange elements from
`rocq-num-analysis` (Coquelicot-based) *or* the measure/integration depth of
MathComp-Analysis — not both. This must be chosen early and deliberately; it is
not a detail that can be deferred to the loop.

Mathlib ingredients that exist and matter: equivalence of norms on
finite-dimensional spaces; change of variables for the Lebesgue integral under
affine maps (Haar-measure scaling); polynomial spaces (`MvPolynomial`) and their
finite-dimensionality in bounded degree. The divergence theorem is present only
for **rectangular boxes** (Kudryashov, ITP 2022): there is no surface measure
and no outward normal, so on a simplex the boundary integral is not merely
unproved but *not statable*. This bears directly on T4.

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

`simplex_divergence`. This is the hardest genuinely-new piece, and the July 2026
re-check made it *harder*, not easier — both previously-hoped shortcuts are now
closed:

- Reusing the bounded-domain IBP of the 2604.05984 infrastructure is no longer
  a plan: that development is standalone, un-upstreamed, 3 commits, unmaintained
  since April 2026, and its Sobolev layer is not packaged for reuse. Vendoring
  ~56k unmaintained lines to obtain one identity is a worse trade than proving
  the identity.
- Mapping Mathlib's box divergence theorem onto a simplex does not typecheck as
  a plan either: that result is stated for rectangular boxes with no surface
  measure and no outward normal, so the simplex boundary integral cannot even be
  *stated* against it. The prerequisite is the missing statement, not a
  transport argument.

What remains is the direct route, for the *specific integrands needed* — all are
products of polynomials with an alpha in W^{1,infty}: approximate alpha by smooth
functions and use a smooth-field divergence theorem on the simplex, obtained by
the usual iterated-integral proof over a simplex written as a graph domain. This
requires first defining a surface measure and outward normal on a simplex face.
Honest assessment: a paper-sized formalisation in its own right — and, given the
above, that is now the expected case rather than the fallback. That Green's
theorem exists only in HOL Light and Isabelle, and only in the plane, is the
ecosystem's verdict on this difficulty.

Everything else in Class G is bookkeeping on top of it:
`H_skew`/`H_ibp_vp`/`H_ibp_qu` are T4 plus density/zero-trace facts on the whole
domain (which no system currently packages reusably), and the assembly
identities are finite combinatorics over the face lattice once traces of
continuous piecewise polynomials are defined.

### T5. Face bounds `H_face_p/c` (Class I + G)

Hoelder on a face (Mathlib has Hoelder for integrals) + T2 on both
neighbours + `face_measure_bound` + alpha <= alpha_K. Assembly-level work
once T2/T4 exist.

### What NOT to attempt first

Do not start from `Horth`: like `HBS` before it became a theorem, it
presupposes the full variational setup — it needs bilinearity of B_S in its
first argument, which the scalar encoding does not give the kernel, i.e. it
presupposes a formalised discrete problem. It is now the largest Class-G item.
The dependency-minimal path is T1 -> T2 -> T3 (pure inverse-estimate
technology, no integration by parts), which already discharges `S3` and all
eight `Hw_*` — at that point *lemma:Stability's trusted base shrinks to the two
diagonal Green identities `H_skew_diag`/`H_ibp_diag`*, and lemma:Continuity's to
Class G + the data assumptions.

That endpoint is a materially better one than the roadmap could previously
offer, and the improvement is qualitative rather than arithmetic. The old
endpoint was `HBS`, a single monolithic identity that presupposed the whole
variational setup and the projector's idempotence — i.e. not attackable at all
until a discrete problem had been formalised. The new endpoint is two *diagonal
instances of Green's identity at fixed atoms*: they need T4's divergence
primitive and nothing else, no projector and no discrete problem. Stability's
Green-type base is now reachable by the same primitive Class G already needed,
so nothing on stability's path is gated behind the variational formalisation any
more. (The tractability gained is real but bounded: T4 is itself the hardest
target here, so "reachable via T4" still means paper-sized.)

(This paragraph used to advise against starting from `HBS`. That advice is now
moot: `HBS` was discharged not by formalising the variational setup, but by
*defining* B_S(U_h,U_h) to be the eighteen-term expression and proving the
tested form from the two diagonal identities. The projector-idempotence content
it consumed did not disappear — it moved into the definitional reading of the
atoms, which is outside the hypothesis list and outside this roadmap's scope.)

## Suggested loop protocol

0. Decide the target system deliberately, before any of the below. This
   roadmap targets Lean/Mathlib, but the July 2026 check makes the Coq route
   genuinely competitive for T6: `rocq-num-analysis` already supplies Hilbert
   spaces, Lax–Milgram and Céa, and 2604.20345 supplies Lagrange elements —
   none of which Mathlib has assembled. The catch is that the choice is
   irreversible within one development: MathComp-Analysis and Coquelicot "are
   incompatible: they cannot both be used in the same development" (Inria
   RR-9590 §3.3), so the Coq route buys the FE stack at the cost of
   MathComp-Analysis's measure/integration depth — which is exactly what T4
   wants. Lean keeps the integration depth and starts the FE stack from
   nothing. Neither side has Bramble–Hilbert, so no choice avoids T6.
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


## T6 — Interpolation estimates (new: lem:continterp's Class-I additions)

With `AbstractInterpolation.v` in the kernel, Class I gains the seven
interpolation hypotheses `HI_*` (eq:interp / eq:interpinfty via the
replacement table eq:interpdivvisc–eq:interpzero). Lean target: the
Bramble–Hilbert / Deny–Lions lemma on a shape-regular simplex plus the
scaling argument — Mathlib has the Sobolev and polynomial-approximation
ingredients but not the assembled estimate. The L∞ variants additionally
need the (already-targeted) H² ↪ C⁰ embedding for d ≤ 3. Discharging T6
together with T1–T2 removes every Class-I hypothesis of all four abstract
theorems.

This is the target the ecosystem check bears on most directly, and the Céa
comparison is the crispest way to see why it is hard rather than merely
undone. Céa's lemma is fully formalised in Coq because it is abstract
Hilbert-space quasi-optimality and needs no Sobolev spaces; what it yields is
inf_{v_h} ||u - v_h||, with no rate. T6 *is* the step that turns that infimum
into O(h^k) — and that step exists in no proof assistant. The Rocq Lagrange-
element work (2604.20345) shows the shape of the obstacle: it has the
interpolation operator but no mesh and no element diameter, so the estimate is
not yet expressible there at all. T6 therefore includes building the metric
scaffolding (diameter, shape regularity, the reference-to-physical map) that
makes the statement *sayable*, before any of it can be proved. Budget
accordingly; this is not a lemma-porting exercise.

## T7 — Consistency (the convergence file's Class-G addition)

`Horth` = eq:consistency + bilinearity: Green's identity for the strong operator
applied to the exact solution (regularity eq:regularity) plus linearity
of every component map — the latter is definitional in any concrete
instantiation; the former is T3's divergence-theorem primitive again. No
new primitive is required beyond T1–T5.

(`HBS_W` was listed here too, as "the same tested-identity material as `HBS`".
`HBS` is now a Coq theorem and `HBS_W` is gone outright; neither is a Lean
target any more, and the only Class-G addition the convergence file still makes
is `Horth`.)
